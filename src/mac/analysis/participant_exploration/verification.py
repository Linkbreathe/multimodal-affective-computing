"""Independent arithmetic checks, provenance integrity, and completion marking."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import verify_lock, write_csv, write_json
from .extraction import source_integrity
from .specification import FEATURES, PARTICIPANTS, file_hash, spec_hash


def verify(output: Path) -> dict:
    verify_lock(output)
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("Completed output is immutable.")
    expected = {
        "participant_condition_features.csv": 81,
        "feature_quality.csv": 891,
        "within_person_centered.csv": 1053,
        "participant_descriptive_summary.csv": 117,
        "individual_associations.csv": 198,
        "condition_response_summary.csv": 117,
        "cross_participant_summary.csv": 22,
    }
    tables = {name: pd.read_csv(output / name) for name in expected}
    checks = {name + "_rows": len(tables[name]) == count for name, count in expected.items()}
    wide = tables["participant_condition_features.csv"]
    q = tables["feature_quality.csv"]
    long = tables["within_person_centered.csv"]
    assoc = tables["individual_associations.csv"]
    cross = tables["cross_participant_summary.csv"]
    condition = tables["condition_response_summary.csv"]
    checks["unique_condition_keys"] = not wide.duplicated(["participant_id", "condition"]).any()
    checks["unique_quality_keys"] = not q.duplicated(
        ["participant_id", "condition", "feature"]
    ).any()
    checks["unique_association_keys"] = not assoc.duplicated(
        ["participant_id", "feature", "outcome"]
    ).any()
    checks["cohort_exact"] = sorted(wide.participant_id.unique()) == list(PARTICIPANTS)
    checks["all_eleven_features_retained"] = set(q.feature) == set(FEATURES)
    checks["availability_equals_finite_estimates"] = bool(
        np.array_equal(q.analysis_available, np.isfinite(q.analysis_value))
        and np.array_equal(q.raw_available, np.isfinite(q.raw_value))
    )
    checks["missing_estimates_have_reasons"] = bool(
        q.loc[~q.analysis_available, "reason"].fillna("").str.len().gt(0).all()
    )
    checks["coverage_bounded"] = bool(q.valid_coverage.dropna().between(0, 1 + 1e-8).all())
    checks["fixation_proportion_bounded"] = bool(
        wide.gaze_fixation_time_proportion_raw.dropna().between(0, 1).all()
    )
    checks["transforms_match"] = all(
        np.allclose(wide[f + "_analysis"], np.log(wide[f + "_raw"]), equal_nan=True)
        if FEATURES[f][2] == "ln"
        else np.allclose(wide[f + "_analysis"], wide[f + "_raw"], equal_nan=True)
        for f in FEATURES
    )
    checks["centered_sums_zero"] = bool(
        long.groupby(["participant_id", "variable"]).analysis_deviation.sum().abs().lt(1e-8).all()
    )
    checks["raw_centered_sums_zero"] = bool(
        long.groupby(["participant_id", "variable"]).raw_deviation.sum().abs().lt(1e-6).all()
    )
    arithmetic = []
    for r in assoc.itertuples():
        p = wide.loc[wide.participant_id.eq(r.participant_id)].sort_values("condition")
        x = p[r.feature + "_analysis"]
        y = p[r.outcome + "_raw"]
        valid = x.notna() & y.notna()
        conditions = ";".join(p.loc[valid, "condition"])
        good = int(valid.sum()) == r.n_valid_pairs and conditions == (
            r.included_conditions if isinstance(r.included_conditions, str) else ""
        )
        estimable = int(valid.sum()) >= 6 and x[valid].nunique() > 1 and y[valid].nunique() > 1
        good &= estimable == (r.estimation_status == "estimable")
        if estimable:
            reference = float(spearmanr(x[valid], y[valid]).statistic)
            good &= bool(np.isclose(reference, r.spearman_rho, atol=1e-12))
        else:
            good &= bool(pd.isna(r.spearman_rho) and pd.notna(r.reason))
        arithmetic.append(good)
    checks["independent_scipy_correlations_match"] = all(arithmetic)
    checks["cross_direction_counts_sum_to_nine"] = bool(
        (cross.n_positive + cross.n_negative + cross.n_zero + cross.n_non_estimable).eq(9).all()
    )
    checks["cross_estimable_counts_match"] = bool(
        (cross.n_positive + cross.n_negative + cross.n_zero).eq(cross.n_estimable).all()
    )
    for scale in ["raw", "analysis"]:
        checks[scale + "_condition_direction_counts_match"] = bool(
            (
                condition[scale + "_n_positive"]
                + condition[scale + "_n_negative"]
                + condition[scale + "_n_zero"]
            )
            .eq(condition[scale + "_n_valid"])
            .all()
        )
    # An EEG failure must not erase another available modality in that same record.
    failure = wide.loc[wide.eeg_alpha_power_analysis.isna()]
    checks["feature_specific_missingness_preserved"] = bool(
        len(failure) == 0 or failure.head_mean_angular_speed_analysis.notna().any()
    )
    integrity = source_integrity(output)
    write_csv(output / "input_integrity_verification.csv", pd.DataFrame(integrity))
    checks["original_sources_and_thesis_inputs_unchanged"] = all(x["unchanged"] for x in integrity)
    video = pd.read_csv(output / "consumed_video_frames.csv").drop_duplicates("path")
    valid_inputs = video.loc[video.sha256.notna()]

    def unchanged(row):
        p = Path(row.path)
        return p.is_file() and file_hash(p) == row.sha256

    with ThreadPoolExecutor(max_workers=4) as pool:
        frame_results = list(pool.map(unchanged, valid_inputs.itertuples(index=False)))
    checks["consumed_video_frame_hashes_unchanged"] = all(frame_results)
    forbidden = {
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".pdf",
        ".eps",
        ".tiff",
        ".gif",
        ".mp4",
        ".flo",
        ".html",
    }
    checks["no_graphical_outputs"] = not any(
        p.suffix.lower() in forbidden for p in output.rglob("*") if p.is_file()
    )
    checks["no_extraction_exceptions"] = all(
        not json.loads((output / "cache" / p / "errors.json").read_text()) for p in PARTICIPANTS
    )
    checks["chinese_report_and_draft_present"] = all(
        (output / name).is_file()
        for name in [
            "report_zh.txt",
            "thesis_supplement_zh.txt",
            "data_dictionary.csv",
            "RUNNING.txt",
        ]
    )
    phrase = "This analysis was conducted after the main prediction analyses and is post hoc and exploratory."
    checks["explicit_posthoc_statement"] = all(
        phrase in (output / name).read_text(encoding="utf-8")
        for name in ["report_zh.txt", "thesis_supplement_zh.txt"]
    )
    dictionary = pd.read_csv(output / "data_dictionary.csv")
    checks["all_core_columns_documented"] = all(
        set(dictionary.loc[dictionary.file.eq(name), "field"]) == set(frame.columns)
        for name, frame in tables.items()
    )
    package = Path(__file__).parent
    repo = package.parents[3]
    code_files = sorted(package.glob("*.py")) + [
        repo / "src/mac/data/index.py",
        repo / "tests/test_participant_exploration.py",
    ]
    write_csv(
        output / "analysis_code_manifest.csv",
        pd.DataFrame(
            [
                dict(path=str(p.relative_to(repo)), sha256=file_hash(p), bytes=p.stat().st_size)
                for p in code_files
            ]
        ),
    )
    payload = dict(
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
        expected_dimensions=expected,
        source_files_checked=len(integrity),
        video_files_checked=len(frame_results),
        specification_sha256=spec_hash(),
        n_estimable_correlations=int(assoc.estimation_status.eq("estimable").sum()),
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(output / "verification.json", payload)
    if not all(checks.values()):
        raise AssertionError(f"Verification failed: {[k for k, v in checks.items() if not v]}")
    write_json(output / "COMPLETE.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload
