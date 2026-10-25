"""Read-only provenance and cohort alignment. No feature/rating association logic."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mac.data.alignment import condition_boundaries, load_marker_events, marker_alignment_qc
from mac.data.index import discover_sources
from mac.data.labels import find_painting_workbook, parse_condition_labels
from .common import read_logged, write_csv, write_json
from .specification import PARTICIPANTS, SPEC, file_hash, spec_hash

THESIS_INPUTS = (
    "chapters/method.tex",
    "chapters/experiment.tex",
    "chapters/results.tex",
    "derived/global_restructure/participant_folds_fmq9.csv",
    "derived/rq2_global_revision/rq2_audit_summary.json",
    "derived/rq1_global_restructure/subjective_ratings_fc15.csv",
    "derived/final_intuitive_revision/source/rq2_group_summary.csv",
    "derived/final_intuitive_revision/source/rq2_participant_metrics.csv",
    "derived/final_intuitive_revision/source/rq2_fusion_details.csv",
)


def audit(
    raw_root: Path, thesis_root: Path, labels_root: Path, output: Path, workers: int = 4
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    cohort_file = thesis_root / THESIS_INPUTS[3]
    cohorts = pd.read_csv(cohort_file)
    ids = sorted(cohorts.test_participant_id.unique())
    if ids != list(PARTICIPANTS):
        raise ValueError(f"Thesis cohort differs from fixed nine participants: {ids}")
    contract = json.loads((thesis_root / THESIS_INPUTS[4]).read_text(encoding="utf-8"))["contract"]
    if sorted(contract["participants"]) != ids:
        raise ValueError("Thesis cohort records disagree.")
    # Persist the specification before extracting any features or computing associations.
    output.mkdir(parents=True)
    write_json(output / "feature_specification.json", SPEC)
    config = dict(
        raw_root=str(raw_root.resolve()),
        thesis_root=str(thesis_root.resolve()),
        labels_root=str(labels_root.resolve()),
        output_dir=str(output.resolve()),
        workers=min(4, max(1, workers)),
        specification_sha256=spec_hash(),
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
    )
    config["versions"] = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "pandas", "pyxdf", "opencv-python", "neurokit2"]
    }
    write_json(output / "resolved_config.json", config)
    sources = discover_sources(raw_root, list(PARTICIPANTS))
    write_csv(output / "recording_sources.csv", pd.DataFrame(sources))
    workbook = find_painting_workbook(labels_root)
    labels = pd.DataFrame(
        parse_condition_labels(
            labels_root, list(PARTICIPANTS), SPEC["intensities"], SPEC["frequencies"]
        )
    )
    thesis_labels = pd.read_csv(thesis_root / THESIS_INPUTS[5]).rename(
        columns={"condition_id": "condition"}
    )
    joined = labels.merge(
        thesis_labels,
        on=["participant_id", "condition"],
        suffixes=("_workbook", "_thesis"),
        validate="one_to_one",
    )
    if len(joined) != 81 or any(
        not np.array_equal(joined[c + "_workbook"], joined[c + "_thesis"])
        for c in ["relaxation_raw", "discomfort_raw"]
    ):
        raise ValueError("Workbook ratings disagree with current thesis ratings.")
    if not np.array_equal(joined.presentation_position, joined.presentation_order):
        raise ValueError("Workbook order differs from thesis order.")
    records, issues, source_records = [], [], []
    input_paths = [(workbook, "rating_workbook")] + [
        (thesis_root / p, "thesis_reference") for p in THESIS_INPUTS
    ]
    repo_root = Path(__file__).resolve().parents[4]
    input_paths.extend(
        (repo_root / p, "existing_extraction_definition")
        for p in [
            "configs/project.yaml",
            "src/mac/features/physio.py",
            "src/mac/features/eye.py",
            "src/mac/features/head.py",
            "src/mac/features/video.py",
            "analysis/supplementary/recompute_ecg_neurokit_features.py",
        ]
    )
    for source in sources:
        pid = source["participant_id"]
        for key in [
            "xdf_path",
            "events_csv",
            "samples_csv",
            "eye_tracking_csv",
            "video_frames_csv",
        ]:
            if source.get(key):
                input_paths.append((Path(source[key]), f"{pid}:{key}"))
            else:
                issues.append(
                    dict(participant_id=pid, condition="", scope=key, reason="missing_source")
                )
        if source["session_dir"]:
            manifest = Path(source["session_dir"]) / "video_manifest.json"
            if manifest.exists():
                input_paths.append((manifest, f"{pid}:video_manifest"))
        events = read_logged(Path(source["events_csv"])) if source["events_csv"] else pd.DataFrame()
        if not events.empty:
            allowed = set(SPEC["participant_aliases"].get(pid, [pid]))
            observed = set(events.participant_id.dropna().astype(str))
            if not observed <= allowed:
                raise ValueError(f"Unrecognized recording identity {pid}: {observed}")
            starts = events.loc[events.event_type.eq("condition_start")].sort_values("unix_time_ms")
            expected = (
                labels.loc[labels.participant_id.eq(pid)]
                .sort_values("presentation_position")
                .condition.tolist()
            )
            if starts.condition_id.tolist() != expected:
                raise ValueError(f"Presentation order mismatch: {pid}")
        try:
            markers = load_marker_events(Path(source["xdf_path"]), "WaterLiliesExperimentMarkers")
            boundaries = {b.condition: b for b in condition_boundaries(markers, pid)}
            qc = marker_alignment_qc(markers)
        except Exception as error:
            markers, boundaries, qc = [], {}, {}
            issues.append(
                dict(
                    participant_id=pid,
                    condition="",
                    scope="xdf_boundaries",
                    reason=f"{type(error).__name__}: {error}",
                )
            )
        for label in labels.loc[labels.participant_id.eq(pid)].to_dict("records"):
            c = label["condition"]
            row = {
                k: label[k]
                for k in [
                    "participant_id",
                    "condition",
                    "intensity",
                    "frequency",
                    "intensity_index",
                    "frequency_index",
                    "relaxation_raw",
                    "discomfort_raw",
                    "label_source_row",
                ]
            }
            row["presentation_order"] = label["presentation_position"]
            row["intensity_level"] = label["intensity_index"] + 1
            row["frequency_level"] = label["frequency_index"] + 1
            row.update(
                xdf_is_backup=source["xdf_is_backup"],
                logged_identity=";".join(sorted(observed)) if not events.empty else "",
                identity_alias_applied=pid == "P003",
                alignment_status="unavailable",
                alignment_reason="missing_boundaries",
            )
            if c in boundaries:
                b = boundaries[c]
                row.update(
                    start_xdf=b.start_xdf,
                    end_xdf=b.end_xdf,
                    start_unix_ms=b.start_unix_ms,
                    end_unix_ms=b.end_unix_ms,
                    duration_s=b.duration_seconds,
                    marker_median_residual_ms=qc.get("median_abs_residual_ms"),
                    marker_max_residual_ms=qc.get("max_abs_residual_ms"),
                    alignment_status="available",
                    alignment_reason="",
                )
                for endpoint in ["start", "end"]:
                    matched = (
                        events.loc[
                            events.event_type.eq("condition_" + endpoint)
                            & events.condition_id.eq(c)
                        ]
                        if not events.empty
                        else pd.DataFrame()
                    )
                    if len(matched) == 1:
                        row[endpoint + "_event_marker_difference_ms"] = abs(
                            float(matched.unix_time_ms.iloc[0]) - getattr(b, endpoint + "_unix_ms")
                        )
                    else:
                        row["alignment_status"], row["alignment_reason"] = (
                            "unavailable",
                            "missing_or_duplicate_unity_boundary",
                        )
                differences = [
                    row.get(s + "_event_marker_difference_ms", np.inf) for s in ["start", "end"]
                ]
                if max(differences) > SPEC["time"]["alignment_exclusion_ms"]:
                    row["alignment_status"], row["alignment_reason"] = (
                        "unavailable",
                        "boundary_discrepancy_over_250ms",
                    )
                row["alignment_warning"] = bool(
                    qc.get("max_abs_residual_ms", 0) > 50 or max(differences) > 50
                )
            # Independent Unity modalities can proceed if the XDF alone is absent.
            if not events.empty:
                for endpoint in ["start", "end"]:
                    m = events.loc[
                        events.event_type.eq("condition_" + endpoint) & events.condition_id.eq(c)
                    ]
                    if len(m) == 1:
                        row["unity_" + endpoint + "_unix_ms"] = float(m.unix_time_ms.iloc[0])
            records.append(row)
        print(f"audit {pid}: {len(boundaries)} XDF condition boundaries", flush=True)
    for path, role in input_paths:
        source_records.append(
            dict(
                path=str(path.resolve()),
                role=role,
                exists=path.is_file(),
                bytes=path.stat().st_size if path.is_file() else np.nan,
                sha256=file_hash(path) if path.is_file() else "",
            )
        )
    manifest = pd.DataFrame(records).sort_values(["participant_id", "condition"])
    write_csv(output / "condition_manifest.csv", manifest)
    write_csv(
        output / "condition_alignment_audit.csv",
        manifest.drop(columns=["relaxation_raw", "discomfort_raw"]),
    )
    write_csv(output / "input_files.csv", pd.DataFrame(source_records))
    write_csv(
        output / "source_issues.csv",
        pd.DataFrame(issues, columns=["participant_id", "condition", "scope", "reason"]),
    )
    write_json(
        output / "audit_status.json",
        dict(
            status="complete",
            condition_count=len(records),
            rating_agreement=True,
            presentation_order_agreement=True,
        ),
    )
