#!/usr/bin/env python
"""Rebuild only corrected EEG handcrafted features in a versioned Relax run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
from mac.data.relax_foundation import (  # noqa: E402
    EEG_DISABLED_PARTICIPANTS,
    RELAX_EEG_RUN_TAG,
    RELAX_EEG_XDF_COLUMNS,
    relax_eeg_contract_payload,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_run_inputs(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("preprocessed", "manifests", "metrics"):
        source_dir = source / name
        if source_dir.exists():
            shutil.copytree(source_dir, destination / name, dirs_exist_ok=True)
    (destination / "features").mkdir(parents=True, exist_ok=True)


def rebuild(
    *,
    source_run: Path,
    destination_run: Path,
    relax_model_src: Path,
    run_tag: str,
) -> dict[str, object]:
    _copy_run_inputs(source_run, destination_run)
    sys.path.insert(0, str(relax_model_src))

    from mac.data.tables import write_parquet_if_available
    from mac.features.extract import _load_physio
    from mac.features.physio import eeg_features, eeg_quality_coverage
    from mac.data.condition_data import aggregate_window_frame
    from mac.eeg_contract import (
        RELAX_EEG_MONTAGE as MODEL_MONTAGE,
        RELAX_EEG_XDF_COLUMNS as MODEL_XDF_COLUMNS,
        relax_eeg_contract_payload as model_contract_payload,
    )

    if tuple(MODEL_XDF_COLUMNS) != RELAX_EEG_XDF_COLUMNS:
        raise RuntimeError("Relax-Model and fusion repository disagree on EEG XDF columns")
    if model_contract_payload() != relax_eeg_contract_payload():
        raise RuntimeError("Relax-Model and fusion repository disagree on the EEG channel contract")

    source_manifest_path = source_run / "manifests" / "source_manifest.csv"
    source_manifest = pd.read_csv(source_manifest_path)
    source_by_participant = {
        str(row["participant_id"]): Path(str(row["xdf_path"]))
        for _, row in source_manifest.iterrows()
    }
    old_window_path = source_run / "features" / "window_features.csv"
    old_windows = pd.read_csv(old_window_path)
    old_windows["participant_id"] = old_windows["participant_id"].astype(str)
    old_windows["condition"] = old_windows["condition"].astype(str)

    old_eeg_columns = [column for column in old_windows.columns if column.startswith("eeg_")]
    retained = old_windows.drop(columns=old_eeg_columns).copy()
    rebuilt_rows: list[dict[str, float | bool]] = []
    eeg_available_windows = 0
    handcrafted_eeg_usable_windows = 0
    for participant, participant_rows in old_windows.groupby("participant_id", sort=True):
        xdf_path = source_by_participant.get(participant)
        if xdf_path is None or not xdf_path.exists():
            raise RuntimeError(f"Missing locked XDF source for {participant}: {xdf_path}")
        samples, timestamps, sample_rate = _load_physio(xdf_path, "eeg")
        for row_index, row in participant_rows.sort_values("window_start_xdf").iterrows():
            start = float(row["window_start_xdf"])
            end = float(row["window_end_xdf"])
            left, right = np.searchsorted(timestamps, [start, end], side="left")
            window_samples = samples[int(left) : int(right)]
            record: dict[str, float | bool] = {}
            disabled = participant in EEG_DISABLED_PARTICIPANTS
            if disabled:
                coverage = 0.0
                usable = False
            else:
                eeg_available_windows += 1
                eeg = window_samples[:, RELAX_EEG_XDF_COLUMNS]
                coverage = eeg_quality_coverage(eeg, float(sample_rate), 350.0, 0.2)
                usable = coverage >= 0.60
                if usable:
                    record.update(
                        eeg_features(
                            eeg,
                            float(sample_rate),
                            {
                                "delta": [1.0, 4.0],
                                "theta": [4.0, 8.0],
                                "alpha": [8.0, 13.0],
                                "beta": [13.0, 30.0],
                                "gamma": [30.0, 45.0],
                            },
                            False,
                            channel_names=MODEL_MONTAGE,
                        )
                    )
                    handcrafted_eeg_usable_windows += 1
            retained.loc[row_index, "qc_eeg_strict_coverage"] = float(coverage)
            retained.loc[row_index, "qc_eeg_usable"] = bool(usable)
            retained.loc[row_index, "qc_eeg_disabled_by_participant_qc"] = bool(disabled)
            rebuilt_rows.append({"row_index": int(row_index), **record})

    eeg_frame = pd.DataFrame(rebuilt_rows).set_index("row_index")
    corrected = retained.join(eeg_frame, how="left")
    corrected = corrected.loc[old_windows.index]
    corrected_window_path = destination_run / "features" / "window_features.csv"
    corrected.to_csv(corrected_window_path, index=False)
    write_parquet_if_available(
        destination_run / "features" / "window_features.parquet",
        corrected.to_dict(orient="records"),
    )

    condition = aggregate_window_frame(corrected)
    corrected_condition_path = destination_run / "features" / "condition_features.csv"
    condition.to_csv(corrected_condition_path, index=False)
    write_parquet_if_available(
        destination_run / "features" / "condition_features.parquet",
        condition.to_dict(orient="records"),
    )

    non_eeg_columns = [
        column
        for column in old_windows.columns
        if not column.startswith("eeg_") and not column.startswith("qc_eeg_")
    ]
    left_values = old_windows[non_eeg_columns].copy()
    right_values = corrected[non_eeg_columns].copy()
    if list(left_values.columns) != list(right_values.columns) or not left_values.equals(right_values):
        raise RuntimeError("Non-EEG window feature values changed during corrected EEG rebuild")

    run_manifest_path = destination_run / "manifests" / "run_manifest.json"
    previous_manifest = {}
    if run_manifest_path.exists():
        previous_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest = {
        **copy.deepcopy(previous_manifest),
        "run_id": run_tag,
        "parent_run": str(source_run),
        "eeg_channel_contract": relax_eeg_contract_payload(),
        "handcrafted_eeg_montage": list(MODEL_MONTAGE),
        "eeg_available_windows": eeg_available_windows,
        "handcrafted_eeg_usable_windows": handcrafted_eeg_usable_windows,
        "eeg_disabled_windows": int(len(corrected) - eeg_available_windows),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_window_features_sha256": _sha256(old_window_path),
    }
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    errors_path = destination_run / "metrics" / "feature_extraction_errors.json"
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.write_text("[]\n", encoding="utf-8")
    audit = {
        "run_tag": run_tag,
        "window_rows": len(corrected),
        "condition_rows": len(condition),
        "valid_eeg_embedding_windows_expected": eeg_available_windows,
        "handcrafted_eeg_usable_windows": handcrafted_eeg_usable_windows,
        "handcrafted_eeg_qc_failed_windows": eeg_available_windows - handcrafted_eeg_usable_windows,
        "expected_missing_eeg_windows": int(len(corrected) - eeg_available_windows),
        "channel_contract": relax_eeg_contract_payload(),
        "non_eeg_window_features_identical": True,
        "window_features_sha256": _sha256(corrected_window_path),
        "condition_features_sha256": _sha256(corrected_condition_path),
    }
    (destination_run / "metrics" / "eegmap_rebuild_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--destination-run", required=True, type=Path)
    parser.add_argument("--relax-model-src", required=True, type=Path)
    parser.add_argument("--run-tag", default=RELAX_EEG_RUN_TAG)
    args = parser.parse_args()
    audit = rebuild(
        source_run=args.source_run,
        destination_run=args.destination_run,
        relax_model_src=args.relax_model_src,
        run_tag=args.run_tag,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
