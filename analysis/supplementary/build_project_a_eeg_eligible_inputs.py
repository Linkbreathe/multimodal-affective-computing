"""Build Project A inputs for the frozen nine-participant common-window contract."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from analysis.supplementary.build_alignment_contract import apply_masks_to_window_features
from real_time_ml.evaluation.alignment import validate_alignment_contract


PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
MODALITIES = ("eeg", "ecg", "eye", "head", "video")
PROJECT_A_MODALITIES = ("eeg", "ecg", "eye", "head")
KEYS = ["participant_id", "condition", "condition_window_index"]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, rows: int | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }
    if rows is not None:
        output["rows"] = int(rows)
    return output


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not set(normalized).issubset({"true", "false", "1", "0", "1.0", "0.0"}):
        raise ValueError(f"Unrecognized Boolean values in {series.name}")
    return normalized.isin({"true", "1", "1.0"})


def build_inputs(contract_dir: Path, output_dir: Path) -> dict[str, Any]:
    contract = validate_alignment_contract(contract_dir)
    if contract.get("schema_version") != "eeg_eligible_modality_ablation_contract_v1":
        raise ValueError("Project A nine-participant inputs require the EEG-eligible contract")
    if tuple(contract.get("participants", ())) != PARTICIPANTS:
        raise ValueError("The contract participant cohort is not the frozen nine-participant cohort")
    if int(contract.get("observation_count", -1)) != 81:
        raise ValueError("The contract must contain 81 participant-condition observations")

    contract_path = contract_dir / "contract.json"
    base_contract_path = Path(str(contract["base_contract"]["path"]))
    if file_sha256(base_contract_path) != contract["base_contract"]["sha256"]:
        raise ValueError("The referenced 15-participant base contract hash changed")
    base_contract = validate_alignment_contract(base_contract_path.parent)
    source_record = base_contract["files"]["project_a_window_features"]
    source_path = base_contract_path.parent / source_record["path"]
    if file_sha256(source_path) != source_record["sha256"]:
        raise ValueError("The Project A base feature table hash changed")

    features = pd.read_csv(source_path)
    features = features.loc[features["participant_id"].astype(str).isin(PARTICIPANTS)].copy()
    masks_path = contract_dir / contract["files"]["common_masks"]["path"]
    masks = pd.read_csv(masks_path)
    if len(features) != 567 or features.duplicated(KEYS).any():
        raise ValueError("Project A source features must cover 567 unique cohort windows")
    if len(masks) != 567 or masks.duplicated(KEYS).any():
        raise ValueError("The common mask must cover 567 unique cohort windows")
    if set(map(tuple, features[KEYS].to_numpy())) != set(map(tuple, masks[KEYS].to_numpy())):
        raise ValueError("Project A features and the common-window mask have different keys")

    common = _as_bool(masks["common_valid"])
    if int(common.sum()) != 545:
        raise ValueError("The fixed common-window mask must contain exactly 545 valid windows")
    for modality in MODALITIES:
        if not _as_bool(masks[f"{modality}_valid"]).equals(common):
            raise ValueError(f"{modality} validity differs from the common mask")

    # Only pass the five contract validity decisions into the masking helper.
    # Source-valid audit columns and common_valid itself must not become model
    # inputs after condition aggregation.
    mask_columns = [*KEYS, *[f"{modality}_valid" for modality in MODALITIES]]
    masked = apply_masks_to_window_features(features, masks[mask_columns])
    masked = masked.sort_values(KEYS).reset_index(drop=True)
    for modality in PROJECT_A_MODALITIES:
        modality_columns = [
            column
            for column in masked.columns
            if column.startswith(f"{modality}_") and not column.startswith("qc_")
        ]
        invalid = ~_as_bool(masked[f"mask_{modality}_valid"])
        if modality_columns and np.isfinite(
            masked.loc[invalid, modality_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
        ).any():
            raise ValueError(f"{modality} contains values outside the 545 common-valid windows")

    labels_path = contract_dir / contract["files"]["labels"]["path"]
    labels = pd.read_csv(labels_path)
    observed_labels = masked.groupby(["participant_id", "condition"], as_index=False)[
        ["relaxation", "discomfort"]
    ].first()
    checked = observed_labels.merge(
        labels[["participant_id", "condition", "relaxation", "discomfort"]],
        on=["participant_id", "condition"],
        suffixes=("_features", "_contract"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(checked) != 81 or not (checked["_merge"] == "both").all():
        raise ValueError("Project A feature labels do not cover the 81 contract observations")
    for target in ("relaxation", "discomfort"):
        if not np.allclose(
            checked[f"{target}_features"], checked[f"{target}_contract"], atol=1e-8, rtol=0.0
        ):
            raise ValueError(f"Project A {target} labels differ from the contract")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "project_a_window_features_common.csv"
    valid_keys_path = output_dir / "common_valid_window_keys.csv"
    masked.to_csv(feature_path, index=False, lineterminator="\n")
    masks.loc[common, KEYS].sort_values(KEYS).to_csv(
        valid_keys_path, index=False, lineterminator="\n"
    )
    manifest = {
        "schema_version": "project_a_eeg_eligible_inputs_v1",
        "contract": _record(contract_path),
        "base_contract": _record(base_contract_path),
        "source_project_a_features": _record(source_path, len(features)),
        "common_masks": _record(masks_path, len(masks)),
        "outputs": {
            "window_features": _record(feature_path, len(masked)),
            "common_valid_window_keys": _record(valid_keys_path, int(common.sum())),
        },
        "participants": list(PARTICIPANTS),
        "participant_condition_observations": 81,
        "source_windows": 567,
        "common_valid_windows": 545,
        "project_a_modalities": list(PROJECT_A_MODALITIES),
        "mask_policy": "fixed_five_way_intersection",
        "zero_common_window_observations": contract["zero_common_window_observations"],
    }
    manifest_path = output_dir / "project_a_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["manifest"] = _record(manifest_path)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = ROOT / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=root / "contract")
    parser.add_argument("--output-dir", type=Path, default=root / "project_a" / "inputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_inputs(args.contract_dir.resolve(), args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
