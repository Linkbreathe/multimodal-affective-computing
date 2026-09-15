"""Build the frozen nine-participant, common-window ablation contract."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd


PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
MODALITIES = ("eeg", "ecg", "eye", "head", "video")
SEEDS = (20260705, 20260706, 20260707)
WINDOW_KEYS = ["participant_id", "condition", "condition_window_index"]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _file_record(path: Path, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false", "1", "0", "1.0", "0.0"})
    if unknown:
        raise ValueError(f"Unrecognized mask values in {series.name}: {unknown}")
    return normalized.isin({"true", "1", "1.0"})


def build_contract(base_contract_dir: Path, output_dir: Path) -> dict[str, Any]:
    base_contract_path = base_contract_dir / "contract.json"
    base_contract = json.loads(base_contract_path.read_text(encoding="utf-8"))
    for name, record in base_contract["files"].items():
        source = base_contract_dir / record["path"]
        if not source.is_file() or file_sha256(source) != record["sha256"]:
            raise ValueError(f"Base contract file failed its hash check: {name}")

    cohorts = json.loads((base_contract_dir / base_contract["files"]["cohorts"]["path"]).read_text(encoding="utf-8"))
    if tuple(cohorts.get("eeg_eligible", ())) != PARTICIPANTS:
        raise ValueError("Base contract EEG-eligible cohort differs from the frozen nine participants")

    labels_source = pd.read_csv(base_contract_dir / base_contract["files"]["labels"]["path"])
    windows_source = pd.read_csv(base_contract_dir / base_contract["files"]["windows"]["path"])
    masks_source = pd.read_csv(base_contract_dir / base_contract["files"]["modality_masks"]["path"])
    labels = labels_source.loc[labels_source["participant_id"].astype(str).isin(PARTICIPANTS)].copy()
    windows = windows_source.loc[windows_source["participant_id"].astype(str).isin(PARTICIPANTS)].copy()
    masks = masks_source.loc[masks_source["participant_id"].astype(str).isin(PARTICIPANTS)].copy()

    if len(labels) != 81 or labels.duplicated(["participant_id", "condition"]).any():
        raise ValueError("The frozen cohort must contain 81 unique participant-condition labels")
    if len(windows) != 567 or windows.duplicated(WINDOW_KEYS).any():
        raise ValueError("The frozen cohort must contain 567 unique windows")
    if len(masks) != 567 or masks.duplicated(WINDOW_KEYS).any():
        raise ValueError("The frozen cohort mask must contain 567 unique windows")
    if set(map(tuple, windows[WINDOW_KEYS].to_numpy())) != set(map(tuple, masks[WINDOW_KEYS].to_numpy())):
        raise ValueError("Base windows and modality masks have different keys in the frozen cohort")

    source_valid_columns = [f"{modality}_valid" for modality in MODALITIES]
    source_mask = masks[WINDOW_KEYS].copy()
    for column in source_valid_columns:
        source_mask[f"source_{column}"] = _bool_series(masks[column])
    source_mask["common_valid"] = source_mask[
        [f"source_{column}" for column in source_valid_columns]
    ].all(axis=1)
    for modality in MODALITIES:
        source_mask[f"{modality}_valid"] = source_mask["common_valid"]
    common_masks = windows[WINDOW_KEYS].merge(source_mask, on=WINDOW_KEYS, how="left", validate="one_to_one")
    if common_masks["common_valid"].isna().any():
        raise ValueError("A cohort window lacks a common validity decision")
    if int(common_masks["common_valid"].sum()) != 545:
        raise ValueError("Expected exactly 545 five-way-common valid windows")
    if any(
        not common_masks[f"{modality}_valid"].equals(common_masks["common_valid"])
        for modality in MODALITIES
    ):
        raise ValueError("A modality-specific ablation mask differs from the common window mask")

    split_rows: list[dict[str, Any]] = []
    for fold_index, test_participant in enumerate(PARTICIPANTS, start=1):
        validation_participant = PARTICIPANTS[fold_index % len(PARTICIPANTS)]
        for participant in PARTICIPANTS:
            role = (
                "test"
                if participant == test_participant
                else "validation"
                if participant == validation_participant
                else "train"
            )
            split_rows.append(
                {
                    "fold_index": fold_index,
                    "test_participant": test_participant,
                    "validation_participant": validation_participant,
                    "participant_id": participant,
                    "role": role,
                }
            )
    splits = pd.DataFrame(split_rows)
    if len(splits) != 81 or set(splits.groupby("fold_index")["role"].value_counts().to_dict().values()) != {1, 7}:
        raise ValueError("Generated split manifest is not nine folds of 7/1/1")

    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "condition_labels.csv"
    window_path = output_dir / "windows.csv"
    cohort_path = output_dir / "cohorts.json"
    split_path = output_dir / "split_manifest.csv"
    mask_path = output_dir / "common_valid_window_masks.csv"
    labels.to_csv(label_path, index=False, lineterminator="\n")
    windows.to_csv(window_path, index=False, lineterminator="\n")
    splits.to_csv(split_path, index=False, lineterminator="\n")
    common_masks.to_csv(mask_path, index=False, lineterminator="\n")
    _write_json(
        cohort_path,
        {
            "eeg_eligible": list(PARTICIPANTS),
            "definition": {
                "eeg_eligible": "frozen nine-participant EEG-eligible modality-ablation cohort"
            },
        },
    )

    per_condition = common_masks.groupby(["participant_id", "condition"])["common_valid"].sum()
    zero_common = [
        {"participant_id": str(participant), "condition": str(condition)}
        for participant, condition in per_condition.index[per_condition == 0]
    ]
    contract = {
        "schema_version": "eeg_eligible_modality_ablation_contract_v1",
        "base_contract": {
            "path": str(base_contract_path.resolve()),
            "sha256": file_sha256(base_contract_path),
        },
        "cohort": "eeg_eligible",
        "participants": list(PARTICIPANTS),
        "participant_count": len(PARTICIPANTS),
        "observation_count": len(labels),
        "window_count": len(windows),
        "common_valid_window_count": int(common_masks["common_valid"].sum()),
        "zero_common_window_observations": zero_common,
        "fold_contract": {
            "folds": 9,
            "train_participants": 7,
            "validation_participants": 1,
            "test_participants": 1,
        },
        "seeds": list(SEEDS),
        "architectures": ["late", "early", "mid", "qformer", "healnet", "mm_lego"],
        "modality_configurations": {
            "full": list(MODALITIES),
            **{
                f"no_{removed}": [modality for modality in MODALITIES if modality != removed]
                for removed in MODALITIES
            },
        },
        "mask_policy": {
            "name": "fixed_five_way_intersection",
            "definition": "A window is valid only when EEG, ECG, eye, head, and video are all valid in the base shared mask. The same decision is copied to every retained modality in every ablation.",
            "source_valid_columns": source_valid_columns,
        },
        "files": {
            "labels": _file_record(label_path, len(labels)),
            "windows": _file_record(window_path, len(windows)),
            "cohorts": _file_record(cohort_path),
            "split_manifest": _file_record(split_path, len(splits)),
            "common_masks": _file_record(mask_path, len(common_masks)),
        },
    }
    _write_json(output_dir / "contract.json", contract)
    return contract


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2] / "artifacts" / "cross_project_alignment_2026-07-16"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-contract-dir", type=Path, default=root / "alignment_contract")
    parser.add_argument("--output-dir", type=Path, default=root / "eeg_eligible_ablation" / "contract")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    contract = build_contract(args.base_contract_dir, args.output_dir)
    print(json.dumps(contract, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
