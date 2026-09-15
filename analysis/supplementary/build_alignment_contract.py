"""Build and validate the immutable shared cross-project alignment contract."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mac.data.video import load_video_index, uniform_clip_frames
from mac.evaluation.alignment import load_split_manifest, validate_alignment_contract
from mac.data.condition_data import aggregate_window_frame


KEYS = ["participant_id", "condition", "condition_window_index"]
TARGETS = ["relaxation", "discomfort"]
MODALITIES = ("eeg", "ecg", "eye", "head", "video")
SEEDS = (20260705, 20260706, 20260707)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    textual = series.astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0}
    )
    return numeric.fillna(textual).fillna(0.0).astype(float).ne(0.0)


def _video_mask(windows: pd.DataFrame, source_manifest: Path) -> pd.Series:
    sources = pd.read_csv(source_manifest, dtype=str).set_index("participant_id")
    output = pd.Series(False, index=windows.index, dtype=bool)
    for participant, indexes in windows.groupby("participant_id", sort=True).groups.items():
        if participant not in sources.index:
            continue
        source = sources.loc[participant]
        index = load_video_index(
            Path(source["video_frames_csv"]) if source.get("video_frames_csv") else None,
            Path(source["session_dir"]) if source.get("session_dir") else None,
            str(participant),
        )
        for row_index in indexes:
            row = windows.loc[row_index]
            frames = uniform_clip_frames(
                index,
                float(row["window_start_unix_ms"]),
                float(row["window_end_unix_ms"]),
                count=16,
            )
            output.loc[row_index] = len(frames) == 16 and all(frame.path.is_file() for frame in frames)
    return output


def build_shared_modality_masks(
    windows: pd.DataFrame,
    project_a_features: pd.DataFrame,
    source_manifest: Path,
) -> pd.DataFrame:
    """Build one row-level availability policy consumed by both projects."""

    if windows.duplicated(KEYS).any() or project_a_features.duplicated(KEYS).any():
        raise ValueError("Window or feature inputs contain duplicate alignment keys")
    merged = windows[KEYS].merge(
        project_a_features,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(windows):
        raise ValueError("Project A feature rows do not cover every shared window")
    masks = windows[KEYS].copy()
    qc_columns = {
        "eeg": "qc_eeg_usable",
        "ecg": "qc_ecg_usable",
        "eye": "qc_eye_usable",
        "head": "qc_head_usable",
    }
    for modality, column in qc_columns.items():
        if column not in merged:
            raise ValueError(f"Project A feature table lacks required mask column {column}")
        masks[f"{modality}_valid"] = _boolean(merged[column]).to_numpy(dtype=bool)
    masks["video_valid"] = _video_mask(windows, source_manifest).to_numpy(dtype=bool)
    return masks


def apply_masks_to_window_features(
    features: pd.DataFrame,
    masks: pd.DataFrame,
) -> pd.DataFrame:
    """Apply shared masks without changing supervision or window membership."""

    merged = features.merge(masks, on=KEYS, how="left", validate="one_to_one", sort=False)
    if merged[[f"{modality}_valid" for modality in MODALITIES]].isna().any().any():
        raise ValueError("Shared modality mask does not cover every feature row")
    for modality in MODALITIES:
        valid_column = f"{modality}_valid"
        feature_columns = [
            column
            for column in features.columns
            if column.startswith(f"{modality}_") and not column.startswith("qc_")
        ]
        if feature_columns:
            merged.loc[~merged[valid_column].astype(bool), feature_columns] = np.nan
        merged[f"mask_{modality}_valid"] = merged[valid_column].astype(bool)
    return merged.drop(columns=[f"{modality}_valid" for modality in MODALITIES])


def aggregate_project_b_handcrafted(masked_features: pd.DataFrame) -> pd.DataFrame:
    """Create the corrected Ridge-CV condition table under the shared masks."""

    return aggregate_window_frame(masked_features)


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def write_contract(
    *,
    labels_path: Path,
    windows_path: Path,
    features_path: Path,
    split_manifest_path: Path,
    source_manifest_path: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_path)
    windows = pd.read_csv(windows_path)
    features = pd.read_csv(features_path)
    if len(labels) != 135 or labels.duplicated(["participant_id", "condition"]).any():
        raise ValueError("Shared labels must contain exactly 135 unique participant-condition rows")
    if len(windows) != 946 or windows.duplicated(KEYS).any():
        raise ValueError("Shared windows must contain exactly 946 unique rows")
    if len(features) != 946 or features.duplicated(KEYS).any():
        raise ValueError("Project A features must contain exactly 946 unique shared windows")
    participants = sorted(labels["participant_id"].astype(str).unique())
    load_split_manifest(split_manifest_path, expected_participants=participants, expected_train_count=13)

    labels_output = output_dir / "condition_labels.csv"
    windows_output = output_dir / "windows.csv"
    split_output = output_dir / "split_manifest.csv"
    shutil.copyfile(labels_path, labels_output)
    shutil.copyfile(windows_path, windows_output)
    shutil.copyfile(split_manifest_path, split_output)

    masks = build_shared_modality_masks(windows, features, source_manifest_path)
    masks_output = output_dir / "shared_modality_masks.csv"
    masks.to_csv(masks_output, index=False)
    masked = apply_masks_to_window_features(features, masks)
    project_a_output = output_dir / "project_a_window_features_masked.csv"
    project_b_window_output = output_dir / "project_b_window_features_masked.csv"
    masked.to_csv(project_a_output, index=False)
    masked.to_csv(project_b_window_output, index=False)
    project_b_condition = aggregate_project_b_handcrafted(masked)
    project_b_condition_output = output_dir / "project_b_condition_features_masked.csv"
    project_b_condition.to_csv(project_b_condition_output, index=False)

    eeg_participants = sorted(
        masks.loc[masks["eeg_valid"], "participant_id"].astype(str).unique()
    )
    cohorts = {
        "all_135": participants,
        "eeg_eligible": eeg_participants,
        "definition": {
            "all_135": "all 15 participants with nine condition labels each",
            "eeg_eligible": "participants with at least one shared-mask-valid EEG window",
        },
    }
    cohorts_output = output_dir / "cohorts.json"
    cohorts_output.write_text(json.dumps(cohorts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    files = {
        "labels": _record(labels_output, output_dir),
        "windows": _record(windows_output, output_dir),
        "cohorts": _record(cohorts_output, output_dir),
        "split_manifest": _record(split_output, output_dir),
        "modality_masks": _record(masks_output, output_dir),
        "project_a_window_features": _record(project_a_output, output_dir),
        "project_b_window_features": _record(project_b_window_output, output_dir),
        "project_b_condition_features": _record(project_b_condition_output, output_dir),
    }
    payload = {
        "schema_version": "cross_project_alignment_v1",
        "participants": participants,
        "participant_count": len(participants),
        "label_count": len(labels),
        "window_count": len(windows),
        "targets": TARGETS,
        "window_seconds": 10.0,
        "fold_contract": {"train_participants": 13, "validation_participants": 1, "test_participants": 1},
        "seeds": list(SEEDS),
        "modalities": list(MODALITIES),
        "mask_valid_windows": {
            modality: int(masks[f"{modality}_valid"].sum()) for modality in MODALITIES
        },
        "files": files,
        "source_files": {
            "labels": {"path": str(labels_path.resolve()), "sha256": _sha256(labels_path)},
            "windows": {"path": str(windows_path.resolve()), "sha256": _sha256(windows_path)},
            "project_a_features": {"path": str(features_path.resolve()), "sha256": _sha256(features_path)},
            "source_manifest": {"path": str(source_manifest_path.resolve()), "sha256": _sha256(source_manifest_path)},
        },
        "notes": {
            "project_a_full_modalities": ["eeg", "ecg", "eye", "head"],
            "project_b_full_modalities": ["eeg", "ecg", "eye", "head", "video"],
            "mask_rule": "same row-level validity manifest is applied before both projects train",
        },
    }
    contract_path = output_dir / "contract.json"
    contract_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validate_contract(output_dir)
    return contract_path


def validate_contract(output_dir: Path) -> dict[str, Any]:
    payload = validate_alignment_contract(output_dir)
    masks = pd.read_csv(output_dir / payload["files"]["modality_masks"]["path"])
    labels = pd.read_csv(output_dir / payload["files"]["labels"]["path"])
    windows = pd.read_csv(output_dir / payload["files"]["windows"]["path"])
    condition = pd.read_csv(output_dir / payload["files"]["project_b_condition_features"]["path"])
    if len(masks) != 946 or masks.duplicated(KEYS).any():
        raise ValueError("Contract modality mask coverage is invalid")
    if len(labels) != 135 or len(condition) != 135:
        raise ValueError("Contract label/condition coverage is invalid")
    if len(windows) != 946:
        raise ValueError("Contract window coverage is invalid")
    observed = {modality: int(_boolean(masks[f"{modality}_valid"]).sum()) for modality in MODALITIES}
    if observed != payload["mask_valid_windows"]:
        raise ValueError(f"Contract mask counts changed: {observed}")
    return {
        "ok": True,
        "contract": str((output_dir / "contract.json").resolve()),
        "labels": len(labels),
        "windows": len(windows),
        "conditions": len(condition),
        "folds": len(load_split_manifest(output_dir / "split_manifest.csv")),
        "mask_valid_windows": observed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--labels", type=Path, default=ROOT / "artifacts/preprocessed/condition_labels.csv")
    build.add_argument("--windows", type=Path, default=ROOT / "artifacts/preprocessed/windows.csv")
    build.add_argument("--project-a-features", type=Path, default=ROOT / "artifacts/features/window_features.csv")
    build.add_argument(
        "--split-manifest",
        type=Path,
        default=ROOT / "artifacts/cross_project_alignment_2026-07-16/recommended_split_manifest.csv",
    )
    build.add_argument("--source-manifest", type=Path, default=ROOT / "artifacts/manifests/source_manifest.csv")
    build.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/cross_project_alignment_2026-07-16/alignment_contract",
    )
    validate = subparsers.add_parser("validate")
    validate.add_argument(
        "--contract-dir",
        type=Path,
        default=ROOT / "artifacts/cross_project_alignment_2026-07-16/alignment_contract",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        path = write_contract(
            labels_path=args.labels.resolve(),
            windows_path=args.windows.resolve(),
            features_path=args.project_a_features.resolve(),
            split_manifest_path=args.split_manifest.resolve(),
            source_manifest_path=args.source_manifest.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        result = validate_contract(path.parent)
    else:
        result = validate_contract(args.contract_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
