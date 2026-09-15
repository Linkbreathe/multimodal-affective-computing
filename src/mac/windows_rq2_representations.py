"""Windows-side RQ2 contract, representations, and temporal encoders.

This module is intentionally separate from the older project experiments.  It
owns only the Windows representation track: the locked FMQ-9 contract,
handcrafted condition vectors, and modality-specific temporal embeddings.  It
does not implement a downstream fusion model and it never reads the WSL
pretrained-encoder project.

The module is designed to be run from ``analysis/supplementary`` with the
``rtml-p002-p016`` environment.  All shared artifacts use paths relative to
the shared root and all canonical keys use participant/condition identifiers,
never local source paths.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable
import zipfile

import numpy as np
import pandas as pd

from real_time_ml.config import ProjectConfig
from real_time_ml.data.index import build_index
from real_time_ml.data.io import discover_session_dir
from real_time_ml.data.labels import parse_condition_labels
from real_time_ml.data.video import load_video_index, uniform_clip_frames
from real_time_ml.evaluation.alignment import file_sha256
from real_time_ml.features.dynamic_texture import (
    DYNAMIC_TEXTURE_COLUMNS,
    DYNAMIC_TEXTURE_VERSION,
    DynamicTextureSettings,
    dynamic_texture_descriptor,
    load_analysis_clip,
)
from real_time_ml.modeling.condition_data import aggregate_window_frame


PARTICIPANTS = (
    "P003",
    "P004",
    "P007",
    "P008",
    "P009",
    "P011",
    "P012",
    "P013",
    "P015",
)
CONDITIONS = tuple(f"C{i}" for i in range(1, 10))
MODALITIES = ("EEG", "ECG", "Eye", "Head", "Video")
SEEDS = (20260705, 20260706, 20260707)
TARGETS = ("relaxation", "discomfort")
SEQUENCE_LENGTH = 8
SOURCE_WINDOW_COUNT = 7
CONTRACT_FILENAMES = (
    "condition_manifest.csv",
    "window_manifest.csv",
    "folds.csv",
    "condition_anchors.csv",
    "rq2_contract.json",
)
AGGREGATION_STATS = (
    "missing_ratio",
    "mean",
    "std",
    "min",
    "max",
    "median",
    "first",
    "last",
    "slope",
)
INDEX_COLUMNS = (
    "contract_hash",
    "environment",
    "run_id",
    "representation_family",
    "modality",
    "fold",
    "seed",
    "split_role",
    "train_participants",
    "validation_participant",
    "test_participant",
    "participant",
    "condition",
    "intensity",
    "frequency",
    "artifact_path",
    "artifact_row",
    "representation_dimension",
    "representation_available",
    "valid_window_count",
    "fallback_used",
    "encoder_checkpoint_id",
    "feature_or_encoder_version",
    "artifact_sha256",
    "representation_key",
    "unavailable_reason",
)
STATS_COLUMNS = tuple(stat for stat in AGGREGATION_STATS)
PARTICIPANT_RANK = {participant: rank for rank, participant in enumerate(PARTICIPANTS)}
CONDITION_RANK = {condition: rank for rank, condition in enumerate(CONDITIONS)}


class DataContractError(RuntimeError):
    """Raised when the declared FMQ-9 source contract cannot be verified."""


@dataclass(frozen=True)
class ContractBundle:
    repo_root: Path
    raw_root: Path
    shared_root: Path
    contract_dir: Path
    results_dir: Path
    labels: pd.DataFrame
    source_windows: pd.DataFrame
    common_windows: pd.DataFrame
    base_features: pd.DataFrame
    window_manifest: pd.DataFrame
    condition_manifest: pd.DataFrame
    folds: pd.DataFrame
    anchors: pd.DataFrame
    contract_hash: str
    source_hashes: dict[str, str]
    inventory: dict[str, dict[str, Any]]
    contract: dict[str, Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != data:
            raise RuntimeError(f"Refusing to overwrite immutable artifact with different bytes: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _write_immutable_text(path: Path, text: str) -> None:
    _write_immutable_bytes(path, text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))


def _csv_bytes(frame: pd.DataFrame, columns: Iterable[str]) -> bytes:
    ordered = list(columns)
    missing = set(ordered) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV frame is missing columns: {sorted(missing)}")
    buffer = io.StringIO()
    frame.loc[:, ordered].to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _write_immutable_csv(path: Path, frame: pd.DataFrame, columns: Iterable[str]) -> None:
    _write_immutable_bytes(path, _csv_bytes(frame, columns))


def _write_immutable_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_immutable_text(path, text)


def _write_immutable_npz(path: Path, **arrays: Any) -> None:
    """Write a timestamp-independent NPZ, reusing an identical existing artifact."""
    expected = {name: np.asarray(value) for name, value in arrays.items()}
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as existing:
                same_names = set(existing.files) == set(expected)
                def equal(left: np.ndarray, right: np.ndarray) -> bool:
                    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
                        return bool(np.allclose(left, right, rtol=1e-7, atol=1e-8, equal_nan=True))
                    return bool(np.array_equal(left, right))
                same_values = same_names and all(equal(existing[name], value) for name, value in expected.items())
            if same_values:
                return
        except Exception:
            pass
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(expected):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, expected[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    _write_immutable_bytes(path, buffer.getvalue())


def _ordered_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    if "participant" in output.columns:
        output["_participant_rank"] = output["participant"].map(PARTICIPANT_RANK)
    elif "participant_id" in output.columns:
        output["_participant_rank"] = output["participant_id"].map(PARTICIPANT_RANK)
    if "condition" in output.columns:
        output["_condition_rank"] = output["condition"].map(CONDITION_RANK)
    sort_columns = [column for column in ("_participant_rank", "_condition_rank", "condition_window_index") if column in output]
    if "fold_index" in output.columns:
        sort_columns = ["fold_index", *sort_columns]
    if "target" in output.columns:
        sort_columns.append("target")
    if sort_columns:
        output = output.sort_values(sort_columns, kind="stable")
    return output.drop(columns=["_participant_rank", "_condition_rank"], errors="ignore").reset_index(drop=True)


def _key_series(frame: pd.DataFrame) -> pd.Series:
    return frame["participant_id"].astype(str) + "+" + frame["condition"].astype(str)


def condition_key(participant: str, condition: str) -> str:
    return f"{participant}+{condition}"


def window_key(participant: str, condition: str, window_id: str) -> str:
    return f"{participant}+{condition}+{window_id}"


def representation_key(
    representation_family: str,
    modality: str,
    fold: str,
    seed: str,
    participant: str,
    condition: str,
) -> str:
    return "+".join((representation_family, modality, fold, seed, participant, condition))


def final_oof_key(model_id: str, fold: str, seed: str, target: str, participant: str, condition: str) -> str:
    return "+".join((model_id, fold, seed, target, participant, condition))


def _legacy_contract_dir(root: Path) -> Path:
    return root / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation" / "contract"


def _relative_source_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _normalise_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _verify_equal_numeric(left: pd.Series, right: pd.Series, name: str) -> list[str]:
    values_left = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    values_right = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    if len(values_left) != len(values_right) or not np.allclose(values_left, values_right, equal_nan=True):
        return [f"{name} differs from the parsed label source"]
    return []


def _audit_source(config: ProjectConfig, root: Path) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    errors: list[str] = []
    legacy = _legacy_contract_dir(root)
    required = {
        "condition_labels": legacy / "condition_labels.csv",
        "windows": legacy / "windows.csv",
        "common_mask": legacy / "common_valid_window_masks.csv",
        "base_features": root / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation" / "project_a" / "inputs" / "project_a_window_features_common.csv",
    }
    for name, path in required.items():
        if not path.exists():
            errors.append(f"Missing required Windows source artifact {name}: {_relative_source_path(path, root)}")
    if errors:
        raise DataContractError("\n".join(errors))

    labels = pd.DataFrame(parse_condition_labels(
        config.path("labels_root"),
        list(PARTICIPANTS),
        [float(value) for value in config.get("conditions.intensities")],
        [float(value) for value in config.get("conditions.frequencies")],
    ))
    labels = labels.rename(columns={"participant_id": "participant_id"})
    source_labels = pd.read_csv(required["condition_labels"], dtype={"participant_id": str, "condition": str})
    expected_label_keys = {(participant, condition) for participant in PARTICIPANTS for condition in CONDITIONS}
    actual_label_keys = set(zip(labels["participant_id"], labels["condition"], strict=False))
    if len(labels) != 81 or actual_label_keys != expected_label_keys:
        errors.append(f"Parsed labels must contain exactly the FMQ-9 81 keys; found {len(labels)} rows and {len(actual_label_keys)} keys")
    source_keys = set(zip(source_labels["participant_id"], source_labels["condition"], strict=False))
    if len(source_labels) != 81 or source_keys != expected_label_keys:
        errors.append(f"Existing condition_labels.csv must contain exactly 81 FMQ-9 keys; found {len(source_labels)} rows and {len(source_keys)} keys")
    if not errors:
        left = labels.sort_values(["participant_id", "condition"]).reset_index(drop=True)
        right = source_labels.sort_values(["participant_id", "condition"]).reset_index(drop=True)
        for column in ("presentation_position", "intensity", "frequency", "relaxation", "discomfort"):
            errors.extend(_verify_equal_numeric(left[column], right[column], column))

    source_windows = pd.read_csv(required["windows"], dtype={"participant_id": str, "condition": str, "window_id": str})
    mask = pd.read_csv(required["common_mask"], dtype={"participant_id": str, "condition": str})
    base_features = pd.read_csv(required["base_features"], dtype={"participant_id": str, "condition": str, "window_id": str})
    source_windows["condition_window_index"] = pd.to_numeric(source_windows["condition_window_index"], errors="coerce").astype("Int64")
    mask["condition_window_index"] = pd.to_numeric(mask["condition_window_index"], errors="coerce").astype("Int64")
    base_features["condition_window_index"] = pd.to_numeric(base_features["condition_window_index"], errors="coerce").astype("Int64")
    source_key = source_windows["participant_id"].astype(str) + "+" + source_windows["condition"] + "+" + source_windows["condition_window_index"].astype(str)
    mask_key = mask["participant_id"].astype(str) + "+" + mask["condition"] + "+" + mask["condition_window_index"].astype(str)
    base_key = base_features["participant_id"].astype(str) + "+" + base_features["condition"] + "+" + base_features["condition_window_index"].astype(str)
    if len(source_windows) != 567 or source_key.nunique() != 567:
        errors.append(f"Source windows must contain exactly 567 unique keys; found {len(source_windows)} rows and {source_key.nunique()} keys")
    per_condition = source_windows.groupby(["participant_id", "condition"], sort=False).size()
    if len(per_condition) != 81 or not (per_condition == 7).all():
        errors.append("Every FMQ-9 participant-condition must have exactly seven source windows")
    if len(mask) != 567 or mask_key.nunique() != 567 or set(source_key) != set(mask_key):
        errors.append("The common-valid mask must align one-to-one with all 567 source windows")
    if len(base_features) != 567 or base_key.nunique() != 567 or set(source_key) != set(base_key):
        errors.append("The existing handcrafted feature table must align one-to-one with all 567 source windows")
    mask_columns = ("common_valid", "eeg_valid", "ecg_valid", "eye_valid", "head_valid", "video_valid")
    missing_mask_columns = set(mask_columns) - set(mask.columns)
    if missing_mask_columns:
        errors.append(f"Common-valid mask is missing columns: {sorted(missing_mask_columns)}")
    else:
        for column in mask_columns:
            mask[column] = _normalise_bool_series(mask[column])
        common_count = int(mask["common_valid"].sum())
        if common_count != 545:
            errors.append(f"Expected exactly 545 common-valid windows, found {common_count}")
        common = mask.loc[mask["common_valid"]].copy()
        for column in ("eeg_valid", "ecg_valid", "eye_valid", "head_valid", "video_valid"):
            if not bool(common[column].all()):
                errors.append(f"Common-valid rows contain a false {column} flag")
        zero_conditions = set(
            tuple(values)
            for values in mask.groupby(["participant_id", "condition"], sort=False)["common_valid"].sum().reset_index().loc[lambda frame: frame["common_valid"] == 0, ["participant_id", "condition"]].itertuples(index=False, name=None)
        )
        if zero_conditions != {("P004", "C6")}:
            errors.append(f"All-missing participant-condition inventory must be exactly P004/C6; found {sorted(zero_conditions)}")

    feature_columns = {
        modality: tuple(
            column for column in base_features.columns
            if column.startswith(prefix)
        )
        for modality, prefix in (("EEG", "eeg_"), ("ECG", "ecg_"), ("Eye", "eye_"), ("Head", "head_"))
    }
    expected_feature_counts = {"EEG": 33, "ECG": 18, "Eye": 11, "Head": 18}
    for modality, expected_count in expected_feature_counts.items():
        if len(feature_columns[modality]) != expected_count:
            errors.append(f"{modality} handcrafted source feature count changed: expected {expected_count}, found {len(feature_columns[modality])}")
    if len(DYNAMIC_TEXTURE_COLUMNS) != 12 or DYNAMIC_TEXTURE_VERSION != "dynamic_texture_v1":
        errors.append("The dynamic-texture descriptor contract is not dynamic_texture_v1 with 12 columns")
    if source_windows["condition_window_index"].min() != 0 or source_windows["condition_window_index"].max() != 6:
        errors.append("The verified source sequence slots must be condition_window_index 0 through 6")

    inventory: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    raw_root = config.path("raw_root")
    for participant in PARTICIPANTS:
        participant_dir = raw_root / participant
        session_dir = discover_session_dir(participant_dir) if participant_dir.exists() else None
        video_csv = session_dir / "video_frames.csv" if session_dir and (session_dir / "video_frames.csv").exists() else None
        required_streams = {
            name: session_dir / filename if session_dir and (session_dir / filename).exists() else None
            for name, filename in (("samples", "samples.csv"), ("eye", "eye_tracking.csv"), ("video", "video_frames.csv"))
        }
        missing_streams = [name for name, path in required_streams.items() if path is None]
        if missing_streams:
            errors.append(f"{participant} is missing required raw streams: {missing_streams}")
        if video_csv is None:
            video_index = load_video_index(None, session_dir, participant)
        else:
            video_index = load_video_index(video_csv, session_dir, participant)
        if video_index.reason != "ok" or len(video_index.frames) < 16:
            errors.append(f"{participant} video inventory is unusable: reason={video_index.reason}, frames={len(video_index.frames)}")
        inventory[participant] = {
            "participant_dir": _relative_source_path(participant_dir, root),
            "session_dir": session_dir.relative_to(raw_root).as_posix() if session_dir else None,
            "video_frames_csv": video_csv.relative_to(raw_root).as_posix() if video_csv else None,
            "video_timestamp_source": video_index.timestamp_source,
            "video_index_reason": video_index.reason,
            "video_frame_count": len(video_index.frames),
        }
        for name, path in required_streams.items():
            if path is not None and path.exists():
                source_hashes[f"{participant}:{name}"] = file_sha256(path)
    for name, path in required.items():
        source_hashes[f"contract_source:{name}"] = file_sha256(path)
    source_hashes["dynamic_texture_module"] = file_sha256(Path(__file__).resolve().parents[0] / "features" / "dynamic_texture.py")

    if errors:
        raise DataContractError("\n".join(errors))

    common_windows = source_windows.merge(
        mask,
        on=["participant_id", "condition", "condition_window_index"],
        how="inner",
        validate="one_to_one",
    )
    common_windows = common_windows.loc[common_windows["common_valid"]].copy()
    if len(common_windows) != 545:
        raise DataContractError(f"Post-merge common-valid inventory changed to {len(common_windows)} rows")
    return labels, source_windows, common_windows, base_features, inventory, source_hashes


def _build_condition_manifest(labels: pd.DataFrame) -> pd.DataFrame:
    output = labels.loc[:, [
        "participant_id", "condition", "presentation_position", "intensity", "frequency",
        "relaxation", "discomfort",
    ]].rename(columns={
        "participant_id": "participant",
        "presentation_position": "presentation_order",
        "relaxation": "normalized_relaxation",
        "discomfort": "normalized_discomfort",
    })
    output["fallback_status"] = np.where(
        (output["participant"] == "P004") & (output["condition"] == "C6"),
        "condition_only",
        "none",
    )
    output = _ordered_frame(output, output.columns)
    columns = (
        "participant", "condition", "presentation_order", "intensity", "frequency",
        "normalized_relaxation", "normalized_discomfort", "fallback_status",
    )
    return output.loc[:, columns]


def _build_window_manifest(common_windows: pd.DataFrame) -> pd.DataFrame:
    output = common_windows.loc[:, [
        "participant_id", "condition", "window_id", "condition_window_index",
        "eeg_valid", "ecg_valid", "eye_valid", "head_valid", "video_valid",
    ]].rename(columns={"participant_id": "participant"})
    output = _ordered_frame(output, output.columns)
    columns = (
        "participant", "condition", "window_id", "condition_window_index",
        "eeg_valid", "ecg_valid", "eye_valid", "head_valid", "video_valid",
    )
    return output.loc[:, columns]


def _build_folds() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold_index, test_participant in enumerate(PARTICIPANTS, start=1):
        validation_participant = PARTICIPANTS[fold_index % len(PARTICIPANTS)]
        train = [participant for participant in PARTICIPANTS if participant not in {test_participant, validation_participant}]
        for participant in PARTICIPANTS:
            role = "test" if participant == test_participant else "validation" if participant == validation_participant else "train"
            records.append({
                "fold_index": fold_index,
                "participant": participant,
                "role": role,
                "train_participants": ";".join(train),
                "validation_participant": validation_participant,
                "test_participant": test_participant,
            })
    return pd.DataFrame(records, columns=[
        "fold_index", "participant", "role", "train_participants",
        "validation_participant", "test_participant",
    ])


def _build_anchors(labels: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    label_lookup = labels.set_index(["participant_id", "condition"])
    for fold_index in range(1, len(PARTICIPANTS) + 1):
        fold = folds.loc[folds["fold_index"] == fold_index].iloc[0]
        train = str(fold["train_participants"]).split(";")
        for condition in CONDITIONS:
            for target in TARGETS:
                values = [float(label_lookup.loc[(participant, condition), target]) for participant in train]
                records.append({
                    "fold_index": fold_index,
                    "condition": condition,
                    "target": target,
                    "condition_anchor": float(np.mean(values)),
                    "train_participants": ";".join(train),
                    "training_participant_count": len(values),
                })
    return pd.DataFrame(records, columns=[
        "fold_index", "condition", "target", "condition_anchor",
        "train_participants", "training_participant_count",
    ])


def _comparison_contract() -> dict[str, Any]:
    return {
        "family_1_representation_comparisons": {
            "correction": "Holm",
            "two_sided": True,
            "delta_definition": "MAE(left) - MAE(right)",
            "negative_delta_means_left_better": True,
            "comparisons": [
                {"comparison_id": "F1_1", "left": "frozen_simplex_full5", "right": "handcrafted_simplex_full5", "target": "relaxation"},
                {"comparison_id": "F1_2", "left": "frozen_simplex_full5", "right": "handcrafted_simplex_full5", "target": "discomfort"},
                {"comparison_id": "F1_3", "left": "frozen_simplex_full5", "right": "temporal_1dcnn_simplex_full5", "target": "relaxation"},
                {"comparison_id": "F1_4", "left": "frozen_simplex_full5", "right": "temporal_1dcnn_simplex_full5", "target": "discomfort"},
            ],
        },
        "family_2_fusion_comparisons": {
            "correction": "Holm",
            "two_sided": True,
            "delta_definition": "MAE(left) - MAE(right)",
            "negative_delta_means_left_better": True,
            "comparisons": [
                {"comparison_id": "F2_1", "left": "frozen_simplex_full5", "right": "frozen_concat_ridge_full5", "target": "relaxation"},
                {"comparison_id": "F2_2", "left": "frozen_simplex_full5", "right": "frozen_concat_ridge_full5", "target": "discomfort"},
                {"comparison_id": "F2_3", "left": "frozen_healnet_full5", "right": "frozen_concat_ridge_full5", "target": "relaxation"},
                {"comparison_id": "F2_4", "left": "frozen_healnet_full5", "right": "frozen_concat_ridge_full5", "target": "discomfort"},
            ],
        },
    }


def _make_contract(
    root: Path,
    labels: pd.DataFrame,
    source_windows: pd.DataFrame,
    common_windows: pd.DataFrame,
    inventory: dict[str, dict[str, Any]],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    base_feature_counts = {
        "EEG": int(sum(column.startswith("eeg_") for column in pd.read_csv(root / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation" / "project_a" / "inputs" / "project_a_window_features_common.csv", nrows=0).columns)),
        "ECG": int(sum(column.startswith("ecg_") for column in pd.read_csv(root / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation" / "project_a" / "inputs" / "project_a_window_features_common.csv", nrows=0).columns)),
        "Eye": int(sum(column.startswith("eye_") for column in pd.read_csv(root / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation" / "project_a" / "inputs" / "project_a_window_features_common.csv", nrows=0).columns)),
        "Head": int(sum(column.startswith("head_") for column in pd.read_csv(root / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation" / "project_a" / "inputs" / "project_a_window_features_common.csv", nrows=0).columns)),
    }
    return {
        "schema_version": "rq2_windows_contract_v1",
        "analysis_id": "rq2_fmq9",
        "environment_boundary": {
            "windows_project_root": root.name,
            "shared_root_role": "cross_environment_contract_and_results",
            "windows_outputs": ["handcrafted", "temporal_1dcnn"],
            "pretrained_encoder_owner": "wsl",
            "windows_must_not_access_wsl_project": True,
        },
        "cohort": {
            "participant_order": list(PARTICIPANTS),
            "participant_count": 9,
            "conditions": list(CONDITIONS),
            "condition_count_per_participant": 9,
            "condition_rating_count": 81,
            "primary_cohort": "FMQ-9",
            "secondary_full_15_participant_cohort_allowed": True,
            "secondary_results_must_not_mix_with_primary": True,
        },
        "targets": {
            "names": list(TARGETS),
            "normalization": "(raw_1_to_7 - 1) / 6",
            "range": [0.0, 1.0],
        },
        "modalities": list(MODALITIES),
        "representation_families": {
            "windows": ["handcrafted", "temporal_1dcnn"],
            "all_cross_environment_families": ["handcrafted", "temporal_1dcnn", "pretrained"],
            "pretrained_source": "shared_root/wsl_results",
            "handcrafted": {
                "aggregation": "nine fixed window statistics per source feature",
                "statistics": list(AGGREGATION_STATS),
                "source_window_policy": "only window_manifest rows",
                "video_source": DYNAMIC_TEXTURE_VERSION,
                "feature_counts": base_feature_counts | {"Video": len(DYNAMIC_TEXTURE_COLUMNS)},
                "missing_statistic_policy": "non-finite aggregate values encoded as 0; missing_ratio retained",
            },
            "temporal_1dcnn": {
                "sequence_length": SEQUENCE_LENGTH,
                "source_window_slots": list(range(SOURCE_WINDOW_COUNT)),
                "structural_padding_slot": SOURCE_WINDOW_COUNT,
                "missing_window_policy": "zero-filled structural slot; no substitution from another window",
                "feature_counts": base_feature_counts | {"Video": len(DYNAMIC_TEXTURE_COLUMNS) + 1},
                "video_validity_feature": "video_dynamic_validity",
                "fold_local_standardization": True,
                "trainable_encoder_is_modality_specific": True,
                "temporary_head_targets": ["relaxation_residual", "discomfort_residual"],
            },
        },
        "folds": {
            "count": 9,
            "test_rule": "participant i in ordered participant list",
            "validation_rule": "next participant with wraparound",
            "training_participant_count": 7,
            "anchor_source": "seven training participants only",
            "anchors_saved_once_and_reused_across_seeds": True,
        },
        "seeds": list(SEEDS),
        "residual_target_rule": {
            "condition_anchor": "mean training-participant rating for same condition",
            "true_residual": "true_rating - condition_anchor",
            "final_prediction": "clip(condition_anchor + predicted_residual, 0, 1)",
            "validation_and_test_ratings_excluded_from_anchor": True,
        },
        "models": {
            "frozen_fusion_models": [
                "frozen_simplex_full5",
                "frozen_concat_ridge_full5",
                "frozen_healnet_full5",
            ],
            "full_five_modalities": list(MODALITIES),
            "no_video_comparison": "full-five/no-video comparison is required in WSL",
            "frozen_simplex_run_identity": "the same run_id and OOF predictions must serve Family 1 and Family 2",
            "controlled_representation_downstream": "one shared WSL Ridge-expert and target-specific Simplex pipeline",
            "windows_must_not_fit_final_fusion": True,
        },
        "downstream_pipeline": {
            "pca_allocation": {"EEG": 3, "ECG": 4, "Eye": 1, "Head": 1, "Video": 3},
            "ridge_alpha_grid": [10, 100, 1000],
            "simplex": {
                "weight_grid": [0.25, 0.5, 0.75, 1.0],
                "active_expert_min_weight": 0.02,
                "missing_expert_renormalization": True,
                "expert_correction_clip": [-0.2, 0.2],
                "correction_scale_grid": [0.25, 0.5, 0.75, 1.0],
                "final_prediction_clip": [0.0, 1.0],
                "target_specific": True,
            },
            "aggregation_of_cached_pretrained_representations": "mean_over_valid_windows",
            "primary_metric": "participant_macro_mae",
            "secondary_metrics": [
                "participant_macro_rmse",
                "condition_macro_mae",
                "participant_macro_bias",
                "participant_macro_median_absolute_error",
            ],
            "bootstrap": {"unit": "participant", "replicates": 10000, "random_seed": 20260705},
            "sign_flip_test": {
                "type": "exact_two_sided_participant_sign_flip",
                "unit": "participant",
                "enumerate_all_sign_assignments": True,
                "statistic": "mean participant-level paired MAE delta",
            },
            "comparison_families": _comparison_contract(),
        },
        "canonical_keys": {
            "condition": "participant + condition",
            "window": "participant + condition + window_id",
            "representation": "representation_family + modality + fold + seed + participant + condition",
            "final_oof": "model_id + fold + seed + target + participant + condition",
            "machine_specific_source_paths_excluded": True,
        },
        "completeness": {
            "source_window_rows": len(source_windows),
            "common_valid_window_rows": len(common_windows),
            "condition_manifest_rows": 81,
            "window_manifest_rows": 545,
            "handcrafted_rows": "5 modalities x 81 keys in index; P004/C6 unavailable",
            "temporal_rows": "9 folds x 3 seeds x 5 modalities x 81 keys in index; P004/C6 unavailable",
        },
        "fallback": {
            "participant": "P004",
            "condition": "C6",
            "status": "condition_only",
            "representation_vector": "unavailable",
            "condition_anchor_prediction_returned_by_downstream": True,
        },
        "source_audit": {
            "source_feature_artifact": "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/project_a/inputs/project_a_window_features_common.csv",
            "source_contract_directory": "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract",
            "inventory": inventory,
            "source_sha256": source_hashes,
        },
        "artifact_format": {
            "vectors": "compressed NPZ with participant, condition, feature_names, vectors arrays",
            "text_encoding": "UTF-8",
            "line_endings": "LF",
            "row_order": "participant order then C1-C9 then condition_window_index",
            "index_paths": "relative to shared root with forward slashes",
        },
        "contract_hash_definition": "SHA-256 over sorted filename<TAB>individual_sha256 lines for the five contract files; rq2_contract.sha256 excluded",
    }


def _write_data_error(shared_root: Path, message: str) -> Path:
    results = shared_root / "windows_results"
    results.mkdir(parents=True, exist_ok=True)
    text = "# Windows data error\n\nThe locked FMQ-9 preflight failed. No model training or hand-off marker was produced.\n\n## Exact mismatches\n\n" + "\n".join(f"- {line}" for line in message.splitlines()) + "\n"
    path = results / "WINDOWS_DATA_ERROR.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def prepare_contract(config: ProjectConfig, shared_root: str | Path) -> ContractBundle:
    root = repo_root()
    shared = Path(shared_root).resolve()
    contract_dir = shared / "contract"
    results_dir = shared / "windows_results"
    for path in (contract_dir, results_dir, shared / "wsl_results", shared / "combined"):
        path.mkdir(parents=True, exist_ok=True)
    labels, source_windows, common_windows, base_features, inventory, source_hashes = _audit_source(config, root)
    condition_manifest = _build_condition_manifest(labels)
    window_manifest = _build_window_manifest(common_windows)
    folds = _build_folds()
    anchors = _build_anchors(labels, folds)
    contract = _make_contract(root, labels, source_windows, common_windows, inventory, source_hashes)

    frames = {
        "condition_manifest.csv": (condition_manifest, (
            "participant", "condition", "presentation_order", "intensity", "frequency",
            "normalized_relaxation", "normalized_discomfort", "fallback_status",
        )),
        "window_manifest.csv": (window_manifest, (
            "participant", "condition", "window_id", "condition_window_index",
            "eeg_valid", "ecg_valid", "eye_valid", "head_valid", "video_valid",
        )),
        "folds.csv": (folds, (
            "fold_index", "participant", "role", "train_participants",
            "validation_participant", "test_participant",
        )),
        "condition_anchors.csv": (anchors, (
            "fold_index", "condition", "target", "condition_anchor",
            "train_participants", "training_participant_count",
        )),
    }
    for filename, (frame, columns) in frames.items():
        _write_immutable_csv(contract_dir / filename, frame, columns)
    _write_immutable_json(contract_dir / "rq2_contract.json", contract)

    individual_hashes = {filename: file_sha256(contract_dir / filename) for filename in CONTRACT_FILENAMES}
    canonical_lines = "\n".join(f"{filename}\t{individual_hashes[filename]}" for filename in sorted(CONTRACT_FILENAMES)) + "\n"
    contract_hash = _sha256_text(canonical_lines)
    hash_text = canonical_lines + f"contract_hash\t{contract_hash}\n"
    _write_immutable_text(contract_dir / "rq2_contract.sha256", hash_text)
    return ContractBundle(
        repo_root=root,
        raw_root=config.path("raw_root"),
        shared_root=shared,
        contract_dir=contract_dir,
        results_dir=results_dir,
        labels=labels,
        source_windows=source_windows,
        common_windows=common_windows,
        base_features=base_features,
        window_manifest=window_manifest,
        condition_manifest=condition_manifest,
        folds=folds,
        anchors=anchors,
        contract_hash=contract_hash,
        source_hashes=source_hashes,
        inventory=inventory,
        contract=contract,
    )


def load_contract_hash(contract_dir: str | Path) -> str:
    directory = Path(contract_dir)
    path = directory / "rq2_contract.sha256"
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    values = dict(line.split("\t", 1) for line in lines if "\t" in line)
    missing = set(CONTRACT_FILENAMES) - set(values)
    if missing or "contract_hash" not in values:
        raise DataContractError(f"rq2_contract.sha256 is missing entries: {sorted(missing)}")
    actual = {filename: file_sha256(directory / filename) for filename in CONTRACT_FILENAMES}
    if any(actual[name] != values[name] for name in CONTRACT_FILENAMES):
        raise DataContractError("One or more immutable contract file hashes do not match rq2_contract.sha256")
    canonical = "\n".join(f"{filename}\t{actual[filename]}" for filename in sorted(CONTRACT_FILENAMES)) + "\n"
    calculated = _sha256_text(canonical)
    if calculated != values["contract_hash"]:
        raise DataContractError("The calculated contract_hash does not match rq2_contract.sha256")
    return calculated


def _source_feature_columns(base_features: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    return {
        "EEG": tuple(column for column in base_features.columns if column.startswith("eeg_")),
        "ECG": tuple(column for column in base_features.columns if column.startswith("ecg_")),
        "Eye": tuple(column for column in base_features.columns if column.startswith("eye_")),
        "Head": tuple(column for column in base_features.columns if column.startswith("head_")),
        "Video": tuple(DYNAMIC_TEXTURE_COLUMNS),
    }


def _video_indexes(bundle: ContractBundle) -> dict[str, Any]:
    indexes: dict[str, Any] = {}
    for participant, record in bundle.inventory.items():
        session_dir = bundle.raw_root / str(record["session_dir"]) if record.get("session_dir") else None
        video_csv = bundle.raw_root / str(record["video_frames_csv"]) if record.get("video_frames_csv") else None
        indexes[participant] = load_video_index(video_csv, session_dir, participant)
    return indexes


def build_dynamic_texture_table(bundle: ContractBundle) -> pd.DataFrame:
    """Build or verify dynamic_texture_v1 only for the 545 locked windows."""
    cache_path = bundle.results_dir / "cache" / "dynamic_texture_v1_window.csv"
    key_columns = ["participant", "condition", "condition_window_index"]
    if cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"participant": str, "condition": str})
        expected_columns = key_columns + list(DYNAMIC_TEXTURE_COLUMNS) + ["video_dynamic_validity"]
        if len(cached) == 545 and set(expected_columns) <= set(cached.columns):
            cached = _ordered_frame(cached, cached.columns)
            return cached.loc[:, expected_columns]
        raise DataContractError(f"Existing dynamic-texture cache does not match the locked 545-row schema: {cache_path}")

    indexes = _video_indexes(bundle)
    settings = DynamicTextureSettings()
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for number, row in enumerate(bundle.common_windows.itertuples(index=False), start=1):
        participant = str(row.participant_id)
        index = indexes[participant]
        frames = uniform_clip_frames(index, float(row.start_unix_ms), float(row.end_unix_ms), settings.frames_per_window)
        if len(frames) != settings.frames_per_window:
            raise DataContractError(
                f"{participant}/{row.condition}/window {row.condition_window_index} cannot provide {settings.frames_per_window} distinct video frames"
            )
        analysis_frames, timestamps_ms, _ = load_analysis_clip(frames, settings)
        descriptors = dynamic_texture_descriptor(analysis_frames, timestamps_ms, settings)
        records.append({
            "participant": participant,
            "condition": str(row.condition),
            "condition_window_index": int(row.condition_window_index),
            **descriptors,
            "video_dynamic_validity": 1.0,
        })
        if number == 1 or number % 25 == 0 or number == len(bundle.common_windows):
            elapsed = time.perf_counter() - started
            print(f"dynamic_texture {number}/{len(bundle.common_windows)} ({elapsed:.1f}s)", flush=True)
    output = pd.DataFrame(records)
    expected_columns = key_columns + list(DYNAMIC_TEXTURE_COLUMNS) + ["video_dynamic_validity"]
    if len(output) != 545 or output[expected_columns].isna().any().any():
        raise DataContractError("dynamic_texture_v1 generation did not produce 545 complete rows")
    output = _ordered_frame(output, output.columns).loc[:, expected_columns]
    _write_immutable_csv(cache_path, output, expected_columns)
    return output


def _feature_frame_for_modality(bundle: ContractBundle, dynamic: pd.DataFrame, modality: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    source_columns = _source_feature_columns(bundle.base_features)
    key_columns = ["participant_id", "condition", "condition_window_index"]
    metadata_columns = ["relaxation", "discomfort", "presentation_position", "intensity", "frequency", "condition_window_count"]
    common_keys = bundle.common_windows.loc[:, key_columns]
    common_base = bundle.base_features.merge(common_keys, on=key_columns, how="inner", validate="one_to_one")
    if modality == "Video":
        frame = common_base.loc[:, [*key_columns, "presentation_position", "intensity", "frequency", "condition_window_count", "relaxation", "discomfort"]].merge(
            dynamic,
            left_on=["participant_id", "condition", "condition_window_index"],
            right_on=["participant", "condition", "condition_window_index"],
            how="left",
            validate="one_to_one",
        )
        frame = frame.drop(columns=["participant"])
        feature_columns = tuple(DYNAMIC_TEXTURE_COLUMNS)
    else:
        feature_columns = source_columns[modality]
        frame = common_base.loc[:, key_columns + metadata_columns + list(feature_columns)].copy()
    return frame, feature_columns


def _condition_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    labels: pd.DataFrame,
) -> tuple[np.ndarray, list[tuple[str, str]], list[str]]:
    columns = [
        "participant_id", "condition", "condition_window_index", "relaxation", "discomfort",
        "presentation_position", "intensity", "frequency", "condition_window_count",
        *feature_columns,
    ]
    aggregated = aggregate_window_frame(frame.loc[:, columns])
    aggregated = _ordered_frame(aggregated, aggregated.columns)
    vector_columns = tuple(f"{column}__{stat}" for column in feature_columns for stat in AGGREGATION_STATS)
    missing = set(vector_columns) - set(aggregated.columns)
    if missing:
        raise DataContractError(f"Condition aggregation lost expected feature columns: {sorted(missing)[:5]}")
    keys = [(str(row.participant_id), str(row.condition)) for row in aggregated.itertuples(index=False)]
    matrix = aggregated.loc[:, vector_columns].to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    expected_keys = set(zip(labels["participant_id"], labels["condition"], strict=False)) - {("P004", "C6")}
    if set(keys) != expected_keys:
        raise DataContractError(f"Condition representation keys do not match the 80 available keys: {len(keys)}")
    return matrix, keys, list(vector_columns)


def _relative_shared(path: Path, shared_root: Path) -> str:
    return path.resolve().relative_to(shared_root.resolve()).as_posix()


def _index_row(
    *,
    contract_hash: str,
    run_id: str,
    family: str,
    modality: str,
    fold: str,
    seed: str,
    split_role: str,
    train_participants: str,
    validation_participant: str,
    test_participant: str,
    participant: str,
    condition: str,
    intensity: float,
    frequency: float,
    artifact_path: str,
    artifact_row: int | None,
    dimension: int,
    available: bool,
    valid_window_count: int,
    fallback_used: bool,
    checkpoint_id: str,
    version: str,
    artifact_hash: str,
    unavailable_reason: str = "",
) -> dict[str, Any]:
    return {
        "contract_hash": contract_hash,
        "environment": "windows",
        "run_id": run_id,
        "representation_family": family,
        "modality": modality,
        "fold": fold,
        "seed": seed,
        "split_role": split_role,
        "train_participants": train_participants,
        "validation_participant": validation_participant,
        "test_participant": test_participant,
        "participant": participant,
        "condition": condition,
        "intensity": float(intensity),
        "frequency": float(frequency),
        "artifact_path": artifact_path,
        "artifact_row": "" if artifact_row is None else int(artifact_row),
        "representation_dimension": int(dimension),
        "representation_available": bool(available),
        "valid_window_count": int(valid_window_count),
        "fallback_used": bool(fallback_used),
        "encoder_checkpoint_id": checkpoint_id,
        "feature_or_encoder_version": version,
        "artifact_sha256": artifact_hash,
        "representation_key": representation_key(family, modality, fold, seed, participant, condition),
        "unavailable_reason": unavailable_reason,
    }


def export_handcrafted(
    bundle: ContractBundle,
    run_id: str,
    dynamic: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = bundle.condition_manifest.set_index(["participant", "condition"])
    window_counts = bundle.window_manifest.groupby(["participant", "condition"], sort=False).size().to_dict()
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for modality in MODALITIES:
        frame, feature_columns = _feature_frame_for_modality(bundle, dynamic, modality)
        matrix, keys, vector_columns = _condition_matrix(frame, feature_columns, bundle.labels)
        artifact = bundle.results_dir / "representations" / "handcrafted" / f"{modality.lower()}.npz"
        _write_immutable_npz(
            artifact,
            participants=np.asarray([key[0] for key in keys], dtype="U4"),
            conditions=np.asarray([key[1] for key in keys], dtype="U2"),
            vectors=matrix,
            feature_names=np.asarray(vector_columns, dtype="U128"),
            representation_family=np.asarray(["handcrafted"]),
            modality=np.asarray([modality]),
            feature_version=np.asarray([f"handcrafted_aggregate_v1|{DYNAMIC_TEXTURE_VERSION}"], dtype="U64"),
        )
        artifact_hash = file_sha256(artifact)
        key_to_row = {key: row for row, key in enumerate(keys)}
        for participant in PARTICIPANTS:
            for condition in CONDITIONS:
                label = labels.loc[(participant, condition)]
                available = (participant, condition) in key_to_row
                rows.append(_index_row(
                    contract_hash=bundle.contract_hash,
                    run_id=run_id,
                    family="handcrafted",
                    modality=modality,
                    fold="global",
                    seed="deterministic",
                    split_role="global",
                    train_participants="",
                    validation_participant="",
                    test_participant="",
                    participant=participant,
                    condition=condition,
                    intensity=float(label["intensity"]),
                    frequency=float(label["frequency"]),
                    artifact_path=_relative_shared(artifact, bundle.shared_root),
                    artifact_row=key_to_row.get((participant, condition)),
                    dimension=matrix.shape[1],
                    available=available,
                    valid_window_count=int(window_counts.get((participant, condition), 0)),
                    fallback_used=not available,
                    checkpoint_id="",
                    version=f"handcrafted_aggregate_v1|{DYNAMIC_TEXTURE_VERSION}",
                    artifact_hash=artifact_hash,
                    unavailable_reason="all_missing_common_windows" if not available else "",
                ))
        audits.append({
            "representation_family": "handcrafted",
            "modality": modality,
            "fold": "global",
            "seed": "deterministic",
            "feature_or_encoder_version": f"handcrafted_aggregate_v1|{DYNAMIC_TEXTURE_VERSION}",
            "representation_dimension": int(matrix.shape[1]),
            "index_rows": 81,
            "artifact_rows": int(matrix.shape[0]),
            "missing_index_rows": 1,
            "missing_keys": "P004+C6",
            "source_hashes": json.dumps(bundle.source_hashes, sort_keys=True),
            "artifact_sha256": artifact_hash,
            "status": "complete",
            "warnings": "P004/C6 retained as unavailable condition-only fallback",
        })
    return rows, audits


def _modality_feature_frame(bundle: ContractBundle, dynamic: pd.DataFrame, modality: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame, columns = _feature_frame_for_modality(bundle, dynamic, modality)
    if modality == "Video":
        frame = frame.copy()
        frame["video_dynamic_validity"] = 1.0
        columns = tuple(columns) + ("video_dynamic_validity",)
    return frame, columns


def _fit_scaler(frame: pd.DataFrame, feature_columns: tuple[str, ...], train_participants: list[str]) -> tuple[np.ndarray, np.ndarray]:
    training = frame.loc[frame["participant_id"].isin(train_participants), list(feature_columns)]
    values = training.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    means = np.zeros(values.shape[1], dtype=np.float64)
    scales = np.ones(values.shape[1], dtype=np.float64)
    for index in range(values.shape[1]):
        finite = values[:, index][np.isfinite(values[:, index])]
        if finite.size:
            means[index] = float(np.mean(finite))
            spread = float(np.std(finite, ddof=0))
            scales[index] = spread if spread > 1e-12 else 1.0
    return means.astype(np.float32), scales.astype(np.float32)


def _sequence_arrays(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    means: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    ordered = frame.sort_values(["participant_id", "condition", "condition_window_index"], kind="stable")
    sequences: list[np.ndarray] = []
    keys: list[tuple[str, str]] = []
    for (participant, condition), group in ordered.groupby(["participant_id", "condition"], sort=False):
        sequence = np.zeros((len(feature_columns), SEQUENCE_LENGTH), dtype=np.float32)
        for row in group.itertuples(index=False):
            slot = int(row.condition_window_index)
            if not 0 <= slot < SOURCE_WINDOW_COUNT:
                raise DataContractError(f"Invalid temporal slot {slot} for {participant}/{condition}")
            values = np.asarray([getattr(row, column) for column in feature_columns], dtype=float)
            values = (values - means) / scales
            values[~np.isfinite(values)] = 0.0
            sequence[:, slot] = values.astype(np.float32)
        sequences.append(sequence)
        keys.append((str(participant), str(condition)))
    if not sequences:
        return np.empty((0, len(feature_columns), SEQUENCE_LENGTH), dtype=np.float32), []
    return np.stack(sequences).astype(np.float32), keys


def _code_version(root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True,
        ).stdout.strip()
    except OSError:
        commit = "unknown"
    return f"git:{commit or 'unknown'}|rq2_windows_sha256:{file_sha256(Path(__file__))}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _temporal_model(feature_count: int):
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_count = feature_count
            self.conv1 = nn.Conv1d(feature_count, 16 * feature_count, kernel_size=3, padding=1, groups=feature_count)
            self.conv2 = nn.Conv1d(16 * feature_count, 32 * feature_count, kernel_size=3, padding=1, groups=feature_count)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(0.30)
            self.embedding_dimension = 64 * feature_count

        def forward(self, values):
            values = self.dropout(self.pool(self.relu(self.conv1(values))))
            values = self.dropout(self.pool(self.relu(self.conv2(values))))
            return values.reshape(values.shape[0], -1)

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(64 * feature_count, 64),
                nn.ReLU(),
                nn.Dropout(0.30),
                nn.Linear(64, 2),
            )

        def forward(self, values):
            return self.layers(values)

    return Encoder(), Head()


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _fold_info(bundle: ContractBundle, fold_index: int) -> tuple[list[str], str, str, dict[str, str]]:
    fold = bundle.folds.loc[bundle.folds["fold_index"] == fold_index].copy()
    if len(fold) != len(PARTICIPANTS):
        raise DataContractError(f"Fold {fold_index} is not a complete participant-role assignment")
    first = fold.iloc[0]
    train = str(first["train_participants"]).split(";")
    validation = str(first["validation_participant"])
    test = str(first["test_participant"])
    roles = dict(zip(fold["participant"].astype(str), fold["role"].astype(str), strict=True))
    return train, validation, test, roles


def _anchor_lookup(bundle: ContractBundle, fold_index: int, target: str) -> dict[str, float]:
    frame = bundle.anchors.loc[(bundle.anchors["fold_index"] == fold_index) & (bundle.anchors["target"] == target)]
    return {str(row.condition): float(row.condition_anchor) for row in frame.itertuples(index=False)}


def train_temporal_group(
    bundle: ContractBundle,
    run_id: str,
    dynamic: pd.DataFrame,
    fold_index: int,
    seed: int,
    modality: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import torch.nn as nn

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame, feature_columns = _modality_feature_frame(bundle, dynamic, modality)
    train_participants, validation_participant, test_participant, roles = _fold_info(bundle, fold_index)
    means, scales = _fit_scaler(frame, feature_columns, train_participants)
    sequences, keys = _sequence_arrays(frame, feature_columns, means, scales)
    key_to_row = {key: index for index, key in enumerate(keys)}
    available_keys = set(keys)
    label_lookup = bundle.labels.set_index(["participant_id", "condition"])
    roles_by_key = {key: roles[key[0]] for key in keys}
    train_keys = [key for key in keys if roles_by_key[key] == "train"]
    validation_keys = [key for key in keys if roles_by_key[key] == "validation"]
    test_keys = [key for key in keys if roles_by_key[key] == "test"]
    if not train_keys or not validation_keys or not test_keys:
        raise DataContractError(f"Fold {fold_index} {modality} lacks train/validation/test condition keys")
    anchors = {target: _anchor_lookup(bundle, fold_index, target) for target in TARGETS}

    def target_matrix(selected: list[tuple[str, str]]) -> np.ndarray:
        return np.asarray([
            [float(label_lookup.loc[key, target]) - anchors[target][key[1]] for target in TARGETS]
            for key in selected
        ], dtype=np.float32)

    train_x = torch.from_numpy(np.stack([sequences[key_to_row[key]] for key in train_keys])).to(device)
    validation_x = torch.from_numpy(np.stack([sequences[key_to_row[key]] for key in validation_keys])).to(device)
    train_y = torch.from_numpy(target_matrix(train_keys)).to(device)
    validation_y = torch.from_numpy(target_matrix(validation_keys)).to(device)
    encoder, head = _temporal_model(len(feature_columns))
    encoder.to(device)
    head.to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=0.001,
        weight_decay=0.0001,
    )
    loss_function = nn.MSELoss()
    best_loss = math.inf
    best_epoch = 0
    best_encoder_state: dict[str, Any] | None = None
    best_head_state: dict[str, Any] | None = None
    patience = 8
    max_epochs = 40
    batch_size = 16
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        encoder.train()
        head.train()
        permutation = torch.randperm(len(train_x), device=device)
        for start in range(0, len(train_x), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            predictions = head(encoder(train_x[indices]))
            loss = loss_function(predictions, train_y[indices])
            loss.backward()
            optimizer.step()
        encoder.eval()
        head.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(head(encoder(validation_x)), validation_y).item())
        if np.isfinite(validation_loss) and validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch
            best_encoder_state = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
            best_head_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break
    if best_encoder_state is None or best_head_state is None:
        raise RuntimeError(f"Temporal encoder did not obtain a finite validation checkpoint for fold {fold_index}, seed {seed}, {modality}")
    encoder.load_state_dict(best_encoder_state)
    head.load_state_dict(best_head_state)
    encoder.eval()
    checkpoint_payload = {
        "encoder_state_dict": best_encoder_state,
        "fold_index": fold_index,
        "seed": seed,
        "modality": modality,
        "feature_columns": list(feature_columns),
        "embedding_dimension": int(encoder.embedding_dimension),
        "encoder_parameter_count": _parameter_count(encoder),
        "trainable_parameter_count_with_temporary_head": _parameter_count(encoder) + _parameter_count(head),
        "selected_epoch": best_epoch,
        "validation_mse": best_loss,
        "scaler_mean": means,
        "scaler_scale": scales,
        "architecture": {
            "conv_channels": [16, 32],
            "kernel_size": 3,
            "pool_size": 2,
            "dropout": 0.30,
            "sequence_length": SEQUENCE_LENGTH,
            "temporary_head": "Linear(64F,64)->ReLU->Dropout(0.30)->Linear(64,2)",
        },
    }
    checkpoint = bundle.results_dir / "checkpoints" / f"fold_{fold_index:02d}" / f"seed_{seed}" / f"{modality.lower()}.pt"
    if checkpoint.exists():
        try:
            existing_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
            same_state = set(existing_checkpoint.get("encoder_state_dict", {})) == set(checkpoint_payload["encoder_state_dict"])
            if same_state:
                same_state = all(
                    torch.equal(existing_checkpoint["encoder_state_dict"][name], checkpoint_payload["encoder_state_dict"][name])
                    for name in checkpoint_payload["encoder_state_dict"]
                )
            for field in ("fold_index", "seed", "modality", "feature_columns", "embedding_dimension", "selected_epoch"):
                same_state = same_state and existing_checkpoint.get(field) == checkpoint_payload.get(field)
            same_state = same_state and np.array_equal(existing_checkpoint.get("scaler_mean"), means, equal_nan=True) and np.array_equal(existing_checkpoint.get("scaler_scale"), scales, equal_nan=True)
            if not same_state:
                raise RuntimeError(f"Existing temporal checkpoint differs from the reproducible training result: {checkpoint}")
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(f"Could not verify existing temporal checkpoint {checkpoint}: {error}") from error
    else:
        buffer = io.BytesIO()
        torch.save(checkpoint_payload, buffer)
        _write_immutable_bytes(checkpoint, buffer.getvalue())
    checkpoint_hash = file_sha256(checkpoint)

    all_x = torch.from_numpy(sequences).to(device)
    with torch.no_grad():
        embeddings = encoder(all_x).detach().cpu().numpy().astype(np.float32)
    export_keys = [key for key in keys if key in available_keys]
    artifact = bundle.results_dir / "representations" / "temporal_1dcnn" / f"fold_{fold_index:02d}" / f"seed_{seed}" / f"{modality.lower()}.npz"
    _write_immutable_npz(
        artifact,
        participants=np.asarray([key[0] for key in export_keys], dtype="U4"),
        conditions=np.asarray([key[1] for key in export_keys], dtype="U2"),
        vectors=embeddings,
        feature_names=np.asarray([f"embedding_{index:04d}" for index in range(embeddings.shape[1])], dtype="U32"),
        representation_family=np.asarray(["temporal_1dcnn"]),
        modality=np.asarray([modality]),
        fold=np.asarray([fold_index]),
        seed=np.asarray([seed]),
        encoder_version=np.asarray(["temporal_1dcnn_v1"], dtype="U32"),
    )
    artifact_hash = file_sha256(artifact)
    label_lookup_manifest = bundle.condition_manifest.set_index(["participant", "condition"])
    window_counts = bundle.window_manifest.groupby(["participant", "condition"], sort=False).size().to_dict()
    rows: list[dict[str, Any]] = []
    key_to_embedding_row = {key: index for index, key in enumerate(export_keys)}
    for participant in PARTICIPANTS:
        for condition in CONDITIONS:
            label = label_lookup_manifest.loc[(participant, condition)]
            key = (participant, condition)
            available = key in key_to_embedding_row
            rows.append(_index_row(
                contract_hash=bundle.contract_hash,
                run_id=run_id,
                family="temporal_1dcnn",
                modality=modality,
                fold=f"{fold_index:02d}",
                seed=str(seed),
                split_role=roles[participant],
                train_participants=";".join(train_participants),
                validation_participant=validation_participant,
                test_participant=test_participant,
                participant=participant,
                condition=condition,
                intensity=float(label["intensity"]),
                frequency=float(label["frequency"]),
                artifact_path=_relative_shared(artifact, bundle.shared_root),
                artifact_row=key_to_embedding_row.get(key),
                dimension=int(embeddings.shape[1]),
                available=available,
                valid_window_count=int(window_counts.get(key, 0)),
                fallback_used=not available,
                checkpoint_id=f"fold_{fold_index:02d}_seed_{seed}_{modality}",
                version="temporal_1dcnn_v1",
                artifact_hash=artifact_hash,
                unavailable_reason="all_missing_common_windows" if not available else "",
            ))
    runtime = time.perf_counter() - started
    run_record = {
        "run_id": run_id,
        "representation_family": "temporal_1dcnn",
        "modality": modality,
        "fold": f"{fold_index:02d}",
        "seed": seed,
        "contract_hash": bundle.contract_hash,
        "code_version": _code_version(bundle.repo_root),
        "data_hash": _sha256_text("\n".join(f"{key}:{value}" for key, value in sorted(bundle.source_hashes.items()))),
        "model_configuration": "grouped Conv1D 16/32, pool 2/2, dropout .30, temporary dual residual head",
        "selected_epoch": best_epoch,
        "validation_mse": best_loss,
        "device": str(device),
        "runtime_seconds": round(runtime, 3),
        "status": "complete",
        "warnings": "P004/C6 unavailable; test participant excluded from scaler, training, and epoch selection",
        "failures": "",
        "train_participants": ";".join(train_participants),
        "validation_participant": validation_participant,
        "test_participant": test_participant,
        "scaler_fit_participants": ";".join(train_participants),
        "test_used_for_training": False,
        "encoder_checkpoint_id": f"fold_{fold_index:02d}_seed_{seed}_{modality}",
        "encoder_checkpoint_sha256": checkpoint_hash,
        "representation_artifact_sha256": artifact_hash,
        "representation_dimension": int(embeddings.shape[1]),
        "trainable_parameter_count": int(checkpoint_payload["trainable_parameter_count_with_temporary_head"]),
        "encoder_parameter_count": int(checkpoint_payload["encoder_parameter_count"]),
    }
    return rows, run_record


def export_temporal(
    bundle: ContractBundle,
    run_id: str,
    dynamic: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    total = len(PARTICIPANTS) * len(SEEDS) * len(MODALITIES)
    number = 0
    for fold_index in range(1, len(PARTICIPANTS) + 1):
        for seed in SEEDS:
            for modality in MODALITIES:
                number += 1
                print(f"temporal_1dcnn {number}/{total}: fold={fold_index:02d} seed={seed} modality={modality}", flush=True)
                group_rows, run_record = train_temporal_group(bundle, run_id, dynamic, fold_index, seed, modality)
                rows.extend(group_rows)
                runs.append(run_record)
    return rows, runs


def _load_npz_artifact(path: Path) -> tuple[np.ndarray, list[tuple[str, str]], list[str]]:
    with np.load(path, allow_pickle=False) as data:
        participants = [str(value) for value in data["participants"].tolist()]
        conditions = [str(value) for value in data["conditions"].tolist()]
        vectors = np.asarray(data["vectors"])
        feature_names = [str(value) for value in data["feature_names"].tolist()]
    keys = list(zip(participants, conditions, strict=True))
    return vectors, keys, feature_names


def validate_representation_index(
    bundle: ContractBundle,
    index: pd.DataFrame,
    run_records: list[dict[str, Any]],
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    audit_rows: list[dict[str, Any]] = []
    if tuple(index.columns) != INDEX_COLUMNS:
        failures.append(f"Representation index columns do not match the locked order: {list(index.columns)}")
    if index["contract_hash"].nunique() != 1 or str(index["contract_hash"].iloc[0]) != bundle.contract_hash:
        failures.append("Representation index contains inconsistent contract_hash values")
    if index["environment"].ne("windows").any():
        failures.append("Representation index contains a non-Windows environment")
    if index["representation_key"].duplicated().any():
        failures.append("Representation index contains duplicate representation keys")
    expected_conditions = {(participant, condition) for participant in PARTICIPANTS for condition in CONDITIONS}
    expected_available = expected_conditions - {("P004", "C6")}
    for (family, modality, fold, seed), group in index.groupby(["representation_family", "modality", "fold", "seed"], sort=True):
        keys = set(zip(group["participant"], group["condition"], strict=False))
        if keys != expected_conditions or len(group) != 81:
            failures.append(f"{family}/{modality}/{fold}/{seed} does not contain exactly 81 participant-condition index rows")
        path_values = group["artifact_path"].dropna().astype(str).unique()
        if len(path_values) != 1:
            failures.append(f"{family}/{modality}/{fold}/{seed} does not have one artifact path")
            continue
        artifact = bundle.shared_root / Path(path_values[0])
        if not artifact.exists():
            failures.append(f"Missing representation artifact: {path_values[0]}")
            continue
        actual_hash = file_sha256(artifact)
        if set(group["artifact_sha256"].astype(str)) != {actual_hash}:
            failures.append(f"Artifact hash mismatch in index for {path_values[0]}")
        try:
            vectors, artifact_keys, feature_names = _load_npz_artifact(artifact)
        except Exception as error:  # pragma: no cover - diagnostic path
            failures.append(f"Could not load representation artifact {path_values[0]}: {error}")
            continue
        available_mask = _normalise_bool_series(group["representation_available"])
        available_keys = set(zip(
            group.loc[available_mask, "participant"],
            group.loc[available_mask, "condition"],
            strict=False,
        ))
        if available_keys != set(artifact_keys) or set(artifact_keys) != expected_available:
            failures.append(f"Artifact/index key mismatch in {path_values[0]}")
        if len(set(artifact_keys)) != len(artifact_keys):
            failures.append(f"Duplicate vector keys in {path_values[0]}")
        if vectors.ndim != 2 or vectors.shape[0] != len(artifact_keys) or vectors.shape[1] != len(feature_names):
            failures.append(f"Invalid vector shape or feature-name mapping in {path_values[0]}")
        dimensions = set(pd.to_numeric(group["representation_dimension"], errors="coerce").dropna().astype(int))
        if dimensions != {int(vectors.shape[1])}:
            failures.append(f"Dimension mismatch in index/artifact for {path_values[0]}")
        if family == "handcrafted":
            if fold != "global" or seed != "deterministic" or set(group["split_role"]) != {"global"}:
                failures.append(f"Handcrafted group {modality} is not marked fold-independent")
        elif family == "temporal_1dcnn":
            expected_fold = int(fold)
            _, validation, test, roles = _fold_info(bundle, expected_fold)
            for row in group.itertuples(index=False):
                if row.split_role != roles[str(row.participant)] or row.validation_participant != validation or row.test_participant != test:
                    failures.append(f"Temporal split-role mismatch for {row.representation_key}")
                    break
        audit_rows.append({
            "representation_family": family,
            "modality": modality,
            "fold": fold,
            "seed": seed,
            "feature_or_encoder_version": str(group["feature_or_encoder_version"].iloc[0]),
            "representation_dimension": int(vectors.shape[1]),
            "index_rows": len(group),
            "artifact_rows": len(artifact_keys),
            "missing_index_rows": int((~available_mask).sum()),
            "missing_keys": ";".join(sorted(
                f"{participant}+{condition}" for participant, condition in expected_conditions - available_keys
            )),
            "source_hashes": json.dumps(bundle.source_hashes, sort_keys=True),
            "artifact_sha256": actual_hash,
            "status": "complete" if not failures else "review",
            "warnings": "P004/C6 unavailable; artifact rows verified against index",
        })

    temporal_records = [record for record in run_records if record.get("representation_family") == "temporal_1dcnn"]
    for record in temporal_records:
        if record.get("test_used_for_training") is not False:
            failures.append(f"Temporal record violates test exclusion: {record.get('encoder_checkpoint_id')}")
        if record.get("scaler_fit_participants") != record.get("train_participants"):
            failures.append(f"Temporal scaler participants differ from training participants: {record.get('encoder_checkpoint_id')}")
        checkpoint_id = record.get("encoder_checkpoint_id", "")
        if not checkpoint_id or not record.get("encoder_checkpoint_sha256"):
            failures.append(f"Temporal record lacks checkpoint identity: {checkpoint_id}")
    expected_temporal_records = len(PARTICIPANTS) * len(SEEDS) * len(MODALITIES)
    if len(temporal_records) != expected_temporal_records:
        failures.append(f"Expected {expected_temporal_records} temporal run records, found {len(temporal_records)}")
    return not failures, audit_rows, failures


def write_windows_outputs(
    bundle: ContractBundle,
    run_id: str,
    index_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    index = pd.DataFrame(index_rows, columns=INDEX_COLUMNS)
    index = _ordered_frame(index, index.columns)
    index_path = bundle.results_dir / "windows_representation_index.csv"
    _write_immutable_csv(index_path, index, INDEX_COLUMNS)
    audit = pd.DataFrame(audit_rows)
    audit_columns = (
        "representation_family", "modality", "fold", "seed", "feature_or_encoder_version",
        "representation_dimension", "index_rows", "artifact_rows", "missing_index_rows",
        "missing_keys", "source_hashes", "artifact_sha256", "status", "warnings",
    )
    audit = audit.loc[:, audit_columns]
    _write_immutable_csv(bundle.results_dir / "windows_representation_audit.csv", audit, audit_columns)
    run_columns = (
        "run_id", "representation_family", "modality", "fold", "seed", "contract_hash", "code_version",
        "data_hash", "model_configuration", "selected_epoch", "validation_mse", "device", "runtime_seconds",
        "status", "warnings", "failures", "train_participants", "validation_participant", "test_participant",
        "scaler_fit_participants", "test_used_for_training", "encoder_checkpoint_id", "encoder_checkpoint_sha256",
        "representation_artifact_sha256", "representation_dimension", "trainable_parameter_count",
        "encoder_parameter_count",
    )
    run_manifest = pd.DataFrame(run_records)
    for column in run_columns:
        if column not in run_manifest:
            run_manifest[column] = ""
    _write_immutable_csv(bundle.results_dir / "windows_run_manifest.csv", run_manifest.loc[:, run_columns], run_columns)
    return {"index": index, "audit": audit, "run_manifest": run_manifest.loc[:, run_columns]}


def write_windows_report(
    bundle: ContractBundle,
    run_id: str,
    index: pd.DataFrame,
    audit: pd.DataFrame,
    run_manifest: pd.DataFrame,
    checks: list[str],
    failures: list[str],
) -> Path:
    status = "PASS" if not failures else "FAIL"
    family_counts = index.groupby(["representation_family", "modality"], sort=True).size().to_dict() if not index.empty else {}
    handcrafted_group_count = sum(1 for family, _ in family_counts if family == "handcrafted")
    temporal_group_count = int(index.loc[index["representation_family"] == "temporal_1dcnn"].groupby(["modality", "fold", "seed"]).ngroups)
    lines = [
        "# Windows RQ2 hand-off report",
        "",
        f"Status: **{status}**",
        "",
        "This report covers only the Windows representation stage. It does not rank representation families or compare fusion models; those comparisons require the shared WSL Ridge-expert and Simplex pipeline.",
        "",
        f"- Project root: `{bundle.repo_root}`",
        f"- Shared root: `{bundle.shared_root}`",
        f"- Run ID: `{run_id}`",
        f"- Contract hash: `{bundle.contract_hash}`",
        f"- Cohort: FMQ-9 ({len(PARTICIPANTS)} participants, 81 condition ratings)",
        f"- Common-valid windows: {len(bundle.window_manifest)}",
        "- WSL pretrained-encoder project: not accessed",
        "",
        "## Data audit",
        "",
        "The parsed workbook and existing FMQ-9 alignment artifacts agree on the 81 participant-condition labels. The source inventory contains seven ordered ten-second windows per condition, with 545 rows in the locked five-modality common-valid mask. P004/C6 remains in the condition manifest and has no common-valid window; it is marked condition-only and has no representation vector.",
        "",
        f"The existing source feature dimensions are EEG={len(_source_feature_columns(bundle.base_features)['EEG'])}, ECG={len(_source_feature_columns(bundle.base_features)['ECG'])}, Eye={len(_source_feature_columns(bundle.base_features)['Eye'])}, Head={len(_source_feature_columns(bundle.base_features)['Head'])}; Video dynamic_texture_v1 contributes {len(DYNAMIC_TEXTURE_COLUMNS)} descriptors.",
        "",
        "## Completed Windows representations",
        "",
        f"- Handcrafted groups: {handcrafted_group_count} modality groups; each index group has 81 keys and 80 vector rows.",
        f"- Temporal 1D-CNN groups: {temporal_group_count} fold/seed/modality groups; each index group has 81 keys and 80 vector rows.",
        f"- Representation index rows: {len(index)}",
        f"- Temporal training records: {len(run_manifest.loc[run_manifest['representation_family'] == 'temporal_1dcnn'])}",
        "",
        "## Output checks",
        "",
    ]
    lines.extend(f"- {check}" for check in checks)
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.extend([
        "",
        "## Shared downstream requirements",
        "",
        "The contract locks the three representation families, frozen model IDs, anchor reuse, fold-local residual rules, PCA allocation, Ridge grid, target-specific Simplex rules, full-five/no-video comparison, VideoMAE V2 probes, participant-macro primary MAE, secondary metrics, two separate four-test Holm families, 10,000 participant bootstraps, and the exact two-sided participant sign-flip test. The shared frozen_simplex_full5 run must have one run ID and one set of OOF predictions in both comparison families.",
        "",
    ])
    path = bundle.results_dir / "WINDOWS_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def write_done_marker(
    bundle: ContractBundle,
    run_id: str,
    outputs: dict[str, Any],
    checks: list[str],
) -> Path:
    payload = {
        "status": "complete",
        "contract_hash": bundle.contract_hash,
        "code_version": _code_version(bundle.repo_root),
        "completed_run_ids": [run_id],
        "row_counts": {
            "condition_manifest": len(bundle.condition_manifest),
            "window_manifest": len(bundle.window_manifest),
            "representation_index": len(outputs["index"]),
            "representation_audit": len(outputs["audit"]),
            "run_manifest": len(outputs["run_manifest"]),
        },
        "files": [
            "contract/condition_manifest.csv",
            "contract/window_manifest.csv",
            "contract/folds.csv",
            "contract/condition_anchors.csv",
            "contract/rq2_contract.json",
            "contract/rq2_contract.sha256",
            "windows_results/windows_representation_index.csv",
            "windows_results/windows_representation_audit.csv",
            "windows_results/windows_run_manifest.csv",
            "windows_results/WINDOWS_REPORT.md",
            "windows_results/WINDOWS_DONE.json",
        ],
        "checks": checks,
    }
    path = bundle.results_dir / "WINDOWS_DONE.json"
    _write_immutable_json(path, payload)
    return path


def run_windows_rq2(config: ProjectConfig, shared_root: str | Path, run_id: str) -> dict[str, Any]:
    try:
        bundle = prepare_contract(config, shared_root)
    except DataContractError as error:
        _write_data_error(Path(shared_root).resolve(), str(error))
        raise
    dynamic = build_dynamic_texture_table(bundle)
    handcrafted_rows, handcrafted_audit = export_handcrafted(bundle, run_id, dynamic)
    temporal_rows, temporal_runs = export_temporal(bundle, run_id, dynamic)
    index_rows = handcrafted_rows + temporal_rows
    run_records = [{
        "run_id": run_id,
        "representation_family": "handcrafted",
        "modality": "all",
        "fold": "global",
        "seed": "deterministic",
        "contract_hash": bundle.contract_hash,
        "code_version": _code_version(bundle.repo_root),
        "data_hash": _sha256_text("\n".join(f"{key}:{value}" for key, value in sorted(bundle.source_hashes.items()))),
        "model_configuration": "nine-statistic condition aggregation from manifest-listed windows",
        "selected_epoch": "",
        "validation_mse": "",
        "device": "n/a",
        "runtime_seconds": "",
        "status": "complete",
        "warnings": "P004/C6 unavailable; no final fusion fitted in Windows",
        "failures": "",
        "train_participants": "",
        "validation_participant": "",
        "test_participant": "",
        "scaler_fit_participants": "",
        "test_used_for_training": False,
        "encoder_checkpoint_id": "",
        "encoder_checkpoint_sha256": "",
        "representation_artifact_sha256": "",
        "representation_dimension": "",
        "trainable_parameter_count": "",
        "encoder_parameter_count": "",
    }, *temporal_runs]
    outputs = write_windows_outputs(bundle, run_id, index_rows, handcrafted_audit + [
        {
            "representation_family": "temporal_1dcnn",
            "modality": modality,
            "fold": f"{fold_index:02d}",
            "seed": str(seed),
            "feature_or_encoder_version": "temporal_1dcnn_v1",
            "representation_dimension": int(group[0]["representation_dimension"]),
            "index_rows": 81,
            "artifact_rows": 80,
            "missing_index_rows": 1,
            "missing_keys": "P004+C6",
            "source_hashes": json.dumps(bundle.source_hashes, sort_keys=True),
            "artifact_sha256": str(group[0]["artifact_sha256"]),
            "status": "complete",
            "warnings": "P004/C6 retained as unavailable condition-only fallback",
        }
        for (fold_index, seed, modality), group in _group_temporal_index_rows(temporal_rows)
    ], run_records)
    checks = [
        "contract manifests contain exactly 81 condition rows and 545 common-valid window rows",
        "handcrafted and temporal_1dcnn indexes contain all five modalities and 81 keys per group",
        "temporal split roles match folds.csv and handcrafted rows are fold-independent",
        "all vector dimensions, artifact row mappings, and artifact SHA-256 values reproduce from the index",
        "P004/C6 is present in every index group as unavailable with condition-only fallback",
        "all temporal scalers use training participants only; test_used_for_training is false for every temporal job",
        "all representation keys are unique and carry the same contract hash",
    ]
    valid, _, failures = validate_representation_index(bundle, outputs["index"], run_records)
    if not valid:
        write_windows_report(bundle, run_id, outputs["index"], outputs["audit"], outputs["run_manifest"], checks, failures)
        raise RuntimeError("Windows RQ2 output validation failed: " + "; ".join(failures))
    write_windows_report(bundle, run_id, outputs["index"], outputs["audit"], outputs["run_manifest"], checks, [])
    write_done_marker(bundle, run_id, outputs, checks)
    return {"bundle": bundle, "outputs": outputs, "checks": checks}


def _group_temporal_index_rows(rows: list[dict[str, Any]]):
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["fold"]), int(row["seed"]), str(row["modality"]))
        groups.setdefault(key, []).append(row)
    return sorted(groups.items())


__all__ = [
    "CONDITIONS",
    "ContractBundle",
    "DataContractError",
    "MODALITIES",
    "PARTICIPANTS",
    "SEEDS",
    "TARGETS",
    "build_dynamic_texture_table",
    "condition_key",
    "final_oof_key",
    "load_contract_hash",
    "prepare_contract",
    "representation_key",
    "run_windows_rq2",
    "validate_representation_index",
    "window_key",
]
