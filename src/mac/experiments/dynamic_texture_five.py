"""Research-only paired five-modality Project A experiments.

This module has no runtime imports or checkpoint names in common with the
deployed four-modality path.  It reuses Project A's existing classical
residual-ensemble selection helpers, while the deep branch implements an
explicit post-encoder video gate shared by ``full_five`` and ``no_video``.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr

from real_time_ml.config import ProjectConfig
from real_time_ml.evaluation.alignment import (
    AlignedFold,
    file_sha256,
    indexes_for_fold,
    load_split_manifest,
    validate_alignment_contract,
)
from real_time_ml.features.dynamic_texture import DYNAMIC_TEXTURE_COLUMNS
from real_time_ml.modeling.condition_data import build_condition_dataset
from real_time_ml.modeling.condition_models import (
    ModelSpec,
    apply_condition_baseline,
    condition_baseline,
)
from real_time_ml.modeling.condition_train import (
    _feature_columns as classical_feature_columns,
    _fit_target_models,
    _predict_target_models,
    _rank_regression_on_validation,
)
from real_time_ml.modeling.dcnn import _variant_columns as native_dcnn_columns
from real_time_ml.utils import write_json


TARGETS = ("relaxation", "discomfort")
MODALITIES = ("eeg", "ecg", "eye", "head", "video")
VARIANTS = ("full_five", "no_video")
SEQUENCE_LENGTH = 8
CONTEXT_COLUMNS = ("intensity", "frequency")
ZERO_WINDOW_KEY = ("P004", "C6")


@dataclass(frozen=True)
class FiveModalitySequences:
    values: dict[str, np.ndarray]
    context: np.ndarray
    targets: np.ndarray
    participant_ids: np.ndarray
    conditions: np.ndarray
    presentation_positions: np.ndarray
    lengths: np.ndarray
    video_available: np.ndarray
    feature_columns: dict[str, tuple[str, ...]]


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("The paired five-branch experiment requires PyTorch") from error
    return torch


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, Any]) -> str:
    """Stable hash over tensor names, shapes, dtypes, and bytes."""
    digest = sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_signature(model: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "parameter_count": int(parameter.numel()),
        }
        for name, parameter in model.named_parameters()
    ]


def concordance_correlation_coefficient(truth: np.ndarray, prediction: np.ndarray) -> float:
    y_true = np.asarray(truth, dtype=float)
    y_pred = np.asarray(prediction, dtype=float)
    covariance = float(np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))))
    denominator = (
        float(np.var(y_true))
        + float(np.var(y_pred))
        + float((np.mean(y_true) - np.mean(y_pred)) ** 2)
    )
    return float(2.0 * covariance / denominator) if denominator > 0 else float("nan")


def project_b_style_metrics(frame: Any) -> dict[str, Any]:
    """Recompute the fixed Project B metric contract from a normalized OOF frame."""
    target_metrics: dict[str, Any] = {}
    for target in TARGETS:
        truth = frame[f"true_{target}"].to_numpy(dtype=float)
        prediction = frame[f"pred_{target}"].to_numpy(dtype=float)
        participant_mae = (
            frame.assign(_absolute_error=np.abs(truth - prediction))
            .groupby("participant_id", sort=True)["_absolute_error"]
            .mean()
        )
        target_metrics[target] = {
            "participant_macro_mae": float(participant_mae.mean()),
            "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
            "spearman": float(spearmanr(truth, prediction).statistic),
            "pearson": float(pearsonr(truth, prediction).statistic),
            "ccc": concordance_correlation_coefficient(truth, prediction),
        }
    return {
        "n_predictions": int(len(frame)),
        "targets": target_metrics,
        "macro_mae": float(
            np.mean([target_metrics[target]["participant_macro_mae"] for target in TARGETS])
        ),
    }


def _contract_paths(contract_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = validate_alignment_contract(contract_dir)
    files = contract["files"]
    paths = {
        "contract": contract_dir / "contract.json",
        "split_manifest": contract_dir / files["split_manifest"]["path"],
        "masks": contract_dir / files["common_masks"]["path"],
        "labels": contract_dir / files["labels"]["path"],
        "windows": contract_dir / files["windows"]["path"],
    }
    if int(contract.get("observation_count", 0)) != 81:
        raise ValueError("Five-modality experiments require exactly 81 labels")
    if int(contract.get("window_count", 0)) != 567:
        raise ValueError("Five-modality experiments require exactly 567 source windows")
    if int(contract.get("common_valid_window_count", 0)) != 545:
        raise ValueError("Five-modality experiments require exactly 545 common-valid windows")
    if contract.get("zero_common_window_observations") != [
        {"participant_id": "P004", "condition": "C6"}
    ]:
        raise ValueError("P004/C6 must be the unique zero-common-window observation")
    return contract, paths


def _folds(
    contract: dict[str, Any], paths: dict[str, Path], participants: Iterable[str]
) -> list[AlignedFold]:
    return load_split_manifest(
        paths["split_manifest"],
        expected_participants=sorted(set(str(value) for value in participants)),
        expected_train_count=int(contract["fold_contract"]["train_participants"]),
    )


def _classical_columns(condition_frame: Any, variant: str) -> tuple[list[str], list[str]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown dynamic-texture variant: {variant!r}")
    raw = classical_feature_columns(condition_frame)
    descriptor_prefixes = tuple(f"{name}__" for name in DYNAMIC_TEXTURE_COLUMNS)
    all_columns = [
        name
        for name in raw
        if (
            not name.startswith(("video_", "qc_video_", "mask_video_"))
            or name.startswith(descriptor_prefixes)
        )
        and not name.startswith("video_dynamic_validity__")
    ]
    video_columns = [name for name in all_columns if name.startswith(descriptor_prefixes)]
    if len(video_columns) != 108:
        raise ValueError(
            f"Classical full-five must expose exactly 108 video candidates; found {len(video_columns)}"
        )
    selected = (
        all_columns
        if variant == "full_five"
        else [name for name in all_columns if name not in set(video_columns)]
    )
    return sorted(selected), sorted(video_columns)


def _condition_baseline_for_rows(
    train: Any,
    test: Any,
    target: str,
) -> tuple[np.ndarray, dict[str, float], float]:
    mapping, fallback = condition_baseline(train, target)
    return apply_condition_baseline(test["condition"], mapping, fallback), mapping, fallback


def _is_zero_window_rows(frame: Any) -> np.ndarray:
    return (
        frame["participant_id"].astype(str).eq(ZERO_WINDOW_KEY[0])
        & frame["condition"].astype(str).eq(ZERO_WINDOW_KEY[1])
    ).to_numpy(dtype=bool)


def run_classical_dynamic_texture(
    config: ProjectConfig,
    *,
    window_features: str | Path,
    contract_dir: str | Path,
    run_root: str | Path,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    """Run one formal classical residual-ensemble OOF experiment."""
    import joblib
    import pandas as pd

    if int(config.get("modeling.random_seed")) != int(seed):
        raise ValueError("Resolved config seed differs from the requested classical run")
    root = Path(run_root).resolve()
    for child in ("features", "models", "metrics", "predictions", "manifests", "logs"):
        (root / child).mkdir(parents=True, exist_ok=True)
    source = Path(window_features).resolve()
    contract_root = Path(contract_dir).resolve()
    contract, paths = _contract_paths(contract_root)
    condition_path = root / "features" / "condition_features.csv"
    frame = build_condition_dataset(source, condition_path)
    frame = frame.dropna(subset=["participant_id", "condition", *TARGETS]).reset_index(drop=True)
    if len(frame) != 81 or frame.duplicated(["participant_id", "condition"]).any():
        raise ValueError("Classical aggregation did not preserve 81 unique labels")
    columns, video_columns = _classical_columns(frame, variant)
    groups = frame["participant_id"].astype(str).to_numpy()
    folds = _folds(contract, paths, groups)
    model_names = list(config.get("modeling.candidates"))
    predictions = {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
    baselines = {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
    row_fold = np.full(len(frame), -1, dtype=int)
    row_validation = np.full(len(frame), "", dtype=object)
    fold_records: list[dict[str, Any]] = []
    selected_counts = {target: Counter() for target in TARGETS}
    for fold in folds:
        train_indexes, validation_indexes, test_indexes = indexes_for_fold(groups, fold)
        train = frame.iloc[train_indexes]
        validation = frame.iloc[validation_indexes]
        test = frame.iloc[test_indexes]
        fold_bundle: dict[str, Any] = {
            "schema_version": "dynamic_texture_classical_fold_v1",
            "fold_index": int(fold.fold_index),
            "variant": variant,
            "seed": int(seed),
            "feature_columns": columns,
            "targets": {},
        }
        fold_record: dict[str, Any] = {
            "fold_index": int(fold.fold_index),
            "train_participants": list(fold.train_participants),
            "validation_participant": fold.validation_participant,
            "test_participant": fold.test_participant,
            "n_train_participants": len(fold.train_participants),
            "n_validation_participants": 1,
            "n_test_participants": 1,
            "feature_count": len(columns),
            "video_candidate_count": len(video_columns) if variant == "full_five" else 0,
            "targets": {},
        }
        for target in TARGETS:
            feature_count_trials = _rank_regression_on_validation(
                train,
                validation,
                columns,
                target,
                [
                    ModelSpec("ridge", int(count))
                    for count in config.get("modeling.condition_level.feature_counts")
                ],
                config,
                pd,
            )
            selected_count = int(feature_count_trials[0]["spec"].feature_count)
            ranked = _rank_regression_on_validation(
                train,
                validation,
                columns,
                target,
                [ModelSpec(name, selected_count) for name in model_names],
                config,
                pd,
            )
            selected = [
                item["spec"]
                for item in ranked[: int(config.get("modeling.condition_level.ensemble_size"))]
            ]
            mapping, fallback, models = _fit_target_models(
                train, columns, target, selected, config, pd
            )
            prediction = _predict_target_models(test, columns, mapping, fallback, models, pd)
            baseline = apply_condition_baseline(test["condition"], mapping, fallback)
            zero = _is_zero_window_rows(test)
            prediction[zero] = baseline[zero]
            predictions[target][test_indexes] = prediction
            baselines[target][test_indexes] = baseline
            selected_counts[target].update(spec.label() for spec in selected)
            fold_record["targets"][target] = {
                "selected": [asdict(spec) for spec in selected],
                "selected_feature_count": selected_count,
                "validation_ranking": [
                    {
                        **{key: value for key, value in item.items() if key != "spec"},
                        "spec": asdict(item["spec"]),
                    }
                    for item in ranked
                ],
                "feature_count_trials": [
                    {
                        **{key: value for key, value in item.items() if key != "spec"},
                        "spec": asdict(item["spec"]),
                    }
                    for item in feature_count_trials
                ],
            }
            fold_bundle["targets"][target] = {
                "baseline_by_condition": mapping,
                "baseline_fallback": fallback,
                "models": models,
                "selected": [asdict(spec) for spec in selected],
            }
        row_fold[test_indexes] = int(fold.fold_index)
        row_validation[test_indexes] = fold.validation_participant
        model_path = root / "models" / f"fold_{fold.fold_index:02d}.joblib"
        joblib.dump(fold_bundle, model_path)
        fold_record["model_path"] = str(model_path)
        fold_record["model_sha256"] = file_sha256(model_path)
        fold_records.append(fold_record)

    prediction_frame = frame[
        ["participant_id", "condition", "presentation_position", *TARGETS]
    ].copy()
    prediction_frame = prediction_frame.rename(
        columns={target: f"true_{target}" for target in TARGETS}
    )
    for target in TARGETS:
        prediction_frame[f"pred_{target}"] = predictions[target]
        prediction_frame[f"condition_only_{target}"] = baselines[target]
    prediction_frame["project"] = "A"
    prediction_frame["model_family"] = "classical"
    prediction_frame["variant"] = variant
    prediction_frame["modalities"] = (
        "eeg+ecg+eye+head+video" if variant == "full_five" else "eeg+ecg+eye+head"
    )
    prediction_frame["seed"] = int(seed)
    prediction_frame["fold_index"] = row_fold
    prediction_frame["test_participant"] = prediction_frame["participant_id"].astype(str)
    prediction_frame["validation_participant"] = row_validation
    prediction_frame["split_protocol"] = "shared_7_train_1_validation_1_test"
    prediction_frame["device"] = "cpu"
    prediction_frame["cuda_used"] = False
    prediction_frame["video_representation"] = (
        "dynamic_texture_v1" if variant == "full_five" else "gated_absent"
    )
    prediction_frame["p004_c6_condition_only_fallback"] = _is_zero_window_rows(prediction_frame)
    if prediction_frame[[f"pred_{target}" for target in TARGETS]].isna().any().any():
        raise ValueError("Classical OOF predictions are incomplete")
    zero = _is_zero_window_rows(prediction_frame)
    for target in TARGETS:
        if not np.array_equal(
            prediction_frame.loc[zero, f"pred_{target}"].to_numpy(),
            prediction_frame.loc[zero, f"condition_only_{target}"].to_numpy(),
        ):
            raise AssertionError("P004/C6 did not use the exact Condition-only fallback")
    metrics = project_b_style_metrics(prediction_frame)
    prediction_path = root / "predictions" / "oof_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    metrics_payload = {
        "schema_version": "project_a_dynamic_texture_classical_run_v1",
        "research_only": True,
        "project": "A",
        "model_family": "classical",
        "variant": variant,
        "modalities": list(MODALITIES if variant == "full_five" else MODALITIES[:-1]),
        "video_representation": (
            "dynamic_texture_v1" if variant == "full_five" else "gated_absent"
        ),
        "seed": int(seed),
        "split_protocol": "shared_7_train_1_validation_1_test",
        "candidate_models": model_names,
        "feature_counts": list(config.get("modeling.condition_level.feature_counts")),
        "ensemble_size": int(config.get("modeling.condition_level.ensemble_size")),
        "feature_count": len(columns),
        "video_candidate_count": len(video_columns) if variant == "full_five" else 0,
        "selected_spec_frequency": {
            target: dict(counter) for target, counter in selected_counts.items()
        },
        "metrics": metrics,
        "folds": fold_records,
        "fallback": {
            "participant_id": ZERO_WINDOW_KEY[0],
            "condition": ZERO_WINDOW_KEY[1],
            "exact_condition_only": True,
        },
        "runtime": {"device": "cpu", "cuda_used": False},
    }
    metrics_path = root / "metrics" / "metrics.json"
    write_json(metrics_path, metrics_payload)
    manifest = {
        "schema_version": "project_a_dynamic_texture_run_manifest_v1",
        "run_id": root.name,
        "seed": int(seed),
        "model_family": "classical",
        "variant": variant,
        "sources": {
            "window_features": {"path": str(source), "sha256": file_sha256(source)},
            **{
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in paths.items()
            },
        },
        "outputs": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": file_sha256(prediction_path),
                "rows": len(prediction_frame),
            },
            "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
        },
        "folds": [asdict(fold) for fold in folds],
        "runtime": {"device": "cpu", "cuda_used": False},
    }
    write_json(root / "manifests" / "run_manifest.json", manifest)
    return metrics_payload


def build_five_modality_sequences(
    path: str | Path,
    *,
    sequence_length: int = SEQUENCE_LENGTH,
) -> FiveModalitySequences:
    """Build one label row with five temporal branches per participant-condition."""
    import pandas as pd

    frame = pd.read_csv(path)
    required = {
        "participant_id",
        "condition",
        "condition_window_index",
        "presentation_position",
        *TARGETS,
        *CONTEXT_COLUMNS,
        "video_dynamic_validity",
        *DYNAMIC_TEXTURE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Five-modality window table lacks fields: {sorted(missing)}")
    if any("rgb" in name.lower() and name.startswith("video_") for name in frame.columns):
        raise ValueError("RGB tensors or RGB feature arrays are forbidden at the training boundary")
    native = native_dcnn_columns(frame.columns, "full")
    feature_columns = {
        modality: tuple(name for name in native if name.startswith(f"{modality}_"))
        for modality in MODALITIES[:-1]
    }
    feature_columns["video"] = (*DYNAMIC_TEXTURE_COLUMNS, "video_dynamic_validity")
    if any(not columns for columns in feature_columns.values()):
        empty = [name for name, columns in feature_columns.items() if not columns]
        raise ValueError(f"Five-modality sequence has empty branches: {empty}")
    values: dict[str, list[np.ndarray]] = {modality: [] for modality in MODALITIES}
    context: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    participants: list[str] = []
    conditions: list[str] = []
    positions: list[float] = []
    lengths: list[int] = []
    video_available: list[float] = []
    for (participant, condition), group in frame.groupby(
        ["participant_id", "condition"], sort=True
    ):
        ordered = group.copy()
        ordered["condition_window_index"] = pd.to_numeric(
            ordered["condition_window_index"], errors="coerce"
        )
        ordered = ordered.dropna(subset=["condition_window_index"]).sort_values(
            "condition_window_index"
        )
        indexes = ordered["condition_window_index"].astype(int).to_numpy()
        if (
            not len(indexes)
            or indexes.min() < 0
            or indexes.max() >= sequence_length
            or len(np.unique(indexes)) != len(indexes)
        ):
            raise ValueError(f"Invalid temporal indexes for {participant}/{condition}")
        if ordered[list(TARGETS)].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent labels for {participant}/{condition}")
        for modality in MODALITIES:
            columns = list(feature_columns[modality])
            matrix = np.full((len(columns), sequence_length), np.nan, dtype=float)
            numeric = ordered[columns].apply(pd.to_numeric, errors="coerce")
            matrix[:, indexes] = numeric.to_numpy(dtype=float).T
            if modality == "video":
                validity_index = columns.index("video_dynamic_validity")
                matrix[validity_index] = np.where(
                    np.isfinite(matrix[validity_index]), matrix[validity_index], 0.0
                )
            values[modality].append(matrix)
        first = ordered.iloc[0]
        context.append(
            pd.to_numeric(first[list(CONTEXT_COLUMNS)], errors="coerce").to_numpy(dtype=float)
        )
        targets.append(first[list(TARGETS)].to_numpy(dtype=float))
        participants.append(str(participant))
        conditions.append(str(condition))
        positions.append(float(first["presentation_position"]))
        lengths.append(int(indexes.max() + 1))
        video_available.append(
            float(
                pd.to_numeric(ordered["video_dynamic_validity"], errors="coerce")
                .fillna(0.0)
                .gt(0.5)
                .any()
            )
        )
    sequences = FiveModalitySequences(
        values={modality: np.stack(rows) for modality, rows in values.items()},
        context=np.stack(context),
        targets=np.stack(targets),
        participant_ids=np.asarray(participants, dtype=str),
        conditions=np.asarray(conditions, dtype=str),
        presentation_positions=np.asarray(positions, dtype=float),
        lengths=np.asarray(lengths, dtype=int),
        video_available=np.asarray(video_available, dtype=np.float32),
        feature_columns=feature_columns,
    )
    keys = set(zip(sequences.participant_ids, sequences.conditions, strict=True))
    if len(sequences.targets) != 81 or len(keys) != 81:
        raise ValueError("Five-modality sequence builder must produce exactly 81 unique labels")
    unavailable = [
        (str(sequences.participant_ids[index]), str(sequences.conditions[index]))
        for index in np.flatnonzero(sequences.video_available < 0.5)
    ]
    if unavailable != [ZERO_WINDOW_KEY]:
        raise ValueError(f"Unexpected zero-video conditions: {unavailable}")
    return sequences


def fit_five_modality_scaler(
    sequences: FiveModalitySequences,
    indexes: np.ndarray,
    *,
    min_non_missing_fraction: float,
) -> dict[str, Any]:
    scaler: dict[str, Any] = {"modalities": {}}
    for modality in MODALITIES:
        columns = list(sequences.feature_columns[modality])
        values = sequences.values[modality][indexes]
        if modality == "video":
            descriptor_count = len(DYNAMIC_TEXTURE_COLUMNS)
            descriptor_values = values[:, :descriptor_count, :]
            valid_fraction = np.mean(np.isfinite(descriptor_values), axis=(0, 2))
            selected = np.flatnonzero(valid_fraction >= min_non_missing_fraction)
            if len(selected) != descriptor_count:
                raise ValueError(
                    "All twelve video descriptors must survive fold-local availability"
                )
            selected_values = descriptor_values[:, selected, :]
            means = np.nanmean(selected_values, axis=(0, 2))
            scales = np.nanstd(selected_values, axis=(0, 2))
            scaler["modalities"][modality] = {
                "feature_indexes": selected.astype(int),
                "feature_columns": [columns[index] for index in selected],
                "feature_mean": np.where(np.isfinite(means), means, 0.0),
                "feature_scale": np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0),
                "append_validity": True,
            }
            continue
        valid_fraction = np.mean(np.isfinite(values), axis=(0, 2))
        selected = np.flatnonzero(valid_fraction >= min_non_missing_fraction)
        if not len(selected):
            raise ValueError(f"No {modality} features survive fold-local availability")
        selected_values = values[:, selected, :]
        means = np.nanmean(selected_values, axis=(0, 2))
        scales = np.nanstd(selected_values, axis=(0, 2))
        scaler["modalities"][modality] = {
            "feature_indexes": selected.astype(int),
            "feature_columns": [columns[index] for index in selected],
            "feature_mean": np.where(np.isfinite(means), means, 0.0),
            "feature_scale": np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0),
            "append_validity": False,
        }
    train_context = sequences.context[indexes]
    means = np.nanmean(train_context, axis=0)
    scales = np.nanstd(train_context, axis=0)
    scaler["context_mean"] = np.where(np.isfinite(means), means, 0.0)
    scaler["context_scale"] = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    return scaler


def transform_five_modality_sequences(
    sequences: FiveModalitySequences,
    indexes: np.ndarray,
    scaler: dict[str, Any],
    *,
    variant: str,
    prefix_lengths: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown dynamic-texture variant: {variant!r}")
    transformed: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        node = scaler["modalities"][modality]
        selected = np.asarray(node["feature_indexes"], dtype=int)
        values = sequences.values[modality][indexes][:, selected, :].copy()
        if prefix_lengths is not None:
            for position, prefix in enumerate(np.asarray(prefix_lengths, dtype=int)):
                values[position, :, max(0, int(prefix)) :] = np.nan
        means = np.asarray(node["feature_mean"], dtype=float)[None, :, None]
        scales = np.asarray(node["feature_scale"], dtype=float)[None, :, None]
        values = (values - means) / scales
        values = np.where(np.isfinite(values), values, 0.0).astype(np.float32)
        if modality == "video":
            validity = sequences.values["video"][indexes][
                :, len(DYNAMIC_TEXTURE_COLUMNS) : len(DYNAMIC_TEXTURE_COLUMNS) + 1, :
            ].copy()
            if prefix_lengths is not None:
                for position, prefix in enumerate(np.asarray(prefix_lengths, dtype=int)):
                    validity[position, :, max(0, int(prefix)) :] = 0.0
            validity = np.where(np.isfinite(validity), validity, 0.0).astype(np.float32)
            values = np.concatenate((values, validity), axis=1)
            if variant == "no_video":
                values.fill(0.0)
        transformed[modality] = values
    context = sequences.context[indexes].copy()
    context = (context - np.asarray(scaler["context_mean"], dtype=float)) / np.asarray(
        scaler["context_scale"], dtype=float
    )
    context = np.where(np.isfinite(context), context, 0.0).astype(np.float32)
    video_available = sequences.video_available[indexes].astype(np.float32)
    if variant == "no_video":
        video_available = np.zeros_like(video_available)
    if variant == "no_video" and (
        np.any(transformed["video"] != 0.0) or np.any(video_available != 0.0)
    ):
        raise AssertionError("no_video must present exact zero video input, validity, and gate")
    return transformed, context, video_available


def _encoder_output_size(
    feature_count: int,
    sequence_length: int,
    conv_channels: tuple[int, ...],
    pool_sizes: tuple[int, ...],
) -> int:
    output_length = int(sequence_length)
    for pool in pool_sizes:
        output_length //= int(pool)
    if output_length < 1:
        raise ValueError("Pooling collapses the five-branch temporal sequence")
    return int(feature_count * conv_channels[-1] * output_length)


def make_five_branch_model(
    feature_counts: dict[str, int],
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    context_size: int = len(CONTEXT_COLUMNS),
    conv_channels: tuple[int, ...] = (16, 32),
    kernel_sizes: tuple[int, ...] = (3, 3),
    pool_sizes: tuple[int, ...] = (2, 2),
    mlp_hidden: int = 64,
    dropout: float = 0.30,
) -> Any:
    """Instantiate the fixed-capacity five-branch network."""
    torch = _torch()
    nn = torch.nn
    if set(feature_counts) != set(MODALITIES):
        raise ValueError("Five-branch feature counts must declare all five modalities")
    if not (len(conv_channels) == len(kernel_sizes) == len(pool_sizes)):
        raise ValueError("Convolution, kernel, and pool specifications must have equal lengths")

    class PerFeatureEncoder(nn.Module):
        def __init__(self, feature_count: int) -> None:
            super().__init__()
            if feature_count < 1:
                raise ValueError("Each modality branch requires at least one feature stream")
            layers: list[Any] = []
            input_channels = 1
            for output_channels, kernel, pool in zip(
                conv_channels, kernel_sizes, pool_sizes, strict=True
            ):
                layers.extend(
                    (
                        nn.Conv1d(
                            feature_count * input_channels,
                            feature_count * output_channels,
                            kernel_size=kernel,
                            padding=kernel // 2,
                            groups=feature_count,
                        ),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=pool),
                        nn.Dropout(dropout),
                    )
                )
                input_channels = output_channels
            self.network = nn.Sequential(*layers)

        def forward(self, value: Any) -> Any:
            if value.ndim != 3:
                raise ValueError("Every modality branch expects [batch, feature, time]")
            return self.network(value).flatten(1)

    class DynamicTextureFiveBranchCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoders = nn.ModuleDict(
                {
                    modality: PerFeatureEncoder(int(feature_counts[modality]))
                    for modality in MODALITIES[:-1]
                }
            )
            self.video_encoder = PerFeatureEncoder(int(feature_counts["video"]))
            embedding_sizes = {
                modality: _encoder_output_size(
                    int(feature_counts[modality]),
                    sequence_length,
                    conv_channels,
                    pool_sizes,
                )
                for modality in MODALITIES
            }
            self.embedding_sizes = embedding_sizes
            self.fusion_input_dim = int(sum(embedding_sizes.values()) + context_size)
            self.fc1 = nn.Linear(self.fusion_input_dim, mlp_hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(mlp_hidden, len(TARGETS))

        def forward(
            self,
            inputs: dict[str, Any],
            context: Any,
            video_available: Any,
            *,
            return_video_embedding: bool = False,
        ) -> Any:
            if set(inputs) != set(MODALITIES):
                raise ValueError("Five-branch forward requires exactly five modality tensors")
            video_input = inputs["video"]
            if video_input.ndim == 5:
                raise ValueError("RGB tensors are rejected by the five-branch training interface")
            embeddings = [self.encoders[name](inputs[name]) for name in MODALITIES[:-1]]
            video_embedding = self.video_encoder(video_input)
            gate = video_available.reshape(-1, 1).to(
                device=video_embedding.device, dtype=video_embedding.dtype
            )
            video_embedding = video_embedding * gate
            fused = torch.cat([*embeddings, video_embedding, context], dim=1)
            output = torch.sigmoid(self.output(self.dropout(self.relu(self.fc1(fused)))))
            return (output, video_embedding) if return_video_embedding else output

    return DynamicTextureFiveBranchCNN()


def _architecture(config: ProjectConfig) -> dict[str, Any]:
    node = dict(config.get("modeling.dcnn", {}))
    architecture = {
        "sequence_length": int(node.get("sequence_length", SEQUENCE_LENGTH)),
        "conv_channels": tuple(int(value) for value in node.get("conv_channels", [16, 32])),
        "kernel_sizes": tuple(int(value) for value in node.get("kernel_sizes", [3, 3])),
        "pool_sizes": tuple(int(value) for value in node.get("pool_sizes", [2, 2])),
        "mlp_hidden": int(node.get("mlp_hidden", 64)),
        "dropout": float(node.get("dropout", 0.30)),
    }
    if architecture["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("Formal Project A five-branch runs require sequence_length=8")
    return architecture


def _feature_counts(scaler: dict[str, Any]) -> dict[str, int]:
    return {
        modality: int(len(scaler["modalities"][modality]["feature_indexes"]))
        + (1 if modality == "video" else 0)
        for modality in MODALITIES
    }


def _make_model_for_scaler(scaler: dict[str, Any], architecture: dict[str, Any]) -> Any:
    return make_five_branch_model(
        _feature_counts(scaler),
        sequence_length=int(architecture["sequence_length"]),
        conv_channels=tuple(architecture["conv_channels"]),
        kernel_sizes=tuple(architecture["kernel_sizes"]),
        pool_sizes=tuple(architecture["pool_sizes"]),
        mlp_hidden=int(architecture["mlp_hidden"]),
        dropout=float(architecture["dropout"]),
    )


def _to_torch_inputs(
    inputs: dict[str, np.ndarray], context: np.ndarray, gate: np.ndarray, device: Any
) -> tuple[dict[str, Any], Any, Any]:
    torch = _torch()
    return (
        {
            modality: torch.as_tensor(value, dtype=torch.float32, device=device)
            for modality, value in inputs.items()
        },
        torch.as_tensor(context, dtype=torch.float32, device=device),
        torch.as_tensor(gate, dtype=torch.float32, device=device),
    )


def _condition_baseline_arrays(
    sequences: FiveModalitySequences,
    train_indexes: np.ndarray,
    target_indexes: np.ndarray,
) -> np.ndarray:
    fallback = np.mean(sequences.targets[train_indexes], axis=0)
    by_condition: dict[str, np.ndarray] = {}
    for condition in np.unique(sequences.conditions[train_indexes]):
        rows = train_indexes[sequences.conditions[train_indexes] == condition]
        by_condition[str(condition)] = np.mean(sequences.targets[rows], axis=0)
    return np.asarray(
        [by_condition.get(str(sequences.conditions[index]), fallback) for index in target_indexes],
        dtype=float,
    )


def _zero_sequence_mask(sequences: FiveModalitySequences, indexes: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (
                str(sequences.participant_ids[index]) == ZERO_WINDOW_KEY[0]
                and str(sequences.conditions[index]) == ZERO_WINDOW_KEY[1]
            )
            for index in indexes
        ],
        dtype=bool,
    )


def _predict_dcnn(
    model: Any,
    sequences: FiveModalitySequences,
    indexes: np.ndarray,
    scaler: dict[str, Any],
    *,
    variant: str,
    device: Any,
) -> np.ndarray:
    torch = _torch()
    inputs, context, gate = transform_five_modality_sequences(
        sequences, indexes, scaler, variant=variant
    )
    torch_inputs, torch_context, torch_gate = _to_torch_inputs(inputs, context, gate, device)
    with torch.no_grad():
        model.eval()
        return model(torch_inputs, torch_context, torch_gate).detach().cpu().numpy().astype(float)


def _train_dcnn_variant(
    sequences: FiveModalitySequences,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    scaler: dict[str, Any],
    architecture: dict[str, Any],
    config: ProjectConfig,
    *,
    variant: str,
    seed: int,
    device: Any,
    initial_state: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    node = dict(config.get("modeling.dcnn", {}))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = _make_model_for_scaler(scaler, architecture).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(node.get("learning_rate", 0.001)),
        weight_decay=float(node.get("weight_decay", 0.0001)),
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = int(node.get("batch_size", 16))
    maximum_epochs = int(node.get("max_epochs", 40))
    patience = int(node.get("early_stopping_patience", 8))
    rng = np.random.default_rng(int(seed))
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    maximum_video_gradient = 0.0
    for epoch in range(maximum_epochs):
        prefixes = np.asarray(
            [rng.integers(1, int(sequences.lengths[index]) + 1) for index in train_indexes],
            dtype=int,
        )
        inputs, context, gate = transform_five_modality_sequences(
            sequences,
            train_indexes,
            scaler,
            variant=variant,
            prefix_lengths=prefixes,
        )
        order = rng.permutation(len(train_indexes))
        model.train()
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            global_indexes = train_indexes[batch]
            keep = ~_zero_sequence_mask(sequences, global_indexes)
            if not np.any(keep):
                continue
            batch_inputs = {modality: value[batch][keep] for modality, value in inputs.items()}
            batch_context = context[batch][keep]
            batch_gate = gate[batch][keep]
            torch_inputs, torch_context, torch_gate = _to_torch_inputs(
                batch_inputs, batch_context, batch_gate, device
            )
            target = torch.as_tensor(
                sequences.targets[global_indexes][keep],
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(torch_inputs, torch_context, torch_gate), target)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.video_encoder.parameters()
                if parameter.grad is not None
            ]
            gradient = max(
                (float(value.detach().abs().max().item()) for value in gradients),
                default=0.0,
            )
            maximum_video_gradient = max(maximum_video_gradient, gradient)
            if variant == "no_video":
                if gradient != 0.0:
                    raise AssertionError("no_video produced a nonzero Video encoder gradient")
                for parameter in model.video_encoder.parameters():
                    parameter.grad = None
            optimizer.step()
        validation_prediction = _predict_dcnn(
            model,
            sequences,
            validation_indexes,
            scaler,
            variant=variant,
            device=device,
        )
        validation_baseline = _condition_baseline_arrays(
            sequences, train_indexes, validation_indexes
        )
        zero = _zero_sequence_mask(sequences, validation_indexes)
        validation_prediction[zero] = validation_baseline[zero]
        validation_loss = float(
            np.mean((validation_prediction - sequences.targets[validation_indexes]) ** 2)
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Five-branch training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    if variant == "full_five" and maximum_video_gradient <= 0.0:
        raise AssertionError("full_five never produced a nonzero Video encoder gradient")
    if variant == "no_video" and maximum_video_gradient != 0.0:
        raise AssertionError("no_video accumulated a nonzero Video encoder gradient")
    return model, {
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch + 1),
        "validation_loss": float(best_loss),
        "maximum_video_gradient": float(maximum_video_gradient),
    }


def audit_video_gate(
    model: Any,
    inputs: dict[str, np.ndarray],
    context: np.ndarray,
    *,
    device: Any,
) -> dict[str, Any]:
    """Prove exact post-encoder isolation for a no-video forward pass."""
    torch = _torch()
    zero_inputs = {name: value.copy() for name, value in inputs.items()}
    zero_inputs["video"].fill(0.0)
    zero_gate = np.zeros(len(context), dtype=np.float32)
    torch_inputs, torch_context, torch_gate = _to_torch_inputs(
        zero_inputs, context, zero_gate, device
    )
    model.eval()
    with torch.no_grad():
        prediction_before, embedding_before = model(
            torch_inputs,
            torch_context,
            torch_gate,
            return_video_embedding=True,
        )
    video_state = deepcopy(
        {name: value.detach().cpu() for name, value in model.video_encoder.state_dict().items()}
    )
    with torch.no_grad():
        for parameter in model.video_encoder.parameters():
            parameter.add_(7.0)
        prediction_perturbed, embedding_perturbed = model(
            torch_inputs,
            torch_context,
            torch_gate,
            return_video_embedding=True,
        )
    parameter_invariant = bool(torch.equal(prediction_before, prediction_perturbed))
    model.video_encoder.load_state_dict(video_state)
    bias_model = deepcopy(model)
    with torch.no_grad():
        for name, parameter in bias_model.video_encoder.named_parameters():
            if name.endswith("bias"):
                parameter.fill_(3.25)
        _, embedding_with_bias = bias_model(
            torch_inputs,
            torch_context,
            torch_gate,
            return_video_embedding=True,
        )
    gradient_model = deepcopy(model)
    gradient_model.train()
    gradient_model.zero_grad(set_to_none=True)
    prediction = gradient_model(torch_inputs, torch_context, torch_gate)
    prediction.sum().backward()
    gradient_max = max(
        (
            float(parameter.grad.detach().abs().max().item())
            for parameter in gradient_model.video_encoder.parameters()
            if parameter.grad is not None
        ),
        default=0.0,
    )
    record = {
        "no_video_input_max_abs": float(np.max(np.abs(zero_inputs["video"]))),
        "no_video_gate_max_abs": float(np.max(np.abs(zero_gate))),
        "embedding_max_abs": float(embedding_before.detach().abs().max().item()),
        "embedding_after_parameter_perturbation_max_abs": float(
            embedding_perturbed.detach().abs().max().item()
        ),
        "embedding_with_nonzero_bias_max_abs": float(
            embedding_with_bias.detach().abs().max().item()
        ),
        "prediction_parameter_invariant_exact": parameter_invariant,
        "video_encoder_gradient_max_abs": float(gradient_max),
    }
    if (
        any(
            record[name] != 0.0
            for name in (
                "no_video_input_max_abs",
                "no_video_gate_max_abs",
                "embedding_max_abs",
                "embedding_after_parameter_perturbation_max_abs",
                "embedding_with_nonzero_bias_max_abs",
                "video_encoder_gradient_max_abs",
            )
        )
        or not parameter_invariant
    ):
        raise AssertionError(f"Post-encoder no-video gate audit failed: {record}")
    return record


def _runtime_provenance(device: Any) -> dict[str, Any]:
    torch = _torch()
    return {
        "device": str(device),
        "cuda_used": bool(device.type == "cuda"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
    }


def run_paired_dcnn_dynamic_texture(
    config: ProjectConfig,
    *,
    window_features: str | Path,
    contract_dir: str | Path,
    run_roots: dict[str, str | Path],
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Train the formal full/no-video pair with identical fold initialization."""
    import pandas as pd

    torch = _torch()
    if set(run_roots) != set(VARIANTS):
        raise ValueError("Paired DCNN run roots must contain full_five and no_video")
    if int(config.get("modeling.random_seed")) != int(seed):
        raise ValueError("Resolved config seed differs from the requested DCNN pair")
    requested_device = str(config.get("modeling.dcnn.device", "cuda"))
    if not requested_device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Formal Project A five-branch runs require an available CUDA device")
    device = torch.device(requested_device)
    runtime = _runtime_provenance(device)
    source = Path(window_features).resolve()
    contract_root = Path(contract_dir).resolve()
    contract, paths = _contract_paths(contract_root)
    sequences = build_five_modality_sequences(source)
    folds = _folds(contract, paths, sequences.participant_ids)
    architecture = _architecture(config)
    min_fraction = float(config.get("modeling.dcnn.min_non_missing_fraction", 0.40))
    roots = {variant: Path(run_roots[variant]).resolve() for variant in VARIANTS}
    for root in roots.values():
        for child in ("models", "metrics", "predictions", "manifests", "logs"):
            (root / child).mkdir(parents=True, exist_ok=True)
    predictions = {
        variant: np.full_like(sequences.targets, np.nan, dtype=float) for variant in VARIANTS
    }
    baselines = {
        variant: np.full_like(sequences.targets, np.nan, dtype=float) for variant in VARIANTS
    }
    row_fold = np.full(len(sequences.targets), -1, dtype=int)
    row_validation = np.full(len(sequences.targets), "", dtype=object)
    fold_records: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    gating_records: list[dict[str, Any]] = []
    for fold in folds:
        train_indexes, validation_indexes, test_indexes = indexes_for_fold(
            sequences.participant_ids, fold
        )
        scaler = fit_five_modality_scaler(
            sequences,
            train_indexes,
            min_non_missing_fraction=min_fraction,
        )
        feature_counts = _feature_counts(scaler)
        torch.manual_seed(int(seed + fold.fold_index))
        torch.cuda.manual_seed_all(int(seed + fold.fold_index))
        initial_model = _make_model_for_scaler(scaler, architecture).to(device)
        initial_state = deepcopy(
            {name: value.detach().cpu() for name, value in initial_model.state_dict().items()}
        )
        initialization_hash = state_dict_sha256(initial_state)
        signature = parameter_signature(initial_model)
        parameter_count = int(sum(item["parameter_count"] for item in signature))
        fusion_input_dim = int(initial_model.fusion_input_dim)
        transformed_full, context_full, gate_full = transform_five_modality_sequences(
            sequences, test_indexes, scaler, variant="full_five"
        )
        transformed_no, context_no, gate_no = transform_five_modality_sequences(
            sequences, test_indexes, scaler, variant="no_video"
        )
        nonvideo_full_hash = {
            modality: _array_hash(transformed_full[modality]) for modality in MODALITIES[:-1]
        }
        nonvideo_no_hash = {
            modality: _array_hash(transformed_no[modality]) for modality in MODALITIES[:-1]
        }
        if nonvideo_full_hash != nonvideo_no_hash or not np.array_equal(context_full, context_no):
            raise AssertionError("Paired variants received different non-video inputs")
        initial_gate_audit = audit_video_gate(
            initial_model, transformed_no, context_no, device=device
        )
        del initial_model
        fold_models: dict[str, Any] = {}
        for variant in VARIANTS:
            model, training_record = _train_dcnn_variant(
                sequences,
                train_indexes,
                validation_indexes,
                scaler,
                architecture,
                config,
                variant=variant,
                seed=seed + fold.fold_index,
                device=device,
                initial_state=initial_state,
            )
            fold_models[variant] = model
            prediction = _predict_dcnn(
                model,
                sequences,
                test_indexes,
                scaler,
                variant=variant,
                device=device,
            )
            baseline = _condition_baseline_arrays(sequences, train_indexes, test_indexes)
            zero = _zero_sequence_mask(sequences, test_indexes)
            prediction[zero] = baseline[zero]
            predictions[variant][test_indexes] = prediction
            baselines[variant][test_indexes] = baseline
            checkpoint = {
                "schema_version": "project_a_dynamic_texture_dcnn_fold_v1",
                "research_only": True,
                "variant": variant,
                "seed": int(seed),
                "training_seed": int(seed + fold.fold_index),
                "fold": asdict(fold),
                "architecture": architecture,
                "feature_counts": feature_counts,
                "feature_columns": {
                    modality: list(scaler["modalities"][modality]["feature_columns"])
                    + (["video_dynamic_validity"] if modality == "video" else [])
                    for modality in MODALITIES
                },
                "scaler": scaler,
                "initialization_state_sha256": initialization_hash,
                "parameter_signature": signature,
                "parameter_count": parameter_count,
                "fusion_input_dim": fusion_input_dim,
                "training": training_record,
                "runtime": runtime,
                "state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
            }
            checkpoint_path = roots[variant] / "models" / f"fold_{fold.fold_index:02d}.pt"
            torch.save(checkpoint, checkpoint_path)
            fold_records[variant].append(
                {
                    "fold_index": int(fold.fold_index),
                    "train_participants": list(fold.train_participants),
                    "validation_participant": fold.validation_participant,
                    "test_participant": fold.test_participant,
                    "n_train_participants": len(fold.train_participants),
                    "n_validation_participants": 1,
                    "n_test_participants": 1,
                    "feature_counts": feature_counts,
                    "parameter_count": parameter_count,
                    "fusion_input_dim": fusion_input_dim,
                    "initialization_state_sha256": initialization_hash,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    **training_record,
                    **runtime,
                }
            )
        trained_signature = {
            variant: parameter_signature(fold_models[variant]) for variant in VARIANTS
        }
        if trained_signature["full_five"] != trained_signature["no_video"]:
            raise AssertionError("Paired variants have different parameter names or shapes")
        trained_gate_audit = audit_video_gate(
            fold_models["no_video"], transformed_no, context_no, device=device
        )
        full_test_available = bool(np.any(gate_full > 0.5))
        gating_records.append(
            {
                "fold_index": int(fold.fold_index),
                "test_participant": fold.test_participant,
                "initialization_state_sha256_full": initialization_hash,
                "initialization_state_sha256_no_video": initialization_hash,
                "parameter_signature_equal": True,
                "parameter_count_full": parameter_count,
                "parameter_count_no_video": parameter_count,
                "fusion_input_dim_full": fusion_input_dim,
                "fusion_input_dim_no_video": fusion_input_dim,
                "nonvideo_input_sha256_full": sha256(
                    json.dumps(nonvideo_full_hash, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "nonvideo_input_sha256_no_video": sha256(
                    json.dumps(nonvideo_no_hash, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "nonvideo_context_equal": bool(np.array_equal(context_full, context_no)),
                "full_test_video_available": full_test_available,
                "p004_c6_full_gate_zero": bool(
                    np.all(gate_full[_zero_sequence_mask(sequences, test_indexes)] == 0.0)
                ),
                **{f"initial_{key}": value for key, value in initial_gate_audit.items()},
                **{f"trained_{key}": value for key, value in trained_gate_audit.items()},
            }
        )
        row_fold[test_indexes] = int(fold.fold_index)
        row_validation[test_indexes] = fold.validation_participant
        print(
            f"Dynamic-texture paired DCNN fold {fold.fold_index}/{len(folds)}: "
            f"test={fold.test_participant}",
            flush=True,
        )
        del fold_models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    outputs: dict[str, dict[str, Any]] = {}
    gating_frame = pd.DataFrame(gating_records)
    for variant in VARIANTS:
        root = roots[variant]
        prediction_frame = pd.DataFrame(
            {
                "participant_id": sequences.participant_ids,
                "condition": sequences.conditions,
                "presentation_position": sequences.presentation_positions,
                "true_relaxation": sequences.targets[:, 0],
                "true_discomfort": sequences.targets[:, 1],
                "pred_relaxation": predictions[variant][:, 0],
                "pred_discomfort": predictions[variant][:, 1],
                "condition_only_relaxation": baselines[variant][:, 0],
                "condition_only_discomfort": baselines[variant][:, 1],
            }
        )
        prediction_frame["project"] = "A"
        prediction_frame["model_family"] = "1dcnn"
        prediction_frame["variant"] = variant
        prediction_frame["modalities"] = (
            "eeg+ecg+eye+head+video" if variant == "full_five" else "eeg+ecg+eye+head"
        )
        prediction_frame["seed"] = int(seed)
        prediction_frame["fold_index"] = row_fold
        prediction_frame["test_participant"] = prediction_frame["participant_id"]
        prediction_frame["validation_participant"] = row_validation
        prediction_frame["split_protocol"] = "shared_7_train_1_validation_1_test"
        prediction_frame["device"] = str(device)
        prediction_frame["cuda_used"] = True
        prediction_frame["video_representation"] = (
            "dynamic_texture_v1" if variant == "full_five" else "post_encoder_gated_zero"
        )
        prediction_frame["video_available"] = (
            sequences.video_available if variant == "full_five" else 0.0
        )
        prediction_frame["p004_c6_condition_only_fallback"] = _zero_sequence_mask(
            sequences, np.arange(len(sequences.targets))
        )
        zero = prediction_frame["p004_c6_condition_only_fallback"].to_numpy(dtype=bool)
        for target in TARGETS:
            if not np.array_equal(
                prediction_frame.loc[zero, f"pred_{target}"].to_numpy(),
                prediction_frame.loc[zero, f"condition_only_{target}"].to_numpy(),
            ):
                raise AssertionError("P004/C6 DCNN prediction is not exact Condition-only")
        if prediction_frame[[f"pred_{target}" for target in TARGETS]].isna().any().any():
            raise ValueError("DCNN OOF predictions are incomplete")
        metrics = project_b_style_metrics(prediction_frame)
        prediction_path = root / "predictions" / "oof_predictions.csv"
        prediction_frame.to_csv(prediction_path, index=False)
        gate_path = root / "manifests" / "gating_audit.csv"
        gating_frame.to_csv(gate_path, index=False)
        metrics_payload = {
            "schema_version": "project_a_dynamic_texture_dcnn_run_v1",
            "research_only": True,
            "project": "A",
            "model_family": "1dcnn",
            "variant": variant,
            "modalities": list(MODALITIES if variant == "full_five" else MODALITIES[:-1]),
            "video_representation": (
                "dynamic_texture_v1" if variant == "full_five" else "post_encoder_gated_zero"
            ),
            "seed": int(seed),
            "split_protocol": "shared_7_train_1_validation_1_test",
            "architecture": architecture,
            "metrics": metrics,
            "folds": fold_records[variant],
            "fallback": {
                "participant_id": ZERO_WINDOW_KEY[0],
                "condition": ZERO_WINDOW_KEY[1],
                "exact_condition_only": True,
            },
            "runtime": runtime,
        }
        metrics_path = root / "metrics" / "metrics.json"
        write_json(metrics_path, metrics_payload)
        manifest = {
            "schema_version": "project_a_dynamic_texture_run_manifest_v1",
            "run_id": root.name,
            "seed": int(seed),
            "model_family": "1dcnn",
            "variant": variant,
            "sources": {
                "window_features": {"path": str(source), "sha256": file_sha256(source)},
                **{
                    name: {"path": str(path), "sha256": file_sha256(path)}
                    for name, path in paths.items()
                },
            },
            "outputs": {
                "predictions": {
                    "path": str(prediction_path),
                    "sha256": file_sha256(prediction_path),
                    "rows": len(prediction_frame),
                },
                "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
                "gating_audit": {"path": str(gate_path), "sha256": file_sha256(gate_path)},
            },
            "folds": [asdict(fold) for fold in folds],
            "runtime": runtime,
        }
        write_json(root / "manifests" / "run_manifest.json", manifest)
        outputs[variant] = metrics_payload
    return outputs


__all__ = [
    "CONTEXT_COLUMNS",
    "MODALITIES",
    "SEQUENCE_LENGTH",
    "TARGETS",
    "VARIANTS",
    "FiveModalitySequences",
    "audit_video_gate",
    "build_five_modality_sequences",
    "concordance_correlation_coefficient",
    "fit_five_modality_scaler",
    "make_five_branch_model",
    "parameter_signature",
    "project_b_style_metrics",
    "run_classical_dynamic_texture",
    "run_paired_dcnn_dynamic_texture",
    "state_dict_sha256",
    "transform_five_modality_sequences",
]
