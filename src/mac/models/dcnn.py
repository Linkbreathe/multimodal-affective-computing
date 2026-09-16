"""Condition-level multi-stream 1D-CNN regression.

The layout follows egoEMOTION's DCNN design: every extracted feature has its own
one-dimensional convolutional stream and the stream embeddings are concatenated.
Here the sequence axis is the seven causal 10-second windows in a Condition and
the head is a two-target regression head instead of a classifier. The sequence
holds up to eight windows because P005/C2 contains an eighth complete window.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mac.config import ProjectConfig
from mac.data.io import condition_parameters
from mac.evaluation.alignment import (
    indexes_for_fold,
    load_split_manifest,
    validate_alignment_contract,
    write_alignment_manifest,
)
from mac.evaluation.safety import deployment_guard
from mac.utils import atomic_write_text, write_json


MODEL_KIND = "dcnn_condition_regressor_v1"
TARGETS = ("relaxation", "discomfort")
CONTEXT_COLUMNS = ("intensity", "frequency")
PROJECT_A_MODALITIES = ("eeg", "ecg", "eye", "head")
VARIANTS = ("full", "no_eeg", "no_ecg", "no_eye", "no_head", "behavior_only")
REALTIME_PREFIXES = ("eeg_", "ecg_", "head_", "eye_")


@dataclass(frozen=True)
class ConditionSequences:
    """In-memory tensor representation of one sequence per participant/Condition.

    ``values`` has shape ``[N, F, T]``: N condition labels, F selected
    realtime features, and T causal 10-second positions.  Missing future
    windows are represented by NaN until ``_transform`` converts them to the
    model's zero-filled input.
    """

    values: np.ndarray
    context: np.ndarray
    targets: np.ndarray
    participant_ids: np.ndarray
    conditions: np.ndarray
    presentation_positions: np.ndarray
    lengths: np.ndarray
    feature_columns: tuple[str, ...]


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "DCNN requires PyTorch. Install the CUDA build selected for this machine before "
            "running 'rtml train-dcnn-state' or setting modeling.runtime_backend=dcnn."
        ) from error
    return torch


def _device(value: str):
    torch = _torch()
    requested = str(value or "cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"DCNN is configured for {requested}, but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build or explicitly configure device: cpu."
            )
        return torch.device(requested)
    return torch.device("cpu")


def _runtime_provenance(device) -> dict[str, Any]:
    torch = _torch()
    cuda = device.type == "cuda"
    return {
        "device": str(device),
        "cuda_used": bool(cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if cuda else None,
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
    }


@lru_cache(maxsize=1)
def _model_types():
    torch = _torch()
    nn = torch.nn

    class CNNStream(nn.Module):
        def __init__(self, conv_channels: tuple[int, ...], kernel_sizes: tuple[int, ...], pool_sizes: tuple[int, ...], dropout: float) -> None:
            super().__init__()
            layers: list[Any] = []
            in_channels = 1
            for out_channels, kernel, pool in zip(conv_channels, kernel_sizes, pool_sizes, strict=True):
                layers.extend((
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=pool),
                    nn.Dropout(dropout),
                ))
                in_channels = out_channels
            self.network = nn.Sequential(*layers)

        def forward(self, value):
            # The caller supplies one feature's [batch, time] slice.  Adding a
            # singleton channel turns it into Conv1d's [batch, channel, time].
            return self.network(value.unsqueeze(1)).flatten(1)

    class ConditionDCNN(nn.Module):
        def __init__(
            self,
            *,
            n_features: int,
            sequence_length: int,
            context_size: int,
            output_size: int,
            output_activation: str,
            conv_channels: tuple[int, ...],
            kernel_sizes: tuple[int, ...],
            pool_sizes: tuple[int, ...],
            mlp_hidden: int,
            dropout: float,
        ) -> None:
            super().__init__()
            output_length = int(sequence_length)
            for pool in pool_sizes:
                output_length //= int(pool)
            if output_length < 1:
                raise ValueError(
                    "DCNN pooling collapses the sequence. Use the configured two-pool architecture "
                    "for the configured Condition sequence."
                )
            if n_features < 1:
                raise ValueError("DCNN requires at least one temporal feature stream")
            if context_size < 0:
                raise ValueError("DCNN context_size cannot be negative")
            if output_size < 1:
                raise ValueError("DCNN output_size must be positive")
            if output_activation not in {"sigmoid", "tanh"}:
                raise ValueError("DCNN output_activation must be 'sigmoid' or 'tanh'")
            self.streams = nn.ModuleList(
                CNNStream(conv_channels, kernel_sizes, pool_sizes, dropout) for _ in range(n_features)
            )
            flattened = n_features * conv_channels[-1] * output_length
            self.fc1 = nn.Linear(flattened + context_size, mlp_hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(mlp_hidden, output_size)
            self.output_activation = output_activation

        def forward(self, values, context):
            # Separate streams prevent one feature from mixing with another
            # before its own temporal pattern has been encoded.
            features = [stream(values[:, index, :]) for index, stream in enumerate(self.streams)]
            # Stimulus context (intensity/frequency) is not temporal signal;
            # append it only at the final fusion point.
            merged = torch.cat([*features, context], dim=1)
            raw = self.output(self.dropout(self.relu(self.fc1(merged))))
            return torch.sigmoid(raw) if self.output_activation == "sigmoid" else torch.tanh(raw)

    class GroupedConditionDCNN(nn.Module):
        """Equivalent independently-parameterized streams executed as grouped convolutions.

        This is intended for research sweeps with many short temporal models.
        A group owns one input feature and its own convolution kernels, just as
        ``ConditionDCNN.streams`` does, but avoids one Python/CUDA launch path
        per feature.
        """

        def __init__(
            self,
            *,
            n_features: int,
            sequence_length: int,
            context_size: int,
            output_size: int,
            output_activation: str,
            conv_channels: tuple[int, ...],
            kernel_sizes: tuple[int, ...],
            pool_sizes: tuple[int, ...],
            mlp_hidden: int,
            dropout: float,
        ) -> None:
            super().__init__()
            output_length = int(sequence_length)
            for pool in pool_sizes:
                output_length //= int(pool)
            if output_length < 1:
                raise ValueError(
                    "DCNN pooling collapses the sequence. Use the configured two-pool architecture "
                    "for the configured Condition sequence."
                )
            if n_features < 1:
                raise ValueError("DCNN requires at least one temporal feature stream")
            if context_size < 0:
                raise ValueError("DCNN context_size cannot be negative")
            if output_size < 1:
                raise ValueError("DCNN output_size must be positive")
            if output_activation not in {"sigmoid", "tanh"}:
                raise ValueError("DCNN output_activation must be 'sigmoid' or 'tanh'")
            layers: list[Any] = []
            input_channels = 1
            for output_channels, kernel, pool in zip(
                conv_channels, kernel_sizes, pool_sizes, strict=True
            ):
                layers.extend(
                    (
                        nn.Conv1d(
                            n_features * input_channels,
                            n_features * output_channels,
                            kernel_size=kernel,
                            padding=kernel // 2,
                            groups=n_features,
                        ),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=pool),
                        nn.Dropout(dropout),
                    )
                )
                input_channels = output_channels
            self.network = nn.Sequential(*layers)
            flattened = n_features * conv_channels[-1] * output_length
            self.fc1 = nn.Linear(flattened + context_size, mlp_hidden)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(mlp_hidden, output_size)
            self.output_activation = output_activation

        def forward(self, values, context):
            features = self.network(values).flatten(1)
            merged = torch.cat([features, context], dim=1)
            raw = self.output(self.dropout(self.relu(self.fc1(merged))))
            return torch.sigmoid(raw) if self.output_activation == "sigmoid" else torch.tanh(raw)

    ConditionDCNN.grouped_type = GroupedConditionDCNN
    return ConditionDCNN


def _architecture(config: ProjectConfig) -> dict[str, Any]:
    node = dict(config.get("modeling.dcnn", {}))
    architecture = {
        "sequence_length": int(node.get("sequence_length", 8)),
        "conv_channels": tuple(int(value) for value in node.get("conv_channels", [16, 32])),
        "kernel_sizes": tuple(int(value) for value in node.get("kernel_sizes", [3, 3])),
        "pool_sizes": tuple(int(value) for value in node.get("pool_sizes", [2, 2])),
        "mlp_hidden": int(node.get("mlp_hidden", 64)),
        "dropout": float(node.get("dropout", 0.3)),
    }
    if not (len(architecture["conv_channels"]) == len(architecture["kernel_sizes"]) == len(architecture["pool_sizes"])):
        raise ValueError("modeling.dcnn conv_channels, kernel_sizes and pool_sizes must have equal lengths")
    if architecture["sequence_length"] < int(np.prod(architecture["pool_sizes"])):
        raise ValueError("modeling.dcnn.sequence_length is too short for the configured pooling stack")
    return architecture


def _make_model(
    n_features: int,
    architecture: dict[str, Any],
    *,
    context_size: int = len(CONTEXT_COLUMNS),
    output_size: int = len(TARGETS),
    output_activation: str = "sigmoid",
    stream_execution: str = "separate",
):
    """Create the configured stream CNN.

    The defaults are the deployed two-target sigmoid model.  Research-only
    callers may request a different head without changing checkpoint loading
    or runtime inference behavior.
    """
    model_type = _model_types()
    if stream_execution == "grouped":
        model_type = model_type.grouped_type
    elif stream_execution != "separate":
        raise ValueError("DCNN stream_execution must be 'separate' or 'grouped'")
    return model_type(
        n_features=n_features,
        sequence_length=int(architecture["sequence_length"]),
        context_size=int(context_size),
        output_size=int(output_size),
        output_activation=str(output_activation),
        conv_channels=tuple(architecture["conv_channels"]),
        kernel_sizes=tuple(architecture["kernel_sizes"]),
        pool_sizes=tuple(architecture["pool_sizes"]),
        mlp_hidden=int(architecture["mlp_hidden"]),
        dropout=float(architecture["dropout"]),
    )


def _variant_columns(columns: Iterable[str], variant: str) -> list[str]:
    available = [name for name in columns if name.startswith(REALTIME_PREFIXES)]
    if variant == "full":
        return available
    if variant in {"no_eeg", "no_ecg", "no_eye", "no_head"}:
        removed = variant.removeprefix("no_")
        return [name for name in available if not name.startswith(f"{removed}_")]
    if variant == "behavior_only":
        return [name for name in available if name.startswith(("head_", "eye_"))]
    raise ValueError(f"Unknown DCNN model variant: {variant}")


def _variant_modalities(variant: str) -> tuple[str, ...]:
    if variant == "full":
        return PROJECT_A_MODALITIES
    if variant in {"no_eeg", "no_ecg", "no_eye", "no_head"}:
        removed = variant.removeprefix("no_")
        return tuple(modality for modality in PROJECT_A_MODALITIES if modality != removed)
    if variant == "behavior_only":
        return ("eye", "head")
    raise ValueError(f"Unknown DCNN model variant: {variant}")


def build_condition_sequences(path: Path, config: ProjectConfig, variant: str = "full") -> ConditionSequences:
    """Build one temporal sample per participant--Condition label.

    The inherited questionnaire values are verified to be constant but are retained once,
    not converted into independent window-level training rows.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"participant_id", "condition", "condition_window_index", *TARGETS, *CONTEXT_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Window feature table lacks DCNN fields: {sorted(missing)}")
    feature_columns = _variant_columns(frame.columns, variant)
    if not feature_columns:
        raise ValueError(f"No usable real-time feature columns for DCNN variant {variant!r}")
    sequence_length = int(_architecture(config)["sequence_length"])
    values: list[np.ndarray] = []
    context: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    participants: list[str] = []
    conditions: list[str] = []
    positions: list[float] = []
    lengths: list[int] = []
    # Grouping here is the key difference from naive window-level training:
    # each questionnaire label produces one sequence, not many independent
    # supervised rows.
    grouped = frame.groupby(["participant_id", "condition"], sort=True)
    for (participant_id, condition), group in grouped:
        ordered = group.copy()
        ordered["condition_window_index"] = pd.to_numeric(ordered["condition_window_index"], errors="coerce")
        ordered = ordered.dropna(subset=["condition_window_index"]).sort_values("condition_window_index")
        if ordered.empty:
            continue
        if ordered[list(TARGETS)].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent inherited questionnaire labels for {participant_id}/{condition}")
        indexes = ordered["condition_window_index"].astype(int).to_numpy()
        if indexes.min() < 0 or indexes.max() >= sequence_length or len(np.unique(indexes)) != len(indexes):
            raise ValueError(f"Invalid 0..{sequence_length - 1} window indexes for {participant_id}/{condition}")
        # Keep the time axis explicit.  A Condition with only five observed
        # windows has values in positions 0..4 and NaN in later positions.
        matrix = np.full((len(feature_columns), sequence_length), np.nan, dtype=float)
        numeric = ordered[feature_columns].apply(pd.to_numeric, errors="coerce")
        matrix[:, indexes] = numeric.to_numpy(dtype=float).T
        values.append(matrix)
        context.append(
            pd.to_numeric(ordered.iloc[0][list(CONTEXT_COLUMNS)], errors="coerce").to_numpy(dtype=float)
        )
        targets.append(ordered.iloc[0][list(TARGETS)].to_numpy(dtype=float))
        participants.append(str(participant_id))
        conditions.append(str(condition))
        positions.append(float(pd.to_numeric(ordered.iloc[0].get("presentation_position"), errors="coerce")))
        lengths.append(int(indexes.max() + 1))
    if not values:
        raise ValueError("No DCNN Condition sequences could be constructed")
    return ConditionSequences(
        values=np.stack(values),
        context=np.stack(context),
        targets=np.stack(targets),
        participant_ids=np.asarray(participants, dtype=str),
        conditions=np.asarray(conditions, dtype=str),
        presentation_positions=np.asarray(positions, dtype=float),
        lengths=np.asarray(lengths, dtype=int),
        feature_columns=tuple(feature_columns),
    )


def _fit_scaler(sequences: ConditionSequences, indexes: np.ndarray, min_fraction: float) -> dict[str, Any]:
    """Fit feature/context availability and standardization on train rows only."""
    train_values = sequences.values[indexes]
    valid_fraction = np.mean(np.isfinite(train_values), axis=(0, 2))
    # Feature availability is a learned property of the training fold.  Using
    # all participants here would leak test-participant recording quality.
    selected = np.flatnonzero(valid_fraction >= min_fraction)
    if not len(selected):
        raise ValueError("No DCNN features meet the training-fold availability threshold")
    selected_values = train_values[:, selected, :]
    means = np.nanmean(selected_values, axis=(0, 2))
    means = np.where(np.isfinite(means), means, 0.0)
    scales = np.nanstd(selected_values, axis=(0, 2))
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    train_context = sequences.context[indexes]
    context_means = np.nanmean(train_context, axis=0)
    context_means = np.where(np.isfinite(context_means), context_means, 0.0)
    context_scales = np.nanstd(train_context, axis=0)
    context_scales = np.where(np.isfinite(context_scales) & (context_scales > 1e-8), context_scales, 1.0)
    return {
        "feature_indexes": selected.astype(int),
        "feature_mean": means.astype(float),
        "feature_scale": scales.astype(float),
        "context_mean": context_means.astype(float),
        "context_scale": context_scales.astype(float),
    }


def _transform(
    sequences: ConditionSequences,
    indexes: np.ndarray,
    scaler: dict[str, Any],
    prefix_lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_indexes = np.asarray(scaler["feature_indexes"], dtype=int)
    values = sequences.values[indexes][:, feature_indexes, :].copy()
    if prefix_lengths is not None:
        # Prefix masking is used only during training augmentation and causal
        # inference: the network must not see windows that have not occurred.
        for position, prefix in enumerate(np.asarray(prefix_lengths, dtype=int)):
            values[position, :, max(0, int(prefix)) :] = np.nan
    means = np.asarray(scaler["feature_mean"], dtype=float)[None, :, None]
    scales = np.asarray(scaler["feature_scale"], dtype=float)[None, :, None]
    standardized = (values - means) / scales
    # Zero is the neutral value after train-fold standardization and also makes
    # absent windows safe for the convolutional stack.
    standardized = np.where(np.isfinite(standardized), standardized, 0.0).astype(np.float32)
    context = sequences.context[indexes].copy()
    context = (context - np.asarray(scaler["context_mean"], dtype=float)) / np.asarray(scaler["context_scale"], dtype=float)
    context = np.where(np.isfinite(context), context, 0.0).astype(np.float32)
    return standardized, context


def _train_model(
    sequences: ConditionSequences,
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    scaler: dict[str, Any],
    architecture: dict[str, Any],
    config: ProjectConfig,
    seed: int,
    device,
):
    """Train one fold with causal-prefix augmentation and participant validation."""
    torch = _torch()
    node = dict(config.get("modeling.dcnn", {}))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = _make_model(len(scaler["feature_indexes"]), architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(node.get("learning_rate", 1e-4)),
        weight_decay=float(node.get("weight_decay", 5e-5)),
    )
    loss_fn = torch.nn.MSELoss()
    batch_size = int(node.get("batch_size", 16))
    epochs = int(node.get("max_epochs", 100))
    patience = int(node.get("early_stopping_patience", 15))
    rng = np.random.default_rng(int(seed))
    validation_indexes = validation_indexes if len(validation_indexes) else train_indexes
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        # Give the model different amounts of available history each epoch so
        # it does not only learn the fully observed end-of-Condition case.
        prefixes = np.asarray([rng.integers(1, int(sequences.lengths[index]) + 1) for index in train_indexes], dtype=int)
        train_values, train_context = _transform(sequences, train_indexes, scaler, prefixes)
        order = rng.permutation(len(train_indexes))
        model.train()
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            values = torch.as_tensor(train_values[batch], device=device)
            context = torch.as_tensor(train_context[batch], device=device)
            targets = torch.as_tensor(sequences.targets[train_indexes][batch], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = loss_fn(model(values, context), targets)
            loss.backward()
            optimizer.step()
        # Validation uses the complete available sequence and never changes the
        # scaler fitted above.  The best checkpoint is restored after stopping.
        validation_values, validation_context = _transform(sequences, validation_indexes, scaler)
        with torch.no_grad():
            model.eval()
            prediction = model(
                torch.as_tensor(validation_values, device=device),
                torch.as_tensor(validation_context, device=device),
            )
            validation_loss = float(
                loss_fn(
                    prediction,
                    torch.as_tensor(sequences.targets[validation_indexes], dtype=torch.float32, device=device),
                ).item()
            )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("DCNN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss, {"best_epoch": int(best_epoch), "epochs_ran": int(epochs_ran)}


def _predict_model(model, sequences: ConditionSequences, indexes: np.ndarray, scaler: dict[str, Any], device, prefix: int | None = None) -> np.ndarray:
    torch = _torch()
    prefix_lengths = None if prefix is None else np.minimum(sequences.lengths[indexes], int(prefix))
    values, context = _transform(sequences, indexes, scaler, prefix_lengths)
    with torch.no_grad():
        return model(
            torch.as_tensor(values, device=device),
            torch.as_tensor(context, device=device),
        ).detach().cpu().numpy().astype(float)


def _validation_indexes(groups: np.ndarray, train_indexes: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(groups[train_indexes]))
    if len(unique) < 2:
        return train_indexes, np.asarray([], dtype=int)
    validation_group = unique[int(seed) % len(unique)]
    validation = train_indexes[groups[train_indexes] == validation_group]
    core = train_indexes[groups[train_indexes] != validation_group]
    return core, validation


def _condition_baseline(
    sequences: ConditionSequences,
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
) -> np.ndarray:
    fallback = np.mean(sequences.targets[train_indexes], axis=0)
    values: dict[str, np.ndarray] = {}
    for condition in np.unique(sequences.conditions[train_indexes]):
        rows = train_indexes[sequences.conditions[train_indexes] == condition]
        values[str(condition)] = np.mean(sequences.targets[rows], axis=0)
    return np.asarray([values.get(str(sequences.conditions[index]), fallback) for index in test_indexes], dtype=float)


def _history_baseline(sequences: ConditionSequences, test_indexes: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    output = np.full((len(test_indexes), len(TARGETS)), fallback, dtype=float)
    by_participant: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for output_index, sequence_index in enumerate(test_indexes):
        by_participant[str(sequences.participant_ids[sequence_index])].append((output_index, int(sequence_index)))
    for rows in by_participant.values():
        previous = None
        for output_index, sequence_index in sorted(rows, key=lambda item: sequences.presentation_positions[item[1]]):
            if previous is not None:
                output[output_index] = previous
            previous = sequences.targets[sequence_index]
    return output


def _spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    from scipy.stats import spearmanr

    value = float(spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def _ranking_accuracy(sequences: ConditionSequences, prediction: np.ndarray, target_index: int) -> float:
    correct, total = 0, 0
    for participant in np.unique(sequences.participant_ids):
        indexes = np.flatnonzero(sequences.participant_ids == participant)
        for left in range(len(indexes)):
            for right in range(left + 1, len(indexes)):
                difference = sequences.targets[indexes[left], target_index] - sequences.targets[indexes[right], target_index]
                if abs(difference) < 1e-12:
                    continue
                total += 1
                if difference * (prediction[indexes[left]] - prediction[indexes[right]]) > 0:
                    correct += 1
    return float(correct / total) if total else float("nan")


def _risk_metrics(truth: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = score >= threshold
    positives = truth == 1
    true_positive = int(np.sum(predicted & positives))
    false_negative = int(np.sum(~predicted & positives))
    false_positive = int(np.sum(predicted & ~positives))
    recall = true_positive / int(np.sum(positives)) if np.any(positives) else float("nan")
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "false_negatives": false_negative,
        "per_row_threshold_recall": float(recall),
    }


def _scaler_checkpoint(scaler: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    return {
        "feature_columns": columns,
        "feature_mean": np.asarray(scaler["feature_mean"], dtype=float).tolist(),
        "feature_scale": np.asarray(scaler["feature_scale"], dtype=float).tolist(),
        "context_mean": np.asarray(scaler["context_mean"], dtype=float).tolist(),
        "context_scale": np.asarray(scaler["context_scale"], dtype=float).tolist(),
    }


def _checkpoint_from_model(
    model,
    *,
    scaler: dict[str, Any],
    sequences: ConditionSequences,
    architecture: dict[str, Any],
    variant: str,
    metrics: dict[str, Any],
    interval_by_history: dict[str, dict[str, float]],
    config: ProjectConfig,
) -> dict[str, Any]:
    columns = [sequences.feature_columns[index] for index in np.asarray(scaler["feature_indexes"], dtype=int)]
    full_key = str(int(architecture["sequence_length"]))
    return {
        "model_kind": MODEL_KIND,
        "schema_version": config.data["schema_version"],
        "model_variant": variant,
        "targets": list(TARGETS),
        "context_columns": list(CONTEXT_COLUMNS),
        "architecture": {key: list(value) if isinstance(value, tuple) else value for key, value in architecture.items()},
        **_scaler_checkpoint(scaler, columns),
        "interval_half_width": interval_by_history[full_key],
        "interval_half_width_by_history": interval_by_history,
        "metrics": metrics,
        "deployable": bool(metrics["deployable"]),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def _checkpoint_scaler(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_mean": np.asarray(bundle["feature_mean"], dtype=float),
        "feature_scale": np.asarray(bundle["feature_scale"], dtype=float),
        "context_mean": np.asarray(bundle["context_mean"], dtype=float),
        "context_scale": np.asarray(bundle["context_scale"], dtype=float),
    }


def write_dcnn_comparison_report(config: ProjectConfig, reports: dict[str, Any]) -> Path:
    """Write a truthful DCNN/classical comparison when both evaluations exist."""
    classical_path = config.path("reports") / "condition_level_lopo_metrics.json"
    lines = [
        "# DCNN Condition 模型 LOPO 报告",
        "",
        "DCNN 仅在通过相同 LOPO 安全门后才可在配置中启用；默认后端保持 classical。",
        "",
        "| 变体 | Relaxation MAE | Discomfort MAE | 部署门 |",
        "|---|---:|---:|---:|",
    ]
    for variant, report in reports.items():
        metrics = report["metrics"]
        lines.append(
            f"| {variant} | {metrics['targets']['relaxation']['mae']:.4f} | "
            f"{metrics['targets']['discomfort']['mae']:.4f} | {'通过' if metrics['deployable'] else '未通过'} |"
        )
    lines.extend(["", "## 经典模型对照", ""])
    if classical_path.exists():
        classical = json.loads(classical_path.read_text(encoding="utf-8"))["metrics"]
        lines.extend(
            [
                f"- Relaxation MAE：{classical['targets']['relaxation']['mae']:.4f}",
                f"- Discomfort MAE：{classical['targets']['discomfort']['mae']:.4f}",
                f"- 经典模型部署门：{'通过' if classical['deployable'] else '未通过'}",
            ]
        )
    else:
        lines.append("- 经典模型的嵌套 LOPO 搜索尚未完成，当前没有可比较产物；不得自动切换到 DCNN。")
    output = config.path("reports") / "dcnn_condition_comparison_zh.md"
    atomic_write_text(output, "\n".join(lines) + "\n")
    return output


def train_dcnn_state(config: ProjectConfig) -> dict[str, Any]:
    """Train and evaluate the DCNN without replacing the selected classical backend."""
    aligned = bool(config.get("alignment.enabled", False))
    contract_payload: dict[str, Any] | None = None
    expected_labels = 135
    expected_train_count = 13
    if aligned:
        contract_payload = validate_alignment_contract(Path(str(config.get("alignment.contract_dir"))))
        expected_labels = int(
            contract_payload.get("label_count", contract_payload.get("observation_count", 135))
        )
        expected_train_count = int(
            contract_payload.get("fold_contract", {}).get("train_participants", 13)
        )
        source = Path(str(config.get("alignment.window_features")))
    else:
        source = config.path("features") / "window_features.csv"
    if not source.exists():
        raise FileNotFoundError("Run 'rtml extract-features' before DCNN training")
    device = _device(config.get("modeling.dcnn.device", "cuda"))
    if aligned and bool(config.get("alignment.require_cuda", True)) and device.type != "cuda":
        raise RuntimeError("Aligned DCNN experiments require an explicit CUDA device")
    runtime = _runtime_provenance(device)
    from sklearn.model_selection import LeaveOneGroupOut

    reports: dict[str, Any] = {}
    all_prediction_rows: list[dict[str, Any]] = []
    configured_variants = tuple(config.get("modeling.dcnn.variants", VARIANTS))
    unknown_variants = sorted(set(configured_variants) - set(VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown DCNN variants: {unknown_variants}")
    aligned_folds = []
    split_protocol = (
        f"shared_{expected_train_count}_train_1_validation_1_test"
        if aligned
        else "project_a_native_lopo"
    )
    for variant in configured_variants:
        sequences = build_condition_sequences(source, config, variant)
        if len(sequences.targets) != expected_labels or len(
            set(zip(sequences.participant_ids, sequences.conditions, strict=True))
        ) != expected_labels:
            raise ValueError(
                f"DCNN expects exactly {expected_labels} unique participant/Condition labels; "
                f"found {len(sequences.targets)}"
            )
        groups = sequences.participant_ids
        modalities = _variant_modalities(variant)
        if aligned:
            aligned_folds = load_split_manifest(
                Path(str(config.get("alignment.split_manifest"))),
                expected_participants=sorted(set(groups)),
                expected_train_count=expected_train_count,
            )
            split_iterator = [
                (fold.fold_index, fold, *indexes_for_fold(groups, fold))
                for fold in aligned_folds
            ]
        else:
            outer = LeaveOneGroupOut()
            split_iterator = [
                (fold, None, train_indexes, np.asarray([], dtype=int), test_indexes)
                for fold, (train_indexes, test_indexes) in enumerate(
                    outer.split(sequences.values, groups=groups), start=1
                )
            ]
        prediction = np.full_like(sequences.targets, np.nan, dtype=float)
        condition_baseline = np.full_like(sequences.targets, np.nan, dtype=float)
        history_baseline = np.full_like(sequences.targets, np.nan, dtype=float)
        row_fold = np.full(len(sequences.targets), -1, dtype=int)
        row_validation = np.full(len(sequences.targets), "", dtype=object)
        prefix_prediction = {
            length: np.full_like(sequences.targets, np.nan, dtype=float)
            for length in range(1, int(_architecture(config)["sequence_length"]) + 1)
        }
        folds: list[dict[str, Any]] = []
        architecture = _architecture(config)
        min_fraction = float(config.get("modeling.dcnn.min_non_missing_fraction", 0.4))
        seed = int(config.get("modeling.random_seed"))
        for fold, aligned_fold, train_indexes, manifest_validation_indexes, test_indexes in split_iterator:
            if aligned:
                core_indexes = train_indexes
                validation_indexes = manifest_validation_indexes
            else:
                core_indexes, validation_indexes = _validation_indexes(
                    groups, train_indexes, seed + fold
                )
            scaler = _fit_scaler(sequences, core_indexes, min_fraction)
            model, validation_loss, training_record = _train_model(
                sequences, core_indexes, validation_indexes, scaler, architecture, config, seed + fold, device
            )
            prediction[test_indexes] = _predict_model(model, sequences, test_indexes, scaler, device)
            for length, output in prefix_prediction.items():
                output[test_indexes] = _predict_model(model, sequences, test_indexes, scaler, device, prefix=length)
            condition_baseline[test_indexes] = _condition_baseline(sequences, core_indexes, test_indexes)
            history_baseline[test_indexes] = _history_baseline(
                sequences, test_indexes, np.mean(sequences.targets[core_indexes], axis=0)
            )
            validation_participant = str(groups[validation_indexes][0]) if len(validation_indexes) else None
            row_fold[test_indexes] = int(fold)
            row_validation[test_indexes] = validation_participant or ""
            folds.append(
                {
                    "fold": fold,
                    "test_participant": str(groups[test_indexes][0]),
                    "validation_participant": validation_participant,
                    "train_participants": sorted(set(groups[core_indexes])),
                    "n_train_participants": int(len(set(groups[core_indexes]))),
                    "n_validation_participants": int(len(set(groups[validation_indexes]))),
                    "n_test_participants": int(len(set(groups[test_indexes]))),
                    "feature_count": int(len(scaler["feature_indexes"])),
                    "validation_loss": float(validation_loss),
                    "training_seed": int(seed + fold),
                    **training_record,
                    **runtime,
                }
            )
            print(
                f"DCNN {variant} LOPO fold {fold}/{len(split_iterator)}: "
                f"{groups[test_indexes][0]}",
                flush=True,
            )
        high_risk = (sequences.targets[:, 1] >= float(config.get("modeling.condition_level.high_discomfort_label_threshold"))).astype(int)
        threshold = float(config.get("modeling.condition_level.high_discomfort_label_threshold"))
        metrics: dict[str, Any] = {
            "unit_of_analysis": "participant_condition",
            "n_labels": int(len(sequences.targets)),
            "seed": seed,
            "model_variant": variant,
            "modalities": list(modalities),
            "split_protocol": split_protocol,
            "runtime": runtime,
            "targets": {},
        }
        for target_index, target in enumerate(TARGETS):
            truth = sequences.targets[:, target_index]
            metrics["targets"][target] = {
                "mae": float(np.mean(np.abs(truth - prediction[:, target_index]))),
                "spearman": _spearman(truth, prediction[:, target_index]),
                "ranking_accuracy": _ranking_accuracy(sequences, prediction[:, target_index], target_index),
                "condition_only_baseline_mae": float(np.mean(np.abs(truth - condition_baseline[:, target_index]))),
                "history_baseline_mae": float(np.mean(np.abs(truth - history_baseline[:, target_index]))),
            }
        discomfort = metrics["targets"]["discomfort"]
        discomfort["risk_at_fold_tuned_threshold"] = _risk_metrics(high_risk, prediction[:, 1], threshold)
        discomfort["threshold_sweep"] = [
            _risk_metrics(high_risk, prediction[:, 1], float(value))
            for value in config.get("modeling.condition_level.risk_probability_thresholds")
        ]
        deployable, reasons = deployment_guard(metrics)
        metrics["deployable"] = deployable
        metrics["deployment_block_reasons"] = reasons
        interval_by_history: dict[str, dict[str, float]] = {}
        prefix_metrics: dict[str, Any] = {}
        for length, output in prefix_prediction.items():
            available = sequences.lengths >= length
            absolute_error = np.abs(sequences.targets[available] - output[available])
            half_width = np.quantile(absolute_error, 0.90, axis=0) if len(absolute_error) else np.asarray([0.5, 0.5])
            interval_by_history[str(length)] = {
                target: float(min(0.5, max(0.05, half_width[target_index])))
                for target_index, target in enumerate(TARGETS)
            }
            prefix_metrics[str(length)] = {
                "n_conditions": int(np.sum(available)),
                "mae": {
                    target: float(np.mean(absolute_error[:, target_index])) if len(absolute_error) else float("nan")
                    for target_index, target in enumerate(TARGETS)
                },
            }
        if aligned:
            final_fold = aligned_folds[0]
            final_train, final_validation, final_excluded_test = indexes_for_fold(groups, final_fold)
            final_scope = {
                "kind": "fold_1_research_checkpoint",
                "train_participants": list(final_fold.train_participants),
                "validation_participant": final_fold.validation_participant,
                "excluded_test_participant": final_fold.test_participant,
                "excluded_test_rows": int(len(final_excluded_test)),
            }
        else:
            final_train, final_validation = _validation_indexes(groups, np.arange(len(groups)), seed)
            final_scope = {"kind": "native_all_data_with_internal_validation"}
        final_scaler = _fit_scaler(sequences, final_train, min_fraction)
        final_model, _, final_training_record = _train_model(
            sequences, final_train, final_validation, final_scaler, architecture, config, seed, device
        )
        checkpoint = _checkpoint_from_model(
            final_model,
            scaler=final_scaler,
            sequences=sequences,
            architecture=architecture,
            variant=variant,
            metrics=metrics,
            interval_by_history=interval_by_history,
            config=config,
        )
        checkpoint["runtime"] = runtime
        checkpoint["seed"] = seed
        checkpoint["training_scope"] = final_scope
        checkpoint["training_record"] = final_training_record
        torch = _torch()
        checkpoint_path = config.path("models") / f"dcnn_state_{variant}.pt"
        torch.save(checkpoint, checkpoint_path)
        reports[variant] = {
            "model_path": str(checkpoint_path),
            "metrics": metrics,
            "prefix_metrics": prefix_metrics,
            "folds": folds,
            "feature_count_final": int(len(final_scaler["feature_indexes"])),
            "runtime": runtime,
            "seed": seed,
            "training_scope": final_scope,
        }
        for index in range(len(sequences.targets)):
            all_prediction_rows.append(
                {
                    "model_variant": variant,
                    "participant_id": sequences.participant_ids[index],
                    "condition": sequences.conditions[index],
                    "presentation_position": float(sequences.presentation_positions[index]),
                    "relaxation": float(sequences.targets[index, 0]),
                    "discomfort": float(sequences.targets[index, 1]),
                    "pred_relaxation": float(prediction[index, 0]),
                    "pred_discomfort": float(prediction[index, 1]),
                    "condition_only_relaxation": float(condition_baseline[index, 0]),
                    "condition_only_discomfort": float(condition_baseline[index, 1]),
                    "history_relaxation": float(history_baseline[index, 0]),
                    "history_discomfort": float(history_baseline[index, 1]),
                    "seed": seed,
                    "fold_index": int(row_fold[index]),
                    "test_participant": str(sequences.participant_ids[index]),
                    "validation_participant": str(row_validation[index]),
                    "modalities": "+".join(modalities),
                    "split_protocol": split_protocol,
                    "device": str(device),
                    "cuda_used": bool(device.type == "cuda"),
                }
            )
    report_path = config.path("reports") / "dcnn_condition_lopo_metrics.json"
    report_payload = {
        "schema_version": config.data["schema_version"],
        "seed": int(config.get("modeling.random_seed")),
        "split_protocol": split_protocol,
        "runtime": runtime,
        "variants": reports,
    }
    if aligned:
        report_payload["alignment"] = {
            "contract_dir": str(Path(str(config.get("alignment.contract_dir"))).resolve()),
            "split_manifest": str(Path(str(config.get("alignment.split_manifest"))).resolve()),
            "modality_masks": str(Path(str(config.get("alignment.modality_masks"))).resolve()),
            "labels": str(Path(str(config.get("alignment.labels"))).resolve()),
            "windows": str(Path(str(config.get("alignment.windows"))).resolve()),
        }
        write_alignment_manifest(
            config.path("manifests") / "dcnn_alignment_manifest.json",
            split_manifest=config.get("alignment.split_manifest"),
            modality_masks=config.get("alignment.modality_masks"),
            labels=config.get("alignment.labels"),
            windows=config.get("alignment.windows"),
            window_features=source,
            contract=Path(str(config.get("alignment.contract_dir"))) / "contract.json",
            seed=int(config.get("modeling.random_seed")),
            folds=aligned_folds,
            runtime={
                **runtime,
                "model": "dcnn",
                "variants": list(configured_variants),
                "modalities": {
                    variant: list(_variant_modalities(variant))
                    for variant in configured_variants
                },
            },
        )
    write_json(report_path, report_payload)
    import pandas as pd

    prediction_dir = config.path("predictions") if not config.is_legacy else config.path("reports")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_prediction_rows).to_csv(prediction_dir / "dcnn_condition_lopo_predictions.csv", index=False)
    write_dcnn_comparison_report(config, reports)
    return {"metrics_path": str(report_path), "variants": reports}


def load_dcnn_state_model(path: Path, device_name: str) -> dict[str, Any]:
    torch = _torch()
    device = _device(device_name)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_kind") != MODEL_KIND:
        raise ValueError(f"Not a {MODEL_KIND} checkpoint: {path}")
    architecture = dict(checkpoint["architecture"])
    for key in ("conv_channels", "kernel_sizes", "pool_sizes"):
        architecture[key] = tuple(architecture[key])
    model = _make_model(len(checkpoint["feature_columns"]), architecture).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    checkpoint["model"] = model
    checkpoint["device"] = device
    return checkpoint


def predict_dcnn_state(
    bundle: dict[str, Any],
    history: Iterable[dict[str, Any]],
    condition: str,
    config: ProjectConfig,
) -> tuple[dict[str, float], dict[str, float], int]:
    """Predict from the causal current-Condition feature history."""
    torch = _torch()
    records = list(history)
    sequence_length = int(bundle["architecture"]["sequence_length"])
    records = records[-sequence_length:]
    columns = list(bundle["feature_columns"])
    values = np.full((1, len(columns), sequence_length), np.nan, dtype=float)
    for time_index, record in enumerate(records):
        for feature_index, column in enumerate(columns):
            try:
                values[0, feature_index, time_index] = float(record.get(column, np.nan))
            except (TypeError, ValueError):
                continue
    scaler = _checkpoint_scaler(bundle)
    standardized = (values - scaler["feature_mean"][None, :, None]) / scaler["feature_scale"][None, :, None]
    standardized = np.where(np.isfinite(standardized), standardized, 0.0).astype(np.float32)
    parameters = condition_parameters(
        condition,
        list(config.get("conditions.intensities")),
        list(config.get("conditions.frequencies")),
    )
    context = np.asarray([[float(parameters[name]) for name in bundle["context_columns"]]], dtype=float)
    context = (context - scaler["context_mean"]) / scaler["context_scale"]
    context = np.where(np.isfinite(context), context, 0.0).astype(np.float32)
    with torch.no_grad():
        output = bundle["model"](
            torch.as_tensor(standardized, device=bundle["device"]),
            torch.as_tensor(context, device=bundle["device"]),
        ).detach().cpu().numpy()[0]
    history_windows = len(records)
    half_widths = bundle.get("interval_half_width_by_history", {}).get(
        str(history_windows), bundle.get("interval_half_width", {})
    )
    return (
        {target: float(np.clip(output[index], 0.0, 1.0)) for index, target in enumerate(bundle["targets"])},
        {target: float(half_widths.get(target, 0.5)) for target in bundle["targets"]},
        history_windows,
    )
