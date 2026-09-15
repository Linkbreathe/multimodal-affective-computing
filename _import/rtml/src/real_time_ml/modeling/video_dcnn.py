"""VideoMAE2 + physiological 1D-CNN fusion for recorded egocentric replay.

This is deliberately separate from :mod:`real_time_ml.modeling.dcnn`: the
regular DCNN remains deployable only with real-time modalities.  VideoMAE2 is
an offline/replay experiment and carries its own model kind and artifact tree.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.data.io import condition_parameters
from real_time_ml.modeling.safety import deployment_guard
from real_time_ml.utils import atomic_write_text, file_sha256, write_json


VIDEO_MODEL_KIND = "videomae2_fusion_dcnn_condition_regressor_v1"
RELAXATION_VIDEO_MODEL_KIND = "videomae2_fusion_dcnn_relaxation_only_v1"
VIDEO_ENCODER_ABLATION_DUAL_KIND = "videomae2_video_encoder_ablation_dual_v1"
VIDEO_ENCODER_ABLATION_RELAXATION_KIND = "videomae2_video_encoder_ablation_relaxation_v1"
TARGETS = ("relaxation", "discomfort")
CONTEXT_COLUMNS = ("intensity", "frequency")
BASE_PREFIXES = ("eeg_", "ecg_", "head_", "eye_")
VIDEO_ENCODER_NONE = "none"
VIDEO_ENCODER_MASKED_MEAN_MLP = "masked_mean_mlp"
VIDEO_ENCODER_TEMPORAL_DCNN = "temporal_1dcnn"
VIDEO_ENCODER_MODES = {
    VIDEO_ENCODER_NONE,
    VIDEO_ENCODER_MASKED_MEAN_MLP,
    VIDEO_ENCODER_TEMPORAL_DCNN,
}


@dataclass(frozen=True)
class VideoSequences:
    base_values: np.ndarray  # [condition, base feature, 10-second window]
    embeddings: np.ndarray  # [condition, window, 384]
    video_mask: np.ndarray  # [condition, window]
    context: np.ndarray
    targets: np.ndarray
    participant_ids: np.ndarray
    conditions: np.ndarray
    presentation_positions: np.ndarray
    lengths: np.ndarray
    base_columns: tuple[str, ...]
    embedding_columns: tuple[str, ...]
    target_names: tuple[str, ...]


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("VideoMAE2 DCNN requires PyTorch") from error
    return torch


def _device(value: str):
    torch = _torch()
    requested = str(value or "cuda")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("VideoMAE2 DCNN is configured for CUDA but torch.cuda.is_available() is false")
    return torch.device(requested if requested.startswith("cuda") else "cpu")


def _architecture(config: ProjectConfig) -> dict[str, Any]:
    node = dict(config.get("modeling.dcnn", {}))
    architecture = {
        "sequence_length": int(node.get("sequence_length", 8)),
        "conv_channels": tuple(int(item) for item in node.get("conv_channels", [16, 32])),
        "kernel_sizes": tuple(int(item) for item in node.get("kernel_sizes", [3, 3])),
        "pool_sizes": tuple(int(item) for item in node.get("pool_sizes", [2, 2])),
        "mlp_hidden": int(node.get("mlp_hidden", 64)),
        "dropout": float(node.get("dropout", 0.3)),
    }
    if architecture["sequence_length"] < int(np.prod(architecture["pool_sizes"])):
        raise ValueError("DCNN sequence length is too short for its pooling stack")
    return architecture


def _video_encoder_mode(include_video: bool, value: str | None) -> str:
    mode = str(value) if value is not None else (
        VIDEO_ENCODER_TEMPORAL_DCNN if include_video else VIDEO_ENCODER_NONE
    )
    if mode not in VIDEO_ENCODER_MODES:
        raise ValueError(f"Unknown VideoMAE2 video encoder mode: {mode}")
    if include_video and mode == VIDEO_ENCODER_NONE:
        raise ValueError("A visual VideoMAE2 model requires a video encoder mode")
    if not include_video and mode != VIDEO_ENCODER_NONE:
        raise ValueError("A no-video model must use video_encoder_mode='none'")
    return mode


def build_video_sequences(
    window_source: Path,
    embedding_source: Path,
    config: ProjectConfig,
    targets: tuple[str, ...] = TARGETS,
) -> VideoSequences:
    """Join 10-second tabular windows to their keyed frozen visual embeddings."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("VideoMAE2 DCNN requires pandas") from error
    windows = pd.read_csv(window_source)
    embeddings = pd.read_csv(embedding_source)
    keys = ["participant_id", "condition", "condition_window_index"]
    if not targets:
        raise ValueError("VideoMAE2 DCNN requires at least one target")
    required = {*keys, *targets, *CONTEXT_COLUMNS}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"Window feature source is missing {sorted(missing)}")
    embedding_columns = sorted(name for name in embeddings.columns if name.startswith("video_embedding_"))
    if not embedding_columns:
        raise ValueError("VideoMAE2 embedding source has no video_embedding_* columns")
    right = embeddings[[*keys, "video_available", *embedding_columns]].copy()
    frame = windows.merge(right, how="left", on=keys, validate="one_to_one")
    base_columns = tuple(name for name in frame.columns if name.startswith(BASE_PREFIXES))
    if not base_columns:
        raise ValueError("No physiological/head/eye features available for VideoMAE2 fusion")
    sequence_length = int(_architecture(config)["sequence_length"])
    values: list[np.ndarray] = []
    visual: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    targets_values: list[np.ndarray] = []
    participants: list[str] = []
    conditions: list[str] = []
    positions: list[float] = []
    lengths: list[int] = []
    for (participant, condition), group in frame.groupby(["participant_id", "condition"], sort=True):
        ordered = group.copy()
        ordered["condition_window_index"] = pd.to_numeric(ordered["condition_window_index"], errors="coerce")
        ordered = ordered.dropna(subset=["condition_window_index"]).sort_values("condition_window_index")
        indexes = ordered["condition_window_index"].astype(int).to_numpy()
        if not len(indexes) or indexes.min() < 0 or indexes.max() >= sequence_length or len(np.unique(indexes)) != len(indexes):
            raise ValueError(f"Invalid window indexes for {participant}/{condition}")
        if ordered[list(targets)].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Inconsistent inherited labels for {participant}/{condition}")
        base = np.full((len(base_columns), sequence_length), np.nan, dtype=float)
        base[:, indexes] = ordered[list(base_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T
        embedding = np.full((sequence_length, len(embedding_columns)), np.nan, dtype=float)
        embedding[indexes, :] = ordered[embedding_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        mask = np.zeros(sequence_length, dtype=float)
        available = pd.to_numeric(ordered.get("video_available"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mask[indexes] = (available >= 0.5).astype(float)
        values.append(base)
        visual.append(embedding)
        masks.append(mask)
        contexts.append(ordered.iloc[0][list(CONTEXT_COLUMNS)].to_numpy(dtype=float))
        targets_values.append(ordered.iloc[0][list(targets)].to_numpy(dtype=float))
        participants.append(str(participant))
        conditions.append(str(condition))
        positions.append(float(pd.to_numeric(ordered.iloc[0].get("presentation_position"), errors="coerce")))
        lengths.append(int(indexes.max() + 1))
    if not values:
        raise ValueError("No VideoMAE2 Condition sequences could be constructed")
    return VideoSequences(
        base_values=np.stack(values), embeddings=np.stack(visual), video_mask=np.stack(masks),
        context=np.stack(contexts), targets=np.stack(targets_values), participant_ids=np.asarray(participants, dtype=str),
        conditions=np.asarray(conditions, dtype=str), presentation_positions=np.asarray(positions, dtype=float),
        lengths=np.asarray(lengths, dtype=int), base_columns=base_columns, embedding_columns=tuple(embedding_columns),
        target_names=tuple(targets),
    )


def _fit_scaler(sequences: VideoSequences, train_indexes: np.ndarray, config: ProjectConfig, include_video: bool) -> dict[str, Any]:
    minimum = float(config.get("modeling.dcnn.min_non_missing_fraction", 0.4))
    train_base = sequences.base_values[train_indexes]
    availability = np.mean(np.isfinite(train_base), axis=(0, 2))
    selected = np.flatnonzero(availability >= minimum)
    if not len(selected):
        raise ValueError("No base features meet the VideoMAE2 DCNN availability threshold")
    selected_values = train_base[:, selected, :]
    mean = np.nanmean(selected_values, axis=(0, 2))
    scale = np.nanstd(selected_values, axis=(0, 2))
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    context = sequences.context[train_indexes]
    context_mean = np.nanmean(context, axis=0)
    context_scale = np.nanstd(context, axis=0)
    context_mean = np.where(np.isfinite(context_mean), context_mean, 0.0)
    context_scale = np.where(np.isfinite(context_scale) & (context_scale > 1e-8), context_scale, 1.0)
    scaler: dict[str, Any] = {
        "base_indexes": selected.astype(int), "base_mean": mean.astype(float), "base_scale": scale.astype(float),
        "context_mean": context_mean.astype(float), "context_scale": context_scale.astype(float),
        "video_components": 0,
    }
    if not include_video:
        return scaler
    requested = int(config.get("features.video.videomae2.pca_components", config.get("modeling.dcnn.video_pca_components", 32)))
    raw = sequences.embeddings[train_indexes].reshape(-1, sequences.embeddings.shape[-1])
    mask = sequences.video_mask[train_indexes].reshape(-1) >= 0.5
    valid = mask & np.all(np.isfinite(raw), axis=1)
    if int(np.sum(valid)) < requested:
        raise ValueError(f"VideoMAE2 PCA requires {requested} valid training windows; found {int(np.sum(valid))}")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=requested, svd_solver="full", random_state=int(config.get("modeling.random_seed")))
    transformed = pca.fit_transform(raw[valid])
    pca_mean = np.mean(transformed, axis=0)
    pca_scale = np.std(transformed, axis=0)
    pca_scale = np.where(pca_scale > 1e-8, pca_scale, 1.0)
    scaler.update({
        "video_components": requested,
        "video_pca_mean": pca.mean_.astype(float),
        "video_pca_components": pca.components_.astype(float),
        "video_feature_mean": pca_mean.astype(float),
        "video_feature_scale": pca_scale.astype(float),
    })
    return scaler


def _transform(
    sequences: VideoSequences,
    indexes: np.ndarray,
    scaler: dict[str, Any],
    prefix_lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(scaler["base_indexes"], dtype=int)
    base = sequences.base_values[indexes][:, selected, :].copy()
    if prefix_lengths is not None:
        for row, prefix in enumerate(np.asarray(prefix_lengths, dtype=int)):
            base[row, :, max(0, int(prefix)) :] = np.nan
    base = (base - np.asarray(scaler["base_mean"], dtype=float)[None, :, None]) / np.asarray(scaler["base_scale"], dtype=float)[None, :, None]
    base = np.where(np.isfinite(base), base, 0.0).astype(np.float32)
    context = sequences.context[indexes].copy()
    context = (context - np.asarray(scaler["context_mean"], dtype=float)) / np.asarray(scaler["context_scale"], dtype=float)
    context = np.where(np.isfinite(context), context, 0.0).astype(np.float32)
    components = int(scaler.get("video_components", 0))
    if not components:
        return base, np.zeros((len(indexes), 0, base.shape[-1]), dtype=np.float32), context
    raw = sequences.embeddings[indexes].copy()
    mask = sequences.video_mask[indexes].copy()
    if prefix_lengths is not None:
        for row, prefix in enumerate(np.asarray(prefix_lengths, dtype=int)):
            raw[row, max(0, int(prefix)) :, :] = np.nan
            mask[row, max(0, int(prefix)) :] = 0.0
    valid = (mask >= 0.5) & np.all(np.isfinite(raw), axis=2)
    flat = raw.reshape(-1, raw.shape[-1])
    flat_valid = valid.reshape(-1)
    filled = np.broadcast_to(np.asarray(scaler["video_pca_mean"], dtype=float), flat.shape).copy()
    filled[flat_valid] = flat[flat_valid]
    pca = (filled - np.asarray(scaler["video_pca_mean"], dtype=float)) @ np.asarray(scaler["video_pca_components"], dtype=float).T
    pca = (pca - np.asarray(scaler["video_feature_mean"], dtype=float)) / np.asarray(scaler["video_feature_scale"], dtype=float)
    pca[~flat_valid] = 0.0
    video = pca.reshape(raw.shape[0], raw.shape[1], components)
    # The final channel is essential: zero is a missing-video sentinel, not a visual embedding.
    video = np.concatenate([video, valid[..., None].astype(float)], axis=2).transpose(0, 2, 1).astype(np.float32)
    return base, video, context


@lru_cache(maxsize=1)
def _model_type():
    torch = _torch()
    nn = torch.nn

    class Stream(nn.Module):
        def __init__(self, input_channels: int, channels, kernels, pools, dropout) -> None:
            super().__init__()
            layers: list[Any] = []
            current = input_channels
            for out, kernel, pool in zip(channels, kernels, pools, strict=True):
                layers.extend((nn.Conv1d(current, out, kernel, padding=kernel // 2), nn.ReLU(), nn.MaxPool1d(pool), nn.Dropout(dropout)))
                current = out
            self.network = nn.Sequential(*layers)

        def forward(self, value):
            return self.network(value).flatten(1)

    class FusionDCNN(nn.Module):
        def __init__(
            self,
            n_base: int,
            n_video: int,
            architecture: dict[str, Any],
            n_targets: int,
            video_encoder_mode: str,
        ) -> None:
            super().__init__()
            length = int(architecture["sequence_length"])
            for pool in architecture["pool_sizes"]:
                length //= int(pool)
            self.base_streams = nn.ModuleList(Stream(1, architecture["conv_channels"], architecture["kernel_sizes"], architecture["pool_sizes"], architecture["dropout"]) for _ in range(n_base))
            self.video_encoder_mode = video_encoder_mode
            self.video_stream = (
                Stream(n_video, architecture["conv_channels"], architecture["kernel_sizes"], architecture["pool_sizes"], architecture["dropout"])
                if n_video and video_encoder_mode == VIDEO_ENCODER_TEMPORAL_DCNN
                else None
            )
            flattened = n_base * int(architecture["conv_channels"][-1]) * length
            if n_video and video_encoder_mode == VIDEO_ENCODER_TEMPORAL_DCNN:
                flattened += int(architecture["conv_channels"][-1]) * length
            elif n_video and video_encoder_mode == VIDEO_ENCODER_MASKED_MEAN_MLP:
                flattened += n_video
            self.fc1 = nn.Linear(flattened + len(CONTEXT_COLUMNS), int(architecture["mlp_hidden"]))
            self.dropout = nn.Dropout(float(architecture["dropout"]))
            self.output = nn.Linear(int(architecture["mlp_hidden"]), n_targets)

        def forward(self, base, video, context):
            chunks = [stream(base[:, index : index + 1, :]) for index, stream in enumerate(self.base_streams)]
            if self.video_stream is not None:
                chunks.append(self.video_stream(video))
            elif video.shape[1] and self.video_encoder_mode == VIDEO_ENCODER_MASKED_MEAN_MLP:
                video_values, valid = video[:, :-1, :], video[:, -1:, :]
                denominator = valid.sum(dim=2).clamp_min(1.0)
                pooled = (video_values * valid).sum(dim=2) / denominator
                chunks.append(torch.cat([pooled, valid.mean(dim=2)], dim=1))
            merged = torch.cat([*chunks, context], dim=1)
            return torch.sigmoid(self.output(self.dropout(torch.relu(self.fc1(merged)))))

    return FusionDCNN


def _make_model(
    n_base: int,
    n_video: int,
    architecture: dict[str, Any],
    n_targets: int = len(TARGETS),
    video_encoder_mode: str = VIDEO_ENCODER_TEMPORAL_DCNN,
):
    return _model_type()(n_base, n_video, architecture, n_targets, video_encoder_mode)


def _validation_indexes(groups: np.ndarray, train_indexes: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(groups[train_indexes]))
    if len(unique) < 2:
        return train_indexes, np.asarray([], dtype=int)
    selected = unique[int(seed) % len(unique)]
    validation = train_indexes[groups[train_indexes] == selected]
    return train_indexes[groups[train_indexes] != selected], validation


def _train_model(
    sequences,
    train_indexes,
    validation_indexes,
    scaler,
    architecture,
    config,
    seed,
    device,
    video_encoder_mode: str,
):
    torch = _torch()
    node = dict(config.get("modeling.dcnn", {}))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = _make_model(
        len(scaler["base_indexes"]),
        int(scaler.get("video_components", 0)) + int(bool(scaler.get("video_components", 0))),
        architecture,
        sequences.targets.shape[1],
        video_encoder_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(node.get("learning_rate", 1e-4)), weight_decay=float(node.get("weight_decay", 5e-5)))
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(int(seed))
    validation_indexes = validation_indexes if len(validation_indexes) else train_indexes
    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(int(node.get("max_epochs", 100))):
        prefixes = np.asarray([rng.integers(1, int(sequences.lengths[index]) + 1) for index in train_indexes], dtype=int)
        base, video, context = _transform(sequences, train_indexes, scaler, prefixes)
        order = rng.permutation(len(train_indexes))
        model.train()
        for start in range(0, len(order), int(node.get("batch_size", 16))):
            batch = order[start : start + int(node.get("batch_size", 16))]
            optimizer.zero_grad()
            predicted = model(torch.as_tensor(base[batch], device=device), torch.as_tensor(video[batch], device=device), torch.as_tensor(context[batch], device=device))
            target = torch.as_tensor(sequences.targets[train_indexes][batch], dtype=torch.float32, device=device)
            loss = loss_fn(predicted, target)
            loss.backward()
            optimizer.step()
        val_base, val_video, val_context = _transform(sequences, validation_indexes, scaler)
        with torch.no_grad():
            model.eval()
            loss = loss_fn(model(torch.as_tensor(val_base, device=device), torch.as_tensor(val_video, device=device), torch.as_tensor(val_context, device=device)), torch.as_tensor(sequences.targets[validation_indexes], dtype=torch.float32, device=device))
            value = float(loss.item())
        if value < best_loss - 1e-8:
            best_loss = value
            best_state = deepcopy({name: item.detach().cpu() for name, item in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= int(node.get("early_stopping_patience", 15)):
                break
    if best_state is None:
        raise RuntimeError("VideoMAE2 DCNN did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


def _predict(model, sequences, indexes, scaler, device, prefix: int | None = None) -> np.ndarray:
    torch = _torch()
    lengths = None if prefix is None else np.minimum(sequences.lengths[indexes], int(prefix))
    base, video, context = _transform(sequences, indexes, scaler, lengths)
    with torch.no_grad():
        return model(torch.as_tensor(base, device=device), torch.as_tensor(video, device=device), torch.as_tensor(context, device=device)).detach().cpu().numpy().astype(float)


def _condition_baseline(sequences, train_indexes, test_indexes) -> np.ndarray:
    fallback = np.mean(sequences.targets[train_indexes], axis=0)
    mappings = {condition: np.mean(sequences.targets[train_indexes[sequences.conditions[train_indexes] == condition]], axis=0) for condition in np.unique(sequences.conditions[train_indexes])}
    return np.asarray([mappings.get(condition, fallback) for condition in sequences.conditions[test_indexes]], dtype=float)


def _history_baseline(sequences, test_indexes, fallback) -> np.ndarray:
    output = np.full((len(test_indexes), sequences.targets.shape[1]), fallback, dtype=float)
    rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for target, sequence in enumerate(test_indexes):
        rows[str(sequences.participant_ids[sequence])].append((target, int(sequence)))
    for entries in rows.values():
        prior = None
        for output_index, sequence_index in sorted(entries, key=lambda item: sequences.presentation_positions[item[1]]):
            if prior is not None:
                output[output_index] = prior
            prior = sequences.targets[sequence_index]
    return output


def _metrics(sequences, prediction, condition_baseline, history_baseline, config, research_only: bool = False) -> dict[str, Any]:
    from scipy.stats import spearmanr

    metrics: dict[str, Any] = {"unit_of_analysis": "participant_condition", "n_labels": int(len(sequences.targets)), "targets": {}}
    for index, target in enumerate(sequences.target_names):
        truth = sequences.targets[:, index]
        correlation = float(spearmanr(truth, prediction[:, index]).statistic)
        metrics["targets"][target] = {
            "mae": float(np.mean(np.abs(truth - prediction[:, index]))),
            "spearman": correlation if np.isfinite(correlation) else 0.0,
            "condition_only_baseline_mae": float(np.mean(np.abs(truth - condition_baseline[:, index]))),
            "history_baseline_mae": float(np.mean(np.abs(truth - history_baseline[:, index]))),
        }
    if sequences.target_names == TARGETS:
        threshold = float(config.get("modeling.condition_level.high_discomfort_label_threshold"))
        high = sequences.targets[:, 1] >= threshold
        predicted = prediction[:, 1] >= threshold
        recall = float(np.mean(predicted[high])) if np.any(high) else float("nan")
        metrics["targets"]["discomfort"]["risk_at_fold_tuned_threshold"] = {
            "threshold": threshold, "recall": recall, "per_row_threshold_recall": recall,
            "precision": float(np.mean(high[predicted])) if np.any(predicted) else 0.0,
            "false_negatives": int(np.sum(high & ~predicted)),
        }
        if research_only:
            metrics["deployable"] = False
            metrics["deployment_block_reasons"] = ["research_only_video_encoder_ablation"]
        else:
            deployable, reasons = deployment_guard(metrics)
            metrics["deployable"], metrics["deployment_block_reasons"] = deployable, reasons
    else:
        metrics["deployable"] = False
        metrics["deployment_block_reasons"] = ["research_only_relaxation_model"]
    return metrics


def _checkpoint(
    model,
    scaler,
    sequences,
    architecture,
    metrics,
    variant,
    intervals,
    config,
    *,
    model_kind: str = VIDEO_MODEL_KIND,
    research_only: bool = False,
    video_encoder_mode: str = VIDEO_ENCODER_TEMPORAL_DCNN,
) -> dict[str, Any]:
    checkpoint = {
        "model_kind": model_kind, "schema_version": config.data["schema_version"], "model_variant": variant,
        "video_encoder_mode": video_encoder_mode,
        "targets": list(sequences.target_names), "context_columns": list(CONTEXT_COLUMNS), "base_feature_columns": [sequences.base_columns[index] for index in scaler["base_indexes"]],
        "embedding_columns": list(sequences.embedding_columns) if scaler.get("video_components", 0) else [],
        "architecture": {key: list(value) if isinstance(value, tuple) else value for key, value in architecture.items()},
        "scaler": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in scaler.items()},
        "interval_half_width": intervals[str(int(architecture["sequence_length"]))], "interval_half_width_by_history": intervals,
        "metrics": metrics, "deployable": bool(metrics["deployable"]),
        "parameter_count": int(sum(item.numel() for item in model.parameters())),
        "state_dict": {name: item.detach().cpu() for name, item in model.state_dict().items()},
    }
    if research_only:
        checkpoint["research_only"] = True
    return checkpoint


def train_video_dcnn_model(
    config: ProjectConfig, *, window_source: Path, embedding_source: Path, model_path: Path, reports_dir: Path,
    include_video: bool, expected_labels: int, targets: tuple[str, ...] = TARGETS,
    model_kind: str = VIDEO_MODEL_KIND, research_only: bool = False,
    video_encoder_mode: str | None = None, model_variant: str | None = None,
) -> dict[str, Any]:
    """Nested LOPO training for a visual model or its matched tabular fallback."""
    started = perf_counter()
    video_encoder_mode = _video_encoder_mode(include_video, video_encoder_mode)
    sequences = build_video_sequences(window_source, embedding_source, config, targets=targets)
    if len(sequences.targets) != expected_labels or len(set(zip(sequences.participant_ids, sequences.conditions, strict=True))) != expected_labels:
        raise ValueError(f"Expected {expected_labels} unique participant/Condition labels; found {len(sequences.targets)}")
    device = _device(config.get("modeling.dcnn.device", "cuda"))
    groups = sequences.participant_ids
    from sklearn.model_selection import LeaveOneGroupOut
    outer = LeaveOneGroupOut()
    architecture = _architecture(config)
    seed = int(config.get("modeling.random_seed"))
    prediction = np.full_like(sequences.targets, np.nan, dtype=float)
    condition_baseline = np.full_like(sequences.targets, np.nan, dtype=float)
    history_baseline = np.full_like(sequences.targets, np.nan, dtype=float)
    prefix_outputs = {length: np.full_like(sequences.targets, np.nan, dtype=float) for length in range(1, int(architecture["sequence_length"]) + 1)}
    folds = []
    for fold, (train_indexes, test_indexes) in enumerate(outer.split(sequences.base_values, groups=groups), start=1):
        fold_started = perf_counter()
        core, validation = _validation_indexes(groups, train_indexes, seed + fold)
        scaler = _fit_scaler(sequences, core, config, include_video)
        model, loss = _train_model(
            sequences, core, validation, scaler, architecture, config, seed + fold, device,
            video_encoder_mode,
        )
        prediction[test_indexes] = _predict(model, sequences, test_indexes, scaler, device)
        condition_baseline[test_indexes] = _condition_baseline(sequences, train_indexes, test_indexes)
        history_baseline[test_indexes] = _history_baseline(sequences, test_indexes, np.mean(sequences.targets[train_indexes], axis=0))
        for length, output in prefix_outputs.items():
            output[test_indexes] = _predict(model, sequences, test_indexes, scaler, device, length)
        folds.append({
            "fold": fold,
            "test_participant": str(groups[test_indexes][0]),
            "validation_loss": float(loss),
            "base_feature_count": int(len(scaler["base_indexes"])),
            "video_pca_components": int(scaler.get("video_components", 0)),
            "training_seconds": float(perf_counter() - fold_started),
        })
        print(
            f"Video DCNN {video_encoder_mode} LOPO fold {fold}: {groups[test_indexes][0]}",
            flush=True,
        )
    metrics = _metrics(sequences, prediction, condition_baseline, history_baseline, config, research_only)
    intervals: dict[str, dict[str, float]] = {}
    prefix_metrics = {}
    for length, output in prefix_outputs.items():
        available = sequences.lengths >= length
        errors = np.abs(sequences.targets[available] - output[available])
        half = np.quantile(errors, 0.90, axis=0) if len(errors) else np.asarray([0.5, 0.5])
        intervals[str(length)] = {target: float(min(0.5, max(0.05, half[index]))) for index, target in enumerate(sequences.target_names)}
        prefix_metrics[str(length)] = {"n_conditions": int(np.sum(available)), "mae": {target: float(np.mean(errors[:, index])) if len(errors) else float("nan") for index, target in enumerate(sequences.target_names)}}
    final_train, final_validation = _validation_indexes(groups, np.arange(len(groups)), seed)
    final_scaler = _fit_scaler(sequences, final_train, config, include_video)
    final_model, _ = _train_model(
        sequences, final_train, final_validation, final_scaler, architecture, config, seed, device,
        video_encoder_mode,
    )
    variant = model_variant or ("full" if include_video else "no_video")
    checkpoint = _checkpoint(
        final_model, final_scaler, sequences, architecture, metrics, variant, intervals, config,
        model_kind=model_kind, research_only=research_only, video_encoder_mode=video_encoder_mode,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _torch().save(checkpoint, model_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    lopo = pd.DataFrame({
        "participant_id": sequences.participant_ids,
        "condition": sequences.conditions,
        "presentation_position": sequences.presentation_positions,
    })
    for index, target in enumerate(sequences.target_names):
        lopo[target] = sequences.targets[:, index]
        lopo[f"pred_{target}"] = prediction[:, index]
        lopo[f"condition_only_{target}"] = condition_baseline[:, index]
        lopo[f"history_{target}"] = history_baseline[:, index]
    predictions_path = reports_dir / f"{variant}_lopo_predictions.csv"
    lopo.to_csv(predictions_path, index=False)
    result = {
        "model_path": str(model_path),
        "lopo_predictions_path": str(predictions_path),
        "metrics": metrics,
        "folds": folds,
        "prefix_metrics": prefix_metrics,
        "include_video": include_video,
        "video_encoder_mode": video_encoder_mode,
        "targets": list(sequences.target_names),
        "research_only": research_only,
        "parameter_count": checkpoint["parameter_count"],
        "training_seconds": float(perf_counter() - started),
    }
    write_json(reports_dir / f"{variant}_metrics.json", result)
    return result


def _complete_video_window_source(window_source: Path, embedding_source: Path, output: Path):
    import pandas as pd

    windows, embeddings = pd.read_csv(window_source), pd.read_csv(embedding_source)
    keys = ["participant_id", "condition", "condition_window_index"]
    availability = embeddings[keys + ["video_available"]]
    merged = windows.merge(availability, on=keys, how="left", validate="one_to_one")
    merged["video_available"] = pd.to_numeric(merged["video_available"], errors="coerce").fillna(0.0)
    complete = merged.groupby(["participant_id", "condition"])["video_available"].apply(lambda value: bool((value >= 0.5).all()))
    selected = set(complete[complete].index.tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    result = windows[windows.apply(lambda row: (row["participant_id"], row["condition"]) in selected, axis=1)].copy()
    result.to_csv(output, index=False)
    return result, int(len(selected))


def train_videomae2_dcnn(config: ProjectConfig) -> dict[str, Any]:
    """Train visual and no-video fusion DCNNs for main and complete-video cohorts."""
    window_source = config.path("features") / "video_ml" / "window_features.csv"
    embedding_source = config.path("video") / "videomae2" / "window_embeddings.csv"
    if not window_source.exists() or not embedding_source.exists():
        raise FileNotFoundError("Run handcrafted video extraction and 'rtml extract-videomae2' first")
    root = config.path("video")
    models = root / "models" / "videomae2_dcnn"
    reports = root / "reports" / "videomae2_dcnn"
    features = root / "features" / "videomae2_dcnn"
    primary_visual = train_video_dcnn_model(config, window_source=window_source, embedding_source=embedding_source, model_path=models / "dcnn_state_full.pt", reports_dir=reports / "visual", include_video=True, expected_labels=135)
    primary_no_video = train_video_dcnn_model(config, window_source=window_source, embedding_source=embedding_source, model_path=models / "dcnn_state_no_video.pt", reports_dir=reports / "no_video", include_video=False, expected_labels=135)
    sensitivity_source = features / "window_features_video_complete.csv"
    _, sensitivity_labels = _complete_video_window_source(window_source, embedding_source, sensitivity_source)
    sensitivity_visual = train_video_dcnn_model(config, window_source=sensitivity_source, embedding_source=embedding_source, model_path=models / "sensitivity" / "dcnn_state_full.pt", reports_dir=reports / "sensitivity" / "visual", include_video=True, expected_labels=sensitivity_labels)
    sensitivity_no_video = train_video_dcnn_model(config, window_source=sensitivity_source, embedding_source=embedding_source, model_path=models / "sensitivity" / "dcnn_state_no_video.pt", reports_dir=reports / "sensitivity" / "no_video", include_video=False, expected_labels=sensitivity_labels)
    summary = {"primary_masked_fallback": {"videomae2": primary_visual, "no_video": primary_no_video}, "video_complete_sensitivity": {"n_labels": sensitivity_labels, "videomae2": sensitivity_visual, "no_video": sensitivity_no_video}, "runtime": "offline_or_recorded_shadow_replay_only"}
    write_json(reports / "comparison_metrics.json", summary)
    lines = ["# VideoMAE2 + 1DCNN 融合报告", "", "VideoMAE V2 ViT-Small 冻结；每个外层 LOPO 折只用训练参与者窗口拟合 32 维 PCA。", "", "| cohort / 模型 | 标签数 | Relaxation MAE | Discomfort MAE | 高风险召回 | 部署门 |", "|---|---:|---:|---:|---:|---|"]
    for cohort, visual in (("主分析 visual", primary_visual), ("主分析 no-video", primary_no_video), ("完整视频 visual", sensitivity_visual), ("完整视频 no-video", sensitivity_no_video)):
        metrics = visual["metrics"]
        lines.append(f"| {cohort} | {metrics['n_labels']} | {metrics['targets']['relaxation']['mae']:.4f} | {metrics['targets']['discomfort']['mae']:.4f} | {metrics['targets']['discomfort']['risk_at_fold_tuned_threshold']['per_row_threshold_recall']:.1%} | {'通过' if metrics['deployable'] else '未通过'} |")
    lines.extend(["", "默认实时服务不加载这些模型；即使指标通过，仍只能用独立 recorded shadow replay 检查。"])
    atomic_write_text(reports / "comparison_zh.md", "\n".join(lines) + "\n")
    return summary


def train_videomae2_relaxation(config: ProjectConfig) -> dict[str, Any]:
    """Train the isolated relaxation-only VideoMAE2 comparison experiment."""
    window_source = config.path("features") / "video_ml" / "window_features.csv"
    embedding_source = config.path("video") / "videomae2" / "window_embeddings.csv"
    if not window_source.exists() or not embedding_source.exists():
        raise FileNotFoundError("Run handcrafted video extraction and 'rtml extract-videomae2' first")
    root = config.path("video") / "relaxation_only"
    models = root / "models" / "videomae2_dcnn"
    reports = root / "reports" / "videomae2_dcnn"
    features = root / "features" / "videomae2_dcnn"
    options = {
        "targets": ("relaxation",),
        "model_kind": RELAXATION_VIDEO_MODEL_KIND,
        "research_only": True,
    }
    primary_visual = train_video_dcnn_model(
        config, window_source=window_source, embedding_source=embedding_source,
        model_path=models / "dcnn_relaxation_full.pt", reports_dir=reports / "visual",
        include_video=True, expected_labels=135, **options,
    )
    primary_no_video = train_video_dcnn_model(
        config, window_source=window_source, embedding_source=embedding_source,
        model_path=models / "dcnn_relaxation_no_video.pt", reports_dir=reports / "no_video",
        include_video=False, expected_labels=135, **options,
    )
    sensitivity_source = features / "window_features_video_complete.csv"
    _, sensitivity_labels = _complete_video_window_source(window_source, embedding_source, sensitivity_source)
    if sensitivity_labels != 134:
        raise ValueError(f"Expected 134 complete-video sensitivity labels; found {sensitivity_labels}")
    sensitivity_visual = train_video_dcnn_model(
        config, window_source=sensitivity_source, embedding_source=embedding_source,
        model_path=models / "sensitivity" / "dcnn_relaxation_full.pt", reports_dir=reports / "sensitivity" / "visual",
        include_video=True, expected_labels=sensitivity_labels, **options,
    )
    sensitivity_no_video = train_video_dcnn_model(
        config, window_source=sensitivity_source, embedding_source=embedding_source,
        model_path=models / "sensitivity" / "dcnn_relaxation_no_video.pt", reports_dir=reports / "sensitivity" / "no_video",
        include_video=False, expected_labels=sensitivity_labels, **options,
    )
    summary = {
        "targets": ["relaxation"],
        "research_only": True,
        "primary_masked_fallback": {"videomae2": primary_visual, "no_video": primary_no_video},
        "video_complete_sensitivity": {
            "n_labels": sensitivity_labels,
            "videomae2": sensitivity_visual,
            "no_video": sensitivity_no_video,
        },
    }
    write_json(reports / "comparison_metrics.json", summary)
    return summary


def train_videomae2_video_encoder_ablation(config: ProjectConfig) -> dict[str, Any]:
    """Uniformly retrain no-video, direct-video, and temporal-video encoders.

    The suite is intentionally isolated from the existing visual artifacts so
    every comparison uses one source snapshot, seed, LOPO partition, and
    reporting schema.
    """
    window_source = config.path("features") / "video_ml" / "window_features.csv"
    embedding_source = config.path("video") / "videomae2" / "window_embeddings.csv"
    if not window_source.exists() or not embedding_source.exists():
        raise FileNotFoundError("Run handcrafted video extraction and 'rtml extract-videomae2' first")
    root = config.path("video") / "video_encoder_ablation"
    features = root / "features"
    sensitivity_source = features / "window_features_video_complete.csv"
    _, sensitivity_labels = _complete_video_window_source(window_source, embedding_source, sensitivity_source)
    if sensitivity_labels != 134:
        raise ValueError(f"Expected 134 complete-video sensitivity labels; found {sensitivity_labels}")
    cohorts = {
        "primary_masked_fallback": (window_source, 135),
        "video_complete_sensitivity": (sensitivity_source, sensitivity_labels),
    }
    target_suites = {
        "dual_target": (TARGETS, VIDEO_ENCODER_ABLATION_DUAL_KIND),
        "relaxation_only": (("relaxation",), VIDEO_ENCODER_ABLATION_RELAXATION_KIND),
    }
    encoder_modes = {
        "no_video": (False, VIDEO_ENCODER_NONE),
        "video_direct_mlp": (True, VIDEO_ENCODER_MASKED_MEAN_MLP),
        "video_temporal_1dcnn": (True, VIDEO_ENCODER_TEMPORAL_DCNN),
    }
    summaries: dict[str, Any] = {}
    for suite_name, (targets, model_kind) in target_suites.items():
        suite_results: dict[str, Any] = {}
        for cohort_name, (source, expected_labels) in cohorts.items():
            models = root / "models" / suite_name / cohort_name
            reports = root / "reports" / suite_name / cohort_name
            encoder_results: dict[str, Any] = {}
            for encoder_name, (include_video, encoder_mode) in encoder_modes.items():
                encoder_results[encoder_name] = train_video_dcnn_model(
                    config,
                    window_source=source,
                    embedding_source=embedding_source,
                    model_path=models / f"{encoder_name}.pt",
                    reports_dir=reports / encoder_name,
                    include_video=include_video,
                    expected_labels=expected_labels,
                    targets=targets,
                    model_kind=model_kind,
                    research_only=True,
                    video_encoder_mode=encoder_mode,
                    model_variant=encoder_name,
                )
            suite_results[cohort_name] = encoder_results
        summaries[suite_name] = suite_results
    summary = {
        "experiment": "videomae2_video_encoder_ablation_v1",
        "research_only": True,
        "unit_of_analysis": "participant_condition",
        "random_seed": int(config.get("modeling.random_seed")),
        "video_encoder_modes": {
            "no_video": "no VideoMAE2 branch",
            "video_direct_mlp": "masked PCA mean plus video availability directly fused into MLP",
            "video_temporal_1dcnn": "PCA embedding sequence through a temporal video 1DCNN",
        },
        "source_hashes": {
            "window_features": file_sha256(window_source),
            "window_embeddings": file_sha256(embedding_source),
            "complete_video_window_features": file_sha256(sensitivity_source),
        },
        "n_labels": {"primary_masked_fallback": 135, "video_complete_sensitivity": sensitivity_labels},
        "suites": summaries,
    }
    reports_root = root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    write_json(reports_root / "comparison_metrics.json", summary)
    return summary


def _load_video_dcnn_model(
    path: Path,
    device_name: str,
    *,
    expected_model_kind: str,
    research_only: bool,
) -> dict[str, Any]:
    torch = _torch()
    device = _device(device_name)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_kind") != expected_model_kind:
        raise ValueError(f"Not a {expected_model_kind} checkpoint: {path}")
    if bool(checkpoint.get("research_only", False)) != research_only:
        raise ValueError(f"Unexpected research_only marker in checkpoint: {path}")
    if research_only and checkpoint.get("targets") != ["relaxation"]:
        raise ValueError(f"Research-only checkpoint must predict relaxation only: {path}")
    architecture = dict(checkpoint["architecture"])
    for key in ("conv_channels", "kernel_sizes", "pool_sizes"):
        architecture[key] = tuple(architecture[key])
    components = int(checkpoint["scaler"].get("video_components", 0))
    video_encoder_mode = str(checkpoint.get("video_encoder_mode", VIDEO_ENCODER_TEMPORAL_DCNN))
    model = _make_model(
        len(checkpoint["base_feature_columns"]), components + int(bool(components)), architecture,
        len(checkpoint["targets"]), video_encoder_mode,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    checkpoint["model"], checkpoint["device"] = model, device
    return checkpoint


def load_video_dcnn_model(path: Path, device_name: str) -> dict[str, Any]:
    """Load the dual-target recorded-replay model accepted by the existing runtime path."""
    return _load_video_dcnn_model(
        path, device_name, expected_model_kind=VIDEO_MODEL_KIND, research_only=False,
    )


def load_video_relaxation_dcnn_model(path: Path, device_name: str) -> dict[str, Any]:
    """Load an offline-only relaxation comparison checkpoint for evaluation tests."""
    return _load_video_dcnn_model(
        path, device_name, expected_model_kind=RELAXATION_VIDEO_MODEL_KIND, research_only=True,
    )


def _history_input(bundle: dict[str, Any], history: list[dict[str, Any]], condition: str, config: ProjectConfig):
    sequence = int(bundle["architecture"]["sequence_length"])
    base_columns = bundle["base_feature_columns"]
    embedding_columns = bundle.get("embedding_columns", [])
    base = np.full((1, len(base_columns), sequence), np.nan, dtype=float)
    embeddings = np.full((1, sequence, len(embedding_columns)), np.nan, dtype=float)
    mask = np.zeros((1, sequence), dtype=float)
    for index, row in enumerate(history[-sequence:]):
        position = min(index, sequence - 1)
        for column_index, column in enumerate(base_columns):
            try:
                base[0, column_index, position] = float(row.get(column, np.nan))
            except (TypeError, ValueError):
                pass
        for column_index, column in enumerate(embedding_columns):
            try:
                embeddings[0, position, column_index] = float(row.get(column, np.nan))
            except (TypeError, ValueError):
                pass
        try:
            mask[0, position] = float(row.get("video_available", 0.0)) >= 0.5
        except (TypeError, ValueError):
            pass
    context_values = condition_parameters(condition, list(config.get("conditions.intensities")), list(config.get("conditions.frequencies")))
    context = np.asarray([[context_values["intensity"], context_values["frequency"]]], dtype=float)
    return base, embeddings, mask, context


def predict_video_dcnn_state(bundle: dict[str, Any], history: list[dict[str, Any]], condition: str, config: ProjectConfig):
    torch = _torch()
    scaler = bundle["scaler"]
    base, embeddings, mask, context = _history_input(bundle, history, condition, config)
    sequence = int(bundle["architecture"]["sequence_length"])
    base = (base - np.asarray(scaler["base_mean"], dtype=float)[None, :, None]) / np.asarray(scaler["base_scale"], dtype=float)[None, :, None]
    base = np.where(np.isfinite(base), base, 0.0).astype(np.float32)
    context = (context - np.asarray(scaler["context_mean"], dtype=float)) / np.asarray(scaler["context_scale"], dtype=float)
    context = np.where(np.isfinite(context), context, 0.0).astype(np.float32)
    components = int(scaler.get("video_components", 0))
    if components:
        valid = (mask >= 0.5) & np.all(np.isfinite(embeddings), axis=2)
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        valid_flat = valid.reshape(-1)
        filled = np.broadcast_to(np.asarray(scaler["video_pca_mean"], dtype=float), flat.shape).copy()
        filled[valid_flat] = flat[valid_flat]
        pca = (filled - np.asarray(scaler["video_pca_mean"], dtype=float)) @ np.asarray(scaler["video_pca_components"], dtype=float).T
        pca = (pca - np.asarray(scaler["video_feature_mean"], dtype=float)) / np.asarray(scaler["video_feature_scale"], dtype=float)
        pca[~valid_flat] = 0.0
        video = np.concatenate([pca.reshape(1, sequence, components), valid[..., None].astype(float)], axis=2).transpose(0, 2, 1).astype(np.float32)
    else:
        video = np.zeros((1, 0, base.shape[-1]), dtype=np.float32)
    with torch.no_grad():
        result = bundle["model"](torch.as_tensor(base, device=bundle["device"]), torch.as_tensor(video, device=bundle["device"]), torch.as_tensor(context, device=bundle["device"])).detach().cpu().numpy()[0]
    history_windows = min(len(history), sequence)
    widths = bundle.get("interval_half_width_by_history", {}).get(str(history_windows), bundle.get("interval_half_width", {}))
    return ({target: float(np.clip(result[index], 0.0, 1.0)) for index, target in enumerate(bundle["targets"])}, {target: float(widths.get(target, 0.5)) for target in bundle["targets"]}, history_windows)
