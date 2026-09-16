"""Run the formal aligned Relax probe under the shared 7/1/1 protocol.

The input is a strict cache of aligned EEG, ECG, eye, head, and video
embeddings.  This script does not silently re-extract or repair those inputs:
it validates the cohort, masks, hashes, and split manifest, then trains either
a late-fusion baseline or a repository-native fusion head on the outer train
participants only.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch

from mac.data.relax_dataset import RelaxConditionEmbeddingDataset
from mac.fusion.late import LateFusion
from mac.fusion.projector import ModalityProjector


TARGETS = ("relaxation", "discomfort")
FUSION_TYPES = ("late", "early", "mid", "qformer", "healnet", "mm_lego")
FULL_MODALITIES = ("eeg", "ecg", "eye", "head", "video")
STATIC_FEATURE_COLUMNS = {
    "participant_id", "condition", "presentation_position", "intensity", "frequency",
    "intensity_index", "frequency_index", "condition_index", "window_count",
    "window_count_expected", "relaxation", "discomfort", "calm", "pleasantness",
    "monotony", "visual_fit", "relaxation_raw", "discomfort_raw", "arousal_raw",
    "pleasantness_raw", "monotony_raw", "label_source_row",
}


@dataclass(frozen=True)
class Fold:
    fold_index: int
    train_participants: tuple[str, ...]
    validation_participant: str
    test_participant: str


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _load_split_manifest(path: str | Path, participants: Iterable[str], *, strict: bool = True) -> list[Fold]:
    """Load and validate the registered participant split before training."""
    frame = pd.read_csv(path, dtype=str)
    required = {"fold_index", "test_participant", "validation_participant", "participant_id", "role"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Split manifest lacks columns: {sorted(missing)}")
    frame["fold_index"] = pd.to_numeric(frame["fold_index"], errors="raise").astype(int)
    frame["role"] = frame["role"].str.lower().str.strip()
    expected = tuple(sorted(str(value) for value in participants))
    folds: list[Fold] = []
    for fold_index, group in frame.groupby("fold_index", sort=True):
        if group["participant_id"].duplicated().any():
            raise ValueError(f"Fold {fold_index} repeats participants")
        if tuple(sorted(group["participant_id"].astype(str))) != expected:
            raise ValueError(f"Fold {fold_index} participant membership is not the requested cohort")
        roles = {
            role: tuple(sorted(group.loc[group["role"] == role, "participant_id"].astype(str)))
            for role in ("train", "validation", "test")
        }
        expected_train = len(expected) - 2
        if len(roles["train"]) != expected_train or len(roles["validation"]) != 1 or len(roles["test"]) != 1:
            raise ValueError(f"Fold {fold_index} is not {expected_train}/1/1")
        validation, test = roles["validation"][0], roles["test"][0]
        if set(group["validation_participant"]) != {validation} or set(group["test_participant"]) != {test}:
            raise ValueError(f"Fold {fold_index} role headers disagree with role rows")
        folds.append(Fold(int(fold_index), roles["train"], validation, test))
    if strict:
        expected_train = len(expected) - 2
        if len(expected) < 3 or len(folds) != len(expected):
            raise ValueError("Formal aligned runs require one fold per participant")
        if any(len(fold.train_participants) != expected_train for fold in folds):
            raise ValueError(
                f"Formal aligned runs require exactly {expected_train} training participants in every fold"
            )
        if sorted(fold.test_participant for fold in folds) != list(expected):
            raise ValueError("Every participant must be test exactly once")
        if sorted(fold.validation_participant for fold in folds) != list(expected):
            raise ValueError("Every participant must be validation exactly once")
    return folds


def _modality_variant(modalities: Iterable[str]) -> str:
    values = tuple(str(modality) for modality in modalities)
    if len(values) != len(set(values)):
        raise ValueError(f"Modalities contain duplicates: {values}")
    unknown = sorted(set(values) - set(FULL_MODALITIES))
    if unknown:
        raise ValueError(f"Unknown aligned modalities: {unknown}")
    if values == FULL_MODALITIES:
        return "full"
    missing = tuple(modality for modality in FULL_MODALITIES if modality not in values)
    expected_order = tuple(modality for modality in FULL_MODALITIES if modality in values)
    if len(missing) == 1 and values == expected_order:
        return f"no_{missing[0]}"
    raise ValueError(
        "Aligned modality configurations must be full or remove exactly one modality "
        f"in canonical order; received {values}"
    )


def _set_seed(seed: int) -> None:
    """Make numpy, Python, and torch behavior reproducible for one fold."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(args: argparse.Namespace) -> torch.device:
    requested = str(args.device)
    if args.require_cuda and not requested.startswith("cuda"):
        raise RuntimeError("Neural alignment run requires --device cuda when --require-cuda is set")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but torch.cuda.is_available() is false")
    device = torch.device(requested)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("Neural aligned runs may not fall back to CPU")
    return device


def _runtime(device: torch.device) -> dict[str, Any]:
    return {
        "device": str(device),
        "cuda_used": device.type == "cuda",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
    }


def _safe_correlation(truth: np.ndarray, prediction: np.ndarray, method: str) -> float:
    from scipy.stats import pearsonr, spearmanr

    if len(truth) < 2 or np.std(truth) <= 1e-15 or np.std(prediction) <= 1e-15:
        return float("nan")
    result = spearmanr(truth, prediction).statistic if method == "spearman" else pearsonr(truth, prediction).statistic
    return float(result) if np.isfinite(result) else float("nan")


def _ccc(truth: np.ndarray, prediction: np.ndarray) -> float:
    covariance = float(np.mean((truth - truth.mean()) * (prediction - prediction.mean())))
    denominator = float(truth.var() + prediction.var() + (truth.mean() - prediction.mean()) ** 2)
    return 2.0 * covariance / denominator if denominator > 0 else float("nan")


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {"n_predictions": int(len(frame)), "targets": {}}
    participant_macro = []
    for target in TARGETS:
        truth = frame[f"{target}_true"].to_numpy(dtype=float)
        prediction = frame[f"{target}_pred"].to_numpy(dtype=float)
        per_participant = frame.assign(_error=np.abs(truth - prediction)).groupby("participant_id")["_error"].mean()
        metrics = {
            "mae": float(np.mean(np.abs(truth - prediction))),
            "participant_macro_mae": float(per_participant.mean()),
            "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
            "spearman": _safe_correlation(truth, prediction, "spearman"),
            "pearson": _safe_correlation(truth, prediction, "pearson"),
            "ccc": _ccc(truth, prediction),
        }
        output["targets"][target] = metrics
        participant_macro.append(metrics["participant_macro_mae"])
    output["macro_mae"] = float(np.mean(participant_macro))
    return output


def _baseline_predictions(
    labels: pd.DataFrame,
    train_indexes: np.ndarray,
    test_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = labels.iloc[train_indexes]
    test = labels.iloc[test_indexes]
    condition = np.zeros((len(test), 2), dtype=float)
    history = np.zeros((len(test), 2), dtype=float)
    for target_index, target in enumerate(TARGETS):
        fallback = float(train[target].mean())
        means = train.groupby("condition")[target].mean()
        condition[:, target_index] = test["condition"].map(means).fillna(fallback).to_numpy(dtype=float)
        ordered = test.sort_values("presentation_position")
        values = np.r_[fallback, ordered[target].to_numpy(dtype=float)[:-1]]
        positions = {index: position for position, index in enumerate(test.index)}
        for index, value in zip(ordered.index, values, strict=True):
            history[positions[index], target_index] = value
    return condition, history


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if column in STATIC_FEATURE_COLUMNS or "_raw" in column:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            columns.append(column)
    if not columns:
        raise ValueError("No numeric handcrafted features are available")
    return sorted(columns)


def _fold_indexes(participants: np.ndarray, fold: Fold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.flatnonzero(np.isin(participants, fold.train_participants))
    validation = np.flatnonzero(participants == fold.validation_participant)
    test = np.flatnonzero(participants == fold.test_participant)
    if len(set(participants[train])) != len(fold.train_participants) or not len(validation) or not len(test):
        raise ValueError(f"Fold {fold.fold_index} cannot be mapped to dataset rows")
    return train, validation, test


def _run_handcrafted_baseline_cv(
    args: argparse.Namespace,
    labels: pd.DataFrame,
    folds: list[Fold],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    frame = pd.read_csv(args.handcrafted_features)
    frame = frame.merge(
        labels[["participant_id", "condition", "presentation_position", *TARGETS]],
        on=["participant_id", "condition"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_contract"),
    )
    for column in ("presentation_position", *TARGETS):
        contract = f"{column}_contract"
        if contract in frame:
            frame[column] = frame[contract]
    frame = frame.drop(columns=[column for column in frame.columns if column.endswith("_contract")])
    frame = frame.sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    columns = _feature_columns(frame)
    X = frame[columns].apply(pd.to_numeric, errors="coerce")
    y = frame[list(TARGETS)].to_numpy(dtype=float)
    participants = frame["participant_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    alphas = [float(value) for value in args.alphas]
    for fold in folds:
        train, validation, test = _fold_indexes(participants, fold)
        candidates = []
        for alpha in alphas:
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ]
            )
            model.fit(X.iloc[train], y[train])
            prediction = np.clip(model.predict(X.iloc[validation]), 0.0, 1.0)
            candidates.append((float(np.mean(np.abs(y[validation] - prediction))), alpha, model))
        validation_mae, alpha, model = min(candidates, key=lambda item: (item[0], item[1]))
        prediction = np.clip(model.predict(X.iloc[test]), 0.0, 1.0)
        condition_baseline, history_baseline = _baseline_predictions(frame, train, test)
        for local_index, row_index in enumerate(test):
            rows.append(
                {
                    "participant_id": participants[row_index],
                    "condition": str(frame.iloc[row_index]["condition"]),
                    "presentation_position": float(frame.iloc[row_index]["presentation_position"]),
                    "relaxation_true": float(y[row_index, 0]),
                    "discomfort_true": float(y[row_index, 1]),
                    "relaxation_pred": float(prediction[local_index, 0]),
                    "discomfort_pred": float(prediction[local_index, 1]),
                    "condition_only_relaxation": float(condition_baseline[local_index, 0]),
                    "condition_only_discomfort": float(condition_baseline[local_index, 1]),
                    "history_relaxation": float(history_baseline[local_index, 0]),
                    "history_discomfort": float(history_baseline[local_index, 1]),
                    "fold_index": fold.fold_index,
                    "test_participant": fold.test_participant,
                    "validation_participant": fold.validation_participant,
                    "seed": args.seed,
                    "model_variant": "ridge_cv",
                    "modalities": "handcrafted_shared_mask",
                    "device": "cpu",
                    "cuda_used": False,
                }
            )
        fold_records.append(
            {
                **asdict(fold),
                "n_train_participants": len(fold.train_participants),
                "n_validation_participants": 1,
                "n_test_participants": 1,
                "selected_alpha": alpha,
                "validation_macro_mae": validation_mae,
                "feature_count": len(columns),
                "device": "cpu",
                "cuda_used": False,
            }
        )
    return pd.DataFrame(rows), fold_records


class AlignedLateRegressor(torch.nn.Module):
    """Reference aligned head: pool, project, late-fuse, then regress two targets."""

    def __init__(self, embed_dims: dict[str, int], d_common: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.modalities = tuple(embed_dims)
        self.projector = ModalityProjector(embed_dims, d_common)
        self.missing_tokens = torch.nn.ParameterDict(
            {modality: torch.nn.Parameter(torch.zeros(d_common)) for modality in self.modalities}
        )
        self.fusion = LateFusion(
            d_common=d_common,
            num_modalities=len(self.modalities),
            modality_ids=list(self.modalities),
            mode="average",
            dropout=dropout,
            d_out=d_common,
        )
        self.head = torch.nn.Linear(d_common, 2)

    def forward(self, embeddings: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled: dict[str, torch.Tensor] = {}
        present: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            # Masked mean pooling turns a variable number of valid windows into
            # one condition representation without letting padded windows count.
            mask = masks[modality].bool()
            weights = mask.unsqueeze(-1).to(embeddings[modality].dtype)
            pooled[modality] = (embeddings[modality] * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            present[modality] = mask.any(dim=1)
        projected = self.projector(pooled)
        inputs = [
            # A learned token distinguishes "modality absent" from a real
            # all-zero embedding while keeping the tensor shape fixed.
            torch.where(
                present[modality].unsqueeze(-1),
                projected[modality],
                self.missing_tokens[modality].unsqueeze(0),
            )
            for modality in self.modalities
        ]
        fused = self.fusion(inputs, list(self.modalities))
        return torch.sigmoid(self.head(fused))


def _native_fusion(args: argparse.Namespace, modalities: tuple[str, ...]) -> torch.nn.Module:
    if args.fusion == "early":
        from mac.fusion.early import EarlyFusion

        return EarlyFusion(
            d_common=args.d_common,
            num_modalities=len(modalities),
            dropout=args.dropout,
            d_out=args.d_common,
        )
    if args.fusion == "mid":
        from mac.fusion.mid import MidFusion

        return MidFusion(
            d_common=args.d_common,
            modality_ids=list(modalities),
            dropout=args.dropout,
            d_out=args.d_common,
        )
    if args.fusion == "qformer":
        from mac.fusion.qformer import QFormerFusion

        return QFormerFusion(
            d_common=args.d_common,
            n_queries=args.latent_channels,
            d_query=args.d_common,
            n_layers=args.fusion_depth,
            n_heads=args.fusion_heads,
            cross_attn_freq=args.cross_attn_freq,
            dropout=args.dropout,
        )
    if args.fusion == "healnet":
        from mac.fusion.healnet import HEALNetFusion

        return HEALNetFusion(
            d_common=args.d_common,
            memory_size=args.latent_channels,
            n_layers=args.fusion_depth,
            n_heads=args.fusion_heads,
            num_modalities=len(modalities),
            dropout=args.dropout,
        )
    if args.fusion == "mm_lego":
        from mac.fusion.multimodal_lego import MultimodalLegoFusion

        return MultimodalLegoFusion(
            d_common=args.d_common,
            modality_ids=list(modalities),
            mode=args.lego_mode,
            latent_channels=args.latent_channels,
            latent_dim=args.latent_dim,
            depth=args.fusion_depth,
            heads=args.fusion_heads,
            dim_head=args.dim_head,
            attn_dropout=args.dropout,
            ff_dropout=args.dropout,
            frequency_domain=True,
            fourier_dim=1,
            track_imaginary=True,
            normalise=True,
        )
    raise ValueError(f"Unsupported aligned native fusion: {args.fusion}")


class AlignedNativeFusionRegressor(torch.nn.Module):
    """Shared aligned regression head around a repository-native fusion module."""

    def __init__(self, embed_dims: dict[str, int], args: argparse.Namespace) -> None:
        super().__init__()
        self.modalities = tuple(embed_dims)
        self.projector = ModalityProjector(embed_dims, args.d_common)
        self.missing_tokens = torch.nn.ParameterDict(
            {
                modality: torch.nn.Parameter(torch.zeros(args.d_common))
                for modality in self.modalities
            }
        )
        self.fusion = _native_fusion(args, self.modalities)
        self.head = torch.nn.Linear(int(self.fusion.d_out), 2)

    def _sequence_inputs(
        self,
        embeddings: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        projected = self.projector(embeddings)
        inputs: list[torch.Tensor] = []
        safe_masks: list[torch.Tensor] = []
        for modality in self.modalities:
            # Sequence-capable fusions receive every aligned window and an
            # explicit mask.  Invalid windows are zeroed before attention.
            mask = masks[modality].bool()
            values = torch.where(
                mask.unsqueeze(-1),
                projected[modality],
                torch.zeros_like(projected[modality]),
            )
            missing = ~mask.any(dim=1)
            safe_mask = mask.clone()
            if missing.any():
                # Attention implementations need at least one valid token;
                # use the learned missing token as a safe sentinel for a fully
                # absent modality and mark that sentinel valid.
                safe_mask[missing, 0] = True
                values = values.clone()
                values[missing, 0] = self.missing_tokens[modality]
            inputs.append(values)
            safe_masks.append(safe_mask)
        return inputs, safe_masks

    def _pooled_inputs(
        self,
        embeddings: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        pooled: dict[str, torch.Tensor] = {}
        present: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            # Non-sequence fusion modules cannot consume [batch, windows, dim],
            # so aligned windows are reduced to one masked condition vector.
            mask = masks[modality].bool()
            weights = mask.unsqueeze(-1).to(embeddings[modality].dtype)
            pooled[modality] = (
                (embeddings[modality] * weights).sum(dim=1)
                / weights.sum(dim=1).clamp(min=1.0)
            )
            present[modality] = mask.any(dim=1)
        projected = self.projector(pooled)
        return [
            torch.where(
                present[modality].unsqueeze(-1),
                projected[modality],
                self.missing_tokens[modality].unsqueeze(0),
            )
            for modality in self.modalities
        ]

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.fusion.supports_sequence_input:
            inputs, safe_masks = self._sequence_inputs(embeddings, masks)
            fused = self.fusion(inputs, list(self.modalities), safe_masks)
        else:
            inputs = self._pooled_inputs(embeddings, masks)
            fused = self.fusion(inputs, list(self.modalities))
        return torch.sigmoid(self.head(fused))


def _fusion_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "fusion": args.fusion,
        "d_common": args.d_common,
        "dropout": args.dropout,
        "fusion_depth": args.fusion_depth,
        "fusion_heads": args.fusion_heads,
        "cross_attn_freq": args.cross_attn_freq,
        "latent_channels": args.latent_channels,
        "latent_dim": args.latent_dim,
        "dim_head": args.dim_head,
        "lego_mode": args.lego_mode,
    }


def _fit_embedding_scalers(
    dataset: RelaxConditionEmbeddingDataset,
    train: np.ndarray,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Fit one embedding standardizer per modality using outer-train rows only."""
    scalers = {}
    for modality in dataset.modalities:
        values = dataset.embeddings[modality][train]
        mask = dataset.masks[modality][train]
        flattened = values[mask]
        if flattened.numel() == 0:
            mean = torch.zeros(values.shape[-1])
            scale = torch.ones(values.shape[-1])
        else:
            mean = flattened.mean(dim=0)
            scale = flattened.std(dim=0, unbiased=False)
            mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
            scale = torch.where(torch.isfinite(scale) & (scale > 1e-6), scale, torch.ones_like(scale))
        scalers[modality] = mean, scale
    return scalers


def _batch(
    dataset: RelaxConditionEmbeddingDataset,
    indexes: np.ndarray,
    scalers: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Create a device batch while preserving masks as first-class inputs."""
    embeddings = {}
    masks = {}
    for modality in dataset.modalities:
        mean, scale = scalers[modality]
        values = (dataset.embeddings[modality][indexes] - mean) / scale
        embeddings[modality] = torch.where(torch.isfinite(values), values, torch.zeros_like(values)).to(device)
        masks[modality] = dataset.masks[modality][indexes].to(device)
    return embeddings, masks, dataset.targets[indexes].to(device)


def _predict_neural(
    model: torch.nn.Module,
    dataset: RelaxConditionEmbeddingDataset,
    indexes: np.ndarray,
    scalers: dict[str, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        embeddings, masks, _ = _batch(dataset, indexes, scalers, device)
        return model(embeddings, masks).detach().cpu().numpy().astype(float)


def _train_fold(
    args: argparse.Namespace,
    dataset: RelaxConditionEmbeddingDataset,
    fold: Fold,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train one fixed 7/1/1 fold and restore its best validation checkpoint."""
    participants = np.asarray(dataset.participant_ids, dtype=str)
    train, validation, test = _fold_indexes(participants, fold)
    # Scaling is deliberately fitted before the model is built and only from
    # outer-train embeddings; validation/test statistics stay unseen.
    scalers = _fit_embedding_scalers(dataset, train)
    _set_seed(args.seed + fold.fold_index)
    model = (
        AlignedLateRegressor(dataset.embed_dims, args.d_common, args.dropout)
        if args.fusion == "late"
        else AlignedNativeFusionRegressor(dataset.embed_dims, args)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    generator = np.random.default_rng(args.seed + fold.fold_index)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epochs_ran = 0
    for epoch in range(args.max_epochs):
        epochs_ran = epoch + 1
        model.train()
        # The neural head is trained on condition observations, not individual
        # windows.  Window-level masks remain inside each observation.
        order = generator.permutation(train)
        for start in range(0, len(order), args.batch_size):
            indexes = np.asarray(order[start : start + args.batch_size], dtype=int)
            embeddings, masks, targets = _batch(dataset, indexes, scalers, device)
            optimizer.zero_grad()
            loss = loss_fn(model(embeddings, masks), targets)
            loss.backward()
            optimizer.step()
        # Validation controls early stopping; the outer test participant is
        # evaluated only after the best validation checkpoint is restored.
        validation_prediction = _predict_neural(model, dataset, validation, scalers, device)
        validation_truth = dataset.targets[validation].numpy()
        validation_loss = float(np.mean((validation_truth - validation_prediction) ** 2))
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"Fold {fold.fold_index} did not produce a neural checkpoint")
    model.load_state_dict(best_state)
    prediction = _predict_neural(model, dataset, test, scalers, device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fold_{fold.fold_index:02d}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "embed_dims": dataset.embed_dims,
            "modalities": list(dataset.modalities),
            "fusion_config": _fusion_config(args),
            "d_common": args.d_common,
            "seed": args.seed,
            "fold": asdict(fold),
            "scalers": {
                modality: {"mean": mean, "scale": scale}
                for modality, (mean, scale) in scalers.items()
            },
        },
        checkpoint_path,
    )
    record = {
        **asdict(fold),
        "n_train_participants": len(fold.train_participants),
        "n_validation_participants": 1,
        "n_test_participants": 1,
        "training_seed": args.seed + fold.fold_index,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "validation_mse": best_loss,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        **_runtime(device),
        "mask_valid_windows_train": {
            modality: int(dataset.masks[modality][train].sum()) for modality in dataset.modalities
        },
    }
    return prediction, test, record


def _run_neural(
    args: argparse.Namespace,
    labels: pd.DataFrame,
    folds: list[Fold],
    participants: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    device = _device(args)
    dataset = RelaxConditionEmbeddingDataset(
        args.embedding_cache,
        modalities=args.modalities,
        participants=participants,
        mask_manifest=args.mask_manifest,
        strict=args.strict,
    )
    label_lookup = labels.set_index(["participant_id", "condition"])
    for index, key in enumerate(dataset.condition_keys()):
        if key not in label_lookup.index:
            raise ValueError(f"Cache contains unknown label key {key}")
        expected = label_lookup.loc[key, list(TARGETS)].to_numpy(dtype=float)
        if not np.allclose(dataset.targets[index].numpy(), expected, atol=1e-6, rtol=0.0):
            raise ValueError(f"Cache target mismatch for {key}")
    dataset_order = pd.DataFrame(
        {
            "participant_id": dataset.participant_ids,
            "condition": dataset.conditions,
            "presentation_position": dataset.presentation_positions.numpy(),
            "relaxation": dataset.targets[:, 0].numpy(),
            "discomfort": dataset.targets[:, 1].numpy(),
        }
    )
    rows = []
    fold_records = []
    variant = _modality_variant(args.modalities)
    split_protocol = f"shared_{len(participants) - 2}_train_1_validation_1_test"
    checkpoint_dir = args.output_dir / "checkpoints"
    for fold in folds:
        prediction, test, record = _train_fold(args, dataset, fold, device, checkpoint_dir)
        train, _, _ = _fold_indexes(np.asarray(dataset.participant_ids, dtype=str), fold)
        condition_baseline, history_baseline = _baseline_predictions(dataset_order, train, test)
        for local_index, row_index in enumerate(test):
            rows.append(
                {
                    "participant_id": dataset.participant_ids[row_index],
                    "condition": dataset.conditions[row_index],
                    "presentation_position": float(dataset.presentation_positions[row_index]),
                    "relaxation_true": float(dataset.targets[row_index, 0]),
                    "discomfort_true": float(dataset.targets[row_index, 1]),
                    "relaxation_pred": float(prediction[local_index, 0]),
                    "discomfort_pred": float(prediction[local_index, 1]),
                    "condition_only_relaxation": float(condition_baseline[local_index, 0]),
                    "condition_only_discomfort": float(condition_baseline[local_index, 1]),
                    "history_relaxation": float(history_baseline[local_index, 0]),
                    "history_discomfort": float(history_baseline[local_index, 1]),
                    "fold_index": fold.fold_index,
                    "test_participant": fold.test_participant,
                    "validation_participant": fold.validation_participant,
                    "seed": args.seed,
                    "model_variant": f"{args.fusion}_{variant}",
                    "modalities": "+".join(args.modalities),
                    "split_protocol": split_protocol,
                    "device": str(device),
                    "cuda_used": device.type == "cuda",
                }
            )
        fold_records.append(record)
        print(
            f"{args.fusion} {'+'.join(args.modalities)} fold "
            f"{fold.fold_index}/{len(folds)}: {fold.test_participant}",
            flush=True,
        )
    return pd.DataFrame(rows), fold_records, _runtime(device)


def _validate_contract_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[Fold]]:
    labels = pd.read_csv(args.labels)
    windows = pd.read_csv(args.windows)
    masks = pd.read_csv(args.mask_manifest)
    window_keys = ["participant_id", "condition", "condition_window_index"]
    if (
        labels.duplicated(["participant_id", "condition"]).any()
        or windows.duplicated(window_keys).any()
        or masks.duplicated(window_keys).any()
    ):
        raise ValueError("Shared label/window inputs contain duplicate keys")
    cohorts = json.loads(Path(args.cohorts).read_text(encoding="utf-8"))
    if args.cohort not in cohorts:
        raise ValueError(f"Unknown cohort {args.cohort!r}")
    participants = [str(value) for value in cohorts[args.cohort]]
    if len(participants) != len(set(participants)) or len(participants) < 3:
        raise ValueError(f"Cohort {args.cohort!r} is duplicated or too small")
    labels = labels.loc[labels["participant_id"].astype(str).isin(participants)].copy()
    labels = labels.sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    cohort_windows = windows.loc[windows["participant_id"].astype(str).isin(participants)].copy()
    cohort_masks = masks.loc[masks["participant_id"].astype(str).isin(participants)].copy()
    if args.strict:
        expected_participants = set(participants)
        if set(labels["participant_id"].astype(str)) != expected_participants:
            raise ValueError("Formal labels do not cover every cohort participant")
        condition_counts = labels.groupby(labels["participant_id"].astype(str)).size()
        if len(labels) != 9 * len(participants) or set(condition_counts.astype(int)) != {9}:
            raise ValueError("Formal aligned cohorts require nine condition labels per participant")
        window_key_set = set(map(tuple, cohort_windows[window_keys].to_numpy()))
        mask_key_set = set(map(tuple, cohort_masks[window_keys].to_numpy()))
        if not window_key_set or window_key_set != mask_key_set:
            raise ValueError("Formal window and mask keys differ for the requested cohort")
        label_key_set = set(map(tuple, labels[["participant_id", "condition"]].to_numpy()))
        window_label_keys = set(map(tuple, cohort_windows[["participant_id", "condition"]].to_numpy()))
        if label_key_set != window_label_keys:
            raise ValueError("Formal labels and window observations differ for the requested cohort")
    folds = _load_split_manifest(args.split_manifest, participants, strict=args.strict)
    return labels, participants, folds


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    labels, participants, folds = _validate_contract_inputs(args)
    started = time.perf_counter()
    if args.baseline == "relax_handcrafted_ridge_cv":
        predictions, fold_records = _run_handcrafted_baseline_cv(args, labels, folds)
        runtime = _runtime(torch.device("cpu"))
        model_name = "corrected_ridge_cv"
    else:
        predictions, fold_records, runtime = _run_neural(args, labels, folds, participants)
        model_name = f"corrected_{args.fusion}_{_modality_variant(args.modalities)}"
    predictions = predictions.sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    if args.strict:
        if len(predictions) != len(labels) or predictions.duplicated(["participant_id", "condition"]).any():
            raise ValueError("Aligned run did not produce exactly one prediction per label")
        expected_keys = set(map(tuple, labels[["participant_id", "condition"]].to_numpy()))
        observed_keys = set(map(tuple, predictions[["participant_id", "condition"]].to_numpy()))
        if expected_keys != observed_keys:
            raise ValueError("Aligned prediction coverage differs from the shared labels")
        expected_train = len(participants) - 2
        if any(record["n_train_participants"] != expected_train for record in fold_records):
            raise ValueError(
                f"At least one fold did not use exactly {expected_train} training participants"
            )
        if args.baseline == "none" and not all(record["cuda_used"] for record in fold_records):
            raise ValueError("A neural fold did not record CUDA usage")
    run_name = args.run_name or f"{model_name}_s{args.seed}"
    prediction_path = args.output_dir / f"{run_name}_predictions.csv"
    result_path = args.output_dir / f"{run_name}_results.json"
    predictions.to_csv(prediction_path, index=False)
    input_paths = {
        "labels": Path(args.labels),
        "windows": Path(args.windows),
        "split_manifest": Path(args.split_manifest),
        "mask_manifest": Path(args.mask_manifest),
        "cohorts": Path(args.cohorts),
    }
    if args.baseline == "relax_handcrafted_ridge_cv":
        input_paths["handcrafted_features"] = Path(args.handcrafted_features)
    else:
        input_paths["embedding_cache"] = Path(args.embedding_cache)
    payload = {
        "schema_version": "relax_aligned_run_v1",
        "run_name": run_name,
        "model": model_name,
        "seed": args.seed,
        "cohort": args.cohort,
        "modalities": list(args.modalities),
        "split_protocol": f"shared_{len(participants) - 2}_train_1_validation_1_test",
        "participant_count": len(participants),
        "observation_count": len(labels),
        "fold_count": len(folds),
        "metrics": _metrics(predictions),
        "folds": fold_records,
        "runtime": runtime,
        "runtime_seconds": time.perf_counter() - started,
        "prediction_path": str(prediction_path),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in input_paths.items()
        },
        "training": {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "d_common": args.d_common,
        },
        "fusion_config": _fusion_config(args),
    }
    _write_json(result_path, payload)
    return {"prediction_path": str(prediction_path), "result_path": str(result_path), "metrics": payload["metrics"]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--cohort", default="all_135")
    parser.add_argument("--modalities", nargs="+", default=["eeg", "ecg", "eye", "head", "video"])
    parser.add_argument("--fusion", choices=FUSION_TYPES, default="late")
    parser.add_argument("--baseline", choices=["none", "relax_handcrafted_ridge_cv"], default="none")
    parser.add_argument("--handcrafted-features", type=Path)
    parser.add_argument("--target", default="original")
    parser.add_argument("--condition-control", default="none")
    parser.add_argument("--pool", default="sequence")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-common", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fusion-depth", type=int, default=1)
    parser.add_argument("--fusion-heads", type=int, default=2)
    parser.add_argument("--cross-attn-freq", type=int, default=1)
    parser.add_argument("--dim-head", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--lego-mode", choices=["merge-sum"], default="merge-sum")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--run-tag", default="cross_project_aligned_20260716")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    if args.target != "original" or args.condition_control != "none" or args.pool != "sequence":
        parser.error("Aligned comparison locks --target original --condition-control none --pool sequence")
    if args.baseline == "relax_handcrafted_ridge_cv":
        if args.handcrafted_features is None:
            parser.error("--handcrafted-features is required for Ridge-CV")
        args.device = "cpu"
        args.require_cuda = False
    elif args.embedding_cache is None:
        parser.error("--embedding-cache is required for neural fusion")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_model(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.output_dir / "hard_failure.json",
            {"error_type": type(error).__name__, "error": str(error), "seed": args.seed},
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
