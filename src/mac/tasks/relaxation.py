"""Relax condition-level regression, baselines, and fusion probe modules."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mac.data.relax_foundation import CONDITIONS, RelaxHardFailure
from mac.fusion.early import EarlyFusion
from mac.fusion.healnet import HEALNetFusion
from mac.fusion.late import LateFusion
from mac.fusion.mid import MidFusion
from mac.fusion.multimodal_lego import MultimodalLegoFusion
from mac.fusion.qformer import QFormerFusion
from mac.models.head_motion_1dcnn import HeadMotion1DCNN
from mac.evaluation.metrics import concordance_correlation_coefficient


TARGETS: tuple[str, str] = ("relaxation", "discomfort")


def compute_relax_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: tuple[str, ...] = TARGETS,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise RelaxHardFailure(f"Metric shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    metrics: dict[str, float] = {}
    for idx, name in enumerate(target_names):
        truth = y_true[:, idx]
        pred = y_pred[:, idx]
        metrics[f"{name}_mae"] = float(mean_absolute_error(truth, pred))
        metrics[f"{name}_rmse"] = float(np.sqrt(mean_squared_error(truth, pred)))
        metrics[f"{name}_r2"] = float(r2_score(truth, pred)) if len(np.unique(truth)) > 1 else float("nan")
        metrics[f"{name}_ccc"] = concordance_correlation_coefficient(truth, pred)
        if len(truth) > 1 and np.std(truth) > 0 and np.std(pred) > 0:
            metrics[f"{name}_pearson"] = float(np.corrcoef(truth, pred)[0, 1])
        else:
            metrics[f"{name}_pearson"] = float("nan")
    metrics["macro_mae"] = float(np.mean([metrics[f"{name}_mae"] for name in target_names]))
    metrics["macro_rmse"] = float(np.mean([metrics[f"{name}_rmse"] for name in target_names]))
    return metrics


def random_9_condition_baseline(
    train_conditions: pd.DataFrame,
    test_conditions: pd.DataFrame,
    *,
    seeds: Iterable[int] = range(1000),
) -> dict[str, Any]:
    """Uniformly choose one of nine Conditions, using train-only Condition means.

    The random draw never uses test labels. Test labels are used only after the
    predictions are formed to compute evaluation metrics.
    """
    required = {"participant_id", "condition", *TARGETS}
    missing_train = sorted(required - set(train_conditions.columns))
    missing_test = sorted(required - set(test_conditions.columns))
    if missing_train or missing_test:
        raise RelaxHardFailure(
            f"random_9_condition baseline missing columns: train={missing_train}, test={missing_test}"
        )
    means = train_conditions.groupby("condition", sort=False)[list(TARGETS)].mean()
    missing_conditions = [condition for condition in CONDITIONS if condition not in means.index]
    if missing_conditions:
        raise RelaxHardFailure(
            f"random_9_condition baseline training fold missing Condition means: {missing_conditions}"
        )

    test = test_conditions.reset_index(drop=True)
    seed_values = [int(seed) for seed in seeds]
    truth = test[list(TARGETS)].to_numpy(dtype=float)
    condition_values = np.asarray(CONDITIONS)
    choices_by_seed = np.stack(
        [
            np.random.default_rng(seed).choice(condition_values, size=len(test), replace=True)
            for seed in seed_values
        ],
        axis=0,
    ) if seed_values else np.empty((0, len(test)), dtype=str)
    mean_lookup = means.loc[list(CONDITIONS), list(TARGETS)].to_numpy(dtype=float)
    condition_indices = np.searchsorted(condition_values, choices_by_seed)
    predictions_array = mean_lookup[condition_indices] if seed_values else np.empty((0, len(test), len(TARGETS)))

    repeated = test[["participant_id", "condition"]].iloc[
        np.tile(np.arange(len(test)), len(seed_values))
    ].reset_index(drop=True) if seed_values else pd.DataFrame(columns=["participant_id", "condition"])
    repeated["seed"] = np.repeat(seed_values, len(test))
    repeated["chosen_condition"] = choices_by_seed.reshape(-1)
    repeated["relaxation_pred"] = predictions_array[:, :, 0].reshape(-1)
    repeated["discomfort_pred"] = predictions_array[:, :, 1].reshape(-1)
    repeated["relaxation_true"] = np.tile(truth[:, 0], len(seed_values))
    repeated["discomfort_true"] = np.tile(truth[:, 1], len(seed_values))
    predictions = repeated

    batched_metrics: dict[str, np.ndarray] = {}
    for target_index, target in enumerate(TARGETS):
        target_truth = truth[:, target_index]
        target_predictions = predictions_array[:, :, target_index]
        error = target_predictions - target_truth[None, :]
        batched_metrics[f"{target}_mae"] = np.mean(np.abs(error), axis=1)
        batched_metrics[f"{target}_rmse"] = np.sqrt(np.mean(error**2, axis=1))
        denominator = float(np.sum((target_truth - np.mean(target_truth)) ** 2))
        batched_metrics[f"{target}_r2"] = (
            1.0 - np.sum(error**2, axis=1) / denominator
            if denominator > 0
            else np.full(len(seed_values), np.nan)
        )
        pred_mean = np.mean(target_predictions, axis=1)
        truth_mean = float(np.mean(target_truth))
        covariance = np.mean(
            (target_predictions - pred_mean[:, None]) * (target_truth[None, :] - truth_mean),
            axis=1,
        )
        ccc_denominator = (
            np.var(target_predictions, axis=1)
            + float(np.var(target_truth))
            + (pred_mean - truth_mean) ** 2
        )
        batched_metrics[f"{target}_ccc"] = np.divide(
            2.0 * covariance,
            ccc_denominator,
            out=np.full(len(seed_values), np.nan),
            where=ccc_denominator > 0,
        )
        pred_std = np.std(target_predictions, axis=1)
        truth_std = float(np.std(target_truth))
        pearson_denominator = pred_std * truth_std
        batched_metrics[f"{target}_pearson"] = np.divide(
            covariance,
            pearson_denominator,
            out=np.full(len(seed_values), np.nan),
            where=pearson_denominator > 0,
        )
    batched_metrics["macro_mae"] = np.mean(
        np.stack([batched_metrics[f"{target}_mae"] for target in TARGETS], axis=1),
        axis=1,
    )
    batched_metrics["macro_rmse"] = np.mean(
        np.stack([batched_metrics[f"{target}_rmse"] for target in TARGETS], axis=1),
        axis=1,
    )
    summary: dict[str, float] = {"num_seeds": float(len(seed_values))}
    for name, values in sorted(batched_metrics.items()):
        finite = values[np.isfinite(values)]
        summary[f"{name}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
        summary[f"{name}_std"] = float(np.std(finite, ddof=0)) if finite.size else float("nan")
    return {"predictions": predictions, "summary": summary}


class _MaskSafeFusion(nn.Module):
    """Wrap sequence fusions so all-missing rows do not produce NaNs."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.d_out = getattr(module, "d_out", None)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if masks is None:
            return self.module(embeddings, modality_ids, masks=None)
        safe_embeddings: list[torch.Tensor] = []
        safe_masks: list[torch.Tensor] = []
        for emb, mask in zip(embeddings, masks):
            if emb.dim() == 2:
                safe_embeddings.append(emb)
                safe_masks.append(mask)
                continue
            fixed_emb = emb.clone()
            fixed_mask = mask.clone()
            no_valid = ~fixed_mask.any(dim=1)
            if no_valid.any():
                fixed_emb[no_valid] = 0.0
                fixed_mask[no_valid, 0] = True
            safe_embeddings.append(fixed_emb)
            safe_masks.append(fixed_mask)
        return self.module(safe_embeddings, modality_ids, masks=safe_masks)


def create_relax_fusion(
    name: str,
    *,
    modalities: list[str],
    d_common: int = 256,
    **kwargs: Any,
) -> nn.Module:
    """Create a fusion module for Relax, including the ``mm_lego`` CLI choice."""
    name = name.lower()
    if name == "early":
        return EarlyFusion(d_common=d_common, num_modalities=len(modalities), **kwargs)
    if name == "mid":
        return MidFusion(d_common=d_common, modality_ids=modalities, **kwargs)
    if name == "late":
        return LateFusion(d_common=d_common, modality_ids=modalities, num_modalities=len(modalities), **kwargs)
    if name == "qformer":
        return _MaskSafeFusion(QFormerFusion(d_common=d_common, **kwargs))
    if name == "healnet":
        return _MaskSafeFusion(HEALNetFusion(d_common=d_common, num_modalities=len(modalities), **kwargs))
    if name == "mm_lego":
        return _MaskSafeFusion(
            MultimodalLegoFusion(d_common=d_common, modality_ids=modalities, **kwargs)
        )
    raise RelaxHardFailure(f"Unknown Relax fusion method: {name}")


class _MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (x * weights).sum(dim=1) / denom


class RelaxFusionRegressor(nn.Module):
    """Condition-level probe over Relax modality window sequences."""

    def __init__(
        self,
        *,
        embed_dims: dict[str, int | tuple[int, int] | list[int]],
        modalities: list[str],
        fusion_name: str,
        d_common: int = 256,
        pool: str = "sequence",
        fusion_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modalities = list(modalities)
        self.pool = pool
        self.d_common = d_common
        self.projectors = nn.ModuleDict()
        self.head_raw_encoders = nn.ModuleDict()
        self.missing_tokens = nn.ParameterDict()
        self.attention_pools = nn.ModuleDict()

        for modality in self.modalities:
            if modality not in embed_dims:
                raise RelaxHardFailure(f"Missing embedding dimension for modality {modality}")
            dim = embed_dims[modality]
            if isinstance(dim, (tuple, list)):
                if modality != "head" or len(dim) != 2:
                    raise RelaxHardFailure(f"Only raw head-motion dimensions may be tuples, got {modality}: {dim}")
                self.head_raw_encoders[modality] = HeadMotion1DCNN(
                    input_channels=int(dim[0]),
                    embedding_dim=d_common,
                )
            else:
                self.projectors[modality] = nn.Linear(int(dim), d_common)
            self.missing_tokens[modality] = nn.Parameter(torch.zeros(1, 1, d_common))
            self.attention_pools[modality] = _MaskedAttentionPool(d_common)

        self.fusion = create_relax_fusion(
            fusion_name,
            modalities=self.modalities,
            d_common=d_common,
            **(fusion_kwargs or {}),
        )
        d_fused = int(getattr(self.fusion, "d_out", None) or d_common)
        self.head = nn.Sequential(
            nn.LayerNorm(d_fused),
            nn.Dropout(dropout),
            nn.Linear(d_fused, len(TARGETS)),
        )

    def _encode_modality(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        if modality in self.head_raw_encoders:
            if x.ndim != 4:
                raise RelaxHardFailure(f"Raw head-motion tensor must be [B,W,C,T], got {tuple(x.shape)}")
            b, w, c, t = x.shape
            encoded = self.head_raw_encoders[modality](x.reshape(b * w, c, t))
            return encoded.reshape(b, w, self.d_common)
        if x.ndim != 3:
            raise RelaxHardFailure(f"{modality} embedding tensor must be [B,W,D], got {tuple(x.shape)}")
        return self.projectors[modality](x)

    def _ensure_valid_context(
        self,
        modality: str,
        z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z.shape[1] == 0:
            raise RelaxHardFailure(f"{modality} has zero window tokens in a batch")
        fixed_z = z.clone()
        fixed_mask = mask.clone()
        no_valid = ~fixed_mask.any(dim=1)
        if no_valid.any():
            fixed_z[no_valid, 0:1, :] = self.missing_tokens[modality].to(z.dtype)
            fixed_mask[no_valid, 0] = True
        return fixed_z, fixed_mask

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        sequence_fusion = self.pool == "sequence" and getattr(self.fusion, "module", self.fusion).__class__.__name__ in {
            "QFormerFusion",
            "HEALNetFusion",
            "MultimodalLegoFusion",
        }

        for modality in self.modalities:
            x = batch[f"{modality}_emb"]
            mask = batch[f"{modality}_mask"].bool().to(x.device)
            z = self._encode_modality(modality, x)
            z, mask = self._ensure_valid_context(modality, z, mask)
            if sequence_fusion:
                embeddings.append(z)
                masks.append(mask)
            elif self.pool == "attention":
                embeddings.append(self.attention_pools[modality](z, mask))
            else:
                embeddings.append(_masked_mean(z, mask))

        fused = self.fusion(
            embeddings,
            self.modalities,
            masks=masks if sequence_fusion else None,
        )
        return self.head(fused)
