"""Condition-control transforms for Relax foundation probe experiments."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.relax_foundation_dataset import RelaxConditionEmbeddingDataset
from src.data.relax_foundation import RelaxHardFailure


@dataclass(frozen=True)
class FoldConditionResidualizer:
    """Subtract train-fold Condition means from one modality's window tokens.

    The fit step only reads samples indexed by the current training participants.
    Validation and test samples are transformed using those fixed means.
    """

    modality: str
    means: dict[str, torch.Tensor]

    @classmethod
    def fit(
        cls,
        dataset: RelaxConditionEmbeddingDataset,
        *,
        train_indices: list[int],
        modality: str,
    ) -> "FoldConditionResidualizer":
        if modality != "video":
            raise RelaxHardFailure(f"Unsupported Condition residualization modality: {modality}")
        if not train_indices:
            raise RelaxHardFailure("Condition residualizer received an empty training fold")

        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        for index in train_indices:
            sample = dataset[index]
            condition = str(sample["condition"])
            if modality not in sample["embeddings"]:
                raise RelaxHardFailure(f"{sample['participant_id']}/{condition}: missing {modality} embedding")
            emb = sample["embeddings"][modality].float()
            mask = sample["masks"][modality].bool()
            if emb.ndim != 2:
                raise RelaxHardFailure(
                    f"{sample['participant_id']}/{condition}/{modality}: expected [W,D], got {tuple(emb.shape)}"
                )
            if mask.ndim != 1 or mask.shape[0] != emb.shape[0]:
                raise RelaxHardFailure(
                    f"{sample['participant_id']}/{condition}/{modality}: mask shape {tuple(mask.shape)} "
                    f"does not match embedding shape {tuple(emb.shape)}"
                )
            valid = emb[mask]
            if valid.numel() == 0:
                continue
            sums[condition] = sums.get(condition, torch.zeros(emb.shape[-1], dtype=torch.float32)) + valid.sum(dim=0)
            counts[condition] = counts.get(condition, 0) + int(valid.shape[0])

        means: dict[str, torch.Tensor] = {}
        for condition, total in sums.items():
            count = counts.get(condition, 0)
            if count <= 0:
                raise RelaxHardFailure(f"{condition}: no valid train-fold {modality} windows for residualization")
            means[condition] = total / float(count)
        if not means:
            raise RelaxHardFailure(f"No valid train-fold {modality} windows for Condition residualization")
        return cls(modality=modality, means=means)

    def transform_batch(self, batch: dict[str, object]) -> dict[str, object]:
        emb_key = f"{self.modality}_emb"
        mask_key = f"{self.modality}_mask"
        if emb_key not in batch or mask_key not in batch:
            raise RelaxHardFailure(f"Batch is missing {self.modality} tensors for Condition residualization")
        if "conditions" not in batch:
            raise RelaxHardFailure("Batch is missing Condition labels for Condition residualization")

        emb = batch[emb_key]
        mask = batch[mask_key]
        conditions = batch["conditions"]
        if not isinstance(emb, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise RelaxHardFailure(f"Batch {self.modality} tensors must be torch.Tensor objects")
        if emb.ndim != 3:
            raise RelaxHardFailure(f"{self.modality} residualization expects [B,W,D], got {tuple(emb.shape)}")
        if mask.ndim != 2 or mask.shape[:2] != emb.shape[:2]:
            raise RelaxHardFailure(
                f"{self.modality} mask shape {tuple(mask.shape)} does not match embedding shape {tuple(emb.shape)}"
            )
        if not isinstance(conditions, list) or len(conditions) != emb.shape[0]:
            raise RelaxHardFailure("Batch Condition labels do not match residualization batch size")

        out = dict(batch)
        adjusted = emb.clone()
        bool_mask = mask.bool()
        for row, condition in enumerate(conditions):
            if condition not in self.means:
                raise RelaxHardFailure(f"{condition}: no train-fold mean for {self.modality} Condition residualization")
            mean = self.means[condition].to(dtype=adjusted.dtype, device=adjusted.device)
            if mean.shape[-1] != adjusted.shape[-1]:
                raise RelaxHardFailure(
                    f"{condition}: train-fold {self.modality} mean dimension {mean.shape[-1]} "
                    f"does not match batch dimension {adjusted.shape[-1]}"
                )
            adjusted[row] = adjusted[row] - mean
        adjusted = adjusted.masked_fill(~bool_mask.unsqueeze(-1), 0.0)
        out[emb_key] = adjusted
        out[mask_key] = bool_mask
        return out
