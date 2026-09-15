"""Late fusion: per-modality full branches + decision-level fusion."""
from __future__ import annotations

import torch
import torch.nn as nn

from mac.fusion.base import BaseFusionModule


class LateFusion(BaseFusionModule):
    """Per-modality MLP branches fused at decision level."""

    def __init__(
        self,
        d_common: int = 256,
        num_modalities: int = 3,
        modality_ids: list[str] | None = None,
        mode: str = "average",
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_out or d_common)
        self.mode = mode
        self.modality_ids = list(modality_ids) if modality_ids is not None else None
        n_branches = len(self.modality_ids) if self.modality_ids is not None else num_modalities
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_common, d_common),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_common, self.d_out),
            )
            for _ in range(n_branches)
        ])
        if mode == "weighted":
            self.weights = nn.Parameter(torch.ones(n_branches))

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        branch_indices = self._resolve_branch_indices(modality_ids, len(embeddings))
        branch_outs = [
            self.branches[idx](emb)
            for emb, idx in zip(embeddings, branch_indices)
        ]
        stacked = torch.stack(branch_outs, dim=0)

        if self.mode == "average":
            return stacked.mean(dim=0)
        elif self.mode == "weighted":
            w = torch.softmax(self.weights[branch_indices], dim=0).view(-1, 1, 1)
            return (stacked * w).sum(dim=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _resolve_branch_indices(
        self,
        modality_ids: list[str],
        n_embeddings: int,
    ) -> list[int]:
        if len(modality_ids) != n_embeddings:
            raise ValueError("modality_ids and embeddings must have the same length")
        if self.modality_ids is None:
            if len(modality_ids) != len(self.branches):
                raise ValueError(
                    "LateFusion without constructor modality_ids must see all "
                    "modalities on the first forward pass"
                )
            self.modality_ids = list(modality_ids)

        index_by_modality = {mod: i for i, mod in enumerate(self.modality_ids)}
        missing = [mod for mod in modality_ids if mod not in index_by_modality]
        if missing:
            raise ValueError(f"Unknown modalities for LateFusion: {missing}")
        return [index_by_modality[mod] for mod in modality_ids]
