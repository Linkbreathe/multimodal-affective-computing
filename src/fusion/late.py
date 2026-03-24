"""Late fusion: per-modality full branches + decision-level fusion."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class LateFusion(BaseFusionModule):
    """Per-modality MLP branches fused at decision level."""

    def __init__(
        self,
        d_common: int = 256,
        num_modalities: int = 3,
        mode: str = "average",
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_out or d_common)
        self.mode = mode
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_common, d_common),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_common, self.d_out),
            )
            for _ in range(num_modalities)
        ])
        if mode == "weighted":
            self.weights = nn.Parameter(torch.ones(num_modalities))

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        branch_outs = [branch(emb) for branch, emb in zip(self.branches, embeddings)]
        stacked = torch.stack(branch_outs, dim=0)

        if self.mode == "average":
            return stacked.mean(dim=0)
        elif self.mode == "weighted":
            w = torch.softmax(self.weights, dim=0).view(-1, 1, 1)
            return (stacked * w).sum(dim=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
