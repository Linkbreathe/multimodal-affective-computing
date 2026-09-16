"""Early fusion: concatenate + MLP."""
from __future__ import annotations

import torch
import torch.nn as nn

from mac.fusion.base import BaseFusionModule


class EarlyFusion(BaseFusionModule):
    """Concatenate all modality embeddings, then apply one shared MLP.

    This is the simplest baseline: interactions between modalities are learned
    only after concatenation, and every modality must contribute a fixed-width
    projected vector.
    """

    def __init__(
        self,
        d_common: int = 256,
        num_modalities: int = 3,
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_out or d_common)
        d_in = d_common * num_modalities
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_common * 2),
            nn.LayerNorm(d_common * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_common * 2, d_common),
            nn.LayerNorm(d_common),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_common, self.d_out),
        )

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # ``embeddings`` are already projected to d_common by the caller.
        concat = torch.cat(embeddings, dim=-1)
        return self.mlp(concat)
