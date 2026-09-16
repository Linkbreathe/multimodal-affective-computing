"""Mid fusion: per-modality MLP + concatenate + shared MLP."""
from __future__ import annotations

import torch
import torch.nn as nn

from mac.fusion.base import BaseFusionModule


class MidFusion(BaseFusionModule):
    """Refine each modality independently before a shared cross-modal MLP."""

    def __init__(
        self,
        d_common: int = 256,
        modality_ids: list[str] | None = None,
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        modality_ids = modality_ids or ["video", "eye_tracking", "ppg"]
        super().__init__(d_common=d_common, d_out=d_out or d_common)
        self.modality_mlps = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(d_common, d_common),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for mod in modality_ids
        })
        d_in = d_common * len(modality_ids)
        self.cross_modal = nn.Sequential(
            nn.Linear(d_in, d_common * 2),
            nn.LayerNorm(d_common * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_common * 2, self.d_out),
        )

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # The first stage lets each modality clean up its representation before
        # cross-modal information is introduced.
        refined = [self.modality_mlps[m](e) for m, e in zip(modality_ids, embeddings)]
        concat = torch.cat(refined, dim=-1)
        return self.cross_modal(concat)
