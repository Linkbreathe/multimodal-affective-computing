"""SAFFE-style bottleneck fusion for frozen encoder outputs.

Concatenates all modality projections, compresses through a narrow bottleneck,
then expands back. The bottleneck forces selective extraction of complementary
information across modalities, preventing the model from relying on a single
dominant modality.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class BottleneckFusion(BaseFusionModule):
    """Bottleneck MLP fusion: concat -> compress -> expand.

    The bottleneck_ratio controls the compression factor:
        0.5 -> 128d bottleneck (moderate)
        0.25 -> 64d bottleneck (aggressive)

    Uses LayerNorm (no BatchNorm) for batch_size=1 compatibility.
    """

    def __init__(
        self,
        d_common: int = 256,
        num_modalities: int = 3,
        bottleneck_ratio: float = 0.5,
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_out or d_common)
        d_concat = d_common * num_modalities
        d_bottleneck = int(d_common * bottleneck_ratio)

        self.fusion = nn.Sequential(
            nn.Linear(d_concat, d_bottleneck),
            nn.LayerNorm(d_bottleneck),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_bottleneck, self.d_out),
        )

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        concat = torch.cat(embeddings, dim=-1)
        return self.fusion(concat)
