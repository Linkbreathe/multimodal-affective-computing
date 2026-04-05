"""EEGPT two-layer linear probe adapted for mean-pooled embeddings.

Official architecture (per-patch, from EEGPT repo linear_probe_EEGPT_*.py):
    [B, N, 2048] -> Dropout(0.5) -> LinearWithConstraint(2048, 16) -> flatten -> LinearWithConstraint(N*16, C)

Adapted for pre-extracted mean-pooled embeddings:
    [B, 2048] -> Dropout(0.5) -> LinearWithConstraint(2048, 16, max_norm=1) -> LinearWithConstraint(16, C, max_norm=0.25)

The adaptation preserves all key elements: LinearWithConstraint with L2 weight-norm
capping, the 2048->16 bottleneck, specific max_norm values (1.0 and 0.25), and 0.5
dropout rate.
"""
from __future__ import annotations

import torch.nn as nn

from src.models.constrained_layers import LinearWithConstraint


class EEGPTLinearProbe(nn.Module):
    """Official EEGPT two-layer linear probe for mean-pooled embeddings.

    Args:
        num_classes: Number of output classes.
        input_dim: Embedding dimension (2048 for EEGPT large).
        bottleneck_dim: Hidden dimension between the two probe layers.
        dropout: Dropout rate applied before the first probe layer.
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_dim: int = 2048,
        bottleneck_dim: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_probe1 = LinearWithConstraint(
            input_dim, bottleneck_dim, max_norm=1,
        )
        self.linear_probe2 = LinearWithConstraint(
            bottleneck_dim, num_classes, max_norm=0.25,
        )

    def forward(self, x):
        # x: [B, 2048]
        h = self.linear_probe1(self.drop(x))  # [B, 16]
        return self.linear_probe2(h)           # [B, num_classes]
