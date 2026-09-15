"""Avg-pool classification head for EEGPT fine-tuning.

Matches the 'avg_pool' head type from EEG-FM-Bench:
    [B, n_patches, 2048] -> AdaptiveAvgPool1d -> MLP -> [B, num_classes]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AvgPoolClassificationHead(nn.Module):
    """Avg-pool + MLP classification head for per-patch EEGPT features.

    Args:
        input_dim: Per-patch feature dimension (2048 for EEGPT large).
        num_classes: Number of output classes.
        hidden_dims: List of hidden layer dimensions. Default [128].
        dropout: Dropout rate between layers.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        num_classes: int = 5,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, n_patches, D] per-patch features from EEGPT encoder.
        Returns:
            [B, num_classes] logits.
        """
        # Global average pooling over temporal patches
        x = x.mean(dim=1)  # [B, D]
        return self.mlp(x)  # [B, num_classes]
