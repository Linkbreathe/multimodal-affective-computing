"""Papagei frozen encoder wrapper for PPG signals.

NOTE: This is a placeholder implementation. Replace with actual Papagei
model loading once the package is installed. The placeholder uses a simple
1D CNN to produce embeddings of the correct shape.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.encoders.base import BaseEncoder


class PapageiEncoder(BaseEncoder):
    """Wraps Papagei foundation model for PPG embedding extraction.

    Input: [B, T, 1] PPG signal at 125Hz
    Output: [B, 768]

    Currently a placeholder — replace with actual Papagei when available.
    """

    def __init__(self) -> None:
        super().__init__(embed_dim=768)
        # Placeholder: simple 1D CNN that produces correct output shape
        # Replace with actual Papagei model loading when available
        self._placeholder = True
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=5, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=3, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(256, 768)
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 1] -> [B, 1, T] for Conv1d
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder(x)  # [B, 256, 1]
        h = h.squeeze(-1)     # [B, 256]
        return self.proj(h)   # [B, 768]
