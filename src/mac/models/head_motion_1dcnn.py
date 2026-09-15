"""Trainable 1D CNN encoder for Relax head-motion windows."""
from __future__ import annotations

import torch
import torch.nn as nn


class HeadMotion1DCNN(nn.Module):
    """Encode one 10-second head-motion window into a dense embedding.

    Input shape is ``[batch, channels, time]``. The default 10 channels are:
    relative position x/y/z, relative yaw/pitch/roll, velocity x/y/z, and
    angular velocity magnitude in deg/s.
    """

    def __init__(
        self,
        input_channels: int = 10,
        embedding_dim: int = 128,
        conv_channels: tuple[int, int] = (32, 64),
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, conv_channels[0], kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels[0]),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels[1]),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[1], embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"HeadMotion1DCNN expects [B,C,T], got {tuple(x.shape)}")
        return self.proj(self.net(x))
