"""InceptionTime encoder for gaze-only eye tracking."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.encoders.base import BaseEncoder


class InceptionModule(nn.Module):
    """Multi-scale 1D inception block for temporal gaze dynamics."""

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (9, 19, 39),
    ) -> None:
        super().__init__()
        bottleneck_channels = max(1, nb_filters // 4)
        self.bottleneck = nn.Conv1d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            bias=False,
        )
        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    nb_filters,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=False,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm1d(nb_filters * 4)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottlenecked = self.bottleneck(x)
        branches = [conv(bottlenecked) for conv in self.conv_branches]
        branches.append(self.maxpool_branch(x))
        out = torch.cat(branches, dim=1)
        out = self.norm(out)
        return self.activation(out)


class InceptionResidualBlock(nn.Module):
    """Two inception modules plus a residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (9, 19, 39),
    ) -> None:
        super().__init__()
        out_channels = nb_filters * 4
        self.inception_modules = nn.ModuleList(
            [
                InceptionModule(
                    in_channels=in_channels,
                    nb_filters=nb_filters,
                    kernel_sizes=kernel_sizes,
                ),
                InceptionModule(
                    in_channels=out_channels,
                    nb_filters=nb_filters,
                    kernel_sizes=kernel_sizes,
                ),
            ]
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = x
        for module in self.inception_modules:
            out = module(out)
        return self.activation(out + residual)


class InceptionTimeGazeEncoder(BaseEncoder):
    """InceptionTime encoder for raw gaze trajectories.

    Input: ``[B, 2, T]`` channels-first gaze trajectories in radians.
    Output: ``[B, 128]`` global average pooled embeddings.
    """

    def __init__(
        self,
        num_channels: int = 2,
        embed_dim: int = 128,
        nb_filters: int = 32,
        kernel_sizes: tuple[int, int, int] = (9, 19, 39),
    ) -> None:
        super().__init__(embed_dim=embed_dim)
        if embed_dim != nb_filters * 4:
            raise ValueError(
                f"embed_dim must equal 4 * nb_filters ({nb_filters * 4}), got {embed_dim}"
            )
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.nb_filters = nb_filters
        self.residual_blocks = nn.ModuleList(
            [
                InceptionResidualBlock(
                    in_channels=num_channels,
                    nb_filters=nb_filters,
                    kernel_sizes=kernel_sizes,
                ),
                InceptionResidualBlock(
                    in_channels=embed_dim,
                    nb_filters=nb_filters,
                    kernel_sizes=kernel_sizes,
                ),
                InceptionResidualBlock(
                    in_channels=embed_dim,
                    nb_filters=nb_filters,
                    kernel_sizes=kernel_sizes,
                ),
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.num_channels:
            raise ValueError(
                "Expected channels-first [B, 2, T] input for InceptionTime gaze encoding."
            )

        out = x
        for block in self.residual_blocks:
            out = block(out)
        out = self.global_pool(out).squeeze(-1)
        return out
