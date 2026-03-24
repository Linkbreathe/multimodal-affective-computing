"""Abstract base class for frozen encoders."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Base class for all frozen modality encoders."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
