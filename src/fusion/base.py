"""Abstract base class for fusion modules."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFusionModule(ABC, nn.Module):
    def __init__(self, d_common: int, d_out: int | None = None) -> None:
        super().__init__()
        self.d_common = d_common
        self.d_out = d_out or d_common

    @abstractmethod
    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        ...
