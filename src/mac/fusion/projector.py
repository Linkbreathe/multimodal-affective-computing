"""Per-modality linear projection to common dimension."""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    def __init__(self, embed_dims: dict[str, int], d_common: int) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {mod: nn.Linear(dim, d_common) for mod, dim in embed_dims.items()}
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {mod: self.projectors[mod](x) for mod, x in inputs.items()}
