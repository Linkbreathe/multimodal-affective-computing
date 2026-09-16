"""Per-modality linear projection to a common fusion dimension.

Foundation encoders do not produce equally sized vectors (and sequence
embeddings may have different feature widths).  The projector standardizes
only the width; it does not mix modalities.  Fusion happens in the modules
that consume its output.
"""
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
        # Preserve modality names so masks and missing-modality tokens can be
        # aligned after projection.
        return {mod: self.projectors[mod](x) for mod, x in inputs.items()}
