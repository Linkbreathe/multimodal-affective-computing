"""Weight-norm-constrained layers from the official EEGPT repository.

Source: /home/link/Wei/Models/EEGPT/downstream/Modules/Network/utils.py
Reference: Lawhern et al., "EEGNet" (2018) weight constraint technique.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearWithConstraint(nn.Linear):
    """nn.Linear with per-output-neuron L2 weight norm capping via torch.renorm."""

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)
