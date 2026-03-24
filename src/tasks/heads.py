"""Multi-task prediction heads for emotion recognition."""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(self, d_fused: int = 256, num_emotions: int = 9, num_vad: int = 3) -> None:
        super().__init__()
        self.emotion_head = nn.Linear(d_fused, num_emotions)
        self.soft_head = nn.Linear(d_fused, num_emotions)
        self.vad_head = nn.Linear(d_fused, num_vad)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "emotion_logits": self.emotion_head(x),
            "soft_logits": self.soft_head(x),
            "vad_pred": self.vad_head(x),
        }
