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


class TMCTaskHead(nn.Module):
    """Task head for TMC evidential fusion.

    TMC outputs a (B, K) belief vector (already class-level), so:
    - emotion_logits = log(belief + eps) — feeds into CE loss
    - soft_logits = log(belief + eps) — feeds into KL loss
    - vad_pred = Linear(K, num_vad)(belief)
    """

    def __init__(self, num_emotions: int = 9, num_vad: int = 3) -> None:
        super().__init__()
        self.vad_head = nn.Linear(num_emotions, num_vad)

    def forward(self, belief: torch.Tensor) -> dict[str, torch.Tensor]:
        log_belief = torch.log(belief + 1e-8)
        return {
            "emotion_logits": log_belief,
            "soft_logits": log_belief,
            "vad_pred": self.vad_head(belief),
        }
