"""VideoMAE V2 frozen encoder wrapper."""
from __future__ import annotations

import torch
from transformers import VideoMAEModel

from src.encoders.base import BaseEncoder


class VideoMAEV2Encoder(BaseEncoder):
    """Wraps HuggingFace VideoMAE V2 for embedding extraction.

    Input: [B, 3, 16, 224, 224] (16-frame clips)
    Output: [B, 768] (CLS token embedding)
    """

    MODEL_NAME = "MCG-NJU/videomae-base"

    def __init__(self) -> None:
        super().__init__(embed_dim=768)
        self.model = VideoMAEModel.from_pretrained(self.MODEL_NAME)
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> VideoMAE expects [B, T, C, H, W]
        if x.dim() == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]
