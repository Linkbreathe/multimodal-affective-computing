"""VideoMAE frozen encoder wrapper.

Note: HuggingFace hosts VideoMAE v1 (MCG-NJU/videomae-base). The V2 paper
(arXiv 2303.16727) describes a dual-masking pre-training strategy but uses the
same ViT architecture. No official V2 weights are on HuggingFace, so we use
the V1 pretrained weights with the same architecture.
"""
from __future__ import annotations

import torch
from transformers import VideoMAEModel

from src.encoders.base import BaseEncoder


class VideoMAEV2Encoder(BaseEncoder):
    """Wraps HuggingFace VideoMAE for embedding extraction.

    Input: [B, 3, 16, 224, 224] (16-frame clips, C-first)
    Output: [B, 768] (mean-pooled patch embeddings)

    Note: VideoMAE does NOT have a CLS token. The last_hidden_state contains
    all patch embeddings [B, num_patches, 768]. We mean-pool over patches
    to get a single vector per clip.
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
        # Mean-pool over all patch tokens (no CLS token in VideoMAE)
        return outputs.last_hidden_state.mean(dim=1)
