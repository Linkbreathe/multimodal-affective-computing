"""VideoMAE V2 frozen encoder wrapper.

Uses the official OpenGVLab VideoMAE V2 base model pretrained on
UnlabeledHybrid-1M (arXiv 2303.16727, CVPR 2023). This is V2, which uses
dual-masking pre-training and a custom ViT implementation loaded via
``trust_remote_code=True``.

Model: OpenGVLab/VideoMAEv2-Base (86M params, embed_dim=768, tubelet_size=2)
"""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModel

from src.encoders.base import BaseEncoder


class VideoMAEV2Encoder(BaseEncoder):
    """Wraps OpenGVLab VideoMAE V2 Base for embedding extraction.

    Input: [B, 3, 16, 224, 224] (16-frame clips, C-first)
    Output: [B, 768] (mean-pooled patch embeddings)

    The V2 model uses ``use_mean_pooling=True`` and ``num_classes=0``,
    so it returns a mean-pooled [B, 768] tensor directly.
    """

    MODEL_NAME = "OpenGVLab/VideoMAEv2-Base"

    def __init__(self) -> None:
        super().__init__(embed_dim=768)
        config = AutoConfig.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME, config=config, trust_remote_code=True,
        )
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # V2 expects [B, C, T, H, W] which matches our convention.
        # If input arrives as [B, T, C, H, W] (T-first), permute.
        if x.dim() == 5 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4)
        # V2 returns a plain tensor [B, 768] (mean-pooled internally).
        return self.model(x)
