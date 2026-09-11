"""VideoMAE V2 frozen encoder wrapper.

Uses the official OpenGVLab VideoMAE V2 base model pretrained on
UnlabeledHybrid-1M (arXiv 2303.16727, CVPR 2023). This is V2, which uses
dual-masking pre-training and a custom ViT implementation loaded via
``trust_remote_code=True``.

Model: OpenGVLab/VideoMAEv2-Base (86M params, embed_dim=768, tubelet_size=2)
"""
from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

from src.encoders.base import BaseEncoder


class VideoMAEV2Encoder(BaseEncoder):
    """Wraps OpenGVLab VideoMAE V2 Base for embedding extraction.

    Input:  ``[B, C, T, H, W]`` — always C-first, i.e. ``[B, 3, 16, 224, 224]``.
    Output: ``[B, 768]`` (mean-pooled patch embeddings).

    The V2 model uses ``use_mean_pooling=True`` and ``num_classes=0``,
    so it returns a mean-pooled ``[B, 768]`` tensor directly.
    """

    MODEL_NAME = "OpenGVLab/VideoMAEv2-Base"

    def __init__(self) -> None:
        super().__init__(embed_dim=768)
        config = AutoConfig.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True,
        )
        # Recent Transformers versions construct remote-code models on the
        # meta device inside ``from_pretrained``.  VideoMAEv2's constructor
        # calls ``Tensor.item()``, which is invalid for meta tensors.  Build
        # the official architecture normally, then load the same Hub state
        # dict explicitly.
        self.model = AutoModel.from_config(config, trust_remote_code=True)
        weights_path = hf_hub_download(
            repo_id=self.MODEL_NAME,
            filename="model.safetensors",
        )
        state_dict = load_file(weights_path, device="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  Input must be ``[B, 3, 16, 224, 224]`` (C-first)."""
        if x.dim() != 5 or x.shape[1] != 3:
            raise ValueError(
                f"Expected [B, 3, T, H, W] (C-first) input, got shape {tuple(x.shape)}.  "
                "Use src.data.video_transforms.normalize_clip to produce the correct layout."
            )
        return self.model(x)
