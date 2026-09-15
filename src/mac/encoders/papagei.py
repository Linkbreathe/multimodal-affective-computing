"""Papagei frozen encoder wrapper for PPG signals.

Uses the actual PaPaGei-S model (ResNet1DMoE) from Nokia Bell Labs.
Paper: https://arxiv.org/abs/2410.20542 (ICLR 2025)
Repo: https://github.com/Nokia-Bell-Labs/papagei-foundation-model

Architecture: 1D ResNet with Mixture of Experts (18 blocks, 3 experts)
Input: [B, 1, T] single-channel PPG at 125Hz, z-score normalized
Output: [B, 512] projection head embeddings

The model returns 4 outputs: (class_emb, moe1, moe2, backbone_emb)
We use class_emb (out[0]) — the dense projection head output — matching
the official Papagei feature extraction and all published benchmarks.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

from mac.encoders.base import BaseEncoder

# Add papagei repo to path for model imports
_PAPAGEI_REPO = Path(__file__).resolve().parents[4] / "papagei-foundation-model"


def _load_resnet1d_moe():
    """Import ResNet1DMoE from the papagei repo."""
    if str(_PAPAGEI_REPO) not in sys.path:
        sys.path.insert(0, str(_PAPAGEI_REPO))
        sys.path.insert(0, str(_PAPAGEI_REPO / "src"))
    from models.resnet import ResNet1DMoE
    return ResNet1DMoE


class PapageiEncoder(BaseEncoder):
    """Wraps PaPaGei-S (ResNet1DMoE) for PPG embedding extraction.

    Input: [B, 1, T] single-channel PPG at 125Hz (channels-first, Conv1d format)
    Output: [B, 512] embeddings

    Args:
        weights_path: Path to pretrained weights (.pt file).
            Default: weights/papagei/papagei_s.pt
        use_backbone: If True, return backbone embeddings (out[3], dim=512).
            If False (default), return projection head output (out[0], dim=512),
            matching the official Papagei feature extraction pipeline.
    """

    # PaPaGei-S config from the official repo
    MODEL_CONFIG = {
        "base_filters": 32,
        "kernel_size": 3,
        "stride": 2,
        "groups": 1,
        "n_block": 18,
        "n_classes": 512,
        "n_experts": 3,
    }

    def __init__(
        self,
        weights_path: str = "weights/papagei/papagei_s.pt",
        use_backbone: bool = False,
    ) -> None:
        super().__init__(embed_dim=512)
        self.use_backbone = use_backbone

        ResNet1DMoE = _load_resnet1d_moe()
        self.model = ResNet1DMoE(
            in_channels=1,
            base_filters=self.MODEL_CONFIG["base_filters"],
            kernel_size=self.MODEL_CONFIG["kernel_size"],
            stride=self.MODEL_CONFIG["stride"],
            groups=self.MODEL_CONFIG["groups"],
            n_block=self.MODEL_CONFIG["n_block"],
            n_classes=self.MODEL_CONFIG["n_classes"],
            n_experts=self.MODEL_CONFIG["n_experts"],
        )

        # Load pretrained weights
        weights_file = Path(weights_path)
        if weights_file.exists():
            ckpt = torch.load(weights_file, map_location="cpu", weights_only=False)
            # Remove 'module.' prefix from DataParallel training
            if any(k.startswith("module.") for k in ckpt.keys()):
                ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            self.model.load_state_dict(ckpt)
        else:
            log.warning(
                f"Papagei weights not found at {weights_path} — "
                "model will produce random embeddings!"
            )

        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure channels-first: [B, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)          # [B, T] -> [B, 1, T]
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.permute(0, 2, 1)      # [B, T, 1] -> [B, 1, T]

        # ResNet1DMoE returns: (class_emb, moe1, moe2, backbone_emb)
        outputs = self.model(x)

        if self.use_backbone:
            return outputs[3]  # [B, 512] — backbone features before projection
        else:
            return outputs[0]  # [B, 512] — projection head output
