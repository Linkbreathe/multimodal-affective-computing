"""Pulse-PPG frozen encoder wrapper for PPG signals.

Uses the Pulse-PPG model (ResNet1D with morphology-aware contrastive learning).
Repo: https://github.com/maxxu05/pulseppg

Architecture: 1D ResNet (base_filters=128, 12 blocks, 28.5M params)
Input: [B, 1, T] single-channel PPG (channels-first, 125Hz)
Output: [B, 512] embeddings (max-pooled over temporal dim)

The model uses max-pooling over the temporal dimension, so it handles
variable-length inputs. Our 10s segments at 125Hz = 1250 samples work
out of the box despite pre-training on 4-minute windows.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.encoders.base import BaseEncoder

log = logging.getLogger(__name__)

# Path to cloned pulseppg repo (sibling of our project's parent)
_PULSEPPG_REPO = Path(__file__).resolve().parents[3] / "pulseppg"


def _load_net_class():
    """Import Net (ResNet1D) from the pulseppg repo."""
    if str(_PULSEPPG_REPO) not in sys.path:
        sys.path.insert(0, str(_PULSEPPG_REPO))
    try:
        from pulseppg.nets.ResNet1D.ResNet1D_Net import Net
        return Net
    except ImportError:
        raise ImportError(
            f"Could not import Net from pulseppg repo at {_PULSEPPG_REPO}. "
            "Clone the repo: git clone https://github.com/maxxu05/pulseppg "
            f"into {_PULSEPPG_REPO.parent}"
        )


class PulsePPGEncoder(BaseEncoder):
    """Wraps Pulse-PPG (ResNet1D) for PPG embedding extraction.

    Input: [B, 1, T] single-channel PPG at 125Hz (channels-first)
    Output: [B, 512] embeddings

    Args:
        weights_path: Path to pretrained weights (.pt or .pth file).
            Default: weights/pulseppg/pulseppg.pt
    """

    # Config from pulseppg/experiments/configs/PulsePPG_expconfigs.py
    MODEL_CONFIG = {
        "in_channels": 1,
        "base_filters": 128,
        "kernel_size": 11,
        "stride": 2,
        "groups": 1,
        "n_block": 12,
        "finalpool": "max",
    }

    def __init__(
        self,
        weights_path: str = "weights/pulseppg/pulseppg.pt",
    ) -> None:
        super().__init__(embed_dim=512)

        Net = _load_net_class()
        self.model = Net(**self.MODEL_CONFIG)

        # Load pretrained weights
        weights_file = Path(weights_path)
        if weights_file.exists():
            ckpt = torch.load(weights_file, map_location="cpu", weights_only=False)
            # Handle various checkpoint formats:
            # Official pulseppg checkpoint: {"net": state_dict, "optimizer": ..., ...}
            if isinstance(ckpt, dict) and "net" in ckpt:
                state_dict = ckpt["net"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

            # Remove 'module.' prefix from DataParallel training
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=True)
            log.info(f"Loaded Pulse-PPG weights from {weights_path}")
        else:
            log.warning(
                f"Pulse-PPG weights not found at {weights_path} -- "
                "model will produce random embeddings! "
                "Download from: https://github.com/maxxu05/pulseppg"
            )

        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure channels-first: [B, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)          # [B, T] -> [B, 1, T]
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.permute(0, 2, 1)      # [B, T, 1] -> [B, 1, T]

        out = self.model(x)

        # Handle different output formats:
        # If tuple/list, take the first element (embedding head)
        if isinstance(out, (tuple, list)):
            out = out[0]

        # If 3D (B, T', D), pool to (B, D)
        if out.dim() == 3:
            out = out.mean(dim=1)

        return out  # [B, 512]
