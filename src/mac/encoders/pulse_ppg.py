"""Pulse-PPG encoder wrapper for PPG signals with optional fine-tuning.

Uses the Pulse-PPG model (ResNet1D with morphology-aware contrastive learning).
Repo: https://github.com/maxxu05/pulseppg

Architecture: 1D ResNet (base_filters=128, 12 blocks, 28.5M params)
Input: [B, 1, T] single-channel PPG (channels-first, 125Hz)
Output: [B, 512] embeddings (max-pooled over temporal dim)

Block structure (12 BasicBlocks grouped by filter stage):
  - Stage 0: blocks 0-3  (128 filters, 1.4M params)  — low-level temporal
  - Stage 1: blocks 4-7  (256 filters, 5.4M params)  — mid-level features
  - Stage 2: blocks 8-11 (512 filters, 21.6M params) — high-level, task-specific

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

    @property
    def n_blocks(self) -> int:
        """Number of residual blocks in the ResNet1D backbone."""
        return self.MODEL_CONFIG["n_block"]

    def unfreeze(self, from_layer: int | None = None) -> None:
        """Unfreeze ResNet1D blocks for fine-tuning.

        Args:
            from_layer: Block index (0-11). Blocks >= ``from_layer`` are
                unfrozen.  If ``None``, unfreeze all parameters.

        Note:
            BatchNorm layers stay in eval mode — use :meth:`selective_train`
            instead of ``train()`` to avoid degenerate BN stats at BS=1.
        """
        if from_layer is None:
            for param in self.parameters():
                param.requires_grad = True
            return

        if not (0 <= from_layer < self.n_blocks):
            raise ValueError(
                f"from_layer must be in [0, {self.n_blocks}), got {from_layer}"
            )

        for i in range(from_layer, self.n_blocks):
            for param in self.model.basicblock_list[i].parameters():
                param.requires_grad = True

        log.info(
            f"PulsePPG: unfroze blocks {from_layer}-{self.n_blocks - 1} "
            f"({sum(p.numel() for p in self.parameters() if p.requires_grad):,} "
            f"trainable params)"
        )

    def get_layer_groups(self) -> list[dict]:
        """Return parameter groups for layer-wise LR decay.

        Groups:
          - ``first_block``: stem conv+bn (always frozen, lr_scale=0)
          - ``blocks_0_3``: stage 0 (lr_scale=0.02)
          - ``blocks_4_7``: stage 1 (lr_scale=0.05)
          - ``blocks_8_9``: stage 2 lower (lr_scale=0.05)
          - ``blocks_10_11``: stage 2 upper (lr_scale=0.1)
        """
        groups = [
            {
                "name": "ppg_first_block",
                "params": [
                    p for n, p in self.model.named_parameters()
                    if n.startswith("first_block") or n.startswith("instnorm")
                ],
                "lr_scale": 0.0,
            },
        ]

        block_ranges = [
            ("ppg_blocks_0_3", range(0, 4), 0.02),
            ("ppg_blocks_4_7", range(4, 8), 0.05),
            ("ppg_blocks_8_9", range(8, 10), 0.05),
            ("ppg_blocks_10_11", range(10, 12), 0.1),
        ]
        for name, block_indices, lr_scale in block_ranges:
            params = []
            for i in block_indices:
                params.extend(self.model.basicblock_list[i].parameters())
            groups.append({"name": name, "params": params, "lr_scale": lr_scale})

        return groups

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
