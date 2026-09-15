"""ECGFounder frozen encoder wrapper for 1-lead ECG signals.

Architecture: ECGFounder Net1D with 7 residual stages and SE blocks.
Input: [B, 1, 5000] 1-lead ECG, 10 seconds at 500 Hz, z-score normalized.
Output: [B, 1024] global pooled deep features.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from mac.encoders.base import BaseEncoder
from mac.encoders.ecgfounder_model import Net1D

log = logging.getLogger(__name__)


class ECGFounderEncoder(BaseEncoder):
    """Wraps ECGFounder 1-lead checkpoint for ECG feature extraction."""

    MODEL_CONFIG = {
        "in_channels": 1,
        "base_filters": 64,
        "ratio": 1,
        "filter_list": [64, 160, 160, 400, 400, 1024, 1024],
        "m_blocks_list": [2, 2, 2, 3, 3, 4, 4],
        "kernel_size": 16,
        "stride": 2,
        "groups_width": 16,
        "use_bn": False,
        "use_do": False,
        "n_classes": 150,
        "return_features": True,
    }

    def __init__(
        self,
        weights_path: str = "weights/ecgfounder/1_lead_ECGFounder.pth",
        random_init: bool = False,
    ) -> None:
        super().__init__(embed_dim=1024)
        self.model = Net1D(**self.MODEL_CONFIG)
        self._load_pretrained(Path(weights_path), random_init=random_init)
        self.freeze()

    def _load_pretrained(self, weights_path: Path, *, random_init: bool = False) -> None:
        if not weights_path.exists():
            if random_init:
                log.warning(
                    "ECGFounder weights not found at %s; continuing with random init",
                    weights_path,
                )
                return
            raise FileNotFoundError(
                f"ECGFounder weights not found at {weights_path}. "
                "Download 1_lead_ECGFounder.pth into weights/ecgfounder/."
            )

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            k.removeprefix("module."): v
            for k, v in state_dict.items()
            if not k.removeprefix("module.").startswith("dense.")
        }
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"dense.weight", "dense.bias"}
        unexpected = [k for k in unexpected if not k.startswith("dense.")]
        if unexpected:
            raise RuntimeError(f"Unexpected ECGFounder checkpoint keys: {unexpected}")
        disallowed_missing = sorted(set(missing) - allowed_missing)
        if disallowed_missing:
            raise RuntimeError(
                "ECGFounder checkpoint missing backbone keys: "
                f"{disallowed_missing[:10]}"
            )
        log.info("Loaded ECGFounder weights from %s", weights_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.permute(0, 2, 1)

        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError("Expected ECGFounder input shaped [B, 1, T].")

        output = self.model(x)
        if not isinstance(output, tuple):
            raise RuntimeError("ECGFounder model was expected to return features.")
        _, features = output
        return features
