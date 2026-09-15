"""REVE encoder wrapper for EEG signals (frozen or fine-tunable).

Uses the REVE pretrained transformer for EEG embedding extraction.
Architecture: 22-layer Transformer with 4D Fourier positional encoding,
RMSNorm, GEGLU activation, and attention pooling.
Model internalized from EEG-FM-Bench.

Input:  [B, 60, T] — 60 SEED-V channels (10-10 montage, excl. CB1/CB2)
Output: [B, 512]   — attention-pooled embedding

Preprocessing aligned with EEG-FM-Bench (reve_adapter):
  - Z-score normalization (per-channel, per-sample)
  - Electrode position lookup from pos_bank
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.encoders.base import BaseEncoder
from src.encoders.reve_model import Reve, ReveModelArgs
from src.encoders.reve_pos_bank import RevePositionBank
from src.models.lora import (
    count_lora_parameters,
    get_lora_parameters,
    get_lora_state_dict,
    inject_lora as inject_lora_modules,
    is_lora_parameter,
    load_lora_state_dict,
    resolve_lora_targets,
)

log = logging.getLogger(__name__)


class ReveEncoder(BaseEncoder):
    """REVE encoder for EEG signals (frozen inference or fine-tuning).

    Input:  [B, 60, T] (60 SEED-V channels, 10-10 montage excl. CB1/CB2)
    Output: [B, 512] (attention-pooled)

    Args:
        channels: List of channel names for the input data. Default: SEEDV 60-channel layout.
        weights_path: Path to pretrained REVE model checkpoint (.safetensors or .pt).
        pos_bank_path: Path to pretrained position bank checkpoint.
        finetune: If True, skips auto-freeze.
        embed_dim: Embedding dimension. Default 512 (matches EEG-FM-Bench).
        depth: Number of transformer layers. Default 22.
        patch_size: Temporal patch size in samples. Default 200 (1s at 200Hz).
        patch_overlap: Overlap between patches. Default 20.
        random_init: If True, allow missing checkpoints.
    """

    # SEED-V 60-channel order (aligned with EEG-FM-Bench, excl. CB1/CB2)
    SEEDV_60_CHANNELS = [
        "FP1", "FPZ", "FP2", "AF3", "AF4",
        "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
        "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
        "O1", "OZ", "O2",
    ]

    def __init__(
        self,
        channels: list[str] | None = None,
        weights_path: str | None = None,
        pos_bank_path: str | None = None,
        finetune: bool = False,
        embed_dim: int = 512,
        depth: int = 22,
        heads: int = 8,
        head_dim: int = 64,
        patch_size: int = 200,
        patch_overlap: int = 20,
        random_init: bool = False,
    ) -> None:
        super().__init__(embed_dim=embed_dim)
        self._finetune = finetune
        self._lora_active = False
        self._lora_modules: list[str] = []

        # Channel configuration
        if channels is None:
            channels = list(self.SEEDV_60_CHANNELS)
        self._channels = [ch.upper() for ch in channels]

        # Position bank — learnable 3D electrode positions
        self.pos_bank = RevePositionBank()

        # Core REVE model
        cfg = ReveModelArgs(
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            head_dim=head_dim,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )
        self.reve = Reve(cfg)

        # Load pretrained weights — pos_bank is always required for REVE
        if pos_bank_path is not None:
            self._load_pos_bank(Path(pos_bank_path), random_init=random_init)
        elif not random_init:
            raise ValueError(
                "pos_bank_path is required for ReveEncoder (electrode 3D positions). "
                "Pass pos_bank_path or set random_init=True."
            )

        if weights_path is not None:
            self._load_pretrained(Path(weights_path), random_init=random_init)
        elif not random_init:
            raise ValueError(
                "weights_path is required for ReveEncoder. "
                "Pass weights_path or set random_init=True."
            )

        # Pre-compute electrode positions for the fixed channel layout
        with torch.no_grad():
            self._pos_template = self.pos_bank(self._channels)  # [n_ch, 3]

        if not finetune:
            self.freeze()

        log.info(
            "ReveEncoder: %d channels, embed_dim=%d, depth=%d, "
            "patch_size=%d, overlap=%d, finetune=%s",
            len(self._channels), embed_dim, depth,
            patch_size, patch_overlap, finetune,
        )

    def _load_pretrained(self, weights_path: Path, *, random_init: bool = False) -> None:
        if not weights_path.exists():
            if random_init:
                log.warning(
                    "REVE weights not found at %s; continuing with random init",
                    weights_path,
                )
                return
            raise FileNotFoundError(f"REVE weights not found at {weights_path}")

        suffix = weights_path.suffix.lower()
        if suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
                state = load_file(str(weights_path))
            except ImportError:
                raise ImportError("safetensors package required for .safetensors checkpoints")
        else:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)

        # Extract reve model weights (remove prefix if present)
        reve_state = {}
        for k, v in state.items():
            for prefix in ("reve.", "model.", "encoder."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            reve_state[k] = v

        missing, unexpected = self.reve.load_state_dict(reve_state, strict=False)
        log.info(
            "Loaded REVE model: %d keys, missing=%s, unexpected=%s",
            len(reve_state), missing, unexpected,
        )

    def _load_pos_bank(self, pos_bank_path: Path, *, random_init: bool = False) -> None:
        if not pos_bank_path.exists():
            if random_init:
                log.warning(
                    "Position bank weights not found at %s; using random init",
                    pos_bank_path,
                )
                return
            raise FileNotFoundError(f"Position bank weights not found at {pos_bank_path}")

        suffix = pos_bank_path.suffix.lower()
        if suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(pos_bank_path))
        else:
            ckpt = torch.load(pos_bank_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)

        # Extract pos_bank weights
        pb_state = {}
        for k, v in state.items():
            for prefix in ("pos_bank.", "position_bank."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            pb_state[k] = v

        self.pos_bank.load_state_dict(pb_state, strict=False)
        log.info("Loaded REVE position bank: %d keys", len(pb_state))

        # Recompute pos template after loading
        with torch.no_grad():
            self._pos_template = self.pos_bank(self._channels)

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _set_lora_trainable(self, enabled: bool = True) -> None:
        for name, param in self.reve.named_parameters():
            if is_lora_parameter(name):
                param.requires_grad = enabled

    def inject_lora(self, lora_config: dict | None) -> list[str]:
        """Inject LoRA into REVE attention projections."""
        if not lora_config or not lora_config.get("use_lora", False):
            return []
        if self._lora_active:
            return list(self._lora_modules)

        target_modules = resolve_lora_targets(
            lora_config.get("lora_target_modules", ["reve_default"])
        )
        self.reve, injected_modules = inject_lora_modules(
            model=self.reve,
            target_modules=target_modules,
            r=lora_config.get("lora_r", 8),
            lora_alpha=lora_config.get("lora_alpha", 16),
            lora_dropout=lora_config.get("lora_dropout", 0.0),
        )
        if not injected_modules:
            raise ValueError(
                f"No REVE modules matched LoRA targets {target_modules}"
            )

        self._lora_active = True
        self._lora_modules = injected_modules
        self.freeze()

        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        log.info(
            "Injected LoRA into ReveEncoder: modules=%d, total_params=%d, "
            "lora_params=%d, trainable_params=%d",
            len(injected_modules), total_params,
            count_lora_parameters(self.reve), trainable_params,
        )
        return list(self._lora_modules)

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        return get_lora_state_dict(self.reve)

    def load_lora_state_dict(
        self,
        lora_state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        if not self._lora_active:
            raise RuntimeError("Inject LoRA before loading LoRA weights.")
        missing, unexpected = load_lora_state_dict(
            self.reve, lora_state_dict, strict=strict,
        )
        self._set_lora_trainable(True)
        return missing, unexpected

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score normalization (aligned with EEG-FM-Bench reve_adapter).

        EEG-FM-Bench REVE preprocessing:
          - Z-score normalization per channel per sample
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std
        return x

    def _get_positions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get electrode positions for the current channel layout.

        Returns: [B, n_channels, 3]
        """
        pos = self._pos_template.to(device)  # [n_ch, 3]
        return pos.unsqueeze(0).expand(batch_size, -1, -1)  # [B, n_ch, 3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 60, T] EEG input (60 SEED-V channels)

        Returns:
            [B, 512] attention-pooled embedding
        """
        x = self._preprocess(x)                                # [B, C, T]
        pos = self._get_positions(x.size(0), x.device)         # [B, C, 3]
        z = self.reve(x, pos)                                   # [B, C, num_patches, E]
        return self.reve.attention_pooling(z)                   # [B, E]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-patch features WITHOUT attention pooling.

        Returns: [B, C * num_patches, 512]
        """
        x = self._preprocess(x)
        pos = self._get_positions(x.size(0), x.device)
        z = self.reve(x, pos)                                   # [B, C, num_patches, E]
        b, c, h, e = z.shape
        return z.reshape(b, c * h, e)                           # [B, C*num_patches, E]

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze the encoder and keep LoRA trainable when active."""
        for param in self.parameters():
            param.requires_grad = False
        if self._lora_active:
            self._set_lora_trainable(True)
        self.eval()

    def train(self, mode: bool = True) -> "ReveEncoder":
        """Override train() so reve stays in eval when frozen."""
        super().train(mode)
        if not any(p.requires_grad for p in self.reve.parameters()):
            self.reve.eval()
        return self

    def unfreeze(self, from_layer: int | None = None) -> None:
        """Unfreeze parameters for fine-tuning.

        Args:
            from_layer: If given, unfreeze transformer layers from this index
                onwards. If None, unfreeze everything.
        """
        if self._lora_active:
            self.freeze()
            return

        if from_layer is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for i, layer_pair in enumerate(self.reve.transformer.layers):
                if i >= from_layer:
                    for param in layer_pair.parameters():
                        param.requires_grad = True
            # Always unfreeze final LN and cls token
            for param in self.reve.ln.parameters():
                param.requires_grad = True
            self.reve.cls_query_token.requires_grad = True

    def get_layer_groups(self) -> list[dict]:
        if self._lora_active:
            return [
                {
                    "name": "encoder_lora",
                    "params": get_lora_parameters(self.reve),
                    "lr_scale": 1.0,
                }
            ]

        reve = self.reve
        # Embeddings + position encoding — low LR
        embed_params = (
            list(reve.to_patch_embedding.parameters())
            + list(reve.fourier4d.parameters())
            + list(reve.mlp4d.parameters())
            + list(reve.ln.parameters())
            + [reve.cls_query_token]
        )

        n_layers = len(reve.transformer.layers)
        mid = n_layers // 2  # 11 for 22 layers

        return [
            {
                "name": "encoder_embed",
                "params": embed_params,
                "lr_scale": 0.1,
            },
            {
                "name": "encoder_early",
                "params": [
                    p for layer_pair in reve.transformer.layers[:mid]
                    for layer in layer_pair for p in layer.parameters()
                ],
                "lr_scale": 0.1,
            },
            {
                "name": "encoder_late",
                "params": [
                    p for layer_pair in reve.transformer.layers[mid:]
                    for layer in layer_pair for p in layer.parameters()
                ],
                "lr_scale": 0.5,
            },
        ]
