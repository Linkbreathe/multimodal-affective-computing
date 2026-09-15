"""EEGPT encoder wrapper for EEG signals (frozen or fine-tunable).

Uses the EEGPT pretrained transformer (large variant) for EEG embedding extraction.
Architecture: ViT-style transformer with patch embedding over EEG channels/time.
Model internalized from EEG-FM-Bench (no external EEGPT repo dependency).

Input:  [B, 60, T] — 60 SEED-V channels (10-10 montage, excl. CB1/CB2), at 256 Hz
Output: [B, 2048]  — mean-pooled over temporal patches of 2048-dim features

Preprocessing aligned with EEG-FM-Bench:
  - Channel selection from SEED-V 60-channel layout
  - Amplitude scaling (default 0.001, uV → mV)
"""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from mac.encoders.base import BaseEncoder
from mac.encoders.eegpt_model import CHANNEL_DICT, EEGTransformer
from mac.models.lora import (
    count_lora_parameters,
    get_lora_parameters,
    get_lora_state_dict,
    inject_lora as inject_lora_modules,
    is_lora_parameter,
    load_lora_state_dict,
    resolve_lora_targets,
)

log = logging.getLogger(__name__)


class EEGPTEncoder(BaseEncoder):
    """EEGPT encoder for EEG signals (frozen inference or fine-tuning).

    Input:  [B, 60, window_samples] (60 SEED-V channels, 10-10 montage excl. CB1/CB2)
    Output: [B, 2048] (mean-pooled over temporal patches)

    Args:
        channels: List of channel names to use. If None, uses all 58
            pretrained channels. Must be a subset of PRETRAINED_CHANNELS.
        window_samples: Temporal input length in samples. Must be divisible
            by patch_size (64). Default 2560 (= 10s at 256 Hz).
        weights_path: Path to pretrained checkpoint. Must be provided explicitly.
        finetune: If True, enables dropout regularization and skips auto-freeze.
        patch_stride: Temporal patch stride. None = patch_size (non-overlapping).
            Use 32 for overlapping patches (matches EEG-FM-Bench fine-tuning).
        drop_rate: Dropout rate for transformer blocks (0.0 for frozen, 0.1 for fine-tuning).
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        scale: Amplitude scaling factor applied to input data.
            Default 0.001 (uV → mV), matching EEG-FM-Bench adapter.
        random_init: If True, allow missing checkpoints and keep random weights.
    """

    PATCH_SIZE = 64

    # The 58 channel names used in pretraining (CHANNEL_DICT minus AF7, AF8, PO5, PO6)
    PRETRAINED_CHANNELS = [
        "FP1", "FPZ", "FP2", "AF3", "AF4",
        "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POZ", "PO4", "PO8",
        "O1", "OZ", "O2",
    ]

    # Canonical SEED-V 60-channel order (10-10 montage, aligned with EEG-FM-Bench).
    # CB1 and CB2 are excluded to match EEG-FM-Bench's channel definition.
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
        window_samples: int = 2560,
        weights_path: str | None = None,
        finetune: bool = False,
        patch_stride: int | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        scale: float = 0.001,
        random_init: bool = False,
    ) -> None:
        super().__init__(embed_dim=2048)
        self._finetune = finetune
        self._lora_active = False
        self._lora_modules: list[str] = []
        self._scale = scale

        # Resolve channel configuration
        if channels is None:
            channels = list(self.PRETRAINED_CHANNELS)
        else:
            channels = [ch.upper() for ch in channels]
            pretrained_set = set(self.PRETRAINED_CHANNELS)
            invalid = [ch for ch in channels if ch not in pretrained_set]
            if invalid:
                raise ValueError(
                    f"Channels {invalid} are not in EEGPT's pretrained set. "
                    f"Valid channels: {self.PRETRAINED_CHANNELS}"
                )
        self._channels = channels
        n_channels = len(channels)

        if window_samples % self.PATCH_SIZE != 0:
            raise ValueError(
                f"window_samples ({window_samples}) must be divisible by "
                f"patch_size ({self.PATCH_SIZE})"
            )
        self._window_samples = window_samples

        # Deterministic channel selection from SEED-V's 60-channel layout.
        seedv_upper = [ch.upper() for ch in self.SEEDV_60_CHANNELS]
        chan_set = set(channels)
        chan_select = [i for i, ch in enumerate(seedv_upper) if ch in chan_set]
        if len(chan_select) != n_channels:
            raise ValueError(
                f"Expected {n_channels} channels in SEEDV_60_CHANNELS, "
                f"found {len(chan_select)}"
            )
        # Canonicalize channel order to match SEEDV layout
        channels = [seedv_upper[i] for i in chan_select]
        self._channels = channels
        self.register_buffer("chan_select", torch.tensor(chan_select, dtype=torch.long))

        # Core EEGPT encoder (internalized from EEG-FM-Bench)
        self.target_encoder = EEGTransformer(
            img_size=[n_channels, window_samples],
            patch_size=self.PATCH_SIZE,
            patch_stride=patch_stride,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        # Prepare channel IDs — uses the canonicalized channel order
        self.register_buffer(
            "chan_ids",
            self.target_encoder.prepare_chan_ids(channels),
        )

        # Load pretrained weights
        if weights_path is not None:
            self._load_pretrained(Path(weights_path), random_init=random_init)
        elif not random_init:
            log.warning(
                "No weights_path provided for EEGPTEncoder. "
                "Pass weights_path or set random_init=True."
            )

        if not finetune:
            self.freeze()

        log.info(
            "EEGPTEncoder: %d channels, %d samples, patch_stride=%s, "
            "finetune=%s, embed_dim=%d, scale=%s",
            n_channels, window_samples, patch_stride, finetune, self.embed_dim,
            self._scale,
        )

    def _load_pretrained(self, weights_path: Path, *, random_init: bool = False) -> None:
        if not weights_path.exists():
            if random_init:
                log.warning(
                    "EEGPT weights not found at %s; continuing with random initialization "
                    "because random_init=True",
                    weights_path,
                )
                return
            raise FileNotFoundError(
                f"EEGPT weights not found at {weights_path}. "
                "Pass random_init=True only if random initialization is intentional."
            )
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = {}
        prefix = "target_encoder."
        for k, v in ckpt["state_dict"].items():
            if k.startswith(prefix):
                state[k[len(prefix):]] = v
        missing, unexpected = self.target_encoder.load_state_dict(state, strict=False)
        log.info(
            "Loaded EEGPT target_encoder: %d keys, missing=%s, unexpected=%s",
            len(state), missing, unexpected,
        )

    def _set_lora_trainable(self, enabled: bool = True) -> None:
        for name, param in self.target_encoder.named_parameters():
            if is_lora_parameter(name):
                param.requires_grad = enabled

    def inject_lora(self, lora_config: dict | None) -> list[str]:
        """Inject LoRA into EEGPT attention and FFN projections."""
        if not lora_config or not lora_config.get("use_lora", False):
            return []
        if self._lora_active:
            return list(self._lora_modules)

        target_modules = resolve_lora_targets(
            lora_config.get("lora_target_modules", ["default"])
        )
        self.target_encoder, injected_modules = inject_lora_modules(
            model=self.target_encoder,
            target_modules=target_modules,
            r=lora_config.get("lora_r", 8),
            lora_alpha=lora_config.get("lora_alpha", 16),
            lora_dropout=lora_config.get("lora_dropout", 0.0),
        )
        if not injected_modules:
            raise ValueError(
                f"No EEGPT modules matched LoRA targets {target_modules}"
            )

        self._lora_active = True
        self._lora_modules = injected_modules
        self.freeze()

        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        log.info(
            "Injected LoRA into EEGPTEncoder: modules=%d, total_params=%d, "
            "lora_params=%d, trainable_params=%d",
            len(injected_modules),
            total_params,
            count_lora_parameters(self.target_encoder),
            trainable_params,
        )
        return list(self._lora_modules)

    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        return get_lora_state_dict(self.target_encoder)

    def load_lora_state_dict(
        self,
        lora_state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        if not self._lora_active:
            raise RuntimeError("Inject LoRA before loading LoRA weights.")
        missing, unexpected = load_lora_state_dict(
            self.target_encoder,
            lora_state_dict,
            strict=strict,
        )
        self._set_lora_trainable(True)
        return missing, unexpected

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Channel selection + amplitude scaling (aligned with EEG-FM-Bench).

        EEG-FM-Bench preprocessing:
          1. Channel selection from dataset montage
          2. Amplitude scaling (scale=0.001, uV → mV)
        """
        x = x[:, self.chan_select, :]                          # [B, n_ch, T]
        x = x * self._scale                                    # amplitude scaling
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 60, window_samples]
        x = self._preprocess(x)                                # [B, n_ch, window_samples]
        z = self.target_encoder(x, self.chan_ids)               # [B, n_patches, 4, 512]
        z = z.flatten(2)                                        # [B, n_patches, 2048]
        return z.mean(dim=1)                                    # [B, 2048]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-patch features WITHOUT mean-pooling.

        Returns: [B, n_patches, 2048]
        """
        x = self._preprocess(x)
        z = self.target_encoder(x, self.chan_ids)               # [B, n_patches, 4, 512]
        return z.flatten(2)                                     # [B, n_patches, 2048]

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze the base encoder and keep LoRA trainable when active."""
        for param in self.parameters():
            param.requires_grad = False
        if self._lora_active:
            self._set_lora_trainable(True)
        self.eval()

    def train(self, mode: bool = True) -> "EEGPTEncoder":
        """Override train() so target_encoder stays in eval when frozen."""
        super().train(mode)
        if not any(p.requires_grad for p in self.target_encoder.parameters()):
            self.target_encoder.eval()
        return self

    def unfreeze(self, from_layer: int | None = None) -> None:
        """Unfreeze parameters for fine-tuning.

        Args:
            from_layer: If given, unfreeze transformer blocks from this index
                onwards (plus the final norm). If None, unfreeze everything.
        """
        if self._lora_active:
            self.freeze()
            return

        if from_layer is None:
            for param in self.parameters():
                param.requires_grad = True
        else:
            for i, block in enumerate(self.target_encoder.blocks):
                if i >= from_layer:
                    for param in block.parameters():
                        param.requires_grad = True
            for param in self.target_encoder.norm.parameters():
                param.requires_grad = True

    def get_layer_groups(self) -> list[dict]:
        if self._lora_active:
            return [
                {
                    "name": "encoder_lora",
                    "params": get_lora_parameters(self.target_encoder),
                    "lr_scale": 1.0,
                }
            ]

        enc = self.target_encoder
        embed_params = (
            list(enc.patch_embed.parameters())
            + list(enc.chan_embed.parameters())
            + [enc.summary_token]
        )
        return [
            {
                "name": "encoder_embed",
                "params": embed_params,
                "lr_scale": 0.1,
            },
            {
                "name": "encoder_early",
                "params": [
                    p for b in enc.blocks[:4] for p in b.parameters()
                ],
                "lr_scale": 0.1,
            },
            {
                "name": "encoder_late",
                "params": [
                    p for b in enc.blocks[4:] for p in b.parameters()
                ],
                "lr_scale": 0.5,
            },
            {
                "name": "encoder_norm",
                "params": list(enc.norm.parameters()),
                "lr_scale": 1.0,
            },
        ]
