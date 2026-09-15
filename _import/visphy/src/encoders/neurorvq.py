"""Thin, validated adapter around the official NeuroRVQ foundation models.

NeuroRVQ is consumed from an explicit checkout rather than vendored into this
repository.  This keeps the upstream code and checkpoint provenance visible.
The adapter exposes a stable frozen embedding: concatenate the four returned
patch-token branches and mean-pool over electrode/time tokens.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import nn
import yaml


_CONFLICTING_UPSTREAM_MODULES = (
    "NeuroRVQ_modules",
    "RVQ",
    "norm_ema_quantizer",
)
_EXPECTED_MISSING = {
    f"fc_norm_{branch}.{parameter}"
    for branch in range(1, 5)
    for parameter in ("weight", "bias")
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    text = str(path)
    sys.path.insert(0, text)
    try:
        yield
    finally:
        try:
            sys.path.remove(text)
        except ValueError:
            pass


def _load_module(path: Path, name: str, *, dependency_path: Path | None = None) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot construct an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    context = _temporary_sys_path(dependency_path) if dependency_path is not None else _temporary_sys_path(path.parent)
    with context:
        spec.loader.exec_module(module)
    return module


def _upstream_paths(repo: Path, modality: str) -> tuple[Path, Path, Path]:
    normalized = modality.upper()
    if normalized not in {"EEG", "ECG"}:
        raise ValueError(f"NeuroRVQ modality must be EEG or ECG, got {modality!r}")
    model_dir = repo / f"NeuroRVQ_{normalized}"
    model_path = model_dir / "NeuroRVQ.py"
    flags_path = repo / "flags" / f"NeuroRVQ_{normalized}_v1.yml"
    channel_path = repo / "inference" / "modules" / f"NeuroRVQ_{normalized}_FM_inference_modules.py"
    for path in (model_path, flags_path, channel_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required official NeuroRVQ file is missing: {path}")
    return model_path, flags_path, channel_path


def _unexpected_key_is_pretraining_only(key: str) -> bool:
    if key in {"mask_token", "norm_pre.weight", "norm_pre.bias"}:
        return True
    if key.startswith("head_pre_") and key.rsplit(".", 1)[-1] in {"weight", "bias"}:
        return True
    return False


class NeuroRVQFoundationEncoder(nn.Module):
    """Frozen EEG or ECG NeuroRVQ encoder with deterministic token pooling."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        modality: str,
        channels: Sequence[str],
    ) -> None:
        super().__init__()
        self.repo_path = Path(repo_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.modality = modality.upper()
        self.channels = tuple(str(channel).lower() for channel in channels)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"NeuroRVQ checkpoint is missing: {self.checkpoint_path}")

        model_path, flags_path, channel_path = _upstream_paths(self.repo_path, self.modality)
        channel_module = _load_module(
            channel_path,
            f"_relax_neurorvq_{self.modality.lower()}_channels",
        )
        self.global_channels = tuple(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in np.asarray(channel_module.ch_names_global).tolist()
        )
        unknown = sorted(set(self.channels) - set(self.global_channels))
        if unknown:
            raise ValueError(
                f"Channels are outside the official NeuroRVQ {self.modality} vocabulary: {unknown}"
            )

        with flags_path.open("r", encoding="utf-8") as handle:
            flags = yaml.safe_load(handle)
        self.patch_size = int(flags["patch_size"])
        self.maximum_patches = int(flags["n_patches"])
        self.branch_dimension = int(flags["embed_dim_second_stage"])
        self.embedding_dimension = 4 * self.branch_dimension

        for module_name in _CONFLICTING_UPSTREAM_MODULES:
            sys.modules.pop(module_name, None)
        model_module = _load_module(
            model_path,
            f"_relax_neurorvq_{self.modality.lower()}_model",
            dependency_path=model_path.parent,
        )
        model_class = model_module.NeuroRVQFM
        self.model = model_class(
            n_patches=self.maximum_patches,
            patch_size=self.patch_size,
            in_chans=int(flags["in_chans_second_stage"]),
            out_chans=int(flags["out_chans_second_stage"]),
            num_classes=0,
            embed_dim=self.branch_dimension,
            depth=int(flags["depth_second_stage"]),
            num_heads=int(flags["num_heads_second_stage"]),
            mlp_ratio=float(flags["mlp_ratio_second_stage"]),
            qkv_bias=bool(flags["qkv_bias_second_stage"]),
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=float(flags["drop_rate_second_stage"]),
            attn_drop_rate=float(flags["attn_drop_rate_second_stage"]),
            drop_path_rate=float(flags["drop_path_rate_second_stage"]),
            init_values=float(flags["init_values_second_stage"]),
            init_scale=float(flags["init_scale_second_stage"]),
            n_global_electrodes=len(self.global_channels),
            use_as_encoder=True,
            vocab_size=int(flags["n_code"]),
            use_for_pretraining=False,
        )

        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or not state:
            raise TypeError("The NeuroRVQ foundation checkpoint is not a non-empty state dictionary")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if set(missing) != _EXPECTED_MISSING:
            raise RuntimeError(f"Unexpected NeuroRVQ missing keys: {missing}")
        invalid_unexpected = [key for key in unexpected if not _unexpected_key_is_pretraining_only(key)]
        if invalid_unexpected:
            raise RuntimeError(f"Unexpected non-pretraining checkpoint keys: {invalid_unexpected}")
        self.load_diagnostics = {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "checkpoint_sha256": _sha256(self.checkpoint_path),
            "checkpoint_bytes": self.checkpoint_path.stat().st_size,
        }
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "NeuroRVQFoundationEncoder":
        # This adapter is intentionally frozen even if a parent module enters train mode.
        super().train(False)
        self.model.eval()
        return self

    def _embedding_indexes(
        self,
        *,
        batch_size: int,
        n_time: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_time > self.maximum_patches:
            raise ValueError(
                f"Input has {n_time} time patches, exceeding the model limit {self.maximum_patches}"
            )
        temporal = torch.arange(
            self.maximum_patches - n_time,
            self.maximum_patches,
            dtype=torch.long,
            device=device,
        ).repeat(len(self.channels))
        spatial_channel_indexes = torch.tensor(
            [self.global_channels.index(channel) for channel in self.channels],
            dtype=torch.long,
            device=device,
        )
        spatial = torch.repeat_interleave(spatial_channel_indexes, n_time)
        return (
            temporal.unsqueeze(0).expand(batch_size, -1),
            spatial.unsqueeze(0).expand(batch_size, -1),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Encode ``[batch, channels, samples]`` and return one vector/window."""

        if signals.ndim != 3:
            raise ValueError(f"Expected [batch, channels, samples], got {tuple(signals.shape)}")
        batch_size, channels, samples = signals.shape
        if channels != len(self.channels):
            raise ValueError(f"Expected {len(self.channels)} channels, got {channels}")
        if samples % self.patch_size:
            raise ValueError(
                f"Input length {samples} is not divisible by patch size {self.patch_size}"
            )
        n_time = samples // self.patch_size
        if n_time * channels > self.maximum_patches:
            raise ValueError(
                f"Input uses {n_time * channels} electrode/time patches; maximum is {self.maximum_patches}"
            )
        patches = signals.reshape(batch_size, channels, n_time, self.patch_size)
        temporal, spatial = self._embedding_indexes(
            batch_size=batch_size,
            n_time=n_time,
            device=signals.device,
        )
        output = self.model(
            patches,
            temporal,
            spatial,
            return_patch_tokens=True,
        )
        if not isinstance(output, tuple) or len(output) < 4:
            raise RuntimeError("NeuroRVQ did not return its four patch-token branches")
        branches = output[:4]
        expected_tokens = channels * n_time
        for branch in branches:
            if branch.shape != (batch_size, expected_tokens, self.branch_dimension):
                raise RuntimeError(f"Unexpected NeuroRVQ branch shape: {tuple(branch.shape)}")
        embedding = torch.cat(branches, dim=-1).mean(dim=1)
        if embedding.shape != (batch_size, self.embedding_dimension):
            raise RuntimeError(f"Unexpected NeuroRVQ embedding shape: {tuple(embedding.shape)}")
        return embedding


__all__ = ["NeuroRVQFoundationEncoder"]
