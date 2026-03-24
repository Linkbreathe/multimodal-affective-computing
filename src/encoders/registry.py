"""Config-driven modality registry."""
from __future__ import annotations

from typing import Any


class ModalityRegistry:
    """Manages which modalities are enabled and their configurations."""

    def __init__(self, modality_config: dict[str, dict[str, Any]]) -> None:
        self._config = modality_config

    def get_enabled_modalities(self) -> list[str]:
        return [k for k, v in self._config.items() if v.get("enabled", False)]

    def get_embed_dim(self, modality: str) -> int:
        return self._config[modality]["embed_dim"]

    def get_encoder_name(self, modality: str) -> str:
        return self._config[modality]["encoder"]

    def get_all_embed_dims(self) -> dict[str, int]:
        return {m: self.get_embed_dim(m) for m in self.get_enabled_modalities()}
