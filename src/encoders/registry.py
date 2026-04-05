"""Config-driven modality registry."""
from __future__ import annotations

from importlib import import_module
from typing import Any


class ModalityRegistry:
    """Manages which modalities are enabled and their configurations."""

    ENCODER_REGISTRY = {
        "videomaev2": ("src.encoders.video_mae", "VideoMAEV2Encoder"),
        "patchtst": ("src.encoders.patchtst", "PatchTSTEncoder"),
        "inceptiontime": ("src.encoders.inceptiontime", "InceptionTimeGazeEncoder"),
        "papagei": ("src.encoders.papagei", "PapageiEncoder"),
        "pulseppg": ("src.encoders.pulse_ppg", "PulsePPGEncoder"),
        "eegpt": ("src.encoders.eegpt", "EEGPTEncoder"),
        "reve": ("src.encoders.reve", "ReveEncoder"),
    }
    EMBEDDING_DIR_REGISTRY = {
        ("video", "videomaev2"): "video_mae_v2",
        ("eye_tracking", "patchtst"): "patchtst_eye",
        ("eye_tracking", "inceptiontime"): "inceptiontime",
        ("ppg", "papagei"): "papagei_ppg",
        ("ppg", "pulseppg"): "pulseppg_ppg",
        ("eeg", "eegpt"): "eegpt_eeg",
        ("eeg", "reve"): "reve_eeg",
    }

    def __init__(self, modality_config: dict[str, dict[str, Any]]) -> None:
        self._config = modality_config

    @staticmethod
    def _canonicalize_encoder_name(encoder_name: str) -> str:
        return "".join(ch for ch in encoder_name.lower() if ch.isalnum())

    def get_enabled_modalities(self) -> list[str]:
        return [k for k, v in self._config.items() if v.get("enabled", False)]

    def get_embed_dim(self, modality: str) -> int:
        return self._config[modality]["embed_dim"]

    def get_encoder_name(self, modality: str) -> str:
        return self._config[modality]["encoder"]

    def get_encoder_key(self, modality: str) -> str:
        encoder_name = self.get_encoder_name(modality)
        encoder_key = self._canonicalize_encoder_name(encoder_name)
        if encoder_key not in self.ENCODER_REGISTRY:
            raise KeyError(f"Unknown encoder '{encoder_name}' for modality '{modality}'")
        return encoder_key

    def get_encoder_class(self, modality: str) -> type:
        module_name, class_name = self.ENCODER_REGISTRY[self.get_encoder_key(modality)]
        module = import_module(module_name)
        return getattr(module, class_name)

    def get_embedding_dir_name(self, modality: str) -> str:
        key = (modality, self.get_encoder_key(modality))
        if key not in self.EMBEDDING_DIR_REGISTRY:
            raise KeyError(
                f"No embedding directory registered for modality '{modality}' and encoder "
                f"'{self.get_encoder_name(modality)}'"
            )
        return self.EMBEDDING_DIR_REGISTRY[key]

    def get_all_embed_dims(self) -> dict[str, int]:
        return {m: self.get_embed_dim(m) for m in self.get_enabled_modalities()}
