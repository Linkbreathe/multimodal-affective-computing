import pytest
import torch
from src.encoders.base import BaseEncoder
from src.encoders.registry import ModalityRegistry


def test_base_encoder_is_abstract():
    with pytest.raises(TypeError):
        BaseEncoder()


def test_registry_from_config():
    config = {
        "video": {"encoder": "VideoMAEV2", "embed_dim": 768, "enabled": True},
        "ppg": {"encoder": "Papagei", "embed_dim": 768, "enabled": True},
        "eye_tracking": {"encoder": "PatchTST", "embed_dim": 128, "enabled": False},
    }
    registry = ModalityRegistry(config)
    enabled = registry.get_enabled_modalities()
    assert "video" in enabled
    assert "ppg" in enabled
    assert "eye_tracking" not in enabled


def test_registry_embed_dims():
    config = {
        "video": {"encoder": "VideoMAEV2", "embed_dim": 768, "enabled": True},
        "ppg": {"encoder": "Papagei", "embed_dim": 768, "enabled": True},
    }
    registry = ModalityRegistry(config)
    assert registry.get_embed_dim("video") == 768
    assert registry.get_embed_dim("ppg") == 768


def test_registry_get_all_embed_dims():
    config = {
        "video": {"encoder": "VideoMAEV2", "embed_dim": 768, "enabled": True},
        "eye_tracking": {"encoder": "PatchTST", "embed_dim": 128, "enabled": True},
    }
    registry = ModalityRegistry(config)
    dims = registry.get_all_embed_dims()
    assert dims == {"video": 768, "eye_tracking": 128}
