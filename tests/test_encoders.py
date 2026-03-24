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


from src.encoders.video_mae import VideoMAEV2Encoder


def test_video_mae_loads():
    encoder = VideoMAEV2Encoder()
    assert encoder.embed_dim == 768


def test_video_mae_forward_shape():
    encoder = VideoMAEV2Encoder()
    encoder.freeze()
    # 16-frame clip at 224x224
    dummy = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        out = encoder(dummy)
    assert out.shape == (1, 768)


from src.encoders.patchtst import PatchTSTEncoder


def test_patchtst_forward_shape():
    encoder = PatchTSTEncoder(
        num_channels=4, patch_len=45, stride=22,
        d_model=128, n_heads=4, n_layers=3, seq_len=900,
    )
    x = torch.randn(2, 900, 4)
    out = encoder(x)
    expected_patches = (900 - 45) // 22 + 1
    assert out.shape[0] == 2
    assert out.shape[2] == 128
    assert out.shape[1] == expected_patches


def test_patchtst_mean_pool():
    encoder = PatchTSTEncoder(
        num_channels=4, patch_len=45, stride=22,
        d_model=128, n_heads=4, n_layers=3, seq_len=900,
    )
    x = torch.randn(2, 900, 4)
    out = encoder(x)
    pooled = out.mean(dim=1)
    assert pooled.shape == (2, 128)
