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
        "ppg": {"encoder": "Papagei", "embed_dim": 512, "enabled": True},
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
        "ppg": {"encoder": "Papagei", "embed_dim": 512, "enabled": True},
    }
    registry = ModalityRegistry(config)
    assert registry.get_embed_dim("video") == 768
    assert registry.get_embed_dim("ppg") == 512


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


from src.encoders.papagei import PapageiEncoder


def test_papagei_loads():
    encoder = PapageiEncoder()
    assert encoder.embed_dim == 512


def test_papagei_forward_shape():
    encoder = PapageiEncoder()
    encoder.freeze()
    x = torch.randn(2, 1250, 1)  # 10s of PPG at 125Hz
    with torch.no_grad():
        out = encoder(x)
    assert out.shape == (2, 512)


from src.encoders.extract import EmbeddingExtractor


def test_embedding_cache_structure(tmp_path):
    extractor = EmbeddingExtractor(
        data_dir="data/egoemotion_raw",
        output_dir=str(tmp_path / "embeddings"),
        task_times_path="data/egoemotion_raw/task_times.npy",
    )
    path = extractor.get_cache_path("video_mae_v2", "005", 0)
    assert "video_mae_v2" in str(path)
    assert "005" in str(path)
    assert "segment_0000" in str(path)


def test_embedding_save_load(tmp_path):
    extractor = EmbeddingExtractor(
        data_dir="data/egoemotion_raw",
        output_dir=str(tmp_path / "embeddings"),
        task_times_path="data/egoemotion_raw/task_times.npy",
    )
    emb = torch.randn(5, 768)
    meta = {"task_name": "test", "subject_id": "005"}
    extractor.save_embedding("test_enc", "005", 0, emb, meta, config_hash="abc123")
    assert extractor.is_cached("test_enc", "005", 0)

    loaded = extractor.load_embedding("test_enc", "005", 0)
    assert torch.allclose(loaded["embedding"], emb)
    assert loaded["config_hash"] == "abc123"
