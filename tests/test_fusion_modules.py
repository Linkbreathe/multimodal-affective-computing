import pytest
import torch
from src.fusion.base import BaseFusionModule
from src.fusion.projector import ModalityProjector

def test_base_fusion_is_abstract():
    with pytest.raises(TypeError):
        BaseFusionModule(d_common=256)

def test_projector():
    embed_dims = {"video": 768, "eye": 128, "ppg": 768}
    proj = ModalityProjector(embed_dims=embed_dims, d_common=256)
    inputs = {
        "video": torch.randn(4, 768),
        "eye": torch.randn(4, 128),
        "ppg": torch.randn(4, 768),
    }
    outputs = proj(inputs)
    for mod in embed_dims:
        assert outputs[mod].shape == (4, 256)

def test_projector_sequential():
    embed_dims = {"video": 768, "eye": 128}
    proj = ModalityProjector(embed_dims=embed_dims, d_common=256)
    inputs = {
        "video": torch.randn(4, 10, 768),
        "eye": torch.randn(4, 20, 128),
    }
    outputs = proj(inputs)
    assert outputs["video"].shape == (4, 10, 256)
    assert outputs["eye"].shape == (4, 20, 256)
