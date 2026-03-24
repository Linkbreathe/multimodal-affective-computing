import pytest
import torch
from src.fusion.base import BaseFusionModule
from src.fusion.projector import ModalityProjector
from src.fusion.early import EarlyFusion

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

def test_early_fusion_shape():
    fusion = EarlyFusion(d_common=256, num_modalities=3, dropout=0.1)
    embeddings = [torch.randn(4, 256) for _ in range(3)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

def test_early_fusion_variable_modalities():
    fusion = EarlyFusion(d_common=256, num_modalities=2, dropout=0.1)
    embeddings = [torch.randn(4, 256) for _ in range(2)]
    out = fusion(embeddings, modality_ids=["video", "ppg"])
    assert out.shape == (4, 256)


# --- Mid Fusion Tests ---

try:
    from src.fusion.mid import MidFusion
except ImportError:
    MidFusion = None

@pytest.mark.skipif(MidFusion is None, reason="src.fusion.mid not yet implemented")
def test_mid_fusion_shape():
    fusion = MidFusion(d_common=256, modality_ids=["video", "eye", "ppg"], dropout=0.1)
    embeddings = [torch.randn(4, 256) for _ in range(3)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)


# --- Late Fusion Tests ---

try:
    from src.fusion.late import LateFusion
except ImportError:
    LateFusion = None

@pytest.mark.skipif(LateFusion is None, reason="src.fusion.late not yet implemented")
def test_late_fusion_avg():
    fusion = LateFusion(d_common=256, num_modalities=3, mode="average")
    embeddings = [torch.randn(4, 256) for _ in range(3)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

@pytest.mark.skipif(LateFusion is None, reason="src.fusion.late not yet implemented")
def test_late_fusion_weighted():
    fusion = LateFusion(d_common=256, num_modalities=3, mode="weighted")
    embeddings = [torch.randn(4, 256) for _ in range(3)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)


# --- Perceiver IO Fusion Tests ---

from src.fusion.perceiver_io import PerceiverIOFusion

def test_perceiver_io_pooled():
    fusion = PerceiverIOFusion(d_common=256, n_latents=32, n_layers=2, n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

def test_perceiver_io_sequential():
    fusion = PerceiverIOFusion(d_common=256, n_latents=32, n_layers=2, n_heads=4)
    embeddings = [torch.randn(4, 10, 256), torch.randn(4, 20, 256), torch.randn(4, 1, 256)]
    masks = [torch.ones(4, 10, dtype=torch.bool), torch.ones(4, 20, dtype=torch.bool), torch.ones(4, 1, dtype=torch.bool)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"], masks=masks)
    assert out.shape == (4, 256)


# --- Q-Former Fusion Tests ---

try:
    from src.fusion.qformer import QFormerFusion
except ImportError:
    QFormerFusion = None

@pytest.mark.skipif(QFormerFusion is None, reason="src.fusion.qformer not yet implemented")
def test_qformer_pooled():
    fusion = QFormerFusion(d_common=256, n_queries=16, n_layers=2, n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

@pytest.mark.skipif(QFormerFusion is None, reason="src.fusion.qformer not yet implemented")
def test_qformer_sequential():
    fusion = QFormerFusion(d_common=256, n_queries=16, n_layers=2, n_heads=4)
    embeddings = [torch.randn(4, 10, 256), torch.randn(4, 20, 256)]
    masks = [torch.ones(4, 10, dtype=torch.bool), torch.ones(4, 20, dtype=torch.bool)]
    out = fusion(embeddings, modality_ids=["video", "eye"], masks=masks)
    assert out.shape == (4, 256)


# --- HEALNet Fusion Tests ---

try:
    from src.fusion.healnet import HEALNetFusion
except ImportError:
    HEALNetFusion = None

@pytest.mark.skipif(HEALNetFusion is None, reason="src.fusion.healnet not yet implemented")
def test_healnet_pooled():
    fusion = HEALNetFusion(d_common=256, memory_size=16, n_layers=2, n_heads=4, num_modalities=3)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

@pytest.mark.skipif(HEALNetFusion is None, reason="src.fusion.healnet not yet implemented")
def test_healnet_missing_modality():
    fusion = HEALNetFusion(d_common=256, memory_size=16, n_layers=2, n_heads=4, num_modalities=3)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "ppg"])
    assert out.shape == (4, 256)


# --- Multimodal Lego Fusion Tests ---

try:
    from src.fusion.multimodal_lego import MultimodalLegoFusion
except ImportError:
    MultimodalLegoFusion = None

@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_topology_a():
    fusion = MultimodalLegoFusion(d_common=256, modality_ids=["video", "eye", "ppg"], topology="pairwise", n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_topology_b():
    fusion = MultimodalLegoFusion(d_common=256, modality_ids=["video", "eye", "ppg"], topology="hierarchical", n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_topology_c():
    fusion = MultimodalLegoFusion(d_common=256, modality_ids=["video", "eye", "ppg"], topology="gated", n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)
