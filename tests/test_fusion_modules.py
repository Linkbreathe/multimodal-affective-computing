import pytest
import torch
from src.fusion.base import BaseFusionModule
from src.fusion.projector import ModalityProjector
from src.fusion.early import EarlyFusion
from src.fusion.mid import MidFusion

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


@pytest.mark.parametrize(
    "fusion,embeddings,modality_ids",
    [
        (
            EarlyFusion(d_common=32, num_modalities=2, dropout=0.1),
            [torch.randn(1, 32), torch.randn(1, 32)],
            ["video", "ppg"],
        ),
        (
            MidFusion(d_common=32, modality_ids=["video", "ppg"], dropout=0.1),
            [torch.randn(1, 32), torch.randn(1, 32)],
            ["video", "ppg"],
        ),
    ],
)
def test_mlp_fusions_accept_single_sample_batches_in_train_mode(fusion, embeddings, modality_ids):
    fusion.train()
    out = fusion(embeddings, modality_ids=modality_ids)
    assert out.shape == (1, fusion.d_out)


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


@pytest.mark.skipif(LateFusion is None, reason="src.fusion.late not yet implemented")
def test_late_fusion_branches_follow_modality_names_not_input_order():
    torch.manual_seed(7)
    fusion = LateFusion(
        d_common=32,
        modality_ids=["video", "eye"],
        mode="weighted",
        dropout=0.0,
    )
    fusion.eval()
    video = torch.randn(3, 32)
    eye = torch.randn(3, 32)

    out_original = fusion([video, eye], modality_ids=["video", "eye"])
    out_reordered = fusion([eye, video], modality_ids=["eye", "video"])

    assert torch.allclose(out_original, out_reordered, atol=1e-6)


# --- TMC Fusion Tests ---

try:
    from src.fusion.tmc import TMCFusion
except ImportError:
    TMCFusion = None


@pytest.mark.skipif(TMCFusion is None, reason="src.fusion.tmc not yet implemented")
def test_tmc_fusion_branches_follow_modality_names_not_input_order():
    torch.manual_seed(11)
    fusion = TMCFusion(
        d_common=16,
        num_classes=3,
        modality_ids=["video", "eye"],
        dropout=0.0,
    )
    fusion.eval()
    video = torch.randn(2, 16)
    eye = torch.randn(2, 16)

    out_original = fusion([video, eye], modality_ids=["video", "eye"])
    out_reordered = fusion([eye, video], modality_ids=["eye", "video"])

    assert torch.allclose(out_original, out_reordered, atol=1e-6)


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


# --- Custom Topology Fusion Tests (renamed from old MultimodalLegoFusion) ---

try:
    from src.fusion.multimodal_lego import CustomTopologyFusion
except ImportError:
    CustomTopologyFusion = None

@pytest.mark.skipif(CustomTopologyFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_custom_topology_pairwise():
    fusion = CustomTopologyFusion(d_common=256, modality_ids=["video", "eye", "ppg"], topology="pairwise", n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

@pytest.mark.skipif(CustomTopologyFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_custom_topology_hierarchical():
    fusion = CustomTopologyFusion(d_common=256, modality_ids=["video", "eye", "ppg"], topology="hierarchical", n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

@pytest.mark.skipif(CustomTopologyFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_custom_topology_gated():
    fusion = CustomTopologyFusion(d_common=256, modality_ids=["video", "eye", "ppg"], topology="gated", n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)


# --- Multimodal Lego Fusion Tests (paper-faithful implementation) ---

try:
    from src.fusion.multimodal_lego import MultimodalLegoFusion
except ImportError:
    MultimodalLegoFusion = None

LEGO_MODS = ["video", "eye", "ppg"]
LEGO_D = 256
LEGO_LC = 32   # latent channels (smaller for tests)
LEGO_LD = 64   # latent dim


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_merge_sum():
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=LEGO_MODS, mode="merge-sum",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
    )
    embeddings = [torch.randn(4, LEGO_D) for _ in LEGO_MODS]
    out = fusion(embeddings, modality_ids=LEGO_MODS)
    assert out.shape == (4, LEGO_LD)


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_merge_product():
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=LEGO_MODS, mode="merge-product",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
    )
    embeddings = [torch.randn(4, LEGO_D) for _ in LEGO_MODS]
    out = fusion(embeddings, modality_ids=LEGO_MODS)
    assert out.shape == (4, LEGO_LD)


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_merge_mean():
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=LEGO_MODS, mode="merge-mean",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
    )
    embeddings = [torch.randn(4, LEGO_D) for _ in LEGO_MODS]
    out = fusion(embeddings, modality_ids=LEGO_MODS)
    assert out.shape == (4, LEGO_LD)


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_merge_harmonic_2mod():
    mods_2 = ["video", "ppg"]
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=mods_2, mode="merge-harmonic",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
        alpha=0.5,
    )
    embeddings = [torch.randn(4, LEGO_D) for _ in mods_2]
    out = fusion(embeddings, modality_ids=mods_2)
    assert out.shape == (4, LEGO_LD)


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_merge_harmonic_3mod():
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=LEGO_MODS, mode="merge-harmonic",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
    )
    embeddings = [torch.randn(4, LEGO_D) for _ in LEGO_MODS]
    out = fusion(embeddings, modality_ids=LEGO_MODS)
    assert out.shape == (4, LEGO_LD)


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_fuse_stack():
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=LEGO_MODS, mode="fuse-stack",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
    )
    embeddings = [torch.randn(4, LEGO_D) for _ in LEGO_MODS]
    out = fusion(embeddings, modality_ids=LEGO_MODS)
    assert out.shape == (4, LEGO_LD)


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_fuse_weave():
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=LEGO_MODS, mode="fuse-weave",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
    )
    embeddings = [torch.randn(4, LEGO_D) for _ in LEGO_MODS]
    out = fusion(embeddings, modality_ids=LEGO_MODS)
    assert out.shape == (4, LEGO_LD)


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_sequential_input():
    """Test with sequential (3D) embeddings from encoders."""
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=LEGO_MODS, mode="merge-sum",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
    )
    # Different sequence lengths per modality
    embeddings = [
        torch.randn(4, 10, LEGO_D),
        torch.randn(4, 5, LEGO_D),
        torch.randn(4, 1, LEGO_D),
    ]
    out = fusion(embeddings, modality_ids=LEGO_MODS)
    assert out.shape == (4, LEGO_LD)


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_no_frequency_domain():
    """Test with frequency_domain=False (spatial-only mode)."""
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=LEGO_MODS, mode="merge-sum",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
        frequency_domain=False,
    )
    embeddings = [torch.randn(4, LEGO_D) for _ in LEGO_MODS]
    out = fusion(embeddings, modality_ids=LEGO_MODS)
    assert out.shape == (4, LEGO_LD)


@pytest.mark.skipif(MultimodalLegoFusion is None, reason="src.fusion.multimodal_lego not yet implemented")
def test_lego_gradient_flow():
    """Verify gradients propagate through the fusion module."""
    fusion = MultimodalLegoFusion(
        d_common=LEGO_D, modality_ids=LEGO_MODS, mode="merge-sum",
        latent_channels=LEGO_LC, latent_dim=LEGO_LD, depth=2, heads=4, dim_head=32,
    )
    embeddings = [torch.randn(4, LEGO_D, requires_grad=True) for _ in LEGO_MODS]
    out = fusion(embeddings, modality_ids=LEGO_MODS)
    loss = out.sum()
    loss.backward()
    # Check gradients flow to inputs
    for emb in embeddings:
        assert emb.grad is not None
    # Check gradients flow to learnable latents
    for block in fusion.blocks.values():
        assert block.latent.grad is not None
