from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.segments import SegmentExtractor
from src.encoders.registry import ModalityRegistry


def test_registry_resolves_ecgfounder():
    config = {
        "ecg": {"encoder": "ECGFounder", "embed_dim": 1024, "enabled": True},
    }
    registry = ModalityRegistry(config)

    encoder_cls = registry.get_encoder_class("ecg")

    assert encoder_cls.__name__ == "ECGFounderEncoder"
    assert registry.get_embedding_dir_name("ecg") == "ecg_founder"
    assert registry.get_embed_dim("ecg") == 1024


def test_preprocess_ecg_90hz_chunk_to_ecgfounder_input():
    from src.data.ecg_preprocessing import preprocess_ecgfounder_segment

    t = np.linspace(0.0, 10.0, 900, endpoint=False)
    ecg = np.sin(2 * np.pi * 1.3 * t) + 0.05 * np.sin(2 * np.pi * 12.0 * t)

    out = preprocess_ecgfounder_segment(ecg, source_fs=90, target_fs=500)

    assert out.shape == (1, 5000)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert abs(float(out.mean())) < 1e-5
    assert 0.99 < float(out.std()) < 1.01


def test_segment_extractor_loads_ecg_segment(ego_data_dir):
    extractor = SegmentExtractor(
        data_dir=str(ego_data_dir),
        task_times_path=str(ego_data_dir / "task_times.npy"),
    )
    seg = extractor.get_segments("005")[0]

    ecg = extractor.load_ecg_segment("005", seg["start_idx"], seg["start_idx"] + 900)

    assert ecg.ndim == 2
    assert ecg.shape == (900, 1)
    assert np.isfinite(ecg).all()


@pytest.mark.external
def test_ecgfounder_loads_real_weights_and_extracts_real_ecg_features():
    weights = Path("weights/ecgfounder/1_lead_ECGFounder.pth")
    if not weights.exists():
        pytest.skip(f"ECGFounder weights not found at {weights}")

    from src.data.ecg_preprocessing import preprocess_ecgfounder_segment
    from src.encoders.ecgfounder import ECGFounderEncoder

    raw = np.load("data/datasets/egoemotion_raw/005/ecg_90fps.npy")[:900]
    ecg = preprocess_ecgfounder_segment(raw, source_fs=90, target_fs=500)
    x = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0)

    encoder = ECGFounderEncoder(weights_path=str(weights))
    encoder.freeze()
    with torch.no_grad():
        features = encoder(x)

    assert features.shape == (1, 1024)
    assert torch.isfinite(features).all()
