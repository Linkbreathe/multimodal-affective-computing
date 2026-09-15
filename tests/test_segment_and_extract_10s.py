from pathlib import Path

from scripts.segment_and_extract_10s import resolve_encoder_dirs
from src.encoders.registry import ModalityRegistry


def test_10s_extraction_writes_manifest_hash_to_embedding_payloads():
    text = Path("scripts/segment_and_extract_10s.py").read_text()

    assert "compute_manifest_hash" in text
    assert '"manifest_hash": manifest_hash' in text


def test_resolve_encoder_dirs_includes_enabled_ecg_for_all():
    registry = ModalityRegistry({
        "video": {"encoder": "VideoMAEV2", "embed_dim": 768, "enabled": True},
        "eye_tracking": {"encoder": "inceptiontime", "embed_dim": 128, "enabled": True},
        "ppg": {"encoder": "Papagei", "embed_dim": 512, "enabled": True},
        "ecg": {"encoder": "ECGFounder", "embed_dim": 1024, "enabled": True},
    })

    encoders = resolve_encoder_dirs(registry, "all")

    assert encoders == ["video_mae_v2", "inceptiontime", "papagei_ppg", "ecg_founder"]
