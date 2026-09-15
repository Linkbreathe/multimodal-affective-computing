import pytest
import pandas as pd
import torch

from mac.data.egoemotion.manifest import (
    Ego10sManifestError,
    compute_manifest_hash,
    load_egoemotion_10s_by_subject,
)


def _manifest(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "subject": "005",
        "global_seq": 0,
        "task_name": "video_neutral",
        "chunk_idx_in_task": 0,
        "emotion_label": 4,
        "emotion_name": "Neutral",
        "soft_label": "0,0,0,0,1,0,0,0,0",
        "valence": 0.0,
        "arousal": 0.0,
        "dominance": 0.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _save_embedding(
    root,
    encoder: str,
    subject: str,
    seq: int,
    *,
    label: int = 4,
    emotion: str = "Neutral",
    manifest_hash: str | None = None,
    embedding: torch.Tensor | None = None,
) -> None:
    path = root / encoder / subject / f"segment_{seq:04d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding": embedding if embedding is not None else torch.randn(1, 8),
        "subject": subject,
        "segment_idx": seq,
        "label": label,
        "emotion": emotion,
    }
    if manifest_hash is not None:
        payload["manifest_hash"] = manifest_hash
    torch.save(payload, path)


def test_loader_detects_embedding_label_mismatch(tmp_path):
    manifest = _manifest([{"global_seq": 0, "emotion_label": 4}])
    _save_embedding(tmp_path, "video_mae_v2", "005", 0, label=1)

    with pytest.raises(Ego10sManifestError, match="label mismatch"):
        load_egoemotion_10s_by_subject(
            tmp_path,
            ["video_mae_v2"],
            ["video"],
            manifest,
        )


def test_loader_reports_and_limits_missing_modalities(tmp_path):
    manifest = _manifest([
        {"global_seq": 0},
        {"global_seq": 1},
    ])
    _save_embedding(tmp_path, "video_mae_v2", "005", 0)
    _save_embedding(tmp_path, "papagei_ppg", "005", 0)
    _save_embedding(tmp_path, "papagei_ppg", "005", 1)

    data_by_subject, report = load_egoemotion_10s_by_subject(
        tmp_path,
        ["video_mae_v2", "papagei_ppg"],
        ["video", "ppg"],
        manifest,
        max_missing_fraction=0.6,
    )

    assert list(data_by_subject) == ["005"]
    assert len(data_by_subject["005"]) == 1
    assert report.manifest_rows == 2
    assert report.loaded_rows == 1
    assert report.dropped_rows == 1
    assert report.missing_by_encoder == {"video_mae_v2": 1}
    assert report.subjects_in_manifest == 1
    assert report.subjects_loaded == 1

    with pytest.raises(Ego10sManifestError, match="missing embedding fraction"):
        load_egoemotion_10s_by_subject(
            tmp_path,
            ["video_mae_v2", "papagei_ppg"],
            ["video", "ppg"],
            manifest,
            max_missing_fraction=0.0,
        )


def test_loader_validates_manifest_hash_when_required(tmp_path):
    manifest = _manifest([{"global_seq": 0}])
    manifest_hash = compute_manifest_hash(manifest)
    _save_embedding(tmp_path, "video_mae_v2", "005", 0, manifest_hash=manifest_hash)

    data_by_subject, report = load_egoemotion_10s_by_subject(
        tmp_path,
        ["video_mae_v2"],
        ["video"],
        manifest,
        require_manifest_hash=True,
    )

    assert len(data_by_subject["005"]) == 1
    assert report.manifest_hash == manifest_hash

    _save_embedding(tmp_path, "video_mae_v2", "005", 0, manifest_hash="stale")
    with pytest.raises(Ego10sManifestError, match="manifest_hash mismatch"):
        load_egoemotion_10s_by_subject(
            tmp_path,
            ["video_mae_v2"],
            ["video"],
            manifest,
            require_manifest_hash=True,
        )
