from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from real_time_ml.config import ProjectConfig, load_config
from real_time_ml.data.video import load_video_index, sample_window_frames, uniform_clip_frames
from real_time_ml.modeling.video_dcnn import (
    _fit_scaler,
    _transform,
    build_video_sequences,
    load_video_dcnn_model,
    predict_video_dcnn_state,
    train_video_dcnn_model,
)


def _write_p003_style_log(session: Path) -> Path:
    frames = session / "video_frames"
    frames.mkdir(parents=True)
    rows = [
        "participant_id;utc_timestamp_iso;unix_time_ms;frame_index;relative_path",
        "P001;2026-06-04T09:26:01.000+00:00;1,78057E+12;1;video_frames/frame_000001.jpg",
        "P001;2026-06-04T09:26:01.100+00:00;1,78057E+12;2;video_frames/frame_000002.jpg",
        "P001;2026-06-04T09:26:01.200+00:00;1,78057E+12;3;video_frames/frame_000003.jpg",
    ]
    for index in range(1, 4):
        (frames / f"frame_{index:06d}.jpg").write_bytes(b"not decoded in this test")
    log = session / "video_frames.csv"
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return log


def test_p003_timestamp_uses_iso_fallback_and_keeps_outer_participant(tmp_path: Path):
    log = _write_p003_style_log(tmp_path)
    index = load_video_index(log, tmp_path, "P003")

    assert index.participant_id == "P003"
    assert index.timestamp_source == "utc_timestamp_iso"
    assert [frame.unix_time_ms for frame in index.frames] == [1780565161000, 1780565161100, 1780565161200]


def test_uniform_clip_never_duplicates_frames(tmp_path: Path):
    session = tmp_path / "session"
    log = session / "video_frames.csv"
    frame_dir = session / "video_frames"
    frame_dir.mkdir(parents=True)
    lines = ["unix_time_ms,frame_index,relative_path,utc_timestamp_iso"]
    for position in range(100):
        frame = frame_dir / f"frame_{position + 1:06d}.jpg"
        frame.write_bytes(b"x")
        lines.append(f"{1_700_000_000_000 + position * 100},{position + 1},video_frames/{frame.name},2026-01-01T00:00:00+00:00")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    index = load_video_index(log, session, "P002")

    clip = uniform_clip_frames(index, 1_700_000_000_000, 1_700_000_010_000, 16)
    sampled = sample_window_frames(index, 1_700_000_000_000, 1_700_000_010_000, 2.0)
    assert len(clip) == 16
    assert len({frame.frame_index for frame in clip}) == 16
    assert len(sampled) == len({frame.frame_index for frame in sampled})


def _config(tmp_path: Path) -> ProjectConfig:
    data = deepcopy(load_config().data)
    data["paths"] = {key: str(tmp_path / key) for key in data["paths"]}
    data["modeling"]["dcnn"]["device"] = "cpu"
    data["features"]["video"]["videomae2"]["pca_components"] = 2
    return ProjectConfig(source=tmp_path / "project.yaml", data=data)


def test_videomae_pca_uses_train_fold_only(tmp_path: Path):
    config = _config(tmp_path)
    rows = []
    embeddings = []
    for participant, offset in (("P002", 0.0), ("P003", 100.0)):
        for condition in ("C1", "C2"):
            for window in range(7):
                rows.append({
                    "participant_id": participant, "condition": condition, "condition_window_index": window,
                    "presentation_position": int(condition[1:]), "relaxation": 0.4, "discomfort": 0.6,
                    "intensity": 0.08, "frequency": 0.12, "eeg_alpha": offset + window,
                    "ecg_hr_bpm": 60 + window, "head_speed_mean": 0.1 * window, "eye_fixation_fraction_ivt": 0.5,
                })
                embeddings.append({
                    "participant_id": participant, "condition": condition, "condition_window_index": window,
                    "video_available": 1.0, "video_embedding_000": offset + window,
                    "video_embedding_001": offset + window + 1, "video_embedding_002": offset + window + 2,
                })
    windows = tmp_path / "windows.csv"
    visual = tmp_path / "embeddings.csv"
    pd.DataFrame(rows).to_csv(windows, index=False)
    pd.DataFrame(embeddings).to_csv(visual, index=False)
    sequences = build_video_sequences(windows, visual, config)
    train = np.flatnonzero(sequences.participant_ids == "P002")
    scaler = _fit_scaler(sequences, train, config, include_video=True)
    base, video, _ = _transform(sequences, np.arange(len(sequences.targets)), scaler)

    assert scaler["video_components"] == 2
    assert np.max(np.asarray(scaler["video_pca_mean"])) < 10.0
    assert base.shape == (4, 4, 8)
    assert video.shape == (4, 3, 8)  # two PCA dimensions plus explicit missing-video mask


def test_videomae_dcnn_checkpoint_round_trip(tmp_path: Path):
    import pytest

    pytest.importorskip("torch")
    config = _config(tmp_path)
    config.data["modeling"]["dcnn"].update({"max_epochs": 1, "early_stopping_patience": 1, "batch_size": 64})
    rows, embeddings = [], []
    for participant, offset in (("P002", 0.0), ("P003", 10.0)):
        for condition_number in (1, 2):
            for window in range(7):
                rows.append({
                    "participant_id": participant, "condition": f"C{condition_number}", "condition_window_index": window,
                    "presentation_position": condition_number, "relaxation": 0.2 + 0.4 * condition_number,
                    "discomfort": 0.8 - 0.3 * condition_number, "intensity": 0.08, "frequency": 0.12,
                    "eeg_alpha": offset + window, "ecg_hr_bpm": 60 + window,
                    "head_speed_mean": 0.1 * window, "eye_fixation_fraction_ivt": 0.5,
                })
                embeddings.append({
                    "participant_id": participant, "condition": f"C{condition_number}", "condition_window_index": window,
                    "video_available": 1.0, "video_embedding_000": offset + window,
                    "video_embedding_001": offset + window + 1, "video_embedding_002": offset + window + 2,
                })
    window_source = tmp_path / "window.csv"
    embedding_source = tmp_path / "embedding.csv"
    pd.DataFrame(rows).to_csv(window_source, index=False)
    pd.DataFrame(embeddings).to_csv(embedding_source, index=False)
    model_path = tmp_path / "dcnn_state_full.pt"
    train_video_dcnn_model(
        config, window_source=window_source, embedding_source=embedding_source, model_path=model_path,
        reports_dir=tmp_path / "reports", include_video=True, expected_labels=4,
    )
    bundle = load_video_dcnn_model(model_path, "cpu")
    prediction, widths, history = predict_video_dcnn_state(
        bundle, [{**rows[index], **embeddings[index]} for index in range(3)], "C1", config
    )
    assert set(prediction) == {"relaxation", "discomfort"}
    assert history == 3
    assert set(widths) == {"relaxation", "discomfort"}
