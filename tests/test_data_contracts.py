from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from real_time_ml.config import load_config
from real_time_ml.data.io import condition_parameters, normalize_condition, normalize_participant_id, resolve_video_path
from real_time_ml.data.labels import parse_condition_labels


def test_id_normalization_special_cases():
    assert normalize_participant_id("014") == "P014"
    assert normalize_participant_id("N009") == "P009"
    assert normalize_participant_id("p003") == "P003"
    assert normalize_condition("C01") == "C1"
    assert normalize_condition("c9") == "C9"


def test_condition_grid_mapping():
    intensities, frequencies = [0.08, 0.16, 0.25], [0.12, 0.26, 0.41]
    assert condition_parameters("C1", intensities, frequencies)["intensity"] == 0.08
    assert condition_parameters("C1", intensities, frequencies)["frequency"] == 0.12
    assert condition_parameters("C9", intensities, frequencies)["intensity"] == 0.25
    assert condition_parameters("C9", intensities, frequencies)["frequency"] == 0.41


def test_labels_are_15_by_9_and_calm_is_reversed():
    config = load_config()
    rows = parse_condition_labels(
        config.path("labels_root"), config.participants,
        config.get("conditions.intensities"), config.get("conditions.frequencies"),
    )
    assert len(rows) == 135
    assert len({(row["participant_id"], row["condition"]) for row in rows}) == 135
    assert all(abs(row["calm"] - (7.0 - row["arousal_raw"]) / 6.0) < 1e-12 for row in rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["participant_id"]] += 1
    assert set(counts.values()) == {9}


def test_video_path_ignores_stale_absolute_path(tmp_path: Path):
    target = resolve_video_path(tmp_path, {"relative_path": "video_frames/frame_1.jpg", "absolute_path": "Z:/stale.jpg"})
    assert target == (tmp_path / "video_frames" / "frame_1.jpg").resolve()
    with pytest.raises(ValueError):
        resolve_video_path(tmp_path, {"relative_path": "../escape.jpg"})

