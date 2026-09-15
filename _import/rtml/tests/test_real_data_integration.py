from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from real_time_ml.config import load_config
from real_time_ml.data.alignment import condition_boundaries, load_marker_events, make_windows, marker_alignment_qc
from real_time_ml.data.index import build_index


@pytest.mark.integration
@pytest.mark.parametrize("participant", ["P003", "P015"])
def test_real_markers_have_nine_complete_conditions(participant: str):
    config = load_config()
    source = build_index(config, [participant])[0]
    if not source["xdf_path"]:
        pytest.skip("real XDF not present")
    events = load_marker_events(Path(source["xdf_path"]), config.get("streams.marker_name"))
    boundaries = condition_boundaries(events, participant)
    windows = make_windows(boundaries)
    assert len(boundaries) == 9
    assert len(windows) == 63
    by_condition = defaultdict(float)
    for row in windows:
        by_condition[row["condition"]] += row["sample_weight"]
    assert all(abs(value - 1.0) < 1e-12 for value in by_condition.values())
    qc = marker_alignment_qc(events)
    assert qc["median_abs_residual_ms"] <= 10.0
    if participant == "P015":
        assert qc["max_abs_residual_ms"] >= 80.0
        assert qc["outliers"]


@pytest.mark.integration
def test_p011_xdf_fallback_is_explicit_if_needed():
    config = load_config()
    source = build_index(config, ["P011"])[0]
    assert source["xdf_path"]
    if source["xdf_is_backup"]:
        assert any(token in source["xdf_path"].lower() for token in ("backup", "old", "bak"))

