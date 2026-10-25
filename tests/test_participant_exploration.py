"""Mathematical fixtures only; none of these data enter the research export."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

from mac.analysis.participant_exploration.behavior import behavioral_features, rotation_angles
from mac.analysis.participant_exploration.common import feature_result, intervals, write_csv
from mac.analysis.participant_exploration.physiology import (
    band_powers,
    continuity,
    eeg_features,
    rmssd_from_intervals,
)
from mac.analysis.participant_exploration.specification import FEATURES
from mac.analysis.participant_exploration.statistics import association, describe
from mac.data.index import discover_sources
from mac.data.video import load_video_index


def test_counter_rollover_is_not_a_gap():
    assert continuity(np.arange(6) / 500, np.array([253, 254, 255, 0, 1, 2]), 500).all()
    assert not continuity(np.arange(3) / 500, np.array([0, 1, 3]), 500)[1]


def test_intervals_preserve_invalid_rows_and_edges():
    assert list(
        intervals(np.array([True, True, False, True, True]), np.array([True, True, True, False]))
    ) == [(0, 2), (3, 4), (4, 5)]


def test_rotation_sign_and_euler_wrap():
    q = Rotation.from_euler("z", np.array([[179], [-179]]), degrees=True).as_quat()
    assert np.allclose(rotation_angles(q), 2)
    q[1] *= -1
    assert np.allclose(rotation_angles(q), 2)


def test_rr_rejection_does_not_create_successive_beats():
    rr = [
        dict(rr_ms=v, accepted=good, rr_index=i, segment_id=0)
        for i, (v, good) in enumerate([(1000, True), (3000, False), (800, True), (900, True)])
    ]
    value, n = rmssd_from_intervals(rr)
    assert n == 1 and value == 100
    rr[-1]["segment_id"] = 1
    assert rmssd_from_intervals(rr)[1] == 0


def test_band_power_units_and_frequency_definition():
    t = np.arange(5000) / 500
    power = band_powers((2 * np.sin(2 * np.pi * 10 * t))[:, None], 500)
    assert np.isclose(power["alpha"][0], 2, rtol=0.001)
    assert power["theta"][0] < 1e-20 and power["beta"][0] < 1e-20


def test_missing_zero_and_nonpositive_log_are_distinct():
    assert feature_result("head_mean_angular_speed", 0)["analysis_value"] == 0
    assert feature_result("ecg_rmssd", 0)["availability_status"] == "raw_only"
    assert feature_result("ecg_rmssd", reason="missing")["availability_status"] == "unavailable"


def test_spearman_ties_and_missing_values():
    x = np.array([1, 1, 2, 3, 4, 5, np.nan, 7, 8.0])
    y = np.array([3, 2, 4, 5, 5, 1, 2, 7, 6.0])
    c = np.array([f"C{i}" for i in range(1, 10)])
    result = association(x, y, c)
    assert np.isclose(result["spearman_rho"], spearmanr(x, y, nan_policy="omit").statistic)
    assert result["n_valid_pairs"] == 8 and "C7" not in result["included_conditions"]
    assert association(np.ones(9), y, c)["reason"] == "constant_feature"
    assert association(x[:5], y[:5], c[:5])["reason"] == "fewer_than_six_valid_paired_conditions"


def test_time_weighted_gaze_and_tracking_gaps():
    # Unequal intervals: 0.1 s at 0 deg/s, 0.2 s at 150 deg/s.
    angle = np.radians([0, 0, 30])
    f = pd.DataFrame(
        dict(
            unix_time_ms=[0, 100, 300],
            gaze_direction_x=np.sin(angle),
            gaze_direction_y=0,
            gaze_direction_z=np.cos(angle),
            gaze_available=True,
        )
    )
    values, _ = behavioral_features(f, "eye", 0, 310)
    assert np.isclose(values[0]["raw_value"], 100)
    assert np.isclose(values[1]["raw_value"], 1 / 3)
    f.loc[1, "gaze_available"] = False
    values, _ = behavioral_features(f, "eye", 0, 310)
    assert all(not x["analysis_available"] for x in values)


def test_read_only_discovery_and_backup(tmp_path: Path):
    directory = tmp_path / "P011"
    directory.mkdir()
    (directory / "recording_old3.xdf").write_bytes(b"fixture")
    before = set(tmp_path.rglob("*"))
    result = discover_sources(tmp_path, ["P011"])
    assert result[0]["xdf_is_backup"]
    assert set(tmp_path.rglob("*")) == before


def test_csv_numeric_roundtrip_and_small_descriptions(tmp_path: Path):
    f = pd.DataFrame({"number": [1.0, np.nan, 0.0], "reason": ["", "missing", ""]})
    p = tmp_path / "test.csv"
    write_csv(p, f)
    actual = pd.read_csv(p)
    assert actual.number.dtype.kind == "f" and np.isnan(actual.number.iloc[1])
    assert ",missing" in p.read_text()
    assert np.isnan(describe([1])["std"])
    assert len(FEATURES) == 11


def test_p003_timestamp_fallback_does_not_require_writing_images(tmp_path: Path):
    log = tmp_path / "frames.csv"
    log.write_text(
        "participant_id;utc_timestamp_iso;unix_time_ms;frame_index;relative_path\n"
        "P001;2026-06-04T09:26:01.000+00:00;1,78057E+12;1;video_frames/frame_000001.jpg\n"
        "P001;2026-06-04T09:26:01.100+00:00;1,78057E+12;2;video_frames/frame_000002.jpg\n"
        "P001;2026-06-04T09:26:01.200+00:00;1,78057E+12;3;video_frames/frame_000003.jpg\n",
        encoding="utf-8",
    )
    result = load_video_index(log, tmp_path, "P003")
    assert result.participant_id == "P003" and result.timestamp_source == "utc_timestamp_iso"
    assert np.array_equal(np.diff([f.unix_time_ms for f in result.frames]), [100, 100])
    assert not list(tmp_path.rglob("*.jpg"))


def test_eeg_rejects_missing_second_channel_without_selecting_first():
    fs = 500
    t = np.arange(5000) / fs
    reference = np.sin(2 * np.pi * 3 * t)
    raw = np.zeros((len(t), 9))
    raw[:, 0] = np.arange(len(t)) % 256
    raw[:, 1] = reference
    raw[:, 4] = reference
    raw[:, 2] = reference + 5 * np.sin(2 * np.pi * 10 * t)
    raw[:, 3] = reference
    result, audit = eeg_features(raw, t, fs, np.ones(len(t) - 1, dtype=bool), 0, 10)
    assert all(not r["analysis_available"] for r in result)
    assert any(r["channel"] == "TP9" and r["accepted"] for r in audit)
    assert not any(r["channel"] == "TP10" and r["accepted"] for r in audit)


def test_existing_index_writer_preserves_contract(tmp_path: Path):
    from types import SimpleNamespace
    from mac.data.index import build_index

    root = tmp_path / "raw"
    p = root / "P011"
    p.mkdir(parents=True)
    (p / "recording.xdf").write_bytes(b"test fixture")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "4. Painting Reflection (Responses).xlsx").write_bytes(b"only hashed in this test")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("test: true")
    paths = {"raw_root": root, "labels_root": labels, "manifests": tmp_path / "manifests"}
    config = SimpleNamespace(
        participants=["P011"],
        path=lambda key: paths[key],
        source=config_file,
        data={"schema_version": "test"},
    )
    expected = discover_sources(root, ["P011"])
    assert build_index(config) == expected
    assert (paths["manifests"] / "source_manifest.csv").is_file()
    assert len(list(root.rglob("*"))) == 2
