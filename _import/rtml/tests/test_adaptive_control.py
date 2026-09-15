from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from real_time_ml.adaptive_control.contracts import AdaptiveReadinessSnapshot, ControlProfile, json_bytes
from real_time_ml.adaptive_control.models import RealtimeMultimodalWindowAdapter
from real_time_ml.adaptive_control.physio_monitor import build_physio_snapshot, downsample_values, summarize_values
from real_time_ml.adaptive_control.policy import ControlEstimate, AdaptiveControlPolicy
from real_time_ml.cli import build_parser
from real_time_ml.config import load_config
from real_time_ml.modeling.realtime_multimodal import (
    fit_realtime_multimodal_window_model,
    realtime_multimodal_feature_columns,
)


class StubAdapter:
    def __init__(self, current: ControlEstimate, candidates: dict[str, ControlEstimate]) -> None:
        self.current = current
        self.candidates = candidates

    def predict_current(self, features, condition, coverage):
        return self.current

    def predict_candidates(self, features, current_condition, candidates, current):
        return {name: self.candidates[name] for name in candidates}


def _profile(tmp_path) -> ControlProfile:
    path = tmp_path / "profile.json"
    conditions = []
    for index in range(9):
        conditions.append({
            "condition_id": f"C{index + 1}",
            "intensity": [0.08, 0.16, 0.25][index // 3],
            "frequency": [0.12, 0.26, 0.41][index % 3],
        })
    path.write_text(json.dumps({
        "profile_id": "test-adaptive-control",
        "schema_version": "1",
        "baseline": {"intensity": 0.01, "frequency": 0.0},
        "conditions": conditions,
    }), encoding="utf-8")
    return ControlProfile.from_path(path)


def _estimate(relaxation: float, discomfort: float, raw: dict | None = None) -> ControlEstimate:
    return ControlEstimate(relaxation, discomfort, "full", ["eeg", "ecg", "head", "eye"], raw=raw or {})


def test_profile_has_exact_grid_and_orthogonal_adjacency(tmp_path):
    profile = _profile(tmp_path)

    assert profile.values_for("C5").intensity == 0.16
    assert profile.values_for("C5").frequency == 0.26
    assert set(profile.adjacent("C5")) == {"C2", "C4", "C6", "C8"}
    assert profile.is_adjacent_or_same("C1", "C2")
    assert not profile.is_adjacent_or_same("C1", "C5")


def test_adaptive_control_policy_changes_one_adjacent_condition_at_low_hysteresis(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
    )
    adapter = StubAdapter(
        _estimate(0.50, 0.30),
        {
            "C2": _estimate(0.55, 0.30, {"candidate_policy_mode": "stable_probe"}),
            "C4": _estimate(0.51, 0.30),
            "C6": _estimate(0.52, 0.35),
            "C8": _estimate(0.48, 0.20),
        },
    )

    decision = policy.decide(adapter, {}, "C5", {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0})

    assert decision.action == "apply"
    assert decision.target_condition == "C2"
    assert decision.reasons == ["stable_probe"]
    assert decision.utility_delta is not None and decision.utility_delta >= 0.002


def test_adaptive_control_policy_failsafe_only_at_extreme_discomfort(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
    )
    adapter = StubAdapter(_estimate(0.8, 0.95), {})

    decision = policy.decide(adapter, {}, "C5", {})

    assert decision.action == "failsafe"
    assert decision.reasons == ["extreme_predicted_discomfort"]


def test_adaptive_control_policy_explores_after_calm_dwell(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
        calm_exploration_dwell_windows=3,
        calm_exploration_penalty_per_window=0.025,
        calm_exploration_penalty_max=0.12,
    )
    adapter = StubAdapter(
        _estimate(0.78, 0.00),
        {
            "C2": _estimate(0.70, 0.03),
            "C4": _estimate(0.64, 0.02),
        },
    )

    decision = policy.decide(
        adapter,
        {},
        "C1",
        {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0},
        dwell_windows=6,
    )

    assert decision.action == "apply"
    assert decision.target_condition == "C2"
    assert decision.reasons == ["calm_dwell_exploration"]
    assert decision.candidate_utilities["C1"] < 0.60


def test_stochastic_policy_routes_high_discomfort_to_safe_pool(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
        stochastic_exploration_enabled=True,
        exploration_candidate_scope="all",
        exploration_random_seed=3,
        safety_discomfort_min=0.50,
        safety_conditions=["C1", "C2", "C4", "C5"],
    )
    adapter = StubAdapter(
        _estimate(0.45, 0.60),
        {
            "C1": _estimate(0.55, 0.30),
            "C2": _estimate(0.56, 0.28),
            "C4": _estimate(0.54, 0.26),
            "C5": _estimate(0.57, 0.32),
        },
    )

    decision = policy.decide(
        adapter,
        {},
        "C9",
        {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0},
        condition_history=["C9", "C9", "C9"],
    )

    assert decision.action == "apply"
    assert decision.target_condition in {"C1", "C2", "C4", "C5"}
    assert decision.reasons == ["safety_recovery_stochastic"]


def test_stochastic_policy_respects_high_load_cooldown(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
        stochastic_exploration_enabled=True,
        exploration_candidate_scope="all",
        exploration_random_seed=1,
        calm_exploration_dwell_windows=3,
        high_load_conditions=["C9"],
        high_load_penalty=0.0,
    )
    adapter = StubAdapter(
        _estimate(0.80, 0.05),
        {
            "C1": _estimate(0.55, 0.20),
            "C2": _estimate(0.56, 0.20),
            "C3": _estimate(0.57, 0.20),
            "C4": _estimate(0.58, 0.20),
            "C5": _estimate(0.59, 0.20),
            "C7": _estimate(0.60, 0.20),
            "C8": _estimate(0.61, 0.20),
            "C9": _estimate(0.99, 0.00),
        },
    )

    decision = policy.decide(
        adapter,
        {},
        "C6",
        {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0},
        dwell_windows=3,
        condition_history=["C9", "C6", "C6", "C6"],
        condition_cooldowns={"C9": 4},
    )

    assert decision.action == "apply"
    assert decision.target_condition != "C9"
    assert decision.reasons == ["stochastic_calm_exploration"]


def test_sensor_conditioning_downweights_high_load_when_aroused(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
        stochastic_exploration_enabled=True,
        exploration_candidate_scope="all",
        exploration_random_seed=2,
        calm_exploration_dwell_windows=3,
        recent_history_penalty=0.0,
        high_load_penalty=0.0,
        sensor_conditioning_enabled=True,
    )
    adapter = StubAdapter(
        _estimate(0.70, 0.10),
        {
            "C1": _estimate(0.60, 0.10),
            "C2": _estimate(0.60, 0.10),
            "C3": _estimate(0.60, 0.10),
            "C4": _estimate(0.60, 0.10),
            "C6": _estimate(0.60, 0.10),
            "C7": _estimate(0.60, 0.10),
            "C8": _estimate(0.60, 0.10),
            "C9": _estimate(0.60, 0.10),
        },
    )

    decision = policy.decide(
        adapter,
        {
            "ecg_hr_bpm": 150.0,
            "ecg_hrv_30s_rmssd_ms": 50.0,
            "eye_angular_velocity_deg_s_mean": 24.0,
            "eye_saccade_fraction_ivt": 0.08,
            "head_speed_mean": 0.06,
            "head_angular_speed_deg_s_mean": 25.0,
            "head_stationary_fraction": 0.10,
            "eeg_alpha_beta_ratio": 0.20,
            "eeg_theta_beta_ratio": 2.50,
        },
        "C5",
        {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0},
        dwell_windows=3,
        condition_history=["C5", "C5", "C5"],
    )

    assert decision.action == "apply"
    assert decision.candidate_utilities["C1"] > decision.candidate_utilities["C9"]
    assert decision.candidate_utilities["C2"] > decision.candidate_utilities["C9"]
    assert policy.sensor_state(
        {
            "ecg_hr_bpm": 150.0,
            "eye_angular_velocity_deg_s_mean": 24.0,
            "head_speed_mean": 0.06,
            "eeg_theta_beta_ratio": 2.50,
        },
        _estimate(0.70, 0.10),
    )["physiological_arousal"] >= 0.9


def test_stochastic_policy_forces_change_at_max_dwell(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
        stochastic_exploration_enabled=True,
        exploration_candidate_scope="all",
        exploration_random_seed=4,
        min_condition_dwell_windows=0,
        max_condition_dwell_windows=4,
    )
    adapter = StubAdapter(
        _estimate(0.90, 0.02),
        {
            "C1": _estimate(0.50, 0.20),
            "C2": _estimate(0.50, 0.20),
            "C3": _estimate(0.50, 0.20),
            "C4": _estimate(0.50, 0.20),
            "C6": _estimate(0.50, 0.20),
            "C7": _estimate(0.50, 0.20),
            "C8": _estimate(0.50, 0.20),
            "C9": _estimate(0.50, 0.20),
        },
    )

    decision = policy.decide(
        adapter,
        {},
        "C5",
        {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0},
        dwell_windows=4,
        condition_history=["C5", "C5", "C5", "C5"],
    )

    assert decision.action == "apply"
    assert decision.target_condition != "C5"
    assert decision.reasons == ["stochastic_max_dwell_exploration"]


def test_switch_probability_allows_one_window_change_when_aroused(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
        stochastic_exploration_enabled=True,
        exploration_candidate_scope="all",
        exploration_random_seed=1,
        sensor_conditioning_enabled=True,
        switch_probability_enabled=True,
        switch_probability_after_windows=[0.10, 0.45, 0.85],
        switch_probability_force_after_windows=3,
        high_load_penalty=0.0,
        recent_history_penalty=0.0,
    )
    adapter = StubAdapter(
        _estimate(0.70, 0.10),
        {
            "C1": _estimate(0.60, 0.10),
            "C2": _estimate(0.60, 0.10),
            "C3": _estimate(0.60, 0.10),
            "C4": _estimate(0.60, 0.10),
            "C6": _estimate(0.60, 0.10),
            "C7": _estimate(0.60, 0.10),
            "C8": _estimate(0.60, 0.10),
            "C9": _estimate(0.60, 0.10),
        },
    )

    decision = policy.decide(
        adapter,
        {
            "ecg_hr_bpm": 150.0,
            "eye_angular_velocity_deg_s_mean": 24.0,
            "head_speed_mean": 0.06,
            "eeg_theta_beta_ratio": 2.50,
        },
        "C5",
        {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0},
        dwell_windows=0,
        condition_history=["C5"],
    )

    assert decision.action == "apply"
    assert decision.reasons == ["sensor_probability_exploration"]
    assert decision.estimate is not None
    assert decision.estimate.raw["controller_switch_probability"] > 0.10


def test_switch_probability_can_hold_stable_calm_after_one_window(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
        stochastic_exploration_enabled=True,
        exploration_candidate_scope="all",
        exploration_random_seed=1,
        sensor_conditioning_enabled=True,
        switch_probability_enabled=True,
        switch_probability_after_windows=[0.10, 0.45, 0.85],
        switch_probability_force_after_windows=3,
    )
    adapter = StubAdapter(_estimate(0.70, 0.05), {})

    decision = policy.decide(
        adapter,
        {
            "ecg_hr_bpm": 78.0,
            "ecg_hrv_30s_rmssd_ms": 260.0,
            "eye_angular_velocity_deg_s_mean": 5.0,
            "head_speed_mean": 0.010,
            "head_stationary_fraction": 0.50,
            "eeg_alpha_beta_ratio": 0.55,
            "eeg_theta_beta_ratio": 0.45,
        },
        "C5",
        {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0},
        dwell_windows=0,
        condition_history=["C5"],
    )

    assert decision.action == "hold"
    assert decision.reasons == ["sensor_probability_hold"]
    assert decision.estimate is not None
    assert decision.estimate.raw["controller_switch_probability"] < 0.10


def test_switch_probability_forces_change_on_third_window(tmp_path):
    profile = _profile(tmp_path)
    policy = AdaptiveControlPolicy(
        profile,
        relaxation_weight=0.85,
        discomfort_weight=-0.15,
        hysteresis=0.002,
        extreme_discomfort_limit=0.95,
        stochastic_exploration_enabled=True,
        exploration_candidate_scope="all",
        exploration_random_seed=20,
        sensor_conditioning_enabled=True,
        switch_probability_enabled=True,
        switch_probability_after_windows=[0.10, 0.45, 0.85],
        switch_probability_force_after_windows=3,
    )
    adapter = StubAdapter(
        _estimate(0.70, 0.05),
        {
            "C1": _estimate(0.60, 0.10),
            "C2": _estimate(0.60, 0.10),
            "C3": _estimate(0.60, 0.10),
            "C4": _estimate(0.60, 0.10),
            "C6": _estimate(0.60, 0.10),
            "C7": _estimate(0.60, 0.10),
            "C8": _estimate(0.60, 0.10),
            "C9": _estimate(0.60, 0.10),
        },
    )

    decision = policy.decide(
        adapter,
        {
            "ecg_hr_bpm": 78.0,
            "eye_angular_velocity_deg_s_mean": 5.0,
            "head_speed_mean": 0.010,
            "head_stationary_fraction": 0.50,
        },
        "C5",
        {"eeg": 1.0, "ecg": 1.0, "head": 1.0, "eye": 1.0},
        dwell_windows=2,
        condition_history=["C5", "C5", "C5"],
    )

    assert decision.action == "apply"
    assert decision.reasons == ["sensor_probability_force_exploration"]
    assert decision.estimate is not None
    assert decision.estimate.raw["controller_switch_probability"] == 1.0


def test_adaptive_control_has_a_clear_primary_command():
    args = build_parser().parse_args(["adaptive-control"])

    assert args.command == "adaptive-control"


def test_realtime_multimodal_command_is_available():
    args = build_parser().parse_args(["train-realtime-multimodal-window"])

    assert args.command == "train-realtime-multimodal-window"


def test_launcher_uses_configured_default_bundle():
    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((root / "configs" / "adaptive-control.yaml").read_text(encoding="utf-8"))
    launcher = (root / "start-adaptive-control.ps1").read_text(encoding="utf-8")

    assert f'[string]$Bundle = "{settings["models"]["default_bundle"]}"' in launcher


def test_realtime_multimodal_feature_filter_excludes_condition_context():
    columns = [
        "participant_id",
        "condition",
        "intensity",
        "frequency",
        "presentation_position",
        "eeg_tp9_alpha_relative",
        "ecg_hr_bpm",
        "eye_valid_fraction",
        "head_speed_mean",
        "imu_accel_mean",
        "video_motion",
    ]

    selected = realtime_multimodal_feature_columns(columns)

    assert selected == [
        "ecg_hr_bpm",
        "eeg_tp9_alpha_relative",
        "eye_valid_fraction",
        "head_speed_mean",
        "imu_accel_mean",
    ]


def _realtime_bundle(tmp_path):
    import pandas as pd
    import joblib

    rows = []
    for index in range(30):
        calm = index / 29.0
        rows.append({
            "participant_id": f"P{index % 3:03d}",
            "condition": f"C{index % 9 + 1}",
            "intensity": 0.25,
            "frequency": 0.41,
            "presentation_position": index % 9 + 1,
            "sample_weight": 1.0,
            "eeg_tp9_alpha_relative": calm,
            "ecg_hr_bpm": 110.0 - calm * 30.0,
            "eye_valid_fraction": 0.8 + calm * 0.2,
            "head_speed_mean": 0.2 - calm * 0.1,
            "relaxation": calm,
            "discomfort": 1.0 - calm,
        })
    bundle = fit_realtime_multimodal_window_model(
        pd.DataFrame(rows),
        ["relaxation", "discomfort"],
        seed=7,
        n_estimators=50,
        min_samples_leaf=1,
        max_features=1.0,
    )
    model_path = tmp_path / "realtime_multimodal_window_full.joblib"
    joblib.dump(bundle, model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "bundle_id": "realtime_multimodal_window_v1",
        "version": "1.0.0",
        "adapter_id": "realtime_multimodal_window_v1",
        "input_schema_version": "realtime-feature-v1",
        "window_seconds": 10.0,
        "required_modalities": ["eeg", "ecg", "eye", "head"],
        "optional_modalities": ["imu"],
        "target_schema": ["relaxation", "discomfort"],
        "candidate_prediction_supported": True,
        "adaptive_control_compatibility": True,
        "model_files": {
            "full": {
                "path": model_path.name,
                "sha256": digest,
            }
        },
    }), encoding="utf-8")
    return manifest


def test_realtime_multimodal_adapter_predicts_from_window_features(tmp_path):
    adapter = RealtimeMultimodalWindowAdapter(_realtime_bundle(tmp_path))
    report = adapter.preflight(_profile(tmp_path))
    assert report.compatible, report.reasons
    adapter.reset_session("session-1")

    low = adapter.predict_current({
        "eeg_tp9_alpha_relative": 0.0,
        "ecg_hr_bpm": 110.0,
        "eye_valid_fraction": 0.8,
        "head_speed_mean": 0.2,
    }, "C5", {"eeg": 1.0, "ecg": 1.0, "eye": 1.0, "head": 1.0})
    high = adapter.predict_current({
        "eeg_tp9_alpha_relative": 1.0,
        "ecg_hr_bpm": 80.0,
        "eye_valid_fraction": 1.0,
        "head_speed_mean": 0.1,
    }, "C5", {"eeg": 1.0, "ecg": 1.0, "eye": 1.0, "head": 1.0})

    assert low.raw["prediction_source"] == "window_multimodal_model"
    assert low.raw["input_feature_count"] == 4.0
    assert low.raw["missing_feature_count"] == 0.0
    assert set(low.active_modalities) == {"eeg", "ecg", "eye", "head"}
    assert high.relaxation > low.relaxation
    assert not (high.relaxation == low.relaxation == 1.0)


def test_realtime_multimodal_candidates_probe_upward_when_stable(tmp_path):
    adapter = RealtimeMultimodalWindowAdapter(_realtime_bundle(tmp_path))
    assert adapter.preflight(_profile(tmp_path)).compatible
    adapter.reset_session("session-1")
    current = _estimate(0.50, 0.35)

    candidates = adapter.predict_candidates({}, "C1", ["C2", "C4"], current)

    assert set(candidates) == {"C2", "C4"}
    for estimate in candidates.values():
        assert estimate.raw["candidate_policy_mode"] == "stable_probe"
        assert estimate.relaxation > current.relaxation
        assert estimate.discomfort < current.discomfort


def test_realtime_multimodal_adapter_tolerates_missing_features(tmp_path):
    adapter = RealtimeMultimodalWindowAdapter(_realtime_bundle(tmp_path))
    assert adapter.preflight(_profile(tmp_path)).compatible
    adapter.reset_session("session-1")

    estimate = adapter.predict_current({"eeg_tp9_alpha_relative": 0.5}, "C5", {"eeg": 1.0})

    assert np.isfinite(estimate.relaxation)
    assert np.isfinite(estimate.discomfort)
    assert estimate.raw["missing_feature_count"] == 3.0
    assert json_bytes({
        "message_type": "AdaptiveStatusSnapshot",
        "protocol_version": "adaptive-control-v1",
        "prediction_source": estimate.raw["prediction_source"],
        "model_modalities_used": estimate.active_modalities,
    })


def test_readiness_snapshot_has_unity_friendly_flat_fields():
    snapshot = AdaptiveReadinessSnapshot(
        request_id="request-1",
        unix_time_ms=123,
        python_ready=True,
        model_ready=True,
        model_bundle_id="classical_condition_current",
        model_version="1.0.0",
        lsl_eeg_stream_found=True,
        lsl_eeg_sample_received=True,
        eeg_ready=True,
        ecg_ready=True,
        physio_sample_channel_count=9,
        expected_min_channel_count=9,
        reasons=["ready"],
    ).to_dict()

    assert snapshot["message_type"] == "AdaptiveReadinessSnapshot"
    assert snapshot["protocol_version"] == "adaptive-control-v1"
    assert snapshot["eeg_ready"] is True
    assert snapshot["ecg_ready"] is True
    assert snapshot["reasons"] == ["ready"]


def test_physio_monitor_stats_are_json_safe():
    stats = summarize_values(np.asarray([-1.0, 0.0, 3.0, 4.0]))

    assert stats.sample_count == 4
    assert stats.mean == 1.5
    assert np.isclose(stats.std, np.std([-1.0, 0.0, 3.0, 4.0]))
    assert stats.min == -1.0
    assert stats.max == 4.0
    assert np.isclose(stats.rms, np.sqrt(6.5))
    assert stats.peak_to_peak == 5.0


def test_physio_monitor_downsamples_ten_second_500hz_window():
    values = np.arange(5000, dtype=float)

    output = downsample_values(values, 240)

    assert len(output) == 240
    assert output[0] == 0.0
    assert output[-1] == 4999.0


def test_physio_snapshot_encodes_empty_and_valid_windows():
    config = load_config("configs/project.yaml")
    empty = build_physio_snapshot(
        config,
        [],
        1_000,
        stream_found=False,
        sample_received=False,
        last_sample_ms=None,
    ).to_dict()
    assert empty["message_type"] == "AdaptivePhysioSnapshot"
    assert empty["channels"] == []
    assert empty["sample_age_ms"] == -1.0
    assert json_bytes(empty)

    sample = np.zeros(9, dtype=float)
    sample[1:5] = [1.0, 2.0, 3.0, 4.0]
    sample[7] = 10.0
    sample[8] = 2.0
    rows = [sample + index * 0.01 for index in range(5000)]
    valid = build_physio_snapshot(
        config,
        rows,
        11_000,
        stream_found=True,
        sample_received=True,
        last_sample_ms=10_990,
        stream_name="02_007_5_eeg",
        stream_type="eeg",
        channel_count=9,
        nominal_srate=500.0,
    ).to_dict()

    assert valid["sample_age_ms"] == 10.0
    assert len(valid["channels"]) == 5
    assert all(len(channel["raw_values"]) <= 240 for channel in valid["channels"])
    assert json_bytes(valid)
