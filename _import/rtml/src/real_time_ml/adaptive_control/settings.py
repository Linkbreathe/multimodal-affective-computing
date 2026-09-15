"""Adaptive-control-only configuration, deliberately separate from frozen research settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ADAPTIVE_CONTROL_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "adaptive-control.yaml"


@dataclass(frozen=True)
class AdaptiveControlSettings:
    source: Path
    profile_path: Path
    registry_path: Path
    default_bundle: str
    listen_host: str
    unity_to_python_port: int
    python_send_host: str
    python_to_unity_port: int
    sensor_hz: float
    status_hz: float
    command_expiry_ms: int
    command_timeout_seconds: float
    min_modality_coverage: float
    min_active_modalities: int
    consecutive_low_modality_windows: int
    transition_seconds: float
    initial_condition: str
    utility_relaxation_weight: float
    utility_discomfort_weight: float
    utility_hysteresis: float
    extreme_discomfort_limit: float
    calm_exploration_dwell_windows: int
    calm_exploration_penalty_per_window: float
    calm_exploration_penalty_max: float
    calm_exploration_relaxation_min: float
    calm_exploration_discomfort_max: float
    stochastic_exploration_enabled: bool
    exploration_candidate_scope: str
    exploration_random_seed: int | None
    exploration_temperature: float
    exploration_random_floor: float
    sensor_conditioning_enabled: bool
    sensor_conditioning_weight: float
    switch_probability_enabled: bool
    switch_probability_after_windows: list[float]
    switch_probability_force_after_windows: int
    switch_probability_boredom_weight: float
    switch_probability_arousal_weight: float
    switch_probability_discomfort_weight: float
    switch_probability_stable_calm_weight: float
    safety_discomfort_min: float
    safety_conditions: list[str]
    min_condition_dwell_windows: int
    max_condition_dwell_windows: int
    recent_history_window: int
    recent_history_penalty: float
    high_load_conditions: list[str]
    high_load_penalty: float
    high_load_cooldown_windows: int

    def bundle_manifest(self, bundle_id: str | None = None) -> Path:
        selected = bundle_id or self.default_bundle
        candidate = self.registry_path / selected / "manifest.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Adaptive control model bundle manifest not found: {candidate}")
        return candidate


def _resolve(source: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (source.parent / candidate).resolve()


def load_adaptive_control_settings(path: str | Path | None = None) -> AdaptiveControlSettings:
    source = Path(path).resolve() if path else DEFAULT_ADAPTIVE_CONTROL_CONFIG.resolve()
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    network = payload.get("network", {})
    runtime = payload.get("runtime", {})
    policy = payload.get("policy", {})
    models = payload.get("models", {})
    settings = AdaptiveControlSettings(
        source=source,
        profile_path=_resolve(source, str(payload["profile_path"])),
        registry_path=_resolve(source, str(models["registry_path"])),
        default_bundle=str(models["default_bundle"]),
        listen_host=str(network.get("listen_host", "127.0.0.1")),
        unity_to_python_port=int(network.get("unity_to_python_port", 5055)),
        python_send_host=str(network.get("python_send_host", "127.0.0.1")),
        python_to_unity_port=int(network.get("python_to_unity_port", 5056)),
        sensor_hz=float(network.get("sensor_hz", 10.0)),
        status_hz=float(network.get("status_hz", 1.0)),
        command_expiry_ms=int(runtime.get("command_expiry_ms", 9500)),
        command_timeout_seconds=float(runtime.get("command_timeout_seconds", 25.0)),
        min_modality_coverage=float(policy.get("min_modality_coverage", 0.2)),
        min_active_modalities=int(policy.get("min_active_modalities", 2)),
        consecutive_low_modality_windows=int(policy.get("consecutive_low_modality_windows", 3)),
        transition_seconds=float(runtime.get("transition_seconds", 2.0)),
        initial_condition=str(runtime.get("initial_condition", "C5")),
        utility_relaxation_weight=float(policy.get("utility_relaxation_weight", 0.85)),
        utility_discomfort_weight=float(policy.get("utility_discomfort_weight", -0.15)),
        utility_hysteresis=float(policy.get("utility_hysteresis", 0.002)),
        extreme_discomfort_limit=float(policy.get("extreme_discomfort_limit", 0.95)),
        calm_exploration_dwell_windows=int(policy.get("calm_exploration_dwell_windows", 3)),
        calm_exploration_penalty_per_window=float(policy.get("calm_exploration_penalty_per_window", 0.025)),
        calm_exploration_penalty_max=float(policy.get("calm_exploration_penalty_max", 0.12)),
        calm_exploration_relaxation_min=float(policy.get("calm_exploration_relaxation_min", 0.55)),
        calm_exploration_discomfort_max=float(policy.get("calm_exploration_discomfort_max", 0.40)),
        stochastic_exploration_enabled=bool(policy.get("stochastic_exploration_enabled", False)),
        exploration_candidate_scope=str(policy.get("exploration_candidate_scope", "adjacent")),
        exploration_random_seed=(
            int(policy["exploration_random_seed"])
            if policy.get("exploration_random_seed") is not None
            else None
        ),
        exploration_temperature=float(policy.get("exploration_temperature", 0.18)),
        exploration_random_floor=float(policy.get("exploration_random_floor", 0.08)),
        sensor_conditioning_enabled=bool(policy.get("sensor_conditioning_enabled", False)),
        sensor_conditioning_weight=float(policy.get("sensor_conditioning_weight", 1.0)),
        switch_probability_enabled=bool(policy.get("switch_probability_enabled", False)),
        switch_probability_after_windows=[float(value) for value in policy.get("switch_probability_after_windows", [0.10, 0.45, 0.85])],
        switch_probability_force_after_windows=int(policy.get("switch_probability_force_after_windows", 3)),
        switch_probability_boredom_weight=float(policy.get("switch_probability_boredom_weight", 0.25)),
        switch_probability_arousal_weight=float(policy.get("switch_probability_arousal_weight", 0.20)),
        switch_probability_discomfort_weight=float(policy.get("switch_probability_discomfort_weight", 0.30)),
        switch_probability_stable_calm_weight=float(policy.get("switch_probability_stable_calm_weight", 0.15)),
        safety_discomfort_min=float(policy.get("safety_discomfort_min", 0.50)),
        safety_conditions=list(policy.get("safety_conditions", ["C1", "C2", "C4", "C5"])),
        min_condition_dwell_windows=int(policy.get("min_condition_dwell_windows", 0)),
        max_condition_dwell_windows=int(policy.get("max_condition_dwell_windows", 0)),
        recent_history_window=int(policy.get("recent_history_window", 4)),
        recent_history_penalty=float(policy.get("recent_history_penalty", 0.12)),
        high_load_conditions=list(policy.get("high_load_conditions", ["C9"])),
        high_load_penalty=float(policy.get("high_load_penalty", 0.10)),
        high_load_cooldown_windows=int(policy.get("high_load_cooldown_windows", 0)),
    )
    if settings.sensor_hz <= 0 or settings.status_hz <= 0:
        raise ValueError("Adaptive control sensor_hz and status_hz must be positive")
    if settings.transition_seconds <= 0 or settings.command_expiry_ms <= 0:
        raise ValueError("Adaptive control transition_seconds and command_expiry_ms must be positive")
    if not 0 <= settings.min_modality_coverage <= 1:
        raise ValueError("Adaptive control min_modality_coverage must be in [0, 1]")
    if settings.calm_exploration_dwell_windows < 0:
        raise ValueError("Adaptive control calm_exploration_dwell_windows must be non-negative")
    if settings.calm_exploration_penalty_per_window < 0 or settings.calm_exploration_penalty_max < 0:
        raise ValueError("Adaptive control calm exploration penalties must be non-negative")
    if settings.exploration_candidate_scope not in {"adjacent", "all"}:
        raise ValueError("Adaptive control exploration_candidate_scope must be 'adjacent' or 'all'")
    if settings.exploration_temperature <= 0:
        raise ValueError("Adaptive control exploration_temperature must be positive")
    if settings.exploration_random_floor < 0:
        raise ValueError("Adaptive control exploration_random_floor must be non-negative")
    if settings.sensor_conditioning_weight < 0:
        raise ValueError("Adaptive control sensor_conditioning_weight must be non-negative")
    if not settings.switch_probability_after_windows:
        raise ValueError("Adaptive control switch_probability_after_windows must not be empty")
    if any(value < 0 or value > 1 for value in settings.switch_probability_after_windows):
        raise ValueError("Adaptive control switch_probability_after_windows values must be in [0, 1]")
    if settings.switch_probability_force_after_windows < 1:
        raise ValueError("Adaptive control switch_probability_force_after_windows must be >= 1")
    if (
        settings.switch_probability_boredom_weight < 0
        or settings.switch_probability_arousal_weight < 0
        or settings.switch_probability_discomfort_weight < 0
        or settings.switch_probability_stable_calm_weight < 0
    ):
        raise ValueError("Adaptive control switch probability weights must be non-negative")
    if not 0 <= settings.safety_discomfort_min <= 1:
        raise ValueError("Adaptive control safety_discomfort_min must be in [0, 1]")
    if settings.min_condition_dwell_windows < 0 or settings.max_condition_dwell_windows < 0:
        raise ValueError("Adaptive control dwell window limits must be non-negative")
    if settings.max_condition_dwell_windows and settings.max_condition_dwell_windows < settings.min_condition_dwell_windows:
        raise ValueError("Adaptive control max_condition_dwell_windows must be >= min_condition_dwell_windows")
    if settings.recent_history_window < 0 or settings.recent_history_penalty < 0:
        raise ValueError("Adaptive control recent history settings must be non-negative")
    if settings.high_load_penalty < 0 or settings.high_load_cooldown_windows < 0:
        raise ValueError("Adaptive control high-load settings must be non-negative")
    return settings
