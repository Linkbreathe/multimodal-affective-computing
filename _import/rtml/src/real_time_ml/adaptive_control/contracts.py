"""Versioned control contracts shared by Python and Unity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from real_time_ml.data.io import normalize_condition


PROTOCOL_VERSION = "adaptive-control-v1"


@dataclass(frozen=True)
class ControlValues:
    intensity: float
    frequency: float


@dataclass(frozen=True)
class ControlProfile:
    profile_id: str
    schema_version: str
    baseline: ControlValues
    conditions: dict[str, ControlValues]
    sha256: str
    source: Path

    @classmethod
    def from_path(cls, path: str | Path) -> "ControlProfile":
        source = Path(path).resolve()
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        profile_id = str(payload.get("profile_id", "")).strip()
        if not profile_id:
            raise ValueError("Adaptive control profile requires profile_id")
        baseline_payload = payload.get("baseline") or {}
        baseline = ControlValues(
            intensity=float(baseline_payload["intensity"]),
            frequency=float(baseline_payload["frequency"]),
        )
        conditions: dict[str, ControlValues] = {}
        for item in payload.get("conditions", []):
            condition_id = normalize_condition(str(item.get("condition_id", "")))
            if condition_id in conditions:
                raise ValueError(f"Duplicate adaptive-control condition in profile: {condition_id}")
            conditions[condition_id] = ControlValues(float(item["intensity"]), float(item["frequency"]))
        expected = {f"C{index}" for index in range(1, 10)}
        if set(conditions) != expected:
            raise ValueError("Adaptive control profile must define exactly C1 through C9")
        for name, values in {"baseline": baseline, **conditions}.items():
            if values.intensity < 0 or values.frequency < 0:
                raise ValueError(f"Adaptive control profile contains negative values for {name}")
        return cls(
            profile_id=profile_id,
            schema_version=str(payload.get("schema_version", "1")),
            baseline=baseline,
            conditions=conditions,
            sha256=hashlib.sha256(raw).hexdigest(),
            source=source,
        )

    def values_for(self, condition: str) -> ControlValues:
        return self.conditions[normalize_condition(condition)]

    def adjacent(self, condition: str) -> list[str]:
        index = int(normalize_condition(condition)[1:]) - 1
        row, column = divmod(index, 3)
        result = []
        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate_row, candidate_column = row + delta_row, column + delta_column
            if 0 <= candidate_row < 3 and 0 <= candidate_column < 3:
                result.append(f"C{candidate_row * 3 + candidate_column + 1}")
        return result

    def is_adjacent_or_same(self, current: str, candidate: str) -> bool:
        current = normalize_condition(current)
        candidate = normalize_condition(candidate)
        return candidate == current or candidate in self.adjacent(current)


@dataclass
class AdaptiveControlCommand:
    session_id: str
    command_id: str
    cycle_index: int
    issued_unix_ms: int
    expires_unix_ms: int
    action: str
    current_condition: str
    target_condition: str
    intensity: float
    frequency: float
    transition_seconds: float
    utility_delta: float | None
    predicted_relaxation: float | None
    predicted_discomfort: float | None
    model_variant: str
    active_modalities: list[str]
    profile_id: str
    profile_sha256: str
    reasons: list[str] = field(default_factory=list)
    message_type: str = "AdaptiveControlCommand"
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveStatusSnapshot:
    session_id: str | None
    unix_time_ms: int
    runtime_state: str
    next_decision_in_seconds: float
    current_condition: str | None
    target_condition: str | None
    modality_coverage: dict[str, float]
    modality_age_ms: dict[str, float | None]
    model_bundle_id: str | None
    model_version: str | None
    model_variant: str | None
    predictions: dict[str, float | None]
    candidate_utilities: dict[str, float]
    last_command_id: str | None
    # Array mirrors keep Unity's JsonUtility free of dictionary parsing.
    modality_names: list[str] = field(default_factory=list)
    modality_coverage_values: list[float] = field(default_factory=list)
    modality_age_values: list[float] = field(default_factory=list)
    candidate_conditions: list[str] = field(default_factory=list)
    candidate_utility_values: list[float] = field(default_factory=list)
    relaxation: float | None = None
    discomfort: float | None = None
    prediction_source: str | None = None
    model_supervision: str | None = None
    motion_source: str | None = None
    model_input_feature_count: int = 0
    model_available_feature_count: int = 0
    model_missing_feature_count: int = 0
    model_modalities_used: list[str] = field(default_factory=list)
    headset_presence_available: bool = False
    headset_worn: bool = False
    lsl_eeg_stream_found: bool = False
    lsl_eeg_sample_received: bool = False
    warmup_complete: bool = False
    reasons: list[str] = field(default_factory=list)
    message_type: str = "AdaptiveStatusSnapshot"
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveReadinessSnapshot:
    request_id: str | None
    unix_time_ms: int
    python_ready: bool
    model_ready: bool
    model_bundle_id: str | None
    model_version: str | None
    lsl_eeg_stream_found: bool
    lsl_eeg_sample_received: bool
    eeg_ready: bool
    ecg_ready: bool
    physio_sample_channel_count: int
    expected_min_channel_count: int
    reasons: list[str] = field(default_factory=list)
    message_type: str = "AdaptiveReadinessSnapshot"
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptivePhysioStats:
    sample_count: int
    mean: float
    std: float
    min: float
    max: float
    rms: float
    peak_to_peak: float


@dataclass
class AdaptivePhysioChannelSnapshot:
    name: str
    source: str
    unit: str
    raw_values: list[float]
    filtered_values: list[float]
    raw: AdaptivePhysioStats
    filtered: AdaptivePhysioStats


@dataclass
class AdaptivePhysioSnapshot:
    unix_time_ms: int
    lsl_eeg_stream_found: bool
    lsl_eeg_sample_received: bool
    stream_name: str
    stream_type: str
    channel_count: int
    expected_channel_count: int
    nominal_srate: float
    sample_rate_hz: float
    window_seconds: float
    sample_age_ms: float
    sample_count: int
    max_points: int
    channels: list[AdaptivePhysioChannelSnapshot] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    message_type: str = "AdaptivePhysioSnapshot"
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def unix_milliseconds(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
