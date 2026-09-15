from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


REQUIRED_TIMING_FIELDS = (
    "schema_version", "unix_time_ms", "cycle_index", "window_start_ms", "window_end_ms"
)


@dataclass
class StatePrediction:
    schema_version: str
    unix_time_ms: int
    cycle_index: int
    window_start_ms: int
    window_end_ms: int
    participant_id: str | None
    condition: str | None
    predictions: dict[str, float | None]
    intervals: dict[str, list[float] | None]
    modality_coverage: dict[str, float]
    qc: dict[str, Any]
    model_variant: str
    missing_modalities: list[str] = field(default_factory=list)
    degraded: bool = False
    message_type: str = "StatePrediction"

    def validate(self) -> None:
        if self.window_end_ms - self.window_start_ms != 10_000:
            raise ValueError("StatePrediction must describe exactly one 10-second window")
        required = {"relaxation", "discomfort"}
        if set(self.predictions) != required:
            raise ValueError(f"predictions must contain exactly {sorted(required)}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass
class ConditionRecommendation:
    schema_version: str
    unix_time_ms: int
    cycle_index: int
    window_start_ms: int
    window_end_ms: int
    current_condition: str | None
    candidate_condition: str | None
    expected_relaxation_gain: float | None
    conservative_gain: float | None
    predicted_discomfort: float | None
    safe: bool
    action: str
    reasons: list[str]
    shadow: bool = True
    message_type: str = "ConditionRecommendation"

    def validate(self) -> None:
        if self.window_end_ms - self.window_start_ms != 10_000:
            raise ValueError("ConditionRecommendation must describe exactly one 10-second window")
        if not self.shadow:
            raise ValueError("Version 1 is shadow-only")
        if self.action not in {"hold", "recommend"}:
            raise ValueError("action must be 'hold' or 'recommend'")
        if self.action == "hold" and self.candidate_condition != self.current_condition:
            raise ValueError("hold must retain the current condition")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def validate_message(payload: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_TIMING_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"Missing message fields: {missing}")
    if int(payload["window_end_ms"]) - int(payload["window_start_ms"]) != 10_000:
        raise ValueError("Message window must be exactly 10 seconds")
