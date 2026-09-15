from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from real_time_ml.data.io import condition_parameters, normalize_condition
from real_time_ml.schema import ConditionRecommendation


def adjacent_conditions(current: str) -> list[str]:
    index = int(normalize_condition(current)[1:]) - 1
    row, column = divmod(index, 3)
    candidates = {(row, column)}
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, column + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            candidates.add((nr, nc))
    return [f"C{r * 3 + c + 1}" for r, c in sorted(candidates)]


@dataclass
class ResidualCalibrator:
    learning_rate: float = 0.2
    residuals: dict[str, float] = field(default_factory=dict)

    def update(self, participant_id: str, observed: float, predicted: float) -> None:
        residual = float(observed - predicted)
        previous = self.residuals.get(participant_id, 0.0)
        self.residuals[participant_id] = (1.0 - self.learning_rate) * previous + self.learning_rate * residual

    def apply(self, participant_id: str, prediction: float) -> float:
        return float(np.clip(prediction + self.residuals.get(participant_id, 0.0), 0.0, 1.0))


@dataclass
class SafetyPolicy:
    schema_version: str
    intensities: list[float]
    frequencies: list[float]
    discomfort_limit: float = 0.5
    uncertainty_width_max: float = 0.45
    min_modality_coverage: float = 0.5
    min_gain: float = 0.0

    def recommend(
        self,
        *,
        unix_time_ms: int,
        cycle_index: int,
        window_start_ms: int,
        current_condition: str,
        state: dict[str, float | None],
        intervals: dict[str, list[float] | None],
        modality_coverage: dict[str, float],
        model_deployable: bool,
        predict_candidate: Callable[[dict[str, Any]], dict[str, float]],
        force_hold_reasons: list[str] | None = None,
        candidate_interval_half_width: dict[str, float] | None = None,
    ) -> ConditionRecommendation:
        current = normalize_condition(current_condition)
        reasons = list(force_hold_reasons or [])
        if not model_deployable:
            reasons.append("state_model_not_better_than_baselines")
        if not modality_coverage or max(modality_coverage.values(), default=0.0) < self.min_modality_coverage:
            reasons.append("insufficient_modality_coverage")
        widths = [bounds[1] - bounds[0] for bounds in intervals.values() if bounds is not None]
        if widths and max(widths) > self.uncertainty_width_max:
            reasons.append("state_uncertainty_too_high")
        if reasons:
            return self._hold(unix_time_ms, cycle_index, window_start_ms, current, reasons)
        evaluations = []
        uncertain_candidates = 0
        candidate_half_width = candidate_interval_half_width or {}
        for condition in adjacent_conditions(current):
            context = condition_parameters(condition, self.intensities, self.frequencies)
            prediction = predict_candidate({**context, **{f"previous_{key}": value for key, value in state.items()}})
            candidate_widths = [2.0 * float(candidate_half_width.get(name, 0.0)) for name in ("relaxation", "discomfort")]
            if max(candidate_widths, default=0.0) > self.uncertainty_width_max:
                uncertain_candidates += 1
                continue
            discomfort = min(1.0, float(prediction["discomfort"]) + float(candidate_half_width.get("discomfort", 0.0)))
            if discomfort >= self.discomfort_limit:
                continue
            gain = float(prediction["relaxation"] - float(state.get("relaxation") or 0.0))
            candidate_relaxation_lower = max(
                0.0,
                float(prediction["relaxation"]) - float(candidate_half_width.get("relaxation", 0.0)),
            )
            current_relaxation_upper = float(
                (intervals.get("relaxation") or [state.get("relaxation") or 0.0] * 2)[1]
            )
            conservative = candidate_relaxation_lower - current_relaxation_upper
            evaluations.append((conservative, gain, condition, discomfort))
        if not evaluations:
            reason = "candidate_uncertainty_too_high" if uncertain_candidates else "no_safe_candidate"
            return self._hold(unix_time_ms, cycle_index, window_start_ms, current, [reason])
        conservative, gain, condition, discomfort = max(evaluations)
        if condition == current or conservative < self.min_gain:
            return self._hold(unix_time_ms, cycle_index, window_start_ms, current, ["no_positive_conservative_gain"])
        return ConditionRecommendation(
            schema_version=self.schema_version,
            unix_time_ms=unix_time_ms,
            cycle_index=cycle_index,
            window_start_ms=window_start_ms,
            window_end_ms=window_start_ms + 10_000,
            current_condition=current,
            candidate_condition=condition,
            expected_relaxation_gain=gain,
            conservative_gain=conservative,
            predicted_discomfort=discomfort,
            safe=True,
            action="recommend",
            reasons=["highest_safe_conservative_relaxation_gain"],
            shadow=True,
        )

    def _hold(self, unix_time_ms: int, cycle_index: int, start_ms: int, current: str, reasons: list[str]) -> ConditionRecommendation:
        return ConditionRecommendation(
            schema_version=self.schema_version,
            unix_time_ms=unix_time_ms,
            cycle_index=cycle_index,
            window_start_ms=start_ms,
            window_end_ms=start_ms + 10_000,
            current_condition=current,
            candidate_condition=current,
            expected_relaxation_gain=None,
            conservative_gain=None,
            predicted_discomfort=None,
            safe=False,
            action="hold",
            reasons=reasons,
            shadow=True,
        )
