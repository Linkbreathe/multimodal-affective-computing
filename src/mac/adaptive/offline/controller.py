"""Deterministic, safety-first controller for chronological shadow replay.

The controller never receives the current condition label or outcome. Ratings
are supplied explicitly only after the corresponding historical condition has
finished. In shadow mode every recommendation is virtual, so an action outcome
is deliberately never inferred from later historical physiology.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Any, Mapping

from src.adaptive.condition_grid import (
    action_name,
    adjacent_conditions,
    coordinates,
    is_legal_transition,
    level_record,
    load,
    nearest_condition,
    one_step_toward,
    transition_axis,
)


@dataclass(frozen=True)
class ControllerConfig:
    baseline_relaxation: float
    baseline_discomfort: float
    epsilon_relaxation: float
    epsilon_discomfort: float
    smooth_windows: int = 3
    minimum_observation_windows: int = 3
    maximum_dwell_windows: int = 6
    high_discomfort_threshold: float = 0.5
    monotony_threshold: float = 4.0 / 6.0
    fallback_condition: str = "C1"

    def __post_init__(self) -> None:
        for name in (
            "baseline_relaxation",
            "baseline_discomfort",
            "epsilon_relaxation",
            "epsilon_discomfort",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.epsilon_relaxation < 0 or self.epsilon_discomfort < 0:
            raise ValueError("Personal variability thresholds cannot be negative")
        if self.smooth_windows != 3:
            raise ValueError("The frozen protocol requires a three-window smoother")


class AdaptiveController:
    """Stateful controller with a small causal input surface."""

    input_fields = (
        "pred_relaxation",
        "pred_discomfort",
        "signal_valid",
        "gap_reset",
        "revealed_rating",
    )

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.current_condition = config.fallback_condition
        self.controller_state = "warming_up"
        self.relaxation_history: deque[float] = deque(maxlen=config.smooth_windows)
        self.discomfort_history: deque[float] = deque(maxlen=config.smooth_windows)
        self.valid_since_reset = 0
        self.consecutive_invalid = 0
        self.dwell_windows = 0
        self.step_index = 0
        self.last_axis: str | None = None
        self.last_transition: tuple[str, str] | None = None
        self.visit_counts = {f"C{index}": 0 for index in range(1, 10)}
        self.last_visit = {f"C{index}": -1 for index in range(1, 10)}
        self.visit_counts[self.current_condition] = 1
        self.last_visit[self.current_condition] = 0
        self.rating_history: dict[str, dict[str, Any]] = {}
        self.risky_conditions: set[str] = set()
        self.verified_safe_conditions: set[str] = set()
        self.high_intensity_safe_evidence: set[str] = set()
        self.high_frequency_safe_evidence: set[str] = set()
        self.latest_rating: dict[str, Any] | None = None
        self.latest_visual_fit_direction: str | None = None
        self.rating_updates = 0
        self.risk_filters = 0
        self.c9_filter_count = 0
        self.recovery_actions = 0
        self.monotony_actions = 0
        self.low_relaxation_actions = 0
        self.invalid_fallback_actions = 0
        self.history_influenced_actions = 0

    def _reset_observation_history(self) -> None:
        self.relaxation_history.clear()
        self.discomfort_history.clear()
        self.valid_since_reset = 0

    def _smooth(self) -> tuple[float | None, float | None]:
        if not self.relaxation_history:
            return None, None
        return float(median(self.relaxation_history)), float(median(self.discomfort_history))

    def _discomfort_rising(self) -> bool:
        if len(self.discomfort_history) < 3:
            return False
        first, second, third = list(self.discomfort_history)[-3:]
        return (
            second > first
            and third > second
            and third - first > self.config.epsilon_discomfort
        )

    def _update_rating(self, rating: Mapping[str, Any] | None) -> tuple[str | None, bool]:
        if rating is None:
            return None, False
        condition = str(rating["condition"])
        if condition in self.rating_history:
            raise ValueError(f"Rating for {condition} was revealed more than once")
        record = dict(rating)
        self.rating_history[condition] = record
        self.latest_rating = record
        self.rating_updates += 1
        discomfort = float(record["discomfort"])
        direction_value = record.get("visual_fit_direction")
        direction = str(direction_value).lower() if direction_value not in (None, "", "nan") else None
        self.latest_visual_fit_direction = direction
        risky = discomfort >= self.config.high_discomfort_threshold or direction == "too_strong"
        visual_appropriate = bool(record.get("visual_fit_appropriate", False))
        if risky:
            self.risky_conditions.add(condition)
        elif visual_appropriate:
            self.verified_safe_conditions.add(condition)
            intensity, frequency = coordinates(condition)
            if intensity == 2 and frequency < 2:
                self.high_intensity_safe_evidence.add(condition)
            if frequency == 2 and intensity < 2:
                self.high_frequency_safe_evidence.add(condition)
        return condition, risky

    def c9_status(self, smooth_discomfort: float | None = None) -> tuple[bool, list[str]]:
        reasons = []
        if not self.high_intensity_safe_evidence:
            reasons.append("high_intensity_not_separately_verified_safe")
        if not self.high_frequency_safe_evidence:
            reasons.append("high_frequency_not_separately_verified_safe")
        if smooth_discomfort is None or smooth_discomfort >= self.config.high_discomfort_threshold:
            reasons.append("current_discomfort_not_verified_low")
        if self._discomfort_rising():
            reasons.append("current_discomfort_rising")
        if self.latest_rating is None:
            reasons.append("no_revealed_rating")
        else:
            if float(self.latest_rating["discomfort"]) >= self.config.high_discomfort_threshold:
                reasons.append("latest_rating_discomfort_high")
            if self.latest_visual_fit_direction is None:
                reasons.append("visual_fit_direction_unavailable_fail_closed")
            elif self.latest_visual_fit_direction == "too_strong":
                reasons.append("latest_rating_too_strong")
        return bool(reasons), reasons

    def _record_action(self, target: str) -> tuple[str, str]:
        source = self.current_condition
        if not is_legal_transition(source, target):
            raise AssertionError(f"Controller attempted illegal transition {source} -> {target}")
        action = action_name(source, target)
        axis = transition_axis(source, target)
        self.last_transition = (source, target)
        self.last_axis = axis
        self.current_condition = target
        self.visit_counts[target] += 1
        self.last_visit[target] = self.step_index
        self.dwell_windows = 0
        return action, axis or ""

    def _safe_target(self) -> str:
        candidates = set(self.verified_safe_conditions)
        candidates.add(self.config.fallback_condition)
        target = nearest_condition(self.current_condition, candidates)
        return target or self.config.fallback_condition

    def _recovery_target(self) -> str:
        if self.last_transition is not None and self.last_transition[1] == self.current_condition:
            source, target = self.last_transition
            if load(target) > load(source) and source not in self.risky_conditions:
                return source
        safe_target = self._safe_target()
        if safe_target != self.current_condition:
            return one_step_toward(self.current_condition, safe_target)
        if self.current_condition != self.config.fallback_condition:
            return one_step_toward(self.current_condition, self.config.fallback_condition)
        return self.current_condition

    def _candidate_rank(self, candidate: str, *, use_rating_utility: bool = True) -> tuple[Any, ...]:
        axis = transition_axis(self.current_condition, candidate)
        rating = self.rating_history.get(candidate)
        rated_utility = 0.0
        if rating is not None and use_rating_utility:
            rated_utility = float(rating.get("relaxation", 0.0)) + float(rating.get("pleasantness", 0.0))
        untested = candidate not in self.rating_history and self.visit_counts[candidate] == 0
        alternate_axis = self.last_axis is not None and axis != self.last_axis
        return (
            0 if untested else 1,
            self.visit_counts[candidate],
            self.last_visit[candidate],
            load(candidate),
            0 if alternate_axis else 1,
            -rated_utility,
            int(candidate[1:]),
        )

    def _choose_exploration(
        self,
        *,
        smooth_discomfort: float,
        low_relaxation_trigger: bool,
    ) -> tuple[str | None, dict[str, Any]]:
        raw = list(adjacent_conditions(self.current_condition))
        filtered_risk = [candidate for candidate in raw if candidate in self.risky_conditions]
        self.risk_filters += len(filtered_risk)
        candidates = [candidate for candidate in raw if candidate not in self.risky_conditions]
        c9_locked, c9_reasons = self.c9_status(smooth_discomfort)
        if "C9" in candidates and c9_locked:
            candidates.remove("C9")
            self.c9_filter_count += 1
        if low_relaxation_trigger:
            current_intensity = coordinates(self.current_condition)[0]
            candidates = [
                candidate
                for candidate in candidates
                if coordinates(candidate)[0] <= current_intensity
            ]
        if not candidates:
            return None, {
                "candidate_count": len(raw),
                "eligible_candidates": [],
                "risk_filtered": filtered_risk,
                "c9_lock_reasons": c9_reasons,
            }
        target = min(candidates, key=self._candidate_rank)
        no_rating_target = min(
            candidates,
            key=lambda candidate: self._candidate_rank(candidate, use_rating_utility=False),
        )
        history_used = bool(filtered_risk) or target != no_rating_target
        if history_used:
            self.history_influenced_actions += 1
        return target, {
            "candidate_count": len(raw),
            "eligible_candidates": candidates,
            "risk_filtered": filtered_risk,
            "c9_lock_reasons": c9_reasons,
            "history_used_in_ranking": history_used,
            "rating_utility_changed_selected_candidate": target != no_rating_target,
            "revealed_risk_changed_candidate_eligibility": bool(filtered_risk),
        }

    def observe(
        self,
        *,
        pred_relaxation: float,
        pred_discomfort: float,
        signal_valid: bool,
        gap_reset: bool = False,
        revealed_rating: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Consume one causal observation and return a virtual recommendation."""

        self.step_index += 1
        condition_before = self.current_condition
        revealed_condition, revealed_risk = self._update_rating(revealed_rating)
        reset_reason = None
        if gap_reset:
            self._reset_observation_history()
            # A questionnaire/repositioning gap is not time spent under the
            # virtual recommendation. Start a new contiguous dwell epoch while
            # retaining learned ratings and the last recommended condition.
            self.dwell_windows = 0
            reset_reason = "wall_clock_gap"

        if not signal_valid:
            self.consecutive_invalid += 1
            smooth_relaxation, smooth_discomfort = self._smooth()
            action = "hold"
            reason = "signal_invalid_hold"
            if self.consecutive_invalid >= 3:
                target = self._recovery_target()
                self.controller_state = "signal_safe_return"
                if target != self.current_condition:
                    action, _ = self._record_action(target)
                    self.invalid_fallback_actions += 1
                    reason = "three_invalid_windows_safe_return"
                else:
                    reason = "three_invalid_windows_already_at_safe_fallback"
            c9_locked, c9_reasons = self.c9_status(smooth_discomfort)
            return self._decision_record(
                condition_before=condition_before,
                action=action,
                reason=reason,
                smooth_relaxation=smooth_relaxation,
                smooth_discomfort=smooth_discomfort,
                c9_locked=c9_locked,
                c9_reasons=c9_reasons,
                revealed_condition=revealed_condition,
                revealed_risk=revealed_risk,
                reset_reason=reset_reason,
                details={},
            )

        if self.consecutive_invalid:
            self._reset_observation_history()
            reset_reason = "signal_recovery"
        self.consecutive_invalid = 0
        self.relaxation_history.append(float(pred_relaxation))
        self.discomfort_history.append(float(pred_discomfort))
        self.valid_since_reset += 1
        self.dwell_windows += 1
        smooth_relaxation, smooth_discomfort = self._smooth()
        assert smooth_relaxation is not None and smooth_discomfort is not None
        rising = self._discomfort_rising()
        high_discomfort = (
            self.valid_since_reset >= self.config.minimum_observation_windows
            and smooth_discomfort >= self.config.high_discomfort_threshold
        )
        safety_trigger = revealed_risk or high_discomfort or rising

        action = "hold"
        reason = "minimum_observation_warmup"
        details: dict[str, Any] = {
            "discomfort_rising": rising,
            "high_discomfort": high_discomfort,
            "rating_safety_trigger": revealed_risk,
        }
        if safety_trigger:
            target = self._recovery_target()
            self.controller_state = "recovering"
            if target != self.current_condition:
                action, _ = self._record_action(target)
                self.recovery_actions += 1
                reason = (
                    "revealed_rating_safety_recovery"
                    if revealed_risk
                    else "predicted_discomfort_safety_recovery"
                )
            else:
                reason = "safety_trigger_hold_at_safe_fallback"
        elif self.valid_since_reset < self.config.minimum_observation_windows:
            self.controller_state = "warming_up"
        elif self.controller_state in {"recovering", "signal_safe_return"}:
            if self.dwell_windows < self.config.minimum_observation_windows:
                reason = "recovery_minimum_observation"
            elif not rising and smooth_discomfort < self.config.high_discomfort_threshold:
                self.controller_state = "active"
                reason = "recovery_complete_hold"
            else:
                reason = "recovery_hold"
        else:
            self.controller_state = "active"
            monotony_trigger = bool(
                self.latest_rating is not None
                and float(self.latest_rating.get("monotony", 0.0)) >= self.config.monotony_threshold
                and self.dwell_windows >= self.config.minimum_observation_windows
            )
            low_relaxation_trigger = bool(
                smooth_relaxation
                < self.config.baseline_relaxation - self.config.epsilon_relaxation
                and self.dwell_windows >= self.config.minimum_observation_windows
            )
            max_dwell_trigger = self.dwell_windows >= self.config.maximum_dwell_windows
            if monotony_trigger or low_relaxation_trigger or max_dwell_trigger:
                target, candidate_details = self._choose_exploration(
                    smooth_discomfort=smooth_discomfort,
                    low_relaxation_trigger=low_relaxation_trigger and not max_dwell_trigger,
                )
                details.update(candidate_details)
                if target is not None:
                    action, _ = self._record_action(target)
                    if monotony_trigger:
                        self.monotony_actions += 1
                        reason = "revealed_monotony_early_exploration"
                    elif low_relaxation_trigger and not max_dwell_trigger:
                        self.low_relaxation_actions += 1
                        reason = "low_relaxation_reversible_non_intensity_probe"
                    else:
                        reason = "maximum_dwell_history_aware_exploration"
                else:
                    reason = "exploration_candidates_filtered_hold"
            else:
                reason = "active_observation_hold"

        c9_locked, c9_reasons = self.c9_status(smooth_discomfort)
        return self._decision_record(
            condition_before=condition_before,
            action=action,
            reason=reason,
            smooth_relaxation=smooth_relaxation,
            smooth_discomfort=smooth_discomfort,
            c9_locked=c9_locked,
            c9_reasons=c9_reasons,
            revealed_condition=revealed_condition,
            revealed_risk=revealed_risk,
            reset_reason=reset_reason,
            details=details,
        )

    def _decision_record(
        self,
        *,
        condition_before: str,
        action: str,
        reason: str,
        smooth_relaxation: float | None,
        smooth_discomfort: float | None,
        c9_locked: bool,
        c9_reasons: list[str],
        revealed_condition: str | None,
        revealed_risk: bool,
        reset_reason: str | None,
        details: Mapping[str, Any],
    ) -> dict[str, Any]:
        levels = level_record(self.current_condition)
        return {
            "controller_condition_before": condition_before,
            "controller_condition": self.current_condition,
            **levels,
            "smooth_relaxation_30s": smooth_relaxation,
            "smooth_discomfort_30s": smooth_discomfort,
            "state_delta_relaxation_vs_baseline": (
                None if smooth_relaxation is None else smooth_relaxation - self.config.baseline_relaxation
            ),
            "state_delta_discomfort_vs_baseline": (
                None if smooth_discomfort is None else smooth_discomfort - self.config.baseline_discomfort
            ),
            "delta_relaxation": None,
            "delta_discomfort": None,
            "controller_state": self.controller_state,
            "action": action,
            "action_reason": reason,
            "previous_action_outcome": "not_observable",
            "physical_action_applied": False,
            "action_outcome_observable": False,
            "dwell_windows": self.dwell_windows,
            "valid_since_reset": self.valid_since_reset,
            "consecutive_invalid": self.consecutive_invalid,
            "c9_locked": c9_locked,
            "c9_lock_reasons": "|".join(c9_reasons),
            "rating_available": revealed_condition is not None,
            "rating_revealed_condition": revealed_condition,
            "rating_revealed_safety_risk": revealed_risk,
            "observation_reset_reason": reset_reason,
            "synthetic_replay": False,
            "decision_details": dict(details),
        }

    def summary(self) -> dict[str, Any]:
        c9_locked, c9_reasons = self.c9_status(self._smooth()[1])
        return {
            "current_condition": self.current_condition,
            "visit_counts": dict(self.visit_counts),
            "revealed_conditions": list(self.rating_history),
            "risky_conditions": sorted(self.risky_conditions),
            "verified_safe_conditions": sorted(self.verified_safe_conditions),
            "fallback_condition": self.config.fallback_condition,
            "fallback_is_verified_safe": self.config.fallback_condition in self.verified_safe_conditions,
            "high_intensity_safe_evidence": sorted(self.high_intensity_safe_evidence),
            "high_frequency_safe_evidence": sorted(self.high_frequency_safe_evidence),
            "c9_locked": c9_locked,
            "c9_lock_reasons": c9_reasons,
            "rating_updates": self.rating_updates,
            "risk_filters": self.risk_filters,
            "c9_filter_count": self.c9_filter_count,
            "recovery_actions": self.recovery_actions,
            "monotony_actions": self.monotony_actions,
            "low_relaxation_actions": self.low_relaxation_actions,
            "invalid_fallback_actions": self.invalid_fallback_actions,
            "history_influenced_actions": self.history_influenced_actions,
            "action_outcomes_observed": 0,
        }


__all__ = ["AdaptiveController", "ControllerConfig"]
