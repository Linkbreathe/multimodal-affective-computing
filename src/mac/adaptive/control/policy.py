"""Permissive, explicitly adaptive-control-only policy over a pluggable model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import random
from typing import Any, Protocol

from .contracts import ControlProfile


@dataclass(frozen=True)
class ControlEstimate:
    relaxation: float
    discomfort: float
    model_variant: str
    active_modalities: list[str]
    raw: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None


class CandidateModel(Protocol):
    def predict_current(self, features: dict[str, float], condition: str, coverage: dict[str, float]) -> ControlEstimate: ...

    def predict_candidates(
        self,
        features: dict[str, float],
        current_condition: str,
        candidates: list[str],
        current: ControlEstimate,
    ) -> dict[str, ControlEstimate]: ...


@dataclass(frozen=True)
class ControlDecision:
    action: str
    current_condition: str
    target_condition: str
    estimate: ControlEstimate | None
    candidate_utilities: dict[str, float]
    utility_delta: float | None
    reasons: list[str]


class AdaptiveControlPolicy:
    def __init__(
        self,
        profile: ControlProfile,
        *,
        relaxation_weight: float,
        discomfort_weight: float,
        hysteresis: float,
        extreme_discomfort_limit: float,
        calm_exploration_dwell_windows: int = 3,
        calm_exploration_penalty_per_window: float = 0.025,
        calm_exploration_penalty_max: float = 0.12,
        calm_exploration_relaxation_min: float = 0.55,
        calm_exploration_discomfort_max: float = 0.40,
        stochastic_exploration_enabled: bool = False,
        exploration_candidate_scope: str = "adjacent",
        exploration_random_seed: int | None = None,
        exploration_temperature: float = 0.18,
        exploration_random_floor: float = 0.08,
        sensor_conditioning_enabled: bool = False,
        sensor_conditioning_weight: float = 1.0,
        switch_probability_enabled: bool = False,
        switch_probability_after_windows: list[float] | None = None,
        switch_probability_force_after_windows: int = 3,
        switch_probability_boredom_weight: float = 0.25,
        switch_probability_arousal_weight: float = 0.20,
        switch_probability_discomfort_weight: float = 0.30,
        switch_probability_stable_calm_weight: float = 0.15,
        safety_discomfort_min: float = 0.50,
        safety_conditions: list[str] | None = None,
        min_condition_dwell_windows: int = 0,
        max_condition_dwell_windows: int = 0,
        recent_history_window: int = 4,
        recent_history_penalty: float = 0.12,
        high_load_conditions: list[str] | None = None,
        high_load_penalty: float = 0.10,
        high_load_cooldown_windows: int = 0,
    ) -> None:
        self.profile = profile
        self.relaxation_weight = relaxation_weight
        self.discomfort_weight = discomfort_weight
        self.hysteresis = hysteresis
        self.extreme_discomfort_limit = extreme_discomfort_limit
        self.calm_exploration_dwell_windows = calm_exploration_dwell_windows
        self.calm_exploration_penalty_per_window = calm_exploration_penalty_per_window
        self.calm_exploration_penalty_max = calm_exploration_penalty_max
        self.calm_exploration_relaxation_min = calm_exploration_relaxation_min
        self.calm_exploration_discomfort_max = calm_exploration_discomfort_max
        self.stochastic_exploration_enabled = stochastic_exploration_enabled
        self.exploration_candidate_scope = exploration_candidate_scope
        self.exploration_temperature = exploration_temperature
        self.exploration_random_floor = exploration_random_floor
        self.sensor_conditioning_enabled = sensor_conditioning_enabled
        self.sensor_conditioning_weight = sensor_conditioning_weight
        self.switch_probability_enabled = switch_probability_enabled
        self.switch_probability_after_windows = list(switch_probability_after_windows or [0.10, 0.45, 0.85])
        self.switch_probability_force_after_windows = switch_probability_force_after_windows
        self.switch_probability_boredom_weight = switch_probability_boredom_weight
        self.switch_probability_arousal_weight = switch_probability_arousal_weight
        self.switch_probability_discomfort_weight = switch_probability_discomfort_weight
        self.switch_probability_stable_calm_weight = switch_probability_stable_calm_weight
        self.safety_discomfort_min = safety_discomfort_min
        self.safety_conditions = list(safety_conditions or ["C1", "C2", "C4", "C5"])
        self.min_condition_dwell_windows = min_condition_dwell_windows
        self.max_condition_dwell_windows = max_condition_dwell_windows
        self.recent_history_window = recent_history_window
        self.recent_history_penalty = recent_history_penalty
        self.high_load_conditions = set(high_load_conditions or ["C9"])
        self.high_load_penalty = high_load_penalty
        self.high_load_cooldown_windows = high_load_cooldown_windows
        self._rng = random.Random(exploration_random_seed)

    def utility(self, estimate: ControlEstimate) -> float:
        return self.relaxation_weight * estimate.relaxation + self.discomfort_weight * estimate.discomfort

    @staticmethod
    def _condition_sort_key(condition: str) -> tuple[int, str]:
        try:
            return (int(condition[1:]), condition)
        except ValueError:
            return (999, condition)

    def _all_conditions(self) -> list[str]:
        return sorted(self.profile.conditions, key=self._condition_sort_key)

    @staticmethod
    def _finite_feature(features: dict[str, float], name: str) -> float | None:
        try:
            value = float(features.get(name))
        except (TypeError, ValueError):
            return None
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value

    @staticmethod
    def _ramp(value: float | None, low: float, high: float) -> float:
        if value is None:
            return 0.0
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    @staticmethod
    def _inverse_ramp(value: float | None, low: float, high: float) -> float:
        if value is None:
            return 0.0
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (high - value) / (high - low)))

    def sensor_state(self, features: dict[str, float], estimate: ControlEstimate) -> dict[str, Any]:
        """Rule-based state summaries used only by the controller, not the ML estimator."""
        hr = self._finite_feature(features, "ecg_hr_bpm")
        rmssd = self._finite_feature(features, "ecg_hrv_30s_rmssd_ms")
        eye_velocity = self._finite_feature(features, "eye_angular_velocity_deg_s_mean")
        eye_saccade = self._finite_feature(features, "eye_saccade_fraction_ivt")
        head_speed = self._finite_feature(features, "head_speed_mean")
        head_angular = self._finite_feature(features, "head_angular_speed_deg_s_mean")
        stationary = self._finite_feature(features, "head_stationary_fraction")
        alpha_beta = self._finite_feature(features, "eeg_alpha_beta_ratio")
        theta_beta = self._finite_feature(features, "eeg_theta_beta_ratio")

        cardio_arousal = max(self._ramp(hr, 105.0, 145.0), self._inverse_ramp(rmssd, 80.0, 180.0) * 0.5)
        gaze_motion = max(self._ramp(eye_velocity, 8.0, 20.0), self._ramp(eye_saccade, 0.02, 0.10))
        head_motion = max(self._ramp(head_speed, 0.018, 0.045), self._ramp(head_angular, 9.0, 22.0))
        motion_arousal = max(gaze_motion, head_motion)
        stillness = min(
            self._inverse_ramp(eye_velocity, 1.2, 7.0),
            self._inverse_ramp(head_speed, 0.0035, 0.018),
            self._ramp(stationary, 0.80, 0.98),
        )
        eeg_calm = max(self._ramp(alpha_beta, 0.45, 0.80), self._inverse_ramp(theta_beta, 0.35, 0.80) * 0.5)
        eeg_tension = max(self._ramp(theta_beta, 1.25, 2.25), self._inverse_ramp(alpha_beta, 0.20, 0.35) * 0.5)
        calm_estimate = 1.0 if estimate.relaxation >= self.calm_exploration_relaxation_min and estimate.discomfort <= self.calm_exploration_discomfort_max else 0.0
        boredom = min(calm_estimate, stillness)
        physiological_arousal = max(cardio_arousal, motion_arousal, eeg_tension, self._ramp(estimate.discomfort, 0.35, 0.60))

        tags = []
        if cardio_arousal >= 0.65:
            tags.append("cardio_arousal")
        if motion_arousal >= 0.65:
            tags.append("motion_arousal")
        if stillness >= 0.65 and calm_estimate:
            tags.append("calm_stillness")
        if eeg_calm >= 0.65:
            tags.append("eeg_calm")
        if eeg_tension >= 0.65:
            tags.append("eeg_tension")
        if boredom >= 0.65:
            tags.append("possible_boredom")
        return {
            "cardio_arousal": cardio_arousal,
            "motion_arousal": motion_arousal,
            "stillness": stillness,
            "eeg_calm": eeg_calm,
            "eeg_tension": eeg_tension,
            "boredom": boredom,
            "physiological_arousal": physiological_arousal,
            "tags": tags,
        }

    def _condition_metrics(self, condition: str) -> tuple[float, float, float]:
        values = self.profile.values_for(condition)
        max_intensity = max(item.intensity for item in self.profile.conditions.values()) or 1.0
        max_frequency = max(item.frequency for item in self.profile.conditions.values()) or 1.0
        intensity = values.intensity / max_intensity
        frequency = values.frequency / max_frequency
        load = 0.5 * intensity + 0.5 * frequency
        return intensity, frequency, load

    def calm_dwell_penalty(self, estimate: ControlEstimate, dwell_windows: int) -> float:
        if dwell_windows < self.calm_exploration_dwell_windows:
            return 0.0
        if estimate.relaxation < self.calm_exploration_relaxation_min:
            return 0.0
        if estimate.discomfort > self.calm_exploration_discomfort_max:
            return 0.0
        extra_windows = dwell_windows - self.calm_exploration_dwell_windows + 1
        return min(
            self.calm_exploration_penalty_max,
            max(0.0, extra_windows * self.calm_exploration_penalty_per_window),
        )

    def _exploration_candidates(self, current_condition: str, cooldowns: dict[str, int]) -> list[str]:
        if self.exploration_candidate_scope == "all":
            candidates = [condition for condition in self._all_conditions() if condition != current_condition]
        else:
            candidates = list(self.profile.adjacent(current_condition))
        return [condition for condition in candidates if cooldowns.get(condition, 0) <= 0]

    def _selection_utilities(
        self,
        utilities: dict[str, float],
        *,
        current_condition: str,
        condition_history: list[str],
        features: dict[str, float] | None = None,
        estimate: ControlEstimate | None = None,
    ) -> dict[str, float]:
        selected = dict(utilities)
        recent = Counter(condition_history[-self.recent_history_window:]) if self.recent_history_window else Counter()
        state = self.sensor_state(features, estimate) if self.sensor_conditioning_enabled and features is not None and estimate is not None else None
        current_intensity, current_frequency, current_load = self._condition_metrics(current_condition)
        for condition in list(selected):
            if condition == current_condition:
                continue
            selected[condition] -= recent[condition] * self.recent_history_penalty
            if condition in self.high_load_conditions:
                selected[condition] -= self.high_load_penalty
            if state is not None:
                intensity, frequency, load = self._condition_metrics(condition)
                novelty = max(abs(intensity - current_intensity), abs(frequency - current_frequency), abs(load - current_load))
                mid_load = max(0.0, 1.0 - abs(load - 0.65) / 0.65)
                arousal = float(state["physiological_arousal"])
                motion = float(state["motion_arousal"])
                boredom = float(state["boredom"])
                eeg_calm = float(state["eeg_calm"])
                eeg_tension = float(state["eeg_tension"])
                adjustment = 0.0
                adjustment -= arousal * (0.10 * load + 0.04 * frequency)
                adjustment -= motion * (0.04 * intensity + 0.06 * frequency)
                adjustment -= eeg_tension * (0.08 * load)
                if condition in self.safety_conditions:
                    adjustment += arousal * 0.06
                adjustment += boredom * (0.12 * novelty + 0.04 * frequency - 0.03 * intensity)
                adjustment += eeg_calm * (0.08 * mid_load + 0.03 * novelty)
                selected[condition] += self.sensor_conditioning_weight * adjustment
        return selected

    def _weighted_choice(self, scores: dict[str, float], candidates: list[str]) -> str | None:
        if not candidates:
            return None
        max_score = max(scores[condition] for condition in candidates)
        weights = []
        for condition in candidates:
            scaled = (scores[condition] - max_score) / self.exploration_temperature
            weights.append(self.exploration_random_floor + math.exp(max(-60.0, min(60.0, scaled))))
        total = sum(weights)
        if total <= 0.0:
            return candidates[0]
        cursor = self._rng.random() * total
        for condition, weight in zip(candidates, weights):
            cursor -= weight
            if cursor <= 0.0:
                return condition
        return candidates[-1]

    def switch_probability(
        self,
        estimate: ControlEstimate,
        dwell_windows: int,
        state: dict[str, Any] | None,
    ) -> tuple[float, bool]:
        shown_windows = max(1, dwell_windows + 1)
        if shown_windows >= self.switch_probability_force_after_windows:
            return 1.0, True
        index = min(shown_windows - 1, len(self.switch_probability_after_windows) - 1)
        probability = self.switch_probability_after_windows[index]
        if state is not None:
            calm_estimate = (
                estimate.relaxation >= self.calm_exploration_relaxation_min
                and estimate.discomfort <= self.calm_exploration_discomfort_max
            )
            boredom = float(state.get("boredom", 0.0))
            arousal = float(state.get("physiological_arousal", 0.0))
            discomfort = self._ramp(estimate.discomfort, 0.25, self.safety_discomfort_min)
            stable_calm = 0.0
            if calm_estimate:
                stable_calm = min(
                    self._inverse_ramp(arousal, 0.20, 0.60),
                    self._inverse_ramp(boredom, 0.20, 0.65),
                )
            probability += self.switch_probability_boredom_weight * boredom
            probability += self.switch_probability_arousal_weight * arousal
            probability += self.switch_probability_discomfort_weight * discomfort
            probability -= self.switch_probability_stable_calm_weight * stable_calm
        return max(0.0, min(1.0, probability)), False

    def _stochastic_apply(
        self,
        model: CandidateModel,
        features: dict[str, float],
        current_condition: str,
        estimate: ControlEstimate,
        current_utility: float,
        candidates: list[str],
        condition_history: list[str],
        reason: str,
    ) -> ControlDecision | None:
        candidate_estimates = model.predict_candidates(features, current_condition, candidates, estimate)
        utilities = {current_condition: current_utility}
        for condition, candidate_estimate in candidate_estimates.items():
            utilities[condition] = self.utility(candidate_estimate)
        selection_utilities = self._selection_utilities(
            utilities,
            current_condition=current_condition,
            condition_history=condition_history,
            features=features,
            estimate=estimate,
        )
        available = [condition for condition in candidates if condition in candidate_estimates]
        target = self._weighted_choice(selection_utilities, available)
        if target is None:
            return None
        return ControlDecision(
            action="apply",
            current_condition=current_condition,
            target_condition=target,
            estimate=estimate,
            candidate_utilities=selection_utilities,
            utility_delta=selection_utilities[target] - selection_utilities[current_condition],
            reasons=[reason],
        )

    def decide(
        self,
        model: CandidateModel,
        features: dict[str, float],
        current_condition: str,
        coverage: dict[str, float],
        dwell_windows: int = 0,
        condition_history: list[str] | None = None,
        condition_cooldowns: dict[str, int] | None = None,
    ) -> ControlDecision:
        history = list(condition_history or [])
        cooldowns = dict(condition_cooldowns or {})
        estimate = model.predict_current(features, current_condition, coverage)
        current_utility = self.utility(estimate)
        utilities = {current_condition: current_utility}
        if estimate.discomfort >= self.extreme_discomfort_limit:
            return ControlDecision(
                action="failsafe",
                current_condition=current_condition,
                target_condition=current_condition,
                estimate=estimate,
                candidate_utilities=utilities,
                utility_delta=None,
                reasons=["extreme_predicted_discomfort"],
            )
        if self.stochastic_exploration_enabled and estimate.discomfort >= self.safety_discomfort_min:
            if current_condition in self.safety_conditions and dwell_windows < self.min_condition_dwell_windows:
                return ControlDecision(
                    action="hold",
                    current_condition=current_condition,
                    target_condition=current_condition,
                    estimate=estimate,
                    candidate_utilities=utilities,
                    utility_delta=0.0,
                    reasons=["safety_min_condition_dwell"],
                )
            safety_candidates = [
                condition
                for condition in self.safety_conditions
                if condition in self.profile.conditions and condition != current_condition and cooldowns.get(condition, 0) <= 0
            ]
            decision = self._stochastic_apply(
                model,
                features,
                current_condition,
                estimate,
                current_utility,
                safety_candidates,
                history,
                "safety_recovery_stochastic",
            )
            if decision is not None:
                return decision
            return ControlDecision(
                action="hold",
                current_condition=current_condition,
                target_condition=current_condition,
                estimate=estimate,
                candidate_utilities=utilities,
                utility_delta=0.0,
                reasons=["safety_no_available_condition"],
            )
        if (
            self.stochastic_exploration_enabled
            and not self.switch_probability_enabled
            and dwell_windows < self.min_condition_dwell_windows
        ):
            return ControlDecision(
                action="hold",
                current_condition=current_condition,
                target_condition=current_condition,
                estimate=estimate,
                candidate_utilities=utilities,
                utility_delta=0.0,
                reasons=["min_condition_dwell"],
            )
        if self.stochastic_exploration_enabled:
            state = self.sensor_state(features, estimate) if self.sensor_conditioning_enabled else None
            if state is not None:
                estimate.raw["controller_sensor_state"] = state
            calm = (
                estimate.relaxation >= self.calm_exploration_relaxation_min
                and estimate.discomfort <= self.calm_exploration_discomfort_max
            )
            steady = estimate.relaxation >= 0.45 and estimate.discomfort <= self.safety_discomfort_min
            if self.switch_probability_enabled:
                probability, forced = self.switch_probability(estimate, dwell_windows, state)
                draw = 0.0 if forced else self._rng.random()
                estimate.raw["controller_switch_probability"] = probability
                estimate.raw["controller_switch_draw"] = draw
                probability_ready = (
                    forced
                    or calm
                    or steady
                    or (state is not None and float(state.get("physiological_arousal", 0.0)) >= 0.35)
                    or (state is not None and float(state.get("boredom", 0.0)) >= 0.35)
                )
                if probability_ready and draw <= probability:
                    reason = "sensor_probability_force_exploration" if forced else "sensor_probability_exploration"
                    candidates = self._exploration_candidates(current_condition, cooldowns)
                    decision = self._stochastic_apply(
                        model,
                        features,
                        current_condition,
                        estimate,
                        current_utility,
                        candidates,
                        history,
                        reason,
                    )
                    if decision is not None:
                        return decision
                    return ControlDecision(
                        action="hold",
                        current_condition=current_condition,
                        target_condition=current_condition,
                        estimate=estimate,
                        candidate_utilities=utilities,
                        utility_delta=0.0,
                        reasons=["sensor_probability_no_available_condition"],
                    )
                if calm or steady or probability_ready:
                    return ControlDecision(
                        action="hold",
                        current_condition=current_condition,
                        target_condition=current_condition,
                        estimate=estimate,
                        candidate_utilities=utilities,
                        utility_delta=0.0,
                        reasons=["sensor_probability_hold"],
                    )
            else:
                max_dwell_ready = (
                    self.max_condition_dwell_windows > 0
                    and dwell_windows >= self.max_condition_dwell_windows
                    and estimate.discomfort < self.safety_discomfort_min
                )
                calm_dwell_ready = calm and dwell_windows >= self.calm_exploration_dwell_windows
                if max_dwell_ready or calm_dwell_ready:
                    reason = "stochastic_max_dwell_exploration" if max_dwell_ready else "stochastic_calm_exploration"
                    candidates = self._exploration_candidates(current_condition, cooldowns)
                    decision = self._stochastic_apply(
                        model,
                        features,
                        current_condition,
                        estimate,
                        current_utility,
                        candidates,
                        history,
                        reason,
                    )
                    if decision is not None:
                        return decision
            if calm or steady:
                reason = "calm_wait_for_stochastic_exploration" if calm else "steady_wait_for_stochastic_exploration"
                return ControlDecision(
                    action="hold",
                    current_condition=current_condition,
                    target_condition=current_condition,
                    estimate=estimate,
                    candidate_utilities=utilities,
                    utility_delta=0.0,
                    reasons=[reason],
                )
        candidates = self.profile.adjacent(current_condition)
        candidate_estimates = model.predict_candidates(features, current_condition, candidates, estimate)
        for condition, candidate_estimate in candidate_estimates.items():
            utilities[condition] = self.utility(candidate_estimate)
        selection_utilities = dict(utilities)
        dwell_penalty = self.calm_dwell_penalty(estimate, dwell_windows)
        if dwell_penalty > 0.0:
            selection_utilities[current_condition] = current_utility - dwell_penalty
        target, target_utility = max(selection_utilities.items(), key=lambda item: (item[1], item[0]))
        delta = target_utility - selection_utilities[current_condition]
        if target != current_condition and delta >= self.hysteresis:
            target_estimate = candidate_estimates.get(target)
            reason = "adaptive_utility_step"
            if dwell_penalty > 0.0 and utilities.get(target, float("-inf")) <= current_utility:
                reason = "calm_dwell_exploration"
            elif target_estimate is not None:
                mode = str(target_estimate.raw.get("candidate_policy_mode", ""))
                if mode == "stable_probe":
                    reason = "stable_probe"
                elif mode == "calm_exploration":
                    reason = "calm_exploration"
                elif mode == "stress_recovery":
                    reason = "stress_recovery_step"
            return ControlDecision(
                action="apply",
                current_condition=current_condition,
                target_condition=target,
                estimate=estimate,
                candidate_utilities=selection_utilities,
                utility_delta=delta,
                reasons=[reason],
            )
        hold_reason = "dwell_exploration_no_better_neighbor" if dwell_penalty > 0.0 else "stable_no_better_neighbor"
        return ControlDecision(
            action="hold",
            current_condition=current_condition,
            target_condition=current_condition,
            estimate=estimate,
            candidate_utilities=selection_utilities,
            utility_delta=delta,
            reasons=[hold_reason],
        )
