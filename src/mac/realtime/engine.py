from __future__ import annotations

import time
from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.modeling.dcnn import MODEL_KIND, load_dcnn_state_model, predict_dcnn_state
from real_time_ml.modeling.video_dcnn import VIDEO_MODEL_KIND, load_video_dcnn_model, predict_video_dcnn_state
from real_time_ml.modeling.video_ridge import VIDEO_RIDGE_KIND
from real_time_ml.modeling.train import load_state_model, predict_state
from real_time_ml.modeling.condition_data import aggregate_realtime_history
from real_time_ml.modeling.condition_train import predict_condition_risk
from real_time_ml.modeling.safety import deployment_guard
from real_time_ml.policy.recommender import SafetyPolicy
from real_time_ml.schema import ConditionRecommendation, StatePrediction


def _load_runtime_state_model(path, config: ProjectConfig) -> dict[str, Any]:
    """Load only the two-target state bundles permitted by the realtime contract."""
    bundle = load_state_model(path)
    if bool(bundle.get("research_only", False)):
        raise ValueError(f"Realtime inference refuses research-only model bundle: {path}")
    expected_targets = list(config.get("modeling.targets"))
    if bundle.get("targets") != expected_targets:
        raise ValueError(
            f"Realtime inference requires targets {expected_targets}; got {bundle.get('targets')} from {path}"
        )
    return bundle


class InferenceEngine:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.runtime_backend = str(config.get("modeling.runtime_backend", "classical"))
        self.models: dict[str, dict[str, Any]] = {}
        self.condition_histories: dict[tuple[str | None, str | None, str], list[dict[str, Any]]] = {}
        if self.runtime_backend in {"dcnn", "video_dcnn"}:
            for variant in config.get("modeling.fallback_chain"):
                path = config.path("models") / f"dcnn_state_{variant}.pt"
                if path.exists():
                    loader = load_video_dcnn_model if self.runtime_backend == "video_dcnn" else load_dcnn_state_model
                    self.models[variant] = loader(path, str(config.get("modeling.dcnn.device", "cuda")))
        else:
            for variant in config.get("modeling.fallback_chain"):
                path = config.path("models") / f"state_model_{variant}.joblib"
                if path.exists():
                    self.models[variant] = _load_runtime_state_model(path, config)
            default = config.path("models") / "state_model.joblib"
            if default.exists() and "full" not in self.models:
                self.models["full"] = _load_runtime_state_model(default, config)
        self.policy_bundle = None
        policy_path = config.path("models") / "policy_model.joblib"
        if policy_path.exists():
            self.policy_bundle = load_state_model(policy_path)
        self.policy = SafetyPolicy(
            schema_version=config.data["schema_version"],
            intensities=list(config.get("conditions.intensities")),
            frequencies=list(config.get("conditions.frequencies")),
            discomfort_limit=float(config.get("policy.discomfort_limit_normalized")),
            uncertainty_width_max=float(config.get("policy.uncertainty_width_max")),
            min_modality_coverage=float(config.get("policy.min_modality_coverage")),
            min_gain=float(config.get("policy.min_conservative_relaxation_gain")),
        )

    def _select_model(self, coverage: dict[str, float]) -> tuple[str | None, list[str]]:
        if self.runtime_backend == "video_dcnn":
            missing = [name for name in ("eeg", "ecg", "head", "eye", "video") if coverage.get(name, 0.0) < 0.5]
            if coverage.get("video", 0.0) < 0.5 and "no_video" in self.models:
                return "no_video", missing
            if "full" in self.models:
                return "full", missing
            if "no_video" in self.models:
                return "no_video", missing
            return None, missing
        missing = [name for name in ("eeg", "ecg", "head", "eye") if coverage.get(name, 0.0) < 0.5]
        for variant in self.config.get("modeling.fallback_chain"):
            if variant not in self.models:
                continue
            if variant == "full" and "eeg" not in missing:
                return variant, missing
            if variant == "no_eeg" and all(name not in missing for name in ("ecg", "head", "eye")):
                return variant, missing
            if variant == "behavior_only" and any(coverage.get(name, 0.0) >= 0.5 for name in ("head", "eye")):
                return variant, missing
        return None, missing

    def infer(
        self,
        *,
        participant_id: str | None,
        condition: str | None,
        cycle_index: int,
        start_ms: int,
        end_ms: int,
        features: dict[str, Any],
        qc: dict[str, Any],
        coverage: dict[str, float],
        force_hold_reasons: list[str] | None = None,
    ) -> tuple[StatePrediction, ConditionRecommendation]:
        variant, missing = self._select_model(coverage)
        reasons = list(force_hold_reasons or [])
        if variant is None:
            prediction = {name: None for name in self.config.get("modeling.targets")}
            intervals = {name: None for name in prediction}
            reasons.append("no_usable_fallback_model")
            deployable = False
        else:
            bundle = self.models[variant]
            active_features = features
            history: list[dict[str, Any]] | None = None
            if bundle.get("model_kind") in {"condition_residual_ensemble_v1", VIDEO_RIDGE_KIND, MODEL_KIND, VIDEO_MODEL_KIND}:
                history_key = (participant_id, condition, variant)
                if cycle_index == 0:
                    self.condition_histories[history_key] = []
                history = self.condition_histories.setdefault(history_key, [])
                history.append(dict(features))
            if bundle.get("model_kind") in {"condition_residual_ensemble_v1", VIDEO_RIDGE_KIND}:
                static = {"condition": condition}
                if condition:
                    from real_time_ml.data.io import condition_parameters

                    static.update(condition_parameters(condition, list(self.config.get("conditions.intensities")), list(self.config.get("conditions.frequencies"))))
                active_features = aggregate_realtime_history(history, bundle["feature_columns"], static)
            if bundle.get("model_kind") == MODEL_KIND:
                if condition is None:
                    prediction = {name: None for name in self.config.get("modeling.targets")}
                    intervals = {name: None for name in prediction}
                    deployable = False
                    reasons.append("current_condition_missing")
                else:
                    prediction, half_widths, history_windows = predict_dcnn_state(bundle, history or [], condition, self.config)
                    intervals = {
                        name: [
                            max(0.0, value - float(half_widths.get(name, 0.5))),
                            min(1.0, value + float(half_widths.get(name, 0.5))),
                        ]
                        for name, value in prediction.items()
                    }
                    qc["dcnn_history_windows"] = history_windows
                    qc["dcnn_sequence_complete"] = history_windows >= int(bundle["architecture"]["sequence_length"])
                    if history_windows < int(self.config.get("modeling.dcnn.minimum_history_windows", 3)):
                        reasons.append("dcnn_insufficient_history")
                    guard_ok, guard_reasons = deployment_guard(bundle.get("metrics", {}))
                    deployable = bool(bundle.get("deployable", False) and guard_ok)
                    if not guard_ok:
                        reasons.extend(guard_reasons)
            elif bundle.get("model_kind") == VIDEO_MODEL_KIND:
                if condition is None:
                    prediction = {name: None for name in self.config.get("modeling.targets")}
                    intervals = {name: None for name in prediction}
                    deployable = False
                    reasons.append("current_condition_missing")
                else:
                    prediction, half_widths, history_windows = predict_video_dcnn_state(bundle, history or [], condition, self.config)
                    intervals = {
                        name: [max(0.0, value - float(half_widths.get(name, 0.5))), min(1.0, value + float(half_widths.get(name, 0.5)))]
                        for name, value in prediction.items()
                    }
                    qc["dcnn_history_windows"] = history_windows
                    qc["dcnn_sequence_complete"] = history_windows >= int(bundle["architecture"]["sequence_length"])
                    qc["video_model_offline_only"] = True
                    if history_windows < int(self.config.get("modeling.dcnn.minimum_history_windows", 3)):
                        reasons.append("dcnn_insufficient_history")
                    guard_ok, guard_reasons = deployment_guard(bundle.get("metrics", {}))
                    deployable = bool(bundle.get("deployable", False) and guard_ok)
                    if not guard_ok:
                        reasons.extend(guard_reasons)
            else:
                prediction = predict_state(bundle, active_features, condition)
                half_widths = bundle.get("interval_half_width", {name: 0.25 for name in prediction})
                intervals = {
                    name: [max(0.0, value - float(half_widths.get(name, 0.25))), min(1.0, value + float(half_widths.get(name, 0.25)))]
                    for name, value in prediction.items()
                }
                guard_ok, guard_reasons = deployment_guard(bundle.get("metrics", {}))
                deployable = bool(bundle.get("deployable", False) and guard_ok)
                if not guard_ok:
                    reasons.extend(guard_reasons)
                if bundle.get("model_kind") == "condition_residual_ensemble_v1":
                    risk_probability = predict_condition_risk(bundle, active_features, __import__("pandas"))
                    qc["high_discomfort_probability"] = risk_probability
                    qc["high_discomfort_probability_threshold"] = bundle.get("risk_probability_threshold")
                    if risk_probability >= float(bundle.get("risk_probability_threshold", 1.0)):
                        reasons.append("risk_classifier_high_discomfort")
        state = StatePrediction(
            schema_version=self.config.data["schema_version"],
            unix_time_ms=int(time.time() * 1000),
            cycle_index=cycle_index,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
            participant_id=participant_id,
            condition=condition,
            predictions=prediction,
            intervals=intervals,
            modality_coverage=coverage,
            qc=qc,
            model_variant=variant or "none",
            missing_modalities=missing,
            degraded=bool(missing or reasons or not deployable),
        )
        if condition is None or any(value is None for value in prediction.values()) or self.policy_bundle is None:
            if condition is None:
                condition = "C1"
                reasons.append("current_condition_missing")
            if self.policy_bundle is None:
                reasons.append("policy_model_missing")
            recommendation = self.policy._hold(int(time.time() * 1000), cycle_index, start_ms, condition, reasons)
        else:
            def candidate_predict(candidate: dict[str, Any]) -> dict[str, float]:
                context = dict(candidate)
                for target, value in prediction.items():
                    context.setdefault(f"participant_baseline_{target}", value)
                return predict_state(self.policy_bundle, context)

            recommendation = self.policy.recommend(
                unix_time_ms=int(time.time() * 1000),
                cycle_index=cycle_index,
                window_start_ms=start_ms,
                current_condition=condition,
                state=prediction,
                intervals=intervals,
                modality_coverage=coverage,
                model_deployable=deployable,
                predict_candidate=candidate_predict,
                force_hold_reasons=reasons,
                candidate_interval_half_width=self.policy_bundle.get("interval_half_width", {}),
            )
        return state, recommendation
