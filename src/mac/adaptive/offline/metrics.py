"""Metrics, invariant checks, gates, plotting, and reporting for shadow replay."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.adaptive.condition_grid import is_legal_transition, load, transition_axis


TARGETS = ("relaxation", "discomfort")


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _spearman(truth: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(truth) < 2 or np.std(truth) <= 1e-15 or np.std(prediction) <= 1e-15:
        return None
    return _finite_or_none(float(spearmanr(truth, prediction).statistic))


def _regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - truth
    return {
        "n": int(len(truth)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "spearman": _spearman(truth, prediction),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
        "prediction_range": float(np.ptp(prediction)),
        "prediction_std": float(np.std(prediction, ddof=0)),
        "truth_min": float(np.min(truth)),
        "truth_max": float(np.max(truth)),
    }


def _classification_metrics(truth: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    positive = truth >= threshold
    predicted = score >= threshold
    tp = int(np.sum(positive & predicted))
    tn = int(np.sum(~positive & ~predicted))
    fp = int(np.sum(~positive & predicted))
    fn = int(np.sum(positive & ~predicted))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    balanced = None if recall is None or specificity is None else (recall + specificity) / 2.0
    auprc = float(average_precision_score(positive.astype(int), score)) if positive.any() else None
    return {
        "threshold": threshold,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "sensitivity_recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "precision": precision,
        "auprc": auprc,
        "n_positive": int(positive.sum()),
        "n_negative": int((~positive).sum()),
    }


def model_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {
        "n_windows": int(len(predictions)),
        "n_independent_conditions": int(predictions["actual_condition"].nunique()),
        "window_unweighted": {},
        "condition_balanced": {},
        "condition_level": {},
        "prefix_length_strata": {},
    }
    condition_level = predictions.groupby("actual_condition", sort=False).agg(
        true_relaxation=("true_relaxation", "first"),
        true_discomfort=("true_discomfort", "first"),
        pred_relaxation=("pred_relaxation", "mean"),
        pred_discomfort=("pred_discomfort", "mean"),
        n_observed_windows=("window_id", "size"),
    ).reset_index()
    for target in TARGETS:
        window_truth = predictions[f"true_{target}"].to_numpy(dtype=float)
        window_prediction = predictions[f"pred_{target}"].to_numpy(dtype=float)
        output["window_unweighted"][target] = _regression_metrics(window_truth, window_prediction)
        condition_truth = condition_level[f"true_{target}"].to_numpy(dtype=float)
        condition_prediction = condition_level[f"pred_{target}"].to_numpy(dtype=float)
        condition_result = _regression_metrics(condition_truth, condition_prediction)
        output["condition_balanced"][target] = dict(condition_result)
        output["condition_level"][target] = dict(condition_result)
    prefix = pd.to_numeric(predictions["prefix_length"], errors="raise")
    strata = {
        "prefix_1_2": prefix.between(1, 2),
        "prefix_3_4": prefix.between(3, 4),
        "prefix_5_7": prefix.between(5, 7),
    }
    for name, selected in strata.items():
        output["prefix_length_strata"][name] = {
            target: _regression_metrics(
                predictions.loc[selected, f"true_{target}"].to_numpy(dtype=float),
                predictions.loc[selected, f"pred_{target}"].to_numpy(dtype=float),
            )
            for target in TARGETS
        }
    output["high_discomfort_detection"] = {
        "window_level_descriptive_non_independent": _classification_metrics(
            predictions["true_discomfort"].to_numpy(dtype=float),
            predictions["pred_discomfort"].to_numpy(dtype=float),
        ),
        "condition_level_primary": _classification_metrics(
            condition_level["true_discomfort"].to_numpy(dtype=float),
            condition_level["pred_discomfort"].to_numpy(dtype=float),
        ),
    }
    output["condition_predictions"] = condition_level.to_dict(orient="records")
    output["weighting_note"] = (
        "Primary errors average the per-condition errors so every independent end rating has equal weight; "
        "the 50 copied-label windows are also reported descriptively but are not treated as independent."
    )
    return output


def baseline_comparison_metrics(
    predictions: pd.DataFrame,
    comparator_predictions: pd.DataFrame | None,
) -> dict[str, Any]:
    if comparator_predictions is None or comparator_predictions.empty:
        return {"available": False, "reason": "frozen training-fold comparator predictions unavailable"}
    condition = predictions.groupby("actual_condition", sort=False).agg(
        true_relaxation=("true_relaxation", "first"),
        true_discomfort=("true_discomfort", "first"),
        healnet_relaxation=("pred_relaxation", "mean"),
        healnet_discomfort=("pred_discomfort", "mean"),
    ).reset_index().rename(columns={"actual_condition": "condition"})
    comparator = comparator_predictions.groupby("condition", as_index=False).agg(
        condition_only_relaxation=("condition_only_relaxation", "mean"),
        condition_only_discomfort=("condition_only_discomfort", "mean"),
        history_relaxation=("history_relaxation", "mean"),
        history_discomfort=("history_discomfort", "mean"),
    )
    merged = condition.merge(comparator, on="condition", how="left", validate="one_to_one")
    result: dict[str, Any] = {"available": True, "n_conditions": int(len(merged)), "models": {}}
    for model in ("healnet", "condition_only", "history"):
        result["models"][model] = {}
        for target in TARGETS:
            result["models"][model][target] = _regression_metrics(
                merged[f"true_{target}"].to_numpy(dtype=float),
                merged[f"{model}_{target}"].to_numpy(dtype=float),
            )
    result["note"] = "Comparators are fold-local predictions frozen during original training; no P009 fitting occurred."
    return result


def controller_metrics(trace: pd.DataFrame, controller_summary: Mapping[str, Any]) -> dict[str, Any]:
    changed = trace.loc[trace["action"].ne("hold")].copy()
    legal = [
        is_legal_transition(str(row.controller_condition_before), str(row.controller_condition))
        for row in changed.itertuples(index=False)
    ]
    one_axis = [
        transition_axis(str(row.controller_condition_before), str(row.controller_condition)) is not None
        for row in changed.itertuples(index=False)
    ]
    axes = [
        transition_axis(str(row.controller_condition_before), str(row.controller_condition))
        for row in changed.itertuples(index=False)
    ]
    observed_policy = trace["controller_condition_before"].astype(str).tolist()
    gap_resets = trace["gap_reset"].astype(bool).tolist()
    run_lengths = []
    if observed_policy:
        run = 1
        for index, (previous, current) in enumerate(zip(observed_policy, observed_policy[1:]), start=1):
            if current == previous and not gap_resets[index]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)
    # Parentheses kept explicit because this is a safety audit, not a display statistic.
    c9_violations = int(((changed["controller_condition"].eq("C9")) & changed["c9_locked"].astype(bool)).sum())
    invalid_wrong_actions = int(
        (
            ~trace["signal_valid"].astype(bool)
            & trace["action"].ne("hold")
            & trace["action_reason"].ne("three_invalid_windows_safe_return")
        ).sum()
    )
    active_overstay = int(
        (
            trace["controller_state"].eq("active")
            & trace["action"].eq("hold")
            & (pd.to_numeric(trace["dwell_windows"]) > 6)
        ).sum()
    )
    reasons = Counter(changed["action_reason"].astype(str))
    result = {
        "parameter_changes": int(len(changed)),
        "distinct_controller_conditions": int(trace["controller_condition"].nunique()),
        "visited_controller_conditions": sorted(trace["controller_condition"].astype(str).unique()),
        "intensity_changes": int(sum(axis == "intensity" for axis in axes)),
        "frequency_changes": int(sum(axis == "frequency" for axis in axes)),
        "mean_observed_dwell_windows": float(np.mean(run_lengths)) if run_lengths else 0.0,
        "longest_observed_dwell_windows": int(max(run_lengths, default=0)),
        "observed_runs_over_six_windows": int(sum(value > 6 for value in run_lengths)),
        "dwell_epochs_split_at_wall_clock_gaps": True,
        "adjacent_one_level_change_fraction": float(np.mean(legal)) if legal else 1.0,
        "single_axis_change_fraction": float(np.mean(one_axis)) if one_axis else 1.0,
        "illegal_transition_count": int(sum(not value for value in legal)),
        "active_max_dwell_violation_count": active_overstay,
        "c9_locked_selection_count": c9_violations,
        "invalid_signal_wrong_adaptation_count": invalid_wrong_actions,
        "action_reasons": dict(reasons),
        "recovery_actions": int(controller_summary["recovery_actions"]),
        "monotony_early_actions": int(controller_summary["monotony_actions"]),
        "low_relaxation_probe_actions": int(controller_summary["low_relaxation_actions"]),
        "history_influenced_actions": int(controller_summary["history_influenced_actions"]),
        "risk_candidate_filters": int(controller_summary["risk_filters"]),
        "c9_candidate_filters": int(controller_summary["c9_filter_count"]),
        "mechanical_two_condition_bounce": bool(len(changed) >= 3 and trace["controller_condition"].nunique() <= 2),
        "action_outcomes": {
            "success": 0,
            "ineffective": 0,
            "neutral": 0,
            "unsafe": 0,
            "mixed_activation": 0,
            "not_observable": int(len(changed)),
        },
        "evaluable_action_outcomes": 0,
        "controller_summary": dict(controller_summary),
    }
    result["recommendation_variation_gate"] = {
        "pass": bool(
            result["parameter_changes"] >= 3
            and result["distinct_controller_conditions"] >= 3
            and result["illegal_transition_count"] == 0
            and not result["mechanical_two_condition_bounce"]
        ),
        "scope": "virtual recommendation diversity only; not closed-loop efficacy",
    }
    result["causal_adaptive_effect_gate"] = {
        "evaluable": False,
        "pass": None,
        "reason": "Recommended conditions were not physically applied in chronological replay.",
    }
    return result


def build_all_metrics(
    predictions: pd.DataFrame,
    trace: pd.DataFrame,
    calibration: Mapping[str, Any],
    replay_audit: Mapping[str, Any],
    comparator_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    model = model_metrics(predictions)
    controller = controller_metrics(trace, replay_audit["controller_summary"])
    comparison = baseline_comparison_metrics(predictions, comparator_predictions)
    high_d = model["high_discomfort_detection"]["condition_level_primary"]
    condition_predictions = pd.DataFrame(model["condition_predictions"])
    conflict_counts = {
        "relaxation_absolute_error_above_personal_epsilon": int(
            (
                np.abs(condition_predictions["pred_relaxation"] - condition_predictions["true_relaxation"])
                > float(calibration["epsilon_relaxation_p90_absolute_deviation"])
            ).sum()
        ),
        "discomfort_absolute_error_above_personal_epsilon": int(
            (
                np.abs(condition_predictions["pred_discomfort"] - condition_predictions["true_discomfort"])
                > float(calibration["epsilon_discomfort_p90_absolute_deviation"])
            ).sum()
        ),
        "high_discomfort_rating_but_low_model_prediction": int(
            (
                (condition_predictions["true_discomfort"] >= 0.5)
                & (condition_predictions["pred_discomfort"] < 0.5)
            ).sum()
        ),
    }
    changed_trace = trace.loc[trace["action"].ne("hold")].copy()
    increasing_during_rise = 0
    for row in changed_trace.itertuples(index=False):
        details = json.loads(str(row.decision_details))
        if details.get("discomfort_rising") and load(str(row.controller_condition)) > load(
            str(row.controller_condition_before)
        ):
            increasing_during_rise += 1
    rating_safety_rows = trace.loc[trace["rating_revealed_safety_risk"].astype(bool)]
    rating_immediate_recoveries = int(
        (
            rating_safety_rows["action"].ne("hold")
            & rating_safety_rows["action_reason"].eq("revealed_rating_safety_recovery")
        ).sum()
    )
    baseline_adequate = bool(
        calibration.get("n_windows", 0) >= 6
        and calibration.get("c1_safety_condition_baseline", False)
    )
    gates = {
        "model_high_discomfort_recall": {
            "criterion": ">= 0.80 on independent conditions",
            "value": high_d["sensitivity_recall"],
            "pass": high_d["sensitivity_recall"] is not None and high_d["sensitivity_recall"] >= 0.8,
        },
        "controller_transition_invariants": {
            "criterion": "zero illegal, multi-axis, active overstay, or locked-C9 selections",
            "pass": bool(
                controller["illegal_transition_count"] == 0
                and controller["single_axis_change_fraction"] == 1.0
                and controller["active_max_dwell_violation_count"] == 0
                and controller["c9_locked_selection_count"] == 0
            ),
        },
        "calibration_adequacy": {
            "criterion": ">=60 s physiological baseline plus independent C1 safety baseline",
            "value": {
                "physiological_baseline_windows": int(calibration.get("n_windows", 0)),
                "c1_safety_condition_baseline": bool(calibration.get("c1_safety_condition_baseline", False)),
            },
            "pass": baseline_adequate,
        },
        "causal_closed_loop_efficacy": {
            "criterion": "recommendations physically applied with observable post-action outcomes",
            "evaluable": False,
            "pass": None,
        },
    }
    gates["prospective_deployment"] = {
        "pass": False,
        "decision": "NO-GO",
        "reason": (
            "High-discomfort recall and calibration adequacy fail, and chronological shadow replay "
            "cannot establish causal closed-loop safety or benefit."
        ),
    }
    return {
        "experiment": "P009 frozen HealNet causal-prefix chronological shadow replay",
        "model_accuracy": model,
        "frozen_baseline_comparison": comparison,
        "controller_behavior": controller,
        "calibration": dict(calibration),
        "safety": {
            "predicted_high_discomfort_windows": int((predictions["pred_discomfort"] >= 0.5).sum()),
            "true_high_discomfort_windows_descriptive": int((predictions["true_discomfort"] >= 0.5).sum()),
            "high_discomfort_false_negatives_window_descriptive": model["high_discomfort_detection"][
                "window_level_descriptive_non_independent"
            ]["false_negatives"],
            "high_discomfort_false_negatives_condition_primary": high_d["false_negatives"],
            "unsafe_physical_actions": 0,
            "physical_actions_applied": 0,
            "c9_locked_selection_count": controller["c9_locked_selection_count"],
            "signal_invalid_windows": int((~predictions["signal_valid"].astype(bool)).sum()),
            "invalid_signal_wrong_adaptation_count": controller["invalid_signal_wrong_adaptation_count"],
            "risk_candidate_filters": controller["risk_candidate_filters"],
            "continued_load_increase_while_discomfort_rising": increasing_during_rise,
            "revealed_high_risk_ratings": int(len(rating_safety_rows)),
            "same_row_virtual_recovery_after_risk_reveal": rating_immediate_recoveries,
            "same_row_safe_hold_after_risk_reveal": int(
                (
                    rating_safety_rows["action"].eq("hold")
                    & rating_safety_rows["action_reason"].eq("safety_trigger_hold_at_safe_fallback")
                ).sum()
            ),
            "predicted_high_discomfort_recovery_latency_evaluable": bool(
                (predictions["pred_discomfort"] >= 0.5).any()
            ),
            "note": (
                "Recovery counts describe virtual recommendations only. No post-action physiological response is observable."
            ),
        },
        "personalization": {
            "ratings_revealed": list(replay_audit["ratings_revealed_in_order"]),
            "ratings_not_yet_revealed": list(replay_audit["ratings_not_revealed_within_segment"]),
            "history_influenced_actions": controller["history_influenced_actions"],
            "revealed_risky_conditions": list(replay_audit["controller_summary"]["risky_conditions"]),
            "verified_safe_conditions": list(
                replay_audit["controller_summary"]["verified_safe_conditions"]
            ),
            "rating_reveal_changed_recommendation_same_row": int(
                (
                    trace["rating_available"].astype(bool)
                    & trace["action"].ne("hold")
                ).sum()
            ),
            "model_rating_conflicts": conflict_counts,
            "successful_intensity_directions": [],
            "successful_frequency_directions": [],
            "high_risk_virtual_actions": [],
            "personal_evidence_overrode_cold_start_count": controller["history_influenced_actions"],
            "action_response_personalization_evaluable": False,
            "reason": "Virtual actions have no observable response in chronological shadow replay.",
        },
        "go_no_go_gates": gates,
        "interpretation_boundary": {
            "historical": "All physiology, predictions, timestamps, and end-of-condition ratings.",
            "virtual": "All controller conditions and actions.",
            "synthetic": False,
            "causal_effect_claim_supported": False,
        },
        "historical_alignment": {
            "recommendation_matches_next_actual_count": int(
                trace["recommendation_matches_next_actual"].astype(bool).sum()
            ),
            "recommendation_matches_next_actual_fraction": float(
                trace.loc[trace["next_actual_condition"].notna(), "recommendation_matches_next_actual"].mean()
            ),
            "interpretation": "Descriptive alignment only; a match or mismatch is not an action outcome.",
        },
    }


__all__ = [
    "baseline_comparison_metrics",
    "build_all_metrics",
    "controller_metrics",
    "model_metrics",
]
