from __future__ import annotations

from typing import Any


def deployment_guard(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    if metrics.get("unit_of_analysis") == "participant_condition":
        relaxation = metrics.get("targets", {}).get("relaxation", {})
        discomfort = metrics.get("targets", {}).get("discomfort", {})
        risk = discomfort.get("risk_at_fold_tuned_threshold", {})
        reasons = []
        if not (
            relaxation.get("mae", float("inf")) < relaxation.get("condition_only_baseline_mae", float("-inf"))
            and relaxation.get("mae", float("inf")) < relaxation.get("history_baseline_mae", float("-inf"))
            and relaxation.get("spearman", -1.0) > 0.0
        ):
            reasons.append("condition_level_relaxation_gate_failed")
        if not (
            discomfort.get("mae", float("inf")) < discomfort.get("condition_only_baseline_mae", float("-inf"))
            and discomfort.get("mae", float("inf")) < discomfort.get("history_baseline_mae", float("-inf"))
            and risk.get("per_row_threshold_recall", -1.0) >= 0.5
        ):
            reasons.append("condition_level_discomfort_gate_failed")
        return not reasons, reasons
    reasons: list[str] = []
    mae = metrics.get("mae", {})
    condition = metrics.get("condition_baseline_mae", {})
    history = metrics.get("history_baseline_mae", {})
    if not (
        mae.get("relaxation", float("inf")) < condition.get("relaxation", float("-inf"))
        and mae.get("relaxation", float("inf")) < history.get("relaxation", float("-inf"))
    ):
        reasons.append("relaxation_not_better_than_both_baselines")
    if not (
        mae.get("discomfort", float("inf")) < condition.get("discomfort", float("-inf"))
        and mae.get("discomfort", float("inf")) < history.get("discomfort", float("-inf"))
    ):
        reasons.append("discomfort_not_better_than_both_baselines")
    if metrics.get("spearman", {}).get("relaxation", -1.0) <= 0.0:
        reasons.append("relaxation_rank_correlation_not_positive")
    recall = metrics.get("discomfort_high_risk_recall")
    if recall is not None and recall == recall and recall < 0.5:
        reasons.append("discomfort_high_risk_recall_below_0_5")
    return not reasons, reasons
