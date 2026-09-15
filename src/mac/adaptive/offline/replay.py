"""Causal orchestration for chronological (non-interventional) shadow replay."""

from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

from mac.adaptive.offline.condition_grid import level_record
from mac.adaptive.offline.controller import AdaptiveController


RATING_FIELDS = (
    "condition",
    "relaxation",
    "discomfort",
    "pleasantness",
    "calm",
    "monotony",
    "visual_fit_appropriate",
    "visual_fit_direction",
)


def ratings_from_labels(labels: pd.DataFrame, participant_id: str = "P009") -> dict[str, dict[str, Any]]:
    selected = labels.loc[labels["participant_id"].astype(str).eq(participant_id)].copy()
    if selected["condition"].duplicated().any() or len(selected) != 9:
        raise ValueError(f"Expected nine unique condition ratings for {participant_id}")
    ratings: dict[str, dict[str, Any]] = {}
    for row in selected.itertuples(index=False):
        visual_fit = float(getattr(row, "visual_fit"))
        ratings[str(row.condition)] = {
            "condition": str(row.condition),
            "relaxation": float(row.relaxation),
            "discomfort": float(row.discomfort),
            "pleasantness": float(row.pleasantness),
            "calm": float(row.calm),
            "monotony": float(row.monotony),
            "visual_fit_appropriate": visual_fit == 1.0,
            # The source contract collapsed Too weak and Too strong to zero.
            # Direction is therefore intentionally unavailable and fail-closed.
            "visual_fit_direction": None,
        }
    return ratings


def run_chronological_replay(
    predictions: pd.DataFrame,
    ratings: Mapping[str, Mapping[str, Any]],
    controller: AdaptiveController,
    *,
    gap_threshold_seconds: float = 15.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "participant_id",
        "window_id",
        "window_start_unix_ms",
        "window_end_unix_ms",
        "actual_condition",
        "actual_intensity_index",
        "actual_frequency_index",
        "pred_relaxation",
        "pred_discomfort",
        "true_relaxation",
        "true_discomfort",
        "signal_valid",
    }
    missing = required - set(predictions)
    if missing:
        raise ValueError(f"Prefix prediction table lacks replay fields: {sorted(missing)}")
    frame = predictions.sort_values("window_start_unix_ms").reset_index(drop=True)
    if len(frame) != 50:
        raise ValueError(f"Frozen chronological replay requires exactly 50 windows; received {len(frame)}")
    if set(frame["participant_id"].astype(str)) != {"P009"}:
        raise ValueError("Frozen chronological replay is restricted to P009")
    if frame["window_start_unix_ms"].duplicated().any():
        raise ValueError("Replay timestamps are not unique")

    first_start = float(frame.iloc[0]["window_start_unix_ms"])
    rows: list[dict[str, Any]] = []
    revealed: list[str] = []
    previous_actual: str | None = None
    previous_end: float | None = None
    for index, row in frame.iterrows():
        actual = str(row["actual_condition"])
        transition = previous_actual is not None and actual != previous_actual
        gap_seconds = 0.0 if previous_end is None else max(
            0.0, (float(row["window_start_unix_ms"]) - previous_end) / 1000.0
        )
        gap_reset = gap_seconds > gap_threshold_seconds
        revealed_rating = None
        if transition:
            if previous_actual in revealed:
                raise AssertionError(f"Rating for {previous_actual} would be revealed twice")
            revealed_rating = dict(ratings[previous_actual])
            revealed.append(previous_actual)
        if actual in revealed and not transition:
            # A condition is never revisited in this frozen segment. Keeping this
            # assertion makes future protocol changes fail loudly rather than
            # accidentally exposing a rating before a revisit.
            raise AssertionError(f"Frozen replay unexpectedly revisits already revealed {actual}")
        decision = controller.observe(
            pred_relaxation=float(row["pred_relaxation"]),
            pred_discomfort=float(row["pred_discomfort"]),
            signal_valid=bool(row["signal_valid"]),
            gap_reset=gap_reset,
            revealed_rating=revealed_rating,
        )
        actual_levels = level_record(actual)
        output = row.to_dict()
        output.update(
            {
                "replay_index": int(index),
                "time_s": float((index + 1) * 10),
                "wall_elapsed_s": (float(row["window_end_unix_ms"]) - first_start) / 1000.0,
                "actual_condition_transition": transition,
                "wall_clock_gap_s": gap_seconds,
                "gap_reset": gap_reset,
                "actual_intensity_level": actual_levels["intensity_level"],
                "actual_frequency_level": actual_levels["frequency_level"],
                "actual_intensity_value": actual_levels["intensity_value"],
                "actual_frequency_value": actual_levels["frequency_value"],
                **decision,
            }
        )
        output["decision_details"] = json.dumps(output["decision_details"], ensure_ascii=False, sort_keys=True)
        rows.append(output)
        previous_actual = actual
        previous_end = float(row["window_end_unix_ms"])

    trace = pd.DataFrame(rows)
    next_actual = trace["actual_condition"].shift(-1)
    trace["next_actual_condition"] = next_actual
    trace["recommendation_matches_next_actual"] = (
        trace["controller_condition"].eq(next_actual) & next_actual.notna()
    )
    audit = {
        "replay_type": "chronological_shadow_replay",
        "synthetic_replay": False,
        "physical_actions_applied": 0,
        "action_outcomes_observable": 0,
        "controller_input_fields": list(controller.input_fields),
        "controller_forbidden_current_fields": [
            "actual_condition",
            "actual_intensity_index",
            "actual_frequency_index",
            "true_relaxation",
            "true_discomfort",
            "pleasantness",
            "calm",
            "monotony",
            "visual_fit",
        ],
        "ratings_revealed_in_order": revealed,
        "ratings_not_revealed_within_segment": sorted(set(ratings) - set(revealed)),
        "first_rating_reveal_row": {
            condition: int(trace.index[trace["rating_revealed_condition"].eq(condition)][0])
            for condition in revealed
        },
        "current_condition_rating_never_passed": all(
            pd.isna(row.rating_revealed_condition)
            or str(row.rating_revealed_condition) != str(row.actual_condition)
            for row in trace.itertuples(index=False)
        ),
        "visual_fit_direction_available": False,
        "visual_fit_direction_rule_enabled": False,
        "post_recommendation_action_effect_classification_enabled": False,
        "wall_clock_gap_threshold_seconds": gap_threshold_seconds,
        "wall_clock_gap_count": int(trace["gap_reset"].sum()),
        "controller_summary": controller.summary(),
    }
    return trace, audit


__all__ = ["RATING_FIELDS", "ratings_from_labels", "run_chronological_replay"]
