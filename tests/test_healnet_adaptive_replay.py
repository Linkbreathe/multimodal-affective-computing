from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.adaptive.condition_grid import (
    CONDITION_GRID,
    adjacent_conditions,
    coordinates,
    is_legal_transition,
    transition_axis,
)
from src.adaptive.controller import AdaptiveController, ControllerConfig
from src.adaptive.healnet_prefix import EXPECTED_MODALITIES, FrozenHealNetEnsemble
from src.adaptive.replay import run_chronological_replay


def _config() -> ControllerConfig:
    return ControllerConfig(
        baseline_relaxation=0.5,
        baseline_discomfort=0.1,
        epsilon_relaxation=0.05,
        epsilon_discomfort=0.03,
    )


def _observe_valid(controller: AdaptiveController, count: int, relaxation: float = 0.5, discomfort: float = 0.1):
    return [
        controller.observe(
            pred_relaxation=relaxation,
            pred_discomfort=discomfort,
            signal_valid=True,
        )
        for _ in range(count)
    ]


def test_condition_grid_has_only_single_axis_adjacent_edges() -> None:
    assert len(CONDITION_GRID) == 9
    for source in CONDITION_GRID:
        for target in adjacent_conditions(source):
            assert is_legal_transition(source, target)
            assert transition_axis(source, target) in {"intensity", "frequency"}
            assert source in adjacent_conditions(target)
    assert not is_legal_transition("C1", "C3")
    assert not is_legal_transition("C1", "C5")
    assert not is_legal_transition("C1", "C9")


def test_controller_is_deterministic_and_all_changes_are_legal() -> None:
    first = AdaptiveController(_config())
    second = AdaptiveController(_config())
    first_rows = _observe_valid(first, 30)
    second_rows = _observe_valid(second, 30)
    assert first_rows == second_rows
    changed = [row for row in first_rows if row["action"] != "hold"]
    assert len(changed) >= 4
    assert len({row["controller_condition"] for row in changed}) >= 3
    for row in changed:
        assert is_legal_transition(row["controller_condition_before"], row["controller_condition"])
        assert transition_axis(row["controller_condition_before"], row["controller_condition"]) in {
            "intensity",
            "frequency",
        }
        assert row["physical_action_applied"] is False
        assert row["action_outcome_observable"] is False
        assert row["previous_action_outcome"] == "not_observable"


def test_low_relaxation_probe_never_increases_intensity() -> None:
    controller = AdaptiveController(_config())
    rows = _observe_valid(controller, 3, relaxation=0.2, discomfort=0.1)
    changed = [row for row in rows if row["action"] != "hold"]
    assert len(changed) == 1
    row = changed[0]
    before_intensity = coordinates(row["controller_condition_before"])[0]
    after_intensity = coordinates(row["controller_condition"])[0]
    assert after_intensity <= before_intensity
    assert row["action"] != "intensity_increase"
    assert row["action_reason"] == "low_relaxation_reversible_non_intensity_probe"


def test_c9_fails_closed_when_visual_fit_direction_is_unavailable() -> None:
    controller = AdaptiveController(_config())
    safe_high_frequency = {
        "condition": "C6",
        "relaxation": 0.7,
        "discomfort": 0.0,
        "pleasantness": 0.6,
        "calm": 0.6,
        "monotony": 0.1,
        "visual_fit_appropriate": True,
        "visual_fit_direction": None,
    }
    safe_high_intensity = dict(safe_high_frequency, condition="C8")
    controller.observe(
        pred_relaxation=0.5,
        pred_discomfort=0.1,
        signal_valid=True,
        revealed_rating=safe_high_frequency,
    )
    controller.observe(
        pred_relaxation=0.5,
        pred_discomfort=0.1,
        signal_valid=True,
        revealed_rating=safe_high_intensity,
    )
    controller.observe(pred_relaxation=0.5, pred_discomfort=0.1, signal_valid=True)
    locked, reasons = controller.c9_status(0.1)
    assert controller.high_frequency_safe_evidence == {"C6"}
    assert controller.high_intensity_safe_evidence == {"C8"}
    assert locked
    assert "visual_fit_direction_unavailable_fail_closed" in reasons


def test_three_invalid_windows_return_one_safe_step_without_learning_from_invalid() -> None:
    controller = AdaptiveController(_config())
    valid = _observe_valid(controller, 6)
    assert valid[-1]["controller_condition"] == "C2"
    dwell_before = controller.dwell_windows
    invalid = [
        controller.observe(pred_relaxation=1.0, pred_discomfort=1.0, signal_valid=False)
        for _ in range(3)
    ]
    assert invalid[0]["action"] == "hold"
    assert invalid[1]["action"] == "hold"
    assert invalid[2]["controller_condition"] == "C1"
    assert invalid[2]["action_reason"] == "three_invalid_windows_safe_return"
    assert controller.invalid_fallback_actions == 1
    assert controller.dwell_windows == 0
    assert dwell_before == 0


def test_gap_forces_three_window_warmup() -> None:
    controller = AdaptiveController(_config())
    _observe_valid(controller, 5)
    first = controller.observe(
        pred_relaxation=0.5,
        pred_discomfort=0.1,
        signal_valid=True,
        gap_reset=True,
    )
    second = controller.observe(pred_relaxation=0.5, pred_discomfort=0.1, signal_valid=True)
    assert first["observation_reset_reason"] == "wall_clock_gap"
    assert first["valid_since_reset"] == 1 and first["action"] == "hold"
    assert first["dwell_windows"] == 1
    assert second["valid_since_reset"] == 2 and second["action"] == "hold"


class _PrefixLengthModel(torch.nn.Module):
    def forward(self, embeddings, masks):
        # Encode only values made available by FrozenHealNetEnsemble.predict_prefix.
        value = embeddings["eeg"].mean(dim=(1, 2))
        return torch.stack((torch.sigmoid(value), torch.sigmoid(-value)), dim=1)


def test_prefix_inference_does_not_read_future_embeddings() -> None:
    ensemble = FrozenHealNetEnsemble.__new__(FrozenHealNetEnsemble)
    ensemble.device = torch.device("cpu")
    ensemble.modalities = EXPECTED_MODALITIES
    ensemble.models = [_PrefixLengthModel() for _ in range(3)]
    ensemble.records = [{"seed": seed} for seed in (1, 2, 3)]
    dimensions = {"eeg": 4, "ecg": 3, "eye": 2, "head": 2, "video": 3}
    ensemble.scalers = [
        {
            modality: (torch.zeros(dimension), torch.ones(dimension))
            for modality, dimension in dimensions.items()
        }
        for _ in range(3)
    ]
    embeddings = {
        modality: torch.arange(5 * dimension, dtype=torch.float32).reshape(5, dimension)
        for modality, dimension in dimensions.items()
    }
    masks = {modality: torch.ones(5, dtype=torch.bool) for modality in EXPECTED_MODALITIES}
    original = ensemble.predict_prefix(embeddings, masks, 2)
    mutated = {modality: values.clone() for modality, values in embeddings.items()}
    for values in mutated.values():
        values[2:] = 1_000_000
    future_changed = ensemble.predict_prefix(mutated, masks, 2)
    assert original == future_changed
    prefix_grew = ensemble.predict_prefix(mutated, masks, 3)
    assert prefix_grew["pred_relaxation"] != pytest.approx(original["pred_relaxation"])


def _shadow_input() -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    conditions = [
        *(["C4"] * 7),
        *(["C6"] * 7),
        *(["C9"] * 7),
        *(["C8"] * 7),
        *(["C5"] * 7),
        *(["C3"] * 7),
        *(["C1"] * 7),
        "C7",
    ]
    starts = []
    timestamp = 1_000_000
    previous = None
    for condition in conditions:
        if previous is not None and condition != previous:
            timestamp += 60_000
        starts.append(timestamp)
        timestamp += 10_000
        previous = condition
    frame = pd.DataFrame(
        {
            "participant_id": "P009",
            "window_id": [f"W{index:02d}" for index in range(50)],
            "window_start_unix_ms": starts,
            "window_end_unix_ms": np.asarray(starts) + 10_000,
            "actual_condition": conditions,
            "actual_intensity_index": [coordinates(value)[0] for value in conditions],
            "actual_frequency_index": [coordinates(value)[1] for value in conditions],
            "pred_relaxation": np.linspace(0.45, 0.55, 50),
            "pred_discomfort": np.linspace(0.05, 0.08, 50),
            "true_relaxation": np.linspace(0.0, 1.0, 50),
            "true_discomfort": np.linspace(1.0, 0.0, 50),
            "signal_valid": True,
        }
    )
    ratings = {
        condition: {
            "condition": condition,
            "relaxation": 0.5,
            "discomfort": 0.0,
            "pleasantness": 0.5,
            "calm": 0.5,
            "monotony": 0.0,
            "visual_fit_appropriate": True,
            "visual_fit_direction": None,
        }
        for condition in CONDITION_GRID
    }
    return frame, ratings


def test_shadow_replay_ignores_current_and_future_truth_and_reveals_ratings_after_transition() -> None:
    frame, ratings = _shadow_input()
    first_trace, first_audit = run_chronological_replay(frame, ratings, AdaptiveController(_config()))
    poisoned = frame.copy()
    poisoned["true_relaxation"] = poisoned["true_relaxation"].iloc[::-1].to_numpy()
    poisoned["true_discomfort"] = 99.0
    second_trace, _ = run_chronological_replay(poisoned, ratings, AdaptiveController(_config()))
    decision_columns = [
        "controller_condition_before",
        "controller_condition",
        "action",
        "action_reason",
        "controller_state",
        "rating_revealed_condition",
    ]
    pd.testing.assert_frame_equal(first_trace[decision_columns], second_trace[decision_columns])
    assert first_audit["ratings_revealed_in_order"] == ["C4", "C6", "C9", "C8", "C5", "C3", "C1"]
    assert first_audit["ratings_not_revealed_within_segment"] == ["C2", "C7"]
    assert first_audit["current_condition_rating_never_passed"]
    assert not first_trace["physical_action_applied"].any()
    assert not first_trace["action_outcome_observable"].any()
    assert not first_trace["synthetic_replay"].any()
