from __future__ import annotations

from real_time_ml.data.alignment import ConditionBoundary, make_windows
from real_time_ml.realtime.cycle import TenSecondCycleClock
from real_time_ml.schema import ConditionRecommendation, StatePrediction, validate_message


def test_windows_drop_tail_and_weights_sum_one():
    boundary = ConditionBoundary("P003", "C1", 100.0, 171.9, 1_000_000, 1_071_900, 1, 2)
    windows = make_windows([boundary])
    assert len(windows) == 7
    assert all(row["window_end_unix_ms"] - row["window_start_unix_ms"] == 10_000 for row in windows)
    assert abs(sum(row["sample_weight"] for row in windows) - 1.0) < 1e-12


def test_condition_change_resets_cycle_origin_and_out_of_order_is_rejected():
    clock = TenSecondCycleClock()
    assert clock.condition_event("C1", 12_345)
    assert clock.ready_windows(22_344) == []
    assert clock.ready_windows(22_345) == [(0, 12_345, 22_345)]
    assert not clock.condition_event("C1", 22_346)
    assert not clock.condition_event("C2", 20_000)
    assert clock.condition_event("C2", 30_000)
    assert clock.ready_windows(40_000) == [(0, 30_000, 40_000)]


def test_message_schema_is_exactly_ten_seconds_and_shadow_only():
    state = StatePrediction(
        schema_version="1.0", unix_time_ms=20_000, cycle_index=0,
        window_start_ms=10_000, window_end_ms=20_000, participant_id="P003", condition="C1",
        predictions={"relaxation": 0.5, "discomfort": 0.2},
        intervals={name: [0.2, 0.8] for name in ("relaxation", "discomfort")},
        modality_coverage={"head": 1.0}, qc={}, model_variant="behavior_only",
    )
    validate_message(state.to_dict())
    recommendation = ConditionRecommendation(
        schema_version="1.0", unix_time_ms=20_000, cycle_index=0,
        window_start_ms=10_000, window_end_ms=20_000, current_condition="C1",
        candidate_condition="C1", expected_relaxation_gain=None, conservative_gain=None,
        predicted_discomfort=None, safe=False, action="hold", reasons=["test"], shadow=True,
    )
    validate_message(recommendation.to_dict())
