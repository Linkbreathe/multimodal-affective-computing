from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_rq2_modality_ablation import (
    ALL_MODEL_MODALITIES,
    AblationAlignmentError,
    EXPECTED_SEEDS,
    MODALITY_ORDER,
    REMOVED_MODALITY_MODEL,
    TARGETS,
    _determinism_check,
    _resolve_artifact,
    build_ablation_group_summary,
    build_participant_deltas,
    build_plot_data,
)


def _participant_metrics() -> pd.DataFrame:
    rows = []
    participants = [f"P{index:03d}" for index in range(1, 10)]
    for model_index, model_id in enumerate(ALL_MODEL_MODALITIES):
        for target_index, target in enumerate(TARGETS):
            for participant_index, participant in enumerate(participants):
                for seed in EXPECTED_SEEDS:
                    full = 0.1 + target_index * 0.01 + participant_index * 0.001
                    if model_id == "frozen_simplex_full5":
                        mae = full
                    else:
                        removed = next(
                            modality
                            for modality, candidate in REMOVED_MODALITY_MODEL.items()
                            if candidate == model_id
                        )
                        sign = 1 if MODALITY_ORDER.index(removed) % 2 == 0 else -1
                        mae = full + sign * (model_index + 1) * 0.001
                    rows.append(
                        {
                            "model_id": model_id,
                            "target": target,
                            "participant": participant,
                            "seed": seed,
                            "model_mae": mae,
                        }
                    )
    return pd.DataFrame(rows)


def test_participant_deltas_are_seed_averaged_and_paired() -> None:
    deltas = build_participant_deltas(_participant_metrics())
    assert len(deltas) == 90
    assert set(deltas["seed_count"]) == {3}
    assert set(deltas.groupby(["removed_modality", "target"]).size()) == {9}
    eeg = deltas[deltas["removed_modality"].eq("EEG")]
    ecg = deltas[deltas["removed_modality"].eq("ECG")]
    assert (eeg["delta_mae"] > 0).all()
    assert (ecg["delta_mae"] < 0).all()


def test_group_summary_is_deterministic_and_reports_wtl() -> None:
    deltas = build_participant_deltas(_participant_metrics())
    first = build_ablation_group_summary(deltas, bootstrap_replicates=500)
    second = build_ablation_group_summary(deltas, bootstrap_replicates=500)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert len(first) == 10
    assert set(first["participant_count"]) == {9}
    eeg = first[first["removed_modality"].eq("EEG")]
    ecg = first[first["removed_modality"].eq("ECG")]
    assert (eeg[["wins", "ties", "losses"]].to_numpy() == [9, 0, 0]).all()
    assert (ecg[["wins", "ties", "losses"]].to_numpy() == [0, 0, 9]).all()


def test_plot_data_contains_every_participant_and_group_mean() -> None:
    deltas = build_participant_deltas(_participant_metrics())
    summary = build_ablation_group_summary(deltas, bootstrap_replicates=100)
    plot_data = build_plot_data(deltas, summary)
    assert len(plot_data) == 100
    assert (plot_data["row_type"] == "participant").sum() == 90
    assert (plot_data["row_type"] == "group_mean").sum() == 10
    assert plot_data.loc[plot_data["row_type"].eq("group_mean"), "ci_lower"].notna().all()


def test_determinism_check_requires_exact_predictions() -> None:
    frame = pd.DataFrame(
        {
            "model_id": ["frozen_simplex_no_eeg"],
            "seed": [20260705],
            "target": ["relaxation"],
            "participant": ["P003"],
            "condition": ["C1"],
            "condition_anchor": [0.2],
            "true_residual": [0.1],
            "predicted_residual": [0.05],
            "prediction_before_clipping": [0.25],
            "final_prediction": [0.25],
        }
    )
    result = _determinism_check(frame, frame.copy())
    assert result["exact_oof_match"] is True
    changed = frame.copy()
    changed.loc[0, "final_prediction"] = np.nextafter(0.25, 1.0)
    with pytest.raises(AblationAlignmentError, match="not exactly deterministic"):
        _determinism_check(frame, changed)


def test_artifacts_are_resolved_relative_to_shared_root(tmp_path: Path) -> None:
    assert _resolve_artifact(tmp_path, "wsl_results/cache.npz") == tmp_path / "wsl_results/cache.npz"
    absolute = tmp_path / "absolute.npz"
    assert _resolve_artifact(tmp_path / "ignored", str(absolute)) == absolute


def test_participant_pairing_fails_closed_when_a_seed_is_missing() -> None:
    metrics = _participant_metrics()
    missing = metrics[
        ~(
            metrics["model_id"].eq("frozen_simplex_no_eeg")
            & metrics["participant"].eq("P001")
            & metrics["target"].eq("relaxation")
            & metrics["seed"].eq(EXPECTED_SEEDS[0])
        )
    ]
    with pytest.raises(AblationAlignmentError, match="pairing incomplete"):
        build_participant_deltas(missing)
