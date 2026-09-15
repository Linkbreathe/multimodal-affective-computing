from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from real_time_ml.cli import build_parser
from real_time_ml.config import ProjectConfig, load_config
from real_time_ml.modeling.minimal_fusion import (
    COMBINATIONS,
    FEATURES_PER_MODALITY,
    MODALITY_ORDER,
    benchmark_minimal_fusion,
    evaluate_minimal_fusion_frame,
    random_uniform_baseline,
)


def _frame() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for participant_number in range(2, 17):
        for condition_number in range(1, 10):
            presentation_position = ((condition_number + participant_number - 3) % 9) + 1
            relaxation = (condition_number + (participant_number % 4) + 2) / 15.0
            discomfort = ((10 - condition_number) + (participant_number % 3)) / 15.0
            row: dict[str, float | int | str] = {
                "participant_id": f"P{participant_number:03d}",
                "condition": f"C{condition_number}",
                "presentation_position": presentation_position,
                "relaxation": relaxation,
                "discomfort": discomfort,
                "relaxation_raw": relaxation * 10,
                "discomfort_raw": discomfort * 10,
                "condition_index": condition_number,
                "qc_eeg_usable": 1.0,
                "non_modal_leak": relaxation,
            }
            for index in range(23):
                row[f"eeg_feature_{index:02d}"] = relaxation + index * 0.001 + participant_number * 0.0001
            for prefix, value in (("ecg_", discomfort), ("head_", relaxation), ("eye_", discomfort), ("video_", relaxation)):
                for index in range(3):
                    row[f"{prefix}feature_{index:02d}"] = value + index * 0.01 + participant_number * 0.0001
            rows.append(row)
    return pd.DataFrame(rows)


def _config(tmp_path) -> ProjectConfig:
    data = deepcopy(load_config().data)
    data["paths"] = {key: str(tmp_path / key) for key in data["paths"]}
    config = ProjectConfig(source=tmp_path / "project.yaml", data=data)
    config.ensure_artifact_dirs()
    return config


def test_minimal_fusion_combinations_lopo_and_feature_caps():
    result = evaluate_minimal_fusion_frame(_frame(), random_seed=7, random_simulations=7)
    metrics = result["metrics"]
    combinations = metrics.loc[metrics["record_type"].eq("combination")]
    oof = result["oof_predictions"]

    assert COMBINATIONS == (
        "P", "H", "E", "V", "PH", "PE", "PV", "HE", "HV", "EV", "PHE", "PHV", "PEV", "HEV", "PHEV",
    )
    assert set(combinations["combination"]) == set(COMBINATIONS)
    assert not any("condition" in name.lower() for name in COMBINATIONS)
    assert len(oof) == len(COMBINATIONS) * 135 * 2
    assert oof.groupby(["combination", "target"]).size().eq(135).all()
    for modality in MODALITY_ORDER:
        for target in ("relaxation", "discomfort"):
            assert combinations[f"selected_{modality}_{target}_max"].le(FEATURES_PER_MODALITY).all()
    assert len(metrics.loc[metrics["record_type"].eq("ablation")]) == 28


def test_condition_only_is_holdout_label_independent_and_history_is_causal():
    source = _frame()
    original = evaluate_minimal_fusion_frame(source, random_seed=11, random_simulations=4)
    altered_source = source.copy()
    held_out = altered_source["participant_id"].eq("P002")
    altered_source.loc[held_out, "relaxation"] = 1.0 - altered_source.loc[held_out, "relaxation"]
    altered_source.loc[held_out, "discomfort"] = 1.0 - altered_source.loc[held_out, "discomfort"]
    altered = evaluate_minimal_fusion_frame(altered_source, random_seed=11, random_simulations=4)

    original_p002 = original["oof_predictions"].query("combination == 'P' and participant_id == 'P002'")
    altered_p002 = altered["oof_predictions"].query("combination == 'P' and participant_id == 'P002'")
    assert np.allclose(
        original_p002["condition_only_prediction"], altered_p002["condition_only_prediction"]
    )
    assert np.allclose(original_p002["prediction"], altered_p002["prediction"])

    history = original_p002.query("target == 'relaxation'").sort_values("presentation_position")
    fallback = source.loc[~source["participant_id"].eq("P002"), "relaxation"].mean()
    assert history.iloc[0]["history_prediction"] == fallback
    assert np.allclose(history.iloc[1:]["history_prediction"], history.iloc[:-1]["truth"])


def test_random_uniform_is_seed_reproducible():
    truth = {
        "relaxation": np.asarray([0.1, 0.3, 0.7, 0.9]),
        "discomfort": np.asarray([0.0, 0.5, 0.8, 0.2]),
    }
    assert random_uniform_baseline(truth, random_seed=20260621, simulations=31) == random_uniform_baseline(
        truth, random_seed=20260621, simulations=31
    )


def test_minimal_fusion_writes_only_the_three_requested_outputs(tmp_path):
    config = _config(tmp_path)
    source = config.path("features") / "video_ml" / "condition_features.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    _frame().to_csv(source, index=False)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}

    result = benchmark_minimal_fusion(config, random_simulations=5)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    created = after - before

    assert created == {
        (config.path("artifacts") / "fusion_minimal" / "metrics.csv").relative_to(tmp_path),
        (config.path("artifacts") / "fusion_minimal" / "oof_predictions.csv").relative_to(tmp_path),
        (config.path("reports") / "minimal_multimodal_fusion_zh.md").relative_to(tmp_path),
    }
    assert result["n_labels"] == 135
    assert result["n_combinations"] == 15
    assert result["n_ablations"] == 28
    assert build_parser().parse_args(["benchmark-minimal-fusion"]).command == "benchmark-minimal-fusion"
