from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from real_time_ml.config import ProjectConfig
from real_time_ml.evaluation.alignment import (
    indexes_for_fold,
    load_split_manifest,
    validate_alignment_contract,
    write_alignment_manifest,
)
from real_time_ml.modeling.condition_data import STATIC_COLUMNS, build_condition_dataset
from real_time_ml.modeling.condition_models import (
    ModelSpec,
    apply_condition_baseline,
    condition_baseline,
    make_regression_pipeline,
    make_risk_pipeline,
)
from real_time_ml.modeling.groups import is_model_feature_allowed
from real_time_ml.utils import write_json


TARGETS = ("relaxation", "discomfort")
PROJECT_A_MODALITIES = ("eeg", "ecg", "eye", "head")
CONDITION_VARIANTS = ("full", "no_eeg", "no_ecg", "no_eye", "no_head", "behavior_only")


def _dependencies():
    try:
        import joblib
        import pandas as pd
        from sklearn.metrics import average_precision_score, precision_score, recall_score
        from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    except ImportError as error:
        raise RuntimeError("Condition-level training dependencies are missing; create environment.yml") from error
    return {
        "joblib": joblib,
        "pd": pd,
        "average_precision_score": average_precision_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "GroupKFold": GroupKFold,
        "LeaveOneGroupOut": LeaveOneGroupOut,
    }


def _feature_columns(frame) -> list[str]:
    columns = []
    for name in frame.columns:
        if name in STATIC_COLUMNS or name in {"window_count", "window_count_expected"}:
            continue
        if not is_model_feature_allowed(name):
            continue
        converted = __import__("pandas").to_numeric(frame[name], errors="coerce")
        if converted.notna().any():
            columns.append(name)
    # Condition parameters are legitimate, decision-time stimulus context.
    columns.extend(name for name in ("intensity", "frequency", "intensity_index", "frequency_index", "condition_index", "presentation_position") if name in frame.columns)
    return sorted(set(columns))


def _matrix(frame, columns, pd):
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def _ranking_accuracy(frame, prediction: np.ndarray, target: str) -> float:
    correct = 0
    total = 0
    working = frame[["participant_id", "presentation_position", target]].copy()
    working["prediction"] = prediction
    for _, group in working.groupby("participant_id"):
        true = group[target].to_numpy(dtype=float)
        pred = group["prediction"].to_numpy(dtype=float)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                true_difference = true[left] - true[right]
                if abs(true_difference) < 1e-12:
                    continue
                total += 1
                if true_difference * (pred[left] - pred[right]) > 0:
                    correct += 1
    return float(correct / total) if total else float("nan")


def _history_baseline(test_frame, target: str, fallback: float) -> np.ndarray:
    predictions = np.full(len(test_frame), fallback, dtype=float)
    for _, group in test_frame.groupby("participant_id"):
        ordered = group.sort_values("presentation_position")
        previous: float | None = None
        for index, row in ordered.iterrows():
            position = test_frame.index.get_loc(index)
            predictions[position] = fallback if previous is None else previous
            previous = float(row[target])
    return predictions


def _specs(config: ProjectConfig) -> list[ModelSpec]:
    return [
        ModelSpec(name, int(k))
        for name in config.get("modeling.candidates")
        for k in config.get("modeling.condition_level.feature_counts")
    ]


def _fit_target_models(train, columns, target: str, specs: list[ModelSpec], config: ProjectConfig, pd):
    baseline_map, baseline_fallback = condition_baseline(train, target)
    baseline = apply_condition_baseline(train["condition"], baseline_map, baseline_fallback)
    residual = train[target].to_numpy(dtype=float) - baseline
    X = _matrix(train, columns, pd)
    models = []
    for order, spec in enumerate(specs):
        pipeline = make_regression_pipeline(
            spec,
            int(config.get("modeling.random_seed")) + order,
            float(config.get("modeling.condition_level.min_non_missing_fraction")),
            float(config.get("modeling.condition_level.correlation_threshold")),
        )
        pipeline.fit(X, residual)
        models.append(pipeline)
    return baseline_map, baseline_fallback, models


def _predict_target_models(test, columns, baseline_map, fallback, models, pd) -> np.ndarray:
    baseline = apply_condition_baseline(test["condition"], baseline_map, fallback)
    residuals = np.vstack([model.predict(_matrix(test, columns, pd)) for model in models])
    return np.clip(baseline + np.mean(residuals, axis=0), 0.0, 1.0)


def _inner_rank_regression(train, columns, target: str, specs: list[ModelSpec], config: ProjectConfig, pd, GroupKFold):
    groups = train["participant_id"].astype(str).to_numpy()
    splits = min(int(config.get("modeling.inner_cv_splits")), len(np.unique(groups)))
    cv = GroupKFold(n_splits=max(2, splits))
    stats = {spec.label(): {"spec": spec, "mae": [], "rank": []} for spec in specs}
    for inner_train_index, inner_test_index in cv.split(train, groups=groups):
        inner_train = train.iloc[inner_train_index]
        inner_test = train.iloc[inner_test_index]
        baseline_map, fallback = condition_baseline(inner_train, target)
        residual = inner_train[target].to_numpy(dtype=float) - apply_condition_baseline(inner_train["condition"], baseline_map, fallback)
        X_train, X_test = _matrix(inner_train, columns, pd), _matrix(inner_test, columns, pd)
        base_test = apply_condition_baseline(inner_test["condition"], baseline_map, fallback)
        for order, spec in enumerate(specs):
            try:
                pipeline = make_regression_pipeline(
                    spec,
                    int(config.get("modeling.random_seed")) + order,
                    float(config.get("modeling.condition_level.min_non_missing_fraction")),
                    float(config.get("modeling.condition_level.correlation_threshold")),
                )
                pipeline.fit(X_train, residual)
                prediction = np.clip(base_test + pipeline.predict(X_test), 0.0, 1.0)
                stats[spec.label()]["mae"].append(float(np.mean(np.abs(inner_test[target].to_numpy(dtype=float) - prediction))))
                stats[spec.label()]["rank"].append(_ranking_accuracy(inner_test, prediction, target))
            except (ValueError, FloatingPointError):
                stats[spec.label()]["mae"].append(float("inf"))
                stats[spec.label()]["rank"].append(float("nan"))
    ranked = []
    for value in stats.values():
        mae = float(np.mean(value["mae"]))
        rank = float(np.nanmean(value["rank"])) if np.isfinite(value["rank"]).any() else 0.0
        # Ranking is intentionally a secondary tie-breaker for relaxation; the target is still residual MAE.
        score = mae - (0.02 * rank if target == "relaxation" else 0.0)
        ranked.append({"spec": value["spec"], "mae": mae, "ranking_accuracy": rank, "score": score})
    return sorted(ranked, key=lambda item: (item["score"], item["mae"]))


def _rank_regression_on_validation(
    train,
    validation,
    columns,
    target: str,
    specs: list[ModelSpec],
    config: ProjectConfig,
    pd,
):
    """Rank candidates on the manifest validation participant only."""

    baseline_map, fallback = condition_baseline(train, target)
    residual = train[target].to_numpy(dtype=float) - apply_condition_baseline(
        train["condition"], baseline_map, fallback
    )
    validation_baseline = apply_condition_baseline(validation["condition"], baseline_map, fallback)
    X_train = _matrix(train, columns, pd)
    X_validation = _matrix(validation, columns, pd)
    ranked = []
    for order, spec in enumerate(specs):
        try:
            model = make_regression_pipeline(
                spec,
                int(config.get("modeling.random_seed")) + order,
                float(config.get("modeling.condition_level.min_non_missing_fraction")),
                float(config.get("modeling.condition_level.correlation_threshold")),
            )
            model.fit(X_train, residual)
            prediction = np.clip(validation_baseline + model.predict(X_validation), 0.0, 1.0)
            mae = float(np.mean(np.abs(validation[target].to_numpy(dtype=float) - prediction)))
            rank = _ranking_accuracy(validation, prediction, target)
            rank_value = float(rank) if np.isfinite(rank) else 0.0
            score = mae - (0.02 * rank_value if target == "relaxation" else 0.0)
        except (ValueError, FloatingPointError):
            mae, rank_value, score = float("inf"), float("nan"), float("inf")
        ranked.append(
            {
                "spec": spec,
                "mae": mae,
                "ranking_accuracy": rank_value,
                "score": score,
            }
        )
    return sorted(ranked, key=lambda item: (item["score"], item["mae"], item["spec"].label()))


def _choose_feature_count_regression(train, columns, target: str, config: ProjectConfig, pd, GroupKFold):
    """Select K inside the outer training fold before comparing model families."""
    trials = _inner_rank_regression(
        train,
        columns,
        target,
        [ModelSpec("ridge", int(k)) for k in config.get("modeling.condition_level.feature_counts")],
        config,
        pd,
        GroupKFold,
    )
    return int(trials[0]["spec"].feature_count), trials


def _risk_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float, average_precision_score, precision_score, recall_score) -> dict[str, float | int]:
    predicted = probability >= threshold
    recall = float(recall_score(y_true, predicted, zero_division=0))
    precision = float(precision_score(y_true, predicted, zero_division=0))
    false_negatives = int(np.sum((y_true == 1) & ~predicted))
    return {
        "threshold": float(threshold),
        "recall": recall,
        "precision": precision,
        "false_negatives": false_negatives,
        "pr_auc": float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def _positive_probability(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    margin = np.asarray(model.decision_function(X), dtype=float)
    clipped = np.clip(margin, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _choose_risk_threshold(y_true, probability, thresholds, deps):
    rows = [_risk_metrics(y_true, probability, threshold, deps["average_precision_score"], deps["precision_score"], deps["recall_score"]) for threshold in thresholds]
    selected = min(rows, key=lambda row: (row["false_negatives"], -row["recall"], -row["precision"], row["threshold"]))
    return selected, rows


def _inner_rank_risk(train, columns, specs, config, pd, GroupKFold, deps):
    label_threshold = float(config.get("modeling.condition_level.high_discomfort_label_threshold"))
    thresholds = list(config.get("modeling.condition_level.risk_probability_thresholds"))
    groups = train["participant_id"].astype(str).to_numpy()
    y = (train["discomfort"].to_numpy(dtype=float) >= label_threshold).astype(int)
    splits = min(int(config.get("modeling.inner_cv_splits")), len(np.unique(groups)))
    cv = GroupKFold(n_splits=max(2, splits))
    probabilities = {spec.label(): np.full(len(train), np.nan) for spec in specs}
    for train_index, test_index in cv.split(train, y, groups):
        if len(np.unique(y[train_index])) < 2:
            continue
        X_train, X_test = _matrix(train.iloc[train_index], columns, pd), _matrix(train.iloc[test_index], columns, pd)
        for order, spec in enumerate(specs):
            try:
                model = make_risk_pipeline(
                    spec,
                    int(config.get("modeling.random_seed")) + order,
                    float(config.get("modeling.condition_level.min_non_missing_fraction")),
                    float(config.get("modeling.condition_level.correlation_threshold")),
                )
                model.fit(X_train, y[train_index])
                probabilities[spec.label()][test_index] = _positive_probability(model, X_test)
            except (ValueError, FloatingPointError):
                continue
    ranked = []
    for spec in specs:
        probability = probabilities[spec.label()]
        valid = np.isfinite(probability)
        if valid.sum() == 0 or len(np.unique(y[valid])) < 2:
            continue
        selected, table = _choose_risk_threshold(y[valid], probability[valid], thresholds, deps)
        ranked.append({"spec": spec, "selected": selected, "threshold_table": table, "probability": probability})
    return sorted(ranked, key=lambda item: (item["selected"]["false_negatives"], -item["selected"]["recall"], -item["selected"]["precision"]))


def _rank_risk_on_validation(train, validation, columns, specs, config, pd, deps):
    """Rank high-discomfort models without using the held-out test participant."""

    label_threshold = float(config.get("modeling.condition_level.high_discomfort_label_threshold"))
    thresholds = list(config.get("modeling.condition_level.risk_probability_thresholds"))
    y_train = (train["discomfort"].to_numpy(dtype=float) >= label_threshold).astype(int)
    y_validation = (validation["discomfort"].to_numpy(dtype=float) >= label_threshold).astype(int)
    if len(np.unique(y_train)) < 2:
        return []
    X_train = _matrix(train, columns, pd)
    X_validation = _matrix(validation, columns, pd)
    ranked = []
    for order, spec in enumerate(specs):
        try:
            model = make_risk_pipeline(
                spec,
                int(config.get("modeling.random_seed")) + order,
                float(config.get("modeling.condition_level.min_non_missing_fraction")),
                float(config.get("modeling.condition_level.correlation_threshold")),
            )
            model.fit(X_train, y_train)
            probability = _positive_probability(model, X_validation)
            selected, table = _choose_risk_threshold(y_validation, probability, thresholds, deps)
            ranked.append(
                {
                    "spec": spec,
                    "selected": selected,
                    "threshold_table": table,
                    "probability": probability,
                }
            )
        except (ValueError, FloatingPointError):
            continue
    return sorted(
        ranked,
        key=lambda item: (
            item["selected"]["false_negatives"],
            -item["selected"]["recall"],
            -item["selected"]["precision"],
            item["spec"].label(),
        ),
    )


def _choose_feature_count_risk(train, columns, config, pd, GroupKFold, deps):
    trials = _inner_rank_risk(
        train,
        columns,
        [ModelSpec("logistic_regression", int(k)) for k in config.get("modeling.condition_level.feature_counts")],
        config,
        pd,
        GroupKFold,
        deps,
    )
    if not trials:
        return int(config.get("modeling.condition_level.feature_counts")[0]), []
    return int(trials[0]["spec"].feature_count), trials


def _fit_risk_models(train, columns, specs, config, pd):
    threshold = float(config.get("modeling.condition_level.high_discomfort_label_threshold"))
    y = (train["discomfort"].to_numpy(dtype=float) >= threshold).astype(int)
    if len(np.unique(y)) < 2:
        return [], float(np.mean(y))
    models = []
    X = _matrix(train, columns, pd)
    for order, spec in enumerate(specs):
        model = make_risk_pipeline(
            spec,
            int(config.get("modeling.random_seed")) + order,
            float(config.get("modeling.condition_level.min_non_missing_fraction")),
            float(config.get("modeling.condition_level.correlation_threshold")),
        )
        model.fit(X, y)
        models.append(model)
    return models, float(np.mean(y))


def _predict_risk_models(test, columns, models, fallback, pd):
    if not models:
        return np.full(len(test), fallback, dtype=float)
    probabilities = np.vstack([_positive_probability(model, _matrix(test, columns, pd)) for model in models])
    return np.mean(probabilities, axis=0)


def _variant_columns(columns: list[str], variant: str) -> list[str]:
    if variant == "full":
        return columns
    if variant in {"no_eeg", "no_ecg", "no_eye", "no_head"}:
        modality = variant.removeprefix("no_")

        def belongs_to_removed_modality(name: str) -> bool:
            if name.startswith((f"{modality}_", f"qc_{modality}_", f"mask_{modality}_")):
                return True
            # This flag jointly describes the EEG/ECG acquisition stream and
            # must not survive removal of either physiological modality.
            return modality in {"eeg", "ecg"} and name.startswith("qc_physio_complete")

        return [name for name in columns if not belongs_to_removed_modality(name)]
    if variant == "behavior_only":
        return [name for name in columns if name.startswith(("head_", "eye_"))]
    raise ValueError(variant)


def _variant_modalities(variant: str) -> tuple[str, ...]:
    if variant == "full":
        return PROJECT_A_MODALITIES
    if variant in {"no_eeg", "no_ecg", "no_eye", "no_head"}:
        removed = variant.removeprefix("no_")
        return tuple(modality for modality in PROJECT_A_MODALITIES if modality != removed)
    if variant == "behavior_only":
        return ("eye", "head")
    raise ValueError(variant)


def _fit_final_bundle(frame, columns, selected_specs, selected_risk_specs, risk_threshold, config, pd):
    targets: dict[str, Any] = {}
    for target in TARGETS:
        baseline_map, fallback, models = _fit_target_models(frame, columns, target, selected_specs[target], config, pd)
        targets[target] = {
            "baseline_by_condition": baseline_map,
            "baseline_fallback": fallback,
            "models": models,
            "specs": [asdict(spec) for spec in selected_specs[target]],
        }
    risk_models, risk_fallback = _fit_risk_models(frame, columns, selected_risk_specs, config, pd)
    return {
        "model_kind": "condition_residual_ensemble_v1",
        "schema_version": config.data["schema_version"],
        "feature_columns": columns,
        "targets": list(TARGETS),
        "target_models": targets,
        "risk_models": risk_models,
        "risk_specs": [asdict(spec) for spec in selected_risk_specs],
        "risk_fallback_probability": risk_fallback,
        "risk_probability_threshold": risk_threshold,
        "high_discomfort_label_threshold": float(config.get("modeling.condition_level.high_discomfort_label_threshold")),
        "interval_half_width": {},
        "deployable": False,  # only changed after metric safety gate
    }


def _personalized_calibration(frame, predictions: dict[str, np.ndarray], config: ProjectConfig):
    output = {}
    for count in config.get("modeling.condition_level.personalized_calibration_conditions"):
        calibrated = {target: np.full(len(frame), np.nan) for target in TARGETS}
        evaluated = []
        for _, participant in frame.groupby("participant_id"):
            ordered = participant.sort_values("presentation_position")
            calibration = ordered.iloc[: int(count)]
            test = ordered.iloc[int(count) :]
            if test.empty:
                continue
            for target in TARGETS:
                indices_calibration = calibration.index.to_numpy()
                indices_test = test.index.to_numpy()
                bias = float(np.mean(frame.loc[indices_calibration, target].to_numpy(dtype=float) - predictions[target][indices_calibration]))
                calibrated[target][indices_test] = np.clip(predictions[target][indices_test] + bias, 0.0, 1.0)
            evaluated.extend(test.index.tolist())
        indices = np.asarray(evaluated, dtype=int)
        output[str(count)] = {
            "n_predictions": int(len(indices)),
            "relaxation_mae": float(np.mean(np.abs(frame.loc[indices, "relaxation"].to_numpy(dtype=float) - calibrated["relaxation"][indices]))) if len(indices) else float("nan"),
            "relaxation_spearman": float(spearmanr(frame.loc[indices, "relaxation"], calibrated["relaxation"][indices]).statistic) if len(indices) else float("nan"),
            "relaxation_ranking_accuracy": _ranking_accuracy(frame.loc[indices], calibrated["relaxation"][indices], "relaxation") if len(indices) else float("nan"),
            "discomfort_mae": float(np.mean(np.abs(frame.loc[indices, "discomfort"].to_numpy(dtype=float) - calibrated["discomfort"][indices]))) if len(indices) else float("nan"),
        }
    return output


def train_condition_state(
    config: ProjectConfig,
    *,
    source: Path | None = None,
    condition_output: Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    expected_labels: int = 135,
) -> dict[str, Any]:
    """Train one isolated Condition-level model namespace.

    Default arguments preserve the selected production/no-video artifacts.  The
    visual pipelines pass private directories so their experiments can never
    replace ``state_model.joblib`` or the default reports.
    """
    deps = _dependencies()
    pd, joblib = deps["pd"], deps["joblib"]
    aligned = bool(config.get("alignment.enabled", False))
    contract_payload: dict[str, Any] | None = None
    expected_train_count = 13
    if aligned:
        contract_dir = Path(str(config.get("alignment.contract_dir")))
        contract_payload = validate_alignment_contract(contract_dir)
        expected_labels = int(
            contract_payload.get("label_count", contract_payload.get("observation_count", expected_labels))
        )
        expected_train_count = int(
            contract_payload.get("fold_contract", {}).get("train_participants", 13)
        )
        source = source or Path(str(config.get("alignment.window_features")))
    else:
        source = source or config.path("features") / "window_features.csv"
    if not source.exists():
        raise FileNotFoundError("Run 'rtml extract-features' before condition-level training")
    output_path = condition_output or config.path("features") / "condition_features.csv"
    models_dir = models_dir or config.path("models")
    reports_dir = reports_dir or config.path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame = build_condition_dataset(source, output_path)
    frame = frame.dropna(subset=["participant_id", "condition", *TARGETS]).copy()
    frame["presentation_position"] = pd.to_numeric(frame["presentation_position"], errors="coerce")
    if len(frame) != expected_labels or frame[["participant_id", "condition"]].duplicated().any():
        raise ValueError(f"Expected exactly {expected_labels} unique participant/Condition labels; found {len(frame)}")
    frame = frame.reset_index(drop=True)
    all_columns = _feature_columns(frame)
    if aligned:
        # The aligned Project A definition is EEG+ECG+eye+head. Video QC and
        # mask fields are provenance, not Project A predictor variables.
        all_columns = [
            name
            for name in all_columns
            if not name.startswith(("video_", "qc_video_", "mask_video_"))
        ]
    model_variant = str(config.get("modeling.condition_variant", "full"))
    if model_variant not in CONDITION_VARIANTS:
        raise ValueError(f"Unknown classical Condition variant: {model_variant!r}")
    columns = _variant_columns(all_columns, model_variant)
    if not columns:
        raise ValueError(f"No usable features remain for classical variant {model_variant!r}")
    modalities = _variant_modalities(model_variant)
    split_protocol = (
        f"shared_{expected_train_count}_train_1_validation_1_test"
        if aligned
        else "native_14_non_test_nested_lopo"
    )
    model_names = list(config.get("modeling.candidates"))
    groups = frame["participant_id"].astype(str).to_numpy()
    predictions = {target: np.full(len(frame), np.nan) for target in TARGETS}
    condition_baseline_predictions = {target: np.full(len(frame), np.nan) for target in TARGETS}
    history_baseline_predictions = {target: np.full(len(frame), np.nan) for target in TARGETS}
    risk_probability = np.full(len(frame), np.nan)
    risk_thresholds = np.full(len(frame), np.nan)
    fold_records = []
    selected_counts = {target: Counter() for target in TARGETS}
    risk_selected_counts: Counter[str] = Counter()
    row_fold = np.full(len(frame), -1, dtype=int)
    row_validation = np.full(len(frame), "", dtype=object)
    if aligned:
        split_manifest = Path(str(config.get("alignment.split_manifest")))
        folds = load_split_manifest(
            split_manifest,
            expected_participants=sorted(set(groups)),
            expected_train_count=expected_train_count,
        )
        split_iterator = [
            (fold, *indexes_for_fold(groups, fold))
            for fold in folds
        ]
    else:
        outer = deps["LeaveOneGroupOut"]()
        split_iterator = []
        folds = []
        for fold_index, (train_index, test_index) in enumerate(
            outer.split(frame, groups=groups), start=1
        ):
            test_participant = str(groups[test_index][0])
            split_iterator.append((None, train_index, np.asarray([], dtype=int), test_index))

    for fold_position, (aligned_fold, train_index, validation_index, test_index) in enumerate(
        split_iterator, start=1
    ):
        fold_index = aligned_fold.fold_index if aligned_fold is not None else fold_position
        train, test = frame.iloc[train_index], frame.iloc[test_index]
        validation = frame.iloc[validation_index] if len(validation_index) else None
        test_participant = str(test.iloc[0]["participant_id"])
        validation_participant = (
            str(validation.iloc[0]["participant_id"]) if validation is not None else None
        )
        row_fold[test_index] = fold_index
        row_validation[test_index] = validation_participant or ""
        fold_record: dict[str, Any] = {
            "fold": fold_index,
            "test_participant": test_participant,
            "validation_participant": validation_participant,
            "train_participants": sorted(train["participant_id"].astype(str).unique()),
            "n_train_participants": int(train["participant_id"].nunique()),
            "n_validation_participants": int(validation["participant_id"].nunique()) if validation is not None else 0,
            "n_test_participants": int(test["participant_id"].nunique()),
            "targets": {},
        }
        for target in TARGETS:
            if aligned:
                feature_count_trials = _rank_regression_on_validation(
                    train,
                    validation,
                    columns,
                    target,
                    [
                        ModelSpec("ridge", int(k))
                        for k in config.get("modeling.condition_level.feature_counts")
                    ],
                    config,
                    pd,
                )
                selected_k = int(feature_count_trials[0]["spec"].feature_count)
                ranked = _rank_regression_on_validation(
                    train,
                    validation,
                    columns,
                    target,
                    [ModelSpec(name, selected_k) for name in model_names],
                    config,
                    pd,
                )
            else:
                selected_k, feature_count_trials = _choose_feature_count_regression(
                    train, columns, target, config, pd, deps["GroupKFold"]
                )
                ranked = _inner_rank_regression(
                    train,
                    columns,
                    target,
                    [ModelSpec(name, selected_k) for name in model_names],
                    config,
                    pd,
                    deps["GroupKFold"],
                )
            top = [item["spec"] for item in ranked[: int(config.get("modeling.condition_level.ensemble_size"))]]
            baseline_map, fallback, models = _fit_target_models(train, columns, target, top, config, pd)
            prediction = _predict_target_models(test, columns, baseline_map, fallback, models, pd)
            predictions[target][test_index] = prediction
            condition_baseline_predictions[target][test_index] = apply_condition_baseline(test["condition"], baseline_map, fallback)
            history_baseline_predictions[target][test_index] = _history_baseline(test.reset_index(drop=True), target, fallback)
            selected_counts[target].update(spec.label() for spec in top)
            fold_record["targets"][target] = {
                "selected": [asdict(spec) for spec in top],
                "inner_top": [{key: value for key, value in item.items() if key != "spec"} | {"spec": asdict(item["spec"])} for item in ranked[:3]],
                "feature_count_trials": [{key: value for key, value in item.items() if key != "spec"} | {"spec": asdict(item["spec"])} for item in feature_count_trials],
            }
        if aligned:
            risk_feature_count_trials = _rank_risk_on_validation(
                train,
                validation,
                columns,
                [
                    ModelSpec("logistic_regression", int(k))
                    for k in config.get("modeling.condition_level.feature_counts")
                ],
                config,
                pd,
                deps,
            )
            risk_k = int(
                risk_feature_count_trials[0]["spec"].feature_count
                if risk_feature_count_trials
                else config.get("modeling.condition_level.feature_counts")[0]
            )
        else:
            risk_k, risk_feature_count_trials = _choose_feature_count_risk(
                train, columns, config, pd, deps["GroupKFold"], deps
            )
        risk_specs = [ModelSpec("logistic_regression", risk_k), ModelSpec("svm_classifier", risk_k)]
        risk_ranked = (
            _rank_risk_on_validation(train, validation, columns, risk_specs, config, pd, deps)
            if aligned
            else _inner_rank_risk(train, columns, risk_specs, config, pd, deps["GroupKFold"], deps)
        )
        top_risk = [item["spec"] for item in risk_ranked[: int(config.get("modeling.condition_level.risk_ensemble_size"))]]
        if top_risk:
            inner_probability = np.nanmean(np.vstack([next(item["probability"] for item in risk_ranked if item["spec"] == spec) for spec in top_risk]), axis=0)
            valid = np.isfinite(inner_probability)
            selected_threshold, _ = _choose_risk_threshold(
                (
                    validation["discomfort"].to_numpy(dtype=float)[valid]
                    >= float(config.get("modeling.condition_level.high_discomfort_label_threshold"))
                ).astype(int),
                inner_probability[valid],
                config.get("modeling.condition_level.risk_probability_thresholds"), deps,
            )
            threshold = float(selected_threshold["threshold"])
        else:
            threshold = float(min(config.get("modeling.condition_level.risk_probability_thresholds")))
        risk_models, fallback_probability = _fit_risk_models(train, columns, top_risk, config, pd)
        risk_probability[test_index] = _predict_risk_models(test, columns, risk_models, fallback_probability, pd)
        risk_thresholds[test_index] = threshold
        risk_selected_counts.update(spec.label() for spec in top_risk)
        fold_record["risk"] = {
            "selected": [asdict(spec) for spec in top_risk],
            "threshold": threshold,
            "feature_count_trials": [
                {key: value for key, value in item.items() if key not in {"spec", "probability"}} | {"spec": asdict(item["spec"])}
                for item in risk_feature_count_trials
            ],
        }
        fold_records.append(fold_record)
        print(
            f"Condition {model_variant} LOPO fold {fold_index}/{len(split_iterator)}: "
            f"{fold_record['test_participant']}",
            flush=True,
        )

    metrics: dict[str, Any] = {
        "unit_of_analysis": "participant_condition",
        "n_labels": int(len(frame)),
        "seed": int(config.get("modeling.random_seed")),
        "model_variant": model_variant,
        "modalities": list(modalities),
        "split_protocol": split_protocol,
        "runtime": {"device": "cpu", "cuda_used": False},
        "targets": {},
    }
    for target in TARGETS:
        truth = frame[target].to_numpy(dtype=float)
        metrics["targets"][target] = {
            "mae": float(np.mean(np.abs(truth - predictions[target]))),
            "spearman": float(spearmanr(truth, predictions[target]).statistic),
            "ranking_accuracy": _ranking_accuracy(frame, predictions[target], target),
            "condition_only_baseline_mae": float(np.mean(np.abs(truth - condition_baseline_predictions[target]))),
            "history_baseline_mae": float(np.mean(np.abs(truth - history_baseline_predictions[target]))),
        }
    high_risk = (frame["discomfort"].to_numpy(dtype=float) >= float(config.get("modeling.condition_level.high_discomfort_label_threshold"))).astype(int)
    discomfort_metrics = metrics["targets"]["discomfort"]
    discomfort_metrics["risk_at_fold_tuned_threshold"] = _risk_metrics(
        high_risk, risk_probability, float(np.nanmedian(risk_thresholds)), deps["average_precision_score"], deps["precision_score"], deps["recall_score"]
    )
    discomfort_metrics["risk_at_fold_tuned_threshold"]["per_row_threshold_recall"] = float(np.mean((risk_probability[high_risk == 1] >= risk_thresholds[high_risk == 1]))) if np.any(high_risk) else float("nan")
    discomfort_metrics["threshold_sweep"] = [
        _risk_metrics(high_risk, risk_probability, threshold, deps["average_precision_score"], deps["precision_score"], deps["recall_score"])
        for threshold in config.get("modeling.condition_level.risk_probability_thresholds")
    ]
    discomfort_metrics["condition_only_high_risk_recall"] = float(np.mean(condition_baseline_predictions["discomfort"][high_risk == 1] >= 0.5)) if np.any(high_risk) else float("nan")
    discomfort_metrics["history_high_risk_recall"] = float(np.mean(history_baseline_predictions["discomfort"][high_risk == 1] >= 0.5)) if np.any(high_risk) else float("nan")
    metrics["personalized_calibration"] = _personalized_calibration(frame, predictions, config)
    relaxation_ok = (
        metrics["targets"]["relaxation"]["mae"] < metrics["targets"]["relaxation"]["condition_only_baseline_mae"]
        and metrics["targets"]["relaxation"]["mae"] < metrics["targets"]["relaxation"]["history_baseline_mae"]
        and metrics["targets"]["relaxation"]["spearman"] > 0
    )
    discomfort_ok = (
        metrics["targets"]["discomfort"]["mae"] < metrics["targets"]["discomfort"]["condition_only_baseline_mae"]
        and metrics["targets"]["discomfort"]["mae"] < metrics["targets"]["discomfort"]["history_baseline_mae"]
        and discomfort_metrics["risk_at_fold_tuned_threshold"]["per_row_threshold_recall"] >= 0.5
    )
    metrics["deployable"] = bool(relaxation_ok and discomfort_ok)
    metrics["deployment_block_reasons"] = [
        *([] if relaxation_ok else ["condition_level_relaxation_gate_failed"]),
        *([] if discomfort_ok else ["condition_level_discomfort_gate_failed"]),
    ]
    selected_specs = {}
    for target in TARGETS:
        ordered_labels = [label for label, _ in selected_counts[target].most_common(int(config.get("modeling.condition_level.ensemble_size")))]
        lookup = {spec.label(): spec for k in config.get("modeling.condition_level.feature_counts") for spec in [ModelSpec(name, int(k)) for name in model_names]}
        selected_specs[target] = [lookup[label] for label in ordered_labels] or [
            ModelSpec(model_names[0], int(config.get("modeling.condition_level.feature_counts")[0]))
        ]
    risk_lookup = {
        spec.label(): spec
        for k in config.get("modeling.condition_level.feature_counts")
        for spec in (ModelSpec("logistic_regression", int(k)), ModelSpec("svm_classifier", int(k)))
    }
    selected_risk_specs = [risk_lookup[label] for label, _ in risk_selected_counts.most_common(int(config.get("modeling.condition_level.risk_ensemble_size")))] or [risk_specs[0]]
    final_threshold = float(np.nanmedian(risk_thresholds))
    bundles = {}
    bundle_variants = (model_variant,) if aligned else ("full", "no_eeg", "behavior_only")
    for variant in bundle_variants:
        variant_columns = columns if aligned else _variant_columns(all_columns, variant)
        if variant_columns:
            bundle = _fit_final_bundle(frame, variant_columns, selected_specs, selected_risk_specs, final_threshold, config, pd)
            bundle["model_variant"] = variant
            bundle["metrics"] = metrics
            bundle["deployable"] = bool(metrics["deployable"])
            bundle["interval_half_width"] = {
                target: min(0.5, max(0.05, 1.645 * metrics["targets"][target]["mae"]))
                for target in TARGETS
            }
            bundles[variant] = bundle
            joblib.dump(bundle, models_dir / f"state_model_{variant}.joblib")
    joblib.dump(bundles[model_variant if aligned else "full"], models_dir / "state_model.joblib")
    predictions_output = frame[["participant_id", "condition", "presentation_position", *TARGETS]].copy()
    for target in TARGETS:
        predictions_output[f"pred_{target}"] = predictions[target]
        predictions_output[f"condition_only_{target}"] = condition_baseline_predictions[target]
        predictions_output[f"history_{target}"] = history_baseline_predictions[target]
    predictions_output["risk_probability"] = risk_probability
    predictions_output["risk_threshold"] = risk_thresholds
    predictions_output["high_discomfort"] = high_risk
    predictions_output["seed"] = int(config.get("modeling.random_seed"))
    predictions_output["fold_index"] = row_fold
    predictions_output["test_participant"] = predictions_output["participant_id"].astype(str)
    predictions_output["validation_participant"] = row_validation
    predictions_output["model_variant"] = model_variant
    predictions_output["modalities"] = "+".join(modalities)
    predictions_output["split_protocol"] = split_protocol
    predictions_output["device"] = "cpu"
    predictions_output["cuda_used"] = False
    prediction_dir = config.path("predictions") if not config.is_legacy and reports_dir == config.path("reports") else reports_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)
    predictions_output.to_csv(prediction_dir / "condition_level_lopo_predictions.csv", index=False)
    report = {
        "schema_version": config.data["schema_version"],
        "metrics": metrics,
        "model_variant": model_variant,
        "modalities": list(modalities),
        "feature_count_before_fold_selection": len(columns),
        "feature_count_before_ablation": len(all_columns),
        "candidate_models": list(config.get("modeling.candidates")),
        "feature_counts_tried": list(config.get("modeling.condition_level.feature_counts")),
        "selected_spec_frequency": {target: dict(counter) for target, counter in selected_counts.items()},
        "selected_risk_spec_frequency": dict(risk_selected_counts),
        "folds": fold_records,
        "seed": int(config.get("modeling.random_seed")),
        "split_protocol": split_protocol,
        "runtime": {"device": "cpu", "cuda_used": False},
    }
    if aligned:
        report["alignment"] = {
            "contract_dir": str(Path(str(config.get("alignment.contract_dir"))).resolve()),
            "split_manifest": str(Path(str(config.get("alignment.split_manifest"))).resolve()),
            "modality_masks": str(Path(str(config.get("alignment.modality_masks"))).resolve()),
            "labels": str(Path(str(config.get("alignment.labels"))).resolve()),
            "windows": str(Path(str(config.get("alignment.windows"))).resolve()),
        }
        write_alignment_manifest(
            config.path("manifests") / "classical_alignment_manifest.json",
            split_manifest=config.get("alignment.split_manifest"),
            modality_masks=config.get("alignment.modality_masks"),
            labels=config.get("alignment.labels"),
            windows=config.get("alignment.windows"),
            window_features=source,
            contract=Path(str(config.get("alignment.contract_dir"))) / "contract.json",
            seed=int(config.get("modeling.random_seed")),
            folds=folds,
            runtime={
                "device": "cpu",
                "cuda_used": False,
                "model": "classical_residual_ensemble",
                "model_variant": model_variant,
                "modalities": list(modalities),
            },
        )
    write_json(reports_dir / "condition_level_lopo_metrics.json", report)
    return {
        "model_path": str(models_dir / "state_model.joblib"),
        "metrics": metrics,
        "model_variant": model_variant,
        "n_condition_labels": int(len(frame)),
    }


def predict_condition_bundle(bundle: dict[str, Any], features: dict[str, Any], condition: str | None, pd):
    output: dict[str, float] = {}
    condition_value = str(condition or "")
    X = pd.DataFrame([{column: features.get(column, np.nan) for column in bundle["feature_columns"]}])
    for target, target_bundle in bundle["target_models"].items():
        baseline = float(target_bundle["baseline_by_condition"].get(condition_value, target_bundle["baseline_fallback"]))
        residual = float(np.mean([model.predict(X)[0] for model in target_bundle["models"]]))
        output[target] = float(np.clip(baseline + residual, 0.0, 1.0))
    return output


def predict_condition_risk(bundle: dict[str, Any], features: dict[str, Any], pd) -> float:
    if not bundle.get("risk_models"):
        return float(bundle.get("risk_fallback_probability", 0.0))
    X = pd.DataFrame([{column: features.get(column, np.nan) for column in bundle["feature_columns"]}])
    return float(np.mean([_positive_probability(model, X)[0] for model in bundle["risk_models"]]))
