from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from real_time_ml.config import ProjectConfig
from real_time_ml.modeling.groups import TARGETS, columns_for_group
from real_time_ml.modeling.safety import deployment_guard
from real_time_ml.utils import write_json


def _dependencies():
    try:
        import joblib
        import pandas as pd
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("Model training dependencies are missing; create the environment from environment.yml") from error
    return {
        "joblib": joblib,
        "pd": pd,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "VarianceThreshold": VarianceThreshold,
        "SimpleImputer": SimpleImputer,
        "ElasticNet": ElasticNet,
        "Ridge": Ridge,
        "GridSearchCV": GridSearchCV,
        "GroupKFold": GroupKFold,
        "LeaveOneGroupOut": LeaveOneGroupOut,
        "MultiOutputRegressor": MultiOutputRegressor,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def _candidate(name: str, seed: int, deps: dict[str, Any]):
    Pipeline = deps["Pipeline"]
    SimpleImputer = deps["SimpleImputer"]
    VarianceThreshold = deps["VarianceThreshold"]
    StandardScaler = deps["StandardScaler"]
    MultiOutputRegressor = deps["MultiOutputRegressor"]
    if name == "ridge":
        model = deps["Ridge"]()
        grid = {"model__alpha": [1.0, 10.0]}
    elif name == "elastic_net":
        model = MultiOutputRegressor(deps["ElasticNet"](max_iter=10_000, random_state=seed))
        grid = {"model__estimator__alpha": [0.01, 0.1], "model__estimator__l1_ratio": [0.5]}
    elif name == "extra_trees":
        model = deps["ExtraTreesRegressor"](n_estimators=80, min_samples_leaf=4, random_state=seed, n_jobs=-1)
        grid = {"model__max_features": [0.5, 1.0], "model__min_samples_leaf": [4]}
    elif name == "hist_gradient_boosting":
        model = MultiOutputRegressor(
            deps["HistGradientBoostingRegressor"](random_state=seed, max_iter=10, early_stopping=True)
        )
        grid = {"model__estimator__learning_rate": [0.08], "model__estimator__max_leaf_nodes": [7], "model__estimator__l2_regularization": [0.1]}
    else:
        raise ValueError(f"Unknown candidate {name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    ), grid


def _fit_tuned(estimator, grid, X, y, groups, weights, inner, GridSearchCV):
    combinations = int(np.prod([len(values) for values in grid.values()]))
    if combinations == 1:
        parameters = {name: values[0] for name, values in grid.items()}
        estimator.set_params(**parameters)
        estimator.fit(X, y, model__sample_weight=weights)
        return estimator, parameters
    search = GridSearchCV(
        estimator, grid, cv=inner, scoring="neg_mean_absolute_error", n_jobs=1, error_score="raise"
    )
    search.fit(X, y, groups=groups, model__sample_weight=weights)
    return search.best_estimator_, search.best_params_


def _weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(np.abs(y_true - y_pred), axis=0, weights=weights)


def _risk_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
    threshold: float = 0.5,
) -> float:
    actual = y_true >= threshold
    if not np.any(actual):
        return float("nan")
    detected = (y_pred[actual] >= threshold).astype(float)
    return float(np.average(detected, weights=weights[actual] if weights is not None else None))


def _selection_key(result: dict[str, Any]) -> tuple[Any, ...]:
    """Safety-first ranking, with experimental fallback when no model is deployable."""
    recall = float(result.get("discomfort_high_risk_recall", float("nan")))
    if not np.isfinite(recall):
        recall = -1.0
    return (
        not bool(result.get("deployable", False)),
        len(result.get("deployment_block_reasons", [])),
        -recall,
        float(result["mae"]["discomfort"]),
        float(result["mae"]["relaxation"]),
    )


def _baselines(train, test, target_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    global_mean = train[target_names].mean().to_numpy(dtype=float)
    condition_means = train.groupby("condition")[target_names].mean()
    condition_pred = np.vstack([
        condition_means.loc[condition].to_numpy(dtype=float) if condition in condition_means.index else global_mean
        for condition in test["condition"]
    ])
    history_pred = []
    for _, row in test.iterrows():
        prior = test[
            (test["participant_id"] == row["participant_id"])
            & (test["presentation_position"].astype(float) < float(row["presentation_position"]))
        ].sort_values("presentation_position")
        history_pred.append(prior.iloc[-1][target_names].to_numpy(dtype=float) if len(prior) else global_mean)
    return condition_pred, np.asarray(history_pred)


def train_state_window_legacy(config: ProjectConfig) -> dict[str, Any]:
    deps = _dependencies()
    pd = deps["pd"]
    joblib = deps["joblib"]
    GridSearchCV, GroupKFold, LeaveOneGroupOut = deps["GridSearchCV"], deps["GroupKFold"], deps["LeaveOneGroupOut"]
    source = config.path("features") / "window_features.csv"
    if not source.exists():
        raise FileNotFoundError("Run 'rtml extract-features' before training")
    frame = pd.read_csv(source)
    target_names = list(config.get("modeling.targets", TARGETS))
    frame = frame.dropna(subset=target_names + ["participant_id", "condition"])
    frame["presentation_position"] = pd.to_numeric(frame["presentation_position"], errors="coerce")
    frame["sample_weight"] = pd.to_numeric(frame["sample_weight"], errors="coerce").fillna(1.0)
    seed = int(config.get("modeling.random_seed"))
    groups = frame["participant_id"].astype(str).to_numpy()
    outer = LeaveOneGroupOut()
    all_results: list[dict[str, Any]] = []
    fold_predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for feature_group in config.get("modeling.feature_groups"):
        columns = columns_for_group(list(frame.columns), feature_group)
        if not columns:
            all_results.append({"feature_group": feature_group, "status": "skipped", "reason": "no_features"})
            continue
        X = frame[columns].apply(pd.to_numeric, errors="coerce")
        y = frame[target_names].to_numpy(dtype=float)
        for candidate_name in config.get("modeling.candidates"):
            print(f"LOPO: feature_group={feature_group} candidate={candidate_name}", flush=True)
            predictions = np.full_like(y, np.nan)
            lower = np.full_like(y, np.nan)
            upper = np.full_like(y, np.nan)
            fold_meta = []
            for train_index, test_index in outer.split(X, y, groups):
                train_groups = groups[train_index]
                unique_train = np.unique(train_groups)
                splits = min(int(config.get("modeling.inner_cv_splits")), len(unique_train))
                inner = GroupKFold(n_splits=max(2, splits))
                estimator, grid = _candidate(candidate_name, seed, deps)
                fitted, best_params = _fit_tuned(
                    estimator, grid,
                    X.iloc[train_index], y[train_index], groups=train_groups,
                    weights=frame.iloc[train_index]["sample_weight"].to_numpy(dtype=float),
                    inner=inner, GridSearchCV=GridSearchCV,
                )
                prediction = np.clip(fitted.predict(X.iloc[test_index]), 0.0, 1.0)
                predictions[test_index] = prediction
                train_prediction = np.clip(fitted.predict(X.iloc[train_index]), 0.0, 1.0)
                residual = np.abs(y[train_index] - train_prediction)
                quantile = np.quantile(residual, 1.0 - float(config.get("modeling.uncertainty_alpha")), axis=0)
                lower[test_index] = np.clip(prediction - quantile, 0.0, 1.0)
                upper[test_index] = np.clip(prediction + quantile, 0.0, 1.0)
                fold_meta.append({"test_participant": str(groups[test_index][0]), "best_params": best_params})
            weights = frame["sample_weight"].to_numpy(dtype=float)
            mae = _weighted_mae(y, predictions, weights)
            spearman = [float(spearmanr(y[:, i], predictions[:, i], nan_policy="omit").statistic) for i in range(len(target_names))]
            coverage = np.average((y >= lower) & (y <= upper), axis=0, weights=weights)
            condition_baseline, history_baseline = np.full_like(y, np.nan), np.full_like(y, np.nan)
            for train_index, test_index in outer.split(frame, y, groups):
                condition_baseline[test_index], history_baseline[test_index] = _baselines(frame.iloc[train_index], frame.iloc[test_index], target_names)
            condition_mae = _weighted_mae(y, condition_baseline, weights)
            history_mae = _weighted_mae(y, history_baseline, weights)
            result = {
                "feature_group": feature_group,
                "candidate": candidate_name,
                "status": "ok",
                "feature_count": len(columns),
                "mae": dict(zip(target_names, mae.tolist())),
                "spearman": dict(zip(target_names, spearman)),
                "interval_coverage": dict(zip(target_names, coverage.tolist())),
                "interval_mean_width": dict(zip(target_names, np.average(upper - lower, axis=0, weights=weights).tolist())),
                "discomfort_high_risk_recall": _risk_recall(
                    y[:, target_names.index("discomfort")],
                    predictions[:, target_names.index("discomfort")],
                    weights,
                ),
                "discomfort_high_risk_upper_recall": _risk_recall(
                    y[:, target_names.index("discomfort")],
                    upper[:, target_names.index("discomfort")],
                    weights,
                ),
                "condition_baseline_mae": dict(zip(target_names, condition_mae.tolist())),
                "history_baseline_mae": dict(zip(target_names, history_mae.tolist())),
                "folds": fold_meta,
            }
            result["deployable"], result["deployment_block_reasons"] = deployment_guard(result)
            all_results.append(result)
            fold_predictions[(feature_group, candidate_name)] = [
                {
                    "participant_id": frame.iloc[i]["participant_id"],
                    "condition": frame.iloc[i]["condition"],
                    "sample_weight": weights[i],
                    **{f"true_{name}": y[i, j] for j, name in enumerate(target_names)},
                    **{f"pred_{name}": predictions[i, j] for j, name in enumerate(target_names)},
                    **{f"lower_{name}": lower[i, j] for j, name in enumerate(target_names)},
                    **{f"upper_{name}": upper[i, j] for j, name in enumerate(target_names)},
                }
                for i in range(len(frame))
            ]
    valid = [row for row in all_results if row.get("status") == "ok"]
    if not valid:
        raise RuntimeError("No model candidate could be evaluated")
    best = min(valid, key=_selection_key)
    best["selection_rule"] = "deployable_then_fewest_safety_blocks_then_discomfort_recall_mae_then_relaxation_mae"
    columns = columns_for_group(list(frame.columns), best["feature_group"])
    estimator, grid = _candidate(best["candidate"], seed, deps)
    inner = GroupKFold(n_splits=min(int(config.get("modeling.inner_cv_splits")), len(np.unique(groups))))
    final_estimator, final_params = _fit_tuned(
        estimator, grid,
        frame[columns].apply(pd.to_numeric, errors="coerce"), frame[target_names].to_numpy(dtype=float),
        groups=groups, weights=frame["sample_weight"].to_numpy(dtype=float), inner=inner,
        GridSearchCV=GridSearchCV,
    )
    bundle = {
        "schema_version": config.data["schema_version"],
        "estimator": final_estimator,
        "feature_columns": columns,
        "targets": target_names,
        "feature_group": best["feature_group"],
        "candidate": best["candidate"],
        "deployable": best["deployable"],
        "best_params": final_params,
        "metrics": best,
        "interval_half_width": {name: min(0.5, max(0.05, 1.645 * best["mae"][name])) for name in target_names},
    }
    model_path = config.path("models") / "state_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    variant_groups = {"full": "fused", "no_eeg": "no_eeg", "behavior_only": "behavior_only"}
    for variant, group_name in variant_groups.items():
        variant_columns = columns_for_group(list(frame.columns), group_name)
        if not variant_columns:
            continue
        group_results = [row for row in valid if row["feature_group"] == group_name]
        variant_result = min(group_results, key=_selection_key) if group_results else best
        variant_candidate = variant_result["candidate"] if group_results else "ridge"
        variant_estimator, variant_grid = _candidate(variant_candidate, seed, deps)
        variant_fitted, variant_params = _fit_tuned(
            variant_estimator, variant_grid,
            frame[variant_columns].apply(pd.to_numeric, errors="coerce"), frame[target_names].to_numpy(dtype=float),
            groups=groups, weights=frame["sample_weight"].to_numpy(dtype=float), inner=inner,
            GridSearchCV=GridSearchCV,
        )
        variant_bundle = {
            "schema_version": config.data["schema_version"],
            "estimator": variant_fitted,
            "feature_columns": variant_columns,
            "targets": target_names,
            "feature_group": group_name,
            "candidate": variant_candidate,
            "deployable": bool(variant_result.get("deployable", False)) if group_results else False,
            "best_params": variant_params,
            "metrics": variant_result,
            "interval_half_width": {
                name: min(0.5, max(0.05, 1.645 * float(variant_result.get("mae", best["mae"])[name])))
                for name in target_names
            },
        }
        joblib.dump(variant_bundle, config.path("models") / f"state_model_{variant}.joblib")
    write_json(config.path("reports") / "lopo_state_metrics.json", {"selected": best, "all_results": all_results})
    pd.DataFrame(fold_predictions[(best["feature_group"], best["candidate"])]).to_csv(config.path("reports") / "lopo_state_predictions.csv", index=False)
    return {"model_path": str(model_path), "selected": best}


def load_state_model(path: Path):
    deps = _dependencies()
    return deps["joblib"].load(path)


def predict_state(bundle: dict[str, Any], features: dict[str, Any], condition: str | None = None) -> dict[str, float]:
    if bundle.get("model_kind") == "condition_residual_ensemble_v1":
        from real_time_ml.modeling.condition_train import predict_condition_bundle

        deps = _dependencies()
        return predict_condition_bundle(bundle, features, condition, deps["pd"])
    if bundle.get("model_kind") == "video_condition_ridge_v1":
        from real_time_ml.modeling.video_ridge import predict_video_ridge_bundle

        deps = _dependencies()
        return predict_video_ridge_bundle(bundle, features, condition, deps["pd"])
    deps = _dependencies()
    frame = deps["pd"].DataFrame(
        [{name: features.get(name, np.nan) for name in bundle["feature_columns"]}]
    ).apply(deps["pd"].to_numeric, errors="coerce")
    prediction = np.clip(bundle["estimator"].predict(frame)[0], 0.0, 1.0)
    return dict(zip(bundle["targets"], map(float, prediction)))


def train_state(config: ProjectConfig) -> dict[str, Any]:
    """Train only the 135 participant-Condition labels.

    The window-level function retained above is legacy audit code and is deliberately no
    longer callable from the CLI, because inherited 10-second labels are not independent.
    """
    from real_time_ml.modeling.condition_train import train_condition_state

    return train_condition_state(config)
