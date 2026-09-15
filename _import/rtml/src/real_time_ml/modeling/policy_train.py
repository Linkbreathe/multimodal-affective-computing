from __future__ import annotations

from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.modeling.train import _dependencies
from real_time_ml.utils import write_json


POLICY_FEATURES = [
    "intensity", "frequency", "intensity_index", "frequency_index", "presentation_position",
    "previous_relaxation", "previous_discomfort",
    "participant_baseline_relaxation", "participant_baseline_discomfort",
]


def make_policy_table(frame, targets: list[str]):
    frame = frame.sort_values(["participant_id", "presentation_position"]).copy()
    global_means = frame[targets].mean()
    for target in targets:
        previous = frame.groupby("participant_id")[target].shift(1)
        causal_baseline = (
            frame.groupby("participant_id")[target]
            .transform(lambda values: values.shift(1).expanding().mean())
        )
        frame[f"participant_baseline_{target}"] = causal_baseline.fillna(global_means[target])
        frame[f"previous_{target}"] = previous.fillna(global_means[target])
    return frame


def train_policy(config: ProjectConfig) -> dict[str, Any]:
    deps = _dependencies()
    pd, joblib = deps["pd"], deps["joblib"]
    GroupKFold, GridSearchCV = deps["GroupKFold"], deps["GridSearchCV"]
    clone, cross_val_predict = deps["clone"], deps["cross_val_predict"]
    source = config.path("features") / "condition_features.csv"
    if not source.exists():
        raise FileNotFoundError("Run 'rtml extract-features' before policy training")
    frame = pd.read_csv(source)
    targets = list(config.get("modeling.targets"))
    frame = make_policy_table(frame.dropna(subset=targets), targets)
    groups = frame["participant_id"].astype(str).to_numpy()
    estimator, grid = __import__("real_time_ml.modeling.train", fromlist=["_candidate"])._candidate(
        "extra_trees", int(config.get("modeling.random_seed")), deps
    )
    splits = min(int(config.get("modeling.inner_cv_splits")), len(np.unique(groups)))
    cv = GroupKFold(n_splits=splits)
    search = GridSearchCV(estimator, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    search.fit(frame[POLICY_FEATURES], frame[targets].to_numpy(dtype=float), groups=groups)
    y_true = frame[targets].to_numpy(dtype=float)
    oof_prediction = np.clip(
        cross_val_predict(clone(search.best_estimator_), frame[POLICY_FEATURES], y_true, groups=groups, cv=cv, n_jobs=1),
        0.0,
        1.0,
    )
    absolute_error = np.abs(y_true - oof_prediction)
    interval_half_width = dict(zip(targets, np.quantile(absolute_error, 0.90, axis=0).tolist()))
    validation = {
        "group_cv_mae": dict(zip(targets, np.mean(absolute_error, axis=0).tolist())),
        "interval_half_width_90": interval_half_width,
    }
    bundle = {
        "schema_version": config.data["schema_version"],
        "estimator": search.best_estimator_,
        "feature_columns": POLICY_FEATURES,
        "targets": targets,
        "best_params": search.best_params_,
        "predecision_features_only": True,
        "interval_half_width": interval_half_width,
        "validation": validation,
    }
    path = config.path("models") / "policy_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    write_json(config.path("reports") / "policy_model_card.json", {k: v for k, v in bundle.items() if k != "estimator"})
    return {"model_path": str(path), "best_params": search.best_params_, "validation": validation}
