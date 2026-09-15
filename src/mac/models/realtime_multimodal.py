from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.modeling.groups import TARGETS
from real_time_ml.utils import write_json


MODEL_KIND = "realtime_multimodal_window_v1"
ADAPTER_ID = "realtime_multimodal_window_v1"
SUPERVISION = "weak_window_supervision_v1"
ALLOWED_FEATURE_PREFIXES = ("eeg_", "ecg_", "eye_", "head_", "imu_")
BLOCKED_CONTEXT_COLUMNS = {
    "condition",
    "condition_index",
    "intensity",
    "intensity_index",
    "frequency",
    "frequency_index",
    "presentation_position",
}


def _dependencies():
    try:
        import joblib
        import pandas as pd
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as error:
        raise RuntimeError("Realtime multimodal model dependencies are missing; create the environment from environment.yml") from error
    return {
        "joblib": joblib,
        "pd": pd,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "VarianceThreshold": VarianceThreshold,
        "SimpleImputer": SimpleImputer,
        "Pipeline": Pipeline,
    }

def realtime_multimodal_feature_columns(columns: list[str]) -> list[str]:
    selected = [
        name
        for name in columns
        if name not in BLOCKED_CONTEXT_COLUMNS and name.startswith(ALLOWED_FEATURE_PREFIXES)
    ]
    return sorted(set(selected))


def validate_realtime_feature_columns(columns: list[str]) -> None:
    blocked = sorted(set(columns) & BLOCKED_CONTEXT_COLUMNS)
    if blocked:
        raise ValueError("Realtime multimodal model cannot use context columns: " + ",".join(blocked))
    invalid = sorted(name for name in columns if not name.startswith(ALLOWED_FEATURE_PREFIXES))
    if invalid:
        raise ValueError("Realtime multimodal model contains non-realtime feature columns: " + ",".join(invalid[:12]))


def feature_modalities(columns: list[str]) -> list[str]:
    output = []
    for prefix, label in (("eeg_", "eeg"), ("ecg_", "ecg"), ("eye_", "eye"), ("head_", "head"), ("imu_", "imu")):
        if any(name.startswith(prefix) for name in columns):
            output.append(label)
    return output


def fit_realtime_multimodal_window_model(
    frame: Any,
    target_names: list[str],
    *,
    seed: int,
    n_estimators: int = 500,
    min_samples_leaf: int = 4,
    max_features: float = 0.7,
) -> dict[str, Any]:
    deps = _dependencies()
    pd = deps["pd"]
    Pipeline = deps["Pipeline"]
    SimpleImputer = deps["SimpleImputer"]
    VarianceThreshold = deps["VarianceThreshold"]
    ExtraTreesRegressor = deps["ExtraTreesRegressor"]

    feature_columns = realtime_multimodal_feature_columns(list(frame.columns))
    validate_realtime_feature_columns(feature_columns)
    if not feature_columns:
        raise RuntimeError("No realtime multimodal features found; expected eeg_/ecg_/eye_/head_/imu_ columns")

    required = list(target_names)
    training = frame.dropna(subset=required).copy()
    if training.empty:
        raise RuntimeError("No rows with relaxation/discomfort targets are available for realtime multimodal training")

    X = training[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = training[target_names].to_numpy(dtype=float)
    weights = (
        pd.to_numeric(training.get("sample_weight"), errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if "sample_weight" in training
        else None
    )
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("variance", VarianceThreshold()),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=int(n_estimators),
                    min_samples_leaf=int(min_samples_leaf),
                    max_features=float(max_features),
                    random_state=int(seed),
                    n_jobs=-1,
                ),
            ),
        ]
    )
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    estimator.fit(X, y, **fit_kwargs)
    prediction = np.clip(estimator.predict(X), 0.0, 1.0)
    mae = np.mean(np.abs(y - prediction), axis=0)
    bundle = {
        "schema_version": "1.0.0",
        "model_kind": MODEL_KIND,
        "adapter_id": ADAPTER_ID,
        "supervision": SUPERVISION,
        "estimator": estimator,
        "feature_columns": feature_columns,
        "targets": target_names,
        "allowed_feature_prefixes": list(ALLOWED_FEATURE_PREFIXES),
        "blocked_context_columns": sorted(BLOCKED_CONTEXT_COLUMNS),
        "modalities": feature_modalities(feature_columns),
        "deployable": True,
        "metrics": {
            "unit_of_analysis": "10_second_window",
            "supervision": SUPERVISION,
            "n_windows": int(len(training)),
            "n_participants": int(training["participant_id"].astype(str).nunique()) if "participant_id" in training else 0,
            "feature_count": int(len(feature_columns)),
            "modalities": feature_modalities(feature_columns),
            "training_mae": dict(zip(target_names, map(float, mae))),
            "disallowed_context_columns_used": [],
        },
    }
    return bundle


def train_realtime_multimodal_window_model(config: ProjectConfig) -> dict[str, Any]:
    deps = _dependencies()
    pd = deps["pd"]
    joblib = deps["joblib"]
    source = config.path("features") / "window_features.csv"
    if not source.exists():
        raise FileNotFoundError("Run 'rtml extract-features' before training realtime multimodal window model")
    frame = pd.read_csv(source)
    target_names = list(config.get("modeling.targets", TARGETS))
    bundle = fit_realtime_multimodal_window_model(
        frame,
        target_names,
        seed=int(config.get("modeling.random_seed")),
    )
    model_path = config.path("models") / "realtime_multimodal_window_full.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    card = {key: value for key, value in bundle.items() if key != "estimator"}
    report_path = config.path("reports") / "realtime_multimodal_window_model_card.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, card)
    return {
        "model_path": str(model_path),
        "model_kind": MODEL_KIND,
        "supervision": SUPERVISION,
        "feature_count": len(bundle["feature_columns"]),
        "modalities": bundle["modalities"],
        "metrics": bundle["metrics"],
    }
