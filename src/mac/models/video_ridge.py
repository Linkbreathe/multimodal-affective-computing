"""Fast, leakage-safe Condition residual Ridge for visual ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.modeling.condition_data import STATIC_COLUMNS, build_condition_dataset
from real_time_ml.modeling.safety import deployment_guard
from real_time_ml.utils import write_json


VIDEO_RIDGE_KIND = "video_condition_ridge_v1"
RELAXATION_VIDEO_RIDGE_KIND = "video_condition_ridge_relaxation_only_v1"
TARGETS = ("relaxation", "discomfort")
CONTEXT = ("intensity", "frequency", "intensity_index", "frequency_index", "condition_index", "presentation_position")
BASE_PREFIXES = ("eeg_", "ecg_", "head_", "eye_")


def _candidate_columns(frame, include_video: bool) -> list[str]:
    import pandas as pd

    candidates = []
    for name in frame.columns:
        if name in STATIC_COLUMNS or name in {"window_count", "window_count_expected"}:
            continue
        if not name.startswith((*BASE_PREFIXES, "video_")):
            continue
        if name.startswith("video_") and not include_video:
            continue
        if not name.endswith(("__mean", "__std", "__slope", "__missing_ratio")):
            continue
        if pd.to_numeric(frame[name], errors="coerce").notna().any():
            candidates.append(name)
    return sorted(set(candidates))


def _fold_columns(train, candidates: list[str], include_video: bool, limit: int = 120) -> list[str]:
    import pandas as pd

    base = [name for name in candidates if not name.startswith("video_")]
    visual = [name for name in candidates if name.startswith("video_")] if include_video else []
    def rank(names, count):
        values = train[names].apply(pd.to_numeric, errors="coerce")
        valid = values.notna().mean(axis=0) >= 0.4
        variance = values.loc[:, valid].var(axis=0, skipna=True).sort_values(ascending=False)
        return variance.head(count).index.tolist()
    selected = rank(base, limit)
    if include_video:
        selected.extend(rank(visual, limit))
    selected.extend(name for name in CONTEXT if name in train.columns)
    return sorted(set(selected))


def _baseline(train, test, target: str) -> tuple[np.ndarray, dict[str, float], float]:
    fallback = float(train[target].mean())
    mapping = train.groupby("condition")[target].mean().astype(float).to_dict()
    prediction = np.asarray([float(mapping.get(condition, fallback)) for condition in test["condition"]], dtype=float)
    return prediction, mapping, fallback


def _history_baseline(test, fallback: float, target: str) -> np.ndarray:
    output = np.full(len(test), fallback, dtype=float)
    for _, group in test.groupby("participant_id"):
        prior = None
        for index in group.sort_values("presentation_position").index:
            position = test.index.get_loc(index)
            if prior is not None:
                output[position] = prior
            prior = float(test.loc[index, target])
    return output


def _pipeline(alpha: float = 10.0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def train_visual_ridge(
    config: ProjectConfig,
    *,
    source: Path,
    condition_output: Path,
    models_dir: Path,
    reports_dir: Path,
    include_video: bool,
    expected_labels: int,
    targets: tuple[str, ...] = TARGETS,
    model_kind: str = VIDEO_RIDGE_KIND,
    research_only: bool = False,
) -> dict[str, Any]:
    """LOPO residual Ridge with fold-local unsupervised feature filtering."""
    import joblib
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.model_selection import LeaveOneGroupOut

    if not targets:
        raise ValueError("Visual Ridge requires at least one target")
    if research_only and tuple(targets) != ("relaxation",):
        raise ValueError("Research-only visual Ridge bundles must predict relaxation only")
    frame = build_condition_dataset(source, condition_output).dropna(subset=["participant_id", "condition", *targets]).copy()
    frame["presentation_position"] = pd.to_numeric(frame["presentation_position"], errors="coerce")
    if len(frame) != expected_labels or frame[["participant_id", "condition"]].duplicated().any():
        raise ValueError(f"Expected {expected_labels} unique labels; found {len(frame)}")
    frame = frame.reset_index(drop=True)
    candidates = _candidate_columns(frame, include_video)
    if not candidates:
        raise ValueError("No usable visual-Ridge candidate columns")
    groups = frame["participant_id"].astype(str).to_numpy()
    outer = LeaveOneGroupOut()
    prediction = {target: np.full(len(frame), np.nan) for target in targets}
    condition_only = {target: np.full(len(frame), np.nan) for target in targets}
    history = {target: np.full(len(frame), np.nan) for target in targets}
    fold_records: list[dict[str, Any]] = []
    for fold, (train_indexes, test_indexes) in enumerate(outer.split(frame, groups=groups), start=1):
        train, test = frame.iloc[train_indexes], frame.iloc[test_indexes]
        columns = _fold_columns(train, candidates, include_video)
        fold_record = {"fold": fold, "test_participant": str(test.iloc[0]["participant_id"]), "feature_count": len(columns)}
        for target in targets:
            baseline, _, fallback = _baseline(train, test, target)
            train_baseline, _, _ = _baseline(train, train, target)
            model = _pipeline()
            model.fit(train[columns].apply(pd.to_numeric, errors="coerce"), train[target].to_numpy(dtype=float) - train_baseline)
            residual = model.predict(test[columns].apply(pd.to_numeric, errors="coerce"))
            prediction[target][test_indexes] = np.clip(baseline + residual, 0.0, 1.0)
            condition_only[target][test_indexes] = baseline
            history[target][test_indexes] = _history_baseline(test.reset_index(drop=True), fallback, target)
        fold_records.append(fold_record)
    metrics: dict[str, Any] = {"unit_of_analysis": "participant_condition", "n_labels": int(len(frame)), "targets": {}}
    for target in targets:
        truth = frame[target].to_numpy(dtype=float)
        value = float(spearmanr(truth, prediction[target]).statistic)
        metrics["targets"][target] = {
            "mae": float(np.mean(np.abs(truth - prediction[target]))),
            "spearman": value if np.isfinite(value) else 0.0,
            "condition_only_baseline_mae": float(np.mean(np.abs(truth - condition_only[target]))),
            "history_baseline_mae": float(np.mean(np.abs(truth - history[target]))),
        }
    if tuple(targets) == TARGETS and not research_only:
        threshold = float(min(config.get("modeling.condition_level.risk_probability_thresholds")))
        high = frame["discomfort"].to_numpy(dtype=float) >= float(config.get("modeling.condition_level.high_discomfort_label_threshold"))
        detected = prediction["discomfort"] >= threshold
        risk = {
            "threshold": threshold,
            "recall": float(np.mean(detected[high])) if np.any(high) else float("nan"),
            "per_row_threshold_recall": float(np.mean(detected[high])) if np.any(high) else float("nan"),
            "precision": float(np.mean(high[detected])) if np.any(detected) else 0.0,
            "false_negatives": int(np.sum(high & ~detected)),
        }
        metrics["targets"]["discomfort"]["risk_at_fold_tuned_threshold"] = risk
        deployable, reasons = deployment_guard(metrics)
        metrics["deployable"] = deployable
        metrics["deployment_block_reasons"] = reasons
    else:
        threshold = None
        metrics["deployable"] = False
        metrics["deployment_block_reasons"] = ["research_only_relaxation_model"]
    final_columns = _fold_columns(frame, candidates, include_video)
    target_models: dict[str, Any] = {}
    for target in targets:
        baseline, mapping, fallback = _baseline(frame, frame, target)
        model = _pipeline()
        model.fit(frame[final_columns].apply(pd.to_numeric, errors="coerce"), frame[target].to_numpy(dtype=float) - baseline)
        target_models[target] = {"model": model, "baseline_by_condition": mapping, "baseline_fallback": fallback}
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_kind": model_kind, "schema_version": config.data["schema_version"], "model_variant": "full" if include_video else "no_video",
        "feature_columns": final_columns, "targets": list(targets), "target_models": target_models,
        "interval_half_width": {target: float(min(0.5, max(0.05, 1.645 * metrics["targets"][target]["mae"]))) for target in targets},
        "metrics": metrics, "deployable": bool(metrics["deployable"]), "risk_probability_threshold": threshold,
    }
    if research_only:
        bundle["research_only"] = True
    joblib.dump(bundle, models_dir / "state_model_full.joblib")
    output = frame[["participant_id", "condition", "presentation_position", *targets]].copy()
    for target in targets:
        output[f"pred_{target}"] = prediction[target]
        output[f"condition_only_{target}"] = condition_only[target]
        output[f"history_{target}"] = history[target]
    output.to_csv(reports_dir / "condition_level_lopo_predictions.csv", index=False)
    report = {"metrics": metrics, "folds": fold_records, "feature_profile": {"base_top_variance": 120, "video_top_variance": 120 if include_video else 0, "statistics": ["mean", "std", "slope", "missing_ratio"], "residual_model": "Ridge(alpha=10)"}}
    write_json(reports_dir / "condition_level_lopo_metrics.json", report)
    return {"model_path": str(models_dir / "state_model_full.joblib"), "metrics": metrics, "n_condition_labels": int(len(frame)), "feature_profile": report["feature_profile"]}


def predict_video_ridge_bundle(bundle: dict[str, Any], features: dict[str, Any], condition: str | None, pd) -> dict[str, float]:
    condition_value = str(condition or "")
    values = pd.DataFrame([{column: features.get(column, np.nan) for column in bundle["feature_columns"]}]).apply(pd.to_numeric, errors="coerce")
    output = {}
    for target, target_bundle in bundle["target_models"].items():
        baseline = float(target_bundle["baseline_by_condition"].get(condition_value, target_bundle["baseline_fallback"]))
        output[target] = float(np.clip(baseline + float(target_bundle["model"].predict(values)[0]), 0.0, 1.0))
    return output
