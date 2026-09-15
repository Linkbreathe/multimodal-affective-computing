from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.supplementary.run_four_channel_ranking_benchmark import (  # noqa: E402
    A4_PREFIX,
    BINARY_TARGET,
    CONTINUOUS_TARGETS,
    DEFAULT_BASE_FEATURES,
    DEFAULT_OUTPUT_DIR as FOUR_CHANNEL_OUTPUT_DIR,
    DISCOMFORT_THRESHOLD,
    IDENTIFIER_AND_LABEL_COLUMNS,
    MODALITY_ORDER,
    MODALITY_PREFIXES,
    _continuous_metrics,
    _matrix,
)
from real_time_ml.data.io import normalize_participant_id  # noqa: E402
from real_time_ml.modeling import minimal_fusion as old_fusion  # noqa: E402


DEFAULT_A4_FEATURES = FOUR_CHANNEL_OUTPUT_DIR / "a4_condition_features.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "fusion_latent_multitask"
DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "latent_state_multitask_benchmark_zh.md"

LABEL_TARGETS = ("relaxation", "pleasantness", "calm", "discomfort")
COMBINATIONS = tuple(
    "".join(parts)
    for size in range(1, len(MODALITY_ORDER) + 1)
    for parts in combinations(MODALITY_ORDER, size)
)


def _dependencies():
    try:
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Latent benchmark dependencies are missing; use rtml-p002-p016") from error
    return (
        PCA,
        SimpleImputer,
        Ridge,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        Pipeline,
        StandardScaler,
    )


def _fmt(value: Any, digits: int = 4, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.1%}" if percent else f"{number:.{digits}f}"


def _validate_frame(frame: pd.DataFrame, expected_labels: int = 135) -> pd.DataFrame:
    required = {"participant_id", "condition", "presentation_position", *LABEL_TARGETS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input frame missing required columns: {missing}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].map(normalize_participant_id)
    frame["condition"] = frame["condition"].astype(str)
    for name in ("presentation_position", *LABEL_TARGETS):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame[list(required)].isna().any().any():
        raise ValueError("Input has missing identifiers or target labels")
    if frame[["participant_id", "condition"]].duplicated().any():
        raise ValueError("Input must contain one row per participant-condition")
    if len(frame) != expected_labels:
        raise ValueError(f"Expected {expected_labels} rows; found {len(frame)}")
    frame[BINARY_TARGET] = (frame["discomfort"] >= DISCOMFORT_THRESHOLD).astype(int)
    return frame.reset_index(drop=True)


def _modal_columns(frame: pd.DataFrame, modality: str) -> list[str]:
    prefixes = MODALITY_PREFIXES[modality]
    return sorted(
        name
        for name in frame.columns
        if name not in IDENTIFIER_AND_LABEL_COLUMNS and name.startswith(prefixes)
    )


def _matrix_from_labels(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, list(LABEL_TARGETS)].to_numpy(dtype=float)


def _condition_label_baseline(train: pd.DataFrame, target_frame: pd.DataFrame) -> np.ndarray:
    fallback = train.loc[:, list(LABEL_TARGETS)].mean(axis=0).to_numpy(dtype=float)
    by_condition = train.groupby("condition", sort=True)[list(LABEL_TARGETS)].mean()
    rows = []
    for condition in target_frame["condition"]:
        if condition in by_condition.index:
            rows.append(by_condition.loc[condition].to_numpy(dtype=float))
        else:
            rows.append(fallback)
    return np.vstack(rows)


def _history_label_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    fallback = train.loc[:, list(LABEL_TARGETS)].mean(axis=0).to_numpy(dtype=float)
    output = np.tile(fallback, (len(test), 1))
    for _, group in test.groupby("participant_id", sort=False):
        previous: np.ndarray | None = None
        for index in group.sort_values("presentation_position", kind="stable").index:
            position = test.index.get_loc(index)
            output[position] = fallback if previous is None else previous
            previous = test.loc[index, list(LABEL_TARGETS)].to_numpy(dtype=float)
    return output


def _feature_rank_by_latent(train: pd.DataFrame, columns: list[str], latent: np.ndarray, limit: int) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for name in columns:
        values = pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).sum() < 3 or np.nanstd(values) <= 0:
            continue
        best = 0.0
        for component in range(latent.shape[1]):
            signal = latent[:, component]
            valid = np.isfinite(values) & np.isfinite(signal)
            if valid.sum() < 3 or np.nanstd(signal[valid]) <= 0:
                continue
            corr = float(np.corrcoef(values[valid], signal[valid])[0, 1])
            if np.isfinite(corr):
                best = max(best, abs(corr))
        if best > 0:
            ranked.append((best, name))
    return [name for _, name in sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]]


def _select_by_modality(
    train: pd.DataFrame,
    latent: np.ndarray,
    *,
    feature_limit_per_modality: int,
) -> dict[str, list[str]]:
    return {
        modality: _feature_rank_by_latent(
            train,
            _modal_columns(train, modality),
            latent,
            feature_limit_per_modality,
        )
        for modality in MODALITY_ORDER
    }


def _combine_selected(selected_by_modality: dict[str, list[str]], combination: str) -> tuple[list[str], dict[str, int]]:
    selected: list[str] = []
    counts: dict[str, int] = {}
    for modality in MODALITY_ORDER:
        retained = selected_by_modality[modality] if modality in combination else []
        selected.extend(retained)
        counts[modality] = len(retained)
    return sorted(selected), counts


def _ridge_pipeline(alpha: float):
    _, SimpleImputer, Ridge, *_rest, Pipeline, StandardScaler = _dependencies()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _binary_metrics_from_predictions(
    truth: np.ndarray,
    score: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    _, _, _, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, *_ = _dependencies()
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    prediction = np.asarray(prediction, dtype=int)
    if len(np.unique(truth)) == 2:
        roc_auc = float(roc_auc_score(truth, score))
        pr_auc = float(average_precision_score(truth, score))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "false_negatives": float(np.sum((truth == 1) & (prediction == 0))),
        "positives_predicted": float(np.sum(prediction == 1)),
    }


def _choose_recall_threshold(
    truth: np.ndarray,
    score: np.ndarray,
    *,
    min_recall: float,
) -> float:
    truth = np.asarray(truth, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    truth = truth[finite]
    score = score[finite]
    if truth.size == 0 or truth.sum() == 0:
        return 0.5
    candidates = np.unique(np.r_[score, 0.0, 1.0])
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        pred = score >= threshold
        tp = float(np.sum((truth == 1) & pred))
        fp = float(np.sum((truth == 0) & pred))
        fn = float(np.sum((truth == 1) & ~pred))
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        if recall >= min_recall:
            candidate = (precision, threshold, recall)
            if best is None or candidate > best:
                best = candidate
    if best is not None:
        return float(best[1])
    # Fall back to the lowest threshold so a safety policy errs toward recall.
    return float(np.min(candidates))


def _continuous_rows_from_predictions(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    record_type: str,
    baseline: str,
    combination: str,
) -> list[dict[str, Any]]:
    rows = []
    for target in CONTINUOUS_TARGETS:
        rows.append(
            {
                "record_type": record_type,
                "baseline": baseline,
                "combination": combination,
                "target": target,
                **_continuous_metrics(frame, target, predictions[target]),
            }
        )
    return rows


def _array_predictions_by_target(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        target: values[:, LABEL_TARGETS.index(target)].astype(float)
        for target in CONTINUOUS_TARGETS
    }


def evaluate_latent_multitask(
    frame: pd.DataFrame,
    *,
    latent_components: int,
    ridge_alpha: float,
    feature_limit_per_modality: int,
    min_recall: float,
) -> dict[str, Any]:
    PCA, *_rest, StandardScaler = _dependencies()
    frame = _validate_frame(frame)
    n_components = min(latent_components, len(LABEL_TARGETS))
    participants = sorted(frame["participant_id"].unique())
    label_truth = _matrix_from_labels(frame)
    continuous_predictions = {
        combination: np.full((len(frame), len(LABEL_TARGETS)), np.nan, dtype=float)
        for combination in COMBINATIONS
    }
    condition_prediction = np.full_like(label_truth, np.nan, dtype=float)
    history_prediction = np.full_like(label_truth, np.nan, dtype=float)
    fixed_binary_predictions = {
        combination: np.full(len(frame), -1, dtype=int) for combination in COMBINATIONS
    }
    tuned_binary_predictions = {
        combination: np.full(len(frame), -1, dtype=int) for combination in COMBINATIONS
    }
    binary_scores = {combination: np.full(len(frame), np.nan, dtype=float) for combination in COMBINATIONS}
    condition_fixed_binary = np.full(len(frame), -1, dtype=int)
    condition_tuned_binary = np.full(len(frame), -1, dtype=int)
    history_fixed_binary = np.full(len(frame), -1, dtype=int)
    history_tuned_binary = np.full(len(frame), -1, dtype=int)
    condition_scores = np.full(len(frame), np.nan, dtype=float)
    history_scores = np.full(len(frame), np.nan, dtype=float)
    threshold_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []

    for participant in participants:
        test_mask = frame["participant_id"].eq(participant).to_numpy()
        test_indexes = np.flatnonzero(test_mask)
        train = frame.loc[~test_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)
        y_train = _matrix_from_labels(train)
        cond_train = _condition_label_baseline(train, train)
        cond_test = _condition_label_baseline(train, test)
        hist_train = _history_label_baseline(train, train)
        hist_test = _history_label_baseline(train, test)
        residual_train = y_train - cond_train
        label_scaler = StandardScaler()
        residual_scaled = label_scaler.fit_transform(residual_train)
        pca = PCA(n_components=n_components, random_state=20260704)
        latent_train = pca.fit_transform(residual_scaled)
        for component in range(n_components):
            for target_index, target in enumerate(LABEL_TARGETS):
                loading_rows.append(
                    {
                        "held_out_participant": participant,
                        "component": component + 1,
                        "target": target,
                        "loading": float(pca.components_[component, target_index]),
                        "explained_variance_ratio": float(pca.explained_variance_ratio_[component]),
                    }
                )
        selected_by_modality = _select_by_modality(
            train,
            latent_train,
            feature_limit_per_modality=feature_limit_per_modality,
        )

        condition_prediction[test_indexes] = cond_test
        history_prediction[test_indexes] = hist_test
        condition_scores[test_indexes] = np.clip(cond_test[:, LABEL_TARGETS.index("discomfort")], 0.0, 1.0)
        history_scores[test_indexes] = np.clip(hist_test[:, LABEL_TARGETS.index("discomfort")], 0.0, 1.0)

        train_truth_binary = train[BINARY_TARGET].to_numpy(dtype=int)
        cond_train_score = np.clip(cond_train[:, LABEL_TARGETS.index("discomfort")], 0.0, 1.0)
        hist_train_score = np.clip(hist_train[:, LABEL_TARGETS.index("discomfort")], 0.0, 1.0)
        cond_threshold = _choose_recall_threshold(train_truth_binary, cond_train_score, min_recall=min_recall)
        hist_threshold = _choose_recall_threshold(train_truth_binary, hist_train_score, min_recall=min_recall)
        condition_fixed_binary[test_indexes] = (condition_scores[test_indexes] >= DISCOMFORT_THRESHOLD).astype(int)
        history_fixed_binary[test_indexes] = (history_scores[test_indexes] >= DISCOMFORT_THRESHOLD).astype(int)
        condition_tuned_binary[test_indexes] = (condition_scores[test_indexes] >= cond_threshold).astype(int)
        history_tuned_binary[test_indexes] = (history_scores[test_indexes] >= hist_threshold).astype(int)
        threshold_rows.extend(
            [
                {
                    "held_out_participant": participant,
                    "record_type": "baseline",
                    "name": "condition_only",
                    "threshold": cond_threshold,
                },
                {
                    "held_out_participant": participant,
                    "record_type": "baseline",
                    "name": "history",
                    "threshold": hist_threshold,
                },
            ]
        )

        for combination in COMBINATIONS:
            columns, counts = _combine_selected(selected_by_modality, combination)
            if columns:
                model = _ridge_pipeline(ridge_alpha)
                model.fit(_matrix(train, columns), latent_train)
                latent_test = np.asarray(model.predict(_matrix(test, columns)), dtype=float)
                latent_train_pred = np.asarray(model.predict(_matrix(train, columns)), dtype=float)
            else:
                latent_test = np.zeros((len(test), n_components), dtype=float)
                latent_train_pred = np.zeros((len(train), n_components), dtype=float)
            if latent_test.ndim == 1:
                latent_test = latent_test.reshape(-1, 1)
            if latent_train_pred.ndim == 1:
                latent_train_pred = latent_train_pred.reshape(-1, 1)
            residual_test = label_scaler.inverse_transform(pca.inverse_transform(latent_test))
            residual_train_pred = label_scaler.inverse_transform(pca.inverse_transform(latent_train_pred))
            pred_test = np.clip(cond_test + residual_test, 0.0, 1.0)
            pred_train = np.clip(cond_train + residual_train_pred, 0.0, 1.0)
            continuous_predictions[combination][test_indexes] = pred_test
            score_test = pred_test[:, LABEL_TARGETS.index("discomfort")]
            score_train = pred_train[:, LABEL_TARGETS.index("discomfort")]
            threshold = _choose_recall_threshold(train_truth_binary, score_train, min_recall=min_recall)
            binary_scores[combination][test_indexes] = score_test
            fixed_binary_predictions[combination][test_indexes] = (score_test >= DISCOMFORT_THRESHOLD).astype(int)
            tuned_binary_predictions[combination][test_indexes] = (score_test >= threshold).astype(int)
            row: dict[str, Any] = {
                "held_out_participant": participant,
                "record_type": "combination",
                "name": combination,
                "threshold": threshold,
            }
            for modality, count in counts.items():
                row[f"selected_{modality}"] = int(count)
            threshold_rows.append(row)

    if np.isnan(condition_prediction).any() or np.isnan(history_prediction).any():
        raise AssertionError("Missing baseline predictions")
    for combination, prediction in continuous_predictions.items():
        if np.isnan(prediction).any():
            raise AssertionError(f"Missing predictions for {combination}")
        if np.isnan(binary_scores[combination]).any():
            raise AssertionError(f"Missing binary score for {combination}")
        if np.any(fixed_binary_predictions[combination] < 0) or np.any(tuned_binary_predictions[combination] < 0):
            raise AssertionError(f"Missing binary predictions for {combination}")

    continuous_rows: list[dict[str, Any]] = []
    continuous_rows.extend(
        _continuous_rows_from_predictions(
            frame,
            _array_predictions_by_target(condition_prediction),
            record_type="baseline",
            baseline="condition_only",
            combination="",
        )
    )
    continuous_rows.extend(
        _continuous_rows_from_predictions(
            frame,
            _array_predictions_by_target(history_prediction),
            record_type="baseline",
            baseline="history",
            combination="",
        )
    )
    for combination in COMBINATIONS:
        continuous_rows.extend(
            _continuous_rows_from_predictions(
                frame,
                _array_predictions_by_target(continuous_predictions[combination]),
                record_type="combination",
                baseline="",
                combination=combination,
            )
        )

    truth_binary = frame[BINARY_TARGET].to_numpy(dtype=int)
    binary_rows: list[dict[str, Any]] = []
    for name, scores, fixed, tuned in (
        ("condition_only", condition_scores, condition_fixed_binary, condition_tuned_binary),
        ("history", history_scores, history_fixed_binary, history_tuned_binary),
    ):
        binary_rows.append(
            {
                "record_type": "baseline",
                "baseline": name,
                "combination": "",
                "operating_point": "fixed_0.50",
                **_binary_metrics_from_predictions(truth_binary, scores, fixed),
            }
        )
        binary_rows.append(
            {
                "record_type": "baseline",
                "baseline": name,
                "combination": "",
                "operating_point": f"fold_tuned_recall_{min_recall:.2f}",
                **_binary_metrics_from_predictions(truth_binary, scores, tuned),
            }
        )
    for combination in COMBINATIONS:
        binary_rows.append(
            {
                "record_type": "combination",
                "baseline": "",
                "combination": combination,
                "operating_point": "fixed_0.50",
                **_binary_metrics_from_predictions(
                    truth_binary,
                    binary_scores[combination],
                    fixed_binary_predictions[combination],
                ),
            }
        )
        binary_rows.append(
            {
                "record_type": "combination",
                "baseline": "",
                "combination": combination,
                "operating_point": f"fold_tuned_recall_{min_recall:.2f}",
                **_binary_metrics_from_predictions(
                    truth_binary,
                    binary_scores[combination],
                    tuned_binary_predictions[combination],
                ),
            }
        )

    oof_rows: list[dict[str, Any]] = []
    for row_index, source_row in frame.iterrows():
        for combination in COMBINATIONS:
            for target in LABEL_TARGETS:
                target_index = LABEL_TARGETS.index(target)
                oof_rows.append(
                    {
                        "combination": combination,
                        "participant_id": source_row["participant_id"],
                        "condition": source_row["condition"],
                        "presentation_position": int(source_row["presentation_position"]),
                        "target": target,
                        "truth": float(source_row[target]),
                        "prediction": float(continuous_predictions[combination][row_index, target_index]),
                        "condition_only_prediction": float(condition_prediction[row_index, target_index]),
                        "history_prediction": float(history_prediction[row_index, target_index]),
                    }
                )
    return {
        "frame": frame,
        "continuous_metrics": pd.DataFrame(continuous_rows),
        "binary_metrics": pd.DataFrame(binary_rows),
        "oof_predictions": pd.DataFrame(oof_rows),
        "thresholds": pd.DataFrame(threshold_rows),
        "loadings": pd.DataFrame(loading_rows),
    }


def _head_to_head_continuous(continuous: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("V", "A", "single video"),
        ("PV", "PA", "physio + video"),
        ("PHEV", "PHEA", "replace original video in PHEV"),
        ("PHEV", "PHEVA", "add A4 on top of original video"),
        ("PHE", "PHEA", "add A4 to PHE"),
    ]
    rows: list[dict[str, Any]] = []
    combos = continuous.loc[continuous["record_type"].eq("combination")]
    baselines = continuous.loc[continuous["record_type"].eq("baseline")]
    for target in CONTINUOUS_TARGETS:
        target_frame = combos.loc[combos["target"].eq(target)].set_index("combination")
        baseline_frame = baselines.loc[baselines["target"].eq(target)].set_index("baseline")
        for combination in ("A", "V", "PHEA", "PHEVA"):
            if combination in target_frame.index and "condition_only" in baseline_frame.index:
                row = target_frame.loc[combination]
                base = baseline_frame.loc["condition_only"]
                rows.append(
                    {
                        "target": target,
                        "comparison": "vs condition_only baseline",
                        "original_combination": "condition_only",
                        "attended_combination": combination,
                        "delta_spearman": float(row["spearman"] - base["spearman"]),
                        "delta_within_spearman": float(
                            row["within_participant_spearman_mean"]
                            - base["within_participant_spearman_mean"]
                        ),
                        "delta_pairwise_accuracy": float(
                            row["pairwise_ranking_accuracy"] - base["pairwise_ranking_accuracy"]
                        ),
                        "delta_top1_hit": float(row["top1_condition_hit"] - base["top1_condition_hit"]),
                        "delta_top1_regret": float(row["top1_regret"] - base["top1_regret"]),
                    }
                )
        for original, attended, comparison in pairs:
            if original not in target_frame.index or attended not in target_frame.index:
                continue
            left = target_frame.loc[original]
            right = target_frame.loc[attended]
            rows.append(
                {
                    "target": target,
                    "comparison": comparison,
                    "original_combination": original,
                    "attended_combination": attended,
                    "delta_spearman": float(right["spearman"] - left["spearman"]),
                    "delta_within_spearman": float(
                        right["within_participant_spearman_mean"]
                        - left["within_participant_spearman_mean"]
                    ),
                    "delta_pairwise_accuracy": float(
                        right["pairwise_ranking_accuracy"] - left["pairwise_ranking_accuracy"]
                    ),
                    "delta_top1_hit": float(right["top1_condition_hit"] - left["top1_condition_hit"]),
                    "delta_top1_regret": float(right["top1_regret"] - left["top1_regret"]),
                }
            )
    return pd.DataFrame(rows)


def _head_to_head_binary(binary: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("V", "A", "single video"),
        ("PV", "PA", "physio + video"),
        ("PHEV", "PHEA", "replace original video in PHEV"),
        ("PHEV", "PHEVA", "add A4 on top of original video"),
        ("PHE", "PHEA", "add A4 to PHE"),
    ]
    rows: list[dict[str, Any]] = []
    for operating_point, group in binary.groupby("operating_point", sort=False):
        combos = group.loc[group["record_type"].eq("combination")].set_index("combination")
        baselines = group.loc[group["record_type"].eq("baseline")].set_index("baseline")
        for combination in ("A", "V", "PHEA", "PHEVA", "P"):
            if combination in combos.index and "condition_only" in baselines.index:
                row = combos.loc[combination]
                base = baselines.loc["condition_only"]
                rows.append(
                    {
                        "operating_point": operating_point,
                        "comparison": "vs condition_only baseline",
                        "original_combination": "condition_only",
                        "attended_combination": combination,
                        "delta_roc_auc": float(row["roc_auc"] - base["roc_auc"]),
                        "delta_pr_auc": float(row["pr_auc"] - base["pr_auc"]),
                        "delta_recall": float(row["recall"] - base["recall"]),
                        "delta_precision": float(row["precision"] - base["precision"]),
                        "delta_false_negatives": float(
                            row["false_negatives"] - base["false_negatives"]
                        ),
                    }
                )
        for original, attended, comparison in pairs:
            if original not in combos.index or attended not in combos.index:
                continue
            left = combos.loc[original]
            right = combos.loc[attended]
            rows.append(
                {
                    "operating_point": operating_point,
                    "comparison": comparison,
                    "original_combination": original,
                    "attended_combination": attended,
                    "delta_roc_auc": float(right["roc_auc"] - left["roc_auc"]),
                    "delta_pr_auc": float(right["pr_auc"] - left["pr_auc"]),
                    "delta_recall": float(right["recall"] - left["recall"]),
                    "delta_precision": float(right["precision"] - left["precision"]),
                    "delta_false_negatives": float(right["false_negatives"] - left["false_negatives"]),
                }
            )
    return pd.DataFrame(rows)


def _continuous_row(row: pd.Series) -> str:
    return (
        f"| {row['combination']} | {_fmt(row['spearman'])} | "
        f"{_fmt(row['within_participant_spearman_mean'])} | "
        f"{_fmt(row['pairwise_ranking_accuracy'], percent=True)} | "
        f"{_fmt(row['top1_condition_hit'], percent=True)} | "
        f"{_fmt(row['top1_regret'])} |"
    )


def _binary_row(row: pd.Series) -> str:
    return (
        f"| {row['combination']} | {_fmt(row['roc_auc'])} | {_fmt(row['pr_auc'])} | "
        f"{_fmt(row['recall'], percent=True)} | {_fmt(row['precision'], percent=True)} | "
        f"{_fmt(row['f1'])} | {int(row['false_negatives'])} |"
    )


def _loading_summary(loadings: pd.DataFrame) -> pd.DataFrame:
    if loadings.empty:
        return loadings
    return (
        loadings.groupby(["component", "target"], as_index=False)
        .agg(
            loading_mean=("loading", "mean"),
            loading_std=("loading", "std"),
            explained_variance_ratio_mean=("explained_variance_ratio", "mean"),
        )
        .sort_values(["component", "target"])
    )


def write_report(
    report_path: Path,
    *,
    continuous: pd.DataFrame,
    binary: pd.DataFrame,
    head_continuous: pd.DataFrame,
    head_binary: pd.DataFrame,
    loading_summary: pd.DataFrame,
    output_dir: Path,
    parameters: dict[str, Any],
) -> None:
    continuous_combos = continuous.loc[continuous["record_type"].eq("combination")]
    sections: list[str] = []
    for target in CONTINUOUS_TARGETS:
        top = continuous_combos.loc[continuous_combos["target"].eq(target)].sort_values(
            ["within_participant_spearman_mean", "spearman"],
            ascending=[False, False],
        )
        sections.extend(
            [
                f"## {target} latent-residual ranking top combinations",
                "",
                "| Combination | Global Spearman | Within Spearman mean | Pairwise acc | Top1 hit | Top1 regret |",
                "|---|---:|---:|---:|---:|---:|",
                *[_continuous_row(row) for _, row in top.head(10).iterrows()],
                "",
            ]
        )
    fixed_binary = binary.loc[
        (binary["record_type"].eq("combination")) & (binary["operating_point"].eq("fixed_0.50"))
    ].sort_values(["pr_auc", "roc_auc"], ascending=[False, False])
    tuned_binary = binary.loc[
        (binary["record_type"].eq("combination")) & binary["operating_point"].str.startswith("fold_tuned")
    ].sort_values(["recall", "pr_auc"], ascending=[False, False])
    loading_rows = [
        "| Component | Target | Loading mean | Loading sd | EVR mean |",
        "|---:|---|---:|---:|---:|",
    ]
    for _, row in loading_summary.iterrows():
        loading_rows.append(
            f"| {int(row['component'])} | {row['target']} | {_fmt(row['loading_mean'])} | "
            f"{_fmt(row['loading_std'])} | {_fmt(row['explained_variance_ratio_mean'])} |"
        )
    head_bin_rows = [
        "| Operating point | 对照 | 原组合 | 新组合 | Δ ROC-AUC | Δ PR-AUC | Δ recall | Δ precision | Δ FN |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in head_binary.iterrows():
        if row["comparison"] != "vs condition_only baseline" and row["original_combination"] not in {"V", "PHEV", "PHE"}:
            continue
        head_bin_rows.append(
            "| {op} | {comparison} | {original} | {attended} | {d_roc:+.4f} | {d_pr:+.4f} | "
            "{d_rec:+.1%} | {d_prec:+.1%} | {d_fn:+.0f} |".format(
                op=row["operating_point"],
                comparison=row["comparison"],
                original=row["original_combination"],
                attended=row["attended_combination"],
                d_roc=row["delta_roc_auc"],
                d_pr=row["delta_pr_auc"],
                d_rec=row["delta_recall"],
                d_prec=row["delta_precision"],
                d_fn=row["delta_false_negatives"],
            )
        )
    lines = [
        "# Latent-state multi-task benchmark",
        "",
        "## 口径",
        "",
        "- Uses NeuroKit ECG base features plus A4 features.",
        "- Trains fold-local latent residual state from relaxation, pleasantness, calm, discomfort.",
        "- Final label prediction = condition-only label baseline + predicted latent residual.",
        "- Continuous labels are evaluated by ranking metrics; high_discomfort is evaluated as binary risk.",
        "- Latent transform, feature selection, model fit, and threshold selection are all fold-local.",
        "",
        "## Latent axes",
        "",
        *loading_rows,
        "",
        "## High-discomfort head-to-head",
        "",
        *head_bin_rows,
        "",
        *sections,
        "## High-discomfort fixed-threshold top combinations",
        "",
        "| Combination | ROC-AUC | PR-AUC | Recall | Precision | F1 | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[_binary_row(row) for _, row in fixed_binary.head(12).iterrows()],
        "",
        "## High-discomfort fold-tuned recall top combinations",
        "",
        "| Combination | ROC-AUC | PR-AUC | Recall | Precision | F1 | FN |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[_binary_row(row) for _, row in tuned_binary.head(12).iterrows()],
        "",
        "## Outputs",
        "",
        f"- Continuous metrics: `{(output_dir / 'latent_continuous_ranking_metrics.csv').as_posix()}`",
        f"- Binary metrics: `{(output_dir / 'latent_binary_high_discomfort_metrics.csv').as_posix()}`",
        f"- Head-to-head continuous: `{(output_dir / 'latent_head_to_head_continuous.csv').as_posix()}`",
        f"- Head-to-head binary: `{(output_dir / 'latent_head_to_head_binary.csv').as_posix()}`",
        f"- OOF predictions: `{(output_dir / 'latent_oof_predictions.csv').as_posix()}`",
        f"- Latent loadings: `{(output_dir / 'latent_loadings.csv').as_posix()}`",
        "",
        "## Parameters",
        "",
        f"- Latent components: {parameters['latent_components']}.",
        f"- Ridge alpha: {parameters['ridge_alpha']}.",
        f"- Feature cap per modality: {parameters['feature_limit_per_modality']}.",
        f"- Fold-local tuned threshold target recall: {parameters['min_recall']}.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    base_features_path: Path,
    a4_features_path: Path,
    output_dir: Path,
    report_path: Path,
    latent_components: int,
    ridge_alpha: float,
    feature_limit_per_modality: int,
    min_recall: float,
) -> dict[str, Any]:
    if not base_features_path.exists():
        raise FileNotFoundError(base_features_path)
    if not a4_features_path.exists():
        raise FileNotFoundError(a4_features_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(base_features_path)
    a4 = pd.read_csv(a4_features_path)
    frame = base.merge(a4, on=["participant_id", "condition"], how="left", validate="one_to_one")
    a4_cols = [name for name in frame.columns if name.startswith(A4_PREFIX)]
    if not a4_cols:
        raise ValueError("Merged frame has no A4 feature columns")
    if frame[a4_cols].isna().all(axis=1).any():
        raise ValueError("Some participant-condition rows have no A4 features")
    result = evaluate_latent_multitask(
        frame,
        latent_components=latent_components,
        ridge_alpha=ridge_alpha,
        feature_limit_per_modality=feature_limit_per_modality,
        min_recall=min_recall,
    )
    continuous = result["continuous_metrics"]
    binary = result["binary_metrics"]
    oof = result["oof_predictions"]
    thresholds = result["thresholds"]
    loadings = result["loadings"]
    loading_summary = _loading_summary(loadings)
    head_continuous = _head_to_head_continuous(continuous)
    head_binary = _head_to_head_binary(binary)

    continuous.to_csv(output_dir / "latent_continuous_ranking_metrics.csv", index=False)
    binary.to_csv(output_dir / "latent_binary_high_discomfort_metrics.csv", index=False)
    oof.to_csv(output_dir / "latent_oof_predictions.csv", index=False)
    thresholds.to_csv(output_dir / "latent_thresholds.csv", index=False)
    loadings.to_csv(output_dir / "latent_loadings.csv", index=False)
    loading_summary.to_csv(output_dir / "latent_loading_summary.csv", index=False)
    head_continuous.to_csv(output_dir / "latent_head_to_head_continuous.csv", index=False)
    head_binary.to_csv(output_dir / "latent_head_to_head_binary.csv", index=False)
    write_report(
        report_path,
        continuous=continuous,
        binary=binary,
        head_continuous=head_continuous,
        head_binary=head_binary,
        loading_summary=loading_summary,
        output_dir=output_dir,
        parameters={
            "latent_components": latent_components,
            "ridge_alpha": ridge_alpha,
            "feature_limit_per_modality": feature_limit_per_modality,
            "min_recall": min_recall,
        },
    )
    summary = {
        "output_dir": str(output_dir),
        "report": str(report_path),
        "n_rows": int(len(result["frame"])),
        "participants": int(result["frame"]["participant_id"].nunique()),
        "a4_feature_columns": int(len(a4_cols)),
        "positive_high_discomfort": int(result["frame"][BINARY_TARGET].sum()),
        "latent_components": int(latent_components),
        "continuous_metrics": str(output_dir / "latent_continuous_ranking_metrics.csv"),
        "binary_metrics": str(output_dir / "latent_binary_high_discomfort_metrics.csv"),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run latent-state multi-task benchmark.")
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--a4-features", type=Path, default=DEFAULT_A4_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--latent-components", type=int, default=2)
    parser.add_argument("--ridge-alpha", type=float, default=old_fusion.RIDGE_ALPHA)
    parser.add_argument("--feature-limit-per-modality", type=int, default=old_fusion.FEATURES_PER_MODALITY)
    parser.add_argument("--min-recall", type=float, default=0.70)
    args = parser.parse_args()
    summary = run(
        base_features_path=args.base_features,
        a4_features_path=args.a4_features,
        output_dir=args.output_dir,
        report_path=args.report,
        latent_components=args.latent_components,
        ridge_alpha=args.ridge_alpha,
        feature_limit_per_modality=args.feature_limit_per_modality,
        min_recall=args.min_recall,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
