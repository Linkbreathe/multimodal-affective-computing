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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from real_time_ml.data.io import normalize_participant_id  # noqa: E402


DEFAULT_BASE_FEATURES = ROOT / "artifacts" / "features" / "ecg_neurokit" / "condition_features.csv"
DEFAULT_A4_FEATURES = ROOT / "artifacts" / "fusion_four_channel_ranking" / "a4_condition_features.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "pairwise_ranker_warm_start"
DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "pairwise_ranker_warm_start_zh.md"

TARGET = "relaxation"
COMBINATIONS = ("V", "A", "VA", "PHEV", "PHEA", "PHEVA")
PREFIX_COUNTS = (1, 2, 3, 4)
PRIMARY_COMBINATION = "V"
PRIMARY_PREFIX_COUNT = 3
PRIMARY_PREDICTOR = "condition_plus_sensor_ranker"
FEATURE_LIMIT_PER_MODALITY = 20
BOOTSTRAP_ITERATIONS = 5000
SIGN_FLIP_ITERATIONS = 5000
RANDOM_SEED = 20260704

MODALITY_PREFIXES = {
    "P": ("eeg_", "ecg_"),
    "H": ("head_",),
    "E": ("eye_",),
    "V": ("video_",),
    "A": ("a4_",),
}
MODALITY_ORDER = ("P", "H", "E", "V", "A")
PREDICTORS = (
    "condition_only",
    "sensor_ranker",
    "condition_plus_sensor_ranker",
    "residual_sensor_ranker",
    "condition_plus_residual_ranker",
    "calibrated_blend",
    "calibrated_residual_blend",
)


def _dependencies():
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Pairwise ranker dependencies are missing; use rtml-p002-p016") from error
    return SimpleImputer, LogisticRegression, Pipeline, StandardScaler


def _fmt(value: Any, digits: int = 4, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.1%}" if percent else f"{number:.{digits}f}"


def _safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(truth) < 3 or np.nanstd(truth) <= 1e-12 or np.nanstd(prediction) <= 1e-12:
        return 0.0
    value = float(spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def _pairwise_accuracy_arrays(truth: np.ndarray, prediction: np.ndarray) -> float:
    correct = 0.0
    total = 0
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            truth_delta = truth[left] - truth[right]
            prediction_delta = prediction[left] - prediction[right]
            if abs(truth_delta) <= 1e-12:
                continue
            total += 1
            if abs(prediction_delta) <= 1e-12:
                correct += 0.5
            elif np.sign(truth_delta) == np.sign(prediction_delta):
                correct += 1.0
    return float(correct / total) if total else float("nan")


def _participant_metrics(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for participant, group in frame.groupby("participant_id", sort=True):
        ordered = group.sort_values("presentation_position", kind="stable")
        truth = ordered["truth"].to_numpy(dtype=float)
        prediction = ordered[prediction_column].to_numpy(dtype=float)
        if len(truth) == 0:
            continue
        predicted_best = int(np.argmax(prediction))
        rows.append(
            {
                "participant_id": participant,
                "n_eval_conditions": int(len(ordered)),
                "spearman": _safe_spearman(truth, prediction),
                "pairwise_accuracy": _pairwise_accuracy_arrays(truth, prediction),
                "top1_hit": float(truth[predicted_best] == np.max(truth)),
                "top1_regret": float(np.max(truth) - truth[predicted_best]),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_participant_metrics(participant_metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "n_participants": int(participant_metrics["participant_id"].nunique()),
        "n_eval_conditions": int(participant_metrics["n_eval_conditions"].sum()),
        "within_spearman": float(participant_metrics["spearman"].mean()),
        "pairwise_accuracy": float(participant_metrics["pairwise_accuracy"].mean()),
        "top1_hit": float(participant_metrics["top1_hit"].mean()),
        "top1_regret": float(participant_metrics["top1_regret"].mean()),
    }


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(BOOTSTRAP_ITERATIONS, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = float(values.mean())
    if abs(observed) <= 1e-12:
        return 1.0
    rng = np.random.default_rng(RANDOM_SEED)
    signs = rng.choice([-1.0, 1.0], size=(SIGN_FLIP_ITERATIONS, len(values)), replace=True)
    null = (signs * values).mean(axis=1)
    if observed > 0:
        return float((np.sum(null >= observed) + 1) / (len(null) + 1))
    return float((np.sum(null <= observed) + 1) / (len(null) + 1))


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_id", "condition", "presentation_position", TARGET}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input frame missing required columns: {missing}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].map(normalize_participant_id)
    frame["condition"] = frame["condition"].astype(str)
    frame["presentation_position"] = pd.to_numeric(frame["presentation_position"], errors="raise")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
    if frame[["participant_id", "condition"]].duplicated().any():
        raise ValueError("Input must contain one row per participant-condition")
    if len(frame) != 135:
        raise ValueError(f"Expected 135 participant-condition rows; found {len(frame)}")
    return frame.reset_index(drop=True)


def _load_frame(base_features_path: Path, a4_features_path: Path) -> pd.DataFrame:
    if not base_features_path.exists():
        raise FileNotFoundError(base_features_path)
    if not a4_features_path.exists():
        raise FileNotFoundError(a4_features_path)
    base = pd.read_csv(base_features_path)
    a4 = pd.read_csv(a4_features_path)
    base["participant_id"] = base["participant_id"].map(normalize_participant_id)
    a4["participant_id"] = a4["participant_id"].map(normalize_participant_id)
    frame = base.merge(a4, on=["participant_id", "condition"], how="left", validate="one_to_one")
    a4_columns = [column for column in frame.columns if column.startswith("a4_")]
    if not a4_columns:
        raise ValueError("Merged frame has no a4_ columns")
    if frame[a4_columns].isna().all(axis=1).any():
        raise ValueError("Some participant-condition rows have no A4 features")
    return _validate_frame(frame)


def _columns_for_modality(frame: pd.DataFrame, modality: str) -> list[str]:
    prefixes = MODALITY_PREFIXES[modality]
    return sorted(column for column in frame.columns if column.startswith(prefixes))


def _candidate_columns(frame: pd.DataFrame, combination: str) -> dict[str, list[str]]:
    return {
        modality: _columns_for_modality(frame, modality)
        for modality in MODALITY_ORDER
        if modality in combination
    }


def _condition_scores(train: pd.DataFrame) -> dict[str, float]:
    return train.groupby("condition", sort=True)[TARGET].mean().to_dict()


def _score_conditions(frame: pd.DataFrame, score_by_condition: dict[str, float]) -> np.ndarray:
    fallback = float(np.mean(list(score_by_condition.values())))
    return frame["condition"].map(lambda condition: score_by_condition.get(condition, fallback)).to_numpy(dtype=float)


def _pair_indexes(group: pd.DataFrame) -> list[tuple[int, int]]:
    return list(combinations(range(len(group)), 2))


def _pairwise_training_arrays(
    train: pd.DataFrame,
    columns: list[str],
    condition_score: np.ndarray,
    ranking_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = train[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    truth = train[TARGET].to_numpy(dtype=float) if ranking_values is None else np.asarray(ranking_values, dtype=float)
    pair_diffs: list[np.ndarray] = []
    truth_deltas: list[float] = []
    condition_deltas: list[float] = []
    labels: list[int] = []
    for _, group in train.groupby("participant_id", sort=True):
        indexes = group.index.to_numpy(dtype=int)
        local_pairs = _pair_indexes(group)
        for local_left, local_right in local_pairs:
            left = indexes[local_left]
            right = indexes[local_right]
            truth_delta = truth[left] - truth[right]
            if abs(truth_delta) <= 1e-12:
                continue
            pair_diffs.append(matrix[left] - matrix[right])
            truth_deltas.append(float(truth_delta))
            condition_deltas.append(float(condition_score[left] - condition_score[right]))
            labels.append(int(truth_delta > 0))
    if not pair_diffs:
        raise ValueError("No non-tied training pairs")
    return (
        np.vstack(pair_diffs),
        np.asarray(truth_deltas, dtype=float),
        np.asarray(condition_deltas, dtype=float),
        np.asarray(labels, dtype=int),
    )


def _rank_features(
    pair_diffs: np.ndarray,
    residual_deltas: np.ndarray,
    columns: list[str],
    limit: int,
) -> list[str]:
    ranked: list[tuple[float, str]] = []
    target = np.asarray(residual_deltas, dtype=float)
    if np.nanstd(target) <= 1e-12:
        return columns[:limit]
    for index, column in enumerate(columns):
        values = pair_diffs[:, index]
        valid = np.isfinite(values) & np.isfinite(target)
        if valid.sum() < 5 or np.nanstd(values[valid]) <= 1e-12:
            continue
        corr = float(np.corrcoef(values[valid], target[valid])[0, 1])
        if np.isfinite(corr):
            ranked.append((abs(corr), column))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, column in ranked[:limit]]


def _select_features(
    train: pd.DataFrame,
    combination: str,
    condition_score: np.ndarray,
) -> tuple[list[str], dict[str, int]]:
    selected: list[str] = []
    counts: dict[str, int] = {}
    for modality, columns in _candidate_columns(train, combination).items():
        pair_diffs, truth_delta, condition_delta, _ = _pairwise_training_arrays(
            train,
            columns,
            condition_score,
        )
        residual_delta = truth_delta - condition_delta
        retained = _rank_features(
            pair_diffs,
            residual_delta,
            columns,
            FEATURE_LIMIT_PER_MODALITY,
        )
        selected.extend(retained)
        counts[modality] = len(retained)
    return sorted(selected), counts


def _make_pairwise_model():
    SimpleImputer, LogisticRegression, Pipeline, StandardScaler = _dependencies()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _fit_ranker(
    train: pd.DataFrame,
    columns: list[str],
    condition_score: np.ndarray,
    *,
    include_condition_delta: bool,
    ranking_values: np.ndarray | None = None,
) -> Any:
    pair_diffs, _truth_delta, condition_delta, labels = _pairwise_training_arrays(
        train,
        columns,
        condition_score,
        ranking_values=ranking_values,
    )
    if include_condition_delta:
        x = np.column_stack([condition_delta, pair_diffs])
    else:
        x = pair_diffs
    x = np.vstack([x, -x])
    y = np.concatenate([labels, 1 - labels])
    model = _make_pairwise_model()
    model.fit(x, y)
    return model


def _ranker_scores(
    model: Any,
    rows: pd.DataFrame,
    columns: list[str],
    condition_score: np.ndarray,
    *,
    include_condition_delta: bool,
) -> np.ndarray:
    if len(rows) == 1:
        return np.array([0.0], dtype=float)
    matrix = rows[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scores = np.zeros(len(rows), dtype=float)
    counts = np.zeros(len(rows), dtype=float)
    for left, right in combinations(range(len(rows)), 2):
        diff = matrix[left] - matrix[right]
        if include_condition_delta:
            diff = np.concatenate([[condition_score[left] - condition_score[right]], diff])
        probability = float(model.predict_proba(diff.reshape(1, -1))[0, 1])
        scores[left] += probability
        scores[right] += 1.0 - probability
        counts[left] += 1.0
        counts[right] += 1.0
    return np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(np.nanstd(values))
    if sd <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.nanmean(values))) / sd


def _calibration_blend_weight(calibration: pd.DataFrame, ranker_column: str) -> float:
    if len(calibration) < 2:
        return 0.0
    truth = calibration["truth"].to_numpy(dtype=float)
    condition = _zscore(calibration["condition_only"].to_numpy(dtype=float))
    ranker = _zscore(calibration[ranker_column].to_numpy(dtype=float))
    best_weight = 0.0
    best_score = -np.inf
    for weight in np.linspace(0.0, 1.0, 11):
        score = (1.0 - weight) * condition + weight * ranker
        metric = _pairwise_accuracy_arrays(truth, score)
        if np.isfinite(metric) and metric > best_score:
            best_score = metric
            best_weight = float(weight)
    return best_weight


def _evaluate_fold(
    frame: pd.DataFrame,
    participant: str,
    combination: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = frame.loc[~frame["participant_id"].eq(participant)].copy().reset_index(drop=True)
    test = frame.loc[frame["participant_id"].eq(participant)].copy()
    test = test.sort_values("presentation_position", kind="stable").reset_index(drop=True)
    score_by_condition = _condition_scores(train)
    train_condition_score = _score_conditions(train, score_by_condition)
    test_condition_score = _score_conditions(test, score_by_condition)
    selected_columns, counts = _select_features(train, combination, train_condition_score)
    if not selected_columns:
        raise ValueError(f"No selected features for {combination}")
    sensor_model = _fit_ranker(
        train,
        selected_columns,
        train_condition_score,
        include_condition_delta=False,
    )
    condition_sensor_model = _fit_ranker(
        train,
        selected_columns,
        train_condition_score,
        include_condition_delta=True,
    )
    train_residual_values = train[TARGET].to_numpy(dtype=float) - train_condition_score
    residual_model = _fit_ranker(
        train,
        selected_columns,
        train_condition_score,
        include_condition_delta=False,
        ranking_values=train_residual_values,
    )
    all_sensor_score = _ranker_scores(
        sensor_model,
        test,
        selected_columns,
        test_condition_score,
        include_condition_delta=False,
    )
    all_condition_sensor_score = _ranker_scores(
        condition_sensor_model,
        test,
        selected_columns,
        test_condition_score,
        include_condition_delta=True,
    )
    all_residual_score = _ranker_scores(
        residual_model,
        test,
        selected_columns,
        test_condition_score,
        include_condition_delta=False,
    )
    scored = test[
        [
            "participant_id",
            "condition",
            "presentation_position",
            TARGET,
        ]
    ].copy()
    scored = scored.rename(columns={TARGET: "truth"})
    scored["condition_only"] = test_condition_score
    scored["sensor_ranker"] = all_sensor_score
    scored["condition_plus_sensor_ranker"] = all_condition_sensor_score
    scored["residual_sensor_ranker"] = all_residual_score
    prediction_rows: list[dict[str, Any]] = []
    for prefix_count in PREFIX_COUNTS:
        calibration = scored.loc[scored["presentation_position"] <= prefix_count].copy()
        evaluation = scored.loc[scored["presentation_position"] > prefix_count].copy()
        if evaluation.empty:
            continue
        weight = _calibration_blend_weight(calibration, "condition_plus_sensor_ranker")
        residual_weight = _calibration_blend_weight(calibration, "residual_sensor_ranker")
        condition_eval = _zscore(evaluation["condition_only"].to_numpy(dtype=float))
        ranker_eval = _zscore(evaluation["condition_plus_sensor_ranker"].to_numpy(dtype=float))
        residual_eval = _zscore(evaluation["residual_sensor_ranker"].to_numpy(dtype=float))
        evaluation["condition_plus_residual_ranker"] = condition_eval + residual_eval
        evaluation["calibrated_blend"] = (1.0 - weight) * condition_eval + weight * ranker_eval
        evaluation["calibrated_residual_blend"] = (
            (1.0 - residual_weight) * condition_eval + residual_weight * residual_eval
        )
        for _, row in evaluation.iterrows():
            for predictor in PREDICTORS:
                prediction_rows.append(
                    {
                        "participant_id": participant,
                        "condition": row["condition"],
                        "presentation_position": int(row["presentation_position"]),
                        "target": TARGET,
                        "combination": combination,
                        "prefix_count": int(prefix_count),
                        "predictor": predictor,
                        "truth": float(row["truth"]),
                        "score": float(row[predictor]),
                        "calibrated_blend_weight": weight,
                        "calibrated_residual_blend_weight": residual_weight,
                    }
                )
    selection_rows = [
        {
            "held_out_participant": participant,
            "combination": combination,
            "selected_feature_count": int(len(selected_columns)),
            **{f"selected_{modality}": int(counts.get(modality, 0)) for modality in MODALITY_ORDER},
        }
    ]
    return prediction_rows, selection_rows


def _evaluate_pairwise_ranker(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    participants = sorted(frame["participant_id"].unique())
    for participant in participants:
        for combination in COMBINATIONS:
            fold_predictions, fold_selection = _evaluate_fold(frame, participant, combination)
            prediction_rows.extend(fold_predictions)
            selection_rows.extend(fold_selection)
    predictions = pd.DataFrame(prediction_rows)
    selection = pd.DataFrame(selection_rows)
    metric_rows: list[dict[str, Any]] = []
    participant_metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for (combination, prefix_count, predictor), group in predictions.groupby(
        ["combination", "prefix_count", "predictor"],
        sort=True,
    ):
        participant = _participant_metrics(group, "score")
        participant_metric_rows.extend(
            {
                "combination": combination,
                "prefix_count": int(prefix_count),
                "predictor": predictor,
                **row.to_dict(),
            }
            for _, row in participant.iterrows()
        )
        metric_rows.append(
            {
                "combination": combination,
                "prefix_count": int(prefix_count),
                "predictor": predictor,
                **_aggregate_participant_metrics(participant),
            }
        )
    participant_metrics = pd.DataFrame(participant_metric_rows)
    metrics = pd.DataFrame(metric_rows)
    for (combination, prefix_count), group in participant_metrics.groupby(
        ["combination", "prefix_count"],
        sort=True,
    ):
        baseline = group.loc[group["predictor"].eq("condition_only")].set_index("participant_id")
        for predictor in PREDICTORS:
            if predictor == "condition_only":
                continue
            candidate = group.loc[group["predictor"].eq(predictor)].set_index("participant_id")
            common = sorted(set(candidate.index).intersection(baseline.index))
            if not common:
                continue
            for metric in ("pairwise_accuracy", "spearman", "top1_hit", "top1_regret"):
                candidate_values = candidate.loc[common, metric].to_numpy(dtype=float)
                baseline_values = baseline.loc[common, metric].to_numpy(dtype=float)
                if metric == "top1_regret":
                    delta = baseline_values - candidate_values
                else:
                    delta = candidate_values - baseline_values
                low, high = _bootstrap_ci(delta, rng)
                comparison_rows.append(
                    {
                        "combination": combination,
                        "prefix_count": int(prefix_count),
                        "predictor": predictor,
                        "baseline_predictor": "condition_only",
                        "metric": metric,
                        "delta_mean_positive_is_better": float(np.nanmean(delta)),
                        "delta_ci_low": low,
                        "delta_ci_high": high,
                        "sign_flip_p_value": _sign_flip_pvalue(delta),
                    }
                )
    return {
        "predictions": predictions,
        "selection": selection,
        "metrics": metrics,
        "participant_metrics": participant_metrics,
        "comparisons": pd.DataFrame(comparison_rows),
    }


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    percent_columns: set[str] | None = None,
) -> str:
    percent_columns = percent_columns or set()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                values.append(value)
            elif column.startswith("n_") or column in {"prefix_count"}:
                values.append(str(int(value)))
            else:
                values.append(_fmt(value, percent=column in percent_columns))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    report_path: Path,
    *,
    output_dir: Path,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    selection: pd.DataFrame,
) -> None:
    primary_metric = metrics.loc[
        metrics["combination"].eq(PRIMARY_COMBINATION)
        & metrics["prefix_count"].eq(PRIMARY_PREFIX_COUNT)
        & metrics["predictor"].isin(
            [
                "condition_only",
                PRIMARY_PREDICTOR,
                "condition_plus_residual_ranker",
                "calibrated_blend",
                "calibrated_residual_blend",
            ]
        )
    ].copy()
    primary_comparison = comparisons.loc[
        comparisons["combination"].eq(PRIMARY_COMBINATION)
        & comparisons["prefix_count"].eq(PRIMARY_PREFIX_COUNT)
        & comparisons["predictor"].eq(PRIMARY_PREDICTOR)
        & comparisons["metric"].eq("pairwise_accuracy")
    ].iloc[0]
    m3_pairwise = comparisons.loc[
        comparisons["prefix_count"].eq(PRIMARY_PREFIX_COUNT)
        & comparisons["metric"].eq("pairwise_accuracy")
    ].copy()
    best_m3_pairwise = m3_pairwise.sort_values(
        "delta_mean_positive_is_better",
        ascending=False,
    ).iloc[0]
    v_residual_m3 = m3_pairwise.loc[
        m3_pairwise["combination"].eq("V")
        & m3_pairwise["predictor"].eq("condition_plus_residual_ranker")
    ].iloc[0]
    curve = metrics.loc[
        metrics["combination"].isin(["V", "A"])
        & metrics["predictor"].isin(
            [
                "condition_only",
                "condition_plus_sensor_ranker",
                "condition_plus_residual_ranker",
            ]
        )
    ].sort_values(["combination", "prefix_count", "predictor"])
    combo_summary = metrics.loc[
        metrics["prefix_count"].eq(PRIMARY_PREFIX_COUNT)
        & metrics["predictor"].eq("condition_plus_sensor_ranker")
    ].sort_values(["pairwise_accuracy", "within_spearman"], ascending=[False, False])
    delta_summary = comparisons.loc[
        comparisons["metric"].eq("pairwise_accuracy")
        & comparisons["predictor"].isin(
            [
                "condition_plus_sensor_ranker",
                "condition_plus_residual_ranker",
                "calibrated_blend",
                "calibrated_residual_blend",
            ]
        )
        & comparisons["prefix_count"].eq(PRIMARY_PREFIX_COUNT)
    ].sort_values(["predictor", "delta_mean_positive_is_better"], ascending=[True, False])
    selection_summary = (
        selection.groupby("combination", sort=True)
        .agg(
            selected_feature_count_mean=("selected_feature_count", "mean"),
            selected_P_mean=("selected_P", "mean"),
            selected_H_mean=("selected_H", "mean"),
            selected_E_mean=("selected_E", "mean"),
            selected_V_mean=("selected_V", "mean"),
            selected_A_mean=("selected_A", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# Direct pairwise ranker warm-start benchmark",
        "",
        "## 评测口径",
        "",
        "- 主终点固定为 relaxation、V、prefix m=3、pairwise accuracy。",
        "- 训练样本是其它被试的 condition 对；标签是同一被试内左侧 condition 是否更 relaxing。",
        "- 模型输入是两段 condition 的特征差。`condition_plus_sensor_ranker` 同时给入 condition baseline 差和传感器特征差。",
        "- `condition_plus_residual_ranker` 则先学 `truth - condition_baseline` 的残差排序，再加回 condition baseline。",
        "- 评价只看 prefix 之后剩余 condition 的个体内排序；推断单位是 participant。",
        "- `calibrated_blend` 用 prefix 内已知评分在 condition baseline 和 ranker 之间选一个简单 blend weight，是 secondary。",
        "",
        "## 主结果",
        "",
        f"- V / m=3 / condition_plus_sensor_ranker 相对 condition-only 的 pairwise 增量："
        f"{_fmt(primary_comparison['delta_mean_positive_is_better'], percent=True)}，"
        f"95% CI [{_fmt(primary_comparison['delta_ci_low'], percent=True)}, "
        f"{_fmt(primary_comparison['delta_ci_high'], percent=True)}]，"
        f"p={_fmt(primary_comparison['sign_flip_p_value'])}。",
        f"- m=3 所有尝试里 pairwise 增量最高的是 `{best_m3_pairwise['combination']}` / "
        f"`{best_m3_pairwise['predictor']}`："
        f"{_fmt(best_m3_pairwise['delta_mean_positive_is_better'], percent=True)}，"
        f"CI [{_fmt(best_m3_pairwise['delta_ci_low'], percent=True)}, "
        f"{_fmt(best_m3_pairwise['delta_ci_high'], percent=True)}]。",
        f"- 单 V 的 residual-ranker 在 m=3 是 "
        f"{_fmt(v_residual_m3['delta_mean_positive_is_better'], percent=True)}；"
        "说明这次 residual 排序没有把 V 的弱信号放大。",
        "",
        "解释：直接把 loss 改成 pairwise 并没有解决问题。PHEV 出现正向趋势，"
        "但 CI 很宽；A 和 VA 在 direct pairwise 下反而明显拖累。"
        "所以当前证据更支持“V/PHEV 有弱排序信号”，不支持“pairwise ranker 已经击败个性化 baseline”。",
        "",
        "主终点 m=3 明细：",
        "",
        _markdown_table(
            primary_metric[
                [
                    "combination",
                    "prefix_count",
                    "predictor",
                    "pairwise_accuracy",
                    "within_spearman",
                    "top1_hit",
                    "top1_regret",
                ]
            ],
            [
                "combination",
                "prefix_count",
                "predictor",
                "pairwise_accuracy",
                "within_spearman",
                "top1_hit",
                "top1_regret",
            ],
            percent_columns={"pairwise_accuracy", "top1_hit"},
        ),
        "",
        "## V/A warm-start 曲线",
        "",
        _markdown_table(
            curve[
                [
                    "combination",
                    "prefix_count",
                    "predictor",
                    "pairwise_accuracy",
                    "within_spearman",
                    "top1_hit",
                    "top1_regret",
                ]
            ],
            [
                "combination",
                "prefix_count",
                "predictor",
                "pairwise_accuracy",
                "within_spearman",
                "top1_hit",
                "top1_regret",
            ],
            percent_columns={"pairwise_accuracy", "top1_hit"},
        ),
        "",
        "## m=3 各组合 ranker 排名",
        "",
        _markdown_table(
            combo_summary[
                [
                    "combination",
                    "pairwise_accuracy",
                    "within_spearman",
                    "top1_hit",
                    "top1_regret",
                ]
            ],
            [
                "combination",
                "pairwise_accuracy",
                "within_spearman",
                "top1_hit",
                "top1_regret",
            ],
            percent_columns={"pairwise_accuracy", "top1_hit"},
        ),
        "",
        "## m=3 pairwise 增量 vs condition-only",
        "",
        _markdown_table(
            delta_summary[
                [
                    "combination",
                    "predictor",
                    "delta_mean_positive_is_better",
                    "delta_ci_low",
                    "delta_ci_high",
                    "sign_flip_p_value",
                ]
            ],
            [
                "combination",
                "predictor",
                "delta_mean_positive_is_better",
                "delta_ci_low",
                "delta_ci_high",
                "sign_flip_p_value",
            ],
            percent_columns={"delta_mean_positive_is_better", "delta_ci_low", "delta_ci_high"},
        ),
        "",
        "## 特征选择规模",
        "",
        _markdown_table(
            selection_summary,
            [
                "combination",
                "selected_feature_count_mean",
                "selected_P_mean",
                "selected_H_mean",
                "selected_E_mean",
                "selected_V_mean",
                "selected_A_mean",
            ],
        ),
        "",
        "## 输出文件",
        "",
        f"- pairwise predictions: `{(output_dir / 'pairwise_predictions.csv').as_posix()}`",
        f"- metrics: `{(output_dir / 'pairwise_metrics.csv').as_posix()}`",
        f"- participant metrics: `{(output_dir / 'pairwise_participant_metrics.csv').as_posix()}`",
        f"- comparisons: `{(output_dir / 'pairwise_comparisons.csv').as_posix()}`",
        f"- feature selection: `{(output_dir / 'feature_selection_summary.csv').as_posix()}`",
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
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(base_features_path, a4_features_path)
    result = _evaluate_pairwise_ranker(frame)
    result["predictions"].to_csv(output_dir / "pairwise_predictions.csv", index=False)
    result["selection"].to_csv(output_dir / "feature_selection_summary.csv", index=False)
    result["metrics"].to_csv(output_dir / "pairwise_metrics.csv", index=False)
    result["participant_metrics"].to_csv(output_dir / "pairwise_participant_metrics.csv", index=False)
    result["comparisons"].to_csv(output_dir / "pairwise_comparisons.csv", index=False)
    _write_report(
        report_path,
        output_dir=output_dir,
        metrics=result["metrics"],
        comparisons=result["comparisons"],
        selection=result["selection"],
    )
    primary = result["comparisons"].loc[
        result["comparisons"]["combination"].eq(PRIMARY_COMBINATION)
        & result["comparisons"]["prefix_count"].eq(PRIMARY_PREFIX_COUNT)
        & result["comparisons"]["predictor"].eq(PRIMARY_PREDICTOR)
        & result["comparisons"]["metric"].eq("pairwise_accuracy")
    ].iloc[0]
    summary = {
        "output_dir": str(output_dir),
        "report": str(report_path),
        "target": TARGET,
        "primary_combination": PRIMARY_COMBINATION,
        "primary_prefix_count": PRIMARY_PREFIX_COUNT,
        "primary_predictor": PRIMARY_PREDICTOR,
        "primary_delta_pairwise": float(primary["delta_mean_positive_is_better"]),
        "primary_delta_ci_low": float(primary["delta_ci_low"]),
        "primary_delta_ci_high": float(primary["delta_ci_high"]),
        "primary_p_value": float(primary["sign_flip_p_value"]),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct pairwise warm-start ranker.")
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--a4-features", type=Path, default=DEFAULT_A4_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    summary = run(
        base_features_path=args.base_features,
        a4_features_path=args.a4_features,
        output_dir=args.output_dir,
        report_path=args.report,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
