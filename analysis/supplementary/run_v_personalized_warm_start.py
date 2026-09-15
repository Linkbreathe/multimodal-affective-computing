from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from mac.data.io import normalize_participant_id  # noqa: E402


DEFAULT_BASE_FEATURES = ROOT / "artifacts" / "features" / "ecg_neurokit" / "condition_features.csv"
DEFAULT_A4_FEATURES = ROOT / "artifacts" / "fusion_four_channel_ranking" / "a4_condition_features.csv"
DEFAULT_V_K4_OOF = ROOT / "artifacts" / "fusion_latent_multitask_k4" / "latent_oof_predictions.csv"
DEFAULT_A_K1_OOF = ROOT / "artifacts" / "fusion_latent_multitask_k1" / "latent_oof_predictions.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "personalized_warm_start_v_k4"
DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "personalized_warm_start_v_k4_zh.md"

TARGETS = ("relaxation", "pleasantness", "calm", "discomfort")
PREFIX_COUNTS = (1, 2, 3, 4)
PRIMARY_TARGET = "relaxation"
PRIMARY_PREFIX_COUNT = 3
PRIMARY_MODEL = "V_k4"
PRIMARY_MODEL_PREDICTOR = "model_plus_condition_eb"
BOOTSTRAP_ITERATIONS = 3000
PERMUTATION_ITERATIONS = 2000
RANDOM_SEED = 20260704


def _fmt(value: Any, digits: int = 4, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    if percent:
        return f"{number:.1%}"
    return f"{number:.{digits}f}"


def _safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(truth) < 3 or np.nanstd(truth) <= 1e-12 or np.nanstd(prediction) <= 1e-12:
        return 0.0
    value = float(spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def _safe_pearson(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(truth) < 3 or np.nanstd(truth) <= 1e-12 or np.nanstd(prediction) <= 1e-12:
        return 0.0
    value = float(pearsonr(truth, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def _pairwise_accuracy_arrays(truth: np.ndarray, prediction: np.ndarray) -> float:
    correct = 0.0
    total = 0
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            truth_delta = truth[left] - truth[right]
            pred_delta = prediction[left] - prediction[right]
            if abs(truth_delta) <= 1e-12:
                continue
            total += 1
            if abs(pred_delta) <= 1e-12:
                correct += 0.5
            elif np.sign(truth_delta) == np.sign(pred_delta):
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
        pairwise = _pairwise_accuracy_arrays(truth, prediction)
        predicted_best_index = int(np.argmax(prediction))
        rows.append(
            {
                "participant_id": participant,
                "n_eval_conditions": int(len(ordered)),
                "mae": float(np.mean(np.abs(truth - prediction))),
                "spearman": _safe_spearman(truth, prediction),
                "pairwise_accuracy": pairwise,
                "top1_hit": float(truth[predicted_best_index] == np.max(truth)),
                "top1_regret": float(np.max(truth) - truth[predicted_best_index]),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_participant_metrics(participant_metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "n_participants": int(participant_metrics["participant_id"].nunique()),
        "n_eval_conditions": int(participant_metrics["n_eval_conditions"].sum()),
        "mae": float(participant_metrics["mae"].mean()),
        "within_spearman": float(participant_metrics["spearman"].mean()),
        "pairwise_accuracy": float(participant_metrics["pairwise_accuracy"].mean()),
        "top1_hit": float(participant_metrics["top1_hit"].mean()),
        "top1_regret": float(participant_metrics["top1_regret"].mean()),
    }


def _bootstrap_ci(
    participant_rows: pd.DataFrame,
    *,
    value_column: str,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    values = participant_rows[value_column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
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
    signs = rng.choice([-1.0, 1.0], size=(PERMUTATION_ITERATIONS, len(values)), replace=True)
    null = (signs * values).mean(axis=1)
    if observed > 0:
        return float((np.sum(null >= observed) + 1) / (len(null) + 1))
    return float((np.sum(null <= observed) + 1) / (len(null) + 1))


def _validate_oof(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {
        "combination",
        "participant_id",
        "condition",
        "presentation_position",
        "target",
        "truth",
        "prediction",
        "condition_only_prediction",
        "history_prediction",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].map(normalize_participant_id)
    frame["condition"] = frame["condition"].astype(str)
    frame["combination"] = frame["combination"].astype(str)
    frame["target"] = frame["target"].astype(str)
    for column in ("presentation_position", "truth", "prediction", "condition_only_prediction", "history_prediction"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    keys = ["combination", "participant_id", "condition", "target"]
    if frame[keys].duplicated().any():
        raise ValueError(f"{path} has duplicated prediction keys")
    return frame


def _model_frame(path: Path, combination: str, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = _validate_oof(pd.read_csv(path), path)
    model = frame.loc[frame["combination"].eq(combination)].copy()
    if model.empty:
        raise ValueError(f"{path} has no combination {combination!r}")
    model = model.rename(
        columns={
            "prediction": "model_prediction",
            "condition_only_prediction": "condition_prediction",
            "history_prediction": "history_prediction",
        }
    )
    model["model_source"] = source_name
    return model[
        [
            "model_source",
            "participant_id",
            "condition",
            "presentation_position",
            "target",
            "truth",
            "model_prediction",
            "condition_prediction",
            "history_prediction",
        ]
    ].reset_index(drop=True)


def _variance_share_within_condition(frame: pd.DataFrame, value_column: str) -> dict[str, float]:
    values = frame[value_column].to_numpy(dtype=float)
    finite = np.isfinite(values)
    values = values[finite]
    conditions = frame.loc[finite, "condition"].astype(str).to_numpy()
    if len(values) < 3:
        return {
            "total_variance": float("nan"),
            "condition_mean_variance": float("nan"),
            "within_condition_variance": float("nan"),
            "within_condition_share": float("nan"),
        }
    total_ss = float(np.sum((values - np.mean(values)) ** 2))
    if total_ss <= 1e-12:
        return {
            "total_variance": 0.0,
            "condition_mean_variance": 0.0,
            "within_condition_variance": 0.0,
            "within_condition_share": float("nan"),
        }
    condition_mean = pd.Series(values).groupby(conditions).transform("mean").to_numpy(dtype=float)
    within_ss = float(np.sum((values - condition_mean) ** 2))
    return {
        "total_variance": float(total_ss / max(len(values) - 1, 1)),
        "condition_mean_variance": float((total_ss - within_ss) / max(len(values) - 1, 1)),
        "within_condition_variance": float(within_ss / max(len(values) - 1, 1)),
        "within_condition_share": float(within_ss / total_ss),
    }


def _prediction_personalization_diagnostics(models: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostic_rows: list[dict[str, Any]] = []
    permutation_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for (source, target), group in models.groupby(["model_source", "target"], sort=True):
        group = group.sort_values(["participant_id", "presentation_position"], kind="stable").reset_index(drop=True)
        prediction_stats = _variance_share_within_condition(group, "model_prediction")
        truth_stats = _variance_share_within_condition(group, "truth")
        pred_residual = (
            group["model_prediction"]
            - group.groupby("condition", sort=False)["model_prediction"].transform("mean")
        ).to_numpy(dtype=float)
        truth_residual = (
            group["truth"]
            - group.groupby("condition", sort=False)["truth"].transform("mean")
        ).to_numpy(dtype=float)
        condition_residual = (
            group["condition_prediction"]
            - group.groupby("condition", sort=False)["condition_prediction"].transform("mean")
        ).to_numpy(dtype=float)
        diagnostic_rows.append(
            {
                "model_source": source,
                "target": target,
                "prediction_within_condition_share": prediction_stats["within_condition_share"],
                "truth_within_condition_share": truth_stats["within_condition_share"],
                "prediction_residual_pearson_vs_truth_residual": _safe_pearson(truth_residual, pred_residual),
                "prediction_residual_spearman_vs_truth_residual": _safe_spearman(truth_residual, pred_residual),
                "condition_prediction_residual_sd": float(np.nanstd(condition_residual)),
                "model_prediction_residual_sd": float(np.nanstd(pred_residual)),
                "truth_residual_sd": float(np.nanstd(truth_residual)),
            }
        )
        if target != PRIMARY_TARGET:
            continue
        observed_pairwise = _pairwise_accuracy_grouped(group, "model_prediction")
        observed_within = _within_spearman_grouped(group, "model_prediction")
        perm_pairwise = np.empty(PERMUTATION_ITERATIONS, dtype=float)
        perm_within = np.empty(PERMUTATION_ITERATIONS, dtype=float)
        for index in range(PERMUTATION_ITERATIONS):
            shuffled = group.copy()
            shuffled["model_prediction"] = _shuffle_within_condition(
                shuffled,
                "model_prediction",
                rng,
            )
            perm_pairwise[index] = _pairwise_accuracy_grouped(shuffled, "model_prediction")
            perm_within[index] = _within_spearman_grouped(shuffled, "model_prediction")
        permutation_rows.extend(
            [
                {
                    "model_source": source,
                    "target": target,
                    "metric": "pairwise_accuracy",
                    "observed": observed_pairwise,
                    "null_mean": float(np.nanmean(perm_pairwise)),
                    "p_value": float((np.sum(perm_pairwise >= observed_pairwise) + 1) / (len(perm_pairwise) + 1)),
                },
                {
                    "model_source": source,
                    "target": target,
                    "metric": "within_spearman",
                    "observed": observed_within,
                    "null_mean": float(np.nanmean(perm_within)),
                    "p_value": float((np.sum(perm_within >= observed_within) + 1) / (len(perm_within) + 1)),
                },
            ]
        )
    return pd.DataFrame(diagnostic_rows), pd.DataFrame(permutation_rows)


def _pairwise_accuracy_grouped(frame: pd.DataFrame, prediction_column: str) -> float:
    participant = _participant_metrics(frame, prediction_column)
    return float(participant["pairwise_accuracy"].mean())


def _within_spearman_grouped(frame: pd.DataFrame, prediction_column: str) -> float:
    participant = _participant_metrics(frame, prediction_column)
    return float(participant["spearman"].mean())


def _shuffle_within_condition(frame: pd.DataFrame, column: str, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.empty(len(frame), dtype=float)
    for _, indexes in frame.groupby("condition", sort=False).groups.items():
        locs = np.asarray(list(indexes), dtype=int)
        shuffled[locs] = rng.permutation(frame.loc[locs, column].to_numpy(dtype=float))
    return shuffled


def _training_shrinkage_parameters(frame: pd.DataFrame, participant: str, target: str, base_column: str) -> tuple[float, float, float]:
    train = frame.loc[~frame["participant_id"].eq(participant) & frame["target"].eq(target)].copy()
    residual = train["truth"].to_numpy(dtype=float) - train[base_column].to_numpy(dtype=float)
    if len(residual) < 3:
        return 0.0, 0.0, 1.0
    train["residual"] = residual
    participant_means = train.groupby("participant_id")["residual"].mean()
    within_variances = train.groupby("participant_id")["residual"].var(ddof=1).dropna()
    sigma2 = float(within_variances.mean()) if len(within_variances) else float(np.var(residual, ddof=1))
    tau2_raw = float(participant_means.var(ddof=1)) if len(participant_means) > 1 else 0.0
    mean_n = float(train.groupby("participant_id")["residual"].size().mean())
    tau2 = max(0.0, tau2_raw - sigma2 / max(mean_n, 1.0))
    return tau2, max(sigma2, 1e-12), float(participant_means.mean())


def _eb_bias(
    calibration_residual: np.ndarray,
    *,
    tau2: float,
    sigma2: float,
    prior_mean: float,
) -> tuple[float, float]:
    calibration_residual = np.asarray(calibration_residual, dtype=float)
    calibration_residual = calibration_residual[np.isfinite(calibration_residual)]
    if len(calibration_residual) == 0:
        return prior_mean, 0.0
    residual_mean = float(np.mean(calibration_residual))
    if tau2 <= 0.0:
        weight = 0.0
    else:
        weight = float(tau2 / (tau2 + sigma2 / len(calibration_residual)))
    return float(prior_mean + weight * (residual_mean - prior_mean)), weight


def _warm_start_rows(models: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, source_frame in models.groupby("model_source", sort=True):
        for target, target_frame in source_frame.groupby("target", sort=True):
            for participant, participant_frame in target_frame.groupby("participant_id", sort=True):
                ordered = participant_frame.sort_values("presentation_position", kind="stable").reset_index(drop=True)
                for prefix_count in PREFIX_COUNTS:
                    if len(ordered) <= prefix_count:
                        continue
                    calibration = ordered.iloc[:prefix_count]
                    evaluation = ordered.iloc[prefix_count:].copy()
                    cond_tau2, cond_sigma2, cond_prior = _training_shrinkage_parameters(
                        source_frame,
                        participant,
                        target,
                        "condition_prediction",
                    )
                    model_tau2, model_sigma2, model_prior = _training_shrinkage_parameters(
                        source_frame,
                        participant,
                        target,
                        "model_prediction",
                    )
                    condition_bias, condition_weight = _eb_bias(
                        calibration["truth"].to_numpy(dtype=float)
                        - calibration["condition_prediction"].to_numpy(dtype=float),
                        tau2=cond_tau2,
                        sigma2=cond_sigma2,
                        prior_mean=cond_prior,
                    )
                    model_bias, model_weight = _eb_bias(
                        calibration["truth"].to_numpy(dtype=float)
                        - calibration["model_prediction"].to_numpy(dtype=float),
                        tau2=model_tau2,
                        sigma2=model_sigma2,
                        prior_mean=model_prior,
                    )
                    last_truth = float(calibration.sort_values("presentation_position", kind="stable")["truth"].iloc[-1])
                    for _, row in evaluation.iterrows():
                        condition_eb = float(np.clip(row["condition_prediction"] + condition_bias, 0.0, 1.0))
                        model_eb = float(np.clip(row["model_prediction"] + model_bias, 0.0, 1.0))
                        model_plus_condition_eb = float(
                            np.clip(
                                condition_eb + (row["model_prediction"] - row["condition_prediction"]),
                                0.0,
                                1.0,
                            )
                        )
                        rows.append(
                            {
                                "model_source": source,
                                "target": target,
                                "prefix_count": int(prefix_count),
                                "participant_id": participant,
                                "condition": row["condition"],
                                "presentation_position": int(row["presentation_position"]),
                                "truth": float(row["truth"]),
                                "condition_raw": float(row["condition_prediction"]),
                                "model_raw": float(row["model_prediction"]),
                                "history_last": last_truth,
                                "condition_eb": condition_eb,
                                "model_eb": model_eb,
                                "model_plus_condition_eb": model_plus_condition_eb,
                                "condition_bias": condition_bias,
                                "model_bias": model_bias,
                                "condition_shrinkage_weight": condition_weight,
                                "model_shrinkage_weight": model_weight,
                            }
                        )
    return pd.DataFrame(rows)


def _warm_start_metrics(warm_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictors = ("condition_raw", "condition_eb", "history_last", "model_raw", "model_eb", "model_plus_condition_eb")
    metric_rows: list[dict[str, Any]] = []
    participant_metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    for (source, target, prefix_count), group in warm_rows.groupby(
        ["model_source", "target", "prefix_count"],
        sort=True,
    ):
        participant_by_predictor: dict[str, pd.DataFrame] = {}
        for predictor in predictors:
            participant = _participant_metrics(group, predictor)
            participant_by_predictor[predictor] = participant
            participant_metric_rows.extend(
                {
                    "model_source": source,
                    "target": target,
                    "prefix_count": int(prefix_count),
                    "predictor": predictor,
                    **row.to_dict(),
                }
                for _, row in participant.iterrows()
            )
            metric_rows.append(
                {
                    "model_source": source,
                    "target": target,
                    "prefix_count": int(prefix_count),
                    "predictor": predictor,
                    **_aggregate_participant_metrics(participant),
                }
            )
        baseline = participant_by_predictor["condition_eb"].set_index("participant_id")
        for predictor in ("model_raw", "model_eb", "model_plus_condition_eb", "history_last"):
            candidate = participant_by_predictor[predictor].set_index("participant_id")
            common = sorted(set(candidate.index).intersection(baseline.index))
            if not common:
                continue
            for metric in ("pairwise_accuracy", "spearman", "mae", "top1_hit", "top1_regret"):
                candidate_values = candidate.loc[common, metric].to_numpy(dtype=float)
                baseline_values = baseline.loc[common, metric].to_numpy(dtype=float)
                if metric in {"mae", "top1_regret"}:
                    delta = baseline_values - candidate_values
                else:
                    delta = candidate_values - baseline_values
                delta_frame = pd.DataFrame(
                    {
                        "participant_id": common,
                        "delta": delta,
                    }
                )
                low, high = _bootstrap_ci(
                    delta_frame,
                    value_column="delta",
                    rng=rng,
                    iterations=BOOTSTRAP_ITERATIONS,
                )
                comparison_rows.append(
                    {
                        "model_source": source,
                        "target": target,
                        "prefix_count": int(prefix_count),
                        "predictor": predictor,
                        "baseline_predictor": "condition_eb",
                        "metric": metric,
                        "delta_mean_positive_is_better": float(np.nanmean(delta)),
                        "delta_ci_low": low,
                        "delta_ci_high": high,
                        "sign_flip_p_value": _sign_flip_pvalue(delta),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(participant_metric_rows), pd.DataFrame(comparison_rows)


def _feature_variance_decomposition(base_features_path: Path, a4_features_path: Path) -> pd.DataFrame:
    if not base_features_path.exists():
        raise FileNotFoundError(base_features_path)
    if not a4_features_path.exists():
        raise FileNotFoundError(a4_features_path)
    base = pd.read_csv(base_features_path)
    a4 = pd.read_csv(a4_features_path)
    base["participant_id"] = base["participant_id"].map(normalize_participant_id)
    a4["participant_id"] = a4["participant_id"].map(normalize_participant_id)
    frame = base.merge(a4, on=["participant_id", "condition"], how="left", validate="one_to_one")
    groups = {
        "eeg": [column for column in frame.columns if column.startswith("eeg_")],
        "ecg": [column for column in frame.columns if column.startswith("ecg_")],
        "head": [column for column in frame.columns if column.startswith("head_")],
        "eye": [column for column in frame.columns if column.startswith("eye_")],
        "video": [column for column in frame.columns if column.startswith("video_")],
        "a4": [column for column in frame.columns if column.startswith("a4_")],
    }
    rows: list[dict[str, Any]] = []
    for modality, columns in groups.items():
        shares: list[float] = []
        usable = 0
        for column in columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().sum() < 10 or float(numeric.std(skipna=True)) <= 1e-12:
                continue
            temp = frame[["condition"]].copy()
            temp["value"] = numeric
            share = _variance_share_within_condition(temp, "value")["within_condition_share"]
            if np.isfinite(share):
                shares.append(share)
                usable += 1
        rows.append(
            {
                "modality": modality,
                "n_columns": int(len(columns)),
                "n_usable_columns": int(usable),
                "within_condition_share_mean": float(np.mean(shares)) if shares else float("nan"),
                "within_condition_share_median": float(np.median(shares)) if shares else float("nan"),
                "within_condition_share_p25": float(np.quantile(shares, 0.25)) if shares else float("nan"),
                "within_condition_share_p75": float(np.quantile(shares, 0.75)) if shares else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, percent_columns: set[str] | None = None) -> str:
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
    feature_variance: pd.DataFrame,
    diagnostics: pd.DataFrame,
    permutation: pd.DataFrame,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> None:
    primary_metric = metrics.loc[
        metrics["model_source"].eq(PRIMARY_MODEL)
        & metrics["target"].eq(PRIMARY_TARGET)
        & metrics["prefix_count"].eq(PRIMARY_PREFIX_COUNT)
        & metrics["predictor"].isin(["condition_eb", PRIMARY_MODEL_PREDICTOR])
    ].copy()
    primary_comparison = comparisons.loc[
        comparisons["model_source"].eq(PRIMARY_MODEL)
        & comparisons["target"].eq(PRIMARY_TARGET)
        & comparisons["prefix_count"].eq(PRIMARY_PREFIX_COUNT)
        & comparisons["predictor"].eq(PRIMARY_MODEL_PREDICTOR)
        & comparisons["metric"].eq("pairwise_accuracy")
    ].iloc[0]
    diag_key = diagnostics.loc[
        diagnostics["model_source"].eq(PRIMARY_MODEL) & diagnostics["target"].eq(PRIMARY_TARGET)
    ].iloc[0]
    perm_key = permutation.loc[
        permutation["model_source"].eq(PRIMARY_MODEL)
        & permutation["target"].eq(PRIMARY_TARGET)
        & permutation["metric"].eq("pairwise_accuracy")
    ].iloc[0]
    curve = metrics.loc[
        metrics["target"].eq(PRIMARY_TARGET)
        & metrics["model_source"].isin(["V_k4", "A_k1"])
        & metrics["predictor"].isin(["condition_eb", "model_plus_condition_eb"])
    ].copy()
    curve["label"] = curve["model_source"] + ":" + curve["predictor"]
    curve = curve[
        [
            "model_source",
            "prefix_count",
            "predictor",
            "pairwise_accuracy",
            "within_spearman",
            "top1_hit",
            "top1_regret",
        ]
    ].sort_values(["model_source", "prefix_count", "predictor"])
    comparison_key = comparisons.loc[
        comparisons["target"].eq(PRIMARY_TARGET)
        & comparisons["predictor"].eq("model_plus_condition_eb")
        & comparisons["metric"].eq("pairwise_accuracy")
    ].sort_values(["model_source", "prefix_count"])

    lines = [
        "# V-first 个性化 warm-start ranking",
        "",
        "## 评测口径",
        "",
        "- 主终点在看结果前固定：relaxation、V_k4、prefix m=3、pairwise accuracy。",
        "- warm-start 指先把前 m 个已经看过的 condition 当作个人校准标签，只评估剩下的 condition。",
        "- 基线是 condition-only 加 empirical-Bayes 的被试残差截距；这是一个更强的个性化基线。",
        "- `model_plus_condition_eb` 在同一个个性化基线上加入模型残差 `(model_raw - condition_raw)`。",
        "- 推断单位是 participant；CI 用 participant bootstrap，p 值用配对 sign-flip permutation。",
        "",
        "## 主结果",
        "",
        f"- V_k4 预测值的 condition 内方差占比：{_fmt(diag_key['prediction_within_condition_share'], percent=True)}。",
        f"- V_k4 预测残差 vs 真实 condition 残差的 Spearman：{_fmt(diag_key['prediction_residual_spearman_vs_truth_residual'])}。",
        f"- condition 内打乱 V_k4 预测后，pairwise accuracy：原始 {_fmt(perm_key['observed'], percent=True)}，"
        f"置换均值 {_fmt(perm_key['null_mean'], percent=True)}，p={_fmt(perm_key['p_value'])}。",
        f"- warm-start m=3 相对个性化 condition baseline 的增量："
        f"{_fmt(primary_comparison['delta_mean_positive_is_better'], percent=True)} pairwise accuracy，"
        f"95% CI [{_fmt(primary_comparison['delta_ci_low'], percent=True)}, "
        f"{_fmt(primary_comparison['delta_ci_high'], percent=True)}]，"
        f"p={_fmt(primary_comparison['sign_flip_p_value'])}。",
        "",
        "解释：特征层的 video 确实主要是 condition 内、被试间变化；但预测层还没有证明"
        " V_k4 正确利用了这些个体差异。V_k4 的 warm-start 排序增量是正的，"
        "但 CI 跨 0；condition 内置换没有变差，说明当前 V_k4 的个体映射不能作为已坐实结论。",
        "",
        "主终点 m=3 明细：",
        "",
        _markdown_table(
            primary_metric[
                [
                    "model_source",
                    "prefix_count",
                    "predictor",
                    "pairwise_accuracy",
                    "within_spearman",
                    "top1_hit",
                    "top1_regret",
                ]
            ],
            [
                "model_source",
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
        "## 特征方差分解",
        "",
        _markdown_table(
            feature_variance,
            [
                "modality",
                "n_usable_columns",
                "within_condition_share_mean",
                "within_condition_share_median",
                "within_condition_share_p25",
                "within_condition_share_p75",
            ],
            percent_columns={
                "within_condition_share_mean",
                "within_condition_share_median",
                "within_condition_share_p25",
                "within_condition_share_p75",
            },
        ),
        "",
        "## relaxation warm-start 曲线",
        "",
        _markdown_table(
            curve,
            [
                "model_source",
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
        "## 相对个性化 condition baseline 的 pairwise 增量",
        "",
        _markdown_table(
            comparison_key[
                [
                    "model_source",
                    "prefix_count",
                    "delta_mean_positive_is_better",
                    "delta_ci_low",
                    "delta_ci_high",
                    "sign_flip_p_value",
                ]
            ],
            [
                "model_source",
                "prefix_count",
                "delta_mean_positive_is_better",
                "delta_ci_low",
                "delta_ci_high",
                "sign_flip_p_value",
            ],
            percent_columns={"delta_mean_positive_is_better", "delta_ci_low", "delta_ci_high"},
        ),
        "",
        "## 输出文件",
        "",
        f"- 特征方差：`{(output_dir / 'feature_variance_decomposition.csv').as_posix()}`",
        f"- 预测层个性化诊断：`{(output_dir / 'prediction_personalization_diagnostics.csv').as_posix()}`",
        f"- condition 内置换：`{(output_dir / 'prediction_within_condition_permutation.csv').as_posix()}`",
        f"- warm-start 逐行预测：`{(output_dir / 'warm_start_predictions.csv').as_posix()}`",
        f"- warm-start 汇总指标：`{(output_dir / 'warm_start_metrics.csv').as_posix()}`",
        f"- warm-start participant 指标：`{(output_dir / 'warm_start_participant_metrics.csv').as_posix()}`",
        f"- warm-start 对比：`{(output_dir / 'warm_start_comparisons.csv').as_posix()}`",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    base_features_path: Path,
    a4_features_path: Path,
    v_k4_oof_path: Path,
    a_k1_oof_path: Path,
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    v_k4 = _model_frame(v_k4_oof_path, "V", "V_k4")
    a_k1 = _model_frame(a_k1_oof_path, "A", "A_k1")
    models = pd.concat([v_k4, a_k1], ignore_index=True)

    feature_variance = _feature_variance_decomposition(base_features_path, a4_features_path)
    diagnostics, permutation = _prediction_personalization_diagnostics(models)
    warm_rows = _warm_start_rows(models)
    metrics, participant_metrics, comparisons = _warm_start_metrics(warm_rows)

    feature_variance.to_csv(output_dir / "feature_variance_decomposition.csv", index=False)
    diagnostics.to_csv(output_dir / "prediction_personalization_diagnostics.csv", index=False)
    permutation.to_csv(output_dir / "prediction_within_condition_permutation.csv", index=False)
    warm_rows.to_csv(output_dir / "warm_start_predictions.csv", index=False)
    metrics.to_csv(output_dir / "warm_start_metrics.csv", index=False)
    participant_metrics.to_csv(output_dir / "warm_start_participant_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "warm_start_comparisons.csv", index=False)
    _write_report(
        report_path,
        feature_variance=feature_variance,
        diagnostics=diagnostics,
        permutation=permutation,
        metrics=metrics,
        comparisons=comparisons,
        output_dir=output_dir,
    )
    primary = comparisons.loc[
        comparisons["model_source"].eq(PRIMARY_MODEL)
        & comparisons["target"].eq(PRIMARY_TARGET)
        & comparisons["prefix_count"].eq(PRIMARY_PREFIX_COUNT)
        & comparisons["predictor"].eq(PRIMARY_MODEL_PREDICTOR)
        & comparisons["metric"].eq("pairwise_accuracy")
    ].iloc[0]
    summary = {
        "output_dir": str(output_dir),
        "report": str(report_path),
        "primary_target": PRIMARY_TARGET,
        "primary_model": PRIMARY_MODEL,
        "primary_prefix_count": PRIMARY_PREFIX_COUNT,
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
    parser = argparse.ArgumentParser(description="Run V-first personalized warm-start ranking diagnostics.")
    parser.add_argument("--base-features", type=Path, default=DEFAULT_BASE_FEATURES)
    parser.add_argument("--a4-features", type=Path, default=DEFAULT_A4_FEATURES)
    parser.add_argument("--v-k4-oof", type=Path, default=DEFAULT_V_K4_OOF)
    parser.add_argument("--a-k1-oof", type=Path, default=DEFAULT_A_K1_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    summary = run(
        base_features_path=args.base_features,
        a4_features_path=args.a4_features,
        v_k4_oof_path=args.v_k4_oof,
        a_k1_oof_path=args.a_k1_oof,
        output_dir=args.output_dir,
        report_path=args.report,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
