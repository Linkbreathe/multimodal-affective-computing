"""Research-only minimal multimodal LOPO Ridge benchmark.

The module has a deliberately narrow input boundary: one already aggregated
participant--Condition CSV. It neither invokes nor reads any window,
VideoMAE2, deep-model, runtime, or Unity artifact.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.utils import atomic_write_text


TARGETS = ("relaxation", "discomfort")
MODALITY_PREFIXES = {
    "P": ("eeg_", "ecg_"),
    "H": ("head_",),
    "E": ("eye_",),
    "V": ("video_",),
}
MODALITY_ORDER = tuple(MODALITY_PREFIXES)
COMBINATIONS = tuple(
    "".join(parts)
    for size in range(1, len(MODALITY_ORDER) + 1)
    for parts in combinations(MODALITY_ORDER, size)
)

SOURCE_RELATIVE_PATH = Path("video_ml") / "condition_features.csv"
OUTPUT_DIRECTORY = "fusion_minimal"
METRICS_FILENAME = "metrics.csv"
OOF_FILENAME = "oof_predictions.csv"
REPORT_FILENAME = "minimal_multimodal_fusion_zh.md"

EXPECTED_LABELS = 135
RIDGE_ALPHA = 10.0
FEATURES_PER_MODALITY = 20
HIGH_DISCOMFORT_TRUTH_THRESHOLD = 0.50
HIGH_DISCOMFORT_PREDICTION_THRESHOLD = 0.20
RANDOM_SIMULATIONS = 10_000

REQUIRED_COLUMNS = ("participant_id", "condition", "presentation_position", *TARGETS)
IDENTIFIER_AND_LABEL_COLUMNS = {
    "participant_id", "condition", "presentation_position", "label_source_row",
    "intensity", "frequency", "intensity_index", "frequency_index", "condition_index",
    "window_count", "window_count_expected", "relaxation", "discomfort",
    "relaxation_raw", "discomfort_raw", "pleasantness", "pleasantness_raw",
    "arousal_raw", "monotony", "monotony_raw", "visual_fit", "calm",
}


def _dependencies():
    try:
        import pandas as pd
        from scipy.stats import rankdata, spearmanr
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - environment failure
        raise RuntimeError("Minimal fusion benchmark dependencies are missing; create environment.yml") from error
    return pd, rankdata, spearmanr, SimpleImputer, Ridge, Pipeline, StandardScaler


def _validate_input(frame: Any, expected_labels: int | None) -> Any:
    pd, *_ = _dependencies()
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Minimal fusion input is missing required columns: {missing}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    for name in ("presentation_position", *TARGETS):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Minimal fusion input has missing identifier, order, or target values")
    if frame[["participant_id", "condition"]].duplicated().any():
        raise ValueError("Minimal fusion input must contain one row per participant--Condition")
    if expected_labels is not None and len(frame) != expected_labels:
        raise ValueError(f"Expected exactly {expected_labels} participant--Condition labels; found {len(frame)}")
    if frame["participant_id"].nunique() < 2:
        raise ValueError("Minimal fusion LOPO requires at least two participants")
    return frame.reset_index(drop=True)


def _modal_columns(frame: Any, modality: str) -> list[str]:
    """Return raw modality-prefixed features only; labels and context are excluded."""
    prefixes = MODALITY_PREFIXES[modality]
    return sorted(
        name for name in frame.columns
        if name not in IDENTIFIER_AND_LABEL_COLUMNS and name.startswith(prefixes)
    )


def _condition_baseline(train: Any, test: Any, target: str) -> tuple[np.ndarray, dict[str, float], float]:
    fallback = float(train[target].mean())
    by_condition = train.groupby("condition", sort=True)[target].mean().astype(float).to_dict()
    prediction = np.asarray(
        [float(by_condition.get(condition, fallback)) for condition in test["condition"]], dtype=float
    )
    return prediction, by_condition, fallback


def _history_baseline(test: Any, fallback: float, target: str) -> np.ndarray:
    """First Condition uses training mean; later Conditions use preceding truth."""
    output = np.full(len(test), fallback, dtype=float)
    for _, group in test.groupby("participant_id", sort=False):
        previous: float | None = None
        for index in group.sort_values("presentation_position", kind="stable").index:
            position = test.index.get_loc(index)
            output[position] = fallback if previous is None else previous
            previous = float(test.loc[index, target])
    return output


def _rank_features(train: Any, columns: list[str], residual: np.ndarray, limit: int) -> list[str]:
    """Rank valid train-fold features by absolute Pearson correlation and name ties."""
    pd, *_ = _dependencies()
    ranked: list[tuple[float, str]] = []
    residual = np.asarray(residual, dtype=float)
    for name in columns:
        values = pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(residual)
        if valid.sum() < 2:
            continue
        feature_values = values[valid]
        residual_values = residual[valid]
        if np.nanstd(feature_values) <= 0.0 or np.nanstd(residual_values) <= 0.0:
            continue
        correlation = float(np.corrcoef(feature_values, residual_values)[0, 1])
        if np.isfinite(correlation):
            ranked.append((abs(correlation), name))
    return [name for _, name in sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]]


def _selected_by_modality(train: Any, target: str) -> dict[str, list[str]]:
    _, baseline_map, fallback = _condition_baseline(train, train, target)
    baseline = np.asarray(
        [float(baseline_map.get(condition, fallback)) for condition in train["condition"]], dtype=float
    )
    residual = train[target].to_numpy(dtype=float) - baseline
    return {
        modality: _rank_features(train, _modal_columns(train, modality), residual, FEATURES_PER_MODALITY)
        for modality in MODALITY_ORDER
    }


def _select_fold_features(train: Any, target: str, combination: str) -> tuple[list[str], dict[str, int]]:
    """Publicly testable fold-local feature selection for one modality combination."""
    ranked = _selected_by_modality(train, target)
    return _combine_selected_features(ranked, combination)


def _combine_selected_features(
    selected_by_modality: dict[str, list[str]], combination: str
) -> tuple[list[str], dict[str, int]]:
    selected: list[str] = []
    counts: dict[str, int] = {}
    for modality in MODALITY_ORDER:
        retained = selected_by_modality[modality] if modality in combination else []
        selected.extend(retained)
        counts[modality] = len(retained)
    return sorted(selected), counts


def _ridge_pipeline() -> Any:
    _, _, _, SimpleImputer, Ridge, Pipeline, StandardScaler = _dependencies()
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def _matrix(frame: Any, columns: list[str]) -> Any:
    pd, *_ = _dependencies()
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def _safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    _, _, spearmanr, *_ = _dependencies()
    value = float(spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def _point_metrics(truth: np.ndarray, prediction: np.ndarray, *, discomfort: bool) -> dict[str, float]:
    metrics = {
        "mae": float(np.mean(np.abs(truth - prediction))),
        "spearman": _safe_spearman(truth, prediction),
    }
    if discomfort:
        high_truth = truth >= HIGH_DISCOMFORT_TRUTH_THRESHOLD
        high_prediction = prediction >= HIGH_DISCOMFORT_PREDICTION_THRESHOLD
        metrics.update({
            "high_recall": float(np.mean(high_prediction[high_truth])) if high_truth.any() else float("nan"),
            "high_precision": float(np.mean(high_truth[high_prediction])) if high_prediction.any() else 0.0,
            "high_false_negatives": float(np.sum(high_truth & ~high_prediction)),
        })
    return metrics


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p025": float(np.quantile(values, 0.025)),
        "p975": float(np.quantile(values, 0.975)),
    }


def _random_spearman(truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    _, rankdata, *_ = _dependencies()
    truth_rank = rankdata(truth)
    truth_centered = truth_rank - truth_rank.mean()
    prediction_rank = rankdata(predictions, axis=1)
    prediction_centered = prediction_rank - prediction_rank.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(prediction_centered, axis=1) * np.linalg.norm(truth_centered)
    return np.divide(
        prediction_centered @ truth_centered,
        denominator,
        out=np.zeros(len(predictions), dtype=float),
        where=denominator > 0,
    )


def random_uniform_baseline(
    truth_by_target: dict[str, np.ndarray], *, random_seed: int, simulations: int = RANDOM_SIMULATIONS
) -> dict[str, dict[str, dict[str, float]]]:
    """Generate independent Uniform(0,1) predictions per target, draw, and sample."""
    if simulations <= 0:
        raise ValueError("random_uniform simulations must be positive")
    rng = np.random.default_rng(random_seed)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for target in TARGETS:
        truth = np.asarray(truth_by_target[target], dtype=float)
        prediction = rng.uniform(0.0, 1.0, size=(simulations, len(truth)))
        values: dict[str, np.ndarray] = {
            "mae": np.mean(np.abs(prediction - truth), axis=1),
            "spearman": _random_spearman(truth, prediction),
        }
        if target == "discomfort":
            high_truth = truth >= HIGH_DISCOMFORT_TRUTH_THRESHOLD
            high_prediction = prediction >= HIGH_DISCOMFORT_PREDICTION_THRESHOLD
            values["high_recall"] = (
                high_prediction[:, high_truth].mean(axis=1)
                if high_truth.any() else np.full(simulations, np.nan, dtype=float)
            )
            positives = high_prediction.sum(axis=1)
            values["high_precision"] = np.divide(
                (high_prediction & high_truth).sum(axis=1), positives,
                out=np.zeros(simulations, dtype=float), where=positives > 0,
            )
            values["high_false_negatives"] = (high_truth & ~high_prediction).sum(axis=1).astype(float)
        output[target] = {name: _summary(value) for name, value in values.items()}
    return output


def _wide_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "relaxation_mae": float(metrics["relaxation"]["mae"]),
        "relaxation_spearman": float(metrics["relaxation"]["spearman"]),
        "discomfort_mae": float(metrics["discomfort"]["mae"]),
        "discomfort_spearman": float(metrics["discomfort"]["spearman"]),
        "discomfort_high_recall": float(metrics["discomfort"]["high_recall"]),
        "discomfort_high_precision": float(metrics["discomfort"]["high_precision"]),
        "discomfort_high_false_negatives": float(metrics["discomfort"]["high_false_negatives"]),
    }


def _wide_random(summary: dict[str, dict[str, dict[str, float]]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for target, metrics in summary.items():
        for metric, values in metrics.items():
            name = f"{target}_{metric}"
            output[name] = float(values["mean"])
            for statistic, value in values.items():
                output[f"{name}_{statistic}"] = float(value)
    return output


def _improvements(fused: dict[str, float], baseline: dict[str, float], name: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric, value in fused.items():
        if metric not in baseline:
            continue
        direction = -1.0 if metric.endswith(("_mae", "_false_negatives")) else 1.0
        output[f"{metric}_improvement_vs_{name}"] = float(direction * (value - baseline[metric]))
    return output


def _selection_summary(counts: dict[str, dict[str, list[int]]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for target in TARGETS:
        for modality in MODALITY_ORDER:
            values = np.asarray(counts[target][modality], dtype=float)
            prefix = f"selected_{modality}_{target}"
            output[f"{prefix}_min"] = float(values.min())
            output[f"{prefix}_mean"] = float(values.mean())
            output[f"{prefix}_max"] = float(values.max())
    return output


def evaluate_minimal_fusion_frame(
    frame: Any,
    *,
    random_seed: int,
    random_simulations: int = RANDOM_SIMULATIONS,
    expected_labels: int | None = EXPECTED_LABELS,
) -> dict[str, Any]:
    """Evaluate the 15 fixed modality combinations without file I/O."""
    pd, *_ = _dependencies()
    frame = _validate_input(frame, expected_labels)
    participants = sorted(frame["participant_id"].unique())
    truth_by_target = {target: frame[target].to_numpy(dtype=float) for target in TARGETS}
    predictions = {
        combination: {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
        for combination in COMBINATIONS
    }
    condition_only = {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
    history = {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
    selections = {
        combination: {target: {modality: [] for modality in MODALITY_ORDER} for target in TARGETS}
        for combination in COMBINATIONS
    }

    for participant in participants:
        test_mask = frame["participant_id"].eq(participant).to_numpy()
        test_indexes = np.flatnonzero(test_mask)
        train = frame.loc[~test_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)
        for target in TARGETS:
            baseline, _, fallback = _condition_baseline(train, test, target)
            condition_only[target][test_indexes] = baseline
            history[target][test_indexes] = _history_baseline(test, fallback, target)
            train_baseline, _, _ = _condition_baseline(train, train, target)
            residual = train[target].to_numpy(dtype=float) - train_baseline
            selected_by_modality = _selected_by_modality(train, target)
            for combination in COMBINATIONS:
                columns, counts = _combine_selected_features(selected_by_modality, combination)
                for modality, count in counts.items():
                    selections[combination][target][modality].append(count)
                if columns:
                    model = _ridge_pipeline()
                    model.fit(_matrix(train, columns), residual)
                    held_out_residual = model.predict(_matrix(test, columns))
                else:
                    held_out_residual = np.zeros(len(test), dtype=float)
                predictions[combination][target][test_indexes] = np.clip(
                    baseline + held_out_residual, 0.0, 1.0
                )

    for target in TARGETS:
        if np.isnan(condition_only[target]).any() or np.isnan(history[target]).any():
            raise AssertionError("LOPO baseline predictions must cover every participant--Condition row")
    for combination in COMBINATIONS:
        for target in TARGETS:
            if np.isnan(predictions[combination][target]).any():
                raise AssertionError("LOPO fusion predictions must cover every participant--Condition row")

    baseline_metrics = {
        "condition_only": {
            target: _point_metrics(truth_by_target[target], condition_only[target], discomfort=target == "discomfort")
            for target in TARGETS
        },
        "history": {
            target: _point_metrics(truth_by_target[target], history[target], discomfort=target == "discomfort")
            for target in TARGETS
        },
    }
    baseline_wide = {name: _wide_metrics(value) for name, value in baseline_metrics.items()}
    random_summary = random_uniform_baseline(
        truth_by_target, random_seed=random_seed, simulations=random_simulations
    )
    random_wide = _wide_random(random_summary)
    random_point = {
        name: value for name, value in random_wide.items()
        if not name.endswith(("_mean", "_p025", "_p975"))
    }

    combination_metrics: dict[str, dict[str, float]] = {}
    metric_rows: list[dict[str, Any]] = []
    for baseline_name in ("condition_only", "history"):
        metric_rows.append({
            "record_type": "baseline", "baseline": baseline_name, "combination": "",
            "research_only": True, **baseline_wide[baseline_name],
        })
    metric_rows.append({
        "record_type": "baseline", "baseline": "random_uniform", "combination": "",
        "research_only": True, "random_simulations": int(random_simulations), **random_wide,
    })
    for combination in COMBINATIONS:
        point = _wide_metrics({
            target: _point_metrics(
                truth_by_target[target], predictions[combination][target], discomfort=target == "discomfort"
            ) for target in TARGETS
        })
        combination_metrics[combination] = point
        metric_rows.append({
            "record_type": "combination", "baseline": "", "combination": combination,
            "research_only": True, "n_labels": len(frame), "ridge_alpha": RIDGE_ALPHA,
            "feature_limit_per_modality": FEATURES_PER_MODALITY,
            "high_discomfort_truth_threshold": HIGH_DISCOMFORT_TRUTH_THRESHOLD,
            "high_discomfort_prediction_threshold": HIGH_DISCOMFORT_PREDICTION_THRESHOLD,
            **point,
            **{f"condition_only_{name}": value for name, value in baseline_wide["condition_only"].items()},
            **{f"history_{name}": value for name, value in baseline_wide["history"].items()},
            **{f"random_uniform_{name}": value for name, value in random_wide.items()},
            **_improvements(point, random_point, "random_uniform_mean"),
            **_improvements(point, baseline_wide["condition_only"], "condition_only"),
            **_improvements(point, baseline_wide["history"], "history"),
            **_selection_summary(selections[combination]),
        })

    for full_combination in COMBINATIONS:
        if len(full_combination) < 2:
            continue
        for removed_modality in full_combination:
            remaining = "".join(modality for modality in full_combination if modality != removed_modality)
            full, reduced = combination_metrics[full_combination], combination_metrics[remaining]
            changes = {
                metric: (-1.0 if metric.endswith(("_mae", "_false_negatives")) else 1.0)
                * (full[metric] - reduced[metric])
                for metric in full
            }
            metric_rows.append({
                "record_type": "ablation", "baseline": "", "combination": full_combination,
                "full_combination": full_combination, "removed_modality": removed_modality,
                "remaining_combination": remaining, "research_only": True,
                **{f"full_{name}": value for name, value in full.items()},
                **{f"remaining_{name}": value for name, value in reduced.items()},
                **{f"{name}_improvement": value for name, value in changes.items()},
            })

    oof_rows: list[dict[str, Any]] = []
    for combination in COMBINATIONS:
        for row_index, source_row in frame.iterrows():
            for target in TARGETS:
                truth = float(source_row[target])
                prediction = float(predictions[combination][target][row_index])
                condition_prediction = float(condition_only[target][row_index])
                history_prediction = float(history[target][row_index])
                is_discomfort = target == "discomfort"
                oof_rows.append({
                    "combination": combination, "participant_id": source_row["participant_id"],
                    "condition": source_row["condition"],
                    "presentation_position": int(source_row["presentation_position"]), "target": target,
                    "truth": truth, "prediction": prediction, "absolute_error": abs(truth - prediction),
                    "condition_only_prediction": condition_prediction,
                    "condition_only_absolute_error": abs(truth - condition_prediction),
                    "history_prediction": history_prediction,
                    "history_absolute_error": abs(truth - history_prediction),
                    "high_discomfort_truth": int(truth >= HIGH_DISCOMFORT_TRUTH_THRESHOLD) if is_discomfort else np.nan,
                    "high_discomfort_prediction": int(prediction >= HIGH_DISCOMFORT_PREDICTION_THRESHOLD) if is_discomfort else np.nan,
                })
    return {
        "frame": frame, "metric_rows": metric_rows, "metrics": pd.DataFrame(metric_rows),
        "oof_predictions": pd.DataFrame(oof_rows), "combination_metrics": combination_metrics,
        "baseline_metrics": baseline_wide, "random_summary": random_summary,
        "random_wide": random_wide, "selection_counts": selections,
    }


def _format_number(value: float, *, percent: bool = False) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.1%}" if percent else f"{value:.4f}"


def _combination_table(
    metrics: dict[str, dict[str, float]], *, key: str, descending: bool = False, percent: bool = False
) -> list[str]:
    ordered = sorted(metrics.items(), key=lambda item: item[1][key], reverse=descending)
    return [
        f"| {rank} | {combination} | {_format_number(values[key], percent=percent)} |"
        for rank, (combination, values) in enumerate(ordered, start=1)
    ]


def _write_report(result: dict[str, Any], output: Path, *, random_simulations: int) -> None:
    baseline, random = result["baseline_metrics"], result["random_summary"]
    combination_metrics, metrics = result["combination_metrics"], result["metrics"]
    ablations = metrics.loc[metrics["record_type"].eq("ablation")]
    baseline_rows = [
        "| 基线 | Relaxation MAE | Relaxation Spearman | Discomfort MAE | Discomfort Spearman | 高 discomfort recall | precision | false negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in baseline.items():
        baseline_rows.append(
            "| {name} | {rel_mae} | {rel_rho} | {dis_mae} | {dis_rho} | {recall} | {precision} | {fn} |".format(
                name="Condition-only" if name == "condition_only" else "history",
                rel_mae=_format_number(values["relaxation_mae"]), rel_rho=_format_number(values["relaxation_spearman"]),
                dis_mae=_format_number(values["discomfort_mae"]), dis_rho=_format_number(values["discomfort_spearman"]),
                recall=_format_number(values["discomfort_high_recall"], percent=True),
                precision=_format_number(values["discomfort_high_precision"], percent=True),
                fn=int(values["discomfort_high_false_negatives"]),
            )
        )
    random_rows = ["| 指标 | 随机 Uniform(0,1) 均值 [2.5%, 97.5%] |", "|---|---|"]
    for target, metric in (
        ("relaxation", "mae"), ("relaxation", "spearman"), ("discomfort", "mae"),
        ("discomfort", "spearman"), ("discomfort", "high_recall"),
        ("discomfort", "high_precision"), ("discomfort", "high_false_negatives"),
    ):
        values = random[target][metric]
        random_rows.append(
            f"| {target} {metric} | {values['mean']:.4f} [{values['p025']:.4f}, {values['p975']:.4f}] |"
        )
    ablation_rows = [
        "| 完整组合 | 移除 | 剩余组合 | Relaxation MAE 改善 | Discomfort MAE 改善 | Recall 改善 | Precision 改善 | 假阴性减少 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in ablations.iterrows():
        ablation_rows.append(
            "| {full} | {removed} | {remaining} | {relaxation:+.4f} | {discomfort:+.4f} | {recall:+.4f} | {precision:+.4f} | {fn:+.0f} |".format(
                full=row["full_combination"], removed=row["removed_modality"], remaining=row["remaining_combination"],
                relaxation=row["relaxation_mae_improvement"], discomfort=row["discomfort_mae_improvement"],
                recall=row["discomfort_high_recall_improvement"], precision=row["discomfort_high_precision_improvement"],
                fn=row["discomfort_high_false_negatives_improvement"],
            )
        )
    lines = [
        "# 最小多模态融合基准（研究比较，不能部署）", "", "## 输入边界与固定协议", "",
        "- 唯一输入是 `artifacts/features/video_ml/condition_features.csv` 的 135 条 participant–Condition 数据；不读取窗口聚合、特征提取、VideoMAE2、深度模型、实时或 Unity 代码的产物。",
        "- 模态固定为 P=`eeg_`+`ecg_`、H=`head_`、E=`eye_`、V=`video_`；仅比较 15 个非空 P/H/E/V 组合，组合名不包含 Condition。问卷原始列、标签、标识、Condition、上下文及 QC 列均不进入模型。",
        "- 验证是 LOPO。每折先以训练参与者的同 Condition 标签均值建立 Condition-only 基线（未知 Condition 回退训练标签均值），随后对标签残差拟合 Ridge(alpha=10.0)。每个目标、模态、折内最多保留 20 个绝对 Pearson 相关最高的特征，随后中位数插补和标准化。",
        "- history 基线在留出参与者内按 `presentation_position` 排序：第一个 Condition 用训练标签均值，之后使用该参与者真实的前一个 Condition 标签。高 discomfort 的真实阈值为 `>=0.50`，回归报警阈值固定为 `>=0.20`；没有阈值搜索或调参。",
        f"- random_uniform 以配置随机种子独立生成 {random_simulations:,} 次、每目标每样本独立的 Uniform(0,1) 预测。以下所有结论仅是此数据集的 LOPO 离线融合比较，不是部署、实时接入、自动推荐或安全放行结论。",
        "", "## 三类基线", "", *baseline_rows, "", *random_rows, "",
        "## 15 个组合：按 Relaxation MAE（低者优先）", "", "| 排名 | 组合 | Relaxation MAE |", "|---:|---|---:|",
        *_combination_table(combination_metrics, key="relaxation_mae"), "",
        "## 15 个组合：按 Discomfort MAE（低者优先）", "", "| 排名 | 组合 | Discomfort MAE |", "|---:|---|---:|",
        *_combination_table(combination_metrics, key="discomfort_mae"), "",
        "## 15 个组合：按高 discomfort recall（高者优先）", "", "| 排名 | 组合 | 高 discomfort recall |", "|---:|---|---:|",
        *_combination_table(combination_metrics, key="discomfort_high_recall", descending=True, percent=True), "",
        "## 全部 28 条移除单模态消融", "",
        "正值始终表示完整组合更好：MAE 为“移除后 MAE − 完整组合 MAE”；Spearman、recall、precision 为“完整组合 − 移除后”；假阴性为“移除后 − 完整组合”。", "",
        *ablation_rows, "", "## 结论边界", "",
        "本报告最多只能描述此 135 条数据上的 LOPO 基准中哪些组合观察上更有价值。它不构成模型选择、部署、实时接入、自动 Condition 推荐或安全门通过的证据；现有运行时和 Shadow/hold 策略不因本基准而改变。",
    ]
    atomic_write_text(output, "\n".join(lines) + "\n")


def benchmark_minimal_fusion(
    config: ProjectConfig, *, random_simulations: int = RANDOM_SIMULATIONS
) -> dict[str, Any]:
    """Run and persist the fixed research-only minimal fusion benchmark."""
    pd, *_ = _dependencies()
    source = config.path("features") / SOURCE_RELATIVE_PATH
    if not source.exists():
        raise FileNotFoundError(f"Minimal fusion input not found: {source}")
    result = evaluate_minimal_fusion_frame(
        pd.read_csv(source), random_seed=int(config.get("modeling.random_seed")),
        random_simulations=random_simulations, expected_labels=EXPECTED_LABELS,
    )
    if config.is_legacy:
        output_directory = config.path("artifacts") / OUTPUT_DIRECTORY
        metrics_output = output_directory / METRICS_FILENAME
        oof_output = output_directory / OOF_FILENAME
        report_output: Path | None = config.path("reports") / REPORT_FILENAME
    else:
        # New runs expose machine artifacts in their normalized directories.
        # The sole human-readable artifact is produced by ``rtml report``.
        metrics_output = config.path("metrics") / "minimal_fusion_metrics.csv"
        oof_output = config.path("predictions") / "minimal_fusion_oof_predictions.csv"
        report_output = None
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    result["metrics"].to_csv(metrics_output, index=False)
    result["oof_predictions"].to_csv(oof_output, index=False)
    if report_output:
        _write_report(result, report_output, random_simulations=random_simulations)
    return {
        "research_only": True, "source": str(source), "n_labels": int(len(result["frame"])),
        "n_combinations": len(COMBINATIONS),
        "n_ablations": int(sum(len(name) for name in COMBINATIONS if len(name) > 1)),
        "random_simulations": int(random_simulations), "metrics": str(metrics_output),
        "oof_predictions": str(oof_output), "report": str(report_output) if report_output else None,
    }
