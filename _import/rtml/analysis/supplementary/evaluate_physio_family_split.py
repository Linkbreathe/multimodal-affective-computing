from __future__ import annotations

from itertools import combinations
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "artifacts" / "features" / "video_ml" / "condition_features.csv"

TARGETS = ("relaxation", "discomfort")
GROUP_ORDER = ("G", "R", "Q", "H", "E", "V")
SENSITIVITY_COMBINATIONS = ("S", "A", "GRQS", "GRQSA")
RIDGE_ALPHA = 10.0
FEATURES_PER_GROUP = 20
CALIBRATION_COUNTS = (2, 3)
HIGH_DISCOMFORT_TRUTH_THRESHOLD = 0.50
HIGH_DISCOMFORT_PREDICTION_THRESHOLD = 0.20

GROUP_LABELS = {
    "G": "EEG",
    "R": "ECG-rate(HR/RR/peaks)",
    "Q": "ECG-HRV",
    "S": "ECG-signal-IQR",
    "A": "ECG-audit-rr-std",
    "H": "head",
    "E": "eye",
    "V": "video",
}

STATIC_COLUMNS = {
    "participant_id",
    "condition",
    "presentation_position",
    "label_source_row",
    "intensity",
    "frequency",
    "intensity_index",
    "frequency_index",
    "condition_index",
    "window_count",
    "window_count_expected",
    "relaxation",
    "discomfort",
    "relaxation_raw",
    "discomfort_raw",
    "pleasantness",
    "pleasantness_raw",
    "arousal_raw",
    "monotony",
    "monotony_raw",
    "visual_fit",
    "calm",
}


def _main_combinations() -> tuple[str, ...]:
    values = [
        "".join(parts)
        for size in range(1, len(GROUP_ORDER) + 1)
        for parts in combinations(GROUP_ORDER, size)
    ]
    return tuple(values + list(SENSITIVITY_COMBINATIONS))


COMBINATIONS = _main_combinations()


def _safe_spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    value = float(spearmanr(truth, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def _point_metrics(truth: np.ndarray, prediction: np.ndarray, target: str) -> dict[str, float]:
    metrics = {
        "mae": float(np.mean(np.abs(truth - prediction))),
        "spearman": _safe_spearman(truth, prediction),
    }
    if target == "discomfort":
        high_truth = truth >= HIGH_DISCOMFORT_TRUTH_THRESHOLD
        high_prediction = prediction >= HIGH_DISCOMFORT_PREDICTION_THRESHOLD
        metrics.update({
            "high_recall": float(np.mean(high_prediction[high_truth])) if high_truth.any() else float("nan"),
            "high_precision": float(np.mean(high_truth[high_prediction])) if high_prediction.any() else 0.0,
            "high_false_negatives": float(np.sum(high_truth & ~high_prediction)),
        })
    return metrics


def _wide_metrics(frame: pd.DataFrame, predictor_column: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for target in TARGETS:
        rows = frame.loc[frame["target"].eq(target)]
        truth = rows["truth"].to_numpy(dtype=float)
        prediction = rows[predictor_column].to_numpy(dtype=float)
        for metric, value in _point_metrics(truth, prediction, target).items():
            output[f"{target}_{metric}"] = value
    return output


def _condition_baseline(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    fallback = float(train[target].mean())
    by_condition = train.groupby("condition", sort=True)[target].mean().astype(float).to_dict()
    return np.asarray(
        [float(by_condition.get(condition, fallback)) for condition in test["condition"]],
        dtype=float,
    )


def _history_baseline(test: pd.DataFrame, fallback: float, target: str) -> np.ndarray:
    output = np.full(len(test), fallback, dtype=float)
    for _, group in test.groupby("participant_id", sort=False):
        previous: float | None = None
        for index in group.sort_values("presentation_position", kind="stable").index:
            position = test.index.get_loc(index)
            output[position] = fallback if previous is None else previous
            previous = float(test.loc[index, target])
    return output


def _group_columns(frame: pd.DataFrame, group: str) -> list[str]:
    candidates: list[str] = []
    if group == "G":
        candidates = [column for column in frame.columns if column.startswith("eeg_")]
    elif group == "R":
        roots = ("ecg_peak_count", "ecg_hr_bpm", "ecg_rr_mean_ms", "ecg_rr_median_ms")
        candidates = [column for column in frame.columns if column.startswith(roots)]
    elif group == "Q":
        candidates = [column for column in frame.columns if column.startswith("ecg_hrv_")]
    elif group == "S":
        candidates = [column for column in frame.columns if column.startswith("ecg_signal_iqr_uV")]
    elif group == "A":
        candidates = [column for column in frame.columns if column.startswith("ecg_rr_std_ms_audit_only")]
    elif group == "H":
        candidates = [column for column in frame.columns if column.startswith("head_")]
    elif group == "E":
        candidates = [column for column in frame.columns if column.startswith("eye_")]
    elif group == "V":
        candidates = [column for column in frame.columns if column.startswith("video_")]
    else:
        raise ValueError(group)
    return sorted(column for column in candidates if column not in STATIC_COLUMNS)


def _rank_features(
    train: pd.DataFrame,
    columns: list[str],
    residual: np.ndarray,
    limit: int = FEATURES_PER_GROUP,
) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for column in columns:
        values = pd.to_numeric(train[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(residual)
        if valid.sum() < 2:
            continue
        feature_values = values[valid]
        residual_values = residual[valid]
        if np.nanstd(feature_values) <= 0.0 or np.nanstd(residual_values) <= 0.0:
            continue
        correlation = float(np.corrcoef(feature_values, residual_values)[0, 1])
        if np.isfinite(correlation):
            ranked.append((abs(correlation), column))
    return [column for _, column in sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]]


def _matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def _ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def _selected_by_group(
    train: pd.DataFrame,
    group_columns: dict[str, list[str]],
    target: str,
) -> dict[str, list[str]]:
    train_baseline = _condition_baseline(train, train, target)
    residual = train[target].to_numpy(dtype=float) - train_baseline
    return {
        group: _rank_features(train, columns, residual)
        for group, columns in group_columns.items()
    }


def _selected_columns(
    selected_by_group: dict[str, list[str]],
    combination: str,
) -> tuple[list[str], dict[str, int]]:
    columns: list[str] = []
    counts: dict[str, int] = {}
    for group in GROUP_LABELS:
        retained = selected_by_group.get(group, []) if group in combination else []
        columns.extend(retained)
        counts[group] = len(retained)
    return sorted(set(columns)), counts


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_id", "condition", "presentation_position", *TARGETS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    for column in ("presentation_position", *TARGETS):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[["participant_id", "condition"]].duplicated().any():
        raise ValueError("Expected one row per participant-condition")
    return frame.reset_index(drop=True)


def _evaluate_cohort(frame: pd.DataFrame, cohort: str) -> dict[str, Any]:
    frame = _validate_frame(frame)
    group_columns = {group: _group_columns(frame, group) for group in GROUP_LABELS}
    participants = sorted(frame["participant_id"].unique())
    predictions = {
        combination: {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
        for combination in COMBINATIONS
    }
    condition_only = {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
    history = {target: np.full(len(frame), np.nan, dtype=float) for target in TARGETS}
    selection_rows: list[dict[str, Any]] = []

    for participant in participants:
        test_mask = frame["participant_id"].eq(participant).to_numpy()
        test_indexes = np.flatnonzero(test_mask)
        train = frame.loc[~test_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)
        for target in TARGETS:
            baseline = _condition_baseline(train, test, target)
            condition_only[target][test_indexes] = baseline
            history[target][test_indexes] = _history_baseline(test, float(train[target].mean()), target)
            selected = _selected_by_group(train, group_columns, target)
            train_baseline = _condition_baseline(train, train, target)
            residual = train[target].to_numpy(dtype=float) - train_baseline
            for combination in COMBINATIONS:
                columns, counts = _selected_columns(selected, combination)
                if not columns:
                    continue
                model = _ridge_pipeline()
                model.fit(_matrix(train, columns), residual)
                predictions[combination][target][test_indexes] = np.clip(
                    baseline + model.predict(_matrix(test, columns)),
                    0.0,
                    1.0,
                )
                selection_rows.append({
                    "cohort": cohort,
                    "participant_id": participant,
                    "target": target,
                    "combination": combination,
                    "n_features": len(columns),
                    **{f"selected_{group}": counts.get(group, 0) for group in GROUP_LABELS},
                })

    oof_rows: list[dict[str, Any]] = []
    for combination in COMBINATIONS:
        for row_index, source in frame.iterrows():
            for target in TARGETS:
                prediction = predictions[combination][target][row_index]
                if not np.isfinite(prediction):
                    continue
                truth = float(source[target])
                oof_rows.append({
                    "cohort": cohort,
                    "combination": combination,
                    "participant_id": source["participant_id"],
                    "condition": source["condition"],
                    "presentation_position": int(source["presentation_position"]),
                    "target": target,
                    "truth": truth,
                    "prediction": float(prediction),
                    "condition_only_prediction": float(condition_only[target][row_index]),
                    "history_prediction": float(history[target][row_index]),
                })
    oof = pd.DataFrame(oof_rows)
    metrics = _metrics_from_oof(oof)
    return {
        "cohort": cohort,
        "frame": frame,
        "group_columns": group_columns,
        "oof": oof,
        "metrics": metrics,
        "selection": pd.DataFrame(selection_rows),
    }


def _metrics_from_oof(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_seen: set[tuple[str, str]] = set()
    for (cohort, combination), group in oof.groupby(["cohort", "combination"], sort=True):
        rows.append({
            "cohort": cohort,
            "record_type": "combination",
            "combination": combination,
            "predictor": "model_raw",
            "n_predictions": int(len(group.loc[group["target"].eq("relaxation")])),
            **_wide_metrics(group, "prediction"),
        })
        for predictor, column in [
            ("condition_only", "condition_only_prediction"),
            ("history", "history_prediction"),
        ]:
            key = (cohort, predictor)
            if key in baseline_seen:
                continue
            baseline_seen.add(key)
            rows.append({
                "cohort": cohort,
                "record_type": "baseline",
                "combination": "",
                "predictor": predictor,
                "n_predictions": int(len(group.loc[group["target"].eq("relaxation")])),
                **_wide_metrics(group, column),
            })
    return pd.DataFrame(rows)


def _personalized_from_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibrated_rows: list[dict[str, Any]] = []
    for count in CALIBRATION_COUNTS:
        for (cohort, combination, participant, target), group in oof.groupby(
            ["cohort", "combination", "participant_id", "target"],
            sort=True,
        ):
            ordered = group.sort_values("presentation_position", kind="stable")
            calibration = ordered.iloc[:count]
            evaluation = ordered.iloc[count:]
            if evaluation.empty:
                continue
            model_bias = float(np.mean(calibration["truth"] - calibration["prediction"]))
            condition_bias = float(np.mean(calibration["truth"] - calibration["condition_only_prediction"]))
            for _, row in evaluation.iterrows():
                calibrated_rows.append({
                    "calibration_conditions": int(count),
                    "cohort": cohort,
                    "combination": combination,
                    "participant_id": participant,
                    "condition": row["condition"],
                    "presentation_position": int(row["presentation_position"]),
                    "target": target,
                    "truth": float(row["truth"]),
                    "model_raw_prediction": float(row["prediction"]),
                    "model_personalized_prediction": float(np.clip(row["prediction"] + model_bias, 0.0, 1.0)),
                    "condition_only_prediction": float(row["condition_only_prediction"]),
                    "condition_only_personalized_prediction": float(
                        np.clip(row["condition_only_prediction"] + condition_bias, 0.0, 1.0)
                    ),
                    "history_prediction": float(row["history_prediction"]),
                })
    calibrated = pd.DataFrame(calibrated_rows)
    metrics_rows: list[dict[str, Any]] = []
    for (count, cohort, combination), group in calibrated.groupby(
        ["calibration_conditions", "cohort", "combination"],
        sort=True,
    ):
        for predictor in [
            "model_raw",
            "model_personalized",
            "condition_only",
            "condition_only_personalized",
            "history",
        ]:
            metrics_rows.append({
                "calibration_conditions": int(count),
                "cohort": cohort,
                "combination": combination,
                "predictor": predictor,
                "n_predictions": int(len(group.loc[group["target"].eq("relaxation")])),
                **_wide_metrics(group, f"{predictor}_prediction"),
            })
    return calibrated, pd.DataFrame(metrics_rows)


def _best_rows(metrics: pd.DataFrame, predictor: str = "model_raw") -> pd.DataFrame:
    rows = []
    source = metrics.loc[
        metrics["record_type"].eq("combination") & metrics["predictor"].eq(predictor)
    ]
    for cohort, group in source.groupby("cohort", sort=True):
        for metric, ascending in [
            ("relaxation_mae", True),
            ("discomfort_mae", True),
            ("relaxation_spearman", False),
            ("discomfort_spearman", False),
            ("discomfort_high_precision", False),
            ("discomfort_high_recall", False),
        ]:
            row = group.sort_values(metric, ascending=ascending).iloc[0]
            rows.append({
                "cohort": cohort,
                "criterion": metric,
                "combination": row["combination"],
                "value": row[metric],
            })
    return pd.DataFrame(rows)


def _personalized_best_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = metrics.loc[metrics["predictor"].eq("model_personalized")]
    for (count, cohort), group in source.groupby(["calibration_conditions", "cohort"], sort=True):
        for metric, ascending in [
            ("relaxation_mae", True),
            ("discomfort_mae", True),
            ("relaxation_spearman", False),
            ("discomfort_spearman", False),
        ]:
            row = group.sort_values(metric, ascending=ascending).iloc[0]
            rows.append({
                "calibration_conditions": int(count),
                "cohort": cohort,
                "criterion": metric,
                "combination": row["combination"],
                "value": row[metric],
            })
    return pd.DataFrame(rows)


def _pass_counts(raw_metrics: pd.DataFrame, personalized_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    combo_metrics = raw_metrics.loc[raw_metrics["record_type"].eq("combination")]
    baselines = raw_metrics.loc[raw_metrics["record_type"].eq("baseline")]
    for cohort, group in combo_metrics.groupby("cohort", sort=True):
        base = baselines.loc[baselines["cohort"].eq(cohort)].set_index("predictor")
        checks = [
            ("raw relaxation beats condition-only", group["relaxation_mae"] < base.loc["condition_only", "relaxation_mae"]),
            ("raw relaxation beats history", group["relaxation_mae"] < base.loc["history", "relaxation_mae"]),
            ("raw discomfort beats condition-only", group["discomfort_mae"] < base.loc["condition_only", "discomfort_mae"]),
            ("raw discomfort beats history", group["discomfort_mae"] < base.loc["history", "discomfort_mae"]),
        ]
        for criterion, mask in checks:
            passed = group.loc[mask, "combination"].tolist()
            rows.append({
                "cohort": cohort,
                "calibration_conditions": 0,
                "criterion": criterion,
                "n_pass": len(passed),
                "combinations": ",".join(passed),
            })

    for (count, cohort), group in personalized_metrics.groupby(["calibration_conditions", "cohort"], sort=True):
        model = group.loc[group["predictor"].eq("model_personalized")]
        base = group.loc[group["predictor"].isin([
            "condition_only",
            "condition_only_personalized",
            "history",
        ])]
        base = base.drop_duplicates(["predictor"]).set_index("predictor")
        checks = [
            (
                "personalized relaxation beats condition-only",
                model["relaxation_mae"] < base.loc["condition_only", "relaxation_mae"],
            ),
            (
                "personalized relaxation beats personalized condition-only",
                model["relaxation_mae"] < base.loc["condition_only_personalized", "relaxation_mae"],
            ),
            (
                "personalized relaxation beats history",
                model["relaxation_mae"] < base.loc["history", "relaxation_mae"],
            ),
            (
                "personalized discomfort beats condition-only",
                model["discomfort_mae"] < base.loc["condition_only", "discomfort_mae"],
            ),
            (
                "personalized discomfort beats personalized condition-only",
                model["discomfort_mae"] < base.loc["condition_only_personalized", "discomfort_mae"],
            ),
            (
                "personalized discomfort beats history",
                model["discomfort_mae"] < base.loc["history", "discomfort_mae"],
            ),
        ]
        for criterion, mask in checks:
            passed = model.loc[mask, "combination"].tolist()
            rows.append({
                "cohort": cohort,
                "calibration_conditions": int(count),
                "criterion": criterion,
                "n_pass": len(passed),
                "combinations": ",".join(passed),
            })
    return pd.DataFrame(rows)


def _format(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
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
            elif column.startswith("n_") or column.endswith("conditions"):
                values.append(str(int(value)))
            else:
                values.append(_format(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    raw_metrics: pd.DataFrame,
    raw_best: pd.DataFrame,
    personalized_metrics: pd.DataFrame,
    personalized_best: pd.DataFrame,
    pass_counts: pd.DataFrame,
    cohort_info: pd.DataFrame,
    column_counts: pd.DataFrame,
    report: Path,
    source: Path,
) -> None:
    combo_metrics = raw_metrics.loc[raw_metrics["record_type"].eq("combination")]
    baselines = raw_metrics.loc[raw_metrics["record_type"].eq("baseline")]
    singles = combo_metrics.loc[combo_metrics["combination"].isin(["G", "R", "Q", "S", "A", "H", "E", "V"])]
    single_columns = [
        "cohort",
        "combination",
        "relaxation_mae",
        "relaxation_spearman",
        "discomfort_mae",
        "discomfort_spearman",
    ]

    lines = [
        "# Physio 家族拆分 LOPO 探索",
        "",
        "**生成日期:** 2026-07-02",
        f"**输入:** `{source.relative_to(ROOT).as_posix()}`。",
        "**协议:** 仍是 participant-condition LOPO；每折先建 condition-only baseline，再用 Ridge 预测残差。"
        "每个家族每折每目标最多保留 20 个与训练残差相关最高的特征。",
        "",
        "## 家族定义",
        "",
        "- `G` = EEG。",
        "- `R` = ECG rate/timing：`ecg_peak_count`、`ecg_hr_bpm`、`ecg_rr_mean_ms`、`ecg_rr_median_ms`。",
        "- `Q` = ECG HRV：`ecg_hrv_*`。",
        "- `S` = ECG signal IQR sensitivity。",
        "- `A` = audit-only `ecg_rr_std_ms_audit_only` sensitivity。",
        "- `H/E/V` = head / eye / video。",
        "",
        "## Cohort 与列数",
        "",
        _table(cohort_info, ["cohort", "n_participants", "n_labels", "participants"]),
        "",
        _table(column_counts, ["group", "label", "n_columns"]),
        "",
        "## Raw LOPO 单家族结果",
        "",
        _table(singles[single_columns], single_columns),
        "",
        "## Raw LOPO 最佳组合",
        "",
        _table(raw_best, ["cohort", "criterion", "combination", "value"]),
        "",
        "## Baseline",
        "",
        _table(
            baselines[[
                "cohort",
                "predictor",
                "relaxation_mae",
                "relaxation_spearman",
                "discomfort_mae",
                "discomfort_spearman",
            ]],
            ["cohort", "predictor", "relaxation_mae", "relaxation_spearman", "discomfort_mae", "discomfort_spearman"],
        ),
        "",
        "## Personalized 最佳组合",
        "",
        _table(personalized_best, ["calibration_conditions", "cohort", "criterion", "combination", "value"]),
        "",
        "## 通过基线的组合数",
        "",
        _table(pass_counts, ["cohort", "calibration_conditions", "criterion", "n_pass", "combinations"]),
        "",
        "## 解读",
        "",
    ]

    for cohort in sorted(combo_metrics["cohort"].unique()):
        cohort_best = raw_best.loc[raw_best["cohort"].eq(cohort)].set_index("criterion")
        base = baselines.loc[baselines["cohort"].eq(cohort)].set_index("predictor")
        lines.extend([
            f"### {cohort}",
            "",
            f"- raw relaxation 最低 MAE: `{cohort_best.loc['relaxation_mae', 'combination']}` = "
            f"{_format(cohort_best.loc['relaxation_mae', 'value'])}；condition-only = "
            f"{_format(base.loc['condition_only', 'relaxation_mae'])}。",
            f"- raw discomfort 最低 MAE: `{cohort_best.loc['discomfort_mae', 'combination']}` = "
            f"{_format(cohort_best.loc['discomfort_mae', 'value'])}；history = "
            f"{_format(base.loc['history', 'discomfort_mae'])}。",
        ])
        for count in CALIBRATION_COUNTS:
            subset = personalized_best.loc[
                personalized_best["cohort"].eq(cohort)
                & personalized_best["calibration_conditions"].eq(count)
            ].set_index("criterion")
            lines.extend([
                f"- personalized N={count} relaxation 最低 MAE: "
                f"`{subset.loc['relaxation_mae', 'combination']}` = "
                f"{_format(subset.loc['relaxation_mae', 'value'])}；"
                f"discomfort 最低 MAE: `{subset.loc['discomfort_mae', 'combination']}` = "
                f"{_format(subset.loc['discomfort_mae', 'value'])}。",
            ])
        lines.append("")

    lines.extend([
        "这一步的判断边界：`S` 与 `A` 是敏感性检查，不应被写成可部署生理证据。"
        "`A` 来自 audit-only RR std；如果它排名靠前，只能说明旧 ECG 检测器/派生特征需要修，"
        "不能说明 HRV 已经成为可靠实时输入。",
        "",
        "本报告仍是离线研究比较，不改变 `Shadow/hold` 和 real-time 部署状态。",
        "",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    source = Path(os.environ.get("RTML_PHYSIO_SPLIT_SOURCE", str(DEFAULT_SOURCE))).resolve()
    run_id = os.environ.get("RTML_PHYSIO_SPLIT_RUN_ID", "physio_family_split")
    output_dir = ROOT / "artifacts" / "reports" / run_id
    report = ROOT / "artifacts" / "reports" / f"{run_id}_2026-07-02_zh.md"
    if not source.exists():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(source)
    full = _validate_frame(full)

    eeg_columns = [
        column for column in _group_columns(full, "G")
        if not column.endswith("__missing_ratio")
    ]
    eeg_participants = sorted(
        participant
        for participant, group in full.groupby("participant_id", sort=True)
        if group[eeg_columns].notna().any().any()
    )
    cohorts = {
        "all15": full,
        "eeg_available9": full.loc[full["participant_id"].isin(eeg_participants)].reset_index(drop=True),
    }

    cohort_info = pd.DataFrame([
        {
            "cohort": name,
            "n_participants": frame["participant_id"].nunique(),
            "n_labels": len(frame),
            "participants": ",".join(sorted(frame["participant_id"].unique())),
        }
        for name, frame in cohorts.items()
    ])
    column_counts = pd.DataFrame([
        {"group": group, "label": GROUP_LABELS[group], "n_columns": len(_group_columns(full, group))}
        for group in GROUP_LABELS
    ])

    results = [_evaluate_cohort(frame, name) for name, frame in cohorts.items()]
    raw_oof = pd.concat([result["oof"] for result in results], ignore_index=True)
    raw_metrics = pd.concat([result["metrics"] for result in results], ignore_index=True)
    selection = pd.concat([result["selection"] for result in results], ignore_index=True)
    personalized_oof, personalized_metrics = _personalized_from_oof(raw_oof)
    raw_best = _best_rows(raw_metrics)
    personalized_best = _personalized_best_rows(personalized_metrics)
    pass_counts = _pass_counts(raw_metrics, personalized_metrics)

    raw_oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    raw_metrics.to_csv(output_dir / "metrics.csv", index=False)
    selection.to_csv(output_dir / "selection_audit.csv", index=False)
    personalized_oof.to_csv(output_dir / "personalized_oof_predictions.csv", index=False)
    personalized_metrics.to_csv(output_dir / "personalized_metrics.csv", index=False)
    raw_best.to_csv(output_dir / "best_raw.csv", index=False)
    personalized_best.to_csv(output_dir / "best_personalized.csv", index=False)
    pass_counts.to_csv(output_dir / "pass_counts.csv", index=False)
    cohort_info.to_csv(output_dir / "cohort_info.csv", index=False)
    column_counts.to_csv(output_dir / "column_counts.csv", index=False)

    _write_report(
        raw_metrics,
        raw_best,
        personalized_metrics,
        personalized_best,
        pass_counts,
        cohort_info,
        column_counts,
        report,
        source,
    )
    print(report)


if __name__ == "__main__":
    main()
