from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "artifacts" / "fusion_minimal" / "oof_predictions.csv"
OUTPUT_DIR = ROOT / "artifacts" / "reports" / "modality_personalization_followup"
REPORT = ROOT / "artifacts" / "reports" / "modality_personalization_followup_2026-07-02_zh.md"

TARGETS = ("relaxation", "discomfort")
CALIBRATION_COUNTS = (2, 3)
HIGH_DISCOMFORT_TRUTH_THRESHOLD = 0.50
HIGH_DISCOMFORT_PREDICTION_THRESHOLD = 0.20
PREDICTORS = (
    "model_raw",
    "model_personalized",
    "condition_only",
    "condition_only_personalized",
    "history",
)
SINGLE_COMBINATIONS = ("P", "H", "E", "V")


def _fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


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


def _wide_metrics(frame: pd.DataFrame, predictor: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for target in TARGETS:
        rows = frame.loc[frame["target"].eq(target)]
        prediction = rows[f"{predictor}_prediction"].to_numpy(dtype=float)
        truth = rows["truth"].to_numpy(dtype=float)
        for metric, value in _point_metrics(truth, prediction, target).items():
            output[f"{target}_{metric}"] = value
    return output


def _read_oof() -> pd.DataFrame:
    if not OOF.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {OOF}")
    frame = pd.read_csv(OOF)
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
        raise ValueError(f"OOF predictions missing columns: {missing}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    frame["target"] = frame["target"].astype(str)
    frame["presentation_position"] = pd.to_numeric(frame["presentation_position"], errors="raise")
    for column in ("truth", "prediction", "condition_only_prediction", "history_prediction"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _calibrated_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for calibration_count in CALIBRATION_COUNTS:
        for (combination, participant, target), group in frame.groupby(
            ["combination", "participant_id", "target"], sort=True
        ):
            ordered = group.sort_values("presentation_position", kind="stable")
            calibration = ordered.iloc[:calibration_count]
            evaluation = ordered.iloc[calibration_count:]
            if evaluation.empty:
                continue
            model_bias = float(np.mean(calibration["truth"] - calibration["prediction"]))
            condition_bias = float(
                np.mean(calibration["truth"] - calibration["condition_only_prediction"])
            )
            for _, row in evaluation.iterrows():
                model_personalized = float(np.clip(row["prediction"] + model_bias, 0.0, 1.0))
                condition_personalized = float(
                    np.clip(row["condition_only_prediction"] + condition_bias, 0.0, 1.0)
                )
                output.append({
                    "calibration_conditions": int(calibration_count),
                    "combination": combination,
                    "participant_id": participant,
                    "condition": row["condition"],
                    "presentation_position": int(row["presentation_position"]),
                    "target": target,
                    "truth": float(row["truth"]),
                    "model_raw_prediction": float(row["prediction"]),
                    "model_personalized_prediction": model_personalized,
                    "condition_only_prediction": float(row["condition_only_prediction"]),
                    "condition_only_personalized_prediction": condition_personalized,
                    "history_prediction": float(row["history_prediction"]),
                    "model_bias": model_bias,
                    "condition_bias": condition_bias,
                })
    result = pd.DataFrame(output)
    for predictor in PREDICTORS:
        result[f"{predictor}_absolute_error"] = (
            result["truth"] - result[f"{predictor}_prediction"]
        ).abs()
    return result


def _metrics(calibrated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (calibration_count, combination), group in calibrated.groupby(
        ["calibration_conditions", "combination"], sort=True
    ):
        for predictor in PREDICTORS:
            rows.append({
                "calibration_conditions": int(calibration_count),
                "combination": combination,
                "predictor": predictor,
                "n_predictions": int(len(group.loc[group["target"].eq("relaxation")])),
                **_wide_metrics(group, predictor),
            })
    return pd.DataFrame(rows)


def _summary_tables(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    model = metrics.loc[metrics["predictor"].eq("model_personalized")].copy()
    raw = metrics.loc[metrics["predictor"].eq("model_raw")].copy()
    raw = raw.rename(columns={
        "relaxation_mae": "raw_relaxation_mae",
        "discomfort_mae": "raw_discomfort_mae",
    })[["calibration_conditions", "combination", "raw_relaxation_mae", "raw_discomfort_mae"]]
    model = model.merge(raw, on=["calibration_conditions", "combination"], how="left")
    model["relaxation_mae_delta_vs_raw"] = model["raw_relaxation_mae"] - model["relaxation_mae"]
    model["discomfort_mae_delta_vs_raw"] = model["raw_discomfort_mae"] - model["discomfort_mae"]

    best_rows = []
    for calibration_count, group in model.groupby("calibration_conditions", sort=True):
        for metric, ascending in [
            ("relaxation_mae", True),
            ("discomfort_mae", True),
            ("relaxation_spearman", False),
            ("discomfort_spearman", False),
            ("discomfort_high_precision", False),
            ("discomfort_high_recall", False),
        ]:
            row = group.sort_values(metric, ascending=ascending).iloc[0]
            best_rows.append({
                "calibration_conditions": int(calibration_count),
                "criterion": metric,
                "combination": row["combination"],
                "value": row[metric],
            })
    best = pd.DataFrame(best_rows)

    baseline_rows = []
    for calibration_count, group in metrics.groupby("calibration_conditions", sort=True):
        for predictor in ("condition_only", "condition_only_personalized", "history"):
            rows = group.loc[group["predictor"].eq(predictor)]
            if rows.empty:
                continue
            first = rows.iloc[0]
            baseline_rows.append({
                "calibration_conditions": int(calibration_count),
                "predictor": predictor,
                "relaxation_mae": first["relaxation_mae"],
                "relaxation_spearman": first["relaxation_spearman"],
                "discomfort_mae": first["discomfort_mae"],
                "discomfort_spearman": first["discomfort_spearman"],
                "discomfort_high_recall": first["discomfort_high_recall"],
                "discomfort_high_precision": first["discomfort_high_precision"],
                "discomfort_high_false_negatives": first["discomfort_high_false_negatives"],
            })
    baselines = pd.DataFrame(baseline_rows)

    pass_rows = []
    for calibration_count, group in model.groupby("calibration_conditions", sort=True):
        baseline_group = baselines.loc[baselines["calibration_conditions"].eq(calibration_count)]
        lookup = baseline_group.set_index("predictor")
        checks = [
            ("relaxation beats raw model", group["relaxation_mae"] < group["raw_relaxation_mae"]),
            (
                "relaxation beats condition-only",
                group["relaxation_mae"] < lookup.loc["condition_only", "relaxation_mae"],
            ),
            (
                "relaxation beats personalized condition-only",
                group["relaxation_mae"] < lookup.loc["condition_only_personalized", "relaxation_mae"],
            ),
            (
                "relaxation beats history",
                group["relaxation_mae"] < lookup.loc["history", "relaxation_mae"],
            ),
            ("discomfort beats raw model", group["discomfort_mae"] < group["raw_discomfort_mae"]),
            (
                "discomfort beats condition-only",
                group["discomfort_mae"] < lookup.loc["condition_only", "discomfort_mae"],
            ),
            (
                "discomfort beats personalized condition-only",
                group["discomfort_mae"] < lookup.loc["condition_only_personalized", "discomfort_mae"],
            ),
            (
                "discomfort beats history",
                group["discomfort_mae"] < lookup.loc["history", "discomfort_mae"],
            ),
        ]
        for criterion, mask in checks:
            passed = group.loc[mask, "combination"].tolist()
            pass_rows.append({
                "calibration_conditions": int(calibration_count),
                "criterion": criterion,
                "n_pass": len(passed),
                "combinations": ",".join(passed),
            })
    passes = pd.DataFrame(pass_rows)

    singles = model.loc[model["combination"].isin(SINGLE_COMBINATIONS)].copy()
    presence_rows = []
    for calibration_count, group in model.groupby("calibration_conditions", sort=True):
        for modality in SINGLE_COMBINATIONS:
            with_modality = group.loc[group["combination"].str.contains(modality, regex=False)]
            without_modality = group.loc[~group["combination"].str.contains(modality, regex=False)]
            presence_rows.append({
                "calibration_conditions": int(calibration_count),
                "modality": modality,
                "n_with": len(with_modality),
                "n_without": len(without_modality),
                "relaxation_mae_with_mean": with_modality["relaxation_mae"].mean(),
                "relaxation_mae_without_mean": without_modality["relaxation_mae"].mean(),
                "discomfort_mae_with_mean": with_modality["discomfort_mae"].mean(),
                "discomfort_mae_without_mean": without_modality["discomfort_mae"].mean(),
                "relaxation_spearman_with_mean": with_modality["relaxation_spearman"].mean(),
                "relaxation_spearman_without_mean": without_modality["relaxation_spearman"].mean(),
                "discomfort_spearman_with_mean": with_modality["discomfort_spearman"].mean(),
                "discomfort_spearman_without_mean": without_modality["discomfort_spearman"].mean(),
            })
    presence = pd.DataFrame(presence_rows)

    return {
        "model": model,
        "best": best,
        "baselines": baselines,
        "passes": passes,
        "singles": singles,
        "presence": presence,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
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
                values.append(_fmt(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(tables: dict[str, pd.DataFrame]) -> None:
    model = tables["model"]
    baselines = tables["baselines"]
    passes = tables["passes"]
    singles = tables["singles"]
    presence = tables["presence"]
    lines = [
        "# 个性化校准后的逐模态 LOPO 探索",
        "",
        "**生成日期:** 2026-07-02",
        "**输入:** `artifacts/fusion_minimal/oof_predictions.csv`。",
        "**方法:** 对每个留出被试、每个组合、每个目标，使用前 N 个 condition 的真实评分估计 `truth - prediction` bias，"
        "只在剩余 condition 上评估。N=2 时每人评估 7 个 condition，共 105 条；N=3 时每人评估 6 个 condition，共 90 条。",
        "",
        "## 结论",
        "",
    ]

    for calibration_count in CALIBRATION_COUNTS:
        group = model.loc[model["calibration_conditions"].eq(calibration_count)]
        base = baselines.loc[baselines["calibration_conditions"].eq(calibration_count)].set_index("predictor")
        best_relax = group.loc[group["relaxation_mae"].idxmin()]
        best_discomfort = group.loc[group["discomfort_mae"].idxmin()]
        best_discomfort_rho = group.loc[group["discomfort_spearman"].idxmax()]
        video_presence = presence.loc[
            presence["calibration_conditions"].eq(calibration_count)
            & presence["modality"].eq("V")
        ].iloc[0]
        pass_group = passes.loc[passes["calibration_conditions"].eq(calibration_count)]
        pass_lookup = pass_group.set_index("criterion")

        lines.extend([
            f"### N={calibration_count}",
            "",
            f"- relaxation 最低 MAE 是 `{best_relax['combination']}` = {_fmt(best_relax['relaxation_mae'])}；"
            f"同一评估子集的 condition-only = {_fmt(base.loc['condition_only', 'relaxation_mae'])}，"
            f"personalized condition-only = {_fmt(base.loc['condition_only_personalized', 'relaxation_mae'])}，"
            f"history = {_fmt(base.loc['history', 'relaxation_mae'])}。",
            f"- discomfort 最低 MAE 是 `{best_discomfort['combination']}` = {_fmt(best_discomfort['discomfort_mae'])}；"
            f"condition-only = {_fmt(base.loc['condition_only', 'discomfort_mae'])}，"
            f"personalized condition-only = {_fmt(base.loc['condition_only_personalized', 'discomfort_mae'])}，"
            f"history = {_fmt(base.loc['history', 'discomfort_mae'])}。",
            f"- discomfort 最高 Spearman 是 `{best_discomfort_rho['combination']}` = "
            f"{_fmt(best_discomfort_rho['discomfort_spearman'])}。",
            f"- 含 `V` 组合平均 relaxation MAE = {_fmt(video_presence['relaxation_mae_with_mean'])}，"
            f"不含 `V` = {_fmt(video_presence['relaxation_mae_without_mean'])}；"
            f"含 `V` 组合平均 discomfort MAE = {_fmt(video_presence['discomfort_mae_with_mean'])}，"
            f"不含 `V` = {_fmt(video_presence['discomfort_mae_without_mean'])}。",
            f"- 低于 personalized condition-only 的组合数：relaxation "
            f"{int(pass_lookup.loc['relaxation beats personalized condition-only', 'n_pass'])}/15，"
            f"discomfort {int(pass_lookup.loc['discomfort beats personalized condition-only', 'n_pass'])}/15。",
            "",
        ])

    single_columns = [
        "calibration_conditions",
        "combination",
        "relaxation_mae",
        "relaxation_mae_delta_vs_raw",
        "relaxation_spearman",
        "discomfort_mae",
        "discomfort_mae_delta_vs_raw",
        "discomfort_spearman",
    ]
    lines.extend([
        "## 单模态个性化结果",
        "",
        _markdown_table(singles[single_columns], single_columns),
        "",
        "## 基线通过情况",
        "",
        _markdown_table(passes, ["calibration_conditions", "criterion", "n_pass", "combinations"]),
        "",
        "## 解读",
        "",
        "个性化校准确实改变了问题：它把每个被试的整体评分偏移先扣掉，因此比纯 LOPO 更接近未来在线使用。"
        "但这轮结果也说明，简单 bias 校准本身已经很强，模态特征必须超过 personalized condition-only 或 history，"
        "才说明传感器真正提供了增量信息。",
        "",
        "当前最值得继续看的不是 raw video，而是两个方向：一是把 video 拆成刺激复刻成分和用户残差成分；"
        "二是修 ECG 后把 P 拆成 EEG-only、HR-only、HRV-only。"
        "如果这些拆分仍不能超过 personalized condition-only，real-time 阶段应优先做安全监测和个体内策略，而不是部署跨人状态预测器。",
        "",
        "本报告仍是离线研究比较，不改变 `Shadow/hold` 和 real-time 部署状态。",
        "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof = _read_oof()
    calibrated = _calibrated_rows(oof)
    metrics = _metrics(calibrated)
    tables = _summary_tables(metrics)

    calibrated.to_csv(OUTPUT_DIR / "personalized_oof_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "personalized_metrics.csv", index=False)
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    _write_report(tables)
    print(REPORT)


if __name__ == "__main__":
    main()
