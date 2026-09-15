from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "artifacts" / "fusion_minimal" / "metrics.csv"
OUTPUT_DIR = ROOT / "artifacts" / "reports" / "modality_lopo_followup"
REPORT = ROOT / "artifacts" / "reports" / "modality_lopo_followup_2026-07-02_zh.md"

TARGET_METRICS = [
    "relaxation_mae",
    "relaxation_spearman",
    "discomfort_mae",
    "discomfort_spearman",
    "discomfort_high_recall",
    "discomfort_high_precision",
    "discomfort_high_false_negatives",
]

MODALITY_NAMES = {
    "P": "physio(EEG+ECG)",
    "H": "head",
    "E": "eye",
    "V": "video",
}


def _fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _metric_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    lines = [header, sep]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                values.append(value)
            elif column.startswith("n_") or column == "rank":
                values.append(str(int(value)))
            else:
                values.append(_fmt(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _read() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not METRICS.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS}")
    metrics = pd.read_csv(METRICS)
    combinations = metrics.loc[metrics["record_type"].eq("combination")].copy()
    baselines = metrics.loc[metrics["record_type"].eq("baseline")].copy()
    if combinations.empty or baselines.empty:
        raise ValueError("metrics.csv must contain both combination and baseline rows")
    return baselines, combinations


def _single_modality_summary(combinations: pd.DataFrame) -> pd.DataFrame:
    keep = ["P", "H", "E", "V"]
    columns = ["combination", *TARGET_METRICS]
    return combinations.loc[combinations["combination"].isin(keep), columns].sort_values("combination")


def _rankings(combinations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, ascending in [
        ("relaxation_mae", True),
        ("discomfort_mae", True),
        ("relaxation_spearman", False),
        ("discomfort_spearman", False),
        ("discomfort_high_precision", False),
        ("discomfort_high_recall", False),
    ]:
        ordered = combinations.sort_values(metric, ascending=ascending)
        for rank, (_, row) in enumerate(ordered.head(5).iterrows(), start=1):
            rows.append({
                "metric": metric,
                "rank": rank,
                "combination": row["combination"],
                "value": row[metric],
            })
    return pd.DataFrame(rows)


def _baseline_pass_counts(combinations: pd.DataFrame) -> pd.DataFrame:
    checks = [
        ("relaxation_mae", "condition_only_relaxation_mae", "relaxation beats condition-only"),
        ("relaxation_mae", "history_relaxation_mae", "relaxation beats history"),
        ("discomfort_mae", "condition_only_discomfort_mae", "discomfort beats condition-only"),
        ("discomfort_mae", "history_discomfort_mae", "discomfort beats history"),
    ]
    rows = []
    for metric, baseline_metric, label in checks:
        passed = combinations.loc[combinations[metric] < combinations[baseline_metric], "combination"].tolist()
        rows.append({
            "criterion": label,
            "n_pass": len(passed),
            "combinations": ",".join(passed),
        })
    return pd.DataFrame(rows)


def _presence_summary(combinations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality, label in MODALITY_NAMES.items():
        with_modality = combinations.loc[combinations["combination"].str.contains(modality, regex=False)]
        without_modality = combinations.loc[~combinations["combination"].str.contains(modality, regex=False)]
        row = {
            "modality": modality,
            "label": label,
            "n_with": len(with_modality),
            "n_without": len(without_modality),
        }
        for metric in [
            "relaxation_mae",
            "discomfort_mae",
            "relaxation_spearman",
            "discomfort_spearman",
        ]:
            row[f"{metric}_with_mean"] = with_modality[metric].mean()
            row[f"{metric}_without_mean"] = without_modality[metric].mean()
            if metric.endswith("_mae"):
                row[f"{metric}_mean_improvement_when_present"] = (
                    without_modality[metric].mean() - with_modality[metric].mean()
                )
            else:
                row[f"{metric}_mean_improvement_when_present"] = (
                    with_modality[metric].mean() - without_modality[metric].mean()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _addition_summary(combinations: pd.DataFrame) -> pd.DataFrame:
    by_name = combinations.set_index("combination")
    rows = []
    for modality, label in MODALITY_NAMES.items():
        for base in by_name.index:
            if modality in base:
                continue
            if not base:
                continue
            candidate = "".join(item for item in MODALITY_NAMES if item in base or item == modality)
            if candidate not in by_name.index:
                continue
            before = by_name.loc[base]
            after = by_name.loc[candidate]
            rows.append({
                "added_modality": modality,
                "label": label,
                "base_combination": base,
                "with_modality": candidate,
                "relaxation_mae_improvement": before["relaxation_mae"] - after["relaxation_mae"],
                "discomfort_mae_improvement": before["discomfort_mae"] - after["discomfort_mae"],
                "relaxation_spearman_improvement": after["relaxation_spearman"] - before["relaxation_spearman"],
                "discomfort_spearman_improvement": after["discomfort_spearman"] - before["discomfort_spearman"],
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    numeric = [
        "relaxation_mae_improvement",
        "discomfort_mae_improvement",
        "relaxation_spearman_improvement",
        "discomfort_spearman_improvement",
    ]
    grouped = frame.groupby(["added_modality", "label"], as_index=False)[numeric].agg(["mean", "min", "max"])
    grouped.columns = [
        "_".join(item for item in column if item) if isinstance(column, tuple) else column
        for column in grouped.columns
    ]
    return grouped


def _write_report(
    baselines: pd.DataFrame,
    combinations: pd.DataFrame,
    singles: pd.DataFrame,
    rankings: pd.DataFrame,
    passes: pd.DataFrame,
    presence: pd.DataFrame,
    addition: pd.DataFrame,
) -> None:
    condition = baselines.loc[baselines["baseline"].eq("condition_only")].iloc[0]
    history = baselines.loc[baselines["baseline"].eq("history")].iloc[0]
    best_relax = combinations.loc[combinations["relaxation_mae"].idxmin()]
    best_discomfort = combinations.loc[combinations["discomfort_mae"].idxmin()]
    best_discomfort_rho = combinations.loc[combinations["discomfort_spearman"].idxmax()]
    full = combinations.loc[combinations["combination"].eq("PHEV")].iloc[0]
    head = combinations.loc[combinations["combination"].eq("H")].iloc[0]
    physio = combinations.loc[combinations["combination"].eq("P")].iloc[0]
    video_presence = presence.loc[presence["modality"].eq("V")].iloc[0]
    pass_rows = {row["criterion"]: row for _, row in passes.iterrows()}

    top_relax = combinations.sort_values("relaxation_mae").head(8)
    top_discomfort = combinations.sort_values("discomfort_mae").head(8)
    single_columns = ["combination", "relaxation_mae", "relaxation_spearman", "discomfort_mae", "discomfort_spearman"]

    lines = [
        "# 逐模态 LOPO 后续探索报告",
        "",
        "**生成日期:** 2026-07-02",
        "**输入:** `artifacts/features/video_ml/condition_features.csv`，135 条 participant-condition。",
        "**输出依据:** `artifacts/fusion_minimal/metrics.csv` 与 `oof_predictions.csv`。",
        "",
        "## 结论",
        "",
        "这轮最小 Ridge 融合基准没有支持“video 可以直接提升主观状态预测”的说法。"
        "Video 在信号分析里能稳定复刻强度/频率操纵，但在 LOPO 预测 relaxation/discomfort 时，"
        "单独 video 和含 video 组合都没有形成稳定优势。",
        "",
        f"- relaxation 最低 MAE 是 `{best_relax['combination']}` = {_fmt(best_relax['relaxation_mae'])}；"
        f"Condition-only baseline = {_fmt(condition['relaxation_mae'])}。只有 "
        f"{pass_rows['relaxation beats condition-only']['n_pass']} 个组合低于 condition-only："
        f"{pass_rows['relaxation beats condition-only']['combinations'] or 'none'}。",
        f"- discomfort 最低 MAE 是 `{best_discomfort['combination']}` = {_fmt(best_discomfort['discomfort_mae'])}；"
        f"history baseline = {_fmt(history['discomfort_mae'])}。"
        f"{pass_rows['discomfort beats history']['n_pass']} 个组合低于 history baseline。",
        f"- discomfort 最高 Spearman 是 `{best_discomfort_rho['combination']}` = "
        f"{_fmt(best_discomfort_rho['discomfort_spearman'])}，但它的 MAE = "
        f"{_fmt(best_discomfort_rho['discomfort_mae'])}，仍高于 history baseline。",
        f"- full `PHEV` relaxation MAE = {_fmt(full['relaxation_mae'])}，"
        f"discomfort MAE = {_fmt(full['discomfort_mae'])}；它不是任一目标的最佳组合。",
        "",
        "## 单模态结果",
        "",
        _metric_table(singles[single_columns], single_columns),
        "",
        "解释：`H` 在 relaxation 上略低于 condition-only，但幅度只有 "
        f"{_fmt(condition['relaxation_mae'] - head['relaxation_mae'])} MAE；"
        f"`P` 在 discomfort 上低于 condition-only { _fmt(condition['discomfort_mae'] - physio['discomfort_mae']) }，"
        "但仍明显输给 history baseline。`V` 单独并没有兑现信号分析里看到的强剂量信息。",
        "",
        "## 前 8 名组合",
        "",
        "### Relaxation MAE",
        "",
        _metric_table(top_relax[["combination", "relaxation_mae", "relaxation_spearman", "discomfort_mae"]], ["combination", "relaxation_mae", "relaxation_spearman", "discomfort_mae"]),
        "",
        "### Discomfort MAE",
        "",
        _metric_table(top_discomfort[["combination", "discomfort_mae", "discomfort_spearman", "relaxation_mae"]], ["combination", "discomfort_mae", "discomfort_spearman", "relaxation_mae"]),
        "",
        "## Video 的实际边际价值",
        "",
        f"含 `V` 组合的平均 relaxation MAE = {_fmt(video_presence['relaxation_mae_with_mean'])}；"
        f"不含 `V` 组合 = {_fmt(video_presence['relaxation_mae_without_mean'])}。"
        f"含 `V` 组合的平均 discomfort MAE = {_fmt(video_presence['discomfort_mae_with_mean'])}；"
        f"不含 `V` 组合 = {_fmt(video_presence['discomfort_mae_without_mean'])}。",
        "",
        "这说明 video 的强信号更像“条件/刺激复刻”，不是主观 relaxation/discomfort 的稳定跨人预测器。"
        "如果后续继续用 video，应把它作为刺激操纵校验、条件上下文或研究对照，而不是直接当成用户状态传感器。",
        "",
        "## 基线通过情况",
        "",
        _metric_table(passes, ["criterion", "n_pass", "combinations"]),
        "",
        "## 下一步",
        "",
        "1. 做 personalized N=2/N=3 的同一套 P/H/E/V 组合评估，确认个体校准是否比纯 LOPO 更能释放模态价值。",
        "2. 把 video 拆成两类：刺激复刻特征和用户行为残差特征；优先检验 residual-video，而不是原始 video。",
        "3. 修 ECG R-peak 后重算 HR/HRV，再把 `P` 拆成 EEG-only、HR-only、HRV-only，避免坏 HRV 污染 physio 结论。",
        "4. 排查 `eye_gaze_on_painting_fraction`，修好后重新跑 `E` 和 `EV/HEV/PHEV`。",
        "",
        "本报告仍是离线研究比较，不改变 `Shadow/hold` 和 real-time 部署状态。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    baselines, combinations = _read()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    singles = _single_modality_summary(combinations)
    rankings = _rankings(combinations)
    passes = _baseline_pass_counts(combinations)
    presence = _presence_summary(combinations)
    addition = _addition_summary(combinations)

    singles.to_csv(OUTPUT_DIR / "single_modality_summary.csv", index=False)
    rankings.to_csv(OUTPUT_DIR / "top_rankings.csv", index=False)
    passes.to_csv(OUTPUT_DIR / "baseline_pass_counts.csv", index=False)
    presence.to_csv(OUTPUT_DIR / "modality_presence_summary.csv", index=False)
    addition.to_csv(OUTPUT_DIR / "modality_addition_summary.csv", index=False)

    _write_report(baselines, combinations, singles, rankings, passes, presence, addition)
    print(REPORT)


if __name__ == "__main__":
    main()
