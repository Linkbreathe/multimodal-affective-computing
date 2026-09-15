from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OLD_DIR = ROOT / "artifacts" / "reports" / "physio_family_split"
NEW_DIR = ROOT / "artifacts" / "reports" / "physio_family_split_ecg_neurokit"
OUTPUT_DIR = ROOT / "artifacts" / "reports" / "ecg_neurokit_comparison"
REPORT = ROOT / "artifacts" / "reports" / "ecg_neurokit_comparison_2026-07-02_zh.md"


def _fmt(value: float) -> str:
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
                values.append(_fmt(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_split(path: Path, detector: str) -> dict[str, pd.DataFrame]:
    return {
        "detector": detector,
        "metrics": pd.read_csv(path / "metrics.csv"),
        "best_raw": pd.read_csv(path / "best_raw.csv"),
        "best_personalized": pd.read_csv(path / "best_personalized.csv"),
        "pass_counts": pd.read_csv(path / "pass_counts.csv"),
    }


def _single_family(metrics: pd.DataFrame, detector: str) -> pd.DataFrame:
    rows = metrics.loc[
        metrics["record_type"].eq("combination")
        & metrics["combination"].isin(["G", "R", "Q", "S", "A", "H", "E", "V"])
    ].copy()
    rows.insert(0, "detector", detector)
    return rows[[
        "detector",
        "cohort",
        "combination",
        "relaxation_mae",
        "relaxation_spearman",
        "discomfort_mae",
        "discomfort_spearman",
    ]]


def _best_summary(old: dict[str, pd.DataFrame], new: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for item in (old, new):
        best = item["best_raw"].copy()
        best.insert(0, "detector", item["detector"])
        rows.append(best)
    return pd.concat(rows, ignore_index=True)


def _personalized_summary(old: dict[str, pd.DataFrame], new: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for item in (old, new):
        best = item["best_personalized"].copy()
        best.insert(0, "detector", item["detector"])
        rows.append(best)
    return pd.concat(rows, ignore_index=True)


def _pass_summary(old: dict[str, pd.DataFrame], new: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for item in (old, new):
        table = item["pass_counts"].copy()
        table.insert(0, "detector", item["detector"])
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def _write_report(
    single: pd.DataFrame,
    best_raw: pd.DataFrame,
    best_personalized: pd.DataFrame,
    pass_counts: pd.DataFrame,
) -> None:
    new_single = single.loc[single["detector"].eq("neurokit")]
    old_single = single.loc[single["detector"].eq("legacy")]
    new_all = new_single.loc[new_single["cohort"].eq("all15")].set_index("combination")
    old_all = old_single.loc[old_single["cohort"].eq("all15")].set_index("combination")
    new_eeg = new_single.loc[new_single["cohort"].eq("eeg_available9")].set_index("combination")
    old_eeg = old_single.loc[old_single["cohort"].eq("eeg_available9")].set_index("combination")

    lines = [
        "# NeuroKit2 ECG 修复前后建模对照",
        "",
        "**生成日期:** 2026-07-02",
        "**对照输入:** legacy = `artifacts/features/video_ml/condition_features.csv`；"
        "neurokit = `artifacts/features/ecg_neurokit/condition_features.csv`。",
        "",
        "## 结论",
        "",
        "NeuroKit2 修复把 ECG 数值带回合理生理范围，并改变了 ECG 家族的模型排序。"
        "旧 detector 里 raw relaxation 最好是 `R`，修复后全体 15 人变成 `QH`；"
        "但这不是简单的“ECG 变强”，而是旧 ECG 速率特征里一部分异常/幅度信息被削弱，"
        "HRV 与 head 的组合相对更靠前。",
        "",
        f"- all15 单独 `R` relaxation MAE: legacy {_fmt(old_all.loc['R', 'relaxation_mae'])} -> "
        f"neurokit {_fmt(new_all.loc['R', 'relaxation_mae'])}。",
        f"- all15 单独 `Q` relaxation MAE: legacy {_fmt(old_all.loc['Q', 'relaxation_mae'])} -> "
        f"neurokit {_fmt(new_all.loc['Q', 'relaxation_mae'])}。",
        f"- eeg_available9 单独 `R` relaxation MAE: legacy {_fmt(old_eeg.loc['R', 'relaxation_mae'])} -> "
        f"neurokit {_fmt(new_eeg.loc['R', 'relaxation_mae'])}。",
        "- eeg_available9 raw discomfort 最好: legacy `GQH` = 0.0736；"
        "neurokit `H` = 0.0741。EEG 相关组合仍有 Spearman 优势，但 MAE 最低不再依赖 ECG。",
        "- neurokit N=2 all15 discomfort 出现 4 个组合低于 history；legacy 为 0。"
        "这些组合是 `GQH/GQHV/GRQH/QHV`，说明修复后的 HRV 可能对不适有一点增量。",
        "",
        "## 单家族对照",
        "",
        _table(single, [
            "detector",
            "cohort",
            "combination",
            "relaxation_mae",
            "relaxation_spearman",
            "discomfort_mae",
            "discomfort_spearman",
        ]),
        "",
        "## Raw 最佳组合",
        "",
        _table(best_raw, ["detector", "cohort", "criterion", "combination", "value"]),
        "",
        "## Personalized 最佳组合",
        "",
        _table(best_personalized, ["detector", "calibration_conditions", "cohort", "criterion", "combination", "value"]),
        "",
        "## 通过基线计数",
        "",
        _table(pass_counts, ["detector", "cohort", "calibration_conditions", "criterion", "n_pass", "combinations"]),
        "",
        "## 判断",
        "",
        "ECG 现在可以继续作为研究候选，但还不能直接进入 real-time 决策。"
        "更稳的下一步是只保留 neurokit ECG，移除 legacy/audit-only ECG 证据，并做一个更窄的候选："
        "`condition/person calibration + R/Q/H/G`，在 EEG-available cohort 和 all15 上分别报告。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old = _load_split(OLD_DIR, "legacy")
    new = _load_split(NEW_DIR, "neurokit")
    single = pd.concat([
        _single_family(old["metrics"], "legacy"),
        _single_family(new["metrics"], "neurokit"),
    ], ignore_index=True)
    best_raw = _best_summary(old, new)
    best_personalized = _personalized_summary(old, new)
    pass_counts = _pass_summary(old, new)

    single.to_csv(OUTPUT_DIR / "single_family_comparison.csv", index=False)
    best_raw.to_csv(OUTPUT_DIR / "best_raw_comparison.csv", index=False)
    best_personalized.to_csv(OUTPUT_DIR / "best_personalized_comparison.csv", index=False)
    pass_counts.to_csv(OUTPUT_DIR / "pass_counts_comparison.csv", index=False)
    _write_report(single, best_raw, best_personalized, pass_counts)
    print(REPORT)


if __name__ == "__main__":
    main()
