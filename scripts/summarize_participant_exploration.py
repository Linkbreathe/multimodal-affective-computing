"""Render all participant results from frozen CSVs; no extraction, fitting or plotting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str, precision: int = 3) -> str:
    return f"{float(value):.{precision}f}" if value else "缺失"


def render(source: Path) -> str:
    records = read_rows(source / "participant_condition_features.csv")
    associations = read_rows(source / "individual_associations.csv")
    quality = read_rows(source / "feature_quality.csv")
    variables = read_rows(source / "variables.csv")
    features = [v for v in variables if v["modality"] != "Rating"]
    participants = sorted({r["participant_id"] for r in records})
    if len(participants) != 9 or len(records) != 81 or len(associations) != 198:
        raise ValueError("Expected the frozen nine-participant, nine-condition analysis")
    lookup = {(r["participant_id"], r["feature"], r["outcome"]): r for r in associations}
    lines = [
        "九名参与者的条件—特征—体验对应报告",
        "",
        "This analysis was conducted after the main prediction analyses and is post hoc and exploratory.",
        "本文件只重新组织既有 CSV，不重新提取、筛选特征、计算相关或训练模型。",
        "表中数值为原始尺度；相关沿用预定义分析尺度，每项至少六个有效配对条件。",
        "EEG 功率和 RMSSD 使用自然对数；Spearman 正值表示同向排序，负值表示反向排序。",
        "条件按 C1–C9 排列；实际呈现顺序另列。个人均值不是静息基线。",
        "全部十一项特征均展示，包括弱关联、零相关与不可估计结果。",
        "显示值通常保留三位小数；低速比例保留六位，避免掩盖接近1的微小变化。精确值见CSV。",
        "",
    ]
    for participant in participants:
        rows = sorted(
            (r for r in records if r["participant_id"] == participant),
            key=lambda r: int(r["condition"][1:]),
        )
        lines.extend([f"===== {participant} =====", ""])
        order = sorted(rows, key=lambda r: int(r["presentation_order"]))
        lines.append("实际顺序：" + " → ".join(r["condition"] for r in order))
        for outcome, label in [("relaxation", "放松"), ("discomfort", "不适")]:
            values = [float(r[f"{outcome}_raw"]) for r in rows]
            highest = "、".join(
                r["condition"] for r in rows if float(r[f"{outcome}_raw"]) == max(values)
            )
            lowest = "、".join(
                r["condition"] for r in rows if float(r[f"{outcome}_raw"]) == min(values)
            )
            lines.append(
                f"{label}：均值 {sum(values) / len(values):.3f}，"
                f"范围 {min(values):g}–{max(values):g}；最高 {highest}，最低 {lowest}。"
            )
            if len(set(values)) == 1:
                lines.append(f"{label}评分恒定，全部对应相关不可估计，不记作零。")
        lines.extend(["", "条件 | 实际顺序 | 强度 | 频率参数 | 放松 | 不适"])
        for row in rows:
            lines.append(
                " | ".join(
                    row[k]
                    for k in [
                        "condition",
                        "presentation_order",
                        "intensity",
                        "frequency",
                        "relaxation_raw",
                        "discomfort_raw",
                    ]
                )
            )
        lines.extend(["", "逐特征原值与对应关系（保留所有结果）"])
        for feature in features:
            key = feature["variable"]
            precision = 6 if feature["raw_unit"] == "proportion" else 3
            lines.extend(["", f"{feature['name_zh']}；原始单位：{feature['raw_unit']}"])
            lines.append(
                "；".join(
                    f"{row['condition']}={number(row[f'{key}_raw'], precision)}" for row in rows
                )
            )
            for outcome, label in [("relaxation", "放松"), ("discomfort", "不适")]:
                assoc = lookup[participant, key, outcome]
                if assoc["estimation_status"] == "estimable":
                    result = f"ρ={number(assoc['spearman_rho'])}；方向={assoc['direction']}"
                else:
                    result = f"不可估计；原因={assoc['reason']}"
                lines.append(
                    f"与{label}：{result}；配对条件数={assoc['n_valid_pairs']}；"
                    f"纳入条件={assoc['included_conditions']}。"
                )
        missing = [
            r
            for r in quality
            if r["participant_id"] == participant and r["availability_status"] != "available"
        ]
        lines.extend(["", "缺失与质量："])
        lines.extend(f"{r['condition']} / {r['feature']}：{r['reason']}" for r in missing)
        if not missing:
            lines.append("全部核心特征条件值达到本次可用规则；不等于测量含义完全验证。")
        lines.extend(
            [
                "",
                "解释边界：每人最多九条件，相关可能不稳定；不能排除条件、顺序或其他因素。",
                "眼动来源与头前向回退无法区分；低速比例不是注意力；光流混合场景和头动。",
                "频段不直接代表情绪；单次会话不证明跨天稳定、因果或预测/个人校准有效。",
                "",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
