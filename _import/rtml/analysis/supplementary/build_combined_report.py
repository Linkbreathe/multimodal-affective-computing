from __future__ import annotations

from pathlib import Path


REPORT_DIR = Path("artifacts/reports")
OUTPUT = REPORT_DIR / "supplementary_evidence_combined_zh.md"

CHAPTERS = [
    (
        "第一章 数据范围与总体结论",
        REPORT_DIR / "supplementary_evidence_zh.md",
        "保留综合报告中的数据边界、核心结论、解释约束和论文位置映射。",
    ),
    (
        "第二章 剂量-反应统计分析",
        REPORT_DIR / "dose_response_report_zh.md",
        "对应 Task 1；支持 3.7(a)、5.1.2-5.1.7 与 RQ1/H1。",
    ),
    (
        "第三章 模型与模态显著性检验",
        REPORT_DIR / "model_significance_report_zh.md",
        "对应 Task 2；支持 3.7(b) 与 5.2.9。",
    ),
    (
        "第四章 额外经典模型比较",
        REPORT_DIR / "additional_classical_models_report_zh.md",
        "对应 Task 3；支持 3.6.2 与 5.2.2。",
    ),
    (
        "第五章 回归指标与端到端延迟",
        REPORT_DIR / "regression_latency_report_zh.md",
        "对应 Task 5；支持 3.6.6 与 5.2.8。",
    ),
    (
        "第六章 Delta-Beta / CFC 特征检查",
        REPORT_DIR / "cfc_feature_check_report_zh.md",
        "对应 Task 6；支持 5.1.4 的边界说明。",
    ),
]


def shift_headings(text: str, levels: int = 1) -> str:
    output: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            hashes = len(line) - len(line.lstrip("#"))
            if hashes >= 1 and line[hashes : hashes + 1] == " ":
                line = "#" * (hashes + levels) + line[hashes:]
        output.append(line)
    return "\n".join(output).strip()


def strip_first_h1(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).lstrip()
    return text


def main() -> None:
    missing = [path for _, path, _ in CHAPTERS if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source report(s): {missing}")

    lines: list[str] = [
        "# 补充实验与统计分析完整合并报告",
        "",
        "生成日期：2026-07-01。",
        "",
        "本文件将补充统计、模型比较、额外经典模型、回归指标、延迟基准与 CFC 特征检查合并为一份章节化报告。所有分析仍遵守既有边界：监督单元为 participant x Condition，EEG 相关统计仅使用 EEG 可用参与者，10 秒窗口只作为特征切片；这些补充分析不改变当前 `classical`、Shadow-only、`hold` 的部署结论。",
        "",
        "## 目录",
        "",
    ]
    for title, _, note in CHAPTERS:
        anchor = title.replace(" ", "-").lower()
        lines.append(f"- [{title}](#{anchor})：{note}")

    for title, path, note in CHAPTERS:
        source = path.read_text(encoding="utf-8")
        body = shift_headings(strip_first_h1(source), levels=1)
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"章节说明：{note}",
                "",
                f"来源文件：`{path.as_posix()}`。",
                "",
                body,
            ]
        )

    lines.extend(
        [
            "",
            "## 附录：使用边界检查清单",
            "",
            "- 心脏信号统一表述为 ECG、heart rate 或 HRV。",
            "- EEG 表格必须标注 N=9 或 EEG-available subset。",
            "- 统计表同时保留未校正 p 值和校正 p 值；校正方法在对应章节说明。",
            "- `*_raw` 与 audit-only 字段不得作为模型特征使用。",
            "- 所有解释限于观察性关联、组间差异或离线模型比较。",
            "- Study 2 / adaptive-control 相关表述保持 demo-oriented、Shadow-only、hold，不写成正式受试者控制实验。",
            "- 当前补充分析不改变部署结论。",
        ]
    )

    OUTPUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
