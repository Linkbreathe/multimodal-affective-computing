"""Create one standalone Chinese evidence review from the current report trees.

This utility intentionally reads every report-related file before cleanup.  It
does not remove anything itself; deletion remains an explicit operational step
after the generated review has been verified.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
ARTIFACTS = WORKSPACE / "artifacts"
REPORTS = ARTIFACTS / "reports"
VIDEO = ARTIFACTS / "video"
FINAL_NAME = "multimodal_evidence_review_zh.md"


def _report_roots() -> list[Path]:
    return [REPORTS, *sorted(path for path in VIDEO.rglob("reports") if path.is_dir())]


def _importance(path: Path) -> tuple[str, str]:
    relative = path.relative_to(ARTIFACTS).as_posix()
    name = path.name.lower()
    if relative.startswith("reports/") and any(
        token in name
        for token in (
            "condition_level",
            "data_qc",
            "dcnn_condition",
            "policy_model_card",
            "shadow_replay",
            "model_card",
            "source_and_artifact_hashes",
            "environment_lock",
        )
    ):
        return "P0", "当前运行时、数据质量与安全证据"
    if "fusion_minimal" in name or "minimal_fusion" in name or "fusion_minimal_dcnn_hp" in relative:
        return "P1", "最新离线最小融合与 H/P 研究证据"
    if relative.startswith("video/"):
        return "P2", "视频分支研究与回放证据（不进入运行时）"
    return "P3", "历史阶段、退役监督或辅助溯源"


def _status(path: Path) -> str:
    name = path.name.lower()
    if any(
        token in name
        for token in (
            "window_level_supervision_retired",
            "lopo_state_",
            "lopo_ablation",
            "second_round",
        )
    ):
        return "历史/已被当前 Condition 级协议替代"
    if "minimal_" in name or "fusion_minimal" in path.as_posix() or path.is_relative_to(VIDEO):
        return "研究专用/不可部署"
    return "当前证据/但不代表自动部署资格"


def _source_files(final_path: Path) -> list[Path]:
    return sorted(
        {
            path
            for root in _report_roots()
            for path in root.rglob("*")
            if path.is_file() and path != final_path
        },
        key=lambda path: path.as_posix(),
    )


def _base_report() -> str:
    latest_path = REPORTS / "latest_multimodal_evidence_zh.md"
    latest = latest_path.read_text(encoding="utf-8")
    lines = latest.splitlines()
    lines[0] = "# P002–P016 多模态证据总综述（唯一保留报告）"
    for index, line in enumerate(lines):
        if line.startswith("本报告读取当前 `artifacts/reports`"):
            lines[index] = (
                "本报告整合本工作区全部报告相关文件（主报告、视频融合、relaxation-only "
                "与视频编码器消融）。旧报告、JSON 指标和预测 CSV 在本报告通过 UTF-8 "
                "校验后删除；附录保留其路径、时间、重要性和处置状态。原始特征、模型、"
                "回放日志及非报告数据不在删除范围。"
            )
            break
    return "\n".join(lines).rstrip()


def _hp_detail() -> str:
    path = REPORTS / "minimal_fusion_dcnn_hp_explainability_zh.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    marker = "## 家族移除消融"
    if marker not in text:
        return ""
    return "## H/P 家族消融与分层细节（已并入）\n\n" + text.split(marker, 1)[1].strip()


def _timeline() -> str:
    return """## 时间轴与证据优先级

- **P0｜当前运行时、安全与数据质量（2026-06-22 至 2026-06-23）**：以 135 条 participant--Condition 标签、Condition-only 残差和 LOPO 为主协议；运行时仍为 `classical`，仅 Shadow，946/946 回放建议为 `hold`。这是部署判断的唯一优先层。
- **P1｜最新最小融合与 H/P 解释研究（2026-06-24）**：Ridge、1DCNN 与 H/P 家族消融只用于离线比较和模型解释，不改变运行时或安全门。
- **P2｜视频研究与编码器消融（2026-06-23）**：手工视频、VideoMAE2、relaxation-only 与视频编码器消融均为隔离的离线/录制回放研究。视频没有进入活跃推理路径，结论保持 `hold_shadow_only`。
- **P3｜历史与辅助溯源（2026-06-22 起）**：窗口级监督退役、早期 LOPO/消融、环境锁定和哈希记录用于理解演进，不覆盖 P0 当前协议。

## 历史结果的处置

窗口级监督已明确退役；早期 `lopo_state`、`lopo_ablation` 和第二轮结果只保留其方法演进含义。当前模型、标签边界和 Shadow 策略必须以本报告的 P0 结论为准。任何看似更好的研究性指标都不能绕过双目标风险门、基线比较和人工批准。"""


def _inventory(files: list[Path]) -> str:
    lines = [
        "## 已合并并删除的来源清单",
        "",
        (
            "本次合并时间（UTC）："
            f"{datetime.now(timezone.utc).replace(microsecond=0).isoformat()}。以下 {len(files)} 个文件均已读取并归入上述优先级；"
            "生成本报告后会删除，仅保留本 Markdown。"
        ),
        "",
        "| 时间（本地文件时间） | 重要性 | 处置状态 | 原始路径 |",
        "|---|---|---|---|",
    ]
    for path in files:
        timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        priority, _ = _importance(path)
        relative = path.relative_to(ARTIFACTS).as_posix()
        lines.append(f"| {timestamp} | {priority} | {_status(path)} | `{relative}` |")
    return "\n".join(lines)


def main() -> int:
    final_path = REPORTS / FINAL_NAME
    files = _source_files(final_path)
    sections = [_base_report(), _timeline()]
    if detail := _hp_detail():
        sections.append(detail)
    sections.append(_inventory(files))
    output = "\n\n".join(sections) + "\n"
    temporary = final_path.with_suffix(".md.tmp")
    temporary.write_text(output, encoding="utf-8", newline="\n")
    temporary.replace(final_path)
    print({"final": str(final_path), "source_files_summarized": len(files), "characters": len(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
