"""Generate the single final report for a run without legacy-report inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.utils import atomic_write_text


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _metric_highlights(metrics_dir: Path) -> list[str]:
    highlights: list[str] = []
    for path in sorted(metrics_dir.rglob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        selected = payload.get("selected")
        if isinstance(selected, dict):
            deployable = selected.get("deployable")
            block_reasons = selected.get("deployment_block_reasons", [])
            highlights.append(
                f"- `{path.name}`：部署资格 {'通过' if deployable else '未通过'}；"
                f"阻断项 {', '.join(map(str, block_reasons)) or '无'}。"
            )
        condition = payload.get("metrics")
        if isinstance(condition, dict) and "deployable" in condition:
            reasons = condition.get("deployment_block_reasons", [])
            highlights.append(
                f"- `{path.name}`：Condition LOPO 部署资格 {'通过' if condition['deployable'] else '未通过'}；"
                f"阻断项 {', '.join(map(str, reasons)) or '无'}。"
            )
    return highlights


def _relative_paths(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(root.rglob("*")) if path.is_file()]


def write_run_summary(config: ProjectConfig) -> dict[str, Any]:
    """Write ``reports/<run_id>_summary_zh.md`` from this run alone.

    The function intentionally never scans ``artifacts/legacy`` or historical
    Markdown.  This makes the report reproducible from the run manifest plus
    its normalized metrics and prediction artifacts.
    """
    if config.is_legacy:
        raise ValueError("The new report command requires --experiment layered configuration")
    assert config.layout is not None
    layout = config.layout
    manifest_path = layout.manifests / "run_manifest.json"
    manifest = _read_json(manifest_path) or {}
    metrics = _relative_paths(layout.metrics)
    predictions = _relative_paths(layout.predictions)
    model_files = _relative_paths(layout.models)
    highlights = _metric_highlights(layout.metrics)
    targets = manifest.get("targets", ["relaxation", "discomfort"])
    lines = [
        f"# {layout.run_id} 运行总结",
        "",
        "## 协议与边界",
        "",
        f"- 分析单位：`{manifest.get('unit_of_analysis', 'participant_condition')}`。",
        f"- 目标：`{', '.join(map(str, targets))}`。",
        f"- 外层验证：`{manifest.get('outer_cv', 'leave_one_participant_out')}`；10 秒窗口仅作特征时间轴。",
        f"- 运行模式：`{manifest.get('mode', config.get('run.mode'))}`；research_only=`{manifest.get('research_only', False)}`。",
        "- 安全契约：保持 Shadow-only；未满足基线与风险门时动作必须为 `hold`。",
        "",
        "## 指标摘要",
        "",
        *(highlights or ["- 尚未发现 JSON 指标。请先执行训练或评估命令。"]),
        "",
        "## 机器产物清单",
        "",
        f"- metrics（{len(metrics)}）：{', '.join(f'`{name}`' for name in metrics) or '无'}",
        f"- predictions（{len(predictions)}）：{', '.join(f'`{name}`' for name in predictions) or '无'}",
        f"- models（{len(model_files)}）：{', '.join(f'`{name}`' for name in model_files) or '无'}",
        "",
        "## 解释限制",
        "",
        "本报告仅总结当前 run 的规范化机器产物，不导入 legacy artifacts，也不把研究分支结果视作自动部署资格。",
    ]
    output = config.human_report_path()
    atomic_write_text(output, "\n".join(lines) + "\n")
    return {
        "run_id": layout.run_id,
        "report": str(output),
        "metrics": len(metrics),
        "predictions": len(predictions),
        "models": len(model_files),
    }
