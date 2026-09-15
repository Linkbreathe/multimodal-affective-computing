from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.data.tables import read_rows, write_parquet_if_available
from real_time_ml.realtime.engine import InferenceEngine
from real_time_ml.utils import atomic_write_text, write_jsonl


def _float_or_value(value: str) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def replay(
    config: ProjectConfig,
    participants: list[str] | None = None,
    output: Path | None = None,
    *,
    source: Path | None = None,
    reports_dir: Path | None = None,
    report_stem: str = "shadow_replay",
) -> dict[str, Any]:
    source = source or config.path("features") / "window_features.csv"
    if not source.exists():
        raise FileNotFoundError("Run 'rtml extract-features' before replay")
    selected = set(participants or config.participants)
    rows = [row for row in read_rows(source) if row["participant_id"] in selected]
    engine = InferenceEngine(config)
    messages: list[dict[str, Any]] = []
    previous_end: dict[tuple[str, str], int] = {}
    for row in sorted(rows, key=lambda item: (item["participant_id"], float(item["window_start_unix_ms"]))):
        participant, condition = row["participant_id"], row["condition"]
        start, end = int(float(row["window_start_unix_ms"])), int(float(row["window_end_unix_ms"]))
        if end - start != 10_000:
            raise AssertionError(f"Replay encountered a non-10-second window: {participant}/{condition}")
        key = (participant, condition)
        cycle = 0 if key not in previous_end else int(round((start - previous_end[key]) / 10_000)) + int(row.get("condition_window_index", 0))
        cycle = int(float(row.get("condition_window_index", cycle)))
        previous_end[key] = end
        feature_values = {name: _float_or_value(value) for name, value in row.items()}
        coverage = {
            "eeg": float(row.get("qc_eeg_strict_coverage") or 0.0),
            "ecg": float(row.get("qc_ecg_quality") or 0.0),
            "head": float(row.get("qc_head_coverage") or 0.0),
            "eye": float(row.get("qc_eye_coverage") or 0.0),
            "video": float(row.get("qc_video_coverage") or row.get("video_available") or 0.0),
        }
        qc = {name[3:]: _float_or_value(value) for name, value in row.items() if name.startswith("qc_")}
        state, recommendation = engine.infer(
            participant_id=participant,
            condition=condition,
            cycle_index=cycle,
            start_ms=start,
            end_ms=end,
            features=feature_values,
            qc=qc,
            coverage=coverage,
        )
        messages.extend([state.to_dict(), recommendation.to_dict()])
    output = output or config.path("realtime_logs") / f"{report_stem}.jsonl"
    write_jsonl(output, messages)
    write_parquet_if_available(output.with_suffix(".parquet"), messages)
    recommendations = [row for row in messages if row["message_type"] == "ConditionRecommendation"]
    states = [row for row in messages if row["message_type"] == "StatePrediction"]
    hold_reasons = Counter(reason for row in recommendations for reason in row.get("reasons", []))
    summary = {
        "windows": len(rows),
        "state_messages": len(states),
        "recommendation_messages": len(recommendations),
        "hold_messages": sum(row.get("action") == "hold" for row in recommendations),
        "recommend_messages": sum(row.get("action") == "recommend" for row in recommendations),
        "safe_messages": sum(bool(row.get("safe")) for row in recommendations),
        "degraded_state_messages": sum(bool(row.get("degraded")) for row in states),
        "reason_counts": dict(hold_reasons.most_common()),
        "all_windows_10_seconds": all(row["window_end_ms"] - row["window_start_ms"] == 10_000 for row in messages),
        "output": str(output),
    }
    reports_dir = reports_dir or config.path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{report_stem}_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    atomic_write_text(
        reports_dir / f"{report_stem}_report_zh.md",
        "# 10 秒 Shadow 回放报告\n\n"
        f"- 窗口数：{summary['windows']}\n"
        f"- StatePrediction：{summary['state_messages']}\n"
        f"- ConditionRecommendation：{summary['recommendation_messages']}\n"
        f"- Hold / Recommend：{summary['hold_messages']} / {summary['recommend_messages']}\n"
        f"- Safe=True：{summary['safe_messages']}\n"
        f"- Degraded StatePrediction：{summary['degraded_state_messages']}\n"
        f"- 全部严格为 10 秒：{'是' if summary['all_windows_10_seconds'] else '否'}\n\n"
        "主要 hold 原因：\n\n"
        + "\n".join(f"- `{reason}`：{count}" for reason, count in summary["reason_counts"].items())
        + "\n\n所有推荐均为 shadow，未执行 Unity Condition 切换。\n",
    )
    return summary
