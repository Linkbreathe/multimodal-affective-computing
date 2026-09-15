"""Build one Chinese evidence index without changing any model artifact."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.utils import atomic_write_text


REPORT_FILENAME = "latest_multimodal_evidence_zh.md"
SOURCES_FILENAME = "latest_multimodal_evidence_sources.csv"
HP_DIRECTORY = "fusion_minimal_dcnn_hp"
STATUS_ZH = {"current": "当前", "historical": "历史", "research_only": "研究专用"}


def _dependencies():
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - environment failure
        raise RuntimeError("Latest evidence report requires pandas") from error
    return pd


def _relative(config: ProjectConfig, path: Path) -> str:
    try:
        return path.resolve().relative_to(config.path("artifacts").resolve()).as_posix()
    except ValueError:
        return str(path)


def _status_and_category(relative: str) -> tuple[str, str]:
    lowered = relative.lower()
    name = Path(relative).name.lower()
    if "hp_explainability" in name or "fusion_minimal_dcnn_hp" in lowered:
        return "research_only", "hp_explainability"
    if "fusion_minimal" in lowered or "video/" in lowered or "minimal_multimodal" in name:
        return "research_only", "research_benchmark"
    if name in {
        "lopo_state_metrics.json",
        "lopo_state_predictions.csv",
        "lopo_ablation_zh.md",
        "second_round_relaxation_discomfort_zh.md",
        "window_level_supervision_retired_zh.md",
    }:
        return "historical", "retired_or_prior_stage"
    if name.startswith("data_qc") or "feature_extraction_errors" in name:
        return "current", "data_qc"
    if "condition_level" in name:
        return "current", "condition_model"
    if name.startswith("dcnn_condition"):
        return "current", "dcnn_comparison"
    if "shadow_replay" in name or "policy_model_card" in name or name == "model_card_zh.md":
        return "current", "shadow_and_safety"
    if "source_and_artifact_hashes" in name or "environment_lock" in name:
        return "current", "provenance"
    return "current", "supporting_evidence"


def _source_paths(config: ProjectConfig) -> list[Path]:
    reports = config.path("reports")
    output_report = reports / REPORT_FILENAME
    output_sources = reports / SOURCES_FILENAME
    paths: set[Path] = set()
    if reports.exists():
        paths.update(
            path for path in reports.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".json"}
            and path not in {output_report, output_sources}
        )
    video = config.path("video")
    for root in (
        video / "reports",
        video / "relaxation_only" / "reports",
        video / "video_encoder_ablation" / "reports",
    ):
        if root.exists():
            paths.update(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".md", ".json"}
            )
    artifacts = config.path("artifacts")
    for relative in (
        "fusion_minimal/metrics.csv",
        "fusion_minimal/oof_predictions.csv",
        "fusion_minimal_dcnn/metrics.csv",
        "fusion_minimal_dcnn/oof_predictions.csv",
        f"{HP_DIRECTORY}/selection_audit.csv",
        f"{HP_DIRECTORY}/selection_stability.csv",
        f"{HP_DIRECTORY}/feature_family_ablation_metrics.csv",
        f"{HP_DIRECTORY}/subgroup_metrics.csv",
    ):
        path = artifacts / relative
        if path.exists():
            paths.add(path)
    return sorted(paths, key=lambda path: _relative(config, path))


def _source_inventory(config: ProjectConfig, generated_at: str) -> Any:
    pd = _dependencies()
    rows: list[dict[str, Any]] = []
    for path in _source_paths(config):
        relative = _relative(config, path)
        status, category = _status_and_category(relative)
        readable = True
        parse_error = ""
        if path.suffix.lower() in {".md", ".json"}:
            try:
                content = path.read_text(encoding="utf-8")
                if path.suffix.lower() == ".json":
                    json.loads(content)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                readable = False
                parse_error = str(error)
        rows.append(
            {
                "generated_at_utc": generated_at,
                "source_path": relative,
                "source_type": path.suffix.lower().lstrip("."),
                "category": category,
                "status": status,
                "status_zh": STATUS_ZH[status],
                "modified_at_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                "bytes": int(path.stat().st_size),
                "utf8_or_json_readable": readable,
                "read_error": parse_error,
            }
        )
    return pd.DataFrame(rows)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _load_csv(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return _dependencies().read_csv(path)
    except (OSError, ValueError):
        return None


def _number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if np.isfinite(number) else "—"


def _metric_row(name: str, metrics: dict[str, Any] | None) -> str:
    metrics = metrics or {}
    return "| {name} | {rel_mae} | {rel_rho} | {dis_mae} | {dis_rho} | {recall} | {fn} |".format(
        name=name,
        rel_mae=_number(metrics.get("relaxation_mae", metrics.get("mae"))),
        rel_rho=_number(metrics.get("relaxation_spearman", metrics.get("spearman"))),
        dis_mae=_number(metrics.get("discomfort_mae")),
        dis_rho=_number(metrics.get("discomfort_spearman")),
        recall=_number(metrics.get("discomfort_high_recall", metrics.get("risk_recall"))),
        fn=_number(metrics.get("discomfort_high_false_negatives"), 0),
    )


def _minimal_metrics_rows(path: Path, label: str) -> list[str]:
    frame = _load_csv(path)
    if frame is None or not {"record_type", "combination"}.issubset(frame.columns):
        return [f"| {label} | 未找到可读指标 | — | — | — | — | — |"]
    rows = []
    for modality in ("H", "P"):
        matched = frame.loc[
            frame["record_type"].eq("combination") & frame["combination"].eq(modality)
        ]
        if len(matched):
            rows.append(_metric_row(f"{label} {modality}", matched.iloc[0].to_dict()))
    return rows or [f"| {label} | 未找到 H/P 行 | — | — | — | — | — |"]


def _hp_rows(path: Path) -> list[str]:
    frame = _load_csv(path)
    if frame is None or "variant" not in frame.columns:
        return ["| H/P 审计 | 尚未生成 | — | — | — | — |"]
    rows = []
    for modality in ("H", "P"):
        matched = frame.loc[
            frame["modality"].eq(modality) & frame["variant"].eq("full_reference")
        ]
        if len(matched):
            row = matched.iloc[0]
            rows.append(
                "| {modality} | {rel} | {dis} | {recall} | {count} | {source} |".format(
                    modality=modality,
                    rel=_number(row.get("relaxation_mae")),
                    dis=_number(row.get("discomfort_mae")),
                    recall=_number(row.get("discomfort_high_recall")),
                    count=int(row.get("feature_count_after_removal", 0)),
                    source=row.get("reference_source", "—"),
                )
            )
    return rows or ["| H/P 审计 | 无完整参考行 | — | — | — | — |"]


def _write_dataframe(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_latest_multimodal_report(config: ProjectConfig) -> dict[str, Any]:
    """Read present evidence artifacts and write an evidence-indexed Chinese report."""
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    reports = config.path("reports")
    artifacts = config.path("artifacts")
    inventory = _source_inventory(config, generated_at)

    data_qc = _load_json(reports / "data_qc.json") or {}
    qc_participants = data_qc.get("participants", [])
    condition_metrics = _load_json(reports / "condition_level_lopo_metrics.json") or {}
    condition_summary = condition_metrics.get("metrics", {})
    dcnn_metrics = _load_json(reports / "dcnn_condition_lopo_metrics.json") or {}
    dcnn_full = dcnn_metrics.get("variants", {}).get("full", {}).get("metrics", {})
    dcnn_summary = dcnn_full.get("targets", {})
    shadow = _load_json(reports / "shadow_replay_report.json") or {}
    video = _load_json(config.path("video") / "reports" / "egocentric_video_fusion_summary.json") or {}
    video_relaxation = _load_json(
        config.path("video") / "relaxation_only" / "reports" / "video_relaxation_summary.json"
    ) or {}

    condition_targets = condition_summary.get("targets", {})
    classic_metrics = {
        "relaxation_mae": condition_targets.get("relaxation", {}).get("mae"),
        "relaxation_spearman": condition_targets.get("relaxation", {}).get("spearman"),
        "discomfort_mae": condition_targets.get("discomfort", {}).get("mae"),
        "discomfort_spearman": condition_targets.get("discomfort", {}).get("spearman"),
        "discomfort_high_recall": condition_targets.get("discomfort", {})
        .get("risk_at_fold_tuned_threshold", {})
        .get("recall"),
        "discomfort_high_false_negatives": condition_targets.get("discomfort", {})
        .get("risk_at_fold_tuned_threshold", {})
        .get("false_negatives"),
    }
    dcnn_row = {
        "relaxation_mae": dcnn_summary.get("relaxation", {}).get("mae"),
        "relaxation_spearman": dcnn_summary.get("relaxation", {}).get("spearman"),
        "discomfort_mae": dcnn_summary.get("discomfort", {}).get("mae"),
        "discomfort_spearman": dcnn_summary.get("discomfort", {}).get("spearman"),
        "discomfort_high_recall": dcnn_summary.get("discomfort", {})
        .get("risk_at_fold_tuned_threshold", {})
        .get("recall"),
        "discomfort_high_false_negatives": dcnn_summary.get("discomfort", {})
        .get("risk_at_fold_tuned_threshold", {})
        .get("false_negatives"),
    }
    qc_ok = sum(bool(row.get("available")) for row in qc_participants)
    qc_windows = [int(row.get("window_count", 0)) for row in qc_participants]
    qc_residuals = [float(row.get("median_abs_residual_ms", np.nan)) for row in qc_participants]
    video_primary = video.get("primary_masked_fallback", {})

    lines = [
        "# 最新多模态证据总报告（研究与运行时边界）",
        "",
        f"生成时间（UTC）：{generated_at}",
        "",
        "本报告读取当前 `artifacts/reports` 的全部 Markdown/JSON、最小融合 Ridge/1DCNN 基准、"
        "H/P 审计和视频研究报告。每一份来源的类别、状态、生成时间和可读性在"
        " `latest_multimodal_evidence_sources.csv` 中列出；历史与研究专用结果不会覆盖当前运行时结论。",
        "",
        "## 数据与 QC",
        "",
        f"- `data_qc.json` 记录 {len(qc_participants)} 名参与者，其中 {qc_ok} 名可用；"
        f"每人完整十秒窗口数范围为 {min(qc_windows) if qc_windows else 0}–{max(qc_windows) if qc_windows else 0}。",
        f"- 可用参与者的 marker 中位绝对残差最大值为 {_number(np.nanmax(qc_residuals) if qc_residuals else np.nan, 2)} ms。"
        "QC 是输入质量证据，不是目标或模型特征。",
        "",
        "## 标签与监督边界",
        "",
        "- 监督单元是 15×9=135 条 participant--Condition 问卷标签；946 个十秒窗口只是 Condition 内"
        "时间特征切片，不能当作独立监督样本。",
        "- relaxation 与 discomfort 来自问卷归一化；问卷原始列、Condition/刺激上下文和 QC 列不得反向"
        "作为模型特征。该边界同样适用于最小融合、视频与 H/P 审计。",
        "",
        "## 当前经典 Condition 模型与常规 DCNN",
        "",
        "| 模型 | Relaxation MAE | Relaxation Spearman | Discomfort MAE | Discomfort Spearman | 高 discomfort recall | false negatives |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _metric_row("经典 Condition 残差集成", classic_metrics),
        _metric_row("常规 Condition 1DCNN", dcnn_row),
        "",
        f"当前经典模型 `deployable`={condition_summary.get('deployable', '—')}；阻断原因："
        f"{', '.join(condition_summary.get('deployment_block_reasons', [])) or '未找到'}。"
        "因此常规运行时不因该比较自动切换后端。",
        "",
        "## 视频研究（研究专用）",
        "",
        f"- 主分析有 {video.get('video_usable_windows', '—')}/{video.get('video_windows', '—')} 个可用视频窗口；"
        f"视频融合结论为 `{video.get('recommendation', '未找到')}`。",
        f"- 手工视频模型的 relaxation MAE={_number(video_primary.get('ml_handcrafted', {}).get('relaxation_mae'))}，"
        f"discomfort MAE={_number(video_primary.get('ml_handcrafted', {}).get('discomfort_mae'))}；"
        f"VideoMAE2+1DCNN 的相应值为 {_number(video_primary.get('dl_videomae2', {}).get('relaxation_mae'))} / "
        f"{_number(video_primary.get('dl_videomae2', {}).get('discomfort_mae'))}。",
        f"- relaxation-only 视频控制实验仍是 `research_only`={video_relaxation.get('research_only', '—')}，"
        f"部署状态：`{video_relaxation.get('deployment', '未找到')}`；它没有 discomfort 安全预测，不能进入策略。",
        "",
        "## 最小融合基准（研究专用）",
        "",
        "下表仅保留 H/P 以便与专门审计连接；完整 15 组合以及 Ridge/1DCNN 的其他组合保留在各自原报告。",
        "",
        "| 模型/模态 | Relaxation MAE | Relaxation Spearman | Discomfort MAE | Discomfort Spearman | 高 discomfort recall | false negatives |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *_minimal_metrics_rows(artifacts / "fusion_minimal" / "metrics.csv", "最小融合 Ridge"),
        *_minimal_metrics_rows(artifacts / "fusion_minimal_dcnn" / "metrics.csv", "最小融合 1DCNN"),
        "",
        "这些是固定离线 LOPO 对照；不是运行时模型选择或安全放行实验。",
        "",
        "## H/P 可解释性审计（研究专用）",
        "",
        "完整 H/P 参考复用既有最小融合 1DCNN OOF，家族消融在特征选择、插补、标准化和训练前删除"
        "对应家族。它衡量模型内依赖，不是生理因果。H 包含平移速度、角速度、jerk、静止比例、"
        "位置范围和运动频谱熵；本数据没有头部朝向/yaw/pitch/roll 方向变化特征。",
        "",
        "| 模态 | Relaxation MAE | Discomfort MAE | 高 discomfort recall | 移除前列数 | 参考来源 |",
        "|---|---:|---:|---:|---:|---|",
        *_hp_rows(artifacts / HP_DIRECTORY / "feature_family_ablation_metrics.csv"),
        "",
        "`ecg_rr_std_ms_audit_only` 只保留为独立敏感性家族，不可解读为可部署生理证据。H 的 18 列"
        "少于 20 列上限，因此全折入选不能证明单一头动变量重要。EEG-disabled 与 EEG-available 的"
        "P/H OOF 分层、选择稳定性和全部消融结果见 H/P 专用 CSV/报告。",
        "",
        "## 回放、Shadow 与安全结论",
        "",
        f"- 常规 Shadow 回放：{shadow.get('windows', '—')} 个窗口，hold={shadow.get('hold_messages', '—')}，"
        f"recommend={shadow.get('recommend_messages', '—')}，safe={shadow.get('safe_messages', '—')}；"
        f"10 秒窗口完整性={shadow.get('all_windows_10_seconds', '—')}。",
        "- 当前结论保持不变：这些研究基准和解释性分析不改变运行时、自动推荐资格或 Shadow/hold 策略。"
        "在前瞻性验证、双目标安全门和人工批准前，系统只输出 Shadow 建议并保持 hold。",
        "",
        "## 解释约束",
        "",
        "所有特征相关、选择频率和家族移除结果都必须表述为观察性或模型内证据。它们不能证明"
        "生理机制、心理因果、个体诊断或临床可用性。",
    ]
    report_path = reports / REPORT_FILENAME
    sources_path = reports / SOURCES_FILENAME
    _write_dataframe(inventory, sources_path)
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    return {
        "generated_at_utc": generated_at,
        "report": str(report_path),
        "sources": str(sources_path),
        "n_sources": int(len(inventory)),
        "research_only_sources": int(inventory["status"].eq("research_only").sum()) if len(inventory) else 0,
        "historical_sources": int(inventory["status"].eq("historical").sum()) if len(inventory) else 0,
    }
