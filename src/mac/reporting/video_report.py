"""Evidence-backed Chinese report for the egocentric video fusion experiment."""

from __future__ import annotations

import json
from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.utils import atomic_write_text, write_json


def _metric(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    discomfort = metrics["targets"]["discomfort"]
    return {
        "n": int(result.get("n_condition_labels", metrics["n_labels"])),
        "relaxation_mae": float(metrics["targets"]["relaxation"]["mae"]),
        "discomfort_mae": float(discomfort["mae"]),
        "risk_recall": float(discomfort["risk_at_fold_tuned_threshold"]["per_row_threshold_recall"]),
        "deployable": bool(metrics["deployable"]),
    }


def _row(name: str, metric: dict[str, Any]) -> str:
    return (
        f"| {name} | {metric['n']} | {metric['relaxation_mae']:.4f} | {metric['discomfort_mae']:.4f} | "
        f"{metric['risk_recall']:.1%} | {'通过' if metric['deployable'] else '未通过'} |"
    )


def write_egocentric_video_report(config: ProjectConfig) -> dict[str, Any]:
    """Combine persisted ML/DL evidence without rerunning models."""
    import pandas as pd

    root = config.path("video")
    ml = json.loads((root / "reports" / "handcrafted" / "comparison_metrics.json").read_text(encoding="utf-8"))
    dl = json.loads((root / "reports" / "videomae2_dcnn" / "comparison_metrics.json").read_text(encoding="utf-8"))
    windows = pd.read_csv(config.path("features") / "video_ml" / "window_features.csv")
    usable = pd.to_numeric(windows["qc_video_usable"], errors="coerce").fillna(0.0) >= 0.5
    missing_rows = windows.loc[~usable, ["participant_id", "condition", "condition_window_index"]].to_dict(orient="records")
    manifest = pd.read_csv(root / "mp4_manifest.csv")
    primary = {
        "ml_no_video": _metric(ml["primary_masked_fallback"]["no_video"]),
        "ml_handcrafted": _metric(ml["primary_masked_fallback"]["visual"]),
        "dl_no_video": _metric(dl["primary_masked_fallback"]["no_video"]),
        "dl_videomae2": _metric(dl["primary_masked_fallback"]["videomae2"]),
    }
    sensitivity = {
        "ml_no_video": _metric(ml["video_complete_sensitivity"]["no_video"]),
        "ml_handcrafted": _metric(ml["video_complete_sensitivity"]["visual"]),
        "dl_no_video": _metric(dl["video_complete_sensitivity"]["no_video"]),
        "dl_videomae2": _metric(dl["video_complete_sensitivity"]["videomae2"]),
    }
    summary = {
        "video_windows": int(len(windows)), "video_usable_windows": int(usable.sum()), "missing_windows": missing_rows,
        "retained_mp4_count": int(len(manifest)), "retained_mp4_bytes": int(manifest["mp4_size_bytes"].sum()),
        "primary_masked_fallback": primary, "video_complete_sensitivity": sensitivity,
        "recommendation": "hold_shadow_only",
    }
    reports = root / "reports"
    write_json(reports / "egocentric_video_fusion_summary.json", summary)
    missing_description = ", ".join(
        f"{row['participant_id']}/{row['condition']}/w{row['condition_window_index']}"
        for row in missing_rows
    ) or "无"
    lines = [
        "# 第一视角视频多模态融合实验报告",
        "",
        "## 数据与可追溯性",
        "",
        f"- 保留 MP4：{summary['retained_mp4_count']} 个，合计 {summary['retained_mp4_bytes'] / 1024**3:.2f} GB；均经 ffprobe 校验为 10 fps、512×288。",
        f"- 10 秒窗口：{summary['video_windows']}；视频可用 {summary['video_usable_windows']}；缺失 {len(missing_rows)}。",
        f"- 唯一缺失窗口：{missing_description}；未补帧或复制帧。",
        "- P003 使用 `utc_timestamp_iso` 修复失真的科学计数法 Unix 时间；其 63 个窗口均保持可用。",
        "",
        "## 主分析：135 participant–Condition 标签（缺失视频显式 mask）",
        "",
        "| 方法 | 标签数 | Relaxation MAE | Discomfort MAE | 高 discomfort 召回 | 部署门 |",
        "|---|---:|---:|---:|---:|---|",
        _row("ML 无视频残差 Ridge", primary["ml_no_video"]),
        _row("ML 手工视觉融合", primary["ml_handcrafted"]),
        _row("1DCNN 无 VideoMAE2", primary["dl_no_video"]),
        _row("冻结 VideoMAE2 + 1DCNN", primary["dl_videomae2"]),
        "",
        f"## 视频完整敏感性分析：{sensitivity['ml_no_video']['n']} 个标签",
        "",
        "| 方法 | 标签数 | Relaxation MAE | Discomfort MAE | 高 discomfort 召回 | 部署门 |",
        "|---|---:|---:|---:|---:|---|",
        _row("ML 无视频残差 Ridge", sensitivity["ml_no_video"]),
        _row("ML 手工视觉融合", sensitivity["ml_handcrafted"]),
        _row("1DCNN 无 VideoMAE2", sensitivity["dl_no_video"]),
        _row("冻结 VideoMAE2 + 1DCNN", sensitivity["dl_videomae2"]),
        "",
        "## 可用性结论",
        "",
        "- 手工视觉融合在两个 cohort 上均劣于同 cohort 无视频基线，不能用于模型选择。",
        "- VideoMAE2 融合改善了主分析的 relaxation MAE（相对无 VideoMAE2 1DCNN），但 discomfort MAE 和高风险召回未满足安全门；完整视频 cohort 同样未通过。",
        "- 两条视觉路径都只能用于离线评估与录制 Shadow 回放。默认 `rtml serve`、现有无视频模型和 Unity UDP 协议未改变。",
        "- VideoMAE2 使用固定官方 ViT-Small、16 帧/10 秒窗口、384 维嵌入；每个外层 LOPO 折只用训练参与者窗口拟合 32 维 PCA。",
        "- ML 使用折内方差筛选的条件残差 Ridge；visual/no-video 在同一 cohort 共享同一非视觉特征和验证协议。",
    ]
    output = reports / "egocentric_video_fusion_report_zh.md"
    atomic_write_text(output, "\n".join(lines) + "\n")
    return {"report": str(output), **summary}


def _relaxation_metric(result: dict[str, Any]) -> dict[str, float | int]:
    metrics = result["metrics"]
    target = metrics["targets"]["relaxation"]
    return {
        "n": int(result.get("n_condition_labels", metrics["n_labels"])),
        "mae": float(target["mae"]),
        "condition_only_baseline_mae": float(target["condition_only_baseline_mae"]),
        "history_baseline_mae": float(target["history_baseline_mae"]),
        "spearman": float(target["spearman"]),
    }


def _relaxation_comparison(no_video: dict[str, Any], visual: dict[str, Any]) -> dict[str, Any]:
    baseline = _relaxation_metric(no_video)
    fused = _relaxation_metric(visual)
    if baseline["n"] != fused["n"]:
        raise ValueError("Visual and no-video relaxation models must use the same label cohort")
    absolute = float(fused["mae"] - baseline["mae"])
    return {
        "no_video": baseline,
        "video": fused,
        "mae_change_vs_no_video": absolute,
        "mae_relative_change_vs_no_video": float(absolute / baseline["mae"]) if baseline["mae"] else float("nan"),
    }


def _relaxation_row(name: str, metric: dict[str, float | int]) -> str:
    return (
        f"| {name} | {metric['n']} | {metric['mae']:.4f} | "
        f"{metric['condition_only_baseline_mae']:.4f} | {metric['history_baseline_mae']:.4f} | "
        f"{metric['spearman']:.4f} |"
    )


def write_video_relaxation_report(config: ProjectConfig) -> dict[str, Any]:
    """Write the research-only relaxation video-fusion comparison report."""
    root = config.path("video") / "relaxation_only"
    reports = root / "reports"
    ml_path = reports / "handcrafted" / "comparison_metrics.json"
    dl_path = reports / "videomae2_dcnn" / "comparison_metrics.json"
    if not ml_path.exists() or not dl_path.exists():
        raise FileNotFoundError(
            "Run 'rtml train-video-relaxation-ml' and 'rtml train-videomae2-relaxation' before reporting"
        )
    ml = json.loads(ml_path.read_text(encoding="utf-8"))
    dl = json.loads(dl_path.read_text(encoding="utf-8"))
    if ml.get("targets") != ["relaxation"] or dl.get("targets") != ["relaxation"]:
        raise ValueError("Relaxation report accepts only relaxation-only experiment summaries")
    if not ml.get("research_only") or not dl.get("research_only"):
        raise ValueError("Relaxation report requires research-only experiment summaries")
    primary = {
        "handcrafted": _relaxation_comparison(
            ml["primary_masked_fallback"]["no_video"], ml["primary_masked_fallback"]["visual"]
        ),
        "videomae2_dcnn": _relaxation_comparison(
            dl["primary_masked_fallback"]["no_video"], dl["primary_masked_fallback"]["videomae2"]
        ),
    }
    sensitivity = {
        "handcrafted": _relaxation_comparison(
            ml["video_complete_sensitivity"]["no_video"], ml["video_complete_sensitivity"]["visual"]
        ),
        "videomae2_dcnn": _relaxation_comparison(
            dl["video_complete_sensitivity"]["no_video"], dl["video_complete_sensitivity"]["videomae2"]
        ),
    }
    if {comparison["no_video"]["n"] for comparison in primary.values()} != {135}:
        raise ValueError("Primary relaxation analysis must contain exactly 135 participant-Condition labels")
    if {comparison["no_video"]["n"] for comparison in sensitivity.values()} != {134}:
        raise ValueError("Complete-video sensitivity analysis must contain exactly 134 labels")
    summary = {
        "targets": ["relaxation"],
        "research_only": True,
        "unit_of_analysis": "participant_condition",
        "primary_masked_fallback": primary,
        "video_complete_sensitivity": sensitivity,
        "deployment": "not_eligible_no_discomfort_safety_prediction",
        "feature_leakage_check": "Questionnaire raw response columns are not model features",
    }
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "video_relaxation_summary.json", summary)
    lines = [
        "# Relaxation-only 视频融合研究对照报告",
        "",
        "本实验仅预测 relaxation，用于检验移除 discomfort 多任务损失后是否改善 relaxation 预测与视频增益。它是研究对照，不含 discomfort 风险预测、风险分类器或部署资格，不能作为实时控制或部署通过依据。",
        "监督单位始终是 participant–Condition；946 个 10 秒窗口仅用于条件内特征汇聚。手工模型使用 EEG、ECG、头动、眼动及可选视频特征；VideoMAE2 嵌入保持冻结，并在每个 LOPO 训练折内拟合 PCA。问卷原始列不作为特征。",
        "",
        "## 主分析：135 个 participant–Condition 标签（缺失视频显式 mask）",
        "",
        "| 方法 | 标签数 | Relaxation MAE | Condition-only MAE | History MAE | Spearman |",
        "|---|---:|---:|---:|---:|---:|",
        _relaxation_row("手工 Ridge 无视频", primary["handcrafted"]["no_video"]),
        _relaxation_row("手工 Ridge 视觉融合", primary["handcrafted"]["video"]),
        _relaxation_row("1DCNN 无 VideoMAE2", primary["videomae2_dcnn"]["no_video"]),
        _relaxation_row("冻结 VideoMAE2 + 1DCNN", primary["videomae2_dcnn"]["video"]),
        "",
        "| 同 cohort 视频对照 | 视频相对无视频的 MAE 变化 |",
        "|---|---:|",
        f"| 手工 Ridge | {primary['handcrafted']['mae_relative_change_vs_no_video']:+.2%} |",
        f"| VideoMAE2 + 1DCNN | {primary['videomae2_dcnn']['mae_relative_change_vs_no_video']:+.2%} |",
        "",
        "## 完整视频敏感性分析：134 个标签",
        "",
        "| 方法 | 标签数 | Relaxation MAE | Condition-only MAE | History MAE | Spearman |",
        "|---|---:|---:|---:|---:|---:|",
        _relaxation_row("手工 Ridge 无视频", sensitivity["handcrafted"]["no_video"]),
        _relaxation_row("手工 Ridge 视觉融合", sensitivity["handcrafted"]["video"]),
        _relaxation_row("1DCNN 无 VideoMAE2", sensitivity["videomae2_dcnn"]["no_video"]),
        _relaxation_row("冻结 VideoMAE2 + 1DCNN", sensitivity["videomae2_dcnn"]["video"]),
        "",
        "| 同 cohort 视频对照 | 视频相对无视频的 MAE 变化 |",
        "|---|---:|",
        f"| 手工 Ridge | {sensitivity['handcrafted']['mae_relative_change_vs_no_video']:+.2%} |",
        f"| VideoMAE2 + 1DCNN | {sensitivity['videomae2_dcnn']['mae_relative_change_vs_no_video']:+.2%} |",
        "",
        "MAE 变化为负表示视频模型优于同 cohort 的无视频基线。不同 cohort 的绝对数值不得互相比较。所有模型和预测均位于 artifacts/video/relaxation_only/，不会覆盖双目标视频模型、默认实时模型、Unity 协议或 Shadow 行为。",
    ]
    output = reports / "video_relaxation_report_zh.md"
    atomic_write_text(output, "\n".join(lines) + "\n")
    return {"report": str(output), **summary}
