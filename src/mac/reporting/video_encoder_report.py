"""Research report for the VideoMAE2 video-encoder ablation suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.utils import atomic_write_text, write_json


BOOTSTRAP_REPLICATES = 10_000
KEYS = ["participant_id", "condition", "presentation_position"]


def _target_metrics(result: dict[str, Any], target: str) -> dict[str, Any]:
    metrics = result["metrics"]
    values = metrics["targets"][target]
    output = {
        "n_labels": int(metrics["n_labels"]),
        "mae": float(values["mae"]),
        "condition_only_baseline_mae": float(values["condition_only_baseline_mae"]),
        "history_baseline_mae": float(values["history_baseline_mae"]),
        "spearman": float(values["spearman"]),
        "parameter_count": int(result["parameter_count"]),
        "training_seconds": float(result["training_seconds"]),
        "video_encoder_mode": str(result["video_encoder_mode"]),
    }
    if target == "discomfort":
        risk = values.get("risk_at_fold_tuned_threshold", {})
        output["risk"] = {
            "threshold": float(risk.get("threshold", float("nan"))),
            "recall": float(risk.get("per_row_threshold_recall", float("nan"))),
            "precision": float(risk.get("precision", float("nan"))),
            "false_negatives": int(risk.get("false_negatives", 0)),
        }
    return output


def _prediction_frame(result: dict[str, Any], target: str):
    import pandas as pd

    source = Path(result["lopo_predictions_path"])
    frame = pd.read_csv(source)
    required = {*KEYS, target, f"pred_{target}"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"LOPO prediction file is missing {sorted(missing)}: {source}")
    frame = frame[[*KEYS, target, f"pred_{target}"]].copy()
    if frame[KEYS].duplicated().any() or frame[[target, f"pred_{target}"]].isna().any().any():
        raise ValueError(f"LOPO prediction file has duplicate or missing values: {source}")
    return frame


def _paired_error_change(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    target: str,
    seed: int,
) -> dict[str, Any]:
    baseline_frame = _prediction_frame(baseline, target)
    candidate_frame = _prediction_frame(candidate, target)
    merged = baseline_frame.merge(candidate_frame, on=KEYS, suffixes=("_baseline", "_candidate"), validate="one_to_one")
    if len(merged) != len(baseline_frame) or len(merged) != len(candidate_frame):
        raise ValueError("Paired VideoMAE2 comparison must retain exactly the same participant-Condition labels")
    truth_baseline = merged[f"{target}_baseline"].to_numpy(dtype=float)
    truth_candidate = merged[f"{target}_candidate"].to_numpy(dtype=float)
    if not np.allclose(truth_baseline, truth_candidate, atol=1e-12, rtol=0.0):
        raise ValueError("Paired VideoMAE2 comparison has inconsistent target labels")
    delta = np.abs(merged[f"pred_{target}_candidate"] - truth_baseline) - np.abs(
        merged[f"pred_{target}_baseline"] - truth_baseline
    )
    by_participant = delta.groupby(merged["participant_id"], sort=True).mean()
    values = by_participant.to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True).mean(axis=1)
    return {
        "target": target,
        "n_labels": int(len(merged)),
        "n_participants": int(len(values)),
        "mean_absolute_error_change": float(np.mean(values)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "participant_mean_changes": {str(name): float(value) for name, value in by_participant.items()},
        "interpretation": "negative_favors_candidate",
    }


def _metric_row(name: str, metric: dict[str, Any]) -> str:
    return (
        f"| {name} | {metric['n_labels']} | {metric['parameter_count']:,} | {metric['training_seconds']:.1f} | "
        f"{metric['mae']:.4f} | {metric['condition_only_baseline_mae']:.4f} | "
        f"{metric['history_baseline_mae']:.4f} | {metric['spearman']:.4f} |"
    )


def _pair_row(name: str, metric: dict[str, Any]) -> str:
    return (
        f"| {name} | {metric['mean_absolute_error_change']:+.4f} | "
        f"[{metric['ci95_low']:+.4f}, {metric['ci95_high']:+.4f}] | {metric['n_participants']} |"
    )


def _safety_row(name: str, metric: dict[str, Any]) -> str:
    risk = metric["risk"]
    return f"| {name} | {metric['mae']:.4f} | {risk['recall']:.1%} | {risk['precision']:.1%} | {risk['false_negatives']} |"


def _handcrafted_context(config: ProjectConfig) -> dict[str, Any]:
    candidates = {
        "dual_target": config.path("video") / "reports" / "handcrafted" / "comparison_metrics.json",
        "relaxation_only": config.path("video") / "relaxation_only" / "reports" / "handcrafted" / "comparison_metrics.json",
    }
    output: dict[str, Any] = {}
    for suite, source in candidates.items():
        if not source.exists():
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        output[suite] = payload
    return output


def _context_lines(context: dict[str, Any]) -> list[str]:
    if not context:
        return ["当前工作区未找到手工 Ridge 结果；该非 DCNN 上下文未列入本报告。"]
    lines = [
        "手工 Ridge 是独立的非 DCNN 特征族，仅作为上下文；它不参与视频编码器的配对 bootstrap 对比。",
        "",
        "| 目标配置 / cohort | Ridge 无视频 relaxation MAE | Ridge 视觉 relaxation MAE |",
        "|---|---:|---:|",
    ]
    for suite, payload in context.items():
        for cohort_name, cohort in (
            ("135 主分析", payload["primary_masked_fallback"]),
            ("134 完整视频", payload["video_complete_sensitivity"]),
        ):
            no_video = cohort["no_video"]["metrics"]["targets"]["relaxation"]["mae"]
            visual_key = "visual"
            visual = cohort[visual_key]["metrics"]["targets"]["relaxation"]["mae"]
            lines.append(f"| {suite} / {cohort_name} | {no_video:.4f} | {visual:.4f} |")
    return lines


def write_videomae2_video_encoder_ablation_report(config: ProjectConfig) -> dict[str, Any]:
    """Create a research-only comparison of direct and temporal video encoders."""
    root = config.path("video") / "video_encoder_ablation"
    summary_path = root / "reports" / "comparison_metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run 'rtml train-videomae2-video-encoder-ablation' before reporting")
    training = json.loads(summary_path.read_text(encoding="utf-8"))
    if training.get("experiment") != "videomae2_video_encoder_ablation_v1" or not training.get("research_only"):
        raise ValueError("Unexpected VideoMAE2 video-encoder ablation summary")
    expected_counts = {"primary_masked_fallback": 135, "video_complete_sensitivity": 134}
    output_suites: dict[str, Any] = {}
    seed = int(training["random_seed"])
    for suite_name, targets in (("dual_target", ("relaxation", "discomfort")), ("relaxation_only", ("relaxation",))):
        suite = training["suites"].get(suite_name)
        if suite is None:
            raise ValueError(f"Ablation summary is missing suite {suite_name}")
        cohort_summary: dict[str, Any] = {}
        for cohort_name, expected_labels in expected_counts.items():
            models = suite[cohort_name]
            required_modes = {"no_video", "video_direct_mlp", "video_temporal_1dcnn"}
            if set(models) != required_modes:
                raise ValueError(f"Ablation cohort {suite_name}/{cohort_name} has unexpected encoder modes")
            metrics = {
                mode: {target: _target_metrics(result, target) for target in targets}
                for mode, result in models.items()
            }
            if {value["relaxation"]["n_labels"] for value in metrics.values()} != {expected_labels}:
                raise ValueError(f"Ablation cohort {suite_name}/{cohort_name} has an unexpected label count")
            paired = {
                target: {
                    "direct_mlp_vs_no_video": _paired_error_change(
                        models["no_video"], models["video_direct_mlp"], target, seed + 101,
                    ),
                    "temporal_1dcnn_vs_no_video": _paired_error_change(
                        models["no_video"], models["video_temporal_1dcnn"], target, seed + 202,
                    ),
                    "temporal_1dcnn_vs_direct_mlp": _paired_error_change(
                        models["video_direct_mlp"], models["video_temporal_1dcnn"], target, seed + 303,
                    ),
                }
                for target in targets
            }
            cohort_summary[cohort_name] = {"metrics": metrics, "paired_absolute_error": paired}
        output_suites[suite_name] = cohort_summary
    output = {
        "experiment": training["experiment"],
        "research_only": True,
        "unit_of_analysis": "participant_condition",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": seed,
            "resampling_unit": "participant_mean_absolute_error_change",
        },
        "source_hashes": training["source_hashes"],
        "n_labels": training["n_labels"],
        "suites": output_suites,
        "handcrafted_ridge_context": _handcrafted_context(config),
        "deployment": "research_only_not_eligible_for_runtime_or_policy",
    }
    reports = root / "reports"
    write_json(reports / "video_encoder_ablation_summary.json", output)
    lines = [
        "# VideoMAE2 视频 1DCNN 编码器消融对比",
        "",
        "三路模型在同一实验目录内以相同随机种子、LOPO 划分和冻结 VideoMAE2 嵌入统一重训：无视频、视频直连 MLP、视频时间 1DCNN。视频直连 MLP仅使用有效窗口的 PCA 嵌入均值与可用率；时间 1DCNN 则卷积 Condition 内 10 秒窗口序列。",
        "所有结果均为研究用途：这些模型不会加载到默认实时引擎、推荐策略或视频 Shadow replay，不能作为部署通过依据。946 个窗口只用于 Condition 内序列，监督单位是 participant–Condition；问卷原始列不是特征。",
        "",
    ]
    for suite_name, title, targets in (
        ("dual_target", "双目标：relaxation + discomfort", ("relaxation", "discomfort")),
        ("relaxation_only", "单目标：relaxation", ("relaxation",)),
    ):
        lines.extend([f"## {title}", ""])
        for cohort_name, cohort_title in (
            ("primary_masked_fallback", "主分析：135 个标签（缺失视频显式 mask）"),
            ("video_complete_sensitivity", "完整视频敏感性分析：134 个标签"),
        ):
            section = output_suites[suite_name][cohort_name]
            lines.extend([
                f"### {cohort_title}",
                "",
                "| 编码器 | 标签数 | 参数量 | 训练秒数 | Relaxation MAE | Condition-only | History | Spearman |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                _metric_row("无视频", section["metrics"]["no_video"]["relaxation"]),
                _metric_row("VideoMAE2 直连 MLP", section["metrics"]["video_direct_mlp"]["relaxation"]),
                _metric_row("VideoMAE2 时间 1DCNN", section["metrics"]["video_temporal_1dcnn"]["relaxation"]),
                "",
                "| 配对绝对误差差（候选 - 基线） | 平均变化 | 95% participant-cluster bootstrap CI | 参与者数 |",
                "|---|---:|---:|---:|",
                _pair_row("直连 MLP - 无视频", section["paired_absolute_error"]["relaxation"]["direct_mlp_vs_no_video"]),
                _pair_row("时间 1DCNN - 无视频", section["paired_absolute_error"]["relaxation"]["temporal_1dcnn_vs_no_video"]),
                _pair_row("时间 1DCNN - 直连 MLP", section["paired_absolute_error"]["relaxation"]["temporal_1dcnn_vs_direct_mlp"]),
            ])
            if "discomfort" in targets:
                lines.extend([
                    "",
                    "| 编码器 | Discomfort MAE | 高 discomfort 召回 | 精确率 | 漏检数 |",
                    "|---|---:|---:|---:|---:|",
                    _safety_row("无视频", section["metrics"]["no_video"]["discomfort"]),
                    _safety_row("VideoMAE2 直连 MLP", section["metrics"]["video_direct_mlp"]["discomfort"]),
                    _safety_row("VideoMAE2 时间 1DCNN", section["metrics"]["video_temporal_1dcnn"]["discomfort"]),
                    "",
                    "双目标模型的安全指标仅供离线比较；research_only 标记强制其不可部署。",
                ])
            lines.append("")
    lines.extend(["## 手工视觉 Ridge 上下文", "", *_context_lines(output["handcrafted_ridge_context"]), ""])
    lines.append("配对误差变化为负表示候选编码器的绝对误差更小；不同 cohort 的绝对指标不能互相比较。")
    report_path = reports / "video_encoder_ablation_report_zh.md"
    atomic_write_text(report_path, "\n".join(lines) + "\n")
    return {"report": str(report_path), **output}
