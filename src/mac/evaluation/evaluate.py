from __future__ import annotations

import json
import importlib.metadata
import platform
import sys
import csv
from pathlib import Path
from typing import Any

from real_time_ml.config import ProjectConfig
from real_time_ml.modeling.safety import deployment_guard
from real_time_ml.utils import atomic_write_text, file_sha256, write_json


def evaluate(config: ProjectConfig) -> dict[str, Any]:
    condition_metrics_path = config.path("reports") / "condition_level_lopo_metrics.json"
    if condition_metrics_path.exists():
        return _evaluate_condition_level(config, condition_metrics_path)
    metrics_path = config.path("reports") / "lopo_state_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Run 'rtml train-state' before evaluation")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = payload["selected"]
    selected["deployable"], selected["deployment_block_reasons"] = deployment_guard(selected)
    payload["selected"] = selected
    write_json(metrics_path, payload)
    lines = [
        "# P002-P016 双目标状态模型卡（第二轮）",
        "",
        "- 目标：`relaxation`、`discomfort`",
        f"- 特征组：`{selected['feature_group']}`",
        f"- 模型：`{selected['candidate']}`",
        f"- 可部署门槛：`{'通过' if selected['deployable'] else '未通过，推荐器必须 hold'}`",
        f"- Relaxation LOPO MAE：{selected['mae']['relaxation']:.4f}",
        f"- Relaxation 基线 MAE（Condition-only / 历史状态）：{selected['condition_baseline_mae']['relaxation']:.4f} / {selected['history_baseline_mae']['relaxation']:.4f}",
        f"- Relaxation Spearman：{selected['spearman']['relaxation']:.4f}",
        f"- Discomfort LOPO MAE：{selected['mae']['discomfort']:.4f}",
        f"- Discomfort 基线 MAE（Condition-only / 历史状态）：{selected['condition_baseline_mae']['discomfort']:.4f} / {selected['history_baseline_mae']['discomfort']:.4f}",
        f"- 高 discomfort 点预测召回：{selected['discomfort_high_risk_recall']:.4f}",
        f"- 高 discomfort 区间上界召回：{selected['discomfort_high_risk_upper_recall']:.4f}",
        f"- 部署阻断原因：{', '.join(selected['deployment_block_reasons']) or '无'}",
        "",
        "该模型仅用于探索性 Shadow 推理；未经前瞻实验验证不得自动控制 Unity。",
    ]
    report = config.path("reports") / "model_card_zh.md"
    atomic_write_text(report, "\n".join(lines) + "\n")
    ablation_lines = ["# LOPO 与消融报告", "", "| 特征组 | 模型 | Relaxation MAE | Discomfort MAE | 可部署 |", "|---|---:|---:|---:|---:|"]
    for row in payload["all_results"]:
        if row.get("status") != "ok":
            continue
        ablation_lines.append(
            f"| {row['feature_group']} | {row['candidate']} | {row['mae']['relaxation']:.4f} | "
            f"{row['mae']['discomfort']:.4f} | {'是' if row['deployable'] else '否'} |"
        )
    atomic_write_text(config.path("reports") / "lopo_ablation_zh.md", "\n".join(ablation_lines) + "\n")

    labels_path = config.path("preprocessed") / "condition_labels.csv"
    label_rows = []
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8-sig", newline="") as handle:
            label_rows = list(csv.DictReader(handle))
    participant_count = len({row["participant_id"] for row in label_rows})
    high_risk_count = sum(float(row["discomfort"]) >= 0.5 for row in label_rows)
    high_risk_rate = high_risk_count / len(label_rows) if label_rows else float("nan")
    policy_card_path = config.path("reports") / "policy_model_card.json"
    policy_card = json.loads(policy_card_path.read_text(encoding="utf-8")) if policy_card_path.exists() else {}
    policy_validation = policy_card.get("validation", {})
    reason_text = {
        "relaxation_not_better_than_both_baselines": "relaxation 未同时优于 Condition-only 与历史状态基线",
        "discomfort_not_better_than_both_baselines": "discomfort 未同时优于 Condition-only 与历史状态基线",
        "relaxation_rank_correlation_not_positive": "relaxation 排序相关不为正",
        "discomfort_high_risk_recall_below_0_5": "高 discomfort 点预测召回低于 0.5",
    }
    blockers = [reason_text.get(reason, reason) for reason in selected["deployment_block_reasons"]]
    round_two = [
        "# 第二轮双目标训练与 Safety-first 测试报告",
        "",
        "## 结论",
        "",
        (
            "第二轮模型未达到自动推荐门槛，系统必须继续保持 `hold/shadow`。"
            if not selected["deployable"]
            else "第二轮离线门槛已通过，但仍只允许 `shadow`，需要前瞻实验后才能自动切换 Condition。"
        ),
        "",
        "本轮不再尝试覆盖 calm、pleasantness 等完整主观体验，只训练 `relaxation` 与 `discomfort`。"
        "候选 Condition 必须先通过 discomfort 与不确定性门控，之后才比较保守 relaxation 增益。",
        "",
        "## 数据与测试设计",
        "",
        f"- 参与者：{participant_count} 人（P002-P016）",
        f"- Condition 级问卷标签：{len(label_rows)} 条",
        f"- 高 discomfort（归一化分数 >= 0.5，即原始评分 >= 4/7）：{high_risk_count}/{len(label_rows)}（{high_risk_rate:.1%}）",
        "- 外层验证：按参与者 Leave-One-Participant-Out；同一 Condition 的全部 10 秒窗权重总和为 1",
        "- 对照基线：Condition-only 与当前参与者历史状态",
        "- 选择顺序：先看是否通过部署门槛，再看安全阻断项数量、高 discomfort 召回、discomfort MAE，最后看 relaxation MAE",
        "",
        "## 选中模型",
        "",
        f"- 特征组 / 模型：`{selected['feature_group']}` / `{selected['candidate']}`",
        f"- Relaxation：MAE {selected['mae']['relaxation']:.4f}；Condition-only {selected['condition_baseline_mae']['relaxation']:.4f}；历史基线 {selected['history_baseline_mae']['relaxation']:.4f}；Spearman {selected['spearman']['relaxation']:.4f}",
        f"- Discomfort：MAE {selected['mae']['discomfort']:.4f}；Condition-only {selected['condition_baseline_mae']['discomfort']:.4f}；历史基线 {selected['history_baseline_mae']['discomfort']:.4f}",
        f"- 高 discomfort 点预测召回：{selected['discomfort_high_risk_recall']:.1%}",
        f"- 高 discomfort 区间上界召回：{selected['discomfort_high_risk_upper_recall']:.1%}",
        f"- 90% 区间覆盖率（relaxation / discomfort）：{selected['interval_coverage']['relaxation']:.1%} / {selected['interval_coverage']['discomfort']:.1%}",
        f"- 平均区间宽度（relaxation / discomfort）：{selected['interval_mean_width']['relaxation']:.4f} / {selected['interval_mean_width']['discomfort']:.4f}；实时门槛 {float(config.get('policy.uncertainty_width_max')):.4f}",
        "",
        "## Safety-first 判定",
        "",
        f"- 部署门槛：{'通过' if selected['deployable'] else '未通过'}",
        f"- 阻断项：{'；'.join(blockers) if blockers else '无'}",
        "- 运行时规则：状态区间过宽、模态覆盖不足、模型未过双基线或候选 discomfort 上界达到 0.5 时，直接 hold",
        "- 只有安全候选存在且其 relaxation 保守下界高于当前状态上界时，才输出 recommend；输出仍为 shadow，不自动控制 Unity",
    ]
    if policy_validation:
        policy_mae = policy_validation.get("group_cv_mae", {})
        policy_width = policy_validation.get("interval_half_width_90", {})
        round_two.extend([
            "",
            "## Condition 策略模型交叉验证",
            "",
            f"- Relaxation MAE：{float(policy_mae.get('relaxation', float('nan'))):.4f}",
            f"- Discomfort MAE：{float(policy_mae.get('discomfort', float('nan'))):.4f}",
            f"- 90% 半区间宽度（relaxation / discomfort）：{float(policy_width.get('relaxation', float('nan'))):.4f} / {float(policy_width.get('discomfort', float('nan'))):.4f}",
        ])
    replay_path = config.path("reports") / "shadow_replay_report.json"
    if replay_path.exists():
        replay_summary = json.loads(replay_path.read_text(encoding="utf-8"))
        round_two.extend([
            "",
            "## 完整 Shadow 回放",
            "",
            f"- 10 秒窗口：{int(replay_summary.get('windows', 0))}",
            f"- Hold / Recommend：{int(replay_summary.get('hold_messages', 0))} / {int(replay_summary.get('recommend_messages', 0))}",
            f"- Safe=True：{int(replay_summary.get('safe_messages', 0))}",
            f"- 全部窗口严格为 10 秒：{'是' if replay_summary.get('all_windows_10_seconds') else '否'}",
        ])
    round_two.extend([
        "",
        "## 解释与下一步",
        "",
        "高 discomfort 样本仅占约一成，点预测容易向低分集中。区间上界可以提高风险敏感度，但如果区间本身过宽，正确动作仍是 hold，而不是把不确定预测当成安全证据。",
        "下一轮应优先增加高 discomfort Condition 的前瞻样本，或在参与者安全约束下设计更平衡的刺激覆盖；在此之前不应降低阈值来换取更多 recommend。",
    ])
    atomic_write_text(config.path("reports") / "second_round_relaxation_discomfort_zh.md", "\n".join(round_two) + "\n")
    project_root = config.source.parent.parent
    source_files = list((project_root / "src").rglob("*.py")) + list((project_root / "integrations").rglob("*.*"))
    hashes = {str(path): file_sha256(path) for path in [config.source, metrics_path, *source_files] if path.is_file()}
    for model in config.path("models").glob("*.joblib"):
        hashes[str(model)] = file_sha256(model)
    write_json(config.path("reports") / "source_and_artifact_hashes.json", hashes)
    distributions = ["numpy", "scipy", "pandas", "pyarrow", "PyYAML", "pyxdf", "mne", "scikit-learn", "joblib", "opencv-python-headless", "pylsl"]
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(
        config.path("reports") / "environment_lock.json",
        {"python": sys.version, "platform": platform.platform(), "packages": versions},
    )
    return {"model_card": str(report), "selected": selected}


def _evaluate_condition_level(config: ProjectConfig, metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    relaxation = metrics["targets"]["relaxation"]
    discomfort = metrics["targets"]["discomfort"]
    risk = discomfort["risk_at_fold_tuned_threshold"]
    lines = [
        "# P002–P016 Condition 级双目标模型卡",
        "",
        "- 监督单位：135 个 participant–Condition 标签；10 秒窗口仅用于特征汇聚。",
        "- 目标：relaxation（改善）与 discomfort（安全风险）。",
        f"- 特征：汇聚前 {payload['feature_count_before_fold_selection']} 列；折内稀疏、方差、相关性和互信息筛选。",
        f"- 候选回归器：{', '.join(payload['candidate_models'])}。",
        f"- Relaxation：MAE {relaxation['mae']:.4f}；Condition-only {relaxation['condition_only_baseline_mae']:.4f}；History {relaxation['history_baseline_mae']:.4f}；Spearman {relaxation['spearman']:.4f}；排序准确率 {relaxation['ranking_accuracy']:.1%}。",
        f"- Discomfort：MAE {discomfort['mae']:.4f}；Condition-only {discomfort['condition_only_baseline_mae']:.4f}；History {discomfort['history_baseline_mae']:.4f}。",
        f"- 高 discomfort：PR-AUC {risk['pr_auc']:.4f}；召回 {risk['per_row_threshold_recall']:.1%}；假阴性 {risk['false_negatives']}。",
        f"- 部署门：{'通过' if metrics['deployable'] else '未通过，推荐器强制 hold'}；原因：{', '.join(metrics['deployment_block_reasons']) or '无'}。",
        "",
        "该资产只允许 Shadow 模式；未经前瞻实验验证不得自动切换 Unity Condition。",
    ]
    report = config.path("reports") / "model_card_zh.md"
    atomic_write_text(report, "\n".join(lines) + "\n")
    sweep = [
        "# Condition 级 LOPO / 高风险阈值报告",
        "",
        "| 风险概率阈值 | 召回 | 精确率 | 假阴性 | PR-AUC |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in discomfort["threshold_sweep"]:
        sweep.append(f"| {row['threshold']:.2f} | {row['recall']:.1%} | {row['precision']:.1%} | {row['false_negatives']} | {row['pr_auc']:.4f} |")
    sweep.extend(["", "## 个体校准", ""])
    calibration_baselines = {}
    prediction_root = config.path("predictions") if not config.is_legacy else config.path("reports")
    prediction_path = prediction_root / "condition_level_lopo_predictions.csv"
    if prediction_path.exists():
        import pandas as pd

        prediction_frame = pd.read_csv(prediction_path)
        for count in metrics["personalized_calibration"]:
            subset = prediction_frame[prediction_frame["presentation_position"] > int(count)]
            calibration_baselines[count] = {
                "relaxation_condition_only_mae": float((subset["relaxation"] - subset["condition_only_relaxation"]).abs().mean()),
                "relaxation_history_mae": float((subset["relaxation"] - subset["history_relaxation"]).abs().mean()),
                "discomfort_condition_only_mae": float((subset["discomfort"] - subset["condition_only_discomfort"]).abs().mean()),
                "discomfort_history_mae": float((subset["discomfort"] - subset["history_discomfort"]).abs().mean()),
            }
    for count, row in metrics["personalized_calibration"].items():
        baselines = calibration_baselines.get(count, {})
        sweep.append(
            f"- 前 {count} 个 Condition 校准后：n={row['n_predictions']}，relaxation MAE={row['relaxation_mae']:.4f}，"
            f"Spearman={row['relaxation_spearman']:.4f}，排序准确率={row['relaxation_ranking_accuracy']:.1%}，"
            f"discomfort MAE={row['discomfort_mae']:.4f}；同一预测集合的 Condition-only / History MAE（relaxation）="
            f"{baselines.get('relaxation_condition_only_mae', float('nan')):.4f} / {baselines.get('relaxation_history_mae', float('nan')):.4f}，"
            f"（discomfort）={baselines.get('discomfort_condition_only_mae', float('nan')):.4f} / {baselines.get('discomfort_history_mae', float('nan')):.4f}。"
        )
    atomic_write_text(config.path("reports") / "condition_level_lopo_report_zh.md", "\n".join(sweep) + "\n")
    atomic_write_text(
        config.path("reports") / "window_level_supervision_retired_zh.md",
        "# 窗口级监督结果已退役\n\n"
        "`lopo_state_metrics.json` 与 `lopo_state_predictions.csv` 是旧窗口级弱标签训练留下的审计产物，"
        "不可用于模型选择、部署门或推荐。当前唯一有效的监督评估是 `condition_level_lopo_metrics.json`："
        "它以 135 个 participant–Condition 问卷标签为单位，窗口只参与特征汇聚。\n",
    )
    project_root = config.source.parent.parent
    source_files = list((project_root / "src").rglob("*.py")) + list((project_root / "integrations").rglob("*.*"))
    hashes = {str(path): file_sha256(path) for path in [config.source, metrics_path, *source_files] if path.is_file()}
    for model in config.path("models").glob("*.joblib"):
        hashes[str(model)] = file_sha256(model)
    write_json(config.path("reports") / "source_and_artifact_hashes.json", hashes)
    distributions = ["numpy", "scipy", "pandas", "pyarrow", "PyYAML", "pyxdf", "mne", "scikit-learn", "joblib", "opencv-python-headless", "pylsl"]
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(config.path("reports") / "environment_lock.json", {"python": sys.version, "platform": platform.platform(), "packages": versions})
    return {"model_card": str(report), "metrics": metrics, "unit_of_analysis": "participant_condition"}
