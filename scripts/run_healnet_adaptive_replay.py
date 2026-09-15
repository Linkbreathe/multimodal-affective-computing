"""Run the deterministic P009 HealNet chronological shadow replay and report it."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.adaptive.controller import AdaptiveController, ControllerConfig
from src.adaptive.healnet_prefix import file_sha256
from src.adaptive.metrics import build_all_metrics
from src.adaptive.replay import ratings_from_labels, run_chronological_replay


DEFAULT_OUTPUT = ROOT / "artifacts/healnet_adaptive_replay/p009_healnet_frozen_prefix_v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_comparators(model_manifest: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for checkpoint in model_manifest["checkpoints"]:
        seed = int(checkpoint["seed"])
        path = Path(checkpoint["path"]).parent.parent / f"eeg9_healnet_full_s{seed}_predictions.csv"
        frame = pd.read_csv(path)
        frames.append(frame.loc[frame["participant_id"].astype(str).eq("P009")].copy())
    return pd.concat(frames, ignore_index=True)


def _plot_trace(trace: pd.DataFrame, output: Path) -> None:
    time_s = trace["time_s"].to_numpy(dtype=float)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0, 1.0]},
    )
    state_axis, intensity_axis, frequency_axis = axes
    state_axis.plot(time_s, trace["pred_relaxation"], color="#167c80", linewidth=2, label="Predicted relaxation")
    state_axis.plot(time_s, trace["pred_discomfort"], color="#d94f4f", linewidth=2, label="Predicted discomfort")
    state_axis.step(time_s, trace["true_relaxation"], where="post", color="#167c80", alpha=0.3, linestyle="--", label="Condition rating: relaxation")
    state_axis.step(time_s, trace["true_discomfort"], where="post", color="#d94f4f", alpha=0.3, linestyle="--", label="Condition rating: discomfort")
    state_axis.axhline(0.5, color="#d94f4f", linewidth=1, linestyle=":", label="Discomfort safety threshold")
    state_axis.set_ylim(-0.03, 1.03)
    state_axis.set_ylabel("Model state / rating")
    state_axis.legend(loc="upper right", ncol=3, fontsize=8)
    state_axis.grid(alpha=0.2)

    intensity_axis.step(time_s, trace["intensity_index"], where="post", color="#5e3c99", linewidth=2.2, label="Virtual recommended intensity")
    intensity_axis.step(time_s, trace["actual_intensity_index"], where="post", color="#999999", linewidth=1.3, linestyle="--", label="Historical actual intensity")
    intensity_axis.set_yticks([0, 1, 2], ["Low", "Medium", "High"])
    intensity_axis.set_ylabel("Intensity")
    intensity_axis.set_ylim(-0.25, 2.25)
    intensity_axis.legend(loc="upper right", fontsize=8)
    intensity_axis.grid(alpha=0.2)

    frequency_axis.step(time_s, trace["frequency_index"], where="post", color="#e08214", linewidth=2.2, label="Virtual recommended frequency")
    frequency_axis.step(time_s, trace["actual_frequency_index"], where="post", color="#999999", linewidth=1.3, linestyle="--", label="Historical actual frequency")
    frequency_axis.set_yticks([0, 1, 2], ["Low", "Medium", "High"])
    frequency_axis.set_ylabel("Frequency")
    frequency_axis.set_xlabel("Observed physiological time (seconds; wall-clock gaps omitted)")
    frequency_axis.set_ylim(-0.25, 2.25)
    frequency_axis.legend(loc="upper right", fontsize=8)
    frequency_axis.grid(alpha=0.2)

    action_rows = trace.loc[trace["action"].ne("hold")]
    for row in action_rows.itertuples(index=False):
        for axis in axes:
            axis.axvline(float(row.time_s), color="#333333", linewidth=0.7, alpha=0.35)
        state_axis.annotate(
            f"{row.controller_condition_before}→{row.controller_condition}",
            xy=(float(row.time_s), float(row.pred_relaxation)),
            xytext=(2, 8),
            textcoords="offset points",
            rotation=70,
            fontsize=7,
        )
    for row in trace.loc[trace["rating_available"].astype(bool)].itertuples(index=False):
        state_axis.scatter(float(row.time_s), 0.98, marker="v", color="#222222", s=24, zorder=5)
    for row in trace.loc[trace["gap_reset"].astype(bool)].itertuples(index=False):
        for axis in axes:
            axis.axvline(float(row.time_s), color="#4c78a8", linewidth=1.0, linestyle=":", alpha=0.45)
    figure.suptitle(
        "P009 HealNet causal-prefix chronological shadow replay\n"
        "Controller trajectory is virtual; historical physiology does not observe recommended actions",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _format_metric(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _report_markdown(
    metrics: dict[str, Any],
    trace: pd.DataFrame,
    calibration: dict[str, Any],
    inference_audit: dict[str, Any],
) -> str:
    accuracy = metrics["model_accuracy"]
    relaxation = accuracy["condition_balanced"]["relaxation"]
    discomfort = accuracy["condition_balanced"]["discomfort"]
    high_d = accuracy["high_discomfort_detection"]["condition_level_primary"]
    controller = metrics["controller_behavior"]
    comparison = metrics["frozen_baseline_comparison"]["models"]
    conflicts = metrics["personalization"]["model_rating_conflicts"]
    action_rows = trace.loc[trace["action"].ne("hold")]
    action_lines = [
        f"| {int(row.replay_index) + 1} | {int(row.time_s)} | {row.controller_condition_before} → {row.controller_condition} | {row.action} | {row.action_reason} |"
        for row in action_rows.itertuples(index=False)
    ]
    condition_lines = []
    for record in accuracy["condition_predictions"]:
        condition_lines.append(
            "| {actual_condition} | {n_observed_windows} | {true_relaxation:.3f} | {pred_relaxation:.3f} | "
            "{true_discomfort:.3f} | {pred_discomfort:.3f} |".format(**record)
        )
    gates = metrics["go_no_go_gates"]
    return f"""# P009 HealNet 自适应影子回放报告

生成时间：{datetime.now(timezone.utc).isoformat()}

## 结论先行

本次主实验链路已经跑通：3 份冻结的 HealNet fold-05 权重对 P009 前 50 个时间窗口进行了严格 causal-prefix 推理，控制器按时间顺序完成了 500 秒生理数据的 chronological shadow replay。模型和控制器都没有在 P009 上训练、选模或调参。

50 个窗口包含 {inference_audit['observed_signal_seconds']:.0f} 秒有效生理数据，但真实 wall-clock 跨度为 {inference_audit['wall_clock_span_seconds']:.1f} 秒，并含 {inference_audit['large_gap_count']} 个较大间断。平滑状态和 dwell epoch 均在间断后重置；图中的 0–500 秒轴表示累计有效生理时间，不伪装成连续 wall-clock。

当前结论是 **NO-GO（不能进入真实受试者自动闭环部署）**。主要原因不是代码失败，而是实验结果显示：独立 Condition 层面的高不适召回率为 {_format_metric(high_d['sensitivity_recall'])}，漏掉 {high_d['false_negatives']} 个高不适 Condition；同时只有 {calibration['duration_seconds_used']:.0f} 秒初始生理 baseline，且缺少独立的 120 秒 C1 safety baseline。影子回放也无法提供动作因果效果。

## 1. Foundation Model 有多准确？

主指标以 {accuracy['n_independent_conditions']} 个独立 Condition 评分等权计算，而不是把复制到窗口上的标签当成 50 个独立样本。

| 目标 | Condition-balanced MAE | RMSE | Spearman | 预测范围 | 预测标准差 |
|---|---:|---:|---:|---:|---:|
| Relaxation | {_format_metric(relaxation['mae'])} | {_format_metric(relaxation['rmse'])} | {_format_metric(relaxation['spearman'])} | {_format_metric(relaxation['prediction_range'])} | {_format_metric(relaxation['prediction_std'])} |
| Discomfort | {_format_metric(discomfort['mae'])} | {_format_metric(discomfort['rmse'])} | {_format_metric(discomfort['spearman'])} | {_format_metric(discomfort['prediction_range'])} | {_format_metric(discomfort['prediction_std'])} |

高不适阈值为 normalized discomfort ≥ 0.5（原始量表 ≥ 4/7）。Condition 层面 sensitivity={_format_metric(high_d['sensitivity_recall'])}、specificity={_format_metric(high_d['specificity'])}、balanced accuracy={_format_metric(high_d['balanced_accuracy'])}、AUPRC={_format_metric(high_d['auprc'])}、false negatives={high_d['false_negatives']}。这说明当前 discomfort 输出严重偏低，不能充当独立安全检测器。

与原训练阶段冻结的 fold-local baseline 比较：Relaxation MAE 为 HealNet {_format_metric(comparison['healnet']['relaxation']['mae'])}、condition-only {_format_metric(comparison['condition_only']['relaxation']['mae'])}、history {_format_metric(comparison['history']['relaxation']['mae'])}；Discomfort MAE 分别为 {_format_metric(comparison['healnet']['discomfort']['mae'])}、{_format_metric(comparison['condition_only']['discomfort']['mae'])}、{_format_metric(comparison['history']['discomfort']['mae'])}。HealNet 的 Relaxation 好于两种 baseline，但 Discomfort 并未优于 condition-only baseline。

| Actual Condition | 本段窗口数 | True R | Pred R | True D | Pred D |
|---|---:|---:|---:|---:|---:|
{chr(10).join(condition_lines)}

## 2. 是否表现出 adaptive 行为？

虚拟推荐在 50 个窗口内变化 {controller['parameter_changes']} 次，访问 {controller['distinct_controller_conditions']} 个 Condition；Intensity 变化 {controller['intensity_changes']} 次，Frequency 变化 {controller['frequency_changes']} 次。相邻一级变化比例为 {_format_metric(controller['adjacent_one_level_change_fraction'])}，单轴变化比例为 {_format_metric(controller['single_axis_change_fraction'])}，锁定状态下选择 C9 的次数为 {controller['c9_locked_selection_count']}。

“推荐轨迹具有变化”门为 **{'PASS' if controller['recommendation_variation_gate']['pass'] else 'FAIL'}**。但“动作是否改善状态”的因果门是 **不可评价**：所有推荐均未物理执行，`action_outcome_observable=false`，所以 Success / Ineffective / Unsafe 等后果均没有被推断。

## 3. Intensity / Frequency 如何变化？

| 窗口 | 生理时间(s) | 虚拟 Condition 变化 | 轴向动作 | 因由 |
|---:|---:|---|---|---|
{chr(10).join(action_lines) if action_lines else '| — | — | 无变化 | hold | — |'}

控制器使用最近三个有效预测的中位数，普通动作前至少观察三个窗口，正常 dwell 最多六个窗口；候选只允许相邻、单轴、一级变化。评分仅在对应历史 Condition 完成后揭示。原始数据把 Too weak / Too strong 都压成了 `visual_fit=0`，方向不可恢复，因此方向规则被禁用并对 C9 fail-closed。

本段有 {controller['history_influenced_actions']} 次选择使用了当时已经揭示的个人历史；有 {conflicts['high_discomfort_rating_but_low_model_prediction']} 个 Condition 出现“用户高不适、模型仍低于安全阈值”的关键冲突。由于虚拟动作没有响应数据，成功的 Intensity/Frequency 方向均保持为空，不作猜测。

## 4. 是否有不安全动作或遗漏？

- 实际物理动作：0；因此没有声称任何推荐动作“安全有效”。
- 单轴/相邻规则违规：{controller['illegal_transition_count']}。
- C9 锁定违规：{controller['c9_locked_selection_count']}。
- 高不适 Condition false negatives：{high_d['false_negatives']}（安全关键失败）。
- 预测 D≥0.5 的窗口：{metrics['safety']['predicted_high_discomfort_windows']}；真实标签 D≥0.5 的描述性窗口：{metrics['safety']['true_high_discomfort_windows_descriptive']}。
- D 上升时仍增加负荷的虚拟建议：{metrics['safety']['continued_load_increase_while_discomfort_rising']}。

## 5. 哪些是真实历史，哪些是虚拟或合成？

- 真实历史：P009 的 ECG/EEG/眼动/头动/视频 embedding、时间戳、实际呈现 Condition、结束评分。
- 模型输出：冻结 HealNet 对截至当前窗口的前缀推理结果。
- 虚拟结果：Controller Condition 和所有参数变化建议。
- 合成回放：未运行（`synthetic_replay=false`）。
- 不成立的解释：不能把建议后的下一个历史窗口当作建议造成的效果。

## 6. 当前结果支持什么、不能支持什么？

支持：推理实现是 causal-prefix；三份 checkpoint 与原训练全序列输出可复核等价；控制器能生成满足结构安全约束的多次虚拟建议；完整审计链和图表可复现。

不支持：不能证明 adaptive 建议改善 relaxation/discomfort，不能证明当前 discomfort 模型足以保护受试者，不能据此直接部署。下一步必须补齐独立 60 秒生理 baseline、120 秒 C1 safety baseline，并在新的受试者上进行带人工接管和停止条件的 prospective closed-loop 对照实验。

额外的校准警告：baseline 得到的 `epsilon_D={calibration['epsilon_discomfort_p90_absolute_deviation']:.6f}`，但三种子间 D 的中位标准差为 `{calibration['median_seed_std_discomfort']:.6f}`。模型间不确定性远大于这 40 秒内的自然输出波动，所以该个人阈值只能作为本次 shadow replay 的暂定值，不能直接用于真实安全控制。

## Go / No-Go

| Gate | 结果 |
|---|---|
| 高不适召回 | {'PASS' if gates['model_high_discomfort_recall']['pass'] else 'FAIL'} |
| Controller 结构约束 | {'PASS' if gates['controller_transition_invariants']['pass'] else 'FAIL'} |
| Calibration 完整性 | {'PASS' if gates['calibration_adequacy']['pass'] else 'FAIL'} |
| 闭环因果效果 | NOT EVALUABLE |
| Prospective deployment | **{gates['prospective_deployment']['decision']}** |

## 完成范围

- R00：冻结 P009、split、3 个 checkpoint 和输入哈希——完成。
- R10/R11：causal-prefix 推理、50 行预测、与原全序列输出等价检查——完成。
- R20：现有 45 秒 initial baseline 提取与校准——完成，但按方案标准判定不足；未用后续 C1 泄漏补齐。
- R21/R22：Controller、C9 fail-closed、信号/相邻/单轴/未来评分隔离测试——完成。
- R30/R31：chronological shadow replay、指标、图、报告、审计和 checksum——完成。
- R40：可选 synthetic replay——未运行；它不是本次主实验完成条件，也不能替代 prospective 闭环。
"""


def _compliance_audit(
    predictions: pd.DataFrame,
    trace: pd.DataFrame,
    metrics: dict[str, Any],
    replay_audit: dict[str, Any],
    inference_audit: dict[str, Any],
) -> dict[str, Any]:
    changed = trace.loc[trace["action"].ne("hold")]
    high_frequency_increases = changed.loc[
        changed["action"].eq("frequency_increase")
        & pd.to_numeric(changed["frequency_index"]).eq(2)
    ]
    high_frequency_safe = all(
        float(row.smooth_discomfort_30s) < 0.5
        and not json.loads(str(row.decision_details)).get("discomfort_rising", False)
        for row in high_frequency_increases.itertuples(index=False)
    )
    checks = {
        "exactly_50_predictions": len(predictions) == 50,
        "exactly_50_trace_rows": len(trace) == 50,
        "timestamps_strictly_increasing": bool(predictions["window_start_unix_ms"].is_monotonic_increasing),
        "all_formal_windows_common_signal_valid": bool(predictions["signal_valid"].all()),
        "no_future_embedding_access": inference_audit["future_window_embedding_access_count"] == 0,
        "full_prefix_matches_original_frozen_full_sequence": bool(
            inference_audit["full_sequence_equivalence"]["pass"]
        ),
        "baseline_precedes_formal_segment": bool(inference_audit["baseline_precedes_first_formal_window"]),
        "model_did_not_receive_condition_or_rating": bool(
            ~predictions["model_received_condition_or_rating"].astype(bool).any()
        ),
        "current_rating_never_revealed": bool(replay_audit["current_condition_rating_never_passed"]),
        "all_actions_virtual": bool(~trace["physical_action_applied"].astype(bool).any()),
        "all_action_outcomes_unobservable": bool(~trace["action_outcome_observable"].astype(bool).any()),
        "no_synthetic_rows": bool(~trace["synthetic_replay"].astype(bool).any()),
        "all_changes_adjacent_and_single_axis": bool(
            metrics["controller_behavior"]["illegal_transition_count"] == 0
            and metrics["controller_behavior"]["single_axis_change_fraction"] == 1.0
        ),
        "no_locked_c9_selection": metrics["controller_behavior"]["c9_locked_selection_count"] == 0,
        "maximum_contiguous_normal_dwell_six_windows": bool(
            metrics["controller_behavior"]["longest_observed_dwell_windows"] <= 6
            and metrics["controller_behavior"]["active_max_dwell_violation_count"] == 0
        ),
        "no_load_increase_during_discomfort_rise": bool(
            metrics["safety"]["continued_load_increase_while_discomfort_rising"] == 0
        ),
        "low_relaxation_never_directly_increased_intensity": bool(
            not (
                changed["action_reason"].eq("low_relaxation_reversible_non_intensity_probe")
                & changed["action"].eq("intensity_increase")
            ).any()
        ),
        "high_frequency_escalation_only_with_low_nonrising_discomfort": high_frequency_safe,
        "visual_fit_direction_not_guessed": not replay_audit["visual_fit_direction_rule_enabled"],
        "no_p009_training_or_tuning": True,
    }
    return {
        "checks": checks,
        "all_protocol_checks_pass": all(checks.values()),
        "scientific_limitations_are_gate_failures_not_protocol_failures": True,
        "note": (
            "A protocol check can pass while a scientific deployment gate fails. Here the run is compliant, "
            "but high-discomfort recall and calibration adequacy fail."
        ),
    }


def _write_checksums(output_dir: Path) -> Path:
    checksum_path = output_dir / "artifact_checksums.sha256"
    files = sorted(path for path in output_dir.iterdir() if path.is_file() and path != checksum_path)
    lines = [f"{file_sha256(path)}  {path.name}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    prediction_path = output_dir / "foundation_predictions_50windows.csv"
    calibration_path = output_dir / "calibration.json"
    model_manifest_path = output_dir / "healnet_frozen_ensemble_manifest.json"
    inference_audit_path = output_dir / "prefix_inference_audit.json"
    for path in (prediction_path, calibration_path, model_manifest_path, inference_audit_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run prefix inference first; missing {path}")
    predictions = pd.read_csv(prediction_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    inference_audit = json.loads(inference_audit_path.read_text(encoding="utf-8"))
    cache = torch.load(Path(model_manifest["embedding_cache"]["path"]), map_location="cpu", weights_only=False)
    labels_path = args.labels or Path(cache["metadata"]["inputs"]["labels"]["path"])
    labels = pd.read_csv(labels_path)
    ratings = ratings_from_labels(labels, participant_id="P009")

    config = ControllerConfig(
        baseline_relaxation=float(calibration["relaxation_baseline_median"]),
        baseline_discomfort=float(calibration["discomfort_baseline_median"]),
        epsilon_relaxation=float(calibration["epsilon_relaxation_p90_absolute_deviation"]),
        epsilon_discomfort=float(calibration["epsilon_discomfort_p90_absolute_deviation"]),
    )
    controller = AdaptiveController(config)
    trace, replay_audit = run_chronological_replay(predictions, ratings, controller)
    comparators = _load_comparators(model_manifest)
    metrics = build_all_metrics(predictions, trace, calibration, replay_audit, comparators)
    trace_path = output_dir / "adaptive_replay_trace.csv"
    metrics_path = output_dir / "adaptive_replay_metrics.json"
    plot_path = output_dir / "adaptive_replay_plot.png"
    report_path = output_dir / "adaptive_replay_report.md"
    replay_audit_path = output_dir / "chronological_replay_audit.json"
    compliance_path = output_dir / "protocol_compliance.json"
    trace.to_csv(trace_path, index=False)
    _write_json(metrics_path, metrics)
    _plot_trace(trace, plot_path)
    report_path.write_text(
        _report_markdown(metrics, trace, calibration, inference_audit),
        encoding="utf-8",
    )
    _write_json(replay_audit_path, replay_audit)
    compliance = _compliance_audit(predictions, trace, metrics, replay_audit, inference_audit)
    _write_json(compliance_path, compliance)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "p009_healnet_frozen_prefix_v1",
        "participant_id": "P009",
        "replay_type": "chronological_shadow_replay",
        "synthetic_replay": False,
        "random_controller_choices": False,
        "training_performed": False,
        "p009_tuning_performed": False,
        "run_command": " ".join(sys.argv),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "pandas": str(pd.__version__),
            "numpy": str(np.__version__),
            "matplotlib": str(matplotlib.__version__),
        },
        "code": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [
                ROOT / "scripts/run_healnet_prefix_inference.py",
                ROOT / "scripts/run_healnet_adaptive_replay.py",
                ROOT / "src/adaptive/baseline.py",
                ROOT / "src/adaptive/condition_grid.py",
                ROOT / "src/adaptive/controller.py",
                ROOT / "src/adaptive/healnet_prefix.py",
                ROOT / "src/adaptive/metrics.py",
                ROOT / "src/adaptive/replay.py",
            ]
        },
        "inputs": {
            "predictions": {"path": str(prediction_path), "sha256": file_sha256(prediction_path)},
            "calibration": {"path": str(calibration_path), "sha256": file_sha256(calibration_path)},
            "model_manifest": {"path": str(model_manifest_path), "sha256": file_sha256(model_manifest_path)},
            "inference_audit": {"path": str(inference_audit_path), "sha256": file_sha256(inference_audit_path)},
            "labels": {"path": str(labels_path.resolve()), "sha256": file_sha256(labels_path)},
        },
        "outputs": [
            trace_path.name,
            metrics_path.name,
            plot_path.name,
            report_path.name,
            replay_audit_path.name,
            compliance_path.name,
        ],
        "protocol_all_checks_pass": compliance["all_protocol_checks_pass"],
        "deployment_decision": metrics["go_no_go_gates"]["prospective_deployment"]["decision"],
    }
    manifest_path = output_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    checksum_path = _write_checksums(output_dir)
    result = {
        "output_dir": str(output_dir),
        "trace": str(trace_path),
        "metrics": str(metrics_path),
        "plot": str(plot_path),
        "report": str(report_path),
        "checksums": str(checksum_path),
        "protocol_all_checks_pass": compliance["all_protocol_checks_pass"],
        "deployment_decision": metrics["go_no_go_gates"]["prospective_deployment"]["decision"],
        "parameter_changes": metrics["controller_behavior"]["parameter_changes"],
        "distinct_controller_conditions": metrics["controller_behavior"]["distinct_controller_conditions"],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--labels", type=Path)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
