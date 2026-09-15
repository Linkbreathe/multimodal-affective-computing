"""Build the final Chinese report and compact figures for the Relax ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_LABELS = {
    "s0_legacy": "S0 legacy",
    "s1_reve_native_raw4": "S1 REVE-native/raw4",
    "s2_reve_native_linked2": "S2 REVE/linked2",
    "s3_reve_clean_linked2": "S3 clean REVE",
    "s4_neurorvq_clean_z": "S4 NeuroRVQ/z",
    "s5_neurorvq_clean_native": "S5 NeuroRVQ/native",
    "s6_neurorvq_eeg_ecg": "S6 NeuroRVQ EEG+ECG",
    "r3_reve_official_session": "R3 REVE/session",
    "ecg_only_neurorvq": "E1 NeuroRVQ-ECG only",
}
DEFAULT_ROOT = ROOT / "artifacts/relax/neurorvq_embedding_ladder_20260718_corrected"


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def _signed(value: float, digits: int = 6) -> str:
    return f"{float(value):+.{digits}f}"


def _comparison_row(frame: pd.DataFrame, comparison: str, outcome: str) -> pd.Series:
    selected = frame.loc[
        (frame["comparison"] == comparison) & (frame["outcome"] == outcome)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one comparison row for {comparison}/{outcome}")
    return selected.iloc[0]


def _plot_stage_metrics(stage: pd.DataFrame, output: Path) -> None:
    labels = [STAGE_LABELS[value] for value in stage["stage"]]
    x = np.arange(len(stage))
    figure, axis = plt.subplots(figsize=(12.5, 5.2))
    axis.errorbar(
        x,
        stage["macro_mae_mean"],
        yerr=stage["macro_mae_sd"],
        marker="o",
        linewidth=2,
        capsize=3,
        color="#2457C5",
        label="Fusion Macro MAE",
    )
    axis.plot(
        x,
        stage["macro_condition_mae_mean"],
        linestyle="--",
        color="#777777",
        label="Condition-only",
    )
    axis.set_xticks(x, labels, rotation=32, ha="right")
    axis.set_ylabel("Participant-macro Macro MAE (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_deltas(comparisons: pd.DataFrame, output: Path) -> None:
    macro = comparisons.loc[comparisons["outcome"] == "macro"].copy()
    order = [
        "primary_s5_minus_s3",
        "s1_minus_s0",
        "s2_minus_s1",
        "s3_minus_s2",
        "s4_minus_s3",
        "s5_minus_s4",
        "s6_minus_s5",
        "r3_minus_s3",
        "ecg_only_minus_s3",
    ]
    macro = macro.set_index("comparison").loc[order].reset_index()
    y = np.arange(len(macro))
    low = macro["delta_mean"] - macro["participant_bootstrap_ci_low"]
    high = macro["participant_bootstrap_ci_high"] - macro["delta_mean"]
    colors = [
        "#C73E1D" if family == "primary" else "#2457C5" if family == "sequential" else "#6D6D6D"
        for family in macro["family"]
    ]
    figure, axis = plt.subplots(figsize=(9.2, 6.0))
    for index, color in enumerate(colors):
        axis.errorbar(
            float(macro.loc[index, "delta_mean"]),
            y[index],
            xerr=np.asarray([[low.iloc[index]], [high.iloc[index]]]),
            fmt="none",
            ecolor=color,
            capsize=3,
            linewidth=2,
        )
    axis.scatter(macro["delta_mean"], y, c=colors, s=48, zorder=3)
    axis.axvline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(y, macro["comparison"])
    axis.invert_yaxis()
    axis.set_xlabel("Paired Macro MAE delta (new - reference; negative is better)")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build(args: argparse.Namespace) -> Path:
    evaluation_dir = args.root / "evaluation"
    summary = json.loads((evaluation_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    stage = pd.read_csv(evaluation_dir / "stage_metrics.csv")
    comparisons = pd.read_csv(evaluation_dir / "paired_comparisons.csv")
    weights = pd.read_csv(evaluation_dir / "expert_weights.csv")
    discomfort = pd.read_csv(evaluation_dir / "discomfort_strata.csv")
    embedding_qc = pd.read_csv(args.root / "audit/embedding_numerical_qc.csv")
    signal_qc = pd.read_csv(args.root / "audit/signal_qc_summary.csv")
    audit = json.loads((args.root / "audit/audit_summary.json").read_text(encoding="utf-8"))

    figures = args.root / "figures"
    _plot_stage_metrics(stage, figures / "stage_macro_mae.png")
    _plot_deltas(comparisons, figures / "paired_macro_deltas.png")

    primary_macro = _comparison_row(comparisons, "primary_s5_minus_s3", "macro")
    primary_relaxation = _comparison_row(comparisons, "primary_s5_minus_s3", "relaxation")
    primary_discomfort = _comparison_row(comparisons, "primary_s5_minus_s3", "discomfort")
    s3 = stage.set_index("stage").loc["s3_reve_clean_linked2"]
    s5 = stage.set_index("stage").loc["s5_neurorvq_clean_native"]
    best = stage.sort_values("macro_mae_mean").iloc[0]
    relative = 100.0 * primary_macro["delta_mean"] / s3["macro_mae_mean"]
    direction = "改善" if primary_macro["delta_mean"] < 0 else "变差"
    evidence = (
        "较强支持（95% cluster-bootstrap CI 完全低于 0）"
        if summary["success_rules"]["stronger_support_pass"]
        else "尚无较强统计支持"
    )

    stage_rows = []
    for row in stage.itertuples(index=False):
        stage_rows.append(
            "| {label} | {eeg} | {ecg} | {relax} | {discomfort} | {macro} | {delta} |".format(
                label=STAGE_LABELS[row.stage],
                eeg=row.eeg_dimension,
                ecg=row.ecg_dimension,
                relax=_fmt(row.relaxation_mae_mean),
                discomfort=_fmt(row.discomfort_mae_mean),
                macro=_fmt(row.macro_mae_mean),
                delta=_signed(row.macro_delta_vs_condition_mean),
            )
        )

    comparison_rows = []
    macro_rows = comparisons.loc[comparisons["outcome"] == "macro"]
    for row in macro_rows.itertuples(index=False):
        adjusted = "—" if pd.isna(row.holm_adjusted_p) else _fmt(row.holm_adjusted_p, 4)
        comparison_rows.append(
            f"| {row.comparison} | {_signed(row.delta_mean)} | "
            f"[{_signed(row.participant_bootstrap_ci_low)}, {_signed(row.participant_bootstrap_ci_high)}] | "
            f"{_fmt(row.exact_sign_flip_p_two_sided, 4)} | {adjusted} | "
            f"{row.participants_improved}/{row.participants_tied}/{row.participants_worsened} |"
        )

    selected_qc = embedding_qc.loc[
        ((embedding_qc["stage"] == "s3_reve_clean_linked2") & (embedding_qc["modality"] == "eeg"))
        | ((embedding_qc["stage"] == "s4_neurorvq_clean_z") & (embedding_qc["modality"] == "eeg"))
        | ((embedding_qc["stage"] == "s5_neurorvq_clean_native") & (embedding_qc["modality"] == "eeg"))
        | ((embedding_qc["stage"] == "s6_neurorvq_eeg_ecg") & (embedding_qc["modality"] == "ecg"))
    ]
    qc_rows = [
        f"| {STAGE_LABELS[row.stage]} | {row.modality} | {int(row.dimension)} | "
        f"{int(row.exact_unique_vectors)}/{int(row.vectors)} | {row.entropy_effective_rank:.2f} | "
        f"{row.coordinate_std_median:.6f} |"
        for row in selected_qc.itertuples(index=False)
    ]

    weight_summary = (
        weights.loc[weights["stage"].isin(["s3_reve_clean_linked2", "s5_neurorvq_clean_native", "s6_neurorvq_eeg_ecg"])]
        .groupby(["stage", "target", "modality"], sort=True)["weight"]
        .agg(["mean", "std"])
        .reset_index()
    )
    weight_rows = [
        f"| {STAGE_LABELS[row.stage]} | {row.target} | {row.modality} | {row.mean:.4f} ± {row.std:.4f} |"
        for row in weight_summary.itertuples(index=False)
    ]

    strata = (
        discomfort.groupby(["stage", "stratum"], sort=True)
        .agg(observations=("observations", "first"), mae=("mae", "mean"))
        .reset_index()
    )
    strata = strata.loc[strata["stage"].isin(["s3_reve_clean_linked2", "s5_neurorvq_clean_native", "s6_neurorvq_eeg_ecg"])]
    strata_rows = [
        f"| {STAGE_LABELS[row.stage]} | {row.stratum} | {int(row.observations)} | {row.mae:.6f} |"
        for row in strata.itertuples(index=False)
    ]

    native_signal = signal_qc.set_index("stage_signal").loc["clean2_native_uv"]
    ecg_signal = signal_qc.set_index("stage_signal").loc["ecg_lead_i_like_native_mv"]
    lines = [
        "# Relax Foundation-Embedding Ladder 最终报告",
        "",
        "> 仅使用本仓库 `data/datasets/relaxdata` 中的 XDF 原始信号。正式结论来自 corrected root；首次随机 REVE position-buffer 尝试已在相邻目录中明确标记为 invalidated，未进入 fusion。",
        "",
        "## 核心结论",
        "",
        f"- 预注册主比较 S5（NeuroRVQ EEG、native scaling）相对 S3（clean REVE）的 Macro MAE 从 `{_fmt(s3['macro_mae_mean'])}` 到 `{_fmt(s5['macro_mae_mean'])}`，差值 `{_signed(primary_macro['delta_mean'])}`（约 `{relative:+.2f}%`），方向为**{direction}**。",
        f"- participant-cluster 95% CI 为 `[{_signed(primary_macro['participant_bootstrap_ci_low'])}, {_signed(primary_macro['participant_bootstrap_ci_high'])}]`，精确 sign-flip `p={primary_macro['exact_sign_flip_p_two_sided']:.4f}`；判定：**{evidence}**。",
        f"- relaxation 差值 `{_signed(primary_relaxation['delta_mean'])}`；discomfort 差值 `{_signed(primary_discomfort['delta_mean'])}`。预注册 descriptive rule：`{summary['success_rules']['descriptive_pass']}`。",
        f"- 全部 stage 中 Macro MAE 最低的是 **{STAGE_LABELS[best['stage']]}**：`{_fmt(best['macro_mae_mean'])}`。这只作为固定 ladder 内的描述性排序，不替代主比较。",
        "",
        "![各阶段 Macro MAE](figures/stage_macro_mae.png)",
        "",
        "![配对 Macro MAE 差值](figures/paired_macro_deltas.png)",
        "",
        "## 输入通道与预处理",
        "",
        "- Relax XDF 的四个 EEG 数值列按 `[M2, TP9, TP10, M1]` 解释。S0/S1 沿用历史实现，把这 4 路直接送入 REVE；从 S2 起先做 linked-mastoid 重参考：`R=(M1+M2)/2`、`TP9'=TP9-R`、`TP10'=TP10-R`，随后仅把 2 路 `TP9'/TP10'` 送入 encoder。M1/M2 因而是参考电极，不再作为脑信号通道输入。",
        "- NeuroRVQ EEG 的官方 channel vocabulary 有 `TP9/TP10`（也有 `A1/A2`），但没有字面名称 `M1/M2`；REVE position bank 虽能解析 `M1/M2`，这不意味着重参考后仍应把它们作为独立输入。",
        "- S3/S4/S5 的公共 EEG 清洗顺序是：50/60/100 Hz notch（`Q=f/2`）→ 三阶 0.5–45 Hz Butterworth → clip 到 `±500 µV` → FFT resample 到 200 Hz。S3/S4 再做 window/channel z-score；S5 保留物理微伏尺度，以匹配 NeuroRVQ 官方示例。",
        "- R3 是 REVE 预训练风格的敏感性分析：0.5–99.5 Hz、session/channel z-score、clip `±15 SD`、200 Hz。因为使用整段 session 统计量，只能视为 offline/transductive 结果。",
        "- ECG 使用 Relax 的 `LA−RA` Lead-I-like 差分，`µV→mV` 后重采样到 200 Hz，再输入 NeuroRVQ ECG；没有套用 EEG 的 notch/band-pass/z-score 流程。",
        "",
        "## 每一步同一 fusion 的结果",
        "",
        "所有行均使用同一个 `modality_expert_simplex5`、同一 9-fold 7/1/1 split、同一三个 seed、同一 545-window 外部共同 mask。负的 ΔCondition 表示优于 Condition-only。",
        "",
        "| Stage | EEG dim | ECG dim | Relaxation MAE | Discomfort MAE | Macro MAE | ΔCondition |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *stage_rows,
        "",
        "## 配对推断",
        "",
        "差值定义为 new − reference，因此负值更好；I/T/W 分别是改善/持平/变差的 participant 数。",
        "",
        "| Comparison | Macro Δ | 95% participant-bootstrap CI | exact p | Holm p | I/T/W |",
        "|---|---:|---:|---:|---:|---:|",
        *comparison_rows,
        "",
        "## 表征与信号 QC",
        "",
        f"- S1 的 native 1216 维向量重新做旧的 adaptive pooling 后，与历史 S0 的平均余弦为 `{audit['legacy_pool_reconstruction']['cosine']['mean']:.9f}`，证明 S1 基本只隔离 pooling。",
        "- 所有正式 common-valid vectors 都有限、非全零；eye/head/video 的完整 tensor hash 在所有 cache 中相同。",
        f"- NeuroRVQ EEG 的有效秩明显低于 clean REVE；这是实际表征结构而非常数退化。",
        f"- clean native EEG 在重采样后绝对峰值中位数 `{native_signal['absolute_max_median']:.2f} µV`、P95 `{native_signal['absolute_max_p95']:.2f} µV`；先 clip 再 FFT-resample 会有轻微 Gibbs overshoot。",
        f"- ECG Lead-I-like native 输入绝对峰值中位数 `{ecg_signal['absolute_max_median']:.3f} mV`、P95 `{ecg_signal['absolute_max_p95']:.3f} mV`；全部 embedding 有限。",
        "",
        "| Stage | Modality | Dim | Unique/vectors | Effective rank | Median coordinate SD |",
        "|---|---|---:|---:|---:|---:|",
        *qc_rows,
        "",
        "## 专家权重（跨 3 seeds × 9 folds）",
        "",
        "| Stage | Target | Modality | Mean ± SD |",
        "|---|---|---|---:|",
        *weight_rows,
        "",
        "## Discomfort 分层",
        "",
        "| Stage | Stratum | N | MAE |",
        "|---|---|---:|---:|",
        *strata_rows,
        "",
        "## 如何据此决策",
        "",
        "- **当前不替换生产/主实验特征。** S0 仍是最低 Macro MAE；S1–S6 没有任何正式阶段超过它。",
        f"- S4（z-scored NeuroRVQ）相对 S3 描述性改善 `{_signed(_comparison_row(comparisons, 's4_minus_s3', 'macro')['delta_mean'])}`，但 CI 跨 0、Holm `p={_comparison_row(comparisons, 's4_minus_s3', 'macro')['holm_adjusted_p']:.4f}`，且仍比 S0 差 `{_signed(stage.set_index('stage').loc['s4_neurorvq_clean_z', 'macro_mae_mean'] - stage.set_index('stage').loc['s0_legacy', 'macro_mae_mean'])}`。它只能作为下一轮候选，不能称为提升。",
        f"- native scaling 从 S4 到 S5 使 Macro MAE 再增加 `{_signed(_comparison_row(comparisons, 's5_minus_s4', 'macro')['delta_mean'])}`；这说明本数据上的 NeuroRVQ 对幅值域很敏感，不能因为官方示例使用物理单位就假定迁移最优。",
        f"- NeuroRVQ ECG 呈明显 target trade-off：S6 相对 S5 的 discomfort MAE 下降 `{_signed(stage.set_index('stage').loc['s6_neurorvq_eeg_ecg', 'discomfort_mae_mean'] - stage.set_index('stage').loc['s5_neurorvq_clean_native', 'discomfort_mae_mean'])}`，但 relaxation MAE 上升 `{_signed(stage.set_index('stage').loc['s6_neurorvq_eeg_ecg', 'relaxation_mae_mean'] - stage.set_index('stage').loc['s5_neurorvq_clean_native', 'relaxation_mae_mean'])}`；不能用 Macro 掩盖这种方向相反的结果。",
        "- 若继续，最值得预注册的是：保持同一 fusion，在全新的外部参与者或严格 nested-CV 中测试 target-specific hybrid（relaxation 保留 legacy ECG，discomfort 才启用 NeuroRVQ ECG），并把 S4 作为 EEG 候选。该方案来自本次结果，属于 post-hoc 假设，不能在当前 9 人结果上再次调参后宣称验证成功。",
        "- 第二优先级才是改变 NeuroRVQ pooling：当前四分支 token mean 后 EEG 有效秩约 6，可能丢失时间结构；可预注册 `[mean, std, max]` 或仅在训练 fold 学 attention pooling。不要直接全模型微调，9 名 participant 对此过小。",
        "",
        "## 实验解释边界",
        "",
        "- 这是 9 位 participant、81 个 participant-condition 单元的内部探索验证；window 不是独立监督单元。",
        "- R3 使用 full-session statistics，属于 offline/transductive sensitivity，不能作为实时主结论。",
        "- NeuroRVQ checkpoint 官方 foundation forward 没有直接提供稳定的单向量 API；本实验严格固定为四分支 patch token 拼接后 token mean，并记录已知 missing/unexpected key allowlist。",
        "- 三个 seed 的线性流程通常完全确定；仍完整保留三次运行以遵循既有协议。",
        "",
        "## 上游来源",
        "",
        "- NeuroRVQ paper（本次阅读使用 v4，2026-05-17）：https://arxiv.org/abs/2510.13068",
        "- 官方代码（本次固定 commit `3d1d9e5f7ceae596cc72079610a45371b163c22a`）：https://github.com/KonstantinosBarmpas/NeuroRVQ",
        "- 官方权重：https://huggingface.co/ntinosbarmpas/NeuroRVQ",
        "- 本次 REVE encoder / position bank：https://huggingface.co/brain-bzh/reve-large 与 https://huggingface.co/brain-bzh/reve-positions",
        "",
        "## 可复现产物",
        "",
        f"- Embedding manifest: `{args.root / 'embedding_ladder_manifest.json'}`",
        f"- Fusion execution manifest: `{args.root / 'fusion/fusion_execution_manifest.json'}`",
        f"- Evaluation summary: `{evaluation_dir / 'evaluation_summary.json'}`",
        f"- Numerical audit: `{args.root / 'audit/audit_summary.json'}`",
        f"- Stage metrics: `{evaluation_dir / 'stage_metrics.csv'}`",
        f"- Paired comparisons: `{evaluation_dir / 'paired_comparisons.csv'}`",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output.resolve())
    return args.output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.output = args.output or args.root / "FINAL_REPORT_ZH.md"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
