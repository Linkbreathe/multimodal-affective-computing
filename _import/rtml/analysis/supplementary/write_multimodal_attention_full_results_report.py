from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "reports" / "current_multimodal_attention_full_results_2026-07-04_zh.md"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        if value == "nan":
            return ""
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 1000:
        return f"{number:.1f}"
    return f"{number:.{digits}f}"


def _md_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        frame = frame.loc[:, columns].copy()
    else:
        frame = frame.copy()
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _prefix_counts(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prefix in ["eeg_", "ecg_", "head_", "eye_", "video_", "a4_"]:
        rows.append({"prefix": prefix, "n_columns": sum(str(column).startswith(prefix) for column in frame.columns)})
    rows.extend(
        [
            {"prefix": "all_columns", "n_columns": len(frame.columns)},
            {"prefix": "rows", "n_columns": len(frame)},
        ]
    )
    return pd.DataFrame(rows)


def _target_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["relaxation", "pleasantness", "calm", "discomfort"]:
        values = pd.to_numeric(frame[target], errors="coerce")
        rows.append(
            {
                "target": target,
                "n": int(values.notna().sum()),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "median": values.median(),
                "max": values.max(),
                "n_unique": int(values.nunique(dropna=True)),
            }
        )
    high = frame["discomfort"] >= 0.5
    rows.append(
        {
            "target": "high_discomfort>=0.50",
            "n": int(high.sum()),
            "mean": float(high.mean()),
            "std": None,
            "min": None,
            "median": None,
            "max": None,
            "n_unique": 2,
        }
    )
    return pd.DataFrame(rows)


def _state_model_summary() -> pd.DataFrame:
    bundle = joblib.load(ROOT / "artifacts" / "models" / "state_model.joblib")
    columns = bundle["feature_columns"]
    return pd.DataFrame(
        [
            {"field": "runtime_backend", "value": "classical"},
            {"field": "policy_shadow", "value": True},
            {"field": "model_kind", "value": bundle.get("model_kind")},
            {"field": "model_variant", "value": bundle.get("model_variant")},
            {"field": "deployable", "value": bundle.get("deployable")},
            {"field": "feature_columns", "value": len(columns)},
            {"field": "video_feature_columns", "value": sum(str(column).startswith("video_") for column in columns)},
            {"field": "a4_feature_columns", "value": sum(str(column).startswith("a4_") for column in columns)},
        ]
    )


def _four_channel_continuous_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "record_type",
            "baseline",
            "combination",
            "target",
            "spearman",
            "within_participant_spearman_mean",
            "within_participant_spearman_median",
            "pairwise_ranking_accuracy",
            "top1_condition_hit",
            "top3_condition_hit",
            "top1_regret",
        ]
    ].copy()


def _binary_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "record_type",
            "baseline",
            "combination",
            "target",
            "roc_auc",
            "pr_auc",
            "recall",
            "precision",
            "f1",
            "false_negatives",
            "positives_predicted",
            "threshold",
            "positive_labels",
        ]
    ].copy()


def _best_four_channel(fc_cont: pd.DataFrame, fc_bin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in fc_cont.groupby("target", sort=True):
        baseline = group.loc[group["record_type"].eq("baseline")].sort_values(
            "within_participant_spearman_mean",
            ascending=False,
        ).iloc[0]
        best_combo = group.loc[group["record_type"].eq("combination")].sort_values(
            ["within_participant_spearman_mean", "pairwise_ranking_accuracy"],
            ascending=False,
        ).iloc[0]
        rows.extend(
            [
                {
                    "protocol": "A4 ranking",
                    "target": target,
                    "selected": "best_baseline",
                    "name": baseline["baseline"],
                    "within": baseline["within_participant_spearman_mean"],
                    "pairwise": baseline["pairwise_ranking_accuracy"],
                    "top1": baseline["top1_condition_hit"],
                    "note": "baseline",
                },
                {
                    "protocol": "A4 ranking",
                    "target": target,
                    "selected": "best_combination",
                    "name": best_combo["combination"],
                    "within": best_combo["within_participant_spearman_mean"],
                    "pairwise": best_combo["pairwise_ranking_accuracy"],
                    "top1": best_combo["top1_condition_hit"],
                    "note": "best by within then pairwise",
                },
            ]
        )
    binary_group = fc_bin.loc[fc_bin["record_type"].eq("combination")]
    best_binary = binary_group.sort_values(["pr_auc", "roc_auc"], ascending=False).iloc[0]
    rows.append(
        {
            "protocol": "A4 binary",
            "target": "high_discomfort",
            "selected": "best_combination",
            "name": best_binary["combination"],
            "within": None,
            "pairwise": None,
            "top1": None,
            "note": f"PR-AUC={best_binary['pr_auc']:.4f}, recall={best_binary['recall']:.4f}, FN={best_binary['false_negatives']:.0f}",
        }
    )
    return pd.DataFrame(rows)


def _best_latent(latent_k: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, group in latent_k.loc[latent_k["task"].isin(["relaxation", "pleasantness", "calm"])].groupby(
        "task",
        sort=True,
    ):
        best = group.sort_values(["within", "pairwise"], ascending=False).iloc[0]
        rows.append(
            {
                "protocol": "latent k-sensitivity",
                "target": target,
                "k": int(best["k"]),
                "combo": best["combo"],
                "within": best["within"],
                "pairwise": best["pairwise"],
                "top1": best["top1"],
                "delta_within_vs_condition": best["delta_within_vs_cond"],
                "selection": "best by within then pairwise",
            }
        )
    for task, group in latent_k.loc[latent_k["task"].str.startswith("high_discomfort")].groupby("task", sort=True):
        best = group.sort_values(["pr_auc", "roc_auc"], ascending=False).iloc[0]
        rows.append(
            {
                "protocol": "latent k-sensitivity",
                "target": task,
                "k": int(best["k"]),
                "combo": best["combo"],
                "within": None,
                "pairwise": None,
                "top1": None,
                "delta_within_vs_condition": None,
                "selection": f"best PR-AUC={best['pr_auc']:.4f}, ROC={best['roc_auc']:.4f}, recall={best['recall']:.4f}, FN={best['fn']:.0f}",
            }
        )
    return pd.DataFrame(rows)


def _best_warm_start(metrics: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_rows = []
    for (target, prefix), group in metrics.groupby(["target", "prefix_count"], sort=True):
        best = group.sort_values(["pairwise_accuracy", "within_spearman"], ascending=False).iloc[0]
        baseline = group.loc[group["predictor"].eq("condition_eb")].iloc[0]
        best_rows.append(
            {
                "target": target,
                "prefix_count": int(prefix),
                "best_model_source": best["model_source"],
                "best_predictor": best["predictor"],
                "best_pairwise": best["pairwise_accuracy"],
                "best_within": best["within_spearman"],
                "condition_eb_pairwise": baseline["pairwise_accuracy"],
                "condition_eb_within": baseline["within_spearman"],
            }
        )
    comparison_rows = comparisons.loc[
        comparisons["target"].eq("relaxation")
        & comparisons["metric"].eq("pairwise_accuracy")
        & comparisons["predictor"].eq("model_plus_condition_eb")
    ].sort_values("delta_mean_positive_is_better", ascending=False)
    return pd.DataFrame(best_rows), comparison_rows


def _best_pairwise(metrics: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_rows = []
    for prefix, group in metrics.groupby("prefix_count", sort=True):
        best = group.sort_values(["pairwise_accuracy", "within_spearman"], ascending=False).iloc[0]
        baseline = group.loc[group["predictor"].eq("condition_only")].iloc[0]
        best_rows.append(
            {
                "prefix_count": int(prefix),
                "best_combination": best["combination"],
                "best_predictor": best["predictor"],
                "best_pairwise": best["pairwise_accuracy"],
                "best_within": best["within_spearman"],
                "condition_only_pairwise": baseline["pairwise_accuracy"],
                "condition_only_within": baseline["within_spearman"],
            }
        )
    comparison_rows = comparisons.loc[
        comparisons["metric"].eq("pairwise_accuracy")
        & comparisons["prefix_count"].eq(3)
    ].sort_values("delta_mean_positive_is_better", ascending=False)
    return pd.DataFrame(best_rows), comparison_rows


def _manifest_table(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[
        [
            "participant_id",
            "n_video_frames",
            "video_duration_s",
            "tensor_shape",
            "preview_video_bytes",
            "rgb_crop_video_bytes",
            "attention_mask_video_bytes",
            "four_channel_tensor_npz_bytes",
        ]
    ].copy()


def _file_inventory() -> pd.DataFrame:
    paths = [
        ROOT / "artifacts" / "features" / "ecg_neurokit" / "condition_features.csv",
        ROOT / "artifacts" / "fusion_four_channel_ranking" / "a4_condition_features.csv",
        ROOT / "artifacts" / "fusion_four_channel_ranking" / "four_channel_video_manifest.csv",
        ROOT / "artifacts" / "fusion_four_channel_ranking" / "continuous_ranking_metrics.csv",
        ROOT / "artifacts" / "fusion_four_channel_ranking" / "binary_high_discomfort_metrics.csv",
        ROOT / "artifacts" / "fusion_four_channel_ranking" / "head_to_head_continuous.csv",
        ROOT / "artifacts" / "fusion_four_channel_ranking" / "head_to_head_binary.csv",
        ROOT / "artifacts" / "fusion_four_channel_ranking" / "oof_predictions.csv",
        ROOT / "artifacts" / "fusion_latent_multitask" / "latent_k_sensitivity_summary.csv",
        ROOT / "artifacts" / "fusion_latent_multitask" / "latent_loading_summary.csv",
        ROOT / "artifacts" / "fusion_latent_multitask" / "latent_oof_predictions.csv",
        ROOT / "artifacts" / "personalized_warm_start_v_k4" / "feature_variance_decomposition.csv",
        ROOT / "artifacts" / "personalized_warm_start_v_k4" / "prediction_personalization_diagnostics.csv",
        ROOT / "artifacts" / "personalized_warm_start_v_k4" / "prediction_within_condition_permutation.csv",
        ROOT / "artifacts" / "personalized_warm_start_v_k4" / "warm_start_metrics.csv",
        ROOT / "artifacts" / "personalized_warm_start_v_k4" / "warm_start_comparisons.csv",
        ROOT / "artifacts" / "personalized_warm_start_v_k4" / "warm_start_predictions.csv",
        ROOT / "artifacts" / "pairwise_ranker_warm_start" / "pairwise_metrics.csv",
        ROOT / "artifacts" / "pairwise_ranker_warm_start" / "pairwise_comparisons.csv",
        ROOT / "artifacts" / "pairwise_ranker_warm_start" / "pairwise_predictions.csv",
        ROOT / "artifacts" / "pairwise_ranker_warm_start" / "feature_selection_summary.csv",
    ]
    rows = []
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path)
            rows.append(
                {
                    "file": path.relative_to(ROOT).as_posix(),
                    "rows": len(frame),
                    "columns": len(frame.columns),
                    "bytes": path.stat().st_size,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    base = pd.read_csv(ROOT / "artifacts" / "features" / "ecg_neurokit" / "condition_features.csv")
    a4 = pd.read_csv(ROOT / "artifacts" / "fusion_four_channel_ranking" / "a4_condition_features.csv")
    manifest = pd.read_csv(ROOT / "artifacts" / "fusion_four_channel_ranking" / "four_channel_video_manifest.csv")
    fc_cont = pd.read_csv(ROOT / "artifacts" / "fusion_four_channel_ranking" / "continuous_ranking_metrics.csv")
    fc_bin = pd.read_csv(ROOT / "artifacts" / "fusion_four_channel_ranking" / "binary_high_discomfort_metrics.csv")
    fc_h2h_cont = pd.read_csv(ROOT / "artifacts" / "fusion_four_channel_ranking" / "head_to_head_continuous.csv")
    fc_h2h_bin = pd.read_csv(ROOT / "artifacts" / "fusion_four_channel_ranking" / "head_to_head_binary.csv")
    latent_k = pd.read_csv(ROOT / "artifacts" / "fusion_latent_multitask" / "latent_k_sensitivity_summary.csv")
    latent_load = pd.read_csv(ROOT / "artifacts" / "fusion_latent_multitask" / "latent_loading_summary.csv")
    feat_var = pd.read_csv(ROOT / "artifacts" / "personalized_warm_start_v_k4" / "feature_variance_decomposition.csv")
    pred_diag = pd.read_csv(ROOT / "artifacts" / "personalized_warm_start_v_k4" / "prediction_personalization_diagnostics.csv")
    pred_perm = pd.read_csv(ROOT / "artifacts" / "personalized_warm_start_v_k4" / "prediction_within_condition_permutation.csv")
    warm_metrics = pd.read_csv(ROOT / "artifacts" / "personalized_warm_start_v_k4" / "warm_start_metrics.csv")
    warm_comparisons = pd.read_csv(ROOT / "artifacts" / "personalized_warm_start_v_k4" / "warm_start_comparisons.csv")
    pair_metrics = pd.read_csv(ROOT / "artifacts" / "pairwise_ranker_warm_start" / "pairwise_metrics.csv")
    pair_comparisons = pd.read_csv(ROOT / "artifacts" / "pairwise_ranker_warm_start" / "pairwise_comparisons.csv")
    pair_selection = pd.read_csv(ROOT / "artifacts" / "pairwise_ranker_warm_start" / "feature_selection_summary.csv")

    warm_best, warm_relax_delta = _best_warm_start(warm_metrics, warm_comparisons)
    pair_best, pair_m3_delta = _best_pairwise(pair_metrics, pair_comparisons)

    lines = [
        "# Multimodal attention/video 全量结果汇总",
        "",
        "生成日期：2026-07-04",
        "",
        "这是一份单一总文档：所有关键结果、比较、具体数据表都直接放在本文中。本文不要求再跳到其它报告查看结论。",
        "",
        "## 0. 总结论",
        "",
        "当前已经完成 A4/heatmap video、four-channel ranking、latent multi-task、V-first warm-start、direct pairwise ranker 与 residual-pairwise ranker。综合最新与最好结果，最强可保留信号是 `V k=4` 的 relaxation latent 排序突破，以及 `PHEV` 在 direct pairwise ranker 中的弱正向趋势。但这些都还没有稳定通过 participant-level uncertainty，因此不能称为可部署模型。",
        "",
        "## 1. 当前 runtime 与监督单位",
        "",
        _md_table(_state_model_summary()),
        "",
        "监督单位仍是 `participant-condition`，共 `15 x 9 = 135` 行。10 秒 window 只用于特征聚合，不是独立标签样本。",
        "",
        "## 2. 输入数据与标签",
        "",
        "### 2.1 Base feature table 列数",
        "",
        _md_table(_prefix_counts(base)),
        "",
        "### 2.2 A4 feature table 列数",
        "",
        _md_table(_prefix_counts(a4)),
        "",
        "### 2.3 标签分布",
        "",
        _md_table(_target_distribution(base)),
        "",
        "## 3. 当前各实验的最佳点",
        "",
        "### 3.1 Four-channel / A4 benchmark 最佳点",
        "",
        _md_table(_best_four_channel(fc_cont, fc_bin)),
        "",
        "### 3.2 Latent multi-task 最佳点",
        "",
        _md_table(_best_latent(latent_k)),
        "",
        "### 3.3 V-first warm-start 每个 target/prefix 的最好点",
        "",
        _md_table(warm_best),
        "",
        "### 3.4 V-first warm-start relaxation pairwise delta 排名",
        "",
        _md_table(warm_relax_delta[
            [
                "model_source",
                "prefix_count",
                "predictor",
                "delta_mean_positive_is_better",
                "delta_ci_low",
                "delta_ci_high",
                "sign_flip_p_value",
            ]
        ]),
        "",
        "### 3.5 Direct pairwise ranker 每个 prefix 的最好点",
        "",
        _md_table(pair_best),
        "",
        "### 3.6 Direct pairwise ranker m=3 pairwise delta 排名",
        "",
        _md_table(pair_m3_delta[
            [
                "combination",
                "predictor",
                "delta_mean_positive_is_better",
                "delta_ci_low",
                "delta_ci_high",
                "sign_flip_p_value",
            ]
        ]),
        "",
        "## 4. Four-channel artifact 明细",
        "",
        _md_table(_manifest_table(manifest)),
        "",
        "## 5. Four-channel A4 ranking/classification 全部结果",
        "",
        "### 5.1 Continuous ranking metrics",
        "",
        _md_table(_four_channel_continuous_table(fc_cont)),
        "",
        "### 5.2 High-discomfort binary metrics",
        "",
        _md_table(_binary_table(fc_bin)),
        "",
        "### 5.3 Continuous head-to-head",
        "",
        _md_table(fc_h2h_cont),
        "",
        "### 5.4 Binary head-to-head",
        "",
        _md_table(fc_h2h_bin),
        "",
        "## 6. Latent multi-task 全部结果",
        "",
        "### 6.1 Latent loading summary",
        "",
        _md_table(latent_load),
        "",
        "### 6.2 k sensitivity summary",
        "",
        _md_table(latent_k),
        "",
        "## 7. Feature variance 与 V-first warm-start 全部结果",
        "",
        "### 7.1 Feature variance decomposition",
        "",
        _md_table(feat_var),
        "",
        "### 7.2 Prediction personalization diagnostics",
        "",
        _md_table(pred_diag),
        "",
        "### 7.3 Within-condition prediction permutation",
        "",
        _md_table(pred_perm),
        "",
        "### 7.4 Warm-start metrics",
        "",
        _md_table(warm_metrics[
            [
                "model_source",
                "target",
                "prefix_count",
                "predictor",
                "n_eval_conditions",
                "mae",
                "within_spearman",
                "pairwise_accuracy",
                "top1_hit",
                "top1_regret",
            ]
        ]),
        "",
        "### 7.5 Warm-start comparisons",
        "",
        _md_table(warm_comparisons[
            [
                "model_source",
                "target",
                "prefix_count",
                "predictor",
                "metric",
                "delta_mean_positive_is_better",
                "delta_ci_low",
                "delta_ci_high",
                "sign_flip_p_value",
            ]
        ]),
        "",
        "## 8. Direct pairwise / residual pairwise ranker 全部结果",
        "",
        "### 8.1 Pairwise metrics",
        "",
        _md_table(pair_metrics[
            [
                "combination",
                "prefix_count",
                "predictor",
                "n_eval_conditions",
                "within_spearman",
                "pairwise_accuracy",
                "top1_hit",
                "top1_regret",
            ]
        ]),
        "",
        "### 8.2 Pairwise comparisons",
        "",
        _md_table(pair_comparisons[
            [
                "combination",
                "prefix_count",
                "predictor",
                "metric",
                "delta_mean_positive_is_better",
                "delta_ci_low",
                "delta_ci_high",
                "sign_flip_p_value",
            ]
        ]),
        "",
        "### 8.3 Pairwise feature selection",
        "",
        _md_table(pair_selection),
        "",
        "## 9. 文件行数与列数清单",
        "",
        _md_table(_file_inventory()),
        "",
        "## 10. 最终判断",
        "",
        "- `A4` 已完成生成与评估，但当前不是最强预测信号。",
        "- `V k=4` 是 latent multi-task 中 relaxation 连续排序最强点：within `0.1951`，pairwise `0.5917`。",
        "- warm-start 中 `V_k4 + personalized baseline` 在 m=3 的 pairwise 从 `0.6129` 到 `0.6513`，delta `+0.0384`，但 CI `[-0.0621, 0.1354]`，不稳定。",
        "- direct pairwise ranker 的主终点 `V/m=3` 只有 `+0.0049`，CI `[-0.0968, 0.1009]`，没有解决问题。",
        "- direct pairwise ranker 中 `PHEV/m=3` 是 secondary 最好趋势之一，delta `+0.0259`，但 CI `[-0.1245, 0.1507]`。",
        "- 当前不能部署，也不能声称已经稳定击败个性化 baseline。",
        "",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
