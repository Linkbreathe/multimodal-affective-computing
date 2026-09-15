from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "reports" / "current_multimodal_attention_data_appendix_2026-07-04_zh.md"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
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


def _md_table(frame: pd.DataFrame, columns: list[str] | None = None, *, max_rows: int | None = None) -> str:
    if columns is not None:
        frame = frame.loc[:, columns].copy()
    else:
        frame = frame.copy()
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(cols) - 1)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _csv_info(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    return {
        "path": _rel(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": ", ".join(frame.columns),
        "bytes": int(path.stat().st_size),
    }


def _prefix_counts(frame: pd.DataFrame) -> pd.DataFrame:
    prefixes = ["eeg_", "ecg_", "head_", "eye_", "video_", "a4_"]
    rows = []
    for prefix in prefixes:
        rows.append({"prefix": prefix, "n_columns": sum(str(col).startswith(prefix) for col in frame.columns)})
    rows.append({"prefix": "total_columns", "n_columns": len(frame.columns)})
    rows.append({"prefix": "rows", "n_columns": len(frame)})
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
    rows.append(
        {
            "target": "high_discomfort>=0.50",
            "n": int((frame["discomfort"] >= 0.5).sum()),
            "mean": float((frame["discomfort"] >= 0.5).mean()),
            "std": float("nan"),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "n_unique": 2,
        }
    )
    return pd.DataFrame(rows)


def _state_model_summary() -> pd.DataFrame:
    path = ROOT / "artifacts" / "models" / "state_model.joblib"
    bundle = joblib.load(path)
    columns = bundle["feature_columns"]
    return pd.DataFrame(
        [
            {"field": "path", "value": _rel(path)},
            {"field": "model_kind", "value": bundle.get("model_kind")},
            {"field": "model_variant", "value": bundle.get("model_variant")},
            {"field": "deployable", "value": bundle.get("deployable")},
            {"field": "feature_columns", "value": len(columns)},
            {"field": "video_feature_columns", "value": sum(str(col).startswith("video_") for col in columns)},
            {"field": "a4_feature_columns", "value": sum(str(col).startswith("a4_") for col in columns)},
        ]
    )


def _manifest_summary(manifest: pd.DataFrame) -> pd.DataFrame:
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


def _metric_file_inventory() -> pd.DataFrame:
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
    ]
    return pd.DataFrame(_csv_info(path) for path in paths if path.exists())


def _simplify_four_channel_continuous(frame: pd.DataFrame) -> pd.DataFrame:
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


def _simplify_binary(frame: pd.DataFrame) -> pd.DataFrame:
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


def _warm_start_key_tables(metrics: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
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
    comparison_columns = [
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
    return metrics[metric_columns].copy(), comparisons[comparison_columns].copy()


def _pairwise_key_tables(metrics: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "combination",
        "prefix_count",
        "predictor",
        "n_eval_conditions",
        "within_spearman",
        "pairwise_accuracy",
        "top1_hit",
        "top1_regret",
    ]
    comparison_columns = [
        "combination",
        "prefix_count",
        "predictor",
        "metric",
        "delta_mean_positive_is_better",
        "delta_ci_low",
        "delta_ci_high",
        "sign_flip_p_value",
    ]
    return metrics[metric_columns].copy(), comparisons[comparison_columns].copy()


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

    warm_metric_table, warm_comparison_table = _warm_start_key_tables(warm_metrics, warm_comparisons)
    pair_metric_table, pair_comparison_table = _pairwise_key_tables(pair_metrics, pair_comparisons)
    inventory = _metric_file_inventory()
    raw_prediction_inventory = inventory.loc[
        inventory["path"].str.contains("prediction", case=False) | inventory["path"].str.contains("oof", case=False)
    ].copy()

    lines = [
        "# Multimodal attention/video 数据附录",
        "",
        "生成日期：2026-07-04",
        "",
        "这份附录把当前阶段用到的汇总数据完整列出。逐行 OOF/prediction 文件只列路径、行数、列名；完整逐行数据保留在对应 CSV。",
        "",
        "## 1. 文件清单",
        "",
        _md_table(inventory[["path", "rows", "columns", "bytes"]]),
        "",
        "## 2. 逐行预测文件清单",
        "",
        _md_table(raw_prediction_inventory[["path", "rows", "columns", "bytes"]]),
        "",
        "## 3. 当前 runtime state model",
        "",
        _md_table(_state_model_summary()),
        "",
        "## 4. Base feature table",
        "",
        "### 4.1 特征列数量",
        "",
        _md_table(_prefix_counts(base)),
        "",
        "### 4.2 标签分布",
        "",
        _md_table(_target_distribution(base)),
        "",
        "## 5. A4 / four-channel artifact 数据",
        "",
        "### 5.1 A4 特征列数量",
        "",
        _md_table(_prefix_counts(a4)),
        "",
        "### 5.2 每个 participant 的 four-channel artifact",
        "",
        _md_table(_manifest_summary(manifest)),
        "",
        "## 6. Four-channel A4 ranking/classification benchmark",
        "",
        "### 6.1 Continuous ranking metrics 全表",
        "",
        _md_table(_simplify_four_channel_continuous(fc_cont)),
        "",
        "### 6.2 High-discomfort binary metrics 全表",
        "",
        _md_table(_simplify_binary(fc_bin)),
        "",
        "### 6.3 Continuous head-to-head 全表",
        "",
        _md_table(fc_h2h_cont),
        "",
        "### 6.4 Binary head-to-head 全表",
        "",
        _md_table(fc_h2h_bin),
        "",
        "## 7. Latent multi-task 数据",
        "",
        "### 7.1 Latent loading summary",
        "",
        _md_table(latent_load),
        "",
        "### 7.2 k sensitivity summary 全表",
        "",
        _md_table(latent_k),
        "",
        "## 8. Feature variance / V-first warm-start 数据",
        "",
        "### 8.1 Feature variance decomposition",
        "",
        _md_table(feat_var),
        "",
        "### 8.2 Prediction personalization diagnostics",
        "",
        _md_table(pred_diag),
        "",
        "### 8.3 Within-condition prediction permutation",
        "",
        _md_table(pred_perm),
        "",
        "### 8.4 Warm-start metrics 全表",
        "",
        _md_table(warm_metric_table),
        "",
        "### 8.5 Warm-start comparisons 全表",
        "",
        _md_table(warm_comparison_table),
        "",
        "## 9. Direct pairwise ranker 数据",
        "",
        "### 9.1 Pairwise metrics 全表",
        "",
        _md_table(pair_metric_table),
        "",
        "### 9.2 Pairwise comparisons 全表",
        "",
        _md_table(pair_comparison_table),
        "",
        "### 9.3 Feature selection summary 全表",
        "",
        _md_table(pair_selection),
        "",
        "## 10. Run summaries",
        "",
    ]

    for path in [
        ROOT / "artifacts" / "fusion_four_channel_ranking" / "run_summary.json",
        ROOT / "artifacts" / "fusion_latent_multitask" / "run_summary.json",
        ROOT / "artifacts" / "fusion_latent_multitask_k1" / "run_summary.json",
        ROOT / "artifacts" / "fusion_latent_multitask_k3" / "run_summary.json",
        ROOT / "artifacts" / "fusion_latent_multitask_k4" / "run_summary.json",
        ROOT / "artifacts" / "personalized_warm_start_v_k4" / "run_summary.json",
        ROOT / "artifacts" / "pairwise_ranker_warm_start" / "run_summary.json",
    ]:
        if path.exists():
            lines.extend(
                [
                    f"### {_rel(path)}",
                    "",
                    "```json",
                    json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
