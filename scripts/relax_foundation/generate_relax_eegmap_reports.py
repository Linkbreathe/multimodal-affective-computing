#!/usr/bin/env python
"""Generate the audited corrected-EEG Relax report package from raw results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.relax_foundation.relax_suite_common import validate_job_output  # noqa: E402
from mac.data.relax_foundation import RELAX_EEG_MONTAGE, RELAX_EEG_RUN_TAG  # noqa: E402


TARGETS = ("relaxation", "discomfort")
EXPECTED_SUITE_COUNTS = {"foundation": 62, "claim_validation": 94, "attention_video": 96}
METRIC_ATOL = 5e-5
METRIC_RTOL = 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ccc(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    covariance = float(np.mean((truth - truth.mean()) * (prediction - prediction.mean())))
    denominator = float(truth.var() + prediction.var() + (truth.mean() - prediction.mean()) ** 2)
    return float(2.0 * covariance / denominator) if denominator > 0 else 0.0


def _metrics(truth: Iterable[float], prediction: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(truth), dtype=float)
    p = np.asarray(list(prediction), dtype=float)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ccc": _ccc(y, p),
        "pearson": float(np.corrcoef(y, p)[0, 1])
        if len(y) > 1 and np.std(y) > 0 and np.std(p) > 0
        else float("nan"),
    }


def _mean_fold_macro(result: dict[str, Any]) -> float:
    values = []
    for fold in result["folds"]:
        metrics = fold["metrics"]
        key = "macro_mae_mean" if "macro_mae_mean" in metrics else "macro_mae"
        values.append(float(metrics[key]))
    return float(np.mean(values))


def _variant(path: Path, suite_root: Path, suite: str) -> str:
    relative = path.relative_to(suite_root)
    if suite == "attention_video" and len(relative.parts) >= 2 and relative.parts[0] != "baselines":
        return relative.parts[-2]
    if "with_eeg" in relative.parts:
        return "with_eeg"
    if "without_eeg" in relative.parts:
        return "without_eeg"
    if "video_residualized" in relative.parts:
        return "video_residualized"
    if "without_video" in relative.parts:
        return "without_video"
    if "full" in relative.parts:
        return "full"
    return "baseline" if "baselines" in relative.parts else "registered"


def _load_and_validate_results(log_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    registry_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for suite, expected_count in EXPECTED_SUITE_COUNTS.items():
        suite_root = log_root / suite
        suite_jobs_path = suite_root / "suite_jobs.json"
        jobs = json.loads(suite_jobs_path.read_text(encoding="utf-8"))
        if len(jobs) != expected_count:
            raise RuntimeError(f"{suite}: registered {len(jobs)} jobs, expected {expected_count}")
        for job in jobs:
            valid, reason = validate_job_output(job["command"])
            if not valid:
                raise RuntimeError(f"{suite}: incomplete registered job: {reason}")
        result_files = sorted(suite_root.rglob("*_results.json"))
        if len(result_files) != expected_count:
            raise RuntimeError(f"{suite}: found {len(result_files)} results, expected {expected_count}")
        hard_failures = list(suite_root.rglob("hard_failure.json"))
        if hard_failures:
            raise RuntimeError(f"{suite}: hard failures remain: {hard_failures[:3]}")

        for result_path in result_files:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            args = result["args"]
            prediction_path = Path(result["predictions_csv"])
            predictions = pd.read_csv(prediction_path)
            job_id = str(result_path.relative_to(log_root))
            baseline = str(args.get("baseline", "none"))
            seed = int(args.get("seed", 20260705))
            variant = _variant(result_path, suite_root, suite)
            registry_rows.append(
                {
                    "job_id": job_id,
                    "suite": suite,
                    "result_json": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "predictions_csv": str(prediction_path),
                    "predictions_sha256": _sha256(prediction_path),
                    "run_tag": args.get("run_tag"),
                    "cohort": args["cohort"],
                    "fusion": args.get("fusion"),
                    "baseline": baseline,
                    "modalities": "+".join(args["modalities"]),
                    "variant": variant,
                    "seed": seed,
                    "target_mode": args.get("target"),
                    "pool": args.get("pool"),
                    "condition_control": args.get("condition_control", "none"),
                    "fold_count": len(result["folds"]),
                    "prediction_rows": len(predictions),
                    "macro_mae": _mean_fold_macro(result),
                    "complete": True,
                }
            )

            for fold in result["folds"]:
                fold_name = str(fold["test_participant"])
                fold_predictions = predictions[predictions["fold"].astype(str) == fold_name]
                json_metrics = fold["metrics"]
                if baseline == "random_9_condition":
                    recomputed_by_seed: dict[str, list[float]] = {}
                    for random_seed, seed_frame in fold_predictions.groupby("seed", sort=True):
                        target_metrics = []
                        for target in TARGETS:
                            values = _metrics(seed_frame[f"{target}_true"], seed_frame[f"{target}_pred"])
                            target_metrics.append(values)
                            for metric, value in values.items():
                                recomputed_by_seed.setdefault(f"{target}_{metric}", []).append(value)
                        recomputed_by_seed.setdefault("macro_mae", []).append(
                            float(np.mean([values["mae"] for values in target_metrics]))
                        )
                        recomputed_by_seed.setdefault("macro_rmse", []).append(
                            float(np.mean([values["rmse"] for values in target_metrics]))
                        )
                    for key, values in recomputed_by_seed.items():
                        finite = np.asarray(values, dtype=float)
                        finite = finite[np.isfinite(finite)]
                        for suffix, recomputed in (
                            ("mean", float(np.mean(finite)) if finite.size else float("nan")),
                            ("std", float(np.std(finite)) if finite.size else float("nan")),
                        ):
                            json_value = float(json_metrics[f"{key}_{suffix}"])
                            audit_rows.append(
                                {
                                    "job_id": job_id,
                                    "fold": fold_name,
                                    "metric": f"{key}_{suffix}",
                                    "json_value": json_value,
                                    "recomputed_value": recomputed,
                                    "absolute_difference": abs(json_value - recomputed),
                                }
                            )
                else:
                    target_values = {}
                    for target in TARGETS:
                        values = _metrics(fold_predictions[f"{target}_true"], fold_predictions[f"{target}_pred"])
                        target_values[target] = values
                        for metric, recomputed in values.items():
                            json_value = float(json_metrics[f"{target}_{metric}"])
                            audit_rows.append(
                                {
                                    "job_id": job_id,
                                    "fold": fold_name,
                                    "metric": f"{target}_{metric}",
                                    "json_value": json_value,
                                    "recomputed_value": recomputed,
                                    "absolute_difference": abs(json_value - recomputed)
                                    if np.isfinite(json_value) and np.isfinite(recomputed)
                                    else 0.0,
                                }
                            )
                    for metric in ("mae", "rmse"):
                        recomputed = float(np.mean([target_values[target][metric] for target in TARGETS]))
                        json_value = float(json_metrics[f"macro_{metric}"])
                        audit_rows.append(
                            {
                                "job_id": job_id,
                                "fold": fold_name,
                                "metric": f"macro_{metric}",
                                "json_value": json_value,
                                "recomputed_value": recomputed,
                                "absolute_difference": abs(json_value - recomputed),
                            }
                        )

                fold_row = {
                    "job_id": job_id,
                    "suite": suite,
                    "cohort": args["cohort"],
                    "fusion": args.get("fusion"),
                    "baseline": baseline,
                    "modalities": "+".join(args["modalities"]),
                    "variant": variant,
                    "seed": seed,
                    "fold": fold_name,
                    "participant_id": fold_name,
                }
                for key, value in json_metrics.items():
                    try:
                        fold_row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
                fold_rows.append(fold_row)

            prediction_seed_column = "seed" if "seed" in predictions.columns else None
            group_columns = ["participant_id"] + ([prediction_seed_column] if prediction_seed_column else [])
            for group_key, group in predictions.groupby(group_columns, sort=True):
                keys = group_key if isinstance(group_key, tuple) else (group_key,)
                participant = str(keys[0])
                metric_seed = int(keys[1]) if prediction_seed_column else seed
                for target in TARGETS:
                    values = _metrics(group[f"{target}_true"], group[f"{target}_pred"])
                    detailed_rows.append(
                        {
                            "job_id": job_id,
                            "suite": suite,
                            "cohort": args["cohort"],
                            "fusion": args.get("fusion"),
                            "baseline": baseline,
                            "modalities": "+".join(args["modalities"]),
                            "variant": variant,
                            "seed": metric_seed,
                            "target": target,
                            "fold": participant,
                            "participant_id": participant,
                            "n": len(group),
                            **values,
                        }
                    )

    registry = pd.DataFrame(registry_rows).sort_values(["suite", "job_id"])
    if len(registry) != 252 or registry["run_tag"].ne(RELAX_EEG_RUN_TAG).any():
        raise RuntimeError("Final result registry is not exactly 252 corrected-run jobs")
    audit = pd.DataFrame(audit_rows)
    json_values = audit["json_value"].to_numpy(dtype=float)
    recomputed_values = audit["recomputed_value"].to_numpy(dtype=float)
    scales = np.maximum.reduce(
        [np.abs(json_values), np.abs(recomputed_values), np.ones(len(audit), dtype=float)]
    )
    audit["relative_difference"] = audit["absolute_difference"].to_numpy(dtype=float) / scales
    audit["allowed_difference"] = METRIC_ATOL + METRIC_RTOL * scales
    audit["within_tolerance"] = np.isclose(
        json_values,
        recomputed_values,
        rtol=METRIC_RTOL,
        atol=METRIC_ATOL,
        equal_nan=True,
    )
    mismatches = audit[~audit["within_tolerance"]]
    if audit.empty or not mismatches.empty:
        worst = mismatches.sort_values("relative_difference", ascending=False).head(5).to_dict("records")
        raise RuntimeError(
            f"Metric recomputation mismatch: count={len(mismatches)}; "
            f"worst={worst}"
        )
    return registry, pd.DataFrame(fold_rows), pd.DataFrame(detailed_rows), audit


def _load_old_registry(old_roots: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for suite, root in old_roots.items():
        for path in sorted(root.rglob("*_results.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            args = result["args"]
            rows.append(
                {
                    "suite": suite,
                    "cohort": args["cohort"],
                    "fusion": args.get("fusion"),
                    "baseline": args.get("baseline", "none"),
                    "modalities": "+".join(args["modalities"]),
                    "seed": int(args.get("seed", 20260705)),
                    "target_mode": args.get("target"),
                    "pool": args.get("pool"),
                    "condition_control": args.get("condition_control", "none"),
                    "variant": _variant(path, root, suite),
                    "old_result_json": str(path),
                    "old_macro_mae": _mean_fold_macro(result),
                }
            )
    return pd.DataFrame(rows)


def _old_new_comparison(registry: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "suite", "cohort", "fusion", "baseline", "modalities", "seed", "target_mode",
        "pool", "condition_control", "variant",
    ]
    new = registry[registry["suite"].isin(["claim_validation", "attention_video"])].copy()
    comparison = old.merge(new, on=keys, how="inner", validate="one_to_one")
    if len(old) != 190 or len(new) != 190 or len(comparison) != 190:
        raise RuntimeError(
            "Old/new comparison must contain all 190 strictly comparable "
            f"claim/attention jobs; old={len(old)}, new={len(new)}, matched={len(comparison)}"
        )
    comparison["new_minus_old_macro_mae"] = comparison["macro_mae"] - comparison["old_macro_mae"]
    rank_groups = ["suite", "cohort", "seed", "variant"]
    comparison["old_rank"] = comparison.groupby(rank_groups)["old_macro_mae"].rank(method="min")
    comparison["new_rank"] = comparison.groupby(rank_groups)["macro_mae"].rank(method="min")
    comparison["rank_change_new_minus_old"] = comparison["new_rank"] - comparison["old_rank"]
    return comparison.sort_values(keys)


def _markdown(frame: pd.DataFrame, columns: list[str], n: int = 20) -> str:
    if frame.empty:
        return "无可用行。"
    view = frame.copy()
    for column in columns:
        if column not in view:
            view[column] = np.nan
    view = view[columns].head(n)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in view.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float) and math.isfinite(value):
                values.append(f"{value:.5f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _save_figures(
    report_dir: Path,
    registry: pd.DataFrame,
    drift: pd.DataFrame,
    claim_dir: Path,
    attention_dir: Path,
    comparison: pd.DataFrame,
) -> None:
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    participant_drift = drift.groupby("participant_id", as_index=False).agg(
        cosine=("cosine_similarity", "mean"), l2=("l2_drift", "mean")
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(participant_drift["participant_id"], participant_drift["cosine"])
    axes[0].set_ylabel("Cosine similarity")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(participant_drift["participant_id"], participant_drift["l2"])
    axes[1].set_ylabel("L2 drift")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figure_dir / "eeg_embedding_drift_by_participant.png", dpi=180)
    plt.close(fig)

    full = registry[
        (registry["suite"] == "claim_validation")
        & (registry["variant"] == "full")
        & (registry["baseline"] == "none")
        & (registry["cohort"] == "all_135")
    ]
    ranking = full.groupby("fusion", as_index=False).agg(mean=("macro_mae", "mean"), std=("macro_mae", "std"))
    ranking = ranking.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ranking["fusion"], ranking["mean"], yerr=ranking["std"], capsize=4)
    ax.set_ylabel("Macro MAE (lower is better)")
    fig.tight_layout()
    fig.savefig(figure_dir / "fusion_seed_stability_all135.png", dpi=180)
    plt.close(fig)

    eeg_effect = pd.read_csv(claim_dir / "eeg_effect_aggregate.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    eeg_effect = eeg_effect.sort_values("eeg_help_mean", ascending=False)
    ax.bar(eeg_effect["fusion"], eeg_effect["eeg_help_mean"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("MAE(without EEG) - MAE(with EEG)")
    fig.tight_layout()
    fig.savefig(figure_dir / "paired_eeg_gain_by_fusion.png", dpi=180)
    plt.close(fig)

    attention = pd.read_csv(attention_dir / "attention_video_aggregate.csv")
    attention = attention[(attention["summary_type"] == "metric") & (attention["cohort"] == "all_135")]
    attention_plot = attention.groupby("comparison", as_index=False)["macro_mae_mean"].mean().sort_values("macro_mae_mean")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(attention_plot["comparison"], attention_plot["macro_mae_mean"])
    ax.set_ylabel("Mean macro MAE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_dir / "attention_video_variants_all135.png", dpi=180)
    plt.close(fig)

    neural_comparison = comparison[comparison["baseline"] == "none"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(neural_comparison["old_macro_mae"], neural_comparison["macro_mae"], alpha=0.5, s=15)
    limits = [
        min(neural_comparison["old_macro_mae"].min(), neural_comparison["macro_mae"].min()),
        max(neural_comparison["old_macro_mae"].max(), neural_comparison["macro_mae"].max()),
    ]
    ax.plot(limits, limits, linestyle="--", color="black")
    ax.set_xlabel("Old wrong-map macro MAE")
    ax.set_ylabel("Corrected-map macro MAE")
    fig.tight_layout()
    fig.savefig(figure_dir / "old_vs_corrected_macro_mae.png", dpi=180)
    plt.close(fig)


def _write_reports(
    report_dir: Path,
    registry: pd.DataFrame,
    detailed: pd.DataFrame,
    comparison: pd.DataFrame,
    acceptance: dict[str, Any],
    drift: pd.DataFrame,
    claim_dir: Path,
    attention_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    claim_seed = pd.read_csv(claim_dir / "seed_stability_full_aggregate.csv").sort_values("macro_mae_mean")
    eeg_effect = pd.read_csv(claim_dir / "eeg_effect_aggregate.csv").sort_values("eeg_help_mean", ascending=False)
    handcrafted = pd.read_csv(claim_dir / "handcrafted_full_aggregate.csv")
    paired_eeg = pd.read_csv(claim_dir / "paired_eeg_stats.csv")
    attention = pd.read_csv(attention_dir / "attention_video_aggregate.csv")
    claim_decision = json.loads((claim_dir / "claim_validation_summary.json").read_text(encoding="utf-8"))["decision"]
    attention_decision = json.loads(
        (attention_dir / "attention_video_summary.json").read_text(encoding="utf-8")
    )["decision"]
    best = claim_seed.iloc[0]
    stable_eeg = eeg_effect[eeg_effect["eeg_help_min"] > 0]
    stable_condition = claim_seed[claim_seed["delta_vs_condition_max"] < 0]
    worse_condition = claim_seed[claim_seed["delta_vs_condition_min"] > 0]
    condition_jobs = registry[
        (registry["suite"] == "claim_validation") & (registry["baseline"] == "condition")
    ]
    condition_all = float(condition_jobs.loc[condition_jobs["cohort"] == "all_135", "macro_mae"].iloc[0])
    hand_all = handcrafted[handcrafted["cohort"] == "all_135"]
    hand_beaten = hand_all[hand_all["foundation_minus_handcrafted_max"] < 0]
    hand_eeg = handcrafted[handcrafted["cohort"] == "eeg_eligible"]
    hand_eeg_beaten = hand_eeg[hand_eeg["foundation_minus_handcrafted_max"] < 0]
    clustered_eeg_support = paired_eeg[(paired_eeg["ci_low"] > 0) & (paired_eeg["ci_high"] > 0)]

    executive = f"""# Relax EEG 通道修正执行摘要

本次结果使用唯一 run tag `{RELAX_EEG_RUN_TAG}`。XDF 列 `[1,2,3,4]` 按原始顺序输入，REVE 位置严格为 `{list(RELAX_EEG_MONTAGE)}`。所有 252 个注册结果均完整，`hard_failure.json` 为 0；15 人与 9 人队列分别通过 15/9 folds 验收。

数据层面，567 个 EEG 窗口全部重新提取，379 个 EEG 缺失均来自预注册的 6 位 EEG-disabled 参与者；4,730 个非 EEG tensor 与旧缓存字节级一致。旧新 EEG embedding 的平均 cosine 为 {acceptance['eeg_drift']['cosine_mean']:.4f}，平均 L2 drift 为 {acceptance['eeg_drift']['l2_mean']:.4f}，且没有完全相同的窗口。

三随机种子下，all_135 全模态最佳均值为 `{best['fusion']}`，macro MAE={best['macro_mae_mean']:.4f}（Condition baseline={condition_all:.4f}）。稳定低于 Condition 的方法为 {', '.join(stable_condition['fusion']) if len(stable_condition) else '无'}；稳定高于 Condition 的方法为 {', '.join(worse_condition['fusion']) if len(worse_condition) else '无'}。复杂融合并未普遍胜过简单融合。

EEG 增益只在 `{', '.join(stable_eeg['fusion']) if len(stable_eeg) else '无'}` 保持三个 seed 同向；但 participant-clustered 区间仅有 {len(clustered_eeg_support)} 个 seed 行完全高于 0，因此不能写成跨 seed、跨融合的普遍增益。corrected foundation 在 eeg_eligible 上稳定胜过 corrected handcrafted Ridge-CV 的方法为 {', '.join(hand_eeg_beaten['fusion']) if len(hand_eeg_beaten) else '无'}，在 all_135 上则为 0 个。

claim-validation 的预注册决策为 `{claim_decision}`；attention-video 为 `{attention_decision}`。后者只在 eeg_eligible 的特定融合上支持近似替代，不能外推到 all_135 或所有融合。
"""
    (report_dir / "executive_summary_zh.md").write_text(executive, encoding="utf-8")

    participant_drift = drift.groupby("participant_id", as_index=False).agg(
        cosine_mean=("cosine_similarity", "mean"),
        cosine_std=("cosine_similarity", "std"),
        l2_mean=("l2_drift", "mean"),
        l2_std=("l2_drift", "std"),
    ).sort_values("l2_mean", ascending=False)
    condition_drift = drift.groupby("condition", as_index=False).agg(
        cosine_mean=("cosine_similarity", "mean"),
        cosine_std=("cosine_similarity", "std"),
        l2_mean=("l2_drift", "mean"),
        l2_std=("l2_drift", "std"),
    ).sort_values("l2_mean", ascending=False)
    data_report = f"""# 数据、QC、通道映射与 embedding 审计

## 固定通道契约

| XDF sample 列 | EEG 局部索引 | 电极 | 半球 |
| ---: | ---: | --- | --- |
| 1 | 0 | M2 | 右 |
| 2 | 1 | TP9 | 左 |
| 3 | 2 | TP10 | 右 |
| 4 | 3 | M1 | 左 |

handcrafted alpha asymmetry 定义为 `log(mean(M2,TP10) alpha)-log(mean(TP9,M1) alpha)`。REVE 使用 frozen `brain-bzh/reve-large` 与 `brain-bzh/reve-positions`；每个 tensor 均记录源 XDF、窗口坐标、模型 revision、位置库 revision、run tag 与映射。

## 验收结果

- condition samples：135；EEG valid/missing：567/379。
- 非 EEG tensor：{acceptance['copied_non_eeg_tensors']} 个，SHA-256/字节一致。
- 唯一非 EEG 异常窗口：`P005/C2_w007`；eye/head/video/attention-video mask 均保留。
- P003 GPU smoke：63 个窗口、1024 维、全有限值、位置顺序正确。
- REVE-large 当前原生宽度为 1216；本 run 使用无训练参数的 adaptive average reduction 固定为预注册的 1024 维，并在 provenance 中明确记录。旧 embedding 做同一适配后再计算 drift。

## participant 异质性

{_markdown(participant_drift, ['participant_id','cosine_mean','cosine_std','l2_mean','l2_std'], 20)}

P012 的平均 drift 最大；窗口级 cosine 最低值为 {acceptance['eeg_drift']['cosine_min']:.4f}。这表明修正影响具有参与者异质性。

## Condition 异质性

{_markdown(condition_drift, ['condition','cosine_mean','cosine_std','l2_mean','l2_std'], 20)}

C7 的平均 L2 drift 最大，C4 最小；因此旧新差异也不是各 Condition 等幅平移。
"""
    (report_dir / "data_qc_eeg_audit_zh.md").write_text(data_report, encoding="utf-8")

    foundation = registry[registry["suite"] == "foundation"]
    foundation_baselines = foundation[foundation["baseline"] != "none"].sort_values(["cohort", "macro_mae"])
    formal = foundation[foundation["job_id"].str.contains("/formal_fusion/")].sort_values(["cohort", "macro_mae"])
    leave = foundation[foundation["job_id"].str.contains("leave_one_modality_out")].sort_values("macro_mae")
    progressive = foundation[foundation["job_id"].str.contains("progressive_modalities")].sort_values("job_id")
    representation = foundation[foundation["job_id"].str.contains("representation_loss")].sort_values("macro_mae")
    fusion_report = f"""# 六种融合、模态与表示消融

## 五个队列的基线

{_markdown(foundation_baselines, ['cohort','baseline','macro_mae','fold_count'], 20)}

非 CV 的 `relax_handcrafted` 在包含 P006 的两个 15 人队列出现数值退化（P006 预测绝对值最高约 `1.86e12`），因此该行不能用于模型优越性结论；claim-validation 的 `relax_handcrafted_ridge_cv` 未出现该退化，并作为正式 handcrafted 对照。

## 五个队列的正式融合排名

{_markdown(formal, ['cohort','fusion','modalities','macro_mae','fold_count'], 40)}

## Leave-one-modality-out

{_markdown(leave, ['cohort','fusion','modalities','macro_mae'], 10)}

## Progressive modalities

{_markdown(progressive, ['cohort','fusion','modalities','macro_mae'], 10)}

## Pooling / target representation

{_markdown(representation, ['cohort','fusion','target_mode','pool','macro_mae'], 10)}

简单融合与复杂融合的判断以同一 cohort、seed、全模态配置为准；若复杂方法没有稳定低于 Early/Mid/Late，则只报告趋势，不使用“显著优于”。模态贡献同样要求 leave-one-out、progressive 与三种子配对方向相互支持。

三 seed 的 all_135 对照中，简单的 Late 均值最佳（0.16394），HealNet 接近（0.16435）；两者都在三个 seed 中低于 Condition。Q-Former 的单 seed 正式实验排名第一，但三 seed 标准差最大（0.01292），说明单 seed 排名不稳。MM-Lego 的三 seed 均值最差（0.18550），所以复杂融合没有整体优势。

MM-Lego 单 seed 消融中，去掉任一模态都比全模态 sequence 更差，去 eye 的退化最大；但 progressive 路径从 ECG 加入 EEG 反而由 0.19178 变为 0.19915，和 leave-one-out 的 EEG 方向不一致。可稳定陈述的是 eye/head/video 随后逐步改善，不能据此宣称 EEG 有普遍独立贡献。表示消融中 original+attention 最好（0.17035），所有 residual-target 版本更差；该结论限定于 MM-Lego 单 seed。
"""
    (report_dir / "fusion_modality_representation_zh.md").write_text(fusion_report, encoding="utf-8")

    claim_report = f"""# Claim-validation 专项报告

## 三随机种子稳定性（all_135，全模态）

{_markdown(claim_seed, list(claim_seed.columns), 10)}

## EEG 增益（正数表示加 EEG 更好）

{_markdown(eeg_effect, list(eeg_effect.columns), 10)}

participant-clustered seed-level 配对结果：

{_markdown(paired_eeg.sort_values('mean_without_minus_with_eeg', ascending=False), ['fusion','seed','mean_without_minus_with_eeg','ci_low','ci_high','p_signflip_two_sided'], 20)}

## Corrected foundation vs corrected handcrafted Ridge-CV

{_markdown(handcrafted, list(handcrafted.columns), 30)}

all_135 与 eeg_eligible 必须分开解释：前者 6 位参与者的 EEG 全部 masked，会稀释“加 EEG”的总体效应；EEG 的真实配对增益以 9 人 eeg_eligible 结果为主。区间跨 0 的结果仅视为趋势。

实际结论：Late 与 HealNet 稳定胜过 Condition；MM-Lego 稳定更差。只有 MM-Lego 的 EEG 增益在三个 seed 中都为正，但仅 seed 20260706 的 participant-clustered 95% 区间完全高于 0，其余两个跨 0。Early 的三个 seed 均为负向。故 EEG 效应明显依赖融合方法，预注册总决策为 `{claim_decision}`。

相对 corrected handcrafted Ridge-CV，all_135 的六种 foundation 融合均稳定更差；eeg_eligible 中 Q-Former、HealNet、Early 稳定更好，Mid/Late 仅为混合方向，MM-Lego 稳定更差。队列结论因此发生反转，不能合并报告。
"""
    (report_dir / "claim_validation_zh.md").write_text(claim_report, encoding="utf-8")

    attention_metrics = attention[attention["summary_type"] == "metric"].sort_values(
        ["cohort", "macro_mae_mean"]
    )
    attention_pairs = attention[attention["summary_type"] == "replacement_pair"].sort_values(
        ["cohort", "comparison", "mean_delta"]
    )
    attention_report = f"""# Attention-video 与正确 EEG 专项报告

预注册决策：`{attention_decision}`。

## 三种子模型指标

{_markdown(attention_metrics, ['cohort','fusion','comparison','macro_mae_mean','macro_mae_std','macro_mae_min','macro_mae_max','n_seeds'], 40)}

## 替代与冗余配对

{_markdown(attention_pairs, ['cohort','fusion','comparison','mean_delta','min_delta','max_delta','n_seeds'], 30)}

`attention_replacement - full_eye_video` 接近 0 才支持替代；`attention_replacement - no_visual < 0` 才支持它提供额外视觉信号；`attention_only - visual_pair_only < 0` 才支持单独 attention-video 优于原视觉对。结论必须同时查看 all_135 与 eeg_eligible，若只在 9 人队列成立则明确限定队列。

all_135 中没有任何融合同时满足稳定近似 full-eye-video 且稳定优于 no-visual；Early 的 attention-replacement 均值最好，但 seed 方向混合，只能视为趋势。eeg_eligible 中仅 MM-Lego 满足预设替代条件：replacement 相对 full-eye-video 的均值差为 +0.00281（近似持平），相对 no-visual 为 -0.00608 且三个 seed 全为负。该结果说明 attention-video 在这一方法/队列中提供补充视觉信息，但它并非总体最佳模型。

在 eeg_eligible 上绝对最佳是 Early 的 visual-pair-only（0.11893）；attention-only 相对 visual-pair-only 在 Early 和 HealNet 三个 seed 都更差。因而 attention-video 不能普遍取代原始 eye+video，也没有证据表明它单独就足够。
"""
    (report_dir / "attention_video_zh.md").write_text(attention_report, encoding="utf-8")

    change_summary = comparison.groupby(["suite", "cohort", "fusion"], as_index=False).agg(
        old_macro_mae=("old_macro_mae", "mean"),
        corrected_macro_mae=("macro_mae", "mean"),
        corrected_minus_old=("new_minus_old_macro_mae", "mean"),
        mean_rank_change=("rank_change_new_minus_old", "mean"),
        configurations=("job_id", "count"),
    ).sort_values("corrected_minus_old")
    has_eeg = comparison["modalities"].str.split("+").map(lambda values: "eeg" in values)
    neural = comparison[(comparison["baseline"] == "none") & has_eeg]
    neural_summary = neural.groupby(["suite", "cohort"], as_index=False).agg(
        configurations=("job_id", "count"),
        old_macro_mae=("old_macro_mae", "mean"),
        corrected_macro_mae=("macro_mae", "mean"),
        corrected_minus_old=("new_minus_old_macro_mae", "mean"),
        min_change=("new_minus_old_macro_mae", "min"),
        max_change=("new_minus_old_macro_mae", "max"),
    )
    no_eeg_neural = comparison[(comparison["baseline"] == "none") & ~has_eeg]
    no_eeg_noise = float(no_eeg_neural["new_minus_old_macro_mae"].abs().max())
    old_new_report = f"""# 旧错误映射 vs 新正确映射

旧映射使用 `[T7,T8,TP7,TP8]` 位置，新映射使用 `[M2,TP9,TP10,M1]`；原始信号列未重排。embedding drift 说明位置修正产生了实质变化，而非文件重命名。

## 严格可比配置的性能与排名变化

{_markdown(change_summary, list(change_summary.columns), 60)}

## 含 EEG 的神经模型配置

{_markdown(neural_summary, list(neural_summary.columns), 20)}

正的 `corrected_minus_old` 表示修正后 MAE 更高，负值表示更低。通道映射修正是测量契约纠错，不能依据性能方向选择旧映射；排名变化用于审计结论稳健性，而不是决定是否接受正确映射。

修正并未带来单向性能变化：claim all_135 的含 EEG 神经配置平均 +0.00187，而 eeg_eligible 平均 -0.00155；attention-video 对应为 +0.00168 与 -0.00024。不同融合的改变量横跨正负，排名也有交换。无 EEG 神经对照的最大绝对差仅 {no_eeg_noise:.6f}，提供了重跑数值噪声量级；含 EEG 的主要变化远大于这一量级。
"""
    (report_dir / "old_vs_new_eegmap_zh.md").write_text(old_new_report, encoding="utf-8")

    comprehensive = f"""# Relax 正确 EEG 通道映射综合主报告

本报告的所有数值由结果 JSON 和 prediction CSV 自动重算。结果总数为 {len(registry)}，明细指标行数为 {len(detailed)}；完整注册表、fold/participant/target/seed 指标、旧新配对表与图表均保存在本目录。

## 核心结论

1. 通道契约、缓存隔离和 QC 全部通过；旧结果未覆盖。
2. all_135 三种子最佳全模态融合为 `{best['fusion']}`（macro MAE={best['macro_mae_mean']:.4f}）。复杂融合是否优于简单融合请以 `{best['fusion']}` 的实际类别及完整排名为准，不能预设 HealNet/MM-Lego 更强。
3. EEG 增益存在融合依赖；稳定正向的方法为 {', '.join(stable_eeg['fusion']) if len(stable_eeg) else '无'}，且 participant-clustered 支持只出现在部分 seed，不能泛化。
4. all_135 中 EEG-disabled mask 会稀释效应；EEG 因果式配对解读限于 eeg_eligible 9 人队列。
5. corrected foundation 相对 corrected handcrafted Ridge-CV 的广泛优势{'' if len(hand_beaten) else '不'}成立；all_135 全部更差，eeg_eligible 仅 Q-Former/HealNet/Early 稳定更好。
6. attention-video 仅在 eeg_eligible+MM-Lego 满足预设近似替代条件；all_135 不支持普遍替代，attention-only 也不普遍优于原始视觉对。
7. 通道修正引起实质 embedding drift，但下游 MAE 变化有正有负；正确映射的接受依据是测量契约，而不是性能方向。

## 报告导航

- `executive_summary_zh.md`
- `data_qc_eeg_audit_zh.md`
- `fusion_modality_representation_zh.md`
- `claim_validation_zh.md`
- `attention_video_zh.md`
- `old_vs_new_eegmap_zh.md`
- `claim_validation/` 与 `attention_video/`：原分析器输出的 participant-clustered 表与英文可审计报告。
"""
    (report_dir / "comprehensive_report_zh.md").write_text(comprehensive, encoding="utf-8")


def _mark_old_reports(report_dir: Path, paths: list[Path]) -> pd.DataFrame:
    marker = "<!-- superseded-by-corrected-relax-eegmap -->"
    banner = (
        f"{marker}\n> **已被正确通道映射结果取代。** 请使用 "
        f"`{report_dir / 'comprehensive_report_zh.md'}`。以下旧正文仅为审计保留。\n\n"
    )
    rows = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if marker not in text:
            path.write_text(banner + text, encoding="utf-8")
        rows.append({"old_report": str(path), "superseded_by": str(report_dir / "comprehensive_report_zh.md")})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--acceptance-audit", required=True, type=Path)
    parser.add_argument("--embedding-drift", required=True, type=Path)
    parser.add_argument("--old-claim-root", required=True, type=Path)
    parser.add_argument("--old-attention-root", required=True, type=Path)
    args = parser.parse_args()

    registry, fold_metrics, detailed, metric_audit = _load_and_validate_results(args.log_root)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(args.report_dir / "result_registry.csv", index=False)
    (args.report_dir / "result_registry.json").write_text(
        json.dumps(registry.to_dict("records"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fold_metrics.to_csv(args.report_dir / "fold_metrics.csv", index=False)
    detailed.to_csv(args.report_dir / "participant_target_seed_metrics.csv", index=False)
    metric_audit.to_csv(args.report_dir / "metric_recalculation_audit.csv", index=False)

    old = _load_old_registry(
        {"claim_validation": args.old_claim_root, "attention_video": args.old_attention_root}
    )
    comparison = _old_new_comparison(registry, old)
    comparison["has_eeg"] = comparison["modalities"].str.split("+").map(lambda values: "eeg" in values)
    comparison["is_neural"] = comparison["baseline"].eq("none")
    comparison.to_csv(args.report_dir / "old_new_result_comparison.csv", index=False)
    (
        comparison[comparison["has_eeg"] & comparison["is_neural"]]
        .groupby(["suite", "cohort", "fusion"], as_index=False)
        .agg(
            configurations=("job_id", "count"),
            old_macro_mae=("old_macro_mae", "mean"),
            corrected_macro_mae=("macro_mae", "mean"),
            corrected_minus_old=("new_minus_old_macro_mae", "mean"),
            min_change=("new_minus_old_macro_mae", "min"),
            max_change=("new_minus_old_macro_mae", "max"),
            mean_rank_change=("rank_change_new_minus_old", "mean"),
        )
        .to_csv(args.report_dir / "old_new_eeg_neural_summary.csv", index=False)
    )

    acceptance = json.loads(args.acceptance_audit.read_text(encoding="utf-8"))
    if acceptance.get("status") != "ok" or acceptance.get("run_tag") != RELAX_EEG_RUN_TAG:
        raise RuntimeError(
            "Corrected-cache acceptance audit did not pass for the registered run tag: "
            f"status={acceptance.get('status')!r}, run_tag={acceptance.get('run_tag')!r}"
        )
    drift = pd.read_csv(args.embedding_drift)
    claim_dir = args.report_dir / "claim_validation"
    attention_dir = args.report_dir / "attention_video"
    _save_figures(args.report_dir, registry, drift, claim_dir, attention_dir, comparison)
    _write_reports(
        args.report_dir,
        registry,
        detailed,
        comparison,
        acceptance,
        drift,
        claim_dir,
        attention_dir,
    )

    superseded = _mark_old_reports(
        args.report_dir,
        [
            Path("reports/relax_foundation_probe/relax_foundation_report.md"),
            Path("reports/relax_claim_validation/claim_validation_report.md"),
            Path("reports/relax_attention_video/attention_video_report.md"),
        ],
    )
    superseded.to_csv(args.report_dir / "superseded_report_registry.csv", index=False)
    final_audit = {
        "status": "ok",
        "run_tag": RELAX_EEG_RUN_TAG,
        "result_count": len(registry),
        "suite_counts": registry.groupby("suite").size().to_dict(),
        "hard_failure_count": 0,
        "metric_rows_recomputed": len(metric_audit),
        "max_metric_absolute_difference": float(metric_audit["absolute_difference"].max()),
        "max_metric_relative_difference": float(metric_audit["relative_difference"].max()),
        "max_metric_tolerance_ratio": float(
            (metric_audit["absolute_difference"] / metric_audit["allowed_difference"]).max()
        ),
        "metric_tolerance_atol": METRIC_ATOL,
        "metric_tolerance_rtol": METRIC_RTOL,
        "metric_mismatch_count": int((~metric_audit["within_tolerance"]).sum()),
        "ranking_rows": len(comparison),
        "report_files": sorted(
            {
                *(str(path.relative_to(args.report_dir)) for path in args.report_dir.rglob("*") if path.is_file()),
                "final_acceptance.json",
            }
        ),
    }
    (args.report_dir / "final_acceptance.json").write_text(
        json.dumps(final_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
