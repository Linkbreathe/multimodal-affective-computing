"""Research-only H/P explainability audit for the minimal 1DCNN benchmark.

The existing 15-combination benchmark is deliberately left untouched.  This
module reads its H and P out-of-fold references, then trains only *family
removal* variants in memory.  It therefore cannot create a checkpoint or
change the runtime, deployment, or Shadow decision paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from real_time_ml.config import ProjectConfig
from real_time_ml.modeling.dcnn import _architecture, _device, _validation_indexes
from real_time_ml.modeling.minimal_fusion import (
    EXPECTED_LABELS,
    FEATURES_PER_MODALITY,
    HIGH_DISCOMFORT_TRUTH_THRESHOLD,
    TARGETS,
    _condition_baseline,
    _dependencies,
    _point_metrics,
)
from real_time_ml.modeling.minimal_fusion_dcnn import (
    MinimalFusionSequences,
    SOURCE_RELATIVE_PATH,
    _fold_seed,
    _predict_residual_model,
    _train_residual_model,
    build_minimal_fusion_sequences,
)
from real_time_ml.utils import atomic_write_text


OUTPUT_DIRECTORY = "fusion_minimal_dcnn_hp"
SELECTION_AUDIT_FILENAME = "selection_audit.csv"
SELECTION_STABILITY_FILENAME = "selection_stability.csv"
FAMILY_ABLATION_FILENAME = "feature_family_ablation_metrics.csv"
SUBGROUP_FILENAME = "subgroup_metrics.csv"
REPORT_FILENAME = "minimal_fusion_dcnn_hp_explainability_zh.md"

REFERENCE_DIRECTORY = "fusion_minimal_dcnn"
REFERENCE_OOF_FILENAME = "oof_predictions.csv"

H_FAMILIES = (
    "head_translational_speed",
    "head_angular_velocity",
    "head_jerk",
    "head_stationary_fraction",
    "head_position_range",
    "head_motion_spectral_entropy",
)
P_FAMILIES = (
    "ecg_rate_signal_amplitude",
    "ecg_hrv",
    "ecg_rr_std_ms_audit_only",
    "eeg_relative_band_power",
    "eeg_absolute_power_amplitude",
    "eeg_hjorth_spectral_entropy",
    "eeg_ratios_asymmetry",
)
FAMILIES_BY_MODALITY = {"H": H_FAMILIES, "P": P_FAMILIES}

FAMILY_LABELS_ZH = {
    "head_translational_speed": "平移速度",
    "head_angular_velocity": "角速度",
    "head_jerk": "jerk",
    "head_stationary_fraction": "静止比例",
    "head_position_range": "位置范围",
    "head_motion_spectral_entropy": "运动频谱熵",
    "ecg_rate_signal_amplitude": "ECG 速率/信号幅度",
    "ecg_hrv": "ECG HRV",
    "ecg_rr_std_ms_audit_only": "ecg_rr_std_ms_audit_only（审计敏感性）",
    "eeg_relative_band_power": "EEG 相对频带功率",
    "eeg_absolute_power_amplitude": "EEG 绝对功率/幅度",
    "eeg_hjorth_spectral_entropy": "EEG Hjorth/谱熵",
    "eeg_ratios_asymmetry": "EEG 比率/不对称性",
}


@dataclass(frozen=True)
class FamilyAblation:
    modality: str
    removed_family: str
    sequences: MinimalFusionSequences


def feature_family(modality: str, feature_name: str) -> str:
    """Map every permitted H/P feature to exactly one predeclared family."""
    if modality == "H":
        if feature_name.startswith("head_speed_"):
            return "head_translational_speed"
        if feature_name.startswith("head_angular_speed_deg_s_"):
            return "head_angular_velocity"
        if feature_name.startswith("head_jerk_"):
            return "head_jerk"
        if feature_name == "head_stationary_fraction":
            return "head_stationary_fraction"
        if feature_name == "head_position_range":
            return "head_position_range"
        if feature_name == "head_motion_spectral_entropy":
            return "head_motion_spectral_entropy"
    elif modality == "P":
        if feature_name == "ecg_rr_std_ms_audit_only":
            return "ecg_rr_std_ms_audit_only"
        if feature_name.startswith("ecg_hrv_"):
            return "ecg_hrv"
        if feature_name.startswith("ecg_"):
            return "ecg_rate_signal_amplitude"
        if feature_name.endswith("_relative"):
            return "eeg_relative_band_power"
        if feature_name.endswith("_power") or feature_name.endswith("_robust_amplitude_uV"):
            return "eeg_absolute_power_amplitude"
        if "_hjorth_" in feature_name or feature_name.endswith("_spectral_entropy"):
            return "eeg_hjorth_spectral_entropy"
        if feature_name in {
            "eeg_alpha_beta_ratio",
            "eeg_theta_beta_ratio",
            "eeg_alpha_asymmetry_log_right_left",
        }:
            return "eeg_ratios_asymmetry"
    raise ValueError(f"Unmapped {modality} feature in H/P audit: {feature_name}")


def _modality_feature_names(sequences: MinimalFusionSequences, modality: str) -> list[str]:
    prefixes = ("head_",) if modality == "H" else ("eeg_", "ecg_")
    return [name for name in sequences.feature_columns if name.startswith(prefixes)]


def _validate_family_coverage(sequences: MinimalFusionSequences) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = {
        modality: {family: [] for family in families}
        for modality, families in FAMILIES_BY_MODALITY.items()
    }
    for modality in FAMILIES_BY_MODALITY:
        for feature_name in _modality_feature_names(sequences, modality):
            family = feature_family(modality, feature_name)
            grouped[modality][family].append(feature_name)
    for modality, families in grouped.items():
        empty = [family for family, names in families.items() if not names]
        if empty:
            raise ValueError(f"Missing expected {modality} feature families: {empty}")
    return grouped


def _subset_sequences(sequences: MinimalFusionSequences, names: list[str]) -> MinimalFusionSequences:
    lookup = {name: index for index, name in enumerate(sequences.feature_columns)}
    indexes = np.asarray([lookup[name] for name in names], dtype=int)
    return MinimalFusionSequences(
        values=sequences.values[:, indexes, :],
        targets=sequences.targets,
        participant_ids=sequences.participant_ids,
        conditions=sequences.conditions,
        presentation_positions=sequences.presentation_positions,
        lengths=sequences.lengths,
        feature_columns=tuple(names),
    )


def _labels_frame(sequences: MinimalFusionSequences) -> Any:
    pd, *_ = _dependencies()
    return pd.DataFrame(
        {
            "participant_id": sequences.participant_ids,
            "condition": sequences.conditions,
            "presentation_position": sequences.presentation_positions,
            "relaxation": sequences.targets[:, 0],
            "discomfort": sequences.targets[:, 1],
        }
    )


def _rank_features_with_audit(
    sequences: MinimalFusionSequences,
    train_indexes: np.ndarray,
    feature_names: list[str],
    residual: np.ndarray,
    *,
    modality: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Rank one fold and retain enough per-feature evidence to audit selection."""
    lookup = {name: index for index, name in enumerate(sequences.feature_columns)}
    total_windows = int(np.sum(sequences.lengths[train_indexes]))
    scored: list[tuple[float, str, float, int, float, bool]] = []
    residual = np.asarray(residual, dtype=float)
    for feature_name in feature_names:
        feature_index = lookup[feature_name]
        values = sequences.values[train_indexes, feature_index, :]
        valid_time = np.zeros(values.shape, dtype=bool)
        for local_index, sequence_index in enumerate(train_indexes):
            valid_time[local_index, : int(sequences.lengths[sequence_index])] = True
        values = values[valid_time]
        repeated_residual = np.repeat(residual, sequences.lengths[train_indexes])
        valid = np.isfinite(values) & np.isfinite(repeated_residual)
        valid_count = int(valid.sum())
        available = bool(valid_count >= 2 and np.std(values[valid]) > 0.0 and np.std(repeated_residual[valid]) > 0.0)
        correlation = float("nan")
        if available:
            correlation = float(np.corrcoef(values[valid], repeated_residual[valid])[0, 1])
            available = bool(np.isfinite(correlation))
        if available:
            scored.append(
                (
                    abs(correlation),
                    feature_name,
                    correlation,
                    valid_count,
                    float(valid_count / total_windows) if total_windows else 0.0,
                    True,
                )
            )
        else:
            scored.append(
                (float("nan"), feature_name, correlation, valid_count, float(valid_count / total_windows), False)
            )

    ordered = sorted(
        (row for row in scored if row[-1]), key=lambda row: (-float(row[0]), str(row[1]))
    )
    ranks = {name: rank for rank, (_, name, *_rest) in enumerate(ordered, start=1)}
    selected = [name for _, name, *_rest in ordered[:FEATURES_PER_MODALITY]]
    audit = []
    for absolute, name, signed, valid_count, availability, available in sorted(
        scored, key=lambda row: str(row[1])
    ):
        rank = ranks.get(name)
        audit.append(
            {
                "modality": modality,
                "feature_name": name,
                "family": feature_family(modality, name),
                "selection_rank": rank if rank is not None else np.nan,
                "selected": bool(name in selected),
                "signed_pearson_r": signed,
                "absolute_pearson_r": absolute,
                "valid_window_count": valid_count,
                "train_window_count": total_windows,
                "availability_fraction": availability,
                "available_for_ranking": available,
                "feature_limit_per_modality": FEATURES_PER_MODALITY,
                "feature_cap_binding": bool(len(feature_names) > FEATURES_PER_MODALITY),
            }
        )
    return selected, audit


def _reference_oof_path(config: ProjectConfig) -> Path:
    if config.is_legacy:
        return config.path("artifacts") / REFERENCE_DIRECTORY / REFERENCE_OOF_FILENAME
    reference_run_id = config.get("experiment.reference_run_id")
    if not reference_run_id:
        raise ValueError(
            "Layered H/P audit requires experiment.reference_run_id for the prior minimal 1DCNN run"
        )
    return (
        Path(str(config.get("paths.artifacts_root")))
        / "runs"
        / str(reference_run_id)
        / "predictions"
        / "minimal_fusion_dcnn_oof_predictions.csv"
    )


def _load_reference_oof(config: ProjectConfig, labels: Any) -> dict[str, Any]:
    pd, *_ = _dependencies()
    source = _reference_oof_path(config)
    if not source.exists():
        raise FileNotFoundError(
            "H/P audit requires the existing 1DCNN minimal-fusion OOF reference; "
            "run 'rtml benchmark-minimal-fusion-dcnn' first"
        )
    frame = pd.read_csv(source)
    required = {
        "combination", "participant_id", "condition", "presentation_position", "target", "truth", "prediction"
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Existing minimal 1DCNN OOF reference lacks columns: {missing}")
    expected = labels.loc[:, ["participant_id", "condition", "presentation_position"]].copy()
    output: dict[str, Any] = {}
    for modality in FAMILIES_BY_MODALITY:
        subset = frame.loc[frame["combination"].eq(modality)].copy()
        if len(subset) != len(labels) * len(TARGETS):
            raise ValueError(f"Existing {modality} OOF reference does not cover all 135 labels and targets")
        if subset.duplicated(["participant_id", "condition", "target"]).any():
            raise ValueError(f"Existing {modality} OOF reference has duplicate labels")
        for target_index, target in enumerate(TARGETS):
            target_oof = subset.loc[subset["target"].eq(target)].copy()
            check = expected.merge(
                target_oof,
                on=["participant_id", "condition", "presentation_position"],
                how="left",
                validate="one_to_one",
            )
            if len(check) != len(labels) or check["prediction"].isna().any():
                raise ValueError(f"Existing {modality} OOF reference cannot be aligned to the source labels")
            expected_truth = labels[target].to_numpy(dtype=float)
            if not np.allclose(check["truth"].to_numpy(dtype=float), expected_truth):
                raise ValueError(f"Existing {modality} OOF truth does not match current source labels for {target}")
        output[modality] = subset
    return output


def _reference_oof_rows(reference: Any, modality: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in reference.itertuples(index=False):
        rows.append(
            {
                "modality": modality,
                "variant": "full_reference",
                "removed_family": "",
                "participant_id": str(row.participant_id),
                "condition": str(row.condition),
                "presentation_position": float(row.presentation_position),
                "target": str(row.target),
                "truth": float(row.truth),
                "prediction": float(row.prediction),
                "reference_source": f"{REFERENCE_DIRECTORY}/{REFERENCE_OOF_FILENAME}",
            }
        )
    return rows


def _metrics_for_oof(oof: Any) -> dict[str, float]:
    output: dict[str, float] = {}
    for target in TARGETS:
        subset = oof.loc[oof["target"].eq(target)]
        metrics = _point_metrics(
            subset["truth"].to_numpy(dtype=float),
            subset["prediction"].to_numpy(dtype=float),
            discomfort=target == "discomfort",
        )
        output.update({f"{target}_{name}": value for name, value in metrics.items()})
    return output


def _ablation_metric_rows(oof: Any, feature_counts: dict[tuple[str, str], int]) -> Any:
    pd, *_ = _dependencies()
    rows: list[dict[str, Any]] = []
    for modality in FAMILIES_BY_MODALITY:
        reference = oof.loc[
            oof["modality"].eq(modality) & oof["variant"].eq("full_reference")
        ]
        reference_metrics = _metrics_for_oof(reference)
        for variant, removed in (
            oof.loc[oof["modality"].eq(modality), ["variant", "removed_family"]]
            .drop_duplicates()
            .itertuples(index=False)
        ):
            subset = oof.loc[
                oof["modality"].eq(modality)
                & oof["variant"].eq(variant)
                & oof["removed_family"].eq(removed)
            ]
            metrics = _metrics_for_oof(subset)
            row = {
                "modality": modality,
                "variant": variant,
                "removed_family": removed,
                "removed_family_zh": FAMILY_LABELS_ZH.get(removed, ""),
                "research_only": True,
                "n_labels": int(len(subset) // len(TARGETS)),
                "n_participants": int(subset["participant_id"].nunique()),
                "feature_count_after_removal": feature_counts.get((modality, removed), feature_counts[(modality, "")]),
                "feature_removal_stage": "before_fold_selection_imputation_scaling_training",
                "reference_source": f"{REFERENCE_DIRECTORY}/{REFERENCE_OOF_FILENAME}",
                **metrics,
            }
            for name, value in reference_metrics.items():
                row[f"reference_{name}"] = value
                row[f"candidate_minus_reference_{name}"] = metrics[name] - value
            rows.append(row)
    return pd.DataFrame(rows)


def _subgroup_rows(oof: Any, eeg_disabled: set[str]) -> Any:
    pd, *_ = _dependencies()
    rows: list[dict[str, Any]] = []
    for modality, variant, removed in (
        oof.loc[:, ["modality", "variant", "removed_family"]].drop_duplicates().itertuples(index=False)
    ):
        variant_oof = oof.loc[
            oof["modality"].eq(modality)
            & oof["variant"].eq(variant)
            & oof["removed_family"].eq(removed)
        ]
        for subgroup, mask in (
            ("eeg_disabled", variant_oof["participant_id"].isin(eeg_disabled)),
            ("eeg_available", ~variant_oof["participant_id"].isin(eeg_disabled)),
        ):
            for target in TARGETS:
                subset = variant_oof.loc[mask & variant_oof["target"].eq(target)]
                metrics = _point_metrics(
                    subset["truth"].to_numpy(dtype=float),
                    subset["prediction"].to_numpy(dtype=float),
                    discomfort=target == "discomfort",
                )
                rows.append(
                    {
                        "modality": modality,
                        "variant": variant,
                        "removed_family": removed,
                        "subgroup": subgroup,
                        "target": target,
                        "research_only": True,
                        "n_participants": int(subset["participant_id"].nunique()),
                        "n_labels": int(len(subset)),
                        "n_high_discomfort_truth": (
                            int(np.sum(subset["truth"].to_numpy(dtype=float) >= HIGH_DISCOMFORT_TRUTH_THRESHOLD))
                            if target == "discomfort"
                            else 0
                        ),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _selection_stability(audit: Any) -> Any:
    pd, *_ = _dependencies()
    grouped = audit.groupby(["modality", "family", "target", "feature_name"], sort=True)
    rows: list[dict[str, Any]] = []
    for (modality, family, target, feature_name), group in grouped:
        selected = group["selected"].astype(bool)
        ranks = group.loc[group["selection_rank"].notna(), "selection_rank"]
        rows.append(
            {
                "modality": modality,
                "family": family,
                "family_zh": FAMILY_LABELS_ZH[family],
                "target": target,
                "feature_name": feature_name,
                "n_folds": int(len(group)),
                "selected_folds": int(selected.sum()),
                "selection_rate": float(selected.mean()),
                "mean_selection_rank_when_ranked": float(ranks.mean()) if len(ranks) else np.nan,
                "median_selection_rank_when_ranked": float(ranks.median()) if len(ranks) else np.nan,
                "mean_signed_pearson_r": float(group["signed_pearson_r"].mean()),
                "mean_absolute_pearson_r": float(group["absolute_pearson_r"].mean()),
                "mean_valid_window_count": float(group["valid_window_count"].mean()),
                "mean_availability_fraction": float(group["availability_fraction"].mean()),
                "available_for_ranking_folds": int(group["available_for_ranking"].astype(bool).sum()),
                "feature_limit_per_modality": FEATURES_PER_MODALITY,
                "feature_cap_binding": bool(group["feature_cap_binding"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _write_dataframe(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _format(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def _write_report(result: dict[str, Any], output: Path) -> None:
    stability = result["selection_stability"]
    ablations = result["family_ablation_metrics"]
    subgroup = result["subgroup_metrics"]
    h_count = int(result["family_counts"]["H"])
    p_count = int(result["family_counts"]["P"])
    reference = ablations.loc[ablations["variant"].eq("full_reference")]
    ref_rows = []
    for row in reference.itertuples(index=False):
        ref_rows.append(
            "| {modality} | {rel_mae} | {rel_rho} | {dis_mae} | {dis_rho} | {recall} | {fn} |".format(
                modality=row.modality,
                rel_mae=_format(row.relaxation_mae),
                rel_rho=_format(row.relaxation_spearman),
                dis_mae=_format(row.discomfort_mae),
                dis_rho=_format(row.discomfort_spearman),
                recall=_format(row.discomfort_high_recall),
                fn=_format(row.discomfort_high_false_negatives, 0),
            )
        )
    ablation_rows = []
    for row in ablations.loc[ablations["variant"].eq("remove_family")].itertuples(index=False):
        ablation_rows.append(
            "| {modality} | {family} | {rel:+.4f} | {dis:+.4f} | {recall:+.4f} | {fn:+.0f} |".format(
                modality=row.modality,
                family=row.removed_family_zh,
                rel=row.candidate_minus_reference_relaxation_mae,
                dis=row.candidate_minus_reference_discomfort_mae,
                recall=row.candidate_minus_reference_discomfort_high_recall,
                fn=row.candidate_minus_reference_discomfort_high_false_negatives,
            )
        )
    stability_rows = []
    for row in stability.sort_values(
        ["modality", "target", "selection_rate", "mean_absolute_pearson_r", "feature_name"],
        ascending=[True, True, False, False, True],
    ).groupby(["modality", "target"], sort=True).head(8).itertuples(index=False):
        stability_rows.append(
            "| {modality} | {target} | {feature} | {family} | {selected}/{folds} | {rank} | {corr} |".format(
                modality=row.modality,
                target=row.target,
                feature=row.feature_name,
                family=row.family_zh,
                selected=row.selected_folds,
                folds=row.n_folds,
                rank=_format(row.mean_selection_rank_when_ranked, 2),
                corr=_format(row.mean_absolute_pearson_r),
            )
        )
    subgroup_rows = []
    for row in subgroup.loc[subgroup["variant"].eq("full_reference")].itertuples(index=False):
        subgroup_rows.append(
            "| {modality} | {subgroup} | {target} | {people} | {labels} | {mae} | {rho} | {recall} |".format(
                modality=row.modality,
                subgroup=row.subgroup,
                target=row.target,
                people=row.n_participants,
                labels=row.n_labels,
                mae=_format(row.mae),
                rho=_format(row.spearman),
                recall=_format(row.high_recall),
            )
        )
    lines = [
        "# 最小融合 1DCNN 的 H/P 可解释性审计（研究专用）",
        "",
        "## 范围与不变项",
        "",
        "- 本审计复用最小融合 1DCNN 的 135 条 participant--Condition 标签、15 折 LOPO、"
        "每折 Condition-only 残差、8 个十秒窗口、配置随机种子、CUDA/CPU 设备、随机前缀和早停协议。",
        "- 完整 H/P 参考预测直接读取既有 `fusion_minimal_dcnn/oof_predictions.csv`；本命令不重训、"
        "不覆盖该基准，也不保存 checkpoint。",
        "- H/P 仅是研究性离线模型输入。它们不会改写运行时后端、自动推荐资格或 Shadow/hold 策略。",
        "",
        "## 特征家族与解释边界",
        "",
        "- H：平移速度、角速度、jerk、静止比例、位置范围、运动频谱熵。当前表只有 HMD 位置、"
        "由记录的角速度派生的角速度统计；**没有**头部朝向或 yaw/pitch/roll 方向变化特征。",
        "- P：ECG 速率/信号幅度、ECG HRV、`ecg_rr_std_ms_audit_only`、EEG 相对频带功率、"
        "EEG 绝对功率/幅度、EEG Hjorth/谱熵、EEG 比率/不对称性。",
        "- `ecg_rr_std_ms_audit_only` 仅作为独立敏感性家族；即使它在既有 P 基准中被选中，也不能"
        "被解释为可部署的生理证据。",
        f"- H 有 {h_count} 列，低于每模态 {FEATURES_PER_MODALITY} 列上限；因此“全折入选”只说明"
        f"上限未绑定，不能证明任一头动变量单独重要。P 有 {p_count} 列，选择上限会在折内生效。",
        "",
        "## 既有完整 H/P 参考 OOF",
        "",
        "| 模态 | Relaxation MAE | Relaxation Spearman | Discomfort MAE | Discomfort Spearman | 高 discomfort recall | false negatives |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *ref_rows,
        "",
        "## 折内选择稳定性（每个模态/目标列出前 8 项）",
        "",
        "Pearson 仅在训练参与者的有效十秒窗口与训练折残差之间计算；它是观察相关，不能解释为生理机制。",
        "",
        "| 模态 | 目标 | 特征 | 家族 | 入选折数 | 平均排名 | 平均 |r| |",
        "|---|---|---|---|---:|---:|---:|",
        *stability_rows,
        "",
        "完整折内审计见 `selection_audit.csv`；稳定性汇总见 `selection_stability.csv`。前者包含每个"
        "留出参与者、带符号/绝对相关、有效窗口数、可用性、排名及是否入选。",
        "",
        "## 家族移除消融",
        "",
        "每一行在特征选择、插补、标准化和训练**之前**删除对应家族；数值是“移除家族结果 − 完整参考结果”。"
        "MAE/false negatives 的正值表示移除后更差；recall 的负值表示移除后更差。这是模型内依赖检验，"
        "不是因果或生理机制结论。",
        "",
        "| 模态 | 移除家族 | Δ Relaxation MAE | Δ Discomfort MAE | Δ recall | Δ false negatives |",
        "|---|---|---:|---:|---:|---:|",
        *ablation_rows,
        "",
        "## EEG 分层的完整 H/P OOF",
        "",
        "下表只列完整参考。P 的 EEG-disabled 行仍包含可用 ECG；其样本量和标签数必须与具体 OOF 一起解读。",
        "",
        "| 模态 | 分层 | 目标 | 参与者数 | 标签数 | MAE | Spearman | 高 discomfort recall |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        *subgroup_rows,
        "",
        "## 结论",
        "",
        "本结果最多支持该 135 条样本上的离线观察相关、折内选择稳定性和模型内家族移除敏感性。"
        "它不证明头动或生理特征导致放松/不适，也不构成临床、生理因果或部署证据。"
        "现有运行时、自动推荐资格及 Shadow/hold 策略保持不变。",
    ]
    atomic_write_text(output, "\n".join(lines) + "\n")


def analyze_minimal_fusion_dcnn_hp(config: ProjectConfig) -> dict[str, Any]:
    """Write research-only fold audits and H/P feature-family ablations."""
    pd, *_ = _dependencies()
    source = config.path("features") / SOURCE_RELATIVE_PATH
    if not source.exists():
        raise FileNotFoundError(f"Minimal fusion 1DCNN input not found: {source}")
    architecture = _architecture(config)
    sequences = build_minimal_fusion_sequences(
        pd.read_csv(source),
        sequence_length=int(architecture["sequence_length"]),
        expected_labels=EXPECTED_LABELS,
    )
    _validate_family_coverage(sequences)
    labels = _labels_frame(sequences)
    reference = _load_reference_oof(config, labels)
    participants = sorted(labels["participant_id"].unique())
    random_seed = int(config.get("modeling.random_seed"))
    device = _device(config.get("modeling.dcnn.device", "cuda"))
    family_counts = {modality: len(_modality_feature_names(sequences, modality)) for modality in FAMILIES_BY_MODALITY}

    ablations: list[FamilyAblation] = []
    feature_counts: dict[tuple[str, str], int] = {}
    for modality, families in FAMILIES_BY_MODALITY.items():
        modality_names = _modality_feature_names(sequences, modality)
        feature_counts[(modality, "")] = len(modality_names)
        for family in families:
            retained = [name for name in modality_names if feature_family(modality, name) != family]
            if not retained:
                raise ValueError(f"Removing {family} leaves no {modality} features")
            feature_counts[(modality, family)] = len(retained)
            ablations.append(FamilyAblation(modality, family, _subset_sequences(sequences, retained)))

    audit_rows: list[dict[str, Any]] = []
    oof_rows = [row for modality, frame in reference.items() for row in _reference_oof_rows(frame, modality)]
    ablation_predictions = {
        (ablation.modality, ablation.removed_family): {
            target: np.full(len(labels), np.nan, dtype=float) for target in TARGETS
        }
        for ablation in ablations
    }

    for fold, participant in enumerate(participants, start=1):
        test_indexes = np.flatnonzero(labels["participant_id"].eq(participant).to_numpy())
        train_indexes = np.flatnonzero(~labels["participant_id"].eq(participant).to_numpy())
        train = labels.iloc[train_indexes].reset_index(drop=True)
        test = labels.iloc[test_indexes].reset_index(drop=True)
        for target_index, target in enumerate(TARGETS):
            baseline, baseline_map, fallback = _condition_baseline(train, test, target)
            train_baseline = np.asarray(
                [float(baseline_map.get(condition, fallback)) for condition in train["condition"]], dtype=float
            )
            residual = sequences.targets[train_indexes, target_index] - train_baseline
            residual_by_sequence = np.full(len(labels), np.nan, dtype=float)
            residual_by_sequence[train_indexes] = residual
            for modality in FAMILIES_BY_MODALITY:
                _selected, fold_audit = _rank_features_with_audit(
                    sequences,
                    train_indexes,
                    _modality_feature_names(sequences, modality),
                    residual,
                    modality=modality,
                )
                for row in fold_audit:
                    row.update(
                        {
                            "target": target,
                            "held_out_participant": participant,
                            "fold": fold,
                            "n_training_participants": int(train["participant_id"].nunique()),
                        }
                    )
                audit_rows.extend(fold_audit)

            for ablation in ablations:
                selected, _unused_audit = _rank_features_with_audit(
                    ablation.sequences,
                    train_indexes,
                    list(ablation.sequences.feature_columns),
                    residual,
                    modality=ablation.modality,
                )
                if not selected:
                    held_out_residual = np.zeros(len(test_indexes), dtype=float)
                else:
                    feature_lookup = {
                        name: index for index, name in enumerate(ablation.sequences.feature_columns)
                    }
                    feature_indexes = np.asarray([feature_lookup[name] for name in selected], dtype=int)
                    # Keep the existing H/P combination seed across family removals: only the
                    # permitted feature family changes, not fold seed or optimization protocol.
                    seed = _fold_seed(random_seed, target, ablation.modality, fold)
                    core_indexes, validation_indexes = _validation_indexes(
                        ablation.sequences.participant_ids, train_indexes, seed
                    )
                    model, scaler, _ = _train_residual_model(
                        ablation.sequences,
                        core_indexes,
                        validation_indexes,
                        feature_indexes,
                        residual_by_sequence,
                        architecture,
                        config,
                        seed,
                        device,
                    )
                    held_out_residual = _predict_residual_model(
                        model,
                        ablation.sequences,
                        test_indexes,
                        feature_indexes,
                        scaler,
                        device,
                    )
                ablation_predictions[(ablation.modality, ablation.removed_family)][target][test_indexes] = np.clip(
                    baseline + held_out_residual, 0.0, 1.0
                )
        print(f"H/P 1DCNN explainability LOPO fold {fold}/{len(participants)}: {participant}", flush=True)

    for (modality, family), predictions in ablation_predictions.items():
        for target, values in predictions.items():
            if np.isnan(values).any():
                raise AssertionError(f"{modality}/{family}/{target} ablation OOF is incomplete")
            for index, source_row in labels.iterrows():
                oof_rows.append(
                    {
                        "modality": modality,
                        "variant": "remove_family",
                        "removed_family": family,
                        "participant_id": str(source_row["participant_id"]),
                        "condition": str(source_row["condition"]),
                        "presentation_position": float(source_row["presentation_position"]),
                        "target": target,
                        "truth": float(source_row[target]),
                        "prediction": float(values[index]),
                        "reference_source": "new_family_ablation",
                    }
                )

    audit = pd.DataFrame(audit_rows)
    if len(audit) != len(participants) * len(TARGETS) * sum(family_counts.values()):
        raise AssertionError("H/P selection audit does not cover every fold, target, and candidate feature")
    oof = pd.DataFrame(oof_rows)
    expected_variants = sum(1 + len(families) for families in FAMILIES_BY_MODALITY.values())
    if len(oof) != expected_variants * len(labels) * len(TARGETS):
        raise AssertionError("H/P reference and ablation OOF rows are incomplete")
    stability = _selection_stability(audit)
    metrics = _ablation_metric_rows(oof, feature_counts)
    subgroups = _subgroup_rows(oof, set(config.get("participants.eeg_disabled", [])))

    if config.is_legacy:
        output_directory = config.path("artifacts") / OUTPUT_DIRECTORY
        audit_output = output_directory / SELECTION_AUDIT_FILENAME
        selection_output = output_directory / SELECTION_STABILITY_FILENAME
        ablation_output = output_directory / FAMILY_ABLATION_FILENAME
        subgroup_output = output_directory / SUBGROUP_FILENAME
        oof_output: Path | None = None
        report_output: Path | None = config.path("reports") / REPORT_FILENAME
    else:
        audit_output = config.path("metrics") / SELECTION_AUDIT_FILENAME
        selection_output = config.path("metrics") / SELECTION_STABILITY_FILENAME
        ablation_output = config.path("metrics") / FAMILY_ABLATION_FILENAME
        subgroup_output = config.path("metrics") / SUBGROUP_FILENAME
        oof_output = config.path("predictions") / "minimal_fusion_dcnn_hp_oof_predictions.csv"
        report_output = None
    _write_dataframe(audit, audit_output)
    _write_dataframe(stability, selection_output)
    _write_dataframe(metrics, ablation_output)
    _write_dataframe(subgroups, subgroup_output)
    if oof_output:
        _write_dataframe(oof, oof_output)
    result = {
        "research_only": True,
        "source": str(source),
        "reference_oof": str(_reference_oof_path(config)),
        "n_labels": int(len(labels)),
        "n_participants": int(len(participants)),
        "family_counts": family_counts,
        "selection_audit": audit,
        "selection_stability": stability,
        "family_ablation_metrics": metrics,
        "subgroup_metrics": subgroups,
        "oof_predictions": oof,
        "selection_audit_path": str(audit_output),
        "selection_stability_path": str(selection_output),
        "feature_family_ablation_metrics_path": str(ablation_output),
        "subgroup_metrics_path": str(subgroup_output),
        "oof_predictions_path": str(oof_output) if oof_output else None,
        "report": str(report_output) if report_output else None,
    }
    if report_output:
        _write_report(result, report_output)
    return result
