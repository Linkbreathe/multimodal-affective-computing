#!/usr/bin/env python3
"""Build the integrated frozen-feature compression/fusion evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts/relax/foundation_compression_fusion_20260717"
PRIOR_EEG = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/"
    "evaluation/aggregate_modality_effects_overall.csv"
)
METHODS = [
    "joint_block_balanced_pca8",
    "modality_pca2_additive",
    "supervised_pls2",
    "linear_shared_private",
    "modality_expert_simplex",
    "modality_expert_simplex_foundation_only",
]
METHOD_LABELS = {
    "joint_block_balanced_pca8": "Joint block-balanced PCA8",
    "modality_pca2_additive": "Modality-wise PCA2 + additive Ridge",
    "supervised_pls2": "Supervised PLS2",
    "linear_shared_private": "Linear shared-private",
    "modality_expert_simplex": "Modality expert simplex (hybrid)",
    "modality_expert_simplex_foundation_only": "Modality expert simplex (foundation only)",
}
BASELINES = [
    "fold_local_condition_only",
    "nonfoundation_global_relaxation_condition_discomfort",
    "historical_no_eeg_raw_dual",
    "historical_healnet_no_eeg",
]
BASELINE_LABELS = {
    "fold_local_condition_only": "Condition-only",
    "nonfoundation_global_relaxation_condition_discomfort": "Non-foundation hybrid",
    "historical_no_eeg_raw_dual": "Historical no_eeg_raw_dual",
    "historical_healnet_no_eeg": "Historical HEALNet no-EEG",
}
TARGETS = ["relaxation", "discomfort", "macro"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prior-eeg-summary", type=Path, default=PRIOR_EEG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return frame


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    return frame.to_markdown(index=False, floatfmt=floatfmt) if len(frame) else "_No rows._"


def main() -> None:
    args = arguments()
    root = args.artifact_root.resolve()
    evaluation = root / "evaluation"
    audit = root / "audit"
    output = args.output.resolve() if args.output else root / "foundation_compression_fusion_final_report.md"

    candidate = read_csv(evaluation / "candidate_summary.csv", ["candidate"])
    paired = read_csv(
        evaluation / "paired_statistics_18_test_holm.csv",
        [
            "candidate",
            "outcome",
            "mean_paired_delta",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "exact_sign_flip_p_two_sided",
            "holm_p",
            "holm_significant_0_05",
        ],
    )
    methods = read_csv(
        evaluation / "preregistered_method_summary.csv",
        [
            "candidate",
            "role",
            "modalities",
            "declared_compression_dimension",
            "formal_run_audit_passed",
            "candidate_point_success",
        ],
    )
    runs = read_csv(
        evaluation / "run_audit.csv",
        [
            "candidate",
            "seed",
            "valid",
            "observations",
            "folds",
            "common_valid_windows",
            "embedding_cuda_used",
            "embedding_device",
            "head_device",
            "runner_source_sha256",
            "compression_source_sha256",
            "dataset_source_sha256",
            "anchor_runner_source_sha256",
        ],
    )
    dimensions = read_csv(
        evaluation / "selected_dimension_summary.csv",
        [
            "candidate",
            "target",
            "scope",
            "parameter",
            "selections",
            "mean_value",
            "min_value",
            "max_value",
        ],
    )
    weights = read_csv(
        evaluation / "expert_weight_summary.csv",
        ["candidate", "target", "modality", "mean_weight", "min_weight", "max_weight"],
    )
    removals = read_csv(
        evaluation / "runner_validation_modality_ablation_summary.csv",
        [
            "candidate",
            "outcome",
            "modality",
            "mean_delta_without_minus_full",
            "fraction_removal_worsened_validation_mae",
        ],
    )
    head = read_csv(
        evaluation / "head_ablation_summary.csv",
        ["outcome", "mean_delta_foundation_only_minus_hybrid", "interpretation"],
    )
    modality = read_csv(
        audit / "modality_audit.csv",
        [
            "modality",
            "dimension",
            "internal_valid_windows",
            "common_valid_windows",
            "row_l2_median",
            "coordinate_sd_median",
            "entropy_effective_rank",
            "pc_count_95",
            "participant_feature_r2_descriptive",
            "condition_feature_r2_descriptive",
        ],
    )
    joint = read_csv(
        audit / "joint_pca8_fold_audit.csv",
        ["joint_input_dimension", "joint_explained_variance"]
        + [
            f"{name}_{suffix}"
            for name in ("ecg", "eye", "head", "video")
            for suffix in ("dimension_with_presence", "loading_mass_share")
        ],
    )
    probes = read_csv(
        audit / "fixed_target_information_probe.csv",
        ["modality", "target", "delta_vs_condition"],
    )
    redundancy = read_csv(
        audit / "cross_modal_redundancy_balanced_pca8_cka.csv",
        [
            "modality_a",
            "modality_b",
            "linear_cka_balanced_pca8",
            "linear_cka_balanced_pca8_after_participant_condition_residualization",
            "status",
        ],
    )
    evaluation_audit = read_json(evaluation / "evaluation_audit.json")
    figure_manifest = read_json(root / "figures/figure_manifest.json")
    if (
        evaluation_audit.get("status") != "passed"
        or not evaluation_audit["facts"]["figures_generated"]
        or figure_manifest.get("status") != "passed"
        or len(runs) != 18
        or not runs["valid"].all()
        or len(paired) != 18
        or paired["holm_significant_0_05"].any()
    ):
        raise ValueError("Final-report evidence gates did not pass")

    indexed = candidate.set_index("candidate")
    required = set(METHODS + BASELINES)
    if not required.issubset(indexed.index):
        raise ValueError("Candidate/comparator roster is incomplete")

    comparison_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for outcome in TARGETS:
            mae = float(indexed.loc[method, f"{outcome}_mae_mean"])
            row: dict[str, Any] = {
                "candidate": method,
                "display_name": METHOD_LABELS[method],
                "outcome": outcome,
                "mae": mae,
            }
            for baseline in BASELINES:
                row[f"delta_vs_{baseline}"] = (
                    mae - float(indexed.loc[baseline, f"{outcome}_mae_mean"])
                )
            comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(root / "all_baseline_comparison.csv", index=False)

    stability = candidate[candidate["candidate"].isin(METHODS)].copy()
    stability["display_name"] = stability["candidate"].map(METHOD_LABELS)
    stability_columns = ["candidate", "display_name"] + [
        f"{outcome}_{suffix}"
        for outcome in TARGETS
        for suffix in ("mae_mean", "mae_std")
    ]
    stability = stability[stability_columns]
    stability.to_csv(root / "three_seed_stability.csv", index=False)

    fixed = dimensions[
        dimensions["scope"].eq("fold")
        & dimensions["parameter"].eq("compression_dimension")
    ].copy()
    fixed["display_name"] = fixed["candidate"].map(METHOD_LABELS)
    fixed.to_csv(root / "fixed_dimensions_by_fold_target.csv", index=False)

    capacity_proxy = {
        "joint_block_balanced_pca8": "20 Ridge coefficients",
        "modality_pca2_additive": "20 Ridge coefficients",
        "supervised_pls2": "8 final Ridge coefficients + nested PLS state",
        "linear_shared_private": "20 Ridge coefficients",
        "modality_expert_simplex": "≤32 label-conditioned coefficients/weights",
        "modality_expert_simplex_foundation_only": "≤24 label-conditioned coefficients/weights",
    }
    capacity = methods.copy()
    capacity["display_name"] = capacity["candidate"].map(METHOD_LABELS)
    capacity["label_conditioned_capacity_proxy"] = capacity["candidate"].map(capacity_proxy)
    capacity.to_csv(root / "method_capacity_summary.csv", index=False)

    fallback_rows: list[dict[str, Any]] = []
    raw_root = root.parent / "condition_anchor_residual_20260717/runs/no_eeg_raw_dual"
    for path in sorted(raw_root.glob("seed_*/*_results.json")):
        payload = read_json(path)
        for target in ("relaxation", "discomfort"):
            target_rows = [fold["targets"][target] for fold in payload["folds"]]
            fallback_rows.append(
                {
                    "seed": int(payload["seed"]),
                    "target": target,
                    "folds": len(target_rows),
                    "gamma_zero_fallback_folds": sum(
                        bool(row["fallback"]) and float(row["gamma"]) == 0.0
                        for row in target_rows
                    ),
                    "eligible_under_new_positive_gamma_rule": all(
                        (not bool(row["fallback"])) and float(row["gamma"]) > 0
                        for row in target_rows
                    ),
                    "source_sha256": digest(path),
                }
            )
    fallback = pd.DataFrame(fallback_rows)
    if len(fallback) != 6:
        raise ValueError("Historical raw-dual fallback audit is incomplete")
    fallback.to_csv(root / "historical_raw_dual_fallback_audit.csv", index=False)

    prior_eeg_text = "The prior cross-architecture EEG-removal summary was unavailable."
    if args.prior_eeg_summary.is_file():
        prior = read_csv(
            args.prior_eeg_summary,
            [
                "configuration",
                "removed_modality",
                "mean_delta_across_architectures_and_targets",
                "negative_architecture_target_pairs",
                "holm_significant_tests",
            ],
        )
        eeg = prior[
            prior["configuration"].eq("no_eeg")
            & prior["removed_modality"].eq("eeg")
        ].iloc[0]
        prior_eeg_text = (
            "In the prior 108-run neural ablation, removing EEG had a descriptive "
            f"mean delta of {float(eeg['mean_delta_across_architectures_and_targets']):+.6f} "
            "across architectures and targets "
            f"({int(eeg['negative_architecture_target_pairs'])}/12 pairs improved after removal; "
            f"{int(eeg['holm_significant_tests'])} Holm-significant tests)."
        )

    baseline_table = pd.DataFrame(
        [
            {
                "comparator": BASELINE_LABELS[name],
                "relaxation_mae": indexed.loc[name, "relaxation_mae_mean"],
                "discomfort_mae": indexed.loc[name, "discomfort_mae_mean"],
                "macro_mae": indexed.loc[name, "macro_mae_mean"],
                "role": (
                    "canonical"
                    if name == "fold_local_condition_only"
                    else "control"
                    if name == "nonfoundation_global_relaxation_condition_discomfort"
                    else "historical descriptive"
                ),
            }
            for name in BASELINES
        ]
    )
    modality_table = modality[
        [
            "modality",
            "dimension",
            "internal_valid_windows",
            "common_valid_windows",
            "row_l2_median",
            "coordinate_sd_median",
            "entropy_effective_rank",
            "pc_count_95",
            "participant_feature_r2_descriptive",
            "condition_feature_r2_descriptive",
        ]
    ]
    joint_mean = joint.mean(numeric_only=True)
    coordinate_share = {
        name: float(joint_mean[f"{name}_dimension_with_presence"])
        / float(joint_mean["joint_input_dimension"])
        for name in ("ecg", "eye", "head", "video")
    }
    loading_share = {
        name: float(joint_mean[f"{name}_loading_mass_share"])
        for name in ("ecg", "eye", "head", "video")
    }

    primary = indexed.loc["modality_expert_simplex"]
    shared = indexed.loc["linear_shared_private"]
    foundation = indexed.loc["modality_expert_simplex_foundation_only"]
    condition = indexed.loc["fold_local_condition_only"]
    joint_method = indexed.loc["joint_block_balanced_pca8"]
    modality_method = indexed.loc["modality_pca2_additive"]
    expert_shared_gap = float(primary["macro_mae_mean"] - shared["macro_mae_mean"])
    modality_minus_joint = float(
        modality_method["macro_mae_mean"] - joint_method["macro_mae_mean"]
    )
    eeg_probe = probes[probes["modality"].eq("eeg")].set_index("target")

    effects = paired.copy()
    effects["method"] = effects["candidate"].map(METHOD_LABELS)
    effects = effects[
        [
            "method",
            "outcome",
            "mean_paired_delta",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "exact_sign_flip_p_two_sided",
            "holm_p",
            "holm_significant_0_05",
        ]
    ].sort_values(["outcome", "mean_paired_delta"])
    primary_weights = weights[weights["candidate"].eq("modality_expert_simplex")][
        ["target", "modality", "mean_weight", "min_weight", "max_weight"]
    ].sort_values(["target", "mean_weight"], ascending=[True, False])
    primary_removals = removals[
        removals["candidate"].eq("modality_expert_simplex")
    ][
        [
            "outcome",
            "modality",
            "mean_delta_without_minus_full",
            "fraction_removal_worsened_validation_mae",
        ]
    ].sort_values(["outcome", "mean_delta_without_minus_full"], ascending=[True, False])

    hashes = {
        "cache": evaluation_audit["input_contract"]["embedding_cache_sha256"],
        "runner": runs.iloc[0]["runner_source_sha256"],
        "compression": runs.iloc[0]["compression_source_sha256"],
        "dataset": runs.iloc[0]["dataset_source_sha256"],
        "anchor": runs.iloc[0]["anchor_runner_source_sha256"],
    }

    lines: list[str] = []

    def add(*items: str) -> None:
        lines.extend(items)

    add(
        "# Frozen Multimodal Foundation-Feature Compression and Fusion",
        "",
        "## Executive verdict",
        "",
        "**Fact.** No new compression/fusion candidate supports a confirmed improvement over "
        "fold-local Condition-only. All 18 participant-level candidate-outcome bootstrap intervals "
        "cross zero and all Holm-adjusted p-values equal 1.0.",
        "",
        f"**Fact.** The internally locked primary, modality expert simplex, passed every eligibility "
        f"rule and had the lowest new-method macro point estimate ({float(primary['macro_mae_mean']):.6f} "
        f"versus {float(condition['macro_mae_mean']):.6f}; Δ {float(primary['macro_delta_mean']):+.6f}). "
        f"It improved discomfort ({float(primary['discomfort_delta_mean']):+.6f}) but worsened relaxation "
        f"({float(primary['relaxation_delta_mean']):+.6f}), so the both-target success gate failed.",
        "",
        f"**Inference.** The primary is not substantively better than linear shared-private; their macro "
        f"difference is only {expert_shared_gap:+.9f}. If one method is frozen for an independent cohort, "
        "use the expert model because it was the pre-outcome primary and handles unavailable experts "
        "deterministically—not because this negligible internal gap proves superiority.",
        "",
        "**Recommendation.** Retain Condition-only as the benchmark. Do not claim that foundation-feature "
        "fusion outperforms it on this cohort.",
        "",
        "## Evidence labels and protocol",
        "",
        "- **Facts** come directly from the cache, manifests, result JSONs, OOF predictions, or CSVs.",
        "- **Inferences** are bounded interpretations; **recommendations** are next-step decisions.",
        "- The original literature review and feature audit preceded the internal method lock. The explicit "
        "ICLR citation and equal-width CKA export are marked post-execution supplements and did not alter "
        "methods or inference.",
        "",
        "Participants: P003, P004, P007, P008, P009, P011, P012, P013, P015. The fixed protocol is "
        "nine 7-train/1-validation/1-test folds, 81 participant-condition observations, 545 common-valid "
        "windows, and seeds 20260705, 20260706, 20260707. P004/C6 is retained but has no common-valid window.",
        "",
        f"All neural feature extraction used CUDA on {runs.iloc[0]['embedding_device']}. New reducers and "
        f"Ridge heads are intentionally classical CPU methods; no encoder was retrained. Cache SHA-256: "
        f"`{hashes['cache']}`.",
        "",
        "## Literature-to-design evidence",
        "",
        "The targeted review contains 29 tabled primary works plus one ICLR addendum across NeurIPS, ICML, "
        "ICLR, CVPR, ICCV, ECCV, ACL/EMNLP, AAAI, ACM MM, TPAMI, statistics, and medical imaging. "
        "See [the full fact-versus-suitability matrix](literature/frozen_embedding_compression_fusion_review.md).",
        "",
        "- [Supervised principal components](https://doi.org/10.1198/016214505000000628) and "
        "[sparse PLS](https://doi.org/10.1111/j.1467-9868.2009.00723.x) motivate strictly nested "
        "supervised linear reduction.",
        "- [Efficient low-rank multimodal fusion](https://aclanthology.org/P18-1209/) motivates a low-rank "
        "capacity ceiling, not tensor expansion here.",
        "- [Domain Separation Networks](https://proceedings.neurips.cc/paper_files/paper/2016/hash/"
        "45fbc6d3e05ebd93369ce542e8f2322d-Abstract.html), "
        "[MISA](https://doi.org/10.1145/3394171.3413678), and "
        "[ICLR factorized representations](https://openreview.net/forum?id=rygqqsA9KX) support "
        "shared/private structure; only a fixed linear analogue is credible with 63 training rows.",
        "- [Gradient Blending](https://openaccess.thecvf.com/content_CVPR_2020/html/"
        "Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html) and "
        "[quality-aware fusion](https://proceedings.mlr.press/v202/zhang23ar.html) motivate separate "
        "experts and global reliability weighting, not a neural router.",
        "- [Perceiver](https://proceedings.mlr.press/v139/jaegle21a.html), "
        "[attention bottlenecks](https://proceedings.neurips.cc/paper/2021/hash/"
        "76ba9f564ebbc35b1014ac498fafadd0-Abstract.html), and "
        "[BLIP-2](https://proceedings.mlr.press/v202/li23q.html) show bottleneck success at scale; "
        "they do not make a Transformer estimable at seven training participants.",
        "- [HeMIS](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_54), "
        "[ModDrop](https://doi.org/10.1109/TPAMI.2015.2461544), and "
        "[MultiBench](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/"
        "hash/37693cfc748049e45d87b8c7d8b9aacd-Abstract-round1.html) motivate explicit availability, "
        "ablation, capacity, and participant-clustered reporting.",
        "",
        "**Inference.** Neural attention, Q-Former, tensor fusion, per-observation gates, and neural MoE "
        "were correctly excluded. This is a five-family capacity-controlled comparison, not a broad sweep.",
        "",
        "## Frozen-feature audit",
        "",
        table(modality_table, ".4f"),
        "",
        "**Facts.** Raw norms/scales differ sharply and effective ranks are far below raw widths. EEG alone "
        "has 22 internally missing windows; the common mask enforces the same 545-window intersection.",
        "",
        f"The current no-EEG concatenation has 1,942 coordinates. Train-only PCA8 retains "
        f"{float(joint_mean['joint_explained_variance']):.2%} of standardized variance, so "
        f"{1-float(joint_mean['joint_explained_variance']):.2%} is discarded. ECG+video contribute "
        f"{coordinate_share['ecg']+coordinate_share['video']:.2%} of coordinates and receive "
        f"{loading_share['ecg']+loading_share['video']:.2%} of loading mass.",
        "",
        "**Inference.** This is mainly width-proportional dominance, not proof of intrinsic superiority. "
        "The audit proves variance is discarded; it does not prove discarded variance is target-useful. "
        "That remains unresolved without a locked higher-budget sensitivity or independent cohort.",
        "",
        "### Cross-modal redundancy",
        "",
        table(
            redundancy[
                [
                    "modality_a",
                    "modality_b",
                    "linear_cka_balanced_pca8",
                    "linear_cka_balanced_pca8_after_participant_condition_residualization",
                ]
            ],
            ".4f",
        ),
        "",
        "**Fact.** Within the formal no-EEG panel, residualized CKA ranges from 0.0917 (ECG-head) "
        "to 0.3040 (eye-video). Across all five audited modalities, EEG-eye is highest (0.5315).",
        "",
        "**Inference.** This shows neither complete redundancy nor independence and does not establish "
        "complementary predictive information.",
        "",
        "### Target probes and EEG scope",
        "",
        table(probes.sort_values(["target", "delta_vs_condition"])),
        "",
        f"EEG's fixed fold-safe probe worsened relaxation by "
        f"{float(eeg_probe.loc['relaxation','delta_vs_condition']):+.6f} and discomfort by "
        f"{float(eeg_probe.loc['discomfort','delta_vs_condition']):+.6f}. {prior_eeg_text}",
        "",
        "**Scope boundary.** Those descriptive facts motivated the existing no-EEG panel; they do not prove "
        "EEG is useless. Adding EEG after outcomes would be a post hoc seventh candidate and was not done.",
        "",
        "## Locked method panel and capacity",
        "",
        table(
            capacity[
                [
                    "display_name",
                    "role",
                    "modalities",
                    "declared_compression_dimension",
                    "label_conditioned_capacity_proxy",
                    "formal_run_audit_passed",
                    "candidate_point_success",
                ]
            ],
            ".0f",
        ),
        "",
        "Dimensions are fixed in every fold, not validation-selected. Capacity proxies count target-conditioned "
        "coefficients/weights and exclude frozen encoders and unsupervised scaler/PCA state.",
        "",
        table(
            fixed[
                [
                    "display_name",
                    "target",
                    "selections",
                    "mean_value",
                    "min_value",
                    "max_value",
                ]
            ],
            ".1f",
        ),
        "",
        "## Implementation and execution audit",
        "",
        "- `src/fusion/frozen_compression.py` implements deterministic train-index-only reducers.",
        "- `scripts/run_relax_compression_fusion.py` implements two-target residual heads, nested experts, "
        "positive gamma, clipping, availability renormalization, and eligibility checks.",
        "- `scripts/evaluate_relax_compression_fusion.py` audits coverage/hashes and runs participant bootstrap, "
        "exact sign flips, and the 18-test Holm family.",
        "",
        f"All 18/18 runs have 81 unique OOF keys, nine folds, and 545 common-valid windows. All cache "
        f"provenance is CUDA and all new heads report CPU. Identical source hashes: runner "
        f"`{hashes['runner']}`, compression `{hashes['compression']}`, dataset `{hashes['dataset']}`, "
        f"anchor `{hashes['anchor']}`.",
        "",
        "Every scaler, reducer, supervised projection, expert inner prediction, alpha/gamma choice, and "
        "calibration excludes the outer test participant. The primary learns both targets, uses positive "
        "foundation weight floors, and never falls back; P004/C6 is the all-missing exception.",
        "",
        "## Results against all requested comparators",
        "",
        table(baseline_table),
        "",
        "Negative deltas below mean lower participant-macro MAE. Only Condition comparisons belong to the "
        "locked 18-test inferential family.",
        "",
    )

    delta_names = {
        "fold_local_condition_only": "Δ vs Condition",
        "nonfoundation_global_relaxation_condition_discomfort": "Δ vs non-foundation hybrid",
        "historical_no_eeg_raw_dual": "Δ vs raw dual",
        "historical_healnet_no_eeg": "Δ vs HEALNet",
    }
    for outcome in TARGETS:
        subset = comparisons[comparisons["outcome"].eq(outcome)]
        display = pd.DataFrame(
            {
                "method": subset["display_name"],
                "MAE": subset["mae"],
                **{
                    label: subset[f"delta_vs_{name}"]
                    for name, label in delta_names.items()
                },
            }
        )
        add(f"### {outcome.capitalize()}", "", table(display), "")

    add(
        "### Three-seed stability",
        "",
        table(
            stability[
                [
                    "display_name",
                    "relaxation_mae_mean",
                    "relaxation_mae_std",
                    "discomfort_mae_mean",
                    "discomfort_mae_std",
                    "macro_mae_mean",
                    "macro_mae_std",
                ]
            ]
        ),
        "",
        "All new methods are deterministic across seeds (zero or floating-roundoff SD). Seeds verify "
        "reproducibility; they are not additional subjects.",
        "",
        "### Participant-level inference and multiplicity",
        "",
        "Seeds were averaged within participant. Intervals use 10,000 participant-cluster bootstrap "
        "resamples; p-values enumerate all 512 sign assignments; Holm covers six candidates × three outcomes.",
        "",
        table(effects),
        "",
        "![Formal participant-level outcomes](figures/figure1_formal_outcomes.png)",
        "",
        "## Compression and modality diagnostics",
        "",
        f"**Joint versus modality-wise.** Joint block-balanced PCA8 macro MAE is "
        f"{float(joint_method['macro_mae_mean']):.6f}; modality-wise PCA2 is "
        f"{float(modality_method['macro_mae_mean']):.6f}. Modality-wise minus joint is "
        f"{modality_minus_joint:+.6f}; PCA2 did not win this matched eight-axis comparison. "
        "This is not a general rejection of modality-wise fusion.",
        "",
        "**Expert weights (descriptive):**",
        "",
        table(primary_weights),
        "",
        "**Outer-validation removal checks (descriptive):**",
        "",
        table(primary_removals),
        "",
        f"Removing head changes hybrid-versus-foundation-only macro MAE by "
        f"{float(head.loc[head['outcome'].eq('macro'),'mean_delta_foundation_only_minus_hybrid'].iloc[0]):+.6f}; "
        "the relaxation difference is neutral at evaluator tolerance. These are tiny diagnostics, not causal effects.",
        "",
        "![Feature and modality audit](figures/figure2_feature_and_modality_audit.png)",
        "",
        "![Cross-modal redundancy supplement](figures/figure3_cross_modal_redundancy.png)",
        "",
        "## Direct answers",
        "",
        "1. **Most appropriate strategy:** no established winner. Expert simplex is the one exploratory "
        "candidate to freeze for external validation because it was the locked primary, passes both-target "
        "eligibility, protects modality blocks, and handles unavailable experts. Shared-private is practically tied.",
        f"2. **Modality-wise versus joint PCA:** PCA2 is worse by {modality_minus_joint:+.6f} macro MAE in the "
        "matched eight-axis implementations.",
        "3. **Genuine complementarity:** not established. ECG/video have the clearest positive relaxation "
        "removal signals; ECG/video/head have small inconsistent discomfort signals; eye has no positive "
        "removal evidence. Weights, CKA, and removals are descriptive.",
        f"4. **Foundation fusion versus Condition:** no supported win. Foundation-only macro Δ is "
        f"{float(foundation['macro_delta_mean']):+.6f}, relaxation worsens by "
        f"{float(foundation['relaxation_delta_mean']):+.6f}, the both-target gate fails, and no test is corrected-significant.",
        "5. **Next validation:** freeze expert simplex and this evaluator before an independent participant "
        "cohort; keep shared-private secondary and Condition-only canonical.",
        "",
        "## Historical comparator boundaries",
        "",
        "Historical no_eeg_raw_dual has the best descriptive point estimates but was adaptively selected "
        "and violates the new positive-gamma rule. It falls back to Condition in four of nine folds for "
        "each target in every seed:",
        "",
        table(
            fallback[
                [
                    "seed",
                    "target",
                    "folds",
                    "gamma_zero_fallback_folds",
                    "eligible_under_new_positive_gamma_rule",
                ]
            ]
        ),
        "",
        "The non-foundation hybrid uses global relaxation plus direct Condition discomfort, so it is a "
        "control, not an eligible two-target foundation pipeline. HEALNet no-EEG is the historical "
        "validation-selected CUDA neural comparator; it is worse than Condition here and remains descriptive.",
        "",
        "## Limitations and do-not-claim boundaries",
        "",
        "- Nine participant clusters are insufficient for a stable method ranking or broad modality claims.",
        "- The same cohort informed audit, internal design, and evaluation; this is not external confirmation.",
        "- Seeds do not increase biological sample size.",
        "- The common mask cannot validate a learned quality gate or arbitrary missing-modality robustness.",
        "- CKA is a post hoc descriptive export outside the selection and multiplicity families.",
        "- Historical comparators remain adaptively selected and descriptive.",
        "- Do not claim target-useful discarded variance, causal expert weights, useless EEG, genuine "
        "complementarity, or a confirmed winner.",
        "",
        "## Reproduction",
        "",
        "Working directory: `/home/link/Wei/Models/core/real-time-vis-physio-fusion`",
        "",
        "```bash",
        "source /home/link/miniconda3/etc/profile.d/conda.sh",
        "conda activate egoEMOTION",
        "",
        "python scripts/audit_relax_foundation_features.py \\",
        "  --embedding-cache artifacts/relax/aligned_20260716/condition_embeddings.pt \\",
        "  --mask-manifest /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract/common_valid_window_masks.csv \\",
        "  --split-manifest /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract/split_manifest.csv \\",
        "  --cohorts /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract/cohorts.json \\",
        "  --output-dir artifacts/relax/foundation_compression_fusion_20260717/audit",
        "",
        "python scripts/run_relax_compression_fusion_matrix.py \\",
        "  --contract-dir /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract \\",
        "  --embedding-cache artifacts/relax/aligned_20260716/condition_embeddings.pt \\",
        "  --output-root artifacts/relax/foundation_compression_fusion_20260717",
        "",
        "python scripts/plot_relax_compression_fusion.py",
        "python scripts/run_relax_compression_fusion_matrix.py \\",
        "  --contract-dir /mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract \\",
        "  --output-root artifacts/relax/foundation_compression_fusion_20260717 \\",
        "  --evaluate-only",
        "python scripts/build_relax_compression_fusion_final_report.py",
        "```",
        "",
        "The matrix skips existing results unless `--force` is supplied. Run plotting before the final "
        "evaluate-only pass so the evaluator records the figure bundle.",
        "",
        "## Artifact index",
        "",
        "- [Primary-source review](literature/frozen_embedding_compression_fusion_review.md)",
        "- [Feature audit](audit/feature_audit_report.md)",
        "- [Method lock](preregistration/method_preregistration.md)",
        "- [Formal evaluation](evaluation/compression_fusion_evaluation_report.md)",
        "- [Evaluation audit](evaluation/evaluation_audit.json)",
        "- [All-baseline comparison](all_baseline_comparison.csv)",
        "- [Seed stability](three_seed_stability.csv)",
        "- [Fixed dimensions](fixed_dimensions_by_fold_target.csv)",
        "- [Capacity summary](method_capacity_summary.csv)",
        "- [Historical fallback audit](historical_raw_dual_fallback_audit.csv)",
        "- [Figure QA and source data](figures/figure_qa_notes.md)",
        "",
    )

    output.write_text("\n".join(lines), encoding="utf-8")
    generated = [
        "all_baseline_comparison.csv",
        "three_seed_stability.csv",
        "fixed_dimensions_by_fold_target.csv",
        "method_capacity_summary.csv",
        "historical_raw_dual_fallback_audit.csv",
    ]
    manifest = {
        "status": "passed",
        "report": str(output),
        "report_sha256": digest(output),
        "formal_runs": len(runs),
        "formal_tests": len(paired),
        "holm_significant_tests": int(paired["holm_significant_0_05"].sum()),
        "figures_recognized": bool(evaluation_audit["facts"]["figures_generated"]),
        "primary_success": False,
        "generated_tables": {name: digest(root / name) for name in generated},
    }
    (root / "final_report_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
