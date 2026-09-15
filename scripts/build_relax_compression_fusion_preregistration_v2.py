"""Freeze the July 18 five-modality method and inference contract.

The builder must run after the evidence review and feature audit but before any
candidate outcome.  It refuses to write a preregistration when candidate result
or prediction files are already present below the output root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd

from scripts.run_relax_foundation_probe import file_sha256
from scripts.run_relax_condition_anchor_probe import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_OBSERVATIONS,
    EXPECTED_PARTICIPANTS,
    EXPECTED_VALID_WINDOWS,
)


MODALITIES = ("eeg", "ecg", "eye", "head", "video")
SEEDS = (20260705, 20260706, 20260707)
METHOD_NAMES = (
    "joint_block_balanced_pca12",
    "modality_rank_alloc_pca12",
    "targetwise_modality_pls1",
    "linear_gcca_shared_private",
    "modality_expert_simplex5",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _candidate_outputs(root: Path) -> list[str]:
    """Return outcome-like files, excluding evidence/audit/preregistration."""
    found: list[str] = []
    excluded_roots = {"literature", "audit", "preregistration"}
    for path in root.rglob("*") if root.exists() else []:
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in excluded_roots:
            continue
        name = path.name.lower()
        method_named = any(method in name for method in METHOD_NAMES)
        outcome_named = (
            name.endswith("_results.json")
            or name.endswith("_predictions.csv")
            or name == "matrix_status.json"
            or "candidate_comparison" in name
            or "participant_statistics" in name
        )
        if method_named or outcome_named:
            found.append(str(relative))
    return sorted(found)


def _require_hash(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return file_sha256(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    outcomes = _candidate_outputs(args.output_root)
    if outcomes:
        raise RuntimeError(
            "Refusing to create a pre-outcome preregistration after candidate outputs exist: "
            + ", ".join(outcomes[:20])
        )

    literature_sha = _require_hash(args.literature, "literature review")
    audit_sha = _require_hash(args.audit_summary, "feature audit summary")
    audit_report_sha = _require_hash(args.audit_report, "feature audit report")
    cache_sha = _require_hash(args.embedding_cache, "embedding cache")
    if cache_sha != EXPECTED_CACHE_SHA256:
        raise ValueError(f"Frozen cache hash mismatch: {cache_sha}")

    audit = json.loads(args.audit_summary.read_text(encoding="utf-8"))
    protocol = audit.get("protocol", {})
    if tuple(protocol.get("participants", ())) != EXPECTED_PARTICIPANTS:
        raise ValueError("Audit does not certify the fixed EEG-eligible cohort")
    if int(protocol.get("observations", -1)) != EXPECTED_OBSERVATIONS:
        raise ValueError("Audit does not certify 81 observations")
    if int(protocol.get("common_valid_windows", -1)) != EXPECTED_VALID_WINDOWS:
        raise ValueError("Audit does not certify 545 common-valid windows")
    if audit.get("shared_missingness", {}).get("zero_common_window_observations") != [
        {"participant_id": "P004", "condition": "C6"}
    ]:
        raise ValueError("Audit all-missing exception changed")

    chronology = {
        "literature_mtime_utc": datetime.fromtimestamp(
            args.literature.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "audit_summary_mtime_utc": datetime.fromtimestamp(
            args.audit_summary.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "preregistration_created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.literature.stat().st_mtime > args.audit_summary.stat().st_mtime:
        raise RuntimeError("Literature review must predate the completed audit summary")

    methods = [
        {
            "name": "joint_block_balanced_pca12",
            "family": "joint_compression_after_concatenation",
            "formal_modalities": list(MODALITIES),
            "definition": "Outer-train coordinate standardization within each modality; multiply each block by 1/sqrt(raw dimension); concatenate 2,962 window coordinates; weighted PCA12; condition-average scores; append shared presence and quality; separate Ridge residual heads.",
            "compression_dimension": 12,
            "head_input_dimension": 14,
            "parameter_count": {
                "unsupervised_pca_loadings": 35544,
                "label_conditioned_ridge_coefficients_including_intercepts": 30,
            },
            "literature_basis": ["PCA regularization", "block-balanced multimodal bottleneck control"],
            "expected_advantage": "Can retain cross-modal covariance while preventing raw block width from setting total variance mass.",
            "risk": "Shared axes may still preserve participant nuisance and discard modality-private target signal.",
        },
        {
            "name": "modality_rank_alloc_pca12",
            "family": "modality_wise_compression_before_fusion",
            "formal_modalities": list(MODALITIES),
            "definition": "Independent condition-balanced outer-train window PCA per modality; allocate fold-specific k_m with sum 12 from entropy effective rank; condition-average scores; append shared presence and quality; separate Ridge residual heads.",
            "compression_dimension": 12,
            "head_input_dimension": 14,
            "dimension_allocation": "Give every modality one axis; allocate the remaining seven proportional to outer-train entropy effective rank by Hamilton largest remainder; cap by numeric rank and deterministically redistribute overflow; assert sum(k_m)=12.",
            "parameter_count": {
                "unsupervised_pca_loadings": "sum_m raw_dimension_m * k_m, recorded per fold",
                "label_conditioned_ridge_coefficients_including_intercepts": 30,
            },
            "literature_basis": ["protected modality bottlenecks", "multimodal representation-collapse avoidance"],
            "expected_advantage": "Protects small eye/head blocks while assigning more axes to higher-rank ECG, EEG, or video.",
            "risk": "The 12-axis total may still underrepresent high-rank windows; effective rank may retain nuisance rather than target signal.",
        },
        {
            "name": "targetwise_modality_pls1",
            "family": "supervised_compression",
            "formal_modalities": list(MODALITIES),
            "definition": "Begin with the same 12 modality-wise PCA condition scores; for each target and modality fit one outer-train PLS1 direction to cross-fitted Condition residuals; concatenate five target-specific scores with shared presence and quality; fit one Ridge residual head.",
            "compression_dimension_per_target": 5,
            "head_input_dimension_per_target": 7,
            "parameter_count": {
                "label_conditioned_capacity_upper_bound": 40,
                "note": "PLS directions plus Ridge coefficients; exact count recorded per fold",
            },
            "literature_basis": ["Supervised Principal Components", "Sparse PLS"],
            "expected_advantage": "Spends only one direction per modality on covariance with the target residual.",
            "risk": "Outcome-aware directions can be unstable with seven training participants even under strict nesting.",
        },
        {
            "name": "linear_gcca_shared_private",
            "family": "shared_private_decomposition",
            "formal_modalities": list(MODALITIES),
            "definition": "From modality PCA condition scores, fit outer-train Ledoit-Wolf-regularized MAXVAR-GCCA with two shared components; retain one private residual PC per modality; concatenate two shared plus five private scores with shared presence and quality; separate Ridge residual heads.",
            "compression_dimension": 7,
            "head_input_dimension": 9,
            "parameter_count": {
                "reduced_state_parameters_approximate": 36,
                "label_conditioned_ridge_coefficients_including_intercepts": 20,
            },
            "literature_basis": ["JIVE", "Group Factor Analysis", "Domain Separation Networks", "factorized multimodal representations"],
            "expected_advantage": "Tests genuinely shared variation without erasing a private factor from any modality.",
            "risk": "Shared factors can primarily encode participant or condition identity; covariance regularization may dominate tiny folds.",
        },
        {
            "name": "modality_expert_simplex5",
            "family": "modality_specific_experts_weighted_fusion",
            "formal_modalities": list(MODALITIES),
            "definition": "For each modality and target fit an independent Ridge residual expert on its allocated PCA scores. Learn target-specific global nonnegative simplex weights from participant-wise inner-LOPO predictions within the seven outer-training participants. Require every one of the five weights >=0.02 and sum to one; renormalize deterministically over available experts.",
            "compression_dimension": 12,
            "expert_count_per_target": 5,
            "parameter_count": {
                "label_conditioned_capacity_approximate": 43,
                "breakdown": "about 34 expert coefficients, 8 independent target-specific weight degrees, and one shared gamma",
            },
            "literature_basis": ["Gradient-Blending", "quality-aware multimodal fusion", "HeMIS", "linear mixture of experts"],
            "expected_advantage": "Separately regularizes each modality and lets relaxation and discomfort use different global reliability profiles.",
            "risk": "Inner-LOPO weights can remain noisy with only seven participants; the positive floor proves participation, not complementarity.",
        },
    ]

    payload: dict[str, Any] = {
        "schema_version": "relax_foundation_compression_preregistration_v2",
        "frozen_before_candidate_outcomes": True,
        "candidate_outcome_scan": {"found": outcomes, "status": "none_found"},
        "chronology": chronology,
        "literature_sha256": literature_sha,
        "audit_sha256": audit_sha,
        "audit_report_sha256": audit_report_sha,
        "protocol": {
            "participants": list(EXPECTED_PARTICIPANTS),
            "folds": 9,
            "train_participants": 7,
            "validation_participants": 1,
            "test_participants": 1,
            "observations": EXPECTED_OBSERVATIONS,
            "nominal_windows": 567,
            "common_valid_windows": EXPECTED_VALID_WINDOWS,
            "seeds": list(SEEDS),
            "independent_supervision_unit": "participant-condition observation, never a window",
            "encoder_training": False,
            "outer_test_exclusion": "All normalization, covariance, PCA, rank allocation, PLS, GCCA, private projection, Ridge fitting, expert-weight fitting, and alpha/gamma selection exclude the outer test participant.",
        },
        "input_contract": {
            "embedding_cache_sha256": cache_sha,
            "labels_sha256": _require_hash(args.labels, "labels"),
            "windows_sha256": _require_hash(args.windows, "windows"),
            "split_manifest_sha256": _require_hash(args.split_manifest, "split manifest"),
            "common_mask_sha256": _require_hash(args.mask_manifest, "common mask"),
            "cohorts_sha256": _require_hash(args.cohorts, "cohort manifest"),
            "formal_modalities": list(MODALITIES),
            "embedding_dimensions": {"eeg": 1024, "ecg": 1024, "eye": 128, "head": 18, "video": 768},
            "raw_coordinate_total": 2962,
            "head_status": "18 handcrafted Project-A window features; full model is a hybrid foundation-feature pipeline",
            "shared_missingness_policy": "The formal masks are identical. Append one shared presence and one shared common_valid/source_windows quality field, not five duplicate availability signals.",
        },
        "common_model_rules": {
            "window_covariance_weighting": "Each nonempty outer-training participant-condition has total weight one across its common-valid windows.",
            "compress_windows_then_pool": True,
            "supervised_reduction_unit": "condition rows only",
            "training_anchor": "same-condition mean of the other six outer-training participants",
            "validation_test_anchor": "same-condition mean of all seven outer-training participants",
            "targets": ["relaxation", "discomfort"],
            "ridge_alphas": [10.0, 100.0, 1000.0],
            "positive_gammas": [0.25, 0.5, 0.75, 1.0],
            "gamma_zero_allowed": False,
            "residual_cap": 0.2,
            "total_latent_budget": 12,
            "expert_weight_floor": 0.02,
            "selection": "minimize outer-validation Macro MAE using one alpha shared across targets and one gamma shared across targets; exact ties prefer stronger alpha then smaller positive gamma",
            "prediction_bounds": [0.0, 1.0],
            "all_missing_exception": "P004/C6 must receive exactly zero feature correction and the fold-local Condition anchor; this is the only authorized fallback.",
            "foundation_contribution": "Positive gamma is mandatory and nonzero feature corrections must be reported for both targets. Directly copying Condition-only for either target is forbidden.",
        },
        "methods": methods,
        "primary_candidate": "modality_expert_simplex5",
        "primary_eligibility": {
            "formal_modalities": list(MODALITIES),
            "all_five_expert_branches_trained": True,
            "both_targets_learned": True,
            "each_weight_minimum": 0.02,
            "positive_gamma": True,
            "condition_fallback": "forbidden except P004/C6 all-missing zero-correction rule",
        },
        "modality_ablations": [
            {"variant": "no_eeg", "deleted_modality": "eeg", "modalities": ["ecg", "eye", "head", "video"]},
            {"variant": "no_ecg", "deleted_modality": "ecg", "modalities": ["eeg", "eye", "head", "video"]},
            {"variant": "no_eye", "deleted_modality": "eye", "modalities": ["eeg", "ecg", "head", "video"]},
            {"variant": "no_head", "deleted_modality": "head", "modalities": ["eeg", "ecg", "eye", "video"]},
            {"variant": "no_video", "deleted_modality": "video", "modalities": ["eeg", "ecg", "eye", "head"]},
        ],
        "ablation_rules": {
            "role": "mechanism ablation only; no reduced model may replace the all-five primary claim",
            "retrain": True,
            "same_folds_and_seeds": True,
            "same_total_latent_budget": 12,
            "weight_floor": "0.02 over every remaining expert, followed by sum-to-one constraint",
            "no_subset_sweep": "Only the five leave-one-modality-out variants are authorized; no 31-subset search.",
            "genuine_complementarity": "Requires deletion of at least two different modalities to worsen performance consistently; nonzero weights alone are insufficient.",
        },
        "comparators": [
            "fold_local_condition_only",
            "nonfoundation_global_relaxation_condition_discomfort",
            "historical_no_eeg_raw_dual",
            "historical_healnet_no_eeg",
            "july17_best_fusion_descriptive",
        ],
        "evaluation": {
            "primary_metric": "participant-macro Macro MAE",
            "target_metrics": ["relaxation MAE", "discomfort MAE"],
            "seed_stability": "mean, sample standard deviation, min, max, and per-seed delta",
            "participant_inference": "average seeds within participant; 10,000 participant-cluster bootstrap draws; exact 512 participant sign flips",
            "primary_test": "all-five modality_expert_simplex5 versus Condition-only Macro MAE",
            "method_multiplicity": "Holm correction across the five full-five methods versus Condition-only for Macro MAE",
            "target_multiplicity": "Holm correction across relaxation and discomfort for the primary",
            "ablation_multiplicity": "separate Holm family of five deletions for each outcome",
            "direct_strategy_contrast": "paired modality_rank_alloc_pca12 minus joint_block_balanced_pca12 Macro MAE; target contrasts descriptive",
            "discomfort_strata": "report MAE separately for the 65 zero and 16 nonzero discomfort labels",
            "historical_comparators": "descriptive only because prior selection was adaptive",
        },
        "success_rules": {
            "descriptive": "Primary Macro delta versus Condition is negative for all three seeds, mean relaxation and discomfort deltas are both <=0, and feature corrections are nonzero for both targets outside P004/C6.",
            "stronger_support": "Descriptive rule plus participant-cluster Macro delta confidence interval entirely below zero and exact sign-flip p-value passing the declared primary inference.",
            "failure_action": "If the all-five primary fails, retain Condition-only and do not post-hoc promote a reduced ablation or another full method.",
            "claim_scope": "exploratory internal validation on the same nine participants used for development",
        },
        "source_sha256": {
            "preregistration_builder": file_sha256(__file__),
            "audit_script": _require_hash(args.audit_script, "audit script"),
        },
    }
    _write_json(args.output_json, payload)

    method_rows = [
        {
            "method": method["name"],
            "family": method["family"],
            "compressed dimension": method.get(
                "compression_dimension", method.get("compression_dimension_per_target")
            ),
            "head input": method.get("head_input_dimension", method.get("head_input_dimension_per_target")),
            "label-conditioned capacity": method["parameter_count"],
        }
        for method in methods
    ]
    markdown = [
        "# July 18 Five-Modality Compression/Fusion Preregistration",
        "",
        f"Frozen at `{chronology['preregistration_created_utc']}` after literature SHA `{literature_sha}` and audit SHA `{audit_sha}`, with no candidate outcome files present.",
        "",
        "## Fixed protocol",
        "",
        "Nine EEG-eligible participants; 81 participant-condition labels; 545 common-valid windows; fixed nine-fold 7-train/1-validation/1-test LOPO; seeds 20260705, 20260706, and 20260707. Windows estimate frozen-feature covariance only and are never counted as independent labels.",
        "",
        "Every primary method includes EEG, ECG, eye, head, and video and learns residual corrections for both relaxation and discomfort. All transforms and selections exclude the outer test participant.",
        "",
        "## Locked methods",
        "",
        pd.DataFrame(method_rows).to_markdown(index=False),
        "",
        "The all-five `modality_expert_simplex5` is the sole primary. The other four all-five methods are predeclared comparisons. The only reduced-modality runs are retrained `no_eeg`, `no_ecg`, `no_eye`, `no_head`, and `no_video` ablations of the primary.",
        "",
        "## Locked common rules",
        "",
        "- Total unsupervised latent budget: 12.",
        "- Ridge alpha: 10, 100, or 1000; gamma: 0.25, 0.5, 0.75, or 1.0. Gamma zero is forbidden.",
        "- Select one shared alpha/gamma pair by validation Macro MAE; ties prefer stronger alpha then smaller gamma.",
        "- Clip residual corrections to +/-0.2 and final predictions to [0,1].",
        "- P004/C6 is the only all-missing case and must receive exactly the Condition anchor (zero feature correction).",
        "- Expert weights are target-specific, global, nonnegative, at least 0.02 each, and sum to one.",
        "",
        "## Locked inference and claim gate",
        "",
        "Average seeds within participant, use 10,000 participant-cluster bootstrap draws and all 512 participant sign flips. Holm-correct five full-method Macro comparisons; correct the primary's two target tests; correct five ablations in separate outcome families.",
        "",
        "A descriptive all-five claim requires negative Macro delta in every seed, nonworse mean MAE for both targets, and nonzero learned corrections for both targets. Stronger support additionally requires the participant-cluster Macro confidence interval below zero and the declared exact test. If the primary fails, Condition-only remains the model; no ablation or secondary method may be promoted after inspection.",
        "",
    ]
    args.output_markdown.write_text("\n".join(markdown), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--literature", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
