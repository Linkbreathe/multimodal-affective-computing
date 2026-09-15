"""Audit and evaluate the preregistered frozen-feature compression/fusion matrix.

The evaluator is intentionally fail-closed.  It recomputes the canonical fold-local
Condition baseline from the authoritative labels and split manifest, verifies every
new OOF run against that contract, and only then writes evaluation products.  The
six-candidate family and its 18 paired tests come from the frozen preregistration;
historical comparators are reported descriptively and never enter that Holm family.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd


TARGETS = ("relaxation", "discomfort")
OUTCOMES = (*TARGETS, "macro")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260717
FLOAT_ATOL = 5e-7
CORRECTION_ATOL = 1e-12
ZERO_FEATURE_EXCEPTIONS = {("P004", "C6")}
FOUNDATION_MODALITIES = ("ecg", "eye", "video")

DEFAULT_PREREGISTRATION = (
    ROOT
    / "artifacts/relax/foundation_compression_fusion_20260717/preregistration"
    / "method_preregistration.json"
)
DEFAULT_NO_EEG_ROOT = (
    ROOT / "artifacts/relax/condition_anchor_residual_20260717/runs/no_eeg_raw_dual"
)
DEFAULT_HEALNET_ROOT = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/"
    "runs/healnet/no_eeg"
)


def _interpret_head_ablation_delta(delta: float) -> str:
    """Interpret foundation-only minus hybrid MAE without overclaiming ties."""
    if delta > FLOAT_ATOL:
        return "positive_delta_supports_incremental_head_feature_value"
    if delta < -FLOAT_ATOL:
        return "negative_delta_does_not_support_incremental_head_feature_value"
    return "near_zero_delta_is_neutral_for_incremental_head_feature_value"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _candidate_name(payload: Mapping[str, Any]) -> str:
    candidate = payload.get("candidate")
    if isinstance(candidate, Mapping):
        return str(candidate.get("name", ""))
    return str(candidate or payload.get("model", ""))


def _candidate_modalities(payload: Mapping[str, Any], prereg_method: Mapping[str, Any]) -> list[str]:
    candidate = payload.get("candidate")
    if isinstance(candidate, Mapping) and isinstance(candidate.get("modalities"), Sequence):
        return [str(value) for value in candidate["modalities"]]
    if isinstance(payload.get("modalities"), Sequence) and not isinstance(payload.get("modalities"), str):
        return [str(value) for value in payload["modalities"]]
    text = prereg_method.get("modalities")
    if isinstance(text, Sequence) and not isinstance(text, str):
        return [str(value) for value in text]
    definition = str(prereg_method.get("definition", "")).lower()
    return [value for value in ("ecg", "eye", "head", "video") if value in definition]


def _load_preregistration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing preregistration: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "relax_foundation_compression_preregistration_v1":
        raise ValueError(f"Unexpected preregistration schema: {payload.get('schema_version')!r}")
    methods = payload.get("methods")
    sensitivity = payload.get("foundation_only_sensitivity")
    if not isinstance(methods, list) or len(methods) != 5 or not isinstance(sensitivity, Mapping):
        raise ValueError("Preregistration must contain five methods and one foundation-only sensitivity")
    names = [str(method.get("name")) for method in methods] + [str(sensitivity.get("name"))]
    if len(names) != 6 or len(set(names)) != 6:
        raise ValueError(f"Expected six unique candidate names, got {names}")
    protocol = payload.get("protocol", {})
    if protocol.get("folds") != 9 or protocol.get("observations") != 81:
        raise ValueError("Preregistration does not declare the fixed nine-fold/81-observation protocol")
    if list(protocol.get("seeds", [])) != [20260705, 20260706, 20260707]:
        raise ValueError(f"Unexpected seed contract: {protocol.get('seeds')}")
    if payload.get("evaluation", {}).get("multiplicity", "").find("18 tests") < 0:
        raise ValueError("Preregistration does not declare the 18-test Holm family")
    declared = {str(method["name"]): dict(method) for method in methods}
    sensitivity_parent = str(sensitivity.get("method_family", ""))
    if sensitivity_parent not in declared:
        raise ValueError(f"Unknown sensitivity parent method: {sensitivity_parent!r}")
    payload["_candidate_names"] = names
    payload["_method_registry"] = {
        **declared,
        str(sensitivity["name"]): {
            **declared[sensitivity_parent],
            **dict(sensitivity),
            "name": str(sensitivity["name"]),
            "family": str(declared[sensitivity_parent]["family"]),
            "definition": str(sensitivity.get("purpose", "foundation-only sensitivity")),
        },
    }
    payload["_path"] = str(path.resolve())
    payload["_sha256"] = file_sha256(path)
    return payload


def _load_contract(
    labels_path: Path,
    split_path: Path,
    prereg: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    labels = pd.read_csv(labels_path)
    required_labels = {"participant_id", "condition", "presentation_position", *TARGETS}
    missing = required_labels - set(labels.columns)
    if missing:
        raise ValueError(f"Labels are missing required columns: {sorted(missing)}")
    labels = labels.copy()
    labels["participant_id"] = labels["participant_id"].astype(str)
    labels["condition"] = labels["condition"].astype(str)
    labels = labels.sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    key_columns = ["participant_id", "condition"]
    participants = [str(value) for value in prereg["protocol"]["participants"]]
    if len(labels) != 81 or labels.duplicated(key_columns).any():
        raise ValueError("Canonical labels must contain exactly 81 unique participant-condition rows")
    if set(labels["participant_id"]) != set(participants):
        raise ValueError("Canonical label participants differ from the preregistered cohort")
    if not np.isfinite(labels[list(TARGETS)].to_numpy(dtype=float)).all():
        raise ValueError("Canonical target labels contain non-finite values")

    split = pd.read_csv(split_path)
    required_split = {"fold_index", "test_participant", "validation_participant", "participant_id", "role"}
    missing = required_split - set(split.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    split = split.copy()
    for column in ("test_participant", "validation_participant", "participant_id", "role"):
        split[column] = split[column].astype(str)
    split["fold_index"] = split["fold_index"].astype(int)

    folds: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    for fold_index, frame in split.groupby("fold_index", sort=True):
        train = frame.loc[frame["role"] == "train", "participant_id"].tolist()
        validation = frame.loc[frame["role"] == "validation", "participant_id"].tolist()
        test = frame.loc[frame["role"] == "test", "participant_id"].tolist()
        if len(train) != 7 or len(validation) != 1 or len(test) != 1:
            raise ValueError(f"Fold {fold_index} is not 7/1/1")
        test_participant = test[0]
        validation_participant = validation[0]
        if set(frame["test_participant"]) != {test_participant}:
            raise ValueError(f"Fold {fold_index} test column disagrees with roles")
        if set(frame["validation_participant"]) != {validation_participant}:
            raise ValueError(f"Fold {fold_index} validation column disagrees with roles")
        if set(train + validation + test) != set(participants):
            raise ValueError(f"Fold {fold_index} does not cover the exact cohort")
        folds.append(
            {
                "fold_index": int(fold_index),
                "train_participants": train,
                "validation_participant": validation_participant,
                "test_participant": test_participant,
            }
        )
        train_labels = labels[labels["participant_id"].isin(train)]
        test_labels = labels[labels["participant_id"] == test_participant]
        if len(train_labels) != 63 or len(test_labels) != 9:
            raise ValueError(f"Fold {fold_index} does not map to 63 train / 9 test labels")
        condition_means = train_labels.groupby("condition")[list(TARGETS)].mean()
        global_relaxation = float(train_labels["relaxation"].mean())
        for row in test_labels.itertuples(index=False):
            canonical_rows.append(
                {
                    "participant_id": str(row.participant_id),
                    "condition": str(row.condition),
                    "presentation_position": float(row.presentation_position),
                    "fold_index": int(fold_index),
                    "test_participant": test_participant,
                    "validation_participant": validation_participant,
                    "relaxation_true": float(row.relaxation),
                    "discomfort_true": float(row.discomfort),
                    "condition_only_relaxation": float(condition_means.loc[str(row.condition), "relaxation"]),
                    "condition_only_discomfort": float(condition_means.loc[str(row.condition), "discomfort"]),
                    "nonfoundation_relaxation": global_relaxation,
                    "nonfoundation_discomfort": float(condition_means.loc[str(row.condition), "discomfort"]),
                }
            )
    canonical = pd.DataFrame(canonical_rows).sort_values(key_columns).reset_index(drop=True)
    if len(folds) != 9 or len(canonical) != 81 or canonical.duplicated(key_columns).any():
        raise ValueError("Split manifest failed the nine-fold canonical OOF reconstruction")
    return labels, folds, canonical


def _discover_formal_artifacts(
    runs_dir: Path,
    candidate_names: Sequence[str],
    seeds: Sequence[int],
) -> dict[tuple[str, int], tuple[Path, Path, dict[str, Any]]]:
    expected = {(candidate, int(seed)) for candidate in candidate_names for seed in seeds}
    discovered: dict[tuple[str, int], tuple[Path, Path, dict[str, Any]]] = {}
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Missing formal runs directory: {runs_dir}")
    for result_path in sorted(runs_dir.rglob("*_results.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot parse formal result {result_path}: {error}") from error
        pair = (_candidate_name(payload), int(payload.get("seed", -1)))
        if pair not in expected:
            continue
        prediction_path = result_path.with_name(
            result_path.name.removesuffix("_results.json") + "_predictions.csv"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing OOF predictions paired with {result_path}: {prediction_path}")
        if pair in discovered:
            raise ValueError(f"Duplicate formal result identity {pair}: {discovered[pair][0]} and {result_path}")
        discovered[pair] = (result_path, prediction_path, payload)
    if set(discovered) != expected:
        raise FileNotFoundError(
            f"Formal matrix is incomplete: missing={sorted(expected-set(discovered))}; "
            f"observed={sorted(discovered)}"
        )
    return discovered


def _contract_input_hash(contract: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    inputs = contract.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    for alias in aliases:
        record = inputs.get(alias)
        if isinstance(record, Mapping) and record.get("sha256"):
            return str(record["sha256"])
    return None


def _weight_map(fold: Mapping[str, Any], target_record: Mapping[str, Any], target: str) -> dict[str, float]:
    keys = ("expert_weights", "simplex_weights", "modality_weights", "weights")
    for source in (target_record, fold):
        for key in keys:
            value = source.get(key)
            if isinstance(value, Mapping) and target in value and isinstance(value[target], Mapping):
                value = value[target]
            if isinstance(value, Mapping):
                parsed: dict[str, float] = {}
                for modality, weight in value.items():
                    try:
                        parsed[str(modality)] = float(weight)
                    except (TypeError, ValueError):
                        parsed = {}
                        break
                if parsed:
                    return parsed
    return {}


def _selected_dimension_rows(
    candidate: str,
    seed: int,
    fold: Mapping[str, Any],
    target: str,
    target_record: Mapping[str, Any],
    declared_dimension: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(scope: str, parameter: str, value: Any) -> None:
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            rows.append(
                {
                    "candidate": candidate,
                    "seed": seed,
                    "fold_index": int(fold.get("fold_index", -1)),
                    "target": target,
                    "scope": scope,
                    "parameter": parameter,
                    "value": float(value),
                }
            )

    add("preregistered", "compression_dimension", declared_dimension)
    add("fold", "compression_dimension", fold.get("compression_dimension"))
    transform = fold.get("transform")
    if not isinstance(transform, Mapping):
        transform = fold.get("compression") if isinstance(fold.get("compression"), Mapping) else {}
    dimension_keys = (
        "compression_dimension",
        "output_dimension",
        "latent_dimension",
        "selected_dimension",
        "n_components",
        "pca_components",
        "pls_components",
        "shared_components",
        "private_components",
    )
    for key in dimension_keys:
        add("fold_transform", key, transform.get(key))
        add("target_selection", key, target_record.get(key))
    for container_name in (
        "modality_components",
        "selected_dimensions",
        "component_counts",
        "components_per_modality",
        "pre_pca_components_per_modality",
        "private_components_per_modality",
    ):
        for source_name, source in (("fold_transform", transform), ("target_selection", target_record)):
            values = source.get(container_name)
            if isinstance(values, Mapping):
                for modality, value in values.items():
                    add(source_name, f"{container_name}.{modality}", value)

    def walk_dimensions(value: Any, prefix: str = "") -> None:
        if not isinstance(value, Mapping):
            return
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            key_text = str(key).lower()
            if isinstance(child, (int, float, np.integer, np.floating)) and (
                "component" in key_text or "dimension" in key_text
            ):
                add("compressor_provenance", child_prefix, child)
            elif isinstance(child, Mapping):
                walk_dimensions(child, child_prefix)

    walk_dimensions(transform)
    return rows


def _ablation_rows_from_fold(
    candidate: str,
    seed: int,
    fold: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw = fold.get("modality_ablations")
    if raw is None:
        raw = fold.get("validation_modality_ablations")
    if raw is None:
        return []
    rows: list[dict[str, Any]] = []
    if isinstance(raw, Mapping) and any(target in raw for target in TARGETS):
        for target in TARGETS:
            target_record = raw.get(target)
            if not isinstance(target_record, Mapping):
                continue
            full = target_record.get("full_validation_mae", target_record.get("full_mae"))
            without = target_record.get("without", {})
            if not isinstance(without, Mapping):
                continue
            for modality, ablated in without.items():
                if not isinstance(ablated, Mapping):
                    continue
                removed = ablated.get("validation_mae", ablated.get("ablated_mae"))
                delta = ablated.get("delta_without_minus_full", ablated.get("delta"))
                if delta is None and full is not None and removed is not None:
                    delta = float(removed) - float(full)
                if delta is None:
                    continue
                rows.append(
                    {
                        "candidate": candidate,
                        "seed": seed,
                        "fold_index": int(fold.get("fold_index", -1)),
                        "modality": str(modality),
                        "outcome": target,
                        "validation_mae_full": np.nan if full is None else float(full),
                        "validation_mae_without": np.nan if removed is None else float(removed),
                        "delta_without_minus_full": float(delta),
                        "source": "runner_validation_ablation",
                    }
                )
        return rows
    entries: list[Mapping[str, Any]] = []
    if isinstance(raw, list):
        entries = [value for value in raw if isinstance(value, Mapping)]
    elif isinstance(raw, Mapping):
        for modality, value in raw.items():
            if isinstance(value, Mapping) and any(target in value for target in TARGETS):
                for target, target_value in value.items():
                    if target in TARGETS and isinstance(target_value, Mapping):
                        entries.append({"modality": modality, "target": target, **target_value})
            elif isinstance(value, Mapping):
                entries.append({"modality": modality, **value})
    for entry in entries:
        target = str(entry.get("target", entry.get("outcome", "macro")))
        modality = str(entry.get("modality", entry.get("removed_modality", "")))
        full = entry.get("validation_mae_full", entry.get("full_mae"))
        removed = entry.get("validation_mae_without", entry.get("ablated_mae"))
        delta = entry.get("delta_without_minus_full", entry.get("delta"))
        if delta is None and full is not None and removed is not None:
            delta = float(removed) - float(full)
        if not modality or target not in OUTCOMES or delta is None:
            continue
        rows.append(
            {
                "candidate": candidate,
                "seed": seed,
                "fold_index": int(fold.get("fold_index", -1)),
                "modality": modality,
                "outcome": target,
                "validation_mae_full": np.nan if full is None else float(full),
                "validation_mae_without": np.nan if removed is None else float(removed),
                "delta_without_minus_full": float(delta),
                "source": "runner_validation_ablation",
            }
        )
    return rows


def _audit_new_runs(
    artifacts: Mapping[tuple[str, int], tuple[Path, Path, dict[str, Any]]],
    prereg: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected_by_fold = {int(fold["fold_index"]): fold for fold in folds}
    canonical_indexed = canonical.set_index(["participant_id", "condition"]).sort_index()
    contract_hashes = prereg["input_contract"]
    candidate_registry = prereg["_method_registry"]
    primary = str(prereg["primary_candidate"])
    foundation_only = str(prereg["foundation_only_sensitivity"]["name"])
    expert_candidates = {primary, foundation_only}
    primary_rules = prereg["primary_eligibility"]
    min_positive_weight = 0.05

    prediction_frames: list[pd.DataFrame] = []
    payloads: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    runner_ablation_rows: list[dict[str, Any]] = []
    reference_contract_hashes: dict[str, str | None] | None = None
    reference_source_hashes: dict[str, str] | None = None
    expected_source_paths = {
        "runner": ROOT / "scripts/run_relax_compression_fusion.py",
        "compression": ROOT / "src/fusion/frozen_compression.py",
        "dataset": ROOT / "src/data/relax_dataset.py",
        "anchor_runner": ROOT / "scripts/run_relax_condition_anchor_probe.py",
    }
    current_source_hashes = {
        key: file_sha256(path) for key, path in expected_source_paths.items()
    }

    for (candidate, seed), (result_path, prediction_path, payload) in sorted(artifacts.items()):
        issues: list[str] = []
        method = candidate_registry[candidate]
        modalities = _candidate_modalities(payload, method)
        try:
            frame = pd.read_csv(prediction_path)
        except Exception as error:  # pragma: no cover - reported with path context
            raise ValueError(f"Cannot read {prediction_path}: {error}") from error
        required_columns = {
            "participant_id",
            "condition",
            "fold_index",
            "seed",
            "candidate",
            "relaxation_true",
            "discomfort_true",
            "relaxation_pred",
            "discomfort_pred",
            "condition_only_relaxation",
            "condition_only_discomfort",
        }
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            issues.append(f"missing_prediction_columns:{','.join(sorted(missing_columns))}")
        else:
            frame = frame.copy()
            frame["participant_id"] = frame["participant_id"].astype(str)
            frame["condition"] = frame["condition"].astype(str)
            frame["fold_index"] = frame["fold_index"].astype(int)
            keys = list(zip(frame["participant_id"], frame["condition"], strict=True))
            coverage_ok = len(frame) == 81 and len(set(keys)) == 81
            membership_ok = set(keys) == set(canonical_indexed.index.tolist())
            if not coverage_ok:
                issues.append("oof_key_coverage")
            if not membership_ok:
                issues.append("oof_key_membership")
            if set(frame["candidate"].astype(str)) != {candidate} or set(frame["seed"].astype(int)) != {seed}:
                issues.append("prediction_identity")
            if coverage_ok and membership_ok:
                aligned = frame.set_index(["participant_id", "condition"]).sort_index()
                for column in (
                    "fold_index",
                    "relaxation_true",
                    "discomfort_true",
                    "condition_only_relaxation",
                    "condition_only_discomfort",
                ):
                    if column == "fold_index":
                        equal = np.array_equal(
                            aligned[column].to_numpy(dtype=int),
                            canonical_indexed[column].to_numpy(dtype=int),
                        )
                    else:
                        equal = np.allclose(
                            aligned[column].to_numpy(dtype=float),
                            canonical_indexed[column].to_numpy(dtype=float),
                            atol=FLOAT_ATOL,
                            rtol=0.0,
                        )
                    if not equal:
                        issues.append(f"canonical_{column}")
                prediction_values = aligned[[f"{target}_pred" for target in TARGETS]].to_numpy(dtype=float)
                if not np.isfinite(prediction_values).all():
                    issues.append("nonfinite_predictions")
                if np.any((prediction_values < -FLOAT_ATOL) | (prediction_values > 1.0 + FLOAT_ATOL)):
                    issues.append("prediction_bounds")
                for target in TARGETS:
                    fallback_column = f"{target}_fallback"
                    gamma_column = f"{target}_gamma"
                    if fallback_column in aligned and aligned[fallback_column].map(_as_bool).any():
                        issues.append(f"{target}_prediction_fallback")
                    if gamma_column in aligned and not (aligned[gamma_column].astype(float) > 0.0).all():
                        issues.append(f"{target}_prediction_gamma_nonpositive")

        if payload.get("schema_version") != "relax_foundation_compression_fusion_run_v1":
            issues.append("result_schema")
        if _candidate_name(payload) != candidate or int(payload.get("seed", -1)) != seed:
            issues.append("result_identity")
        if int(payload.get("observation_count", -1)) != 81 or int(payload.get("fold_count", -1)) != 9:
            issues.append("result_counts")
        if str(payload.get("prediction_sha256", "")) != file_sha256(prediction_path):
            issues.append("prediction_hash")
        payload_preregistration = payload.get("preregistration", {})
        if not isinstance(payload_preregistration, Mapping) or str(
            payload_preregistration.get("sha256", "")
        ) != str(prereg["_sha256"]):
            issues.append("preregistration_hash")
        payload_source_hashes = payload.get("source_sha256")
        if not isinstance(payload_source_hashes, Mapping):
            issues.append("source_hash_records")
            payload_source_hashes = {}
        observed_source_hashes = {
            key: str(payload_source_hashes.get(key, "")) for key in expected_source_paths
        }
        for key, expected_hash in current_source_hashes.items():
            if observed_source_hashes[key] != expected_hash:
                issues.append(f"source_hash_{key}")
        if reference_source_hashes is None:
            reference_source_hashes = observed_source_hashes
        elif observed_source_hashes != reference_source_hashes:
            issues.append("source_hash_cross_run")

        contract = payload.get("contract", {})
        current_hashes = {
            "embedding_cache": _contract_input_hash(contract, ("embedding_cache", "cache")),
            "labels": _contract_input_hash(contract, ("labels", "condition_labels")),
            "split_manifest": _contract_input_hash(contract, ("split_manifest", "splits")),
            "common_mask": _contract_input_hash(contract, ("mask_manifest", "common_mask", "common_valid_window_masks")),
        }
        for key, expected_key in (
            ("embedding_cache", "embedding_cache_sha256"),
            ("labels", "labels_sha256"),
            ("split_manifest", "split_manifest_sha256"),
            ("common_mask", "common_mask_sha256"),
        ):
            if current_hashes[key] != str(contract_hashes[expected_key]):
                issues.append(f"contract_hash_{key}")
        if reference_contract_hashes is None:
            reference_contract_hashes = current_hashes
        elif current_hashes != reference_contract_hashes:
            issues.append("contract_hash_cross_run")
        if int(contract.get("common_valid_window_count", -1)) != 545:
            issues.append("common_valid_window_count")

        runtime = payload.get("head_runtime", {})
        provenance = payload.get("embedding_provenance", {})
        cuda_used = _as_bool(runtime.get("embedding_extraction_cuda_used"))
        provenance_cuda = _as_bool(provenance.get("cuda_used", cuda_used))
        provenance_device = str(provenance.get("device", "cuda" if provenance_cuda else ""))
        embedding_device = str(
            runtime.get("embedding_extraction_device", provenance.get("cuda_device_name", provenance_device))
        )
        if not cuda_used or not provenance_cuda or "cuda" not in provenance_device.lower():
            issues.append("embedding_cuda_provenance")
        if not embedding_device:
            issues.append("embedding_device_missing")
        if _as_bool(runtime.get("neural_training_performed")):
            issues.append("unexpected_neural_training")
        if str(runtime.get("device", "")).lower() != "cpu":
            issues.append("head_device")

        eligibility = payload.get("eligibility")
        if not isinstance(eligibility, Mapping):
            issues.append("eligibility_record")
            eligibility = {}
        if not _as_bool(eligibility.get("all_folds_eligible")):
            issues.append("all_folds_not_eligible")
        if not _as_bool(eligibility.get("both_targets_learned_in_every_fold")):
            issues.append("eligibility_target_learning")
        if not _as_bool(eligibility.get("gamma_strictly_positive_in_every_fold")):
            issues.append("eligibility_gamma")
        if _as_bool(eligibility.get("condition_fallback_used")):
            issues.append("eligibility_condition_fallback")
        if candidate == primary and not _as_bool(
            eligibility.get("nonzero_correction_both_targets_in_every_fold")
        ):
            issues.append("primary_nonzero_correction")

        payload_folds = payload.get("folds")
        if not isinstance(payload_folds, list) or len(payload_folds) != 9:
            issues.append("fold_records")
            payload_folds = []
        observed_fold_indexes: set[int] = set()
        primary_fold_eligibility: list[bool] = []
        for payload_fold in payload_folds:
            fold_index = int(payload_fold.get("fold_index", -1))
            observed_fold_indexes.add(fold_index)
            expected_fold = expected_by_fold.get(fold_index)
            if expected_fold is None:
                issues.append(f"unknown_fold:{fold_index}")
                continue
            if set(map(str, payload_fold.get("train_participants", []))) != set(expected_fold["train_participants"]):
                issues.append(f"fold_{fold_index}_train_membership")
            if str(payload_fold.get("validation_participant")) != str(expected_fold["validation_participant"]):
                issues.append(f"fold_{fold_index}_validation")
            if str(payload_fold.get("test_participant")) != str(expected_fold["test_participant"]):
                issues.append(f"fold_{fold_index}_test")
            expected_counts = {
                "n_train_participants": 7,
                "n_validation_participants": 1,
                "n_test_participants": 1,
                "n_train_observations": 63,
                "n_validation_observations": 9,
                "n_test_observations": 9,
            }
            for key, expected_value in expected_counts.items():
                if int(payload_fold.get(key, -1)) != expected_value:
                    issues.append(f"fold_{fold_index}_{key}")
            target_records = payload_fold.get("targets")
            if not isinstance(target_records, Mapping):
                issues.append(f"fold_{fold_index}_target_records")
                target_records = {}
            fold_primary_ok = True
            for target in TARGETS:
                target_record = target_records.get(target)
                if not isinstance(target_record, Mapping):
                    issues.append(f"fold_{fold_index}_{target}_record")
                    fold_primary_ok = False
                    continue
                learned = _as_bool(target_record.get("learned", True))
                fallback = _as_bool(target_record.get("fallback", False))
                try:
                    gamma = float(target_record.get("gamma", np.nan))
                except (TypeError, ValueError):
                    gamma = np.nan
                if not learned:
                    issues.append(f"fold_{fold_index}_{target}_not_learned")
                if fallback:
                    issues.append(f"fold_{fold_index}_{target}_fallback")
                if not np.isfinite(gamma) or gamma <= 0.0:
                    issues.append(f"fold_{fold_index}_{target}_gamma_nonpositive")
                weights = _weight_map(payload_fold, target_record, target)
                for modality, weight in weights.items():
                    weight_rows.append(
                        {
                            "candidate": candidate,
                            "seed": seed,
                            "fold_index": fold_index,
                            "target": target,
                            "modality": modality,
                            "weight": float(weight),
                        }
                    )
                if candidate in expert_candidates:
                    if set(weights) != set(modalities):
                        issues.append(f"fold_{fold_index}_{target}_expert_weight_modalities")
                    if not weights or abs(sum(weights.values()) - 1.0) > 1e-6 or any(
                        value < -1e-12 for value in weights.values()
                    ):
                        issues.append(f"fold_{fold_index}_{target}_simplex")
                        fold_primary_ok = False
                if candidate == primary:
                    positive_foundation = sum(
                        weights.get(modality, 0.0) >= min_positive_weight
                        for modality in FOUNDATION_MODALITIES
                    )
                    if not weights or positive_foundation < int(primary_rules["foundation_modalities_with_positive_weight_per_target"]):
                        issues.append(f"fold_{fold_index}_{target}_primary_weights")
                        fold_primary_ok = False
                dimension_rows.extend(
                    _selected_dimension_rows(
                        candidate,
                        seed,
                        payload_fold,
                        target,
                        target_record,
                        method.get("compression_dimension"),
                    )
                )
            primary_fold_eligibility.append(fold_primary_ok)
            runner_ablation_rows.extend(_ablation_rows_from_fold(candidate, seed, payload_fold))
        if observed_fold_indexes != set(expected_by_fold):
            issues.append("fold_index_coverage")

        if candidate == primary and len(set(modalities)) < int(primary_rules["minimum_modalities"]):
            issues.append("primary_minimum_modalities")
        if candidate == primary and required_columns <= set(frame.columns):
            for target in TARGETS:
                correction = frame[f"{target}_pred"].astype(float) - frame[f"condition_only_{target}"].astype(float)
                for fold_index, fold_frame in frame.assign(_correction=correction).groupby("fold_index"):
                    usable = ~fold_frame.apply(
                        lambda row: (str(row["participant_id"]), str(row["condition"])) in ZERO_FEATURE_EXCEPTIONS,
                        axis=1,
                    )
                    if not (np.abs(fold_frame.loc[usable, "_correction"].to_numpy()) > CORRECTION_ATOL).any():
                        issues.append(f"fold_{fold_index}_{target}_zero_correction")

        audit_rows.append(
            {
                "candidate": candidate,
                "seed": seed,
                "valid": not issues,
                "issues": ";".join(dict.fromkeys(issues)),
                "observations": len(frame),
                "folds": len(payload_folds),
                "common_valid_windows": contract.get("common_valid_window_count"),
                "embedding_cuda_used": cuda_used and provenance_cuda,
                "embedding_device": embedding_device,
                "head_device": runtime.get("device"),
                "primary_eligible": candidate != primary or (not issues and all(primary_fold_eligibility)),
                "result_path": str(result_path),
                "prediction_path": str(prediction_path),
                "result_sha256": file_sha256(result_path),
                "prediction_sha256": file_sha256(prediction_path),
                "runner_source_sha256": observed_source_hashes["runner"],
                "compression_source_sha256": observed_source_hashes["compression"],
                "dataset_source_sha256": observed_source_hashes["dataset"],
                "anchor_runner_source_sha256": observed_source_hashes["anchor_runner"],
            }
        )
        if not issues:
            frame["source_kind"] = "new_preregistered_candidate"
            frame["source_file"] = str(prediction_path)
            prediction_frames.append(frame)
            payloads.append(payload)

    audit = pd.DataFrame(audit_rows).sort_values(["candidate", "seed"]).reset_index(drop=True)
    if not audit["valid"].all():
        failures = audit.loc[~audit["valid"], ["candidate", "seed", "issues"]].to_dict("records")
        raise ValueError(f"Formal compression/fusion audit failed: {json.dumps(failures, ensure_ascii=False)}")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_predictions = len(prereg["_candidate_names"]) * len(prereg["protocol"]["seeds"]) * 81
    if len(predictions) != expected_predictions:
        raise ValueError(f"Expected {expected_predictions} audited predictions, got {len(predictions)}")
    return (
        predictions,
        payloads,
        audit,
        pd.DataFrame(dimension_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(runner_ablation_rows),
    )


def _canonical_comparator_predictions(
    canonical: pd.DataFrame,
    seeds: Sequence[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    definitions = {
        "fold_local_condition_only": {
            "relaxation": "condition_only_relaxation",
            "discomfort": "condition_only_discomfort",
            "source_kind": "canonical_baseline",
        },
        "nonfoundation_global_relaxation_condition_discomfort": {
            "relaxation": "nonfoundation_relaxation",
            "discomfort": "nonfoundation_discomfort",
            "source_kind": "nonfoundation_hybrid_control",
        },
    }
    for candidate, definition in definitions.items():
        for seed in seeds:
            frame = canonical.copy()
            frame["candidate"] = candidate
            frame["seed"] = int(seed)
            frame["relaxation_pred"] = frame[str(definition["relaxation"])]
            frame["discomfort_pred"] = frame[str(definition["discomfort"])]
            frame["source_kind"] = definition["source_kind"]
            frame["source_file"] = "recomputed_from_canonical_labels_and_split_manifest"
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _historical_prediction_file(root: Path, seed: int, pattern: str) -> Path:
    seed_dir = root / f"seed_{seed}"
    candidates = sorted(seed_dir.glob(pattern)) if seed_dir.is_dir() else []
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one historical file under {seed_dir} matching {pattern!r}, got {candidates}"
        )
    return candidates[0]


def _normalize_historical(
    path: Path,
    candidate: str,
    seed: int,
    canonical: pd.DataFrame,
    source_kind: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "participant_id",
        "condition",
        "fold_index",
        "relaxation_true",
        "discomfort_true",
        "relaxation_pred",
        "discomfort_pred",
        "condition_only_relaxation",
        "condition_only_discomfort",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Historical comparator {path} lacks {sorted(missing)}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    frame["fold_index"] = frame["fold_index"].astype(int)
    if len(frame) != 81 or frame.duplicated(["participant_id", "condition"]).any():
        raise ValueError(f"Historical comparator {path} lacks exact 81-key OOF coverage")
    if "seed" in frame and set(frame["seed"].astype(int)) != {seed}:
        raise ValueError(f"Historical comparator seed mismatch in {path}")
    indexed = frame.set_index(["participant_id", "condition"]).sort_index()
    expected = canonical.set_index(["participant_id", "condition"]).sort_index()
    if set(indexed.index) != set(expected.index):
        raise ValueError(f"Historical comparator OOF keys differ from canonical labels: {path}")
    for column in (
        "fold_index",
        "relaxation_true",
        "discomfort_true",
        "condition_only_relaxation",
        "condition_only_discomfort",
    ):
        if column == "fold_index":
            valid = np.array_equal(indexed[column].to_numpy(dtype=int), expected[column].to_numpy(dtype=int))
        else:
            valid = np.allclose(
                indexed[column].to_numpy(dtype=float),
                expected[column].to_numpy(dtype=float),
                atol=FLOAT_ATOL,
                rtol=0.0,
            )
        if not valid:
            raise ValueError(f"Historical comparator {path} differs in canonical {column}")
    predictions = indexed[[f"{target}_pred" for target in TARGETS]].to_numpy(dtype=float)
    if not np.isfinite(predictions).all() or np.any((predictions < -FLOAT_ATOL) | (predictions > 1 + FLOAT_ATOL)):
        raise ValueError(f"Historical comparator predictions are invalid: {path}")
    frame["candidate"] = candidate
    frame["seed"] = seed
    frame["source_kind"] = source_kind
    frame["source_file"] = str(path)
    return frame


def _load_historical_comparators(
    no_eeg_root: Path,
    healnet_root: Path,
    seeds: Sequence[int],
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for seed in seeds:
        specifications = (
            (
                no_eeg_root,
                f"no_eeg_raw_dual_s{seed}_predictions.csv",
                "historical_no_eeg_raw_dual",
                "historical_adaptive_condition_anchor",
            ),
            (
                healnet_root,
                "eeg9_healnet_no_eeg_s*_predictions.csv",
                "historical_healnet_no_eeg",
                "historical_validation_selected_neural_fusion",
            ),
        )
        for root, pattern, candidate, source_kind in specifications:
            path = _historical_prediction_file(root, int(seed), pattern)
            frame = _normalize_historical(path, candidate, int(seed), canonical, source_kind)
            if candidate == "historical_healnet_no_eeg":
                if "cuda_used" not in frame or not frame["cuda_used"].map(_as_bool).all():
                    raise ValueError(f"Historical HEALNet run does not prove CUDA use: {path}")
                if "device" not in frame or not frame["device"].astype(str).str.lower().str.contains("cuda").all():
                    raise ValueError(f"Historical HEALNet device is not CUDA: {path}")
            frames.append(frame)
            audit_rows.append(
                {
                    "candidate": candidate,
                    "seed": int(seed),
                    "observations": len(frame),
                    "valid": True,
                    "source_file": str(path),
                    "source_sha256": file_sha256(path),
                    "inferential_status": "descriptive_historical_comparator",
                }
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audit_rows)


def _participant_metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ["candidate", "source_kind", "seed", "participant_id"]
    for (candidate, source_kind, seed, participant), frame in predictions.groupby(group_columns, sort=True):
        target_values: list[tuple[float, float]] = []
        for target in TARGETS:
            model_mae = float(np.mean(np.abs(frame[f"{target}_true"] - frame[f"{target}_pred"])))
            baseline_mae = float(
                np.mean(np.abs(frame[f"{target}_true"] - frame[f"condition_only_{target}"]))
            )
            target_values.append((model_mae, baseline_mae))
            rows.append(
                {
                    "candidate": candidate,
                    "source_kind": source_kind,
                    "seed": int(seed),
                    "participant_id": str(participant),
                    "outcome": target,
                    "model_mae": model_mae,
                    "condition_only_mae": baseline_mae,
                    "delta_vs_condition": model_mae - baseline_mae,
                }
            )
        model_macro = float(np.mean([value[0] for value in target_values]))
        baseline_macro = float(np.mean([value[1] for value in target_values]))
        rows.append(
            {
                "candidate": candidate,
                "source_kind": source_kind,
                "seed": int(seed),
                "participant_id": str(participant),
                "outcome": "macro",
                "model_mae": model_macro,
                "condition_only_mae": baseline_macro,
                "delta_vs_condition": model_macro - baseline_macro,
            }
        )
    return pd.DataFrame(rows)


def _seed_metrics(participant_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, source_kind, seed), frame in participant_metrics.groupby(
        ["candidate", "source_kind", "seed"], sort=True
    ):
        row: dict[str, Any] = {
            "candidate": candidate,
            "source_kind": source_kind,
            "seed": int(seed),
        }
        for outcome in OUTCOMES:
            values = frame[frame["outcome"] == outcome]
            if len(values) != 9:
                raise ValueError(f"{candidate}/{seed}/{outcome} does not contain nine participant metrics")
            row[f"{outcome}_mae"] = float(values["model_mae"].mean())
            row[f"{outcome}_condition_mae"] = float(values["condition_only_mae"].mean())
            row[f"{outcome}_delta_vs_condition"] = float(values["delta_vs_condition"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate, source_kind), frame in seed_metrics.groupby(["candidate", "source_kind"], sort=True):
        if len(frame) != 3:
            raise ValueError(f"{candidate} does not contain all three seed metrics")
        row: dict[str, Any] = {
            "candidate": candidate,
            "source_kind": source_kind,
            "seed_count": len(frame),
        }
        for outcome in OUTCOMES:
            values = frame[f"{outcome}_mae"].to_numpy(dtype=float)
            deltas = frame[f"{outcome}_delta_vs_condition"].to_numpy(dtype=float)
            row[f"{outcome}_mae_mean"] = float(values.mean())
            row[f"{outcome}_mae_std"] = float(values.std(ddof=1))
            row[f"{outcome}_mae_min"] = float(values.min())
            row[f"{outcome}_mae_max"] = float(values.max())
            row[f"{outcome}_delta_mean"] = float(deltas.mean())
            row[f"{outcome}_delta_all_seeds_negative"] = bool(np.all(deltas < -1e-12))
            row[f"{outcome}_delta_three_seed_mean_nonpositive"] = bool(deltas.mean() <= 1e-12)
        row["both_target_means_not_worse"] = bool(
            row["relaxation_delta_three_seed_mean_nonpositive"]
            and row["discomfort_delta_three_seed_mean_nonpositive"]
        )
        row["both_target_means_improve"] = bool(
            row["relaxation_delta_mean"] < -1e-12 and row["discomfort_delta_mean"] < -1e-12
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["macro_mae_mean", "candidate"]).reset_index(drop=True)


def _exact_sign_flip(deltas: np.ndarray) -> tuple[float, int]:
    observed = abs(float(np.mean(deltas)))
    assignments = np.asarray(list(product((-1.0, 1.0), repeat=len(deltas))), dtype=float)
    permuted = np.abs(np.mean(assignments * deltas[None, :], axis=1))
    count = int(np.sum(permuted >= observed - 1e-15))
    return count / len(assignments), len(assignments)


def _holm(pvalues: pd.Series) -> pd.Series:
    raw = pvalues.to_numpy(dtype=float)
    order = np.argsort(raw, kind="stable")
    sorted_p = raw[order]
    adjusted_sorted = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * sorted_p)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=pvalues.index)


def _paired_statistics(
    participant_metrics: pd.DataFrame,
    candidates: Sequence[str],
    *,
    apply_holm: bool,
    family_label: str,
) -> pd.DataFrame:
    subset = participant_metrics[participant_metrics["candidate"].isin(candidates)]
    seed_averaged = (
        subset.groupby(["candidate", "participant_id", "outcome"], as_index=False)["delta_vs_condition"]
        .mean()
        .sort_values(["candidate", "outcome", "participant_id"])
    )
    rows: list[dict[str, Any]] = []
    for index, ((candidate, outcome), frame) in enumerate(
        seed_averaged.groupby(["candidate", "outcome"], sort=True)
    ):
        deltas = frame["delta_vs_condition"].to_numpy(dtype=float)
        if len(deltas) != 9:
            raise ValueError(f"Paired inference requires nine participants for {candidate}/{outcome}")
        rng = np.random.default_rng(BOOTSTRAP_SEED + index)
        sample_indexes = rng.integers(0, 9, size=(BOOTSTRAP_RESAMPLES, 9))
        bootstrap = deltas[sample_indexes].mean(axis=1)
        pvalue, assignments = _exact_sign_flip(deltas)
        rows.append(
            {
                "candidate": candidate,
                "outcome": outcome,
                "mean_paired_delta": float(deltas.mean()),
                "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "bootstrap_cluster": "participant_after_seed_average",
                "exact_sign_flip_p_two_sided": pvalue,
                "sign_flip_assignments": assignments,
                "participants_improved": int(np.sum(deltas < -CORRECTION_ATOL)),
                "participants_tied": int(np.sum(np.isclose(deltas, 0.0, atol=CORRECTION_ATOL))),
                "participants_worsened": int(np.sum(deltas > CORRECTION_ATOL)),
                "participant_count": 9,
                "multiplicity_family": family_label,
                "inferential_status": "preregistered" if apply_holm else "descriptive_only",
            }
        )
    result = pd.DataFrame(rows)
    if apply_holm:
        if len(result) != 18:
            raise ValueError(f"The preregistered Holm family must contain exactly 18 tests, got {len(result)}")
        result["holm_family_size"] = 18
        result["holm_p"] = _holm(result["exact_sign_flip_p_two_sided"])
        result["holm_significant_0_05"] = result["holm_p"] < 0.05
    else:
        result["holm_family_size"] = 0
        result["holm_p"] = np.nan
        result["holm_significant_0_05"] = False
    return result


def _dimension_summary(dimensions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate",
        "target",
        "scope",
        "parameter",
        "selections",
        "mean_value",
        "std_value",
        "min_value",
        "max_value",
    ]
    if dimensions.empty:
        return pd.DataFrame(columns=columns)
    return (
        dimensions.groupby(["candidate", "target", "scope", "parameter"], as_index=False)
        .agg(
            selections=("value", "size"),
            mean_value=("value", "mean"),
            std_value=("value", "std"),
            min_value=("value", "min"),
            max_value=("value", "max"),
        )
        .sort_values(["candidate", "target", "scope", "parameter"])
        .reset_index(drop=True)
    )


def _weight_tables(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_columns = [
        "candidate",
        "target",
        "modality",
        "selections",
        "mean_weight",
        "std_weight",
        "min_weight",
        "max_weight",
        "mean_within_target_rank",
        "fraction_weight_at_least_0_05",
        "interpretation",
    ]
    if weights.empty:
        return weights.copy(), pd.DataFrame(columns=summary_columns)
    detailed = weights.copy()
    detailed["within_target_rank"] = detailed.groupby(
        ["candidate", "seed", "fold_index", "target"]
    )["weight"].rank(method="average", ascending=False)
    summary = (
        detailed.groupby(["candidate", "target", "modality"], as_index=False)
        .agg(
            selections=("weight", "size"),
            mean_weight=("weight", "mean"),
            std_weight=("weight", "std"),
            min_weight=("weight", "min"),
            max_weight=("weight", "max"),
            mean_within_target_rank=("within_target_rank", "mean"),
            fraction_weight_at_least_0_05=("weight", lambda values: float(np.mean(values >= 0.05))),
        )
        .sort_values(["candidate", "target", "mean_weight"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary["interpretation"] = "global_target_specific_simplex_weight_not_causal_effect"
    return detailed, summary[summary_columns]


def _runner_ablation_summary(ablations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate",
        "outcome",
        "modality",
        "fold_seed_evaluations",
        "mean_delta_without_minus_full",
        "std_delta_without_minus_full",
        "median_delta_without_minus_full",
        "min_delta_without_minus_full",
        "max_delta_without_minus_full",
        "fraction_removal_worsened_validation_mae",
        "interpretation",
    ]
    if ablations.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        ablations.groupby(["candidate", "outcome", "modality"], as_index=False)
        .agg(
            fold_seed_evaluations=("delta_without_minus_full", "size"),
            mean_delta_without_minus_full=("delta_without_minus_full", "mean"),
            std_delta_without_minus_full=("delta_without_minus_full", "std"),
            median_delta_without_minus_full=("delta_without_minus_full", "median"),
            min_delta_without_minus_full=("delta_without_minus_full", "min"),
            max_delta_without_minus_full=("delta_without_minus_full", "max"),
            fraction_removal_worsened_validation_mae=(
                "delta_without_minus_full",
                lambda values: float(np.mean(values > CORRECTION_ATOL)),
            ),
        )
        .sort_values(
            ["candidate", "outcome", "mean_delta_without_minus_full"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )
    summary["interpretation"] = (
        "positive_delta_means_removal_worsened_outer_validation_mae_descriptive_only"
    )
    return summary[columns]


def _head_ablation_tables(
    participant_metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    primary: str,
    foundation_only: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, Any]] = []
    for seed in sorted(seed_metrics["seed"].unique()):
        primary_row = seed_metrics[
            (seed_metrics["candidate"] == primary) & (seed_metrics["seed"] == seed)
        ]
        foundation_row = seed_metrics[
            (seed_metrics["candidate"] == foundation_only) & (seed_metrics["seed"] == seed)
        ]
        if len(primary_row) != 1 or len(foundation_row) != 1:
            raise ValueError(f"Missing primary/foundation-only seed metric for seed {seed}")
        for outcome in OUTCOMES:
            hybrid_mae = float(primary_row.iloc[0][f"{outcome}_mae"])
            ablated_mae = float(foundation_row.iloc[0][f"{outcome}_mae"])
            delta = ablated_mae - hybrid_mae
            seed_rows.append(
                {
                    "seed": int(seed),
                    "outcome": outcome,
                    "hybrid_primary": primary,
                    "foundation_only_ablation": foundation_only,
                    "hybrid_mae": hybrid_mae,
                    "foundation_only_mae": ablated_mae,
                    "delta_foundation_only_minus_hybrid": delta,
                    "interpretation": _interpret_head_ablation_delta(delta),
                }
            )
    by_seed = pd.DataFrame(seed_rows)
    summary = (
        by_seed.groupby("outcome", as_index=False)
        .agg(
            seed_count=("seed", "size"),
            hybrid_mae_mean=("hybrid_mae", "mean"),
            foundation_only_mae_mean=("foundation_only_mae", "mean"),
            mean_delta_foundation_only_minus_hybrid=(
                "delta_foundation_only_minus_hybrid",
                "mean",
            ),
            std_delta_foundation_only_minus_hybrid=(
                "delta_foundation_only_minus_hybrid",
                "std",
            ),
            min_delta_foundation_only_minus_hybrid=(
                "delta_foundation_only_minus_hybrid",
                "min",
            ),
            max_delta_foundation_only_minus_hybrid=(
                "delta_foundation_only_minus_hybrid",
                "max",
            ),
        )
        .sort_values("outcome")
        .reset_index(drop=True)
    )
    summary["interpretation"] = summary[
        "mean_delta_foundation_only_minus_hybrid"
    ].map(_interpret_head_ablation_delta)

    subset = participant_metrics[
        participant_metrics["candidate"].isin([primary, foundation_only])
    ][["candidate", "seed", "participant_id", "outcome", "model_mae"]]
    pivot = subset.pivot(
        index=["seed", "participant_id", "outcome"],
        columns="candidate",
        values="model_mae",
    ).reset_index()
    if primary not in pivot or foundation_only not in pivot:
        raise ValueError("Cannot construct the participant-level head ablation")
    pivot["delta_foundation_only_minus_hybrid"] = pivot[foundation_only] - pivot[primary]
    by_participant = (
        pivot.groupby(["participant_id", "outcome"], as_index=False)
        .agg(
            seed_averaged_hybrid_mae=(primary, "mean"),
            seed_averaged_foundation_only_mae=(foundation_only, "mean"),
            seed_averaged_delta_foundation_only_minus_hybrid=(
                "delta_foundation_only_minus_hybrid",
                "mean",
            ),
        )
        .sort_values(["outcome", "participant_id"])
        .reset_index(drop=True)
    )
    return by_seed, summary, by_participant


def _method_table(
    prereg: Mapping[str, Any],
    candidate_summary: pd.DataFrame,
    run_audit: pd.DataFrame,
) -> pd.DataFrame:
    primary = str(prereg["primary_candidate"])
    summary_by_name = candidate_summary.set_index("candidate")
    rows: list[dict[str, Any]] = []
    for candidate in prereg["_candidate_names"]:
        method = prereg["_method_registry"][candidate]
        metric = summary_by_name.loc[candidate]
        run_rows = run_audit[run_audit["candidate"] == candidate]
        audit_passed = len(run_rows) == 3 and bool(run_rows["valid"].all())
        primary_eligible: bool | None = (
            bool(run_rows["primary_eligible"].all()) if candidate == primary else None
        )
        point_success = bool(
            metric["macro_delta_all_seeds_negative"]
            and metric["both_target_means_not_worse"]
        )
        modalities = method.get("modalities")
        if not isinstance(modalities, Sequence) or isinstance(modalities, str):
            modalities = ["ecg", "eye", "head", "video"]
        declared_dimension = method.get("compression_dimension")
        if candidate == str(prereg["foundation_only_sensitivity"]["name"]):
            # The sensitivity uses one PCA2 expert per listed modality; it has no
            # concatenated presence coordinate.  The base method's eight-score
            # dimension therefore must not leak into this three-modality row.
            declared_dimension = 2 * len(modalities)
        rows.append(
            {
                "candidate": candidate,
                "role": (
                    "preregistered_primary"
                    if candidate == primary
                    else "foundation_only_sensitivity"
                    if candidate == str(prereg["foundation_only_sensitivity"]["name"])
                    else "preregistered_secondary"
                ),
                "family": str(method.get("family", "")),
                "modalities": "+".join(map(str, modalities)),
                "declared_compression_dimension": declared_dimension,
                "definition": str(method.get("definition", "")),
                "formal_run_audit_passed": audit_passed,
                "primary_eligibility_passed": primary_eligible,
                "candidate_point_success": point_success,
                "preregistered_primary_success": bool(
                    candidate == primary
                    and audit_passed
                    and bool(primary_eligible)
                    and point_success
                ),
            }
        )
    return pd.DataFrame(rows)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def _format_markdown_value(value: Any) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(float(value))):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if columns is not None:
        frame = frame.loc[:, list(columns)]
    if frame.empty:
        return "_No rows._"
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def _prepare_output_dir(output_dir: Path, force: bool) -> None:
    resolved = output_dir.resolve()
    if output_dir.is_symlink():
        raise ValueError(f"Refusing to replace a symlinked output directory: {output_dir}")
    protected = {Path("/").resolve(), Path.home().resolve(), ROOT.resolve(), *ROOT.resolve().parents}
    if resolved in protected:
        raise ValueError(f"Refusing dangerous output directory: {resolved}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty; pass --force to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _figure_bundle_status(output_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Audit the separately generated publication figure bundle, if present."""
    figures_dir = output_dir.parent / "figures"
    manifest_path = figures_dir / "figure_manifest.json"
    required_stems = (
        "figure1_formal_outcomes",
        "figure2_feature_and_modality_audit",
        "figure3_cross_modal_redundancy",
    )
    required_suffixes = (".svg", ".pdf", ".tiff", ".png")
    required_paths = [
        figures_dir / f"{stem}{suffix}"
        for stem in required_stems
        for suffix in required_suffixes
    ]
    status: dict[str, Any] = {
        "figures_dir": str(figures_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "required_files": [str(path.resolve()) for path in required_paths],
        "manifest_exists": manifest_path.is_file(),
        "required_files_exist": all(path.is_file() for path in required_paths),
    }
    if not manifest_path.is_file() or not status["required_files_exist"]:
        status["valid"] = False
        return False, status
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status["manifest_status"] = manifest.get("status")
    status["backend"] = manifest.get("backend")
    status["valid"] = bool(
        manifest.get("status") == "passed"
        and manifest.get("backend") == "python_matplotlib"
    )
    return bool(status["valid"]), status


def _report_text(
    prereg: Mapping[str, Any],
    run_audit: pd.DataFrame,
    historical_audit: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    method_table: pd.DataFrame,
    paired: pd.DataFrame,
    historical_paired: pd.DataFrame,
    dimension_summary: pd.DataFrame,
    weight_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    head_ablation_summary: pd.DataFrame,
    figures_generated: bool,
    figures_dir: Path,
) -> str:
    candidate_names = list(map(str, prereg["_candidate_names"]))
    primary = str(prereg["primary_candidate"])
    foundation_only = str(prereg["foundation_only_sensitivity"]["name"])
    summary_by_name = candidate_summary.set_index("candidate")
    new_summary = candidate_summary[candidate_summary["candidate"].isin(candidate_names)].copy()
    new_summary = new_summary.sort_values(["macro_mae_mean", "candidate"]).reset_index(drop=True)
    best = str(new_summary.iloc[0]["candidate"])
    best_row = summary_by_name.loc[best]
    baseline_row = summary_by_name.loc["fold_local_condition_only"]
    primary_row = summary_by_name.loc[primary]
    foundation_row = summary_by_name.loc[foundation_only]

    primary_method = method_table.set_index("candidate").loc[primary]
    primary_eligible = bool(primary_method["primary_eligibility_passed"])
    primary_success = bool(primary_method["preregistered_primary_success"])
    best_beats_condition = float(best_row["macro_delta_mean"]) < -CORRECTION_ATOL
    best_point_success = bool(
        best_row["macro_delta_all_seeds_negative"]
        and best_row["both_target_means_not_worse"]
    )

    joint_row = summary_by_name.loc["joint_block_balanced_pca8"]
    modality_row = summary_by_name.loc["modality_pca2_additive"]
    modality_minus_joint = float(modality_row["macro_mae_mean"] - joint_row["macro_mae_mean"])
    compression_comparison = (
        "modality-wise PCA2 was lower"
        if modality_minus_joint < -CORRECTION_ATOL
        else "joint block-balanced PCA8 was lower"
        if modality_minus_joint > CORRECTION_ATOL
        else "the two PCA controls tied"
    )

    foundation_macro_test = paired[
        (paired["candidate"] == foundation_only) & (paired["outcome"] == "macro")
    ]
    foundation_point_success = bool(
        foundation_row["macro_delta_all_seeds_negative"]
        and foundation_row["both_target_means_not_worse"]
    )
    foundation_corrected = bool(
        len(foundation_macro_test) == 1
        and foundation_macro_test.iloc[0]["holm_significant_0_05"]
        and foundation_macro_test.iloc[0]["mean_paired_delta"] < 0.0
    )
    if foundation_point_success and foundation_corrected:
        foundation_verdict = (
            "The foundation-only sensitivity beat Condition-only on the preregistered point criteria "
            "and on the Holm-corrected macro test within this internal nine-participant protocol."
        )
    elif foundation_point_success:
        foundation_verdict = (
            "The foundation-only sensitivity met the preregistered point-estimate criteria, but its "
            "macro comparison was not Holm-significant; this is exploratory point evidence, not a "
            "corrected confirmatory result."
        )
    else:
        foundation_verdict = (
            "The foundation-only sensitivity did not meet the preregistered point-estimate criteria, "
            "so these runs do not support a claim that foundation features beat Condition-only."
        )

    complement_sentences: list[str] = []
    for outcome in TARGETS:
        values = ablation_summary[
            (ablation_summary["candidate"] == primary)
            & (ablation_summary["outcome"] == outcome)
            & (ablation_summary["mean_delta_without_minus_full"] > CORRECTION_ATOL)
        ].sort_values("mean_delta_without_minus_full", ascending=False)
        if values.empty:
            complement_sentences.append(
                f"For {outcome}, no modality had a positive mean validation removal delta."
            )
        else:
            details = ", ".join(
                f"{row.modality} (mean delta {row.mean_delta_without_minus_full:+.6f}; "
                f"removal worsened {row.fraction_removal_worsened_validation_mae:.0%} of fold-seed checks)"
                for row in values.itertuples(index=False)
            )
            complement_sentences.append(
                f"For {outcome}, removal worsened validation MAE on average for {details}."
            )

    head_macro = head_ablation_summary[head_ablation_summary["outcome"] == "macro"]
    head_delta = float(head_macro.iloc[0]["mean_delta_foundation_only_minus_hybrid"])
    if head_delta > CORRECTION_ATOL:
        head_sentence = (
            f"Removing the handcrafted head block increased macro MAE by {head_delta:.6f} on average, "
            "which supports incremental head-feature value in this hybrid."
        )
    elif head_delta < -CORRECTION_ATOL:
        head_sentence = (
            f"Removing the handcrafted head block decreased macro MAE by {-head_delta:.6f} on average; "
            "the head-inclusive primary therefore does not show incremental head-feature value."
        )
    else:
        head_sentence = "The head-inclusive and foundation-only expert variants tied in mean macro MAE."

    best_macro_test = paired[(paired["candidate"] == best) & (paired["outcome"] == "macro")]
    best_corrected = bool(
        len(best_macro_test) == 1
        and best_macro_test.iloc[0]["holm_significant_0_05"]
        and best_macro_test.iloc[0]["mean_paired_delta"] < 0.0
    )
    if best_point_success:
        next_method = (
            f"Carry `{best}` forward as the single compression/fusion candidate for independent-cohort "
            "confirmation"
            + ("; it also passed the corrected internal macro test." if best_corrected else "; its current evidence remains exploratory.")
        )
    else:
        next_method = (
            "Retain fold-local Condition-only as the benchmark because no new candidate satisfied the "
            "predeclared both-target point criterion. If one new method is taken forward only as an exploratory "
            f"diagnostic, use `{best}` because it had the lowest macro MAE; do not promote it as a winner."
        )

    run_fact = (
        run_audit.groupby("candidate", as_index=False)
        .agg(
            seeds=("seed", "size"),
            all_runs_valid=("valid", "all"),
            observations_per_run=("observations", "min"),
            folds_per_run=("folds", "min"),
            cuda_cache_provenance=("embedding_cuda_used", "all"),
            head_device=("head_device", lambda values: ",".join(sorted(set(map(str, values))))),
        )
        .sort_values("candidate")
    )
    new_fact = method_table.merge(
        new_summary[
            [
                "candidate",
                "relaxation_mae_mean",
                "discomfort_mae_mean",
                "macro_mae_mean",
                "relaxation_delta_mean",
                "discomfort_delta_mean",
                "macro_delta_mean",
                "macro_delta_all_seeds_negative",
                "both_target_means_not_worse",
            ]
        ],
        on="candidate",
        how="left",
    ).sort_values(["macro_mae_mean", "candidate"])
    comparator_fact = candidate_summary[~candidate_summary["candidate"].isin(candidate_names)][
        [
            "candidate",
            "source_kind",
            "relaxation_mae_mean",
            "discomfort_mae_mean",
            "macro_mae_mean",
            "macro_delta_mean",
        ]
    ].sort_values(["macro_mae_mean", "candidate"])
    paired_fact = paired[
        [
            "candidate",
            "outcome",
            "mean_paired_delta",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "exact_sign_flip_p_two_sided",
            "holm_p",
            "holm_significant_0_05",
        ]
    ].sort_values(["outcome", "mean_paired_delta", "candidate"])
    historical_fact = historical_paired[
        [
            "candidate",
            "outcome",
            "mean_paired_delta",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "exact_sign_flip_p_two_sided",
            "inferential_status",
        ]
    ].sort_values(["outcome", "mean_paired_delta", "candidate"])
    compact_dimensions = dimension_summary[
        (dimension_summary["scope"] == "fold")
        & (dimension_summary["parameter"] == "compression_dimension")
    ][["candidate", "target", "selections", "mean_value", "min_value", "max_value"]]

    lines = [
        "# Frozen-Embedding Compression/Fusion Evaluation",
        "",
        "This report separates audited facts, bounded inferences, and recommendations. Historical best-model "
        "comparators are descriptive and are not members of the preregistered multiplicity family.",
        "",
        "## Facts",
        "",
        "### Contract and execution audit",
        "",
        "- Protocol: nine fixed outer folds, each with 7 training participants, 1 validation participant, "
        "and 1 test participant; 81 participant-condition OOF keys and 545 common-valid windows.",
        "- Seeds: `20260705`, `20260706`, and `20260707` for every candidate.",
        "- All six new candidates use the exact preregistered cache, labels, split, and mask hashes. Each result "
        "proves that frozen embeddings were extracted on CUDA; compression and Ridge heads ran on CPU and no "
        "neural network was trained in these probes.",
        "- Every new fold learned both targets with a strictly positive gamma and no Condition fallback. The "
        "primary additionally passed its modality-weight and nonzero-correction eligibility checks.",
        "",
        _markdown_table(run_fact),
        "",
        "### New candidate results",
        "",
        "MAE and deltas are participant-macro quantities averaged over the three seeds. Negative delta is better "
        "than the recomputed fold-local Condition baseline.",
        "",
        _markdown_table(
            new_fact,
            [
                "candidate",
                "role",
                "relaxation_mae_mean",
                "discomfort_mae_mean",
                "macro_mae_mean",
                "relaxation_delta_mean",
                "discomfort_delta_mean",
                "macro_delta_mean",
                "macro_delta_all_seeds_negative",
                "both_target_means_not_worse",
                "primary_eligibility_passed",
                "preregistered_primary_success",
            ],
        ),
        "",
        "### Preregistered paired inference",
        "",
        "Seeds were first averaged within participant. Confidence intervals use 10,000 participant-cluster "
        "bootstrap resamples; p-values use all 512 participant-level sign assignments; Holm correction covers "
        "exactly 18 candidate-outcome tests.",
        "",
        _markdown_table(paired_fact),
        "",
        "### Canonical controls and historical comparators",
        "",
        _markdown_table(comparator_fact),
        "",
        "Historical and nonfoundation paired calculations are descriptive only:",
        "",
        _markdown_table(historical_fact),
        "",
        "### Selected dimensions",
        "",
        _markdown_table(compact_dimensions),
        "",
        "The complete fold-, target-, and compressor-provenance dimension records are in "
        "`selected_dimensions.csv` and `selected_dimension_summary.csv`.",
        "",
        "### Expert weights and modality-removal diagnostics",
        "",
        "Simplex weights are global, target-specific reliability weights. They are associative and the formal "
        "expert model constrains foundation weights, so a nonzero weight alone is not causal contribution evidence.",
        "",
        _markdown_table(
            weight_summary,
            [
                "candidate",
                "target",
                "modality",
                "selections",
                "mean_weight",
                "min_weight",
                "max_weight",
                "mean_within_target_rank",
            ],
        ),
        "",
        "Positive removal delta means that removing the modality worsened outer-validation MAE. These diagnostics "
        "are descriptive; they are not extra test-set hypothesis tests.",
        "",
        _markdown_table(
            ablation_summary,
            [
                "candidate",
                "outcome",
                "modality",
                "fold_seed_evaluations",
                "mean_delta_without_minus_full",
                "fraction_removal_worsened_validation_mae",
            ],
        ),
        "",
        "The predeclared head-block ablation compares the hybrid primary with its foundation-only counterpart on "
        "the common test OOF keys:",
        "",
        _markdown_table(head_ablation_summary),
        "",
        "## Inferences",
        "",
        "### Most suitable compression/fusion candidate",
        "",
        f"`{best}` had the lowest three-seed mean macro MAE among the six new candidates "
        f"({float(best_row['macro_mae_mean']):.6f}; delta vs Condition {float(best_row['macro_delta_mean']):+.6f}). "
        + (
            "It improved on Condition-only by the macro point estimate."
            if best_beats_condition
            else f"It did not improve on Condition-only (macro MAE {float(baseline_row['macro_mae_mean']):.6f})."
        )
        + (
            " It also met the predeclared both-target point criterion."
            if best_point_success
            else " It did not meet the predeclared both-target point criterion."
        ),
        "",
        "### Modality-wise versus joint compression",
        "",
        f"The direct equal-budget PCA control shows that {compression_comparison}: modality-wise minus joint macro "
        f"MAE was {modality_minus_joint:+.6f}. This comparison applies only to the preregistered PCA2/PCA8 "
        "implementations, not to all possible early- and late-fusion models.",
        "",
        "### Are modalities complementary?",
        "",
        " ".join(complement_sentences),
        "",
        "These validation removal effects can support a bounded complementarity interpretation when several "
        "modalities have positive deltas, but they do not identify causal physiological contributions. "
        + head_sentence,
        "",
        "### Do foundation features beat Condition-only?",
        "",
        foundation_verdict,
        "",
        f"The head-inclusive primary passed formal eligibility: `{primary_eligible}`; it met the preregistered "
        f"point-success criterion: `{primary_success}`. Its macro delta was "
        f"{float(primary_row['macro_delta_mean']):+.6f}; this hybrid cannot by itself support a foundation-only claim.",
        "",
        "## Recommendations",
        "",
        "1. " + next_method,
        "2. Freeze this evaluator and the selected candidate before collecting an independent participant cohort; "
        "reuse the same label scale, window mask, 7/1/1 participant split logic, and participant-macro metrics.",
        "3. Treat expert weights and validation removal deltas as mechanism diagnostics. Use the foundation-only OOF "
        "sensitivity—not the head-inclusive primary—to make any claim about foundation embeddings.",
        "4. Do not choose a historical comparator as the new winner from this table: those models were selected "
        "adaptively and their calculations are explicitly descriptive.",
        "",
        "## Scope and limitations",
        "",
        "- This is exploratory internal validation on the same nine participants used for method development; it "
        "does not establish out-of-cohort generalization.",
        "- The effective inferential cluster count is nine participants. Seed replication measures optimization or "
        "selection variability, not additional independent subjects.",
        "- P004/C6 has no common-valid foundation window. Its zero feature correction is the preregistered exception "
        "and is not evidence of a learned modality contribution.",
        "- The historical no-EEG and HEALNet rows retain their original adaptive-selection status and cannot enter "
        "the 18-test Holm family.",
        (
            f"- A publication figure bundle with editable SVG/PDF, 600-dpi TIFF, PNG previews, and source-data "
            f"CSVs passed QA at `{figures_dir}`."
            if figures_generated
            else "- No complete publication figure bundle was present when this evaluator ran; all audit, metric, "
            "dimension, weight, and ablation data remain available as CSV tables."
        ),
        "",
    ]
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    prereg = _load_preregistration(args.preregistration)
    expected_inputs = prereg["input_contract"]
    direct_hashes = {
        "labels": file_sha256(args.labels),
        "split_manifest": file_sha256(args.split_manifest),
    }
    if direct_hashes["labels"] != str(expected_inputs["labels_sha256"]):
        raise ValueError(
            f"Evaluator labels hash differs from preregistration: {direct_hashes['labels']}"
        )
    if direct_hashes["split_manifest"] != str(expected_inputs["split_manifest_sha256"]):
        raise ValueError(
            "Evaluator split-manifest hash differs from preregistration: "
            f"{direct_hashes['split_manifest']}"
        )

    _labels, folds, canonical = _load_contract(
        args.labels,
        args.split_manifest,
        prereg,
    )
    seeds = [int(value) for value in prereg["protocol"]["seeds"]]
    candidate_names = list(map(str, prereg["_candidate_names"]))
    artifacts = _discover_formal_artifacts(args.runs_dir, candidate_names, seeds)
    (
        new_predictions,
        _payloads,
        run_audit,
        dimensions,
        weights,
        runner_ablations,
    ) = _audit_new_runs(artifacts, prereg, folds, canonical)

    historical_predictions, historical_audit = _load_historical_comparators(
        args.historical_no_eeg_root,
        args.historical_healnet_root,
        seeds,
        canonical,
    )
    canonical_predictions = _canonical_comparator_predictions(canonical, seeds)
    all_predictions = pd.concat(
        [new_predictions, canonical_predictions, historical_predictions],
        ignore_index=True,
        sort=False,
    )
    expected_candidates = {
        *candidate_names,
        "fold_local_condition_only",
        "nonfoundation_global_relaxation_condition_discomfort",
        "historical_no_eeg_raw_dual",
        "historical_healnet_no_eeg",
    }
    if set(all_predictions["candidate"].astype(str)) != expected_candidates:
        raise ValueError("Combined prediction candidates differ from the frozen evaluation roster")
    coverage = (
        all_predictions.groupby(["candidate", "seed"], as_index=False)
        .agg(
            observations=("condition", "size"),
            participants=("participant_id", "nunique"),
            unique_conditions=("condition", "nunique"),
        )
        .sort_values(["candidate", "seed"])
        .reset_index(drop=True)
    )
    key_counts = (
        all_predictions.groupby(["candidate", "seed"])
        .apply(
            lambda frame: len(frame[["participant_id", "condition"]].drop_duplicates()),
            include_groups=False,
        )
        .rename("unique_oof_keys")
        .reset_index()
    )
    coverage = coverage.merge(key_counts, on=["candidate", "seed"], how="left", validate="one_to_one")
    if (
        len(coverage) != len(expected_candidates) * 3
        or not (coverage["observations"] == 81).all()
        or not (coverage["participants"] == 9).all()
        or not (coverage["unique_conditions"] == 9).all()
        or not (coverage["unique_oof_keys"] == 81).all()
    ):
        raise ValueError("Combined prediction coverage is not exactly 81 OOF keys per candidate and seed")

    participant_metrics = _participant_metric_rows(all_predictions)
    seed_metrics = _seed_metrics(participant_metrics)
    candidate_summary = _candidate_summary(seed_metrics)
    paired = _paired_statistics(
        participant_metrics,
        candidate_names,
        apply_holm=True,
        family_label="six_new_candidates_x_three_outcomes",
    )
    historical_candidates = [
        "nonfoundation_global_relaxation_condition_discomfort",
        "historical_no_eeg_raw_dual",
        "historical_healnet_no_eeg",
    ]
    historical_paired = _paired_statistics(
        participant_metrics,
        historical_candidates,
        apply_holm=False,
        family_label="descriptive_controls_and_historical_comparators",
    )
    dimension_summary = _dimension_summary(dimensions)
    weights_detailed, weight_summary = _weight_tables(weights)
    runner_ablation_summary = _runner_ablation_summary(runner_ablations)
    primary = str(prereg["primary_candidate"])
    foundation_only = str(prereg["foundation_only_sensitivity"]["name"])
    head_by_seed, head_summary, head_by_participant = _head_ablation_tables(
        participant_metrics,
        seed_metrics,
        primary,
        foundation_only,
    )
    methods = _method_table(prereg, candidate_summary, run_audit)

    primary_record = methods[methods["candidate"] == primary]
    if len(primary_record) != 1:
        raise ValueError("Primary candidate is missing from the evaluated method table")
    primary_success = bool(primary_record.iloc[0]["preregistered_primary_success"])
    corrected_significant = paired[paired["holm_significant_0_05"]].copy()
    figures_generated, figure_bundle = _figure_bundle_status(args.output_dir)

    report = _report_text(
        prereg,
        run_audit,
        historical_audit,
        candidate_summary,
        methods,
        paired,
        historical_paired,
        dimension_summary,
        weight_summary,
        runner_ablation_summary,
        head_summary,
        figures_generated,
        Path(figure_bundle["figures_dir"]),
    )

    audit_payload: dict[str, Any] = {
        "schema_version": "relax_foundation_compression_fusion_evaluation_v1",
        "status": "passed",
        "facts": {
            "formal_candidates": len(candidate_names),
            "formal_seeds": seeds,
            "formal_runs": len(run_audit),
            "formal_runs_valid": int(run_audit["valid"].sum()),
            "formal_oof_predictions": len(new_predictions),
            "formal_oof_predictions_expected": 6 * 3 * 81,
            "combined_candidates": len(expected_candidates),
            "combined_prediction_rows": len(all_predictions),
            "participants": 9,
            "folds": 9,
            "observations_per_run": 81,
            "common_valid_windows": 545,
            "cuda_cache_provenance_all_formal_runs": bool(
                run_audit["embedding_cuda_used"].all()
            ),
            "formal_head_devices": sorted(set(map(str, run_audit["head_device"]))),
            "neural_training_performed_in_formal_candidates": False,
            "figures_generated": figures_generated,
        },
        "input_contract": {
            "preregistration": {
                "path": str(args.preregistration.resolve()),
                "sha256": prereg["_sha256"],
            },
            "labels": {"path": str(args.labels.resolve()), "sha256": direct_hashes["labels"]},
            "split_manifest": {
                "path": str(args.split_manifest.resolve()),
                "sha256": direct_hashes["split_manifest"],
            },
            "embedding_cache_sha256": expected_inputs["embedding_cache_sha256"],
            "common_mask_sha256": expected_inputs["common_mask_sha256"],
        },
        "source_contract": {
            "evaluator_sha256": file_sha256(Path(__file__)),
            "runner_sha256": str(run_audit.iloc[0]["runner_source_sha256"]),
            "compression_sha256": str(run_audit.iloc[0]["compression_source_sha256"]),
            "dataset_sha256": str(run_audit.iloc[0]["dataset_source_sha256"]),
            "anchor_runner_sha256": str(run_audit.iloc[0]["anchor_runner_source_sha256"]),
            "identical_across_all_18_formal_runs": True,
        },
        "inference_contract": {
            "seed_aggregation": "average_seeds_within_participant_before_inference",
            "bootstrap_cluster": "participant",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "sign_flip_assignments": 512,
            "holm_family_size": 18,
            "historical_comparators": "descriptive_only",
        },
        "primary": {
            "candidate": primary,
            "eligibility_passed": bool(run_audit.loc[run_audit["candidate"] == primary, "primary_eligible"].all()),
            "point_success": primary_success,
            "corrected_significant_outcomes": corrected_significant.loc[
                corrected_significant["candidate"] == primary, "outcome"
            ].tolist(),
        },
        "figure_bundle": figure_bundle,
        "formal_run_audit": _records(run_audit),
        "historical_comparator_audit": _records(historical_audit),
        "corrected_significant_tests": _records(corrected_significant),
        "scope": str(prereg["evaluation"]["scope"]),
        "limitations": [
            "same nine participants were used for method development and internal evaluation",
            "nine participant clusters; seeds are not independent subjects",
            "P004/C6 is the preregistered all-missing feature exception",
            "historical comparators were adaptively selected and remain descriptive",
        ],
    }

    tables: dict[str, pd.DataFrame] = {
        "run_audit.csv": run_audit,
        "historical_comparator_audit.csv": historical_audit,
        "prediction_coverage.csv": coverage,
        "all_oof_predictions.csv": all_predictions,
        "participant_metrics.csv": participant_metrics,
        "seed_metrics.csv": seed_metrics,
        "candidate_summary.csv": candidate_summary,
        "preregistered_method_summary.csv": methods,
        "paired_statistics_18_test_holm.csv": paired,
        "historical_descriptive_paired_statistics.csv": historical_paired,
        "selected_dimensions.csv": dimensions,
        "selected_dimension_summary.csv": dimension_summary,
        "expert_weights.csv": weights_detailed,
        "expert_weight_summary.csv": weight_summary,
        "runner_validation_modality_ablations.csv": runner_ablations,
        "runner_validation_modality_ablation_summary.csv": runner_ablation_summary,
        "head_ablation_by_seed.csv": head_by_seed,
        "head_ablation_summary.csv": head_summary,
        "head_ablation_by_participant.csv": head_by_participant,
    }

    _prepare_output_dir(args.output_dir, args.force)
    for filename, frame in tables.items():
        frame.to_csv(args.output_dir / filename, index=False)
    report_path = args.output_dir / "compression_fusion_evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    output_hashes = {
        filename: {
            "path": str((args.output_dir / filename).resolve()),
            "sha256": file_sha256(args.output_dir / filename),
            "rows": len(frame),
        }
        for filename, frame in tables.items()
    }
    output_hashes[report_path.name] = {
        "path": str(report_path.resolve()),
        "sha256": file_sha256(report_path),
        "rows": None,
    }
    audit_payload["outputs"] = output_hashes
    audit_path = args.output_dir / "evaluation_audit.json"
    _write_json(audit_path, audit_payload)
    return {
        "status": "passed",
        "output_dir": str(args.output_dir.resolve()),
        "report": str(report_path.resolve()),
        "audit": str(audit_path.resolve()),
        "formal_runs": len(run_audit),
        "formal_predictions": len(new_predictions),
        "primary": primary,
        "primary_success": primary_success,
        "holm_significant_tests": len(corrected_significant),
        "figures_generated": figures_generated,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--historical-no-eeg-root", type=Path, default=DEFAULT_NO_EEG_ROOT)
    parser.add_argument("--historical-healnet-root", type=Path, default=DEFAULT_HEALNET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = evaluate(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
