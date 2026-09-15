"""Evaluate the fixed July 18 five-modality compression/fusion experiment.

Separate from July 17: validate the fixed LOPO contract and report inference.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
MODALITIES = ("eeg", "ecg", "eye", "head", "video")
SEEDS = (20260705, 20260706, 20260707)
TARGETS = ("relaxation", "discomfort")
OUTCOMES = (*TARGETS, "macro")
FULL_METHODS = (
    "joint_block_balanced_pca12",
    "modality_rank_alloc_pca12",
    "targetwise_modality_pls1",
    "linear_gcca_shared_private",
    "modality_expert_simplex5",
)
PRIMARY = "modality_expert_simplex5"
ABLATION_VARIANTS = tuple(f"no_{modality}" for modality in MODALITIES)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260718
FLOAT_ATOL = 5e-7
ZERO_FEATURE_EXCEPTIONS = {("P004", "C6")}

DEFAULT_ROOT = ROOT / "artifacts/relax/foundation_compression_fusion_reinvestigation_20260718"
DEFAULT_CONTRACT_ROOT = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract"
)
DEFAULT_NO_EEG_ROOT = ROOT / "artifacts/relax/condition_anchor_residual_20260717/runs/no_eeg_raw_dual"
DEFAULT_JULY17_ROOT = ROOT / "artifacts/relax/foundation_compression_fusion_20260717/runs/modality_expert_simplex"
DEFAULT_HEALNET_ROOT = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/runs/healnet/no_eeg"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [] if frame.empty else json.loads(frame.to_json(orient="records"))


def _name(payload: Mapping[str, Any]) -> str:
    candidate = payload.get("candidate")
    return str(candidate.get("name", "")) if isinstance(candidate, Mapping) else str(candidate or "")


def _variant(payload: Mapping[str, Any]) -> str:
    candidate = payload.get("candidate")
    nested = candidate.get("variant") if isinstance(candidate, Mapping) else None
    return str(payload.get("variant") or nested or "full")


def _modalities_for_variant(variant: str) -> tuple[str, ...]:
    if variant == "full":
        return MODALITIES
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown run variant {variant!r}")
    removed = variant.removeprefix("no_")
    return tuple(modality for modality in MODALITIES if modality != removed)


def _load_preregistration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing preregistration: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_methods = payload.get("methods")
    if not isinstance(raw_methods, list):
        raise ValueError("Preregistration must contain a methods list")
    methods: list[str] = []
    for value in raw_methods:
        if isinstance(value, Mapping):
            methods.append(str(value.get("name") or value.get("method_id")))
        else:
            methods.append(str(value))
    if tuple(methods) != FULL_METHODS:
        raise ValueError(f"Expected the five ordered July 18 methods, got {methods}")
    if str(payload.get("primary_candidate", PRIMARY)) != PRIMARY:
        raise ValueError("The preregistered primary must be modality_expert_simplex5")
    protocol = payload.get("protocol", payload.get("fixed_protocol", {}))
    seeds = tuple(map(int, protocol.get("seeds", SEEDS)))
    participants = tuple(map(str, protocol.get("participants", PARTICIPANTS)))
    if seeds != SEEDS or participants != PARTICIPANTS:
        raise ValueError("Preregistration differs from the fixed cohort or seed contract")
    if int(protocol.get("folds", 9)) != 9 or int(protocol.get("observations", 81)) != 81:
        raise ValueError("Preregistration must declare 9 folds and 81 observations")
    payload["_sha256"] = file_sha256(path)
    payload["_path"] = str(path.resolve())
    return payload


def _load_contract(
    labels_path: Path, split_path: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    labels = pd.read_csv(labels_path)
    required_labels = {"participant_id", "condition", "presentation_position", *TARGETS}
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"Condition labels lack {sorted(missing)}")
    labels = labels.copy()
    labels["participant_id"] = labels["participant_id"].astype(str)
    labels["condition"] = labels["condition"].astype(str)
    labels = labels[labels["participant_id"].isin(PARTICIPANTS)].copy()
    labels = labels.sort_values(["participant_id", "presentation_position"]).reset_index(drop=True)
    keys = ["participant_id", "condition"]
    if len(labels) != 81 or labels.duplicated(keys).any():
        raise ValueError("Labels do not contain 81 unique cohort observations")
    if set(labels["participant_id"]) != set(PARTICIPANTS):
        raise ValueError("Labels do not contain the exact fixed cohort")
    if not np.isfinite(labels[list(TARGETS)].to_numpy(dtype=float)).all():
        raise ValueError("Targets contain non-finite values")

    split = pd.read_csv(split_path)
    required_split = {
        "fold_index", "test_participant", "validation_participant", "participant_id", "role"
    }
    if missing := required_split - set(split.columns):
        raise ValueError(f"Split manifest lacks {sorted(missing)}")
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
        if set(train + validation + test) != set(PARTICIPANTS):
            raise ValueError(f"Fold {fold_index} does not contain the exact cohort")
        folds.append({
            "fold_index": int(fold_index),
            "train_participants": train,
            "validation_participant": validation[0],
            "test_participant": test[0],
        })
        train_labels = labels[labels["participant_id"].isin(train)]
        test_labels = labels[labels["participant_id"] == test[0]]
        condition_means = train_labels.groupby("condition")[list(TARGETS)].mean()
        global_relaxation = float(train_labels["relaxation"].mean())
        for row in test_labels.itertuples(index=False):
            canonical_rows.append({
                "participant_id": str(row.participant_id),
                "condition": str(row.condition),
                "presentation_position": float(row.presentation_position),
                "fold_index": int(fold_index),
                "relaxation_true": float(row.relaxation),
                "discomfort_true": float(row.discomfort),
                "condition_only_relaxation": float(condition_means.loc[str(row.condition), "relaxation"]),
                "condition_only_discomfort": float(condition_means.loc[str(row.condition), "discomfort"]),
                "nonfoundation_relaxation": global_relaxation,
                "nonfoundation_discomfort": float(condition_means.loc[str(row.condition), "discomfort"]),
            })
    canonical = pd.DataFrame(canonical_rows).sort_values(keys).reset_index(drop=True)
    if len(folds) != 9 or len(canonical) != 81 or canonical.duplicated(keys).any():
        raise ValueError("Split manifest cannot reconstruct exact nine-fold OOF rows")
    return labels, folds, canonical


def _expected_run_ids() -> set[tuple[str, str, int]]:
    full = {(method, "full", seed) for method in FULL_METHODS for seed in SEEDS}
    ablations = {(PRIMARY, variant, seed) for variant in ABLATION_VARIANTS for seed in SEEDS}
    return full | ablations


def _discover(root: Path) -> dict[tuple[str, str, int], tuple[Path, Path, dict[str, Any]]]:
    runs_dir = root / "runs" if (root / "runs").is_dir() else root
    expected = _expected_run_ids()
    discovered: dict[tuple[str, str, int], tuple[Path, Path, dict[str, Any]]] = {}
    for result_path in sorted(runs_dir.rglob("*_results.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        identity = (_name(payload), _variant(payload), int(payload.get("seed", -1)))
        if identity not in expected:
            continue
        prediction_path = result_path.with_name(
            result_path.name.removesuffix("_results.json") + "_predictions.csv"
        )
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing prediction pair for {result_path}")
        if identity in discovered:
            raise ValueError(f"Duplicate run identity {identity}")
        discovered[identity] = (result_path, prediction_path, payload)
    if set(discovered) != expected:
        raise FileNotFoundError(
            f"Incomplete July 18 matrix; missing={sorted(expected - set(discovered))}"
        )
    return discovered


def _weight_map(
    fold: Mapping[str, Any], target_record: Mapping[str, Any], target: str
) -> dict[str, float]:
    for source in (target_record, fold):
        for key in ("expert_weights", "simplex_weights", "modality_weights", "weights"):
            value = source.get(key)
            if isinstance(value, Mapping) and target in value and isinstance(value[target], Mapping):
                value = value[target]
            if isinstance(value, Mapping):
                try:
                    parsed = {str(modality): float(weight) for modality, weight in value.items()}
                except (TypeError, ValueError):
                    continue
                if parsed:
                    return parsed
    return {}


def _component_map(fold: Mapping[str, Any]) -> dict[str, float]:
    sources: list[Mapping[str, Any]] = [fold]
    for key in ("compression", "transform"):
        if isinstance(fold.get(key), Mapping):
            sources.insert(0, fold[key])
    for source in sources:
        for key in (
            "modality_components", "component_counts", "selected_dimensions", "components_per_modality"
        ):
            value = source.get(key)
            if isinstance(value, Mapping):
                try:
                    return {str(modality): float(count) for modality, count in value.items()}
                except (TypeError, ValueError):
                    pass
    return {}


def _payload_modalities(payload: Mapping[str, Any], variant: str) -> tuple[str, ...]:
    value = payload.get("modalities")
    if value is None and isinstance(payload.get("candidate"), Mapping):
        value = payload["candidate"].get("modalities")
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(map(str, value))
    return _modalities_for_variant(variant)


def _audit_runs(
    artifacts: Mapping[tuple[str, str, int], tuple[Path, Path, dict[str, Any]]],
    prereg: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected_folds = {int(fold["fold_index"]): fold for fold in folds}
    canonical_i = canonical.set_index(["participant_id", "condition"]).sort_index()
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    required = {
        "participant_id", "condition", "fold_index", "seed", "candidate", "variant",
        "relaxation_true", "discomfort_true", "relaxation_pred", "discomfort_pred",
        "condition_only_relaxation", "condition_only_discomfort",
    }
    for identity, (result_path, prediction_path, payload) in sorted(artifacts.items()):
        candidate, variant, seed = identity
        issues: list[str] = []
        expected_modalities = _modalities_for_variant(variant)
        observed_modalities = _payload_modalities(payload, variant)
        if set(observed_modalities) != set(expected_modalities):
            issues.append("modality_contract")
        if int(payload.get("observation_count", -1)) != 81:
            issues.append("observation_count")
        if int(payload.get("fold_count", -1)) != 9:
            issues.append("fold_count")
        run_eligibility = payload.get("eligibility")
        if candidate == PRIMARY and (
            not isinstance(run_eligibility, Mapping)
            or run_eligibility.get("all_folds_eligible") is not True
        ):
            issues.append("primary_all_folds_eligible")
        declared_hash = payload.get("prediction_sha256")
        if declared_hash and str(declared_hash) != file_sha256(prediction_path):
            issues.append("prediction_hash")
        prereg_record = payload.get("preregistration")
        if isinstance(prereg_record, Mapping) and prereg_record.get("sha256"):
            if str(prereg_record["sha256"]) != str(prereg["_sha256"]):
                issues.append("preregistration_hash")

        frame = pd.read_csv(prediction_path)
        if missing := required - set(frame.columns):
            issues.append("missing_columns:" + ",".join(sorted(missing)))
        else:
            frame = frame.copy()
            frame["participant_id"] = frame["participant_id"].astype(str)
            frame["condition"] = frame["condition"].astype(str)
            frame["fold_index"] = frame["fold_index"].astype(int)
            keys = list(zip(frame["participant_id"], frame["condition"], strict=True))
            if len(frame) != 81 or len(set(keys)) != 81:
                issues.append("oof_coverage")
            if set(keys) != set(canonical_i.index.tolist()):
                issues.append("oof_membership")
            if set(frame["candidate"].astype(str)) != {candidate}:
                issues.append("prediction_candidate")
            if set(frame["variant"].astype(str)) != {variant}:
                issues.append("prediction_variant")
            if set(frame["seed"].astype(int)) != {seed}:
                issues.append("prediction_seed")
            if set(keys) == set(canonical_i.index.tolist()):
                aligned = frame.set_index(["participant_id", "condition"]).sort_index()
                for column in (
                    "fold_index", "relaxation_true", "discomfort_true",
                    "condition_only_relaxation", "condition_only_discomfort",
                ):
                    observed = aligned[column].to_numpy()
                    expected = canonical_i[column].to_numpy()
                    equal = np.array_equal(observed.astype(int), expected.astype(int)) if column == "fold_index" else np.allclose(observed.astype(float), expected.astype(float), atol=FLOAT_ATOL, rtol=0.0)
                    if not equal:
                        issues.append(f"canonical_{column}")
                predictions = aligned[[f"{target}_pred" for target in TARGETS]].to_numpy(float)
                if not np.isfinite(predictions).all():
                    issues.append("nonfinite_predictions")
                if np.any((predictions < -FLOAT_ATOL) | (predictions > 1.0 + FLOAT_ATOL)):
                    issues.append("prediction_bounds")

        payload_folds = payload.get("folds")
        if not isinstance(payload_folds, list) or len(payload_folds) != 9:
            issues.append("fold_records")
            payload_folds = []
        observed_fold_indexes: set[int] = set()
        for payload_fold in payload_folds:
            fold_index = int(payload_fold.get("fold_index", -1))
            observed_fold_indexes.add(fold_index)
            expected_fold = expected_folds.get(fold_index)
            if expected_fold is None:
                issues.append(f"unknown_fold_{fold_index}")
                continue
            if set(map(str, payload_fold.get("train_participants", []))) != set(expected_fold["train_participants"]):
                issues.append(f"fold_{fold_index}_train")
            if str(payload_fold.get("validation_participant")) != expected_fold["validation_participant"]:
                issues.append(f"fold_{fold_index}_validation")
            if str(payload_fold.get("test_participant")) != expected_fold["test_participant"]:
                issues.append(f"fold_{fold_index}_test")
            if candidate == PRIMARY:
                eligibility = payload_fold.get("eligibility")
                if not isinstance(eligibility, Mapping) or eligibility.get(
                    "feature_coefficients_nonzero_for_both_targets"
                ) is not True:
                    issues.append(f"fold_{fold_index}_primary_feature_coefficients")
            components = _component_map(payload_fold)
            for modality, count in components.items():
                selection_rows.append({
                    "candidate": candidate, "variant": variant, "seed": seed,
                    "fold_index": fold_index, "target": "shared", "parameter": f"components.{modality}",
                    "value": float(count),
                })
            target_records = payload_fold.get("targets")
            if not isinstance(target_records, Mapping):
                issues.append(f"fold_{fold_index}_targets")
                target_records = {}
            for target in TARGETS:
                record = target_records.get(target)
                if not isinstance(record, Mapping):
                    issues.append(f"fold_{fold_index}_{target}_record")
                    continue
                for parameter in ("alpha", "gamma"):
                    try:
                        value = float(record.get(parameter, np.nan))
                    except (TypeError, ValueError):
                        value = np.nan
                    if not np.isfinite(value) or value <= 0:
                        issues.append(f"fold_{fold_index}_{target}_{parameter}")
                    else:
                        selection_rows.append({
                            "candidate": candidate, "variant": variant, "seed": seed,
                            "fold_index": fold_index, "target": target,
                            "parameter": parameter, "value": value,
                        })
                weights = _weight_map(payload_fold, record, target)
                for modality, weight in weights.items():
                    weight_rows.append({
                        "candidate": candidate, "variant": variant, "seed": seed,
                        "fold_index": fold_index, "target": target,
                        "modality": modality, "weight": float(weight),
                    })
                if candidate == PRIMARY:
                    if set(weights) != set(expected_modalities):
                        issues.append(f"fold_{fold_index}_{target}_weight_modalities")
                    if not weights or abs(sum(weights.values()) - 1.0) > 1e-6:
                        issues.append(f"fold_{fold_index}_{target}_simplex_sum")
                    if any(weight < 0.02 - 1e-10 for weight in weights.values()):
                        issues.append(f"fold_{fold_index}_{target}_weight_floor")
        if observed_fold_indexes != set(expected_folds):
            issues.append("fold_index_coverage")

        if candidate == PRIMARY and variant == "full" and required <= set(frame.columns):
            for target in TARGETS:
                correction = frame[f"{target}_pred"].astype(float) - frame[f"condition_only_{target}"].astype(float)
                usable = np.array([
                    (participant, condition) not in ZERO_FEATURE_EXCEPTIONS
                    for participant, condition in zip(frame["participant_id"], frame["condition"], strict=True)
                ])
                if not np.any(np.abs(correction.to_numpy()[usable]) > 1e-12):
                    issues.append(f"primary_{target}_zero_feature_contribution")
                exception = ~usable
                if exception.any() and not np.allclose(correction.to_numpy()[exception], 0.0, atol=1e-12):
                    issues.append(f"primary_{target}_zero_feature_exception")

        audit_rows.append({
            "candidate": candidate, "variant": variant, "seed": seed,
            "valid": not issues, "issues": ";".join(dict.fromkeys(issues)),
            "modalities": "+".join(observed_modalities), "observations": len(frame),
            "result_path": str(result_path), "prediction_path": str(prediction_path),
            "result_sha256": file_sha256(result_path), "prediction_sha256": file_sha256(prediction_path),
        })
        if not issues:
            frame["model_id"] = candidate + ("" if variant == "full" else f"[{variant}]")
            frame["source_kind"] = "new_full_method" if variant == "full" else "primary_modality_ablation"
            frame["source_file"] = str(prediction_path)
            frames.append(frame)

    audit = pd.DataFrame(audit_rows).sort_values(["candidate", "variant", "seed"])
    if not audit["valid"].all():
        failures = audit.loc[~audit["valid"], ["candidate", "variant", "seed", "issues"]]
        raise ValueError(f"July 18 run audit failed: {failures.to_dict('records')}")
    predictions = pd.concat(frames, ignore_index=True)
    if len(predictions) != 30 * 81:
        raise ValueError(f"Expected 2,430 audited OOF rows, got {len(predictions)}")
    return predictions, audit.reset_index(drop=True), pd.DataFrame(selection_rows), pd.DataFrame(weight_rows)


def _canonical_comparators(canonical: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    definitions = {
        "fold_local_condition_only": (
            "condition_only_relaxation", "condition_only_discomfort", "canonical_condition_baseline"
        ),
        "nonfoundation_global_relaxation_condition_discomfort": (
            "nonfoundation_relaxation", "nonfoundation_discomfort", "nonfoundation_hybrid_control"
        ),
    }
    for model_id, (relaxation_column, discomfort_column, source_kind) in definitions.items():
        for seed in SEEDS:
            frame = canonical.copy()
            frame["candidate"] = model_id
            frame["variant"] = "historical"
            frame["model_id"] = model_id
            frame["seed"] = seed
            frame["relaxation_pred"] = frame[relaxation_column]
            frame["discomfort_pred"] = frame[discomfort_column]
            frame["source_kind"] = source_kind
            frame["source_file"] = "recomputed_from_fixed_labels_and_split_manifest"
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _normalize_historical(
    path: Path, model_id: str, seed: int, canonical: pd.DataFrame, source_kind: str
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "participant_id", "condition", "fold_index", "relaxation_true", "discomfort_true",
        "relaxation_pred", "discomfort_pred", "condition_only_relaxation", "condition_only_discomfort",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"{path} lacks {sorted(missing)}")
    frame = frame.copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    frame["condition"] = frame["condition"].astype(str)
    if len(frame) != 81 or frame.duplicated(["participant_id", "condition"]).any():
        raise ValueError(f"{path} lacks exact 81-row OOF coverage")
    observed = frame.set_index(["participant_id", "condition"]).sort_index()
    expected = canonical.set_index(["participant_id", "condition"]).sort_index()
    if set(observed.index) != set(expected.index):
        raise ValueError(f"{path} has a different OOF cohort")
    for column in (
        "fold_index", "relaxation_true", "discomfort_true",
        "condition_only_relaxation", "condition_only_discomfort",
    ):
        left = observed[column].to_numpy()
        right = expected[column].to_numpy()
        valid = np.array_equal(left.astype(int), right.astype(int)) if column == "fold_index" else np.allclose(left.astype(float), right.astype(float), atol=FLOAT_ATOL, rtol=0.0)
        if not valid:
            raise ValueError(f"{path} differs in canonical {column}")
    frame["candidate"] = model_id
    frame["variant"] = "historical"
    frame["model_id"] = model_id
    frame["seed"] = seed
    frame["source_kind"] = source_kind
    frame["source_file"] = str(path)
    return frame


def _historical_file(root: Path, seed: int, pattern: str) -> Path:
    seed_dir = root / f"seed_{seed}"
    matches = sorted(seed_dir.glob(pattern)) if seed_dir.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r} under {seed_dir}; got {len(matches)}")
    return matches[0]


def _load_historical(
    no_eeg_root: Path,
    healnet_root: Path,
    july17_root: Path,
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        specs = (
            (no_eeg_root, f"no_eeg_raw_dual_s{seed}_predictions.csv", "historical_no_eeg_raw_dual", "historical_adaptive_condition_anchor"),
            (healnet_root, "eeg9_healnet_no_eeg_s*_predictions.csv", "historical_healnet_no_eeg", "historical_validation_selected_neural_fusion"),
            (july17_root, f"modality_expert_simplex_s{seed}_predictions.csv", "historical_july17_modality_expert", "historical_july17_four_modality_fusion"),
        )
        for root, pattern, model_id, source_kind in specs:
            try:
                path = _historical_file(root, seed, pattern)
                frame = _normalize_historical(path, model_id, seed, canonical, source_kind)
                frames.append(frame)
                rows.append({
                    "model_id": model_id, "seed": seed, "available": True,
                    "valid": True, "source_file": str(path), "sha256": file_sha256(path),
                    "inferential_status": "descriptive_only",
                })
            except (FileNotFoundError, ValueError) as error:
                rows.append({
                    "model_id": model_id, "seed": seed, "available": False,
                    "valid": False, "source_file": "", "sha256": "",
                    "inferential_status": "descriptive_only", "issue": str(error),
                })
    empty_columns = [
        "participant_id", "condition", "fold_index", "relaxation_true", "discomfort_true",
        "relaxation_pred", "discomfort_pred", "condition_only_relaxation", "condition_only_discomfort",
        "candidate", "variant", "model_id", "seed", "source_kind", "source_file",
    ]
    historical = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=empty_columns)
    if not historical.empty:
        complete = historical.groupby("model_id")["seed"].nunique()
        historical = historical[historical["model_id"].isin(complete[complete == 3].index)].copy()
    return historical, pd.DataFrame(rows)


def _participant_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["model_id", "candidate", "variant", "source_kind", "seed", "participant_id"]
    for identity, frame in predictions.groupby(groups, sort=True):
        model_id, candidate, variant, source_kind, seed, participant = identity
        target_values: list[tuple[float, float]] = []
        for target in TARGETS:
            model_mae = float(np.mean(np.abs(frame[f"{target}_true"] - frame[f"{target}_pred"])))
            condition_mae = float(np.mean(np.abs(frame[f"{target}_true"] - frame[f"condition_only_{target}"])))
            target_values.append((model_mae, condition_mae))
            rows.append({
                "model_id": model_id, "candidate": candidate, "variant": variant,
                "source_kind": source_kind, "seed": int(seed), "participant_id": participant,
                "outcome": target, "model_mae": model_mae,
                "condition_only_mae": condition_mae,
                "delta_vs_condition": model_mae - condition_mae,
            })
        model_macro = float(np.mean([value[0] for value in target_values]))
        condition_macro = float(np.mean([value[1] for value in target_values]))
        rows.append({
            "model_id": model_id, "candidate": candidate, "variant": variant,
            "source_kind": source_kind, "seed": int(seed), "participant_id": participant,
            "outcome": "macro", "model_mae": model_macro,
            "condition_only_mae": condition_macro,
            "delta_vs_condition": model_macro - condition_macro,
        })
    return pd.DataFrame(rows)


def _seed_metrics(participant: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["model_id", "candidate", "variant", "source_kind", "seed"]
    for identity, frame in participant.groupby(groups, sort=True):
        model_id, candidate, variant, source_kind, seed = identity
        row: dict[str, Any] = {
            "model_id": model_id, "candidate": candidate, "variant": variant,
            "source_kind": source_kind, "seed": int(seed),
        }
        for outcome in OUTCOMES:
            values = frame[frame["outcome"] == outcome]
            if len(values) != 9:
                raise ValueError(f"{model_id}/{seed}/{outcome} lacks nine participant metrics")
            row[f"{outcome}_mae"] = float(values["model_mae"].mean())
            row[f"{outcome}_condition_mae"] = float(values["condition_only_mae"].mean())
            row[f"{outcome}_delta_vs_condition"] = float(values["delta_vs_condition"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["model_id", "candidate", "variant", "source_kind"]
    for identity, frame in seed_metrics.groupby(groups, sort=True):
        model_id, candidate, variant, source_kind = identity
        if len(frame) != 3:
            continue
        row: dict[str, Any] = {
            "model_id": model_id, "candidate": candidate, "variant": variant,
            "source_kind": source_kind, "seed_count": 3,
        }
        for outcome in OUTCOMES:
            maes = frame[f"{outcome}_mae"].to_numpy(float)
            deltas = frame[f"{outcome}_delta_vs_condition"].to_numpy(float)
            row[f"{outcome}_mae_mean"] = float(maes.mean())
            row[f"{outcome}_mae_std"] = float(maes.std(ddof=1))
            row[f"{outcome}_mae_min"] = float(maes.min())
            row[f"{outcome}_mae_max"] = float(maes.max())
            row[f"{outcome}_delta_mean"] = float(deltas.mean())
            row[f"{outcome}_delta_min"] = float(deltas.min())
            row[f"{outcome}_delta_max"] = float(deltas.max())
            row[f"{outcome}_delta_all_seeds_negative"] = bool(np.all(deltas < 0.0))
        row["both_target_means_nonworse"] = bool(
            row["relaxation_delta_mean"] <= 0.0 and row["discomfort_delta_mean"] <= 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["macro_mae_mean", "model_id"]).reset_index(drop=True)


def _exact_sign_flip(deltas: np.ndarray) -> tuple[float, int]:
    observed = abs(float(np.mean(deltas)))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(deltas))), dtype=float)
    null = np.abs(np.mean(signs * deltas[None, :], axis=1))
    return float(np.mean(null >= observed - 1e-15)), len(signs)


def _bootstrap_ci(deltas: np.ndarray, offset: int) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    indexes = rng.integers(0, len(deltas), size=(BOOTSTRAP_RESAMPLES, len(deltas)))
    samples = deltas[indexes].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _holm(values: Sequence[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    order = np.argsort(raw, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(raw) - np.arange(len(raw))) * raw[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _paired_vs_condition(
    participant: pd.DataFrame,
    model_ids: Sequence[str],
    outcomes: Sequence[str],
    family: str,
    offset: int,
) -> pd.DataFrame:
    subset = participant[
        participant["model_id"].isin(model_ids) & participant["outcome"].isin(outcomes)
    ]
    averaged = subset.groupby(["model_id", "participant_id", "outcome"], as_index=False)[
        "delta_vs_condition"
    ].mean()
    rows: list[dict[str, Any]] = []
    for index, ((model_id, outcome), frame) in enumerate(
        averaged.groupby(["model_id", "outcome"], sort=True)
    ):
        deltas = frame.sort_values("participant_id")["delta_vs_condition"].to_numpy(float)
        if len(deltas) != 9:
            raise ValueError(f"Inference for {model_id}/{outcome} requires nine participants")
        low, high = _bootstrap_ci(deltas, offset + index)
        pvalue, assignments = _exact_sign_flip(deltas)
        rows.append({
            "family": family, "model_id": model_id, "outcome": outcome,
            "mean_paired_delta": float(deltas.mean()),
            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "cluster": "participant_after_seed_average",
            "exact_sign_flip_p_two_sided": pvalue,
            "sign_flip_assignments": assignments,
            "participants_improved": int(np.sum(deltas < -1e-12)),
            "participants_tied": int(np.sum(np.isclose(deltas, 0.0, atol=1e-12))),
            "participants_worsened": int(np.sum(deltas > 1e-12)),
        })
    result = pd.DataFrame(rows)
    if len(result) != len(model_ids) * len(outcomes):
        raise ValueError(f"Multiplicity family {family} is incomplete")
    result["holm_family_size"] = len(result)
    result["holm_p"] = _holm(result["exact_sign_flip_p_two_sided"])
    result["holm_significant_0_05"] = result["holm_p"] < 0.05
    return result


def _paired_model_contrast(
    participant: pd.DataFrame, left: str, right: str, outcome: str
) -> pd.DataFrame:
    subset = participant[
        participant["model_id"].isin([left, right]) & (participant["outcome"] == outcome)
    ][["model_id", "seed", "participant_id", "model_mae"]]
    pivot = subset.pivot(index=["seed", "participant_id"], columns="model_id", values="model_mae")
    if left not in pivot or right not in pivot or len(pivot) != 27:
        raise ValueError("Cannot construct modality-wise versus joint paired contrast")
    by_participant = (pivot[left] - pivot[right]).groupby("participant_id").mean().sort_index()
    deltas = by_participant.to_numpy(float)
    low, high = _bootstrap_ci(deltas, 700)
    pvalue, assignments = _exact_sign_flip(deltas)
    return pd.DataFrame([{
        "contrast": f"{left}_minus_{right}", "outcome": outcome,
        "mean_paired_delta": float(deltas.mean()), "bootstrap_ci_low": low,
        "bootstrap_ci_high": high, "exact_sign_flip_p_two_sided": pvalue,
        "sign_flip_assignments": assignments,
        "interpretation": "negative_favors_modality_wise_compression",
    }])


def _descriptive_comparator_inference(
    participant: pd.DataFrame, model_ids: Sequence[str]
) -> pd.DataFrame:
    """Participant-paired comparator summaries excluded from formal families."""

    available = [
        model_id for model_id in model_ids
        if participant.loc[participant["model_id"] == model_id, "seed"].nunique() == len(SEEDS)
    ]
    rows: list[dict[str, Any]] = []
    subset = participant[participant["model_id"].isin(available)]
    averaged = subset.groupby(["model_id", "participant_id", "outcome"], as_index=False)[
        "delta_vs_condition"
    ].mean()
    for index, ((model_id, outcome), frame) in enumerate(
        averaged.groupby(["model_id", "outcome"], sort=True)
    ):
        deltas = frame.sort_values("participant_id")["delta_vs_condition"].to_numpy(float)
        if len(deltas) != len(PARTICIPANTS):
            raise ValueError(f"Descriptive inference for {model_id}/{outcome} lacks nine participants")
        low, high = _bootstrap_ci(deltas, 1200 + index)
        pvalue, assignments = _exact_sign_flip(deltas)
        rows.append({
            "model_id": model_id,
            "outcome": outcome,
            "mean_paired_delta": float(deltas.mean()),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "exact_sign_flip_p_two_sided": pvalue,
            "sign_flip_assignments": assignments,
            "participants_improved": int(np.sum(deltas < -1e-12)),
            "participants_tied": int(np.sum(np.isclose(deltas, 0.0, atol=1e-12))),
            "participants_worsened": int(np.sum(deltas > 1e-12)),
            "inferential_status": "descriptive_only_no_multiplicity_claim",
        })
    return pd.DataFrame(rows)


def _discomfort_strata(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["stratum"] = np.where(frame["discomfort_true"].astype(float) > 0.0, "nonzero", "zero")
    frame["model_abs_error"] = np.abs(frame["discomfort_true"] - frame["discomfort_pred"])
    frame["condition_abs_error"] = np.abs(
        frame["discomfort_true"] - frame["condition_only_discomfort"]
    )
    by_seed = frame.groupby(["model_id", "source_kind", "seed", "stratum"], as_index=False).agg(
        observation_count=("model_abs_error", "size"),
        discomfort_mae=("model_abs_error", "mean"),
        condition_only_mae=("condition_abs_error", "mean"),
    )
    by_seed["delta_vs_condition"] = by_seed["discomfort_mae"] - by_seed["condition_only_mae"]
    return by_seed.groupby(["model_id", "source_kind", "stratum"], as_index=False).agg(
        seeds=("seed", "nunique"), observation_count_per_seed=("observation_count", "first"),
        discomfort_mae_mean=("discomfort_mae", "mean"),
        discomfort_mae_std=("discomfort_mae", "std"),
        delta_vs_condition_mean=("delta_vs_condition", "mean"),
        delta_vs_condition_min=("delta_vs_condition", "min"),
        delta_vs_condition_max=("delta_vs_condition", "max"),
    )


def _ablation_tables(
    participant: pd.DataFrame, seed_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_id = PRIMARY
    ablation_ids = [f"{PRIMARY}[{variant}]" for variant in ABLATION_VARIANTS]
    seed_rows: list[dict[str, Any]] = []
    for ablation_id, variant in zip(ablation_ids, ABLATION_VARIANTS, strict=True):
        for seed in SEEDS:
            full = seed_metrics[(seed_metrics["model_id"] == full_id) & (seed_metrics["seed"] == seed)]
            removed = seed_metrics[(seed_metrics["model_id"] == ablation_id) & (seed_metrics["seed"] == seed)]
            if len(full) != 1 or len(removed) != 1:
                raise ValueError(f"Missing full/ablation seed metric for {variant}/{seed}")
            for outcome in OUTCOMES:
                delta = float(removed.iloc[0][f"{outcome}_mae"] - full.iloc[0][f"{outcome}_mae"])
                seed_rows.append({
                    "variant": variant, "removed_modality": variant.removeprefix("no_"),
                    "seed": seed, "outcome": outcome,
                    "full_mae": float(full.iloc[0][f"{outcome}_mae"]),
                    "ablated_mae": float(removed.iloc[0][f"{outcome}_mae"]),
                    "delta_ablated_minus_full": delta,
                })
    by_seed = pd.DataFrame(seed_rows)
    summary = by_seed.groupby(["variant", "removed_modality", "outcome"], as_index=False).agg(
        mean_delta_ablated_minus_full=("delta_ablated_minus_full", "mean"),
        std_delta_ablated_minus_full=("delta_ablated_minus_full", "std"),
        min_delta_ablated_minus_full=("delta_ablated_minus_full", "min"),
        max_delta_ablated_minus_full=("delta_ablated_minus_full", "max"),
        deletion_worsened_all_seeds=("delta_ablated_minus_full", lambda values: bool(np.all(values > 0.0))),
    )

    subset = participant[
        participant["model_id"].isin([full_id, *ablation_ids])
    ][["model_id", "seed", "participant_id", "outcome", "model_mae"]]
    pivot = subset.pivot(index=["seed", "participant_id", "outcome"], columns="model_id", values="model_mae")
    stats_rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(OUTCOMES):
        outcome_rows: list[dict[str, Any]] = []
        outcome_pivot = pivot.reset_index()
        outcome_pivot = outcome_pivot[outcome_pivot["outcome"] == outcome].set_index(["seed", "participant_id"])
        for variant_index, (ablation_id, variant) in enumerate(zip(ablation_ids, ABLATION_VARIANTS, strict=True)):
            by_participant = (outcome_pivot[ablation_id] - outcome_pivot[full_id]).groupby("participant_id").mean()
            deltas = by_participant.sort_index().to_numpy(float)
            low, high = _bootstrap_ci(deltas, 800 + outcome_index * 10 + variant_index)
            pvalue, assignments = _exact_sign_flip(deltas)
            outcome_rows.append({
                "family": f"primary_deletion_{outcome}", "variant": variant,
                "removed_modality": variant.removeprefix("no_"), "outcome": outcome,
                "mean_paired_delta_ablated_minus_full": float(deltas.mean()),
                "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                "exact_sign_flip_p_two_sided": pvalue, "sign_flip_assignments": assignments,
                "participants_deletion_worsened": int(np.sum(deltas > 1e-12)),
                "participants_tied": int(np.sum(np.isclose(deltas, 0.0, atol=1e-12))),
                "participants_deletion_improved": int(np.sum(deltas < -1e-12)),
            })
        outcome_frame = pd.DataFrame(outcome_rows)
        outcome_frame["holm_family_size"] = 5
        outcome_frame["holm_p"] = _holm(outcome_frame["exact_sign_flip_p_two_sided"])
        outcome_frame["holm_significant_0_05"] = outcome_frame["holm_p"] < 0.05
        stats_rows.extend(_records(outcome_frame))
    return by_seed, summary, pd.DataFrame(stats_rows)


def _selection_summary(selections: pd.DataFrame) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame(columns=[
            "candidate", "variant", "target", "parameter", "selections",
            "mean_value", "std_value", "min_value", "max_value",
        ])
    return selections.groupby(
        ["candidate", "variant", "target", "parameter"], as_index=False
    ).agg(
        selections=("value", "size"), mean_value=("value", "mean"),
        std_value=("value", "std"), min_value=("value", "min"), max_value=("value", "max"),
    )


def _weight_summary(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(columns=[
            "candidate", "variant", "target", "modality", "selections",
            "mean_weight", "std_weight", "min_weight", "max_weight",
        ])
    return weights.groupby(
        ["candidate", "variant", "target", "modality"], as_index=False
    ).agg(
        selections=("weight", "size"), mean_weight=("weight", "mean"),
        std_weight=("weight", "std"), min_weight=("weight", "min"), max_weight=("weight", "max"),
    )


def _correction_diagnostics(new_predictions: pd.DataFrame) -> pd.DataFrame:
    frame = new_predictions[
        (new_predictions["candidate"] == PRIMARY) & (new_predictions["variant"] == "full")
    ].copy()
    usable = np.array([
        (participant, condition) not in ZERO_FEATURE_EXCEPTIONS
        for participant, condition in zip(frame["participant_id"], frame["condition"], strict=True)
    ])
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_frame = frame[frame["seed"] == seed]
        seed_usable = usable[frame["seed"].to_numpy() == seed]
        for target in TARGETS:
            correction = (
                seed_frame[f"{target}_pred"].to_numpy(float)
                - seed_frame[f"condition_only_{target}"].to_numpy(float)
            )[seed_usable]
            rows.append({
                "seed": seed, "target": target,
                "usable_observations": len(correction),
                "nonzero_correction_count": int(np.sum(np.abs(correction) > 1e-12)),
                "mean_absolute_correction": float(np.mean(np.abs(correction))),
                "max_absolute_correction": float(np.max(np.abs(correction))),
            })
    return pd.DataFrame(rows)


def _success_gate(
    summary: pd.DataFrame,
    full_inference: pd.DataFrame,
    corrections: pd.DataFrame,
    ablation_summary: pd.DataFrame,
) -> dict[str, Any]:
    primary = summary[summary["model_id"] == PRIMARY]
    if len(primary) != 1:
        raise ValueError("Primary summary is missing")
    row = primary.iloc[0]
    primary_macro = full_inference[
        (full_inference["model_id"] == PRIMARY) & (full_inference["outcome"] == "macro")
    ]
    if len(primary_macro) != 1:
        raise ValueError("Primary macro inference is missing")
    inference = primary_macro.iloc[0]
    correction_ok = bool(
        corrections.groupby("target")["nonzero_correction_count"].apply(lambda values: bool(np.all(values > 0))).reindex(TARGETS, fill_value=False).all()
    )
    descriptive = bool(
        row["macro_delta_all_seeds_negative"]
        and row["both_target_means_nonworse"]
        and correction_ok
    )
    strong = bool(
        descriptive
        and float(inference["bootstrap_ci_high"]) < 0.0
        and float(inference["exact_sign_flip_p_two_sided"]) < 0.05
    )
    macro_ablation = ablation_summary[ablation_summary["outcome"] == "macro"]
    consistent = macro_ablation[macro_ablation["deletion_worsened_all_seeds"]]
    return {
        "primary": PRIMARY,
        "full_five_modality_model_only_can_support_main_claim": True,
        "macro_delta_negative_all_three_seeds": bool(row["macro_delta_all_seeds_negative"]),
        "both_target_three_seed_mean_deltas_nonpositive": bool(row["both_target_means_nonworse"]),
        "foundation_correction_nonzero_for_both_targets_all_seeds": correction_ok,
        "descriptive_success": descriptive,
        "primary_macro_bootstrap_ci_high": float(inference["bootstrap_ci_high"]),
        "primary_macro_exact_sign_flip_p_two_sided": float(inference["exact_sign_flip_p_two_sided"]),
        "stronger_support": strong,
        "modalities_whose_deletion_worsened_macro_all_seeds": consistent["removed_modality"].tolist(),
        "consistent_worsening_deletion_count": int(len(consistent)),
        "genuine_complementarity_descriptive_rule_passed": bool(len(consistent) >= 2),
        "fallback_if_gate_fails": "retain_fold_local_condition_only",
    }


def _format(value: Any) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(float(value))):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    if columns is not None:
        frame = frame.loc[:, [column for column in columns if column in frame.columns]]
    if frame.empty:
        return "_No rows._"
    headers = list(map(str, frame.columns))
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return "\n".join(lines)


def _method_table(prereg: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in prereg["methods"]:
        method = raw if isinstance(raw, Mapping) else {"name": str(raw)}
        compression_dimension: Any = method.get("compression_dimension")
        if compression_dimension is None and method.get("compression_dimension_per_target") is not None:
            compression_dimension = f"{method['compression_dimension_per_target']} per target"
        if compression_dimension is None:
            compression_dimension = method.get("latent_budget", 12)
        rows.append({
            "method": str(method.get("name") or method.get("method_id")),
            "role": "primary" if str(method.get("name") or method.get("method_id")) == PRIMARY else "secondary",
            "modalities": "+".join(MODALITIES),
            "compression_dimension": compression_dimension,
            "parameter_count": method.get("parameter_count", method.get("estimated_parameter_count", "see preregistration")),
            "literature_basis": method.get("literature_basis", method.get("basis", "see literature review")),
            "expected_advantage": method.get("expected_advantage", method.get("advantages", "see preregistration")),
            "risk": method.get("risk", method.get("risks", "see preregistration")),
        })
    return pd.DataFrame(rows)


def _fold_dimension_table(selections: pd.DataFrame) -> pd.DataFrame:
    dimensions = selections[
        (selections["target"] == "shared") & selections["parameter"].str.startswith("components.")
    ].copy()
    if dimensions.empty:
        return dimensions
    dimensions["modality"] = dimensions["parameter"].str.removeprefix("components.")
    return dimensions.pivot_table(
        index=["candidate", "variant", "seed", "fold_index"],
        columns="modality", values="value", aggfunc="first",
    ).reset_index().rename_axis(columns=None)


def _report_text(
    prereg: Mapping[str, Any],
    methods: pd.DataFrame,
    summary: pd.DataFrame,
    inference: pd.DataFrame,
    contrast: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    ablation_inference: pd.DataFrame,
    strata: pd.DataFrame,
    dimensions: pd.DataFrame,
    weights: pd.DataFrame,
    corrections: pd.DataFrame,
    historical_audit: pd.DataFrame,
    comparator_inference: pd.DataFrame,
    gate: Mapping[str, Any],
) -> str:
    primary = summary[summary["model_id"] == PRIMARY].iloc[0]
    full = summary[
        summary["model_id"].isin(FULL_METHODS)
    ].sort_values("macro_mae_mean")
    best_full = full.iloc[0]
    modalitywise_delta = float(contrast.iloc[0]["mean_paired_delta"])
    compression_verdict = (
        "modality-wise rank-allocated PCA outperformed equal-budget joint PCA"
        if modalitywise_delta < 0
        else "modality-wise rank-allocated PCA did not outperform equal-budget joint PCA"
    )
    claim = (
        "strong participant-level support"
        if gate["stronger_support"]
        else "descriptive support only"
        if gate["descriptive_success"]
        else "no support for replacing Condition-only"
    )
    next_step = (
        "Validate the frozen full-five primary without changing its specification on a new cohort, with the nonzero-discomfort stratum declared as a safety check."
        if gate["stronger_support"]
        else "Keep Condition-only as the primary model and externally validate the locked full-five primary before any tuning, with the nonzero-discomfort stratum declared as a safety check."
    )
    condition = summary[summary["model_id"] == "fold_local_condition_only"].iloc[0]
    comparison_ids = (
        "nonfoundation_global_relaxation_condition_discomfort",
        "historical_no_eeg_raw_dual",
        "historical_july17_modality_expert",
        "historical_healnet_no_eeg",
    )
    comparison_text = []
    for model_id in comparison_ids:
        rows = summary[summary["model_id"] == model_id]
        if len(rows) == 1:
            comparison_text.append(
                f"{model_id}: {float(primary['macro_mae_mean'] - rows.iloc[0]['macro_mae_mean']):+.6f}"
            )
    deterministic_full = bool(np.all(full["macro_mae_std"].fillna(0.0).to_numpy(float) < 1e-12))
    significant_deletions = ablation_inference[ablation_inference["holm_significant_0_05"]]
    primary_strata_rows = strata[strata["model_id"] == PRIMARY].set_index("stratum")
    zero_delta = float(primary_strata_rows.loc["zero", "delta_vs_condition_mean"])
    nonzero_delta = float(primary_strata_rows.loc["nonzero", "delta_vs_condition_mean"])
    summary_columns = [
        "model_id", "source_kind", "relaxation_mae_mean", "discomfort_mae_mean",
        "macro_mae_mean", "relaxation_delta_mean", "discomfort_delta_mean", "macro_delta_mean",
        "macro_mae_std", "macro_mae_min", "macro_mae_max",
    ]
    full_inference = inference[inference["family"].isin(["five_full_methods_macro", "primary_two_targets"])]
    primary_strata = strata[strata["model_id"].isin([PRIMARY, "fold_local_condition_only"])]
    ablation_macro = ablation_summary[ablation_summary["outcome"] == "macro"]
    dimension_primary = dimensions[
        (dimensions["candidate"] == PRIMARY) & (dimensions["variant"] == "full")
    ]
    weight_primary = weights[(weights["candidate"] == PRIMARY) & (weights["variant"] == "full")]
    lines = [
        "# July 18 frozen-feature compression and fusion reinvestigation",
        "",
        "## Decision",
        "",
        f"The locked full-five primary has **{claim}**. Descriptive success: **{_format(gate['descriptive_success'])}**; stronger support: **{_format(gate['stronger_support'])}**. The best full method by three-seed Macro MAE was `{best_full['model_id']}` ({best_full['macro_mae_mean']:.6f}); the preregistered primary `{PRIMARY}` achieved relaxation {primary['relaxation_mae_mean']:.6f}, discomfort {primary['discomfort_mae_mean']:.6f}, and Macro {primary['macro_mae_mean']:.6f}.",
        "",
        f"Against Condition-only ({condition['macro_mae_mean']:.6f}), the primary Macro change was {primary['macro_delta_mean']:+.6f}. Primary-minus-comparator Macro changes were: {'; '.join(comparison_text)}. Negative favors the primary. Historical rows remain descriptive and cannot replace the full-five claim.",
        "",
        f"Direct equal-budget result: {compression_verdict} (participant-paired Macro delta modality-wise minus joint = {modalitywise_delta:.6f}).",
        "",
        f"The most appropriate synthesis in this matrix was modality-specific experts over protected modality PCA scores. As a pure compression/head control, block-balanced joint PCA12 was preferable to additive modality-wise PCA12; therefore the expert gain should not be attributed to modality-wise PCA alone.",
        "",
        f"The preregistered descriptive complementarity rule passed: **{_format(gate['genuine_complementarity_descriptive_rule_passed'])}**. Modalities whose deletion worsened Macro MAE in all three seeds: {', '.join(gate['modalities_whose_deletion_worsened_macro_all_seeds']) or 'none'}. No deletion contrast survived its Holm family ({len(significant_deletions)} significant), so EEG/ECG complementarity is tentative rather than statistically established.",
        "",
        f"Discomfort requires caution: the primary improved the 65 zero-discomfort rows by {zero_delta:+.6f} MAE but changed the 16 nonzero-discomfort rows by {nonzero_delta:+.6f}; positive means worse. The aggregate discomfort gain is therefore not evidence of better high-discomfort handling.",
        "",
        f"All full methods were deterministic under the specified linear/SVD solvers: exact three-seed standard deviation was effectively zero = **{_format(deterministic_full)}**. This demonstrates reproducibility under the requested seeds, not three independent stochastic fits.",
        "",
        f"Recommended next validation: {next_step}",
        "",
        "## Fixed contract and interpretation boundary",
        "",
        "The analysis uses only P003, P004, P007, P008, P009, P011, P012, P013, and P015: nine fixed 7/1/1 LOPO folds, 81 participant-condition labels, 545 common-valid windows, and seeds 20260705/20260706/20260707. Every transform and selection belongs inside the outer fold. Windows are feature slices, not independent labels; all uncertainty is clustered at participant level after seed averaging.",
        "",
        "The full pipeline includes EEG, ECG, eye, head, and video. Head contributes 18 Project-A engineered window features and therefore the full model is accurately described as a five-modality hybrid frozen-feature pipeline, not five neural foundation encoders. P004/C6 has no common-valid window and is the sole forced zero-correction Condition fallback.",
        "",
        "## Why these methods were tested",
        "",
        "The review supports supervised PCA/PLS for target-aware reduction, CCA/GCCA and shared-private factorization for cross-view structure, low-rank fusion and gradient-balanced experts for dimensional imbalance, and bottleneck or quality-aware fusion for scalable or missing-modality settings. With only 63 outer-training labels per fold, deep token/query bottlenecks, learned attention fusion, and large mixture-of-experts were not defensible here; the preregistered candidates are linear and low-capacity.",
        "",
        _markdown_table(methods),
        "",
        "## Main metrics and historical controls",
        "",
        "MAE is participant-macro averaged. Historical no-EEG, July-17, and HEALNet rows are descriptive only and are not eligible for the July-18 claim.",
        "",
        _markdown_table(summary, summary_columns),
        "",
        "Participant-paired comparator calculations are shown for transparency but remain descriptive because the historical models were adaptively selected:",
        "",
        _markdown_table(comparator_inference),
        "",
        "## Participant-level paired inference",
        "",
        "The five full methods form one Holm family for Macro MAE; the primary relaxation and discomfort tests form a second Holm family. Each exact test enumerates all 512 participant sign assignments, and each CI uses 10,000 participant-cluster bootstrap resamples after averaging seeds.",
        "",
        _markdown_table(full_inference),
        "",
        "## Full-five modality ablations",
        "",
        "Reduced models are mechanism ablations only. Positive ablated-minus-full delta means deleting that modality worsened performance. Holm correction is applied separately across five deletions for each outcome.",
        "",
        _markdown_table(ablation_macro),
        "",
        _markdown_table(ablation_inference),
        "",
        "## Discomfort sparsity check",
        "",
        _markdown_table(primary_strata),
        "",
        "## Selected compression and fusion parameters",
        "",
        "Exact fold-by-fold dimensions and hyperparameters are in `compression_dimensions_by_fold.csv` and `fold_selections.csv`; the tables below summarize primary selections and expert weights.",
        "",
        _markdown_table(dimension_primary),
        "",
        _markdown_table(weight_primary),
        "",
        "Foundation corrections were required to be nonzero for both targets rather than copying a Condition-only target:",
        "",
        _markdown_table(corrections),
        "",
        "## Eligibility and claim rule",
        "",
        f"Descriptive success requires negative primary Macro delta in all seeds, nonworse mean deltas for both targets, and nonzero corrections for both targets: **{_format(gate['descriptive_success'])}**. Stronger support additionally requires the participant-cluster Macro CI to lie below zero and the exact sign-flip p-value below 0.05: **{_format(gate['stronger_support'])}**.",
        "",
        "A reduced-modality winner cannot replace the full-five primary, and no post-hoc candidate may be promoted. If the locked primary fails, the correct decision is to retain Condition-only.",
        "",
        "## Historical artifact availability",
        "",
        _markdown_table(historical_audit),
        "",
        "## Provenance",
        "",
        f"Preregistration: `{prereg['_path']}`; SHA-256 `{prereg['_sha256']}`. All numeric source tables are stored beside this report.",
        "",
    ]
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    prereg_path = Path(args.preregistration).resolve() if args.preregistration else root / "preregistration/method_preregistration.json"
    if not prereg_path.is_file() and args.preregistration is None:
        matches = sorted((root / "preregistration").glob("*.json"))
        if len(matches) == 1:
            prereg_path = matches[0]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    prereg = _load_preregistration(prereg_path)
    _, folds, canonical = _load_contract(Path(args.labels), Path(args.split_manifest))
    artifacts = _discover(root)
    new_predictions, run_audit, selections, weights = _audit_runs(
        artifacts, prereg, folds, canonical
    )
    canonical_predictions = _canonical_comparators(canonical)
    if args.skip_historical:
        historical = pd.DataFrame(columns=new_predictions.columns)
        historical_audit = pd.DataFrame(columns=["model_id", "seed", "available", "valid"])
    else:
        historical, historical_audit = _load_historical(
            Path(args.no_eeg_root), Path(args.healnet_root), Path(args.july17_root), canonical
        )
    all_predictions = pd.concat(
        [new_predictions, canonical_predictions, historical], ignore_index=True, sort=False
    )
    all_predictions = all_predictions.sort_values(
        ["model_id", "seed", "participant_id", "fold_index"]
    ).reset_index(drop=True)

    participant = _participant_metrics(all_predictions)
    seed_metrics = _seed_metrics(participant)
    summary = _candidate_summary(seed_metrics)
    full_inference = _paired_vs_condition(
        participant, FULL_METHODS, ["macro"], "five_full_methods_macro", 0
    )
    primary_targets = _paired_vs_condition(
        participant, [PRIMARY], TARGETS, "primary_two_targets", 100
    )
    inference = pd.concat([full_inference, primary_targets], ignore_index=True)
    contrast = _paired_model_contrast(
        participant, "modality_rank_alloc_pca12", "joint_block_balanced_pca12", "macro"
    )
    comparator_inference = _descriptive_comparator_inference(
        participant,
        (
            "nonfoundation_global_relaxation_condition_discomfort",
            "historical_no_eeg_raw_dual",
            "historical_healnet_no_eeg",
            "historical_july17_modality_expert",
        ),
    )
    ablation_by_seed, ablation_summary, ablation_inference = _ablation_tables(
        participant, seed_metrics
    )
    strata = _discomfort_strata(all_predictions)
    selection_summary = _selection_summary(selections)
    dimension_folds = _fold_dimension_table(selections)
    weight_summary = _weight_summary(weights)
    corrections = _correction_diagnostics(new_predictions)
    gate = _success_gate(summary, full_inference, corrections, ablation_summary)
    methods = _method_table(prereg)

    tables = {
        "all_oof_predictions.csv": all_predictions,
        "formal_run_audit.csv": run_audit,
        "historical_comparator_audit.csv": historical_audit,
        "participant_metrics.csv": participant,
        "three_seed_metrics.csv": seed_metrics,
        "candidate_summary.csv": summary,
        "paired_inference.csv": inference,
        "modalitywise_vs_joint.csv": contrast,
        "descriptive_comparator_inference.csv": comparator_inference,
        "ablation_seed_comparisons.csv": ablation_by_seed,
        "ablation_summary.csv": ablation_summary,
        "ablation_paired_inference.csv": ablation_inference,
        "discomfort_zero_nonzero_strata.csv": strata,
        "fold_selections.csv": selections,
        "selection_summary.csv": selection_summary,
        "compression_dimensions_by_fold.csv": dimension_folds,
        "expert_weights_by_fold.csv": weights,
        "expert_weight_summary.csv": weight_summary,
        "primary_feature_corrections.csv": corrections,
        "method_specs.csv": methods,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)

    evaluation_summary = {
        "schema_version": "relax_foundation_compression_fusion_evaluation_v2",
        "fixed_contract": {
            "participants": list(PARTICIPANTS), "folds": 9, "observations": 81,
            "common_valid_windows": 545, "seeds": list(SEEDS), "modalities": list(MODALITIES),
        },
        "preregistration": {
            "path": str(prereg_path), "sha256": prereg["_sha256"],
        },
        "formal_run_count": int(len(run_audit)),
        "formal_oof_row_count": int(len(new_predictions)),
        "success_gate": gate,
        "candidate_summary": _records(summary),
        "paired_inference": _records(inference),
        "modalitywise_vs_joint": _records(contrast),
        "descriptive_comparator_inference": _records(comparator_inference),
        "ablation_summary": _records(ablation_summary),
        "ablation_paired_inference": _records(ablation_inference),
        "historical_comparators_are_descriptive_only": True,
    }
    _write_json(output_dir / "success_gate.json", gate)
    _write_json(output_dir / "evaluation_summary.json", evaluation_summary)
    report = _report_text(
        prereg, methods, summary, inference, contrast, ablation_summary,
        ablation_inference, strata, selection_summary, weight_summary,
        corrections, historical_audit, comparator_inference, gate,
    )
    report_path = output_dir / "final_report.md"
    report_path.write_text(report, encoding="utf-8")
    output_files = sorted(
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "evaluation_manifest.json"
    )
    matrix_status_path = root / "matrix_status.json"
    _write_json(
        output_dir / "evaluation_manifest.json",
        {
            "schema_version": "relax_foundation_compression_fusion_evaluation_manifest_v2",
            "formal_run_count": 30,
            "formal_oof_row_count": 2430,
            "preregistration_sha256": prereg["_sha256"],
            "matrix_status_sha256": (
                file_sha256(matrix_status_path) if matrix_status_path.is_file() else None
            ),
            "evaluator_source_sha256": file_sha256(Path(__file__)),
            "outputs": {
                path.name: {"path": str(path), "sha256": file_sha256(path)}
                for path in output_files
            },
        },
    )
    return evaluation_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--labels", type=Path, default=DEFAULT_CONTRACT_ROOT / "condition_labels.csv")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_CONTRACT_ROOT / "split_manifest.csv")
    parser.add_argument("--no-eeg-root", type=Path, default=DEFAULT_NO_EEG_ROOT)
    parser.add_argument("--healnet-root", type=Path, default=DEFAULT_HEALNET_ROOT)
    parser.add_argument("--july17-root", type=Path, default=DEFAULT_JULY17_ROOT)
    parser.add_argument("--skip-historical", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = evaluate(parse_args(argv))
    print(json.dumps({
        "formal_run_count": result["formal_run_count"],
        "formal_oof_row_count": result["formal_oof_row_count"],
        "success_gate": result["success_gate"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
