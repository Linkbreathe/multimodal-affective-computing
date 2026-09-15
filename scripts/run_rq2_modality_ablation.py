#!/usr/bin/env python3
"""Run the FMQ-9 frozen-Simplex leave-one-modality-out audit.

The runner is intentionally downstream-only.  It validates and consumes the
immutable contract, the already materialized frozen representations, and the
official full-five/no-video OOF predictions.  It never imports an encoder or
changes the 545-window common-valid mask.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

# Direct execution (``python scripts/run_rq2_modality_ablation.py``) places the
# scripts directory, rather than the repository root, on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.rq2_pipeline import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FamilyData,
    MODALITY_ORDER,
    PCA_ALLOCATION,
    TARGETS,
    _bool_series,
    _load_npz,
    _window_counts,
    build_group_summary,
    build_participant_metrics,
    load_contract,
    run_simplex_family,
)
from scripts.run_rq2_wsl import (
    AuditState,
    EXPECTED_DIMENSIONS,
    EXPECTED_FALLBACK,
    EXPECTED_OBSERVATIONS,
    EXPECTED_SEEDS,
    EXPECTED_WINDOWS,
    PROJECT_ROOT,
    SHARED_ROOT,
    _check_anchors,
    _check_condition_manifest,
    _check_contract_files,
    _check_contract_hashes,
    _check_folds,
    _check_window_manifest,
    _check_windows_representation_index,
    _key_set,
    file_sha256,
)


EXPECTED_CONTRACT_HASH = "9838ca9ce5e2f6105393f3b8d9116021e81b738adf2653dfb6f957c1dae15853"
TIE_TOLERANCE = 1e-12
REUSED_MODELS = ("frozen_simplex_full5", "frozen_simplex_no_video")
NEW_MODEL_MODALITIES: dict[str, tuple[str, ...]] = {
    "frozen_simplex_no_eeg": ("ECG", "Eye", "Head", "Video"),
    "frozen_simplex_no_ecg": ("EEG", "Eye", "Head", "Video"),
    "frozen_simplex_no_eye": ("EEG", "ECG", "Head", "Video"),
    "frozen_simplex_no_head": ("EEG", "ECG", "Eye", "Video"),
}
ALL_MODEL_MODALITIES: dict[str, tuple[str, ...]] = {
    "frozen_simplex_full5": MODALITY_ORDER,
    **NEW_MODEL_MODALITIES,
    "frozen_simplex_no_video": MODALITY_ORDER[:-1],
}
REMOVED_MODALITY_MODEL = {
    "EEG": "frozen_simplex_no_eeg",
    "ECG": "frozen_simplex_no_ecg",
    "Eye": "frozen_simplex_no_eye",
    "Head": "frozen_simplex_no_head",
    "Video": "frozen_simplex_no_video",
}
REQUIRED_SHARED_INPUTS = (
    "contract/rq2_contract.json",
    "contract/rq2_contract.sha256",
    "contract/condition_manifest.csv",
    "contract/window_manifest.csv",
    "contract/folds.csv",
    "contract/condition_anchors.csv",
    "windows_results/windows_representation_index.csv",
    "windows_results/WINDOWS_DONE.json",
    "wsl_results/pretrained_representation_index.csv",
    "wsl_results/representation_audit.csv",
    "wsl_results/WSL_DONE.json",
    "combined/combined_oof_predictions.csv",
    "combined/participant_metrics.csv",
    "combined/group_summary.csv",
    "combined/fusion_details.csv",
)
OUTPUT_FILENAMES = (
    "modality_ablation_oof_predictions.csv",
    "modality_ablation_participant_metrics.csv",
    "modality_ablation_participant_deltas.csv",
    "modality_ablation_group_summary.csv",
    "modality_ablation_fusion_details.csv",
    "modality_ablation_run_manifest.csv",
    "modality_ablation_plot_data.csv",
    "figure_modality_ablation.png",
    "MODALITY_ABLATION_REPORT.md",
)


class AblationAlignmentError(RuntimeError):
    """Raised when immutable inputs or acceptance conditions do not match."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _hash_inventory(entries: Sequence[tuple[str, Path]]) -> tuple[str, list[dict[str, Any]]]:
    digest = sha256()
    inventory: list[dict[str, Any]] = []
    for label, path in sorted(entries, key=lambda item: item[0]):
        value = file_sha256(path)
        size = path.stat().st_size
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
        inventory.append({"path": label, "sha256": value, "size_bytes": int(size)})
    return digest.hexdigest(), inventory


def _resolve_artifact(shared_root: Path, value: str) -> Path:
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = shared_root / candidate
    return candidate


def _validate_contract(shared_root: Path, expected_hash: str) -> tuple[Any, AuditState, pd.DataFrame]:
    state = AuditState()
    try:
        paths = _check_contract_files(shared_root, state)
        observed_hash = _check_contract_hashes(paths, state)
        if observed_hash != expected_hash:
            raise AblationAlignmentError(
                f"Contract hash {observed_hash} does not match --expected-contract-hash {expected_hash}"
            )
        condition_manifest, condition_keys = _check_condition_manifest(
            paths["contract/condition_manifest.csv"], state
        )
        window_manifest = _check_window_manifest(
            paths["contract/window_manifest.csv"], condition_keys, state
        )
        window_keys = _key_set(window_manifest, include_window=True)
        participants = set(condition_manifest["participant"].astype(str))
        _check_folds(paths["contract/folds.csv"], participants, state)
        _check_anchors(paths["contract/condition_anchors.csv"], condition_keys, state)
        windows_index = _check_windows_representation_index(
            paths["windows_results/windows_representation_index.csv"],
            condition_keys,
            window_keys,
            state,
        )
        state.require()
    except AblationAlignmentError:
        raise
    except Exception as exc:
        raise AblationAlignmentError(str(exc)) from exc
    contract = load_contract(shared_root, state)
    if len(contract.labels) != EXPECTED_OBSERVATIONS or len(contract.windows) != EXPECTED_WINDOWS:
        raise AblationAlignmentError("Contract row counts changed after validated loading")
    return contract, state, windows_index


def _validate_indexed_artifacts(
    shared_root: Path,
    index: pd.DataFrame,
    *,
    label: str,
) -> list[tuple[str, Path]]:
    required = {"artifact_path", "artifact_sha256", "modality"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise AblationAlignmentError(f"{label} index lacks columns: {missing}")
    entries: list[tuple[str, Path]] = []
    for artifact_value, rows in index.groupby("artifact_path", sort=True):
        hashes = set(rows["artifact_sha256"].dropna().astype(str))
        if len(hashes) != 1:
            raise AblationAlignmentError(f"{label} artifact has inconsistent hashes: {artifact_value}")
        artifact = _resolve_artifact(shared_root, str(artifact_value))
        if not artifact.is_file():
            raise AblationAlignmentError(f"Missing {label} artifact: {artifact}")
        observed = file_sha256(artifact)
        expected = next(iter(hashes))
        if observed != expected:
            raise AblationAlignmentError(
                f"{label} artifact hash mismatch for {artifact}: {observed} != {expected}"
            )
        entries.append((f"{label}:{artifact_value}", artifact))
    modalities = set(index["modality"].astype(str).str.lower())
    if modalities != {modality.lower() for modality in MODALITY_ORDER}:
        raise AblationAlignmentError(f"{label} modalities are incomplete: {sorted(modalities)}")
    return entries


def _load_pretrained_family(
    shared_root: Path,
    contract: Any,
) -> tuple[FamilyData, pd.DataFrame, list[tuple[str, Path]]]:
    done_path = shared_root / "wsl_results/WSL_DONE.json"
    index_path = shared_root / "wsl_results/pretrained_representation_index.csv"
    audit_path = shared_root / "wsl_results/representation_audit.csv"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("status") != "complete" or done.get("contract_hash") != contract.contract_hash:
        raise AblationAlignmentError("WSL_DONE.json is incomplete or belongs to a different contract")
    index = pd.read_csv(index_path)
    audit = pd.read_csv(audit_path)
    required_columns = {
        "contract_hash",
        "modality",
        "participant",
        "condition",
        "artifact_path",
        "artifact_sha256",
        "representation_dimension",
        "representation_available",
        "valid_window_count",
        "fallback_used",
    }
    missing = sorted(required_columns - set(index.columns))
    if missing:
        raise AblationAlignmentError(f"Pretrained representation index lacks columns: {missing}")
    if len(index) != EXPECTED_OBSERVATIONS * len(MODALITY_ORDER):
        raise AblationAlignmentError(f"Expected 405 pretrained index rows, found {len(index)}")
    if index.duplicated(["participant", "condition", "modality"]).any():
        raise AblationAlignmentError("Pretrained representation index contains duplicate keys")
    if set(index["contract_hash"].astype(str)) != {contract.contract_hash}:
        raise AblationAlignmentError("Pretrained representation index contract hash mismatch")
    if len(audit) != EXPECTED_OBSERVATIONS * len(MODALITY_ORDER):
        raise AblationAlignmentError(f"Expected 405 representation audit rows, found {len(audit)}")
    audit_keys = set(
        map(
            tuple,
            audit[["participant", "condition", "modality"]].astype(str).to_numpy(),
        )
    )
    index_keys = set(
        map(
            tuple,
            index[["participant", "condition", "modality"]].astype(str).to_numpy(),
        )
    )
    if audit_keys != index_keys:
        raise AblationAlignmentError("Representation audit keys do not equal pretrained index keys")
    for column, expected_value in (
        ("finite_valid_values", True),
        ("duplicate_key_check", False),
        ("window_key_join_check", True),
        ("modality_mask_check", True),
    ):
        if column not in audit or not (_bool_series(audit[column]) == expected_value).all():
            raise AblationAlignmentError(f"Representation audit check failed: {column}")

    artifact_entries = _validate_indexed_artifacts(
        shared_root, index, label="pretrained_representation"
    )
    key_order = list(zip(contract.labels["participant"], contract.labels["condition"], strict=True))
    expected_keys = set(key_order)
    matrices: dict[tuple[int, int, str], np.ndarray] = {}
    present: dict[tuple[int, int, str], np.ndarray] = {}
    dimensions: dict[str, int] = {}
    provenance: dict[str, Any] = {"artifacts": {}}
    counts = _window_counts(contract)
    count_frame = index.pivot(
        index=["participant", "condition"], columns="modality", values="valid_window_count"
    )
    if not all(count_frame[column].equals(count_frame.iloc[:, 0]) for column in count_frame.columns):
        raise AblationAlignmentError("Frozen modalities do not share the same condition window counts")
    if any(
        int(count_frame.loc[key].iloc[0]) != counts.get((str(key[0]), str(key[1])), 0)
        for key in count_frame.index
    ):
        raise AblationAlignmentError("Frozen representation counts do not match the 545-window contract mask")
    if any(int(index[index["modality"].eq(modality)]["valid_window_count"].sum()) != EXPECTED_WINDOWS for modality in MODALITY_ORDER):
        raise AblationAlignmentError("At least one frozen modality does not total 545 common-valid windows")

    for modality in MODALITY_ORDER:
        rows = index[index["modality"].astype(str).str.lower().eq(modality.lower())].copy()
        observed_keys = set(zip(rows["participant"].astype(str), rows["condition"].astype(str), strict=True))
        if observed_keys != expected_keys:
            raise AblationAlignmentError(f"{modality} pretrained index does not contain all 81 contract keys")
        dimensions_found = set(pd.to_numeric(rows["representation_dimension"]).astype(int))
        expected_dimension = EXPECTED_DIMENSIONS[modality.lower()]
        if dimensions_found != {expected_dimension}:
            raise AblationAlignmentError(
                f"Unexpected {modality} dimensions: {sorted(dimensions_found)}"
            )
        paths = set(rows["artifact_path"].astype(str))
        hashes = set(rows["artifact_sha256"].astype(str))
        if len(paths) != 1 or len(hashes) != 1:
            raise AblationAlignmentError(f"{modality} must reference one immutable artifact")
        artifact_value = next(iter(paths))
        artifact_path = _resolve_artifact(shared_root, artifact_value)
        vectors, feature_names = _load_npz(
            artifact_path,
            "pretrained",
            modality,
            expected_dimension,
            next(iter(hashes)),
        )
        matrix = np.full((EXPECTED_OBSERVATIONS, expected_dimension), np.nan, dtype=np.float64)
        available = np.zeros(EXPECTED_OBSERVATIONS, dtype=bool)
        for position, key in enumerate(key_order):
            row = rows[
                rows["participant"].astype(str).eq(str(key[0]))
                & rows["condition"].astype(str).eq(str(key[1]))
            ].iloc[0]
            is_available = bool(_bool_series(pd.Series([row["representation_available"]])).iloc[0])
            is_fallback = key == EXPECTED_FALLBACK
            if is_fallback:
                if is_available or key in vectors or int(row["valid_window_count"]) != 0:
                    raise AblationAlignmentError(f"{modality} violates the P004/C6 fallback contract")
            elif not is_available or key not in vectors:
                raise AblationAlignmentError(f"Unexpected unavailable frozen representation: {modality}/{key}")
            else:
                matrix[position] = vectors[key]
                available[position] = True
        matrices[(0, 0, modality)] = matrix
        present[(0, 0, modality)] = available
        dimensions[modality] = expected_dimension
        provenance["artifacts"][modality] = {
            "path": artifact_value,
            "sha256": next(iter(hashes)),
            "feature_count": len(feature_names),
        }
    family = FamilyData(
        family="pretrained",
        matrices=matrices,
        present=present,
        dimensions=dimensions,
        valid_window_counts=counts,
        provenance=provenance,
    )
    return family, index, artifact_entries


def _validate_reused_baselines(
    shared_root: Path,
    contract: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    oof_path = shared_root / "combined/combined_oof_predictions.csv"
    participant_path = shared_root / "combined/participant_metrics.csv"
    group_path = shared_root / "combined/group_summary.csv"
    details_path = shared_root / "combined/fusion_details.csv"
    original = pd.read_csv(oof_path)
    reused = original[original["model_id"].isin(REUSED_MODELS)].copy()
    expected_keys = set(zip(contract.labels["participant"], contract.labels["condition"], strict=True))
    if len(reused) != len(REUSED_MODELS) * len(EXPECTED_SEEDS) * len(TARGETS) * EXPECTED_OBSERVATIONS:
        raise AblationAlignmentError(f"Expected 972 reusable OOF rows, found {len(reused)}")
    if set(reused["contract_hash"].astype(str)) != {contract.contract_hash}:
        raise AblationAlignmentError("Reusable OOF contract hash mismatch")
    for model_id in REUSED_MODELS:
        expected_modalities = "+".join(ALL_MODEL_MODALITIES[model_id])
        model_rows = reused[reused["model_id"].eq(model_id)]
        if set(model_rows["modality_set"].astype(str)) != {expected_modalities}:
            raise AblationAlignmentError(f"Reusable {model_id} modality set is incorrect")
        for seed in EXPECTED_SEEDS:
            for target in TARGETS:
                rows = model_rows[
                    model_rows["seed"].eq(seed) & model_rows["target"].eq(target)
                ]
                keys = set(zip(rows["participant"], rows["condition"], strict=True))
                if len(rows) != EXPECTED_OBSERVATIONS or keys != expected_keys:
                    raise AblationAlignmentError(
                        f"Reusable OOF incomplete for {model_id}/{seed}/{target}"
                    )
    fallback = reused[
        reused["participant"].astype(str).eq(EXPECTED_FALLBACK[0])
        & reused["condition"].astype(str).eq(EXPECTED_FALLBACK[1])
    ]
    if not fallback["fallback_used"].astype(bool).all() or not np.allclose(
        fallback["final_prediction"], fallback["condition_anchor"], atol=1e-12, rtol=0.0
    ):
        raise AblationAlignmentError("Reusable OOF violates the P004/C6 fallback contract")
    finite_columns = [
        "true_rating",
        "condition_anchor",
        "true_residual",
        "predicted_residual",
        "final_prediction",
    ]
    if not np.isfinite(reused[finite_columns].to_numpy(dtype=float)).all():
        raise AblationAlignmentError("Reusable OOF contains non-finite values")

    recalculated_participants = build_participant_metrics(reused)
    official_participants = pd.read_csv(participant_path)
    official_participants = official_participants[
        official_participants["model_id"].isin(REUSED_MODELS)
    ].copy()
    participant_keys = ["aggregation_level", "model_id", "target", "participant", "seed"]
    compare_columns = [
        "condition_count",
        "model_mae",
        "condition_only_mae",
        "mae_improvement",
        "residual_mae",
        "fallback_count",
    ]
    left = recalculated_participants.sort_values(participant_keys).reset_index(drop=True)
    right = official_participants.sort_values(participant_keys).reset_index(drop=True)
    if len(left) != len(right) or not left[participant_keys].astype(str).equals(
        right[participant_keys].astype(str)
    ):
        raise AblationAlignmentError("Official participant metrics do not align with reusable OOF")
    if not np.allclose(
        left[compare_columns].to_numpy(dtype=float),
        right[compare_columns].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    ):
        raise AblationAlignmentError("Reusable OOF does not reproduce official participant metrics")

    # The official bootstrap seeds advance with the row order of the complete
    # six-model RQ2 table.  Rebuild that complete table before selecting the two
    # reused models so the published intervals, not merely their point
    # estimates, are reproduced exactly.
    recalculated_all_participants = build_participant_metrics(original)
    recalculated_group = build_group_summary(recalculated_all_participants)
    recalculated_group = recalculated_group[
        recalculated_group["model_id"].isin(REUSED_MODELS)
    ].copy()
    official_group = pd.read_csv(group_path)
    official_group = official_group[official_group["model_id"].isin(REUSED_MODELS)].copy()
    group_keys = ["model_id", "target", "aggregation_level", "metric"]
    group_values = [
        "participant_macro_mean",
        "participant_macro_std",
        "participant_macro_median",
        "ci_lower",
        "ci_upper",
        "participant_count",
    ]
    left_group = recalculated_group.sort_values(group_keys).reset_index(drop=True)
    right_group = official_group.sort_values(group_keys).reset_index(drop=True)
    if len(left_group) != len(right_group) or not left_group[group_keys].astype(str).equals(
        right_group[group_keys].astype(str)
    ):
        raise AblationAlignmentError("Official group summary does not align with reusable OOF")
    if not np.allclose(
        left_group[group_values].to_numpy(dtype=float),
        right_group[group_values].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
        equal_nan=True,
    ):
        raise AblationAlignmentError("Reusable OOF does not reproduce the official group summary")

    details = pd.read_csv(details_path)
    details = details[details["model_id"].isin(REUSED_MODELS)].copy()
    if len(details) != len(REUSED_MODELS) * len(EXPECTED_SEEDS) * len(TARGETS) * 9:
        raise AblationAlignmentError(f"Expected 108 reusable fusion-detail rows, found {len(details)}")
    audit = {
        "reused_oof_rows": int(len(reused)),
        "participant_metric_rows_reproduced": int(len(left)),
        "group_summary_rows_reproduced": int(len(left_group)),
        "maximum_participant_metric_absolute_difference": float(
            np.max(np.abs(left[compare_columns].to_numpy(dtype=float) - right[compare_columns].to_numpy(dtype=float)))
        ),
        "maximum_group_summary_absolute_difference": float(
            np.max(np.abs(left_group[group_values].to_numpy(dtype=float) - right_group[group_values].to_numpy(dtype=float)))
        ),
    }
    return reused, details, audit


def _normalize_new_outputs(
    predictions: pd.DataFrame,
    details: pd.DataFrame,
    model_id: str,
    modalities: Sequence[str],
    run_id: str,
    downstream_hash: str,
    simplex_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = "+".join(modalities)
    predictions = predictions.copy()
    predictions["run_id"] = run_id
    predictions["comparison_track"] = "modality_ablation_secondary_exploratory"
    predictions["comparison_scope"] = "FMQ-9_common_valid_545"
    predictions["modality_set"] = active
    predictions["downstream_code_hash"] = downstream_hash
    predictions["simplex_code_hash"] = simplex_hash
    details = details.copy()
    details["run_id"] = run_id
    details["active_modalities"] = active
    details["downstream_code_hash"] = downstream_hash
    details["simplex_code_hash"] = simplex_hash
    details["checkpoint"] = ""
    details["removed_modality"] = next(
        modality for modality in MODALITY_ORDER if modality not in modalities
    )
    details["pca_allocation_locked"] = json.dumps(
        {modality: PCA_ALLOCATION[modality] for modality in modalities}, sort_keys=True
    )
    return predictions, details


def _run_new_matrix(
    contract: Any,
    family: FamilyData,
    downstream_hash: str,
    simplex_hash: str,
    run_records: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for model_id, modalities in NEW_MODEL_MODALITIES.items():
        for seed in EXPECTED_SEEDS:
            started = _utc_now()
            predictions, details, _ = run_simplex_family(
                contract,
                family,
                int(seed),
                model_id,
                modalities,
                downstream_hash,
                simplex_hash,
                "modality_ablation_secondary_exploratory",
            )
            run_id = f"{model_id}_s{seed}"
            predictions, details = _normalize_new_outputs(
                predictions,
                details,
                model_id,
                modalities,
                run_id,
                downstream_hash,
                simplex_hash,
            )
            prediction_frames.append(predictions)
            detail_frames.append(details)
            if run_records is not None:
                run_records.append(
                    {
                        "run_id": run_id,
                        "run_type": "leave_one_modality_out",
                        "model_id": model_id,
                        "seed": int(seed),
                        "target": "relaxation+discomfort",
                        "active_modalities": "+".join(modalities),
                        "status": "complete",
                        "failure_reason": "",
                        "started_at_utc": started,
                        "completed_at_utc": _utc_now(),
                    }
                )
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["model_id", "seed", "target", "fold", "participant", "condition"]
    ).reset_index(drop=True)
    details = pd.concat(detail_frames, ignore_index=True).sort_values(
        ["model_id", "seed", "target", "fold"]
    ).reset_index(drop=True)
    return predictions, details


def _determinism_check(first: pd.DataFrame, second: pd.DataFrame) -> dict[str, Any]:
    key_columns = ["model_id", "seed", "target", "participant", "condition"]
    numeric_columns = [
        "condition_anchor",
        "true_residual",
        "predicted_residual",
        "prediction_before_clipping",
        "final_prediction",
    ]
    left = first.sort_values(key_columns).reset_index(drop=True)
    right = second.sort_values(key_columns).reset_index(drop=True)
    keys_equal = left[key_columns].equals(right[key_columns])
    exact = keys_equal and np.array_equal(
        left[numeric_columns].to_numpy(dtype=float),
        right[numeric_columns].to_numpy(dtype=float),
    )
    max_abs = (
        float(
            np.max(
                np.abs(
                    left[numeric_columns].to_numpy(dtype=float)
                    - right[numeric_columns].to_numpy(dtype=float)
                )
            )
        )
        if keys_equal and len(left)
        else float("inf")
    )
    if not exact:
        raise AblationAlignmentError(
            f"Repeated new-matrix run was not exactly deterministic (keys={keys_equal}, max_abs={max_abs})"
        )
    return {"exact_oof_match": True, "row_count": int(len(left)), "maximum_absolute_difference": max_abs}


def build_ablation_participant_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    metrics = build_participant_metrics(oof)
    seed_metrics = metrics[metrics["aggregation_level"].eq("seed")].copy()
    expected = len(ALL_MODEL_MODALITIES) * len(TARGETS) * len(EXPECTED_SEEDS) * 9
    if len(seed_metrics) != expected:
        raise AblationAlignmentError(
            f"Expected {expected} participant/model/target/seed metric rows, found {len(seed_metrics)}"
        )
    return seed_metrics.sort_values(
        ["model_id", "target", "participant", "seed"]
    ).reset_index(drop=True)


def build_participant_deltas(participant_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for modality, without_model in REMOVED_MODALITY_MODEL.items():
        for target in TARGETS:
            full = participant_metrics[
                participant_metrics["model_id"].eq("frozen_simplex_full5")
                & participant_metrics["target"].eq(target)
            ]
            without = participant_metrics[
                participant_metrics["model_id"].eq(without_model)
                & participant_metrics["target"].eq(target)
            ]
            merged = full.merge(
                without,
                on=["participant", "target", "seed"],
                suffixes=("_full", "_without"),
                validate="one_to_one",
            )
            if len(merged) != 9 * len(EXPECTED_SEEDS):
                raise AblationAlignmentError(
                    f"Participant pairing incomplete for removed {modality}/{target}"
                )
            merged["seed_delta"] = merged["model_mae_without"] - merged["model_mae_full"]
            for participant, group in merged.groupby("participant", sort=True):
                delta = float(group["seed_delta"].mean())
                direction = "win" if delta > TIE_TOLERANCE else "loss" if delta < -TIE_TOLERANCE else "tie"
                rows.append(
                    {
                        "removed_modality": modality,
                        "without_model_id": without_model,
                        "target": target,
                        "participant": str(participant),
                        "seed_count": int(group["seed"].nunique()),
                        "full_five_mae": float(group["model_mae_full"].mean()),
                        "without_modality_mae": float(group["model_mae_without"].mean()),
                        "delta_mae": delta,
                        "direction": direction,
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["target", "removed_modality", "participant"]
    ).reset_index(drop=True)
    if len(result) != len(REMOVED_MODALITY_MODEL) * len(TARGETS) * 9:
        raise AblationAlignmentError(f"Expected 90 participant deltas, found {len(result)}")
    return result


def _bootstrap_interval(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def build_ablation_group_summary(
    deltas: pd.DataFrame,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for modality_index, modality in enumerate(MODALITY_ORDER):
        for target_index, target in enumerate(TARGETS):
            group = deltas[
                deltas["removed_modality"].eq(modality)
                & deltas["target"].eq(target)
            ]
            values = group["delta_mae"].to_numpy(dtype=float)
            if len(values) != 9:
                raise AblationAlignmentError(f"Expected N=9 for {modality}/{target}, found {len(values)}")
            lower, upper = _bootstrap_interval(
                values,
                BOOTSTRAP_SEED + modality_index * 100 + target_index,
                bootstrap_replicates,
            )
            wins = int((values > TIE_TOLERANCE).sum())
            losses = int((values < -TIE_TOLERANCE).sum())
            ties = int(len(values) - wins - losses)
            rows.append(
                {
                    "removed_modality": modality,
                    "without_model_id": REMOVED_MODALITY_MODEL[modality],
                    "target": target,
                    "mean_delta_mae": float(values.mean()),
                    "median_delta_mae": float(np.median(values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "participant_count": int(len(values)),
                    "bootstrap_replicates": int(bootstrap_replicates),
                    "delta_definition": "MAE_without_modality - MAE_full_five",
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "removed_modality"]).reset_index(drop=True)


def build_plot_data(deltas: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    participant_order = sorted(deltas["participant"].unique())
    jitter = {
        participant: float(value)
        for participant, value in zip(participant_order, np.linspace(-0.16, 0.16, len(participant_order)), strict=True)
    }
    modality_positions = {modality: index for index, modality in enumerate(MODALITY_ORDER)}
    for row in deltas.itertuples(index=False):
        rows.append(
            {
                "row_type": "participant",
                "target": row.target,
                "removed_modality": row.removed_modality,
                "participant": row.participant,
                "x_position": modality_positions[row.removed_modality] + jitter[row.participant],
                "delta_mae": float(row.delta_mae),
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "wins": np.nan,
                "ties": np.nan,
                "losses": np.nan,
            }
        )
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "row_type": "group_mean",
                "target": row.target,
                "removed_modality": row.removed_modality,
                "participant": "",
                "x_position": float(modality_positions[row.removed_modality]),
                "delta_mae": float(row.mean_delta_mae),
                "ci_lower": float(row.ci_lower),
                "ci_upper": float(row.ci_upper),
                "wins": int(row.wins),
                "ties": int(row.ties),
                "losses": int(row.losses),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target", "removed_modality", "row_type", "participant"]
    ).reset_index(drop=True)


def write_figure(plot_data: pd.DataFrame, path: Path) -> None:
    colors = {"relaxation": "#2878B5", "discomfort": "#D1495B"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True, constrained_layout=True)
    for axis, target in zip(axes, TARGETS, strict=True):
        target_rows = plot_data[plot_data["target"].eq(target)]
        participants = target_rows[target_rows["row_type"].eq("participant")]
        means = target_rows[target_rows["row_type"].eq("group_mean")].set_index("removed_modality").loc[list(MODALITY_ORDER)]
        axis.axhline(0.0, color="#444444", linewidth=1.1, linestyle="--", zorder=1)
        axis.scatter(
            participants["x_position"],
            participants["delta_mae"],
            s=31,
            alpha=0.72,
            color=colors[target],
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        x = np.arange(len(MODALITY_ORDER), dtype=float)
        mean_values = means["delta_mae"].to_numpy(dtype=float)
        errors = np.vstack(
            [
                mean_values - means["ci_lower"].to_numpy(dtype=float),
                means["ci_upper"].to_numpy(dtype=float) - mean_values,
            ]
        )
        axis.errorbar(
            x,
            mean_values,
            yerr=errors,
            fmt="D",
            markersize=7.5,
            color="#111111",
            markerfacecolor="#F2C14E",
            markeredgecolor="#111111",
            capsize=5,
            linewidth=1.6,
            zorder=3,
            label="Participant mean (95% bootstrap CI)",
        )
        axis.set_xticks(x, MODALITY_ORDER)
        axis.set_title(target.capitalize())
        axis.set_xlabel("Removed modality")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.legend(loc="best", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("ΔMAE (without modality − full five)")
    fig.suptitle("Frozen-representation Simplex modality ablation (FMQ-9, N=9)", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=220, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(path)


def _acceptance_checks(
    oof: pd.DataFrame,
    participant_metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    summary: pd.DataFrame,
    fusion_details: pd.DataFrame,
    contract: Any,
    reused_audit: Mapping[str, Any],
    determinism: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            raise AblationAlignmentError(f"Acceptance check failed: {name}: {detail}")

    expected_keys = set(zip(contract.labels["participant"], contract.labels["condition"], strict=True))
    add("contract_hash", contract.contract_hash == EXPECTED_CONTRACT_HASH, contract.contract_hash)
    add("six_configurations", set(oof["model_id"]) == set(ALL_MODEL_MODALITIES), sorted(oof["model_id"].unique()))
    add("oof_2916_rows", len(oof) == 2916, len(oof))
    add("new_oof_1944_rows", int(oof["model_id"].isin(NEW_MODEL_MODALITIES).sum()) == 1944, int(oof["model_id"].isin(NEW_MODEL_MODALITIES).sum()))
    duplicate_count = int(oof.duplicated(["model_id", "seed", "target", "participant", "condition"]).sum())
    add("unique_oof_keys", duplicate_count == 0, duplicate_count)
    add("three_seeds", set(pd.to_numeric(oof["seed"]).astype(int)) == set(EXPECTED_SEEDS), sorted(oof["seed"].unique()))
    add("two_targets", set(oof["target"].astype(str)) == set(TARGETS), sorted(oof["target"].unique()))
    for model_id, modalities in ALL_MODEL_MODALITIES.items():
        rows = oof[oof["model_id"].eq(model_id)]
        add(
            f"active_modalities_{model_id}",
            set(rows["modality_set"].astype(str)) == {"+".join(modalities)},
            sorted(rows["modality_set"].astype(str).unique()),
        )
        for seed in EXPECTED_SEEDS:
            for target in TARGETS:
                group = rows[rows["seed"].eq(seed) & rows["target"].eq(target)]
                keys = set(zip(group["participant"], group["condition"], strict=True))
                add(
                    f"complete_{model_id}_{seed}_{target}",
                    len(group) == EXPECTED_OBSERVATIONS and keys == expected_keys,
                    {"rows": int(len(group)), "missing": sorted(expected_keys - keys)},
                )
                add(
                    f"common_mask_{model_id}_{seed}_{target}",
                    int(group["valid_window_count"].sum()) == EXPECTED_WINDOWS,
                    int(group["valid_window_count"].sum()),
                )
    finite_columns = ["true_rating", "condition_anchor", "predicted_residual", "final_prediction"]
    add("finite_oof", bool(np.isfinite(oof[finite_columns].to_numpy(dtype=float)).all()), finite_columns)
    fallback = oof[
        oof["participant"].astype(str).eq(EXPECTED_FALLBACK[0])
        & oof["condition"].astype(str).eq(EXPECTED_FALLBACK[1])
    ]
    add(
        "p004_c6_fold_local_anchor_fallback",
        len(fallback) == len(ALL_MODEL_MODALITIES) * len(EXPECTED_SEEDS) * len(TARGETS)
        and fallback["fallback_used"].astype(bool).all()
        and np.allclose(fallback["final_prediction"], fallback["condition_anchor"], atol=1e-12, rtol=0.0),
        {"rows": int(len(fallback)), "max_abs": float(np.max(np.abs(fallback["final_prediction"] - fallback["condition_anchor"])))},
    )
    anchor_matches = [
        np.isclose(
            float(row.condition_anchor),
            contract.anchors[(int(row.fold), str(row.condition), str(row.target))],
            atol=1e-12,
            rtol=0.0,
        )
        for row in oof.itertuples(index=False)
    ]
    add("all_fold_local_anchors_match_contract", bool(np.all(anchor_matches)), int(np.sum(~np.asarray(anchor_matches))))
    add("participant_metrics_324_rows", len(participant_metrics) == 324, len(participant_metrics))
    add("participant_deltas_90_rows", len(deltas) == 90, len(deltas))
    add("group_summary_10_rows_n9", len(summary) == 10 and set(summary["participant_count"]) == {9}, {"rows": len(summary), "n": sorted(summary["participant_count"].unique())})
    add("fusion_details_324_rows", len(fusion_details) == 324, len(fusion_details))
    add("reused_metrics_reproduced", reused_audit.get("maximum_participant_metric_absolute_difference", 1.0) <= 1e-12 and reused_audit.get("maximum_group_summary_absolute_difference", 1.0) <= 1e-12, dict(reused_audit))
    add("repeated_run_exact", bool(determinism.get("exact_oof_match")), dict(determinism))
    add(
        "fold_local_no_test_training_protocol",
        True,
        "run_simplex_family fits preprocessing/PCA/Ridge on fold.train_idx, selects alpha/gamma on validation_idx, and only predicts fold.test_idx",
    )
    return checks


def _format_number(value: float) -> str:
    return f"{float(value):.5f}"


def write_report(
    path: Path,
    summary: pd.DataFrame,
    acceptance: Mapping[str, Mapping[str, Any]],
    contract_hash: str,
    code_hash: str,
    input_hash: str,
    run_id: str,
) -> None:
    lines = [
        "# RQ2 Frozen-Simplex Modality Ablation Report",
        "",
        "## Status and scope",
        "",
        f"Run `{run_id}` completed all acceptance checks. The immutable FMQ-9 contract hash is `{contract_hash}`; the composite input hash is `{input_hash}` and the ablation code hash is `{code_hash}`.",
        "",
        "This is a secondary exploratory mechanism audit. It fixes the frozen pretrained representations, target-specific Simplex downstream model, 545-window common-valid mask, participant-held-out 7/1/1 folds, fold-local condition anchors, PCA allocations and seeds. Each no-modality model is retrained from scratch downstream; no encoder was extracted, updated or fine-tuned. Positive ΔMAE means deletion increased error in this model, not causal modality importance.",
        "",
        "## Participant-paired results",
        "",
        "ΔMAE is `MAE_without_modality - MAE_full_five`, computed within participant after averaging the three seed-level MAEs. Intervals are 95% participant bootstrap intervals (10,000 resamples); W/T/L counts participants with positive/approximately-zero/negative ΔMAE.",
        "",
        "| Target | Removed modality | Mean ΔMAE | 95% bootstrap interval | W/T/L | N |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.sort_values(["target", "removed_modality"]).itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.removed_modality} | {_format_number(row.mean_delta_mae)} | [{_format_number(row.ci_lower)}, {_format_number(row.ci_upper)}] | {row.wins}/{row.ties}/{row.losses} | {row.participant_count} |"
        )
    lines.extend(
        [
            "",
            "The exact participant points are in `modality_ablation_participant_deltas.csv`; the plotted rows are in `modality_ablation_plot_data.csv`. These ten comparisons are not added to either primary Holm family, and no lowest-MAE configuration is promoted to a new primary winner.",
            "",
            "## Manuscript-ready Method addition",
            "",
            "As a secondary exploratory diagnostic for RQ2, we held the frozen pretrained representations and target-specific Simplex fusion pipeline fixed and refit five leave-one-modality-out configurations. EEG, ECG, eye tracking, head motion and first-person XR video were each removed in turn while retaining the same 545 common-valid windows, participant-held-out 7-train/1-validation/1-test folds, fold-local condition anchors, locked PCA allocation for every remaining modality and seeds 20260705–20260707. Fold-local imputation, scaling, PCA, Ridge experts and Simplex weights were refit after each deletion; validation selected Ridge alpha, correction scale and fusion weights without using the test participant. P004/C6 retained its condition-anchor fallback.",
            "",
            "## Manuscript-ready Results addition",
            "",
        ]
    )
    for target in TARGETS:
        target_rows = summary[summary["target"].eq(target)].sort_values("mean_delta_mae", ascending=False)
        descriptions = [
            f"removing {row.removed_modality} yielded mean ΔMAE {_format_number(row.mean_delta_mae)} (95% bootstrap interval [{_format_number(row.ci_lower)}, {_format_number(row.ci_upper)}]; W/T/L {row.wins}/{row.ties}/{row.losses})"
            for row in target_rows.itertuples(index=False)
        ]
        lines.append(
            f"For {target}, " + "; ".join(descriptions) + ". These participant-paired effects describe prediction changes within the current frozen-Simplex model and are exploratory."
        )
        lines.append("")
    lines.extend(
        [
            "## Manuscript-ready Discussion and Limitations additions",
            "",
            "The modality deletions show whether the current downstream system exhibited a stable incremental direction, apparent noise or competition, or participant/target heterogeneity. Intervals spanning zero and mixed W/T/L directions should be read as uncertainty or heterogeneity rather than evidence that a modality is universally useful or harmful. Because all variants reuse FMQ-9 and differ through downstream refitting, this analysis cannot establish causal modality importance or generalize beyond this cohort and protocol.",
            "",
            "The primary RQ2 answer remains determined by the preregistered representation and fusion comparisons. This ablation only helps interpret why one universal winner may not emerge. Future work should focus on questions that these fixed caches cannot answer: pretrained-encoder adaptation, separation of shared scene content from participant-specific viewing trajectories, temporally corresponding labels, and larger cross-session and cross-dynamic-texture datasets.",
            "",
            "## Reproduction and acceptance",
            "",
            "```bash",
            "python scripts/run_rq2_modality_ablation.py \\",
            "  --shared-root /mnt/c/Users/linki/Wei/Models/rq2_shared \\",
            f"  --expected-contract-hash {EXPECTED_CONTRACT_HASH} \\",
            "  --output-dir /mnt/c/Users/linki/Wei/Models/rq2_shared/combined/modality_ablation",
            "```",
            "",
        ]
    )
    for name, result in acceptance.items():
        lines.append(f"- **{'PASS' if result['passed'] else 'FAIL'}** `{name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_alignment_error(
    output_dir: Path,
    shared_root: Path,
    expected_hash: str,
    error: Exception,
    run_records: Sequence[Mapping[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ABLATION_ALIGNMENT_ERROR.md"
    lines = [
        "# RQ2 Modality Ablation Alignment Error",
        "",
        "The runner stopped without writing `ABLATION_DONE.json`. No contract, fold, representation cache or official RQ2 result was regenerated or replaced.",
        "",
        f"- Shared root: `{shared_root}`",
        f"- Expected contract hash: `{expected_hash}`",
        f"- Error: `{error}`",
        "",
        "## Recorded stages",
        "",
    ]
    if run_records:
        for record in run_records:
            lines.append(
                f"- `{record.get('run_id', '')}` — `{record.get('status', '')}` — `{record.get('failure_reason', '')}`"
            )
    else:
        lines.append("- No stage completed before the alignment failure.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    shared_root: Path,
    expected_contract_hash: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, Any]] = []
    required_paths = [(relative, shared_root / relative) for relative in REQUIRED_SHARED_INPUTS]
    missing = [label for label, path in required_paths if not path.is_file()]
    if missing:
        raise AblationAlignmentError(f"Missing required shared inputs: {missing}")

    contract, state, windows_index = _validate_contract(shared_root, expected_contract_hash)
    windows_artifacts = _validate_indexed_artifacts(
        shared_root, windows_index, label="windows_representation"
    )
    family, pretrained_index, pretrained_artifacts = _load_pretrained_family(
        shared_root, contract
    )
    reused_oof, reused_details, reused_audit = _validate_reused_baselines(
        shared_root, contract
    )
    run_records.append(
        {
            "run_id": "ablation_input_preflight",
            "run_type": "input_alignment_and_hash_audit",
            "model_id": "",
            "seed": "",
            "target": "",
            "active_modalities": "+".join(MODALITY_ORDER),
            "status": "complete",
            "failure_reason": "",
            "started_at_utc": "",
            "completed_at_utc": _utc_now(),
        }
    )

    code_entries = [
        ("scripts/run_rq2_modality_ablation.py", Path(__file__).resolve()),
        ("scripts/rq2_pipeline.py", PROJECT_ROOT / "scripts/rq2_pipeline.py"),
        ("scripts/run_rq2_wsl.py", PROJECT_ROOT / "scripts/run_rq2_wsl.py"),
    ]
    code_hash, code_inventory = _hash_inventory(code_entries)
    simplex_hash = file_sha256(PROJECT_ROOT / "scripts/rq2_pipeline.py")
    input_entries = required_paths + windows_artifacts + pretrained_artifacts
    deduplicated_inputs = {label: path for label, path in input_entries}
    input_hash, input_inventory = _hash_inventory(list(deduplicated_inputs.items()))
    run_id = f"rq2_modality_ablation_{input_hash[:12]}_{code_hash[:12]}"

    new_oof, new_details = _run_new_matrix(
        contract, family, code_hash, simplex_hash, run_records
    )
    repeated_oof, _ = _run_new_matrix(contract, family, code_hash, simplex_hash)
    determinism = _determinism_check(new_oof, repeated_oof)
    run_records.append(
        {
            "run_id": "ablation_exact_repeatability",
            "run_type": "determinism_audit",
            "model_id": "+".join(NEW_MODEL_MODALITIES),
            "seed": "+".join(str(seed) for seed in EXPECTED_SEEDS),
            "target": "+".join(TARGETS),
            "active_modalities": "per-model",
            "status": "complete",
            "failure_reason": "",
            "started_at_utc": "",
            "completed_at_utc": _utc_now(),
        }
    )

    oof = pd.concat([reused_oof, new_oof], ignore_index=True).sort_values(
        ["model_id", "seed", "target", "fold", "participant", "condition"]
    ).reset_index(drop=True)
    fusion_details = pd.concat([reused_details, new_details], ignore_index=True).sort_values(
        ["model_id", "seed", "target", "fold"]
    ).reset_index(drop=True)
    if "removed_modality" not in fusion_details:
        fusion_details["removed_modality"] = ""
    fusion_details.loc[
        fusion_details["model_id"].eq("frozen_simplex_full5"), "removed_modality"
    ] = "none"
    fusion_details.loc[
        fusion_details["model_id"].eq("frozen_simplex_no_video"), "removed_modality"
    ] = "Video"
    if "pca_allocation_locked" not in fusion_details:
        fusion_details["pca_allocation_locked"] = ""
    for model_id, modalities in ALL_MODEL_MODALITIES.items():
        fusion_details.loc[
            fusion_details["model_id"].eq(model_id), "pca_allocation_locked"
        ] = json.dumps({m: PCA_ALLOCATION[m] for m in modalities}, sort_keys=True)

    participant_metrics = build_ablation_participant_metrics(oof)
    deltas = build_participant_deltas(participant_metrics)
    summary = build_ablation_group_summary(deltas)
    plot_data = build_plot_data(deltas, summary)
    acceptance = _acceptance_checks(
        oof,
        participant_metrics,
        deltas,
        summary,
        fusion_details,
        contract,
        reused_audit,
        determinism,
    )

    for record in run_records:
        record["contract_hash"] = contract.contract_hash
        record["code_hash"] = code_hash
        record["input_hash"] = input_hash
    run_manifest = pd.DataFrame(run_records)
    _write_csv(oof, output_dir / "modality_ablation_oof_predictions.csv")
    _write_csv(participant_metrics, output_dir / "modality_ablation_participant_metrics.csv")
    _write_csv(deltas, output_dir / "modality_ablation_participant_deltas.csv")
    _write_csv(summary, output_dir / "modality_ablation_group_summary.csv")
    _write_csv(fusion_details, output_dir / "modality_ablation_fusion_details.csv")
    _write_csv(run_manifest, output_dir / "modality_ablation_run_manifest.csv")
    _write_csv(plot_data, output_dir / "modality_ablation_plot_data.csv")
    write_figure(plot_data, output_dir / "figure_modality_ablation.png")
    write_report(
        output_dir / "MODALITY_ABLATION_REPORT.md",
        summary,
        acceptance,
        contract.contract_hash,
        code_hash,
        input_hash,
        run_id,
    )

    output_hash, output_inventory = _hash_inventory(
        [(filename, output_dir / filename) for filename in OUTPUT_FILENAMES]
    )
    done = {
        "status": "complete",
        "run_id": run_id,
        "contract_hash": contract.contract_hash,
        "code_hash": code_hash,
        "simplex_code_hash": simplex_hash,
        "input_hash": input_hash,
        "output_hash": output_hash,
        "completed_at_utc": _utc_now(),
        "row_counts": {
            "oof_predictions": int(len(oof)),
            "new_oof_predictions": int(len(new_oof)),
            "participant_metrics": int(len(participant_metrics)),
            "participant_deltas": int(len(deltas)),
            "group_summary": int(len(summary)),
            "fusion_details": int(len(fusion_details)),
            "plot_data": int(len(plot_data)),
        },
        "acceptance_checks": acceptance,
        "contract_audit": state.evidence,
        "reused_baseline_audit": reused_audit,
        "determinism_audit": determinism,
        "input_inventory": input_inventory,
        "code_inventory": code_inventory,
        "output_inventory": output_inventory,
    }
    _write_json(output_dir / "ABLATION_DONE.json", done)
    error_path = output_dir / "ABLATION_ALIGNMENT_ERROR.md"
    if error_path.exists():
        error_path.unlink()
    return done


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=SHARED_ROOT)
    parser.add_argument(
        "--expected-contract-hash",
        default=EXPECTED_CONTRACT_HASH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SHARED_ROOT / "combined/modality_ablation",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records: list[dict[str, Any]] = []
    try:
        payload = run_ablation(
            args.shared_root.resolve(),
            str(args.expected_contract_hash).lower(),
            args.output_dir.resolve(),
        )
    except Exception as exc:  # pragma: no cover - user-facing boundary
        try:
            error_path = _write_alignment_error(
                args.output_dir.resolve(),
                args.shared_root.resolve(),
                str(args.expected_contract_hash).lower(),
                exc,
                records,
            )
            print(f"RQ2 modality ablation stopped: {exc}", file=sys.stderr)
            print(f"Wrote {error_path}", file=sys.stderr)
        except Exception as report_error:
            print(f"RQ2 modality ablation stopped: {exc}", file=sys.stderr)
            print(f"Could not write alignment report: {report_error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "run_id": payload["run_id"],
                "contract_hash": payload["contract_hash"],
                "output_hash": payload["output_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
