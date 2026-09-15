"""Run the fixed five-expert fusion identically over every embedding stage.

The scientific implementation is imported from
``run_relax_compression_fusion_v2``.  This wrapper only supplies stage-specific
cache hashes/preregistrations and an exact component cache: modality PCA is
independent by construction, so reducers for byte-identical modalities are
reused across stages while changed EEG/ECG reducers are fitted afresh.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch

import scripts.run_relax_compression_fusion_v2 as fusion_runner
import scripts.run_relax_foundation_probe as foundation_probe
from scripts.run_relax_foundation_probe import file_sha256
from src.data.relax_dataset import RelaxConditionEmbeddingDataset
from src.fusion.frozen_compression_v2 import (
    FIVE_MODALITIES,
    ModalityWindowPCABank,
    RankAllocation,
    WindowPCAReducer,
    hamilton_allocate,
)


STAGES = (
    "s0_legacy",
    "s1_reve_native_raw4",
    "s2_reve_native_linked2",
    "s3_reve_clean_linked2",
    "s4_neurorvq_clean_z",
    "s5_neurorvq_clean_native",
    "s6_neurorvq_eeg_ecg",
    "r3_reve_official_session",
    "ecg_only_neurorvq",
)
SEEDS = (20260705, 20260706, 20260707)
DEFAULT_ROOT = ROOT / "artifacts/relax/neurorvq_embedding_ladder_20260718_corrected"
DEFAULT_TEMPLATE = (
    ROOT
    / "artifacts/relax/foundation_compression_fusion_reinvestigation_20260718"
    / "preregistration/method_preregistration.json"
)
DEFAULT_LEGACY_COMPONENTS = (
    ROOT
    / "artifacts/relax/foundation_compression_fusion_reinvestigation_20260718"
    / "compression_cache/modality_banks"
)
DEFAULT_HISTORICAL_RUNS = (
    ROOT
    / "artifacts/relax/foundation_compression_fusion_reinvestigation_20260718"
    / "runs/modality_expert_simplex5/full"
)
DEFAULT_CONTRACT = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/contract"
)


def _install_scipy_19_spearman_compatibility() -> None:
    """Expose SciPy >=1.10's ``.statistic`` name on SciPy 1.9 results.

    This changes no correlation value.  It only bridges the namedtuple field
    rename from ``correlation`` to ``statistic`` used by the frozen evaluator.
    """

    from scipy.stats import pearsonr, spearmanr

    def compatible_safe_correlation(
        truth: np.ndarray,
        prediction: np.ndarray,
        method: str,
    ) -> float:
        if (
            len(truth) < 2
            or np.std(truth) <= 1e-15
            or np.std(prediction) <= 1e-15
        ):
            return float("nan")
        result = (
            spearmanr(truth, prediction)
            if method == "spearman"
            else pearsonr(truth, prediction)
        )
        statistic = getattr(result, "statistic", getattr(result, "correlation", result[0]))
        return float(statistic) if np.isfinite(statistic) else float("nan")

    foundation_probe._safe_correlation = compatible_safe_correlation


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class ComponentizedModalityBankCache:
    """Cache independent modality reducers by tensor, mask and fit cohort."""

    def __init__(
        self,
        *,
        root: Path,
        legacy_composite_dir: Path,
        base_block_hashes: Mapping[str, str],
        source_sha256: str,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_block_hashes = dict(base_block_hashes)
        self.source_sha256 = source_sha256
        # Keep the array reference with the digest so Python cannot recycle an
        # object id for a later stage and accidentally reuse the wrong hash.
        self._array_hash_cache: dict[tuple[int, str], tuple[np.ndarray, str]] = {}
        self._legacy_banks: dict[tuple[str, ...], ModalityWindowPCABank] = {}
        for path in sorted(legacy_composite_dir.glob("*.joblib")):
            bank = joblib.load(path)
            if not isinstance(bank, ModalityWindowPCABank) or bank.fit_indexes_ is None:
                continue
            key = tuple(sorted(str(value) for value in bank.fit_participants_))
            existing = self._legacy_banks.get(key)
            if existing is not None:
                left = existing.provenance()["reducers"]
                right = bank.provenance()["reducers"]
                if left != right:
                    raise ValueError(f"Legacy component banks disagree for fit participants {key}")
                continue
            self._legacy_banks[key] = bank
        if len(self._legacy_banks) != 63:
            raise ValueError(
                f"Expected 63 unique historical train/inner-train banks, got {len(self._legacy_banks)}"
            )

    def _cached_array_hash(self, values: np.ndarray, label: str) -> str:
        key = (id(values), label)
        cached = self._array_hash_cache.get(key)
        if cached is None or cached[0] is not values:
            cached = (values, _hash_array(values))
            self._array_hash_cache[key] = cached
        return cached[1]

    def _component_path(
        self,
        *,
        modality: str,
        values_hash: str,
        mask_hash: str,
        fit_participants: tuple[str, ...],
    ) -> Path:
        digest = sha256()
        digest.update(modality.encode())
        digest.update(values_hash.encode())
        digest.update(mask_hash.encode())
        digest.update("|".join(fit_participants).encode())
        digest.update(self.source_sha256.encode())
        return self.root / modality / f"reducer_{digest.hexdigest()[:32]}.joblib"

    def _load_or_fit_reducer(
        self,
        *,
        modality: str,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        train: np.ndarray,
        participant_ids: np.ndarray,
    ) -> WindowPCAReducer:
        values_hash = self._cached_array_hash(blocks[modality], f"{modality}:values")
        mask_hash = self._cached_array_hash(masks[modality], f"{modality}:mask")
        fit_participants = tuple(sorted(set(participant_ids[np.asarray(train, dtype=int)])))
        path = self._component_path(
            modality=modality,
            values_hash=values_hash,
            mask_hash=mask_hash,
            fit_participants=fit_participants,
        )
        if path.is_file():
            reducer = joblib.load(path)
            if not isinstance(reducer, WindowPCAReducer):
                raise TypeError(f"Unexpected reducer cache type: {path}")
            if reducer.fit_indexes_ is None or not np.array_equal(reducer.fit_indexes_, train):
                raise ValueError(f"Reducer cache fit indexes changed: {path}")
            return reducer

        reducer: WindowPCAReducer
        legacy = self._legacy_banks.get(fit_participants)
        if values_hash == self.base_block_hashes[modality] and legacy is not None:
            if legacy.fit_indexes_ is None or not np.array_equal(legacy.fit_indexes_, train):
                raise ValueError(f"Historical reducer indexes disagree for {fit_participants}")
            reducer = deepcopy(legacy.reducers_[modality])
            origin = "historical_exact_reuse"
            elapsed = 0.0
        else:
            started = time.perf_counter()
            reducer = WindowPCAReducer(
                modality,
                max_components_per_modality := 12,
                clip_to_rank=True,
            ).fit(
                blocks[modality],
                masks[modality],
                train,
                participant_ids=participant_ids,
            )
            if max_components_per_modality != 12:
                raise AssertionError("Formal component cap changed")
            elapsed = time.perf_counter() - started
            origin = "fresh_stage_fit"
            print(
                f"PCA reducer fitted: {modality} participants={','.join(fit_participants)} "
                f"dim={blocks[modality].shape[-1]} time={elapsed:.2f}s",
                flush=True,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(reducer, path)
        _write_json(
            path.with_suffix(".json"),
            {
                "modality": modality,
                "values_sha256": values_hash,
                "mask_sha256": mask_hash,
                "fit_participants": fit_participants,
                "fit_indexes": np.asarray(train, dtype=int).tolist(),
                "origin": origin,
                "runtime_seconds": elapsed,
                "reducer_provenance": reducer.provenance(),
            },
        )
        return reducer

    def load_or_fit_bank(
        self,
        blocks: Mapping[str, np.ndarray],
        masks: Mapping[str, np.ndarray],
        train: np.ndarray,
        participant_ids: np.ndarray,
        cache_dir: Path | None,
    ) -> ModalityWindowPCABank:
        train = np.asarray(train, dtype=int)
        fit_participants = tuple(sorted(set(participant_ids[train])))
        stage_hashes = tuple(
            self._cached_array_hash(blocks[modality], f"{modality}:values")
            for modality in FIVE_MODALITIES
        )
        digest = sha256()
        digest.update("|".join(stage_hashes).encode())
        digest.update("|".join(fit_participants).encode())
        digest.update(self.source_sha256.encode())
        composite_path = (
            None
            if cache_dir is None
            else cache_dir / "componentized_banks" / f"bank_{digest.hexdigest()[:32]}.joblib"
        )
        if composite_path is not None and composite_path.is_file():
            bank = joblib.load(composite_path)
            if not isinstance(bank, ModalityWindowPCABank):
                raise TypeError(f"Unexpected componentized bank type: {composite_path}")
            if bank.fit_indexes_ is None or not np.array_equal(bank.fit_indexes_, train):
                raise ValueError(f"Componentized bank fit indexes changed: {composite_path}")
            return bank

        reducers = {
            modality: self._load_or_fit_reducer(
                modality=modality,
                blocks=blocks,
                masks=masks,
                train=train,
                participant_ids=participant_ids,
            )
            for modality in FIVE_MODALITIES
        }
        spectra = {}
        evidence = {}
        for modality, reducer in reducers.items():
            if reducer.pca.spectrum_ is None:
                raise RuntimeError(f"Reducer spectrum is absent for {modality}")
            spectra[modality] = reducer.pca.spectrum_
            evidence[modality] = dict(reducer.collection_evidence_)
        effective = {
            modality: spectra[modality].entropy_effective_rank
            for modality in FIVE_MODALITIES
        }
        numeric = {modality: spectra[modality].numeric_rank for modality in FIVE_MODALITIES}
        components = hamilton_allocate(
            effective,
            total=12,
            caps={modality: reducers[modality].n_components for modality in FIVE_MODALITIES},
            minimum=1,
        )
        bank = ModalityWindowPCABank(FIVE_MODALITIES, max_components_per_modality=12)
        bank.reducers_ = reducers
        bank.rank_allocation_ = RankAllocation(
            components=components,
            effective_ranks=effective,
            numeric_ranks=numeric,
            spectra=spectra,
            collection_evidence=evidence,
        )
        bank.fit_indexes_ = train.copy()
        bank.fit_participants_ = list(fit_participants)
        bank.allocation_history_ = {}
        bank.allocation_for(FIVE_MODALITIES, total_components=12, variant_name="full_five")

        legacy = self._legacy_banks.get(fit_participants)
        if all(
            stage_hashes[index] == self.base_block_hashes[modality]
            for index, modality in enumerate(FIVE_MODALITIES)
        ):
            if legacy is None:
                raise ValueError(f"Base-stage legacy bank is absent for {fit_participants}")
            current = bank.provenance()
            expected = legacy.provenance()
            for key in ("fit_indexes", "fit_participants", "rank_allocation", "reducers"):
                if current[key] != expected[key]:
                    raise ValueError(f"Componentized base bank differs from historical bank in {key}")

        if composite_path is not None:
            composite_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bank, composite_path)
        return bank


def _base_block_hashes(
    cache_path: Path,
    *,
    participants: Sequence[str],
    mask_manifest: Path,
) -> dict[str, str]:
    dataset = RelaxConditionEmbeddingDataset(
        cache_path,
        modalities=FIVE_MODALITIES,
        participants=participants,
        mask_manifest=mask_manifest,
        strict=True,
    )
    return {
        modality: _hash_array(dataset.embeddings[modality].numpy().astype(np.float64, copy=False))
        for modality in FIVE_MODALITIES
    }


def prepare_preregistrations(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if tuple(manifest["stages"]) != STAGES:
        raise ValueError(f"Unexpected stage order: {tuple(manifest['stages'])}")
    template = json.loads(args.preregistration_template.read_text(encoding="utf-8"))
    protocol_reference = args.root / "preregistration/corrected_run_declaration.json"
    protocol_hash = file_sha256(protocol_reference)
    preregistrations: dict[str, Any] = {}
    for stage in STAGES:
        stage_manifest = manifest["stages"][stage]
        cache = Path(stage_manifest["cache_path"])
        observed_sha = file_sha256(cache)
        if observed_sha != stage_manifest["cache_sha256"]:
            raise ValueError(f"Stage cache hash changed for {stage}")
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        dimensions = {
            modality: int(torch.as_tensor(payload["embeddings"][modality]).shape[-1])
            for modality in FIVE_MODALITIES
        }
        path = args.root / "fusion/preregistration" / f"{stage}.json"
        if path.is_file():
            prereg = json.loads(path.read_text(encoding="utf-8"))
            expected_stage = prereg.get("embedding_ladder_stage", {})
            if (
                expected_stage.get("stage") != stage
                or expected_stage.get("cache_sha256") != observed_sha
                or prereg.get("input_contract", {}).get("embedding_dimensions") != dimensions
            ):
                raise ValueError(f"Frozen stage preregistration no longer matches {stage}: {path}")
        else:
            prereg = deepcopy(template)
            prereg["candidate_outcome_scan"] = {
                "found": [],
                "status": "none_found_in_corrected_root",
            }
            prereg["chronology"]["embedding_ladder_stage_preregistration_created_local"] = (
                pd.Timestamp.now(tz="Europe/Berlin").isoformat()
            )
            prereg["input_contract"].update(
                {
                    "embedding_cache_sha256": observed_sha,
                    "labels_sha256": file_sha256(args.labels),
                    "windows_sha256": file_sha256(args.windows),
                    "split_manifest_sha256": file_sha256(args.split_manifest),
                    "common_mask_sha256": file_sha256(args.mask_manifest),
                    "cohorts_sha256": file_sha256(args.cohorts),
                    "embedding_dimensions": dimensions,
                    "raw_coordinate_total": int(sum(dimensions.values())),
                }
            )
            prereg["embedding_ladder_stage"] = {
                "stage": stage,
                "description": stage_manifest["description"],
                "cache_path": str(cache.resolve()),
                "cache_sha256": observed_sha,
                "corrected_run_declaration": str(protocol_reference.resolve()),
                "corrected_run_declaration_sha256": protocol_hash,
                "comparison_rules": "See fixed corrected run declaration and its referenced base protocol/amendment.",
            }
            prereg["execution_scope"] = {
                "authorized_method": "modality_expert_simplex5",
                "authorized_variant": "full",
                "authorized_seeds": list(SEEDS),
                "same_fusion_for_all_stages": True,
            }
            _write_json(path, prereg)
        preregistrations[stage] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "cache_path": str(cache.resolve()),
            "cache_sha256": observed_sha,
            "embedding_dimensions": dimensions,
        }
    result = {
        "schema_version": "relax_embedding_ladder_fusion_preparation_v1",
        "created_at_local": pd.Timestamp.now(tz="Europe/Berlin").isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "corrected_run_declaration": str(protocol_reference.resolve()),
        "corrected_run_declaration_sha256": protocol_hash,
        "method": "modality_expert_simplex5",
        "variant": "full",
        "seeds": list(SEEDS),
        "stages": preregistrations,
        "fusion_source_sha256": file_sha256(ROOT / "scripts/run_relax_compression_fusion_v2.py"),
        "compression_source_sha256": file_sha256(ROOT / "src/fusion/frozen_compression_v2.py"),
        "wrapper_source_sha256": file_sha256(__file__),
    }
    _write_json(args.root / "fusion/fusion_preparation_manifest.json", result)
    return result


def _validate_s0_reproduction(
    *,
    new_prediction_path: Path,
    historical_prediction_path: Path,
) -> dict[str, Any]:
    current = pd.read_csv(new_prediction_path).sort_values(["participant_id", "condition"]).reset_index(drop=True)
    historical = pd.read_csv(historical_prediction_path).sort_values(["participant_id", "condition"]).reset_index(drop=True)
    keys = ["participant_id", "condition"]
    if not current[keys].equals(historical[keys]):
        raise ValueError("S0 reproduction keys differ from the historical run")
    numeric_columns = sorted(
        column
        for column in set(current).intersection(historical)
        if pd.api.types.is_numeric_dtype(current[column])
        and pd.api.types.is_numeric_dtype(historical[column])
    )
    differences = {
        column: float(
            np.max(
                np.abs(
                    current[column].to_numpy(dtype=float)
                    - historical[column].to_numpy(dtype=float)
                )
            )
        )
        for column in numeric_columns
    }
    weight_columns = [column for column in numeric_columns if column.endswith("_weight")]
    selection_columns = [
        column
        for column in numeric_columns
        if column.endswith("_alpha") or column.endswith("_gamma")
    ]
    prediction_columns = [
        column
        for column in numeric_columns
        if column.endswith("_pred")
        or "_correction" in column
        or column.startswith("condition_only_")
    ]
    maxima = {
        "selection": max(differences[column] for column in selection_columns),
        "weights": max(differences[column] for column in weight_columns),
        "predictions_and_corrections": max(
            differences[column] for column in prediction_columns
        ),
    }
    tolerances = {
        "selection": 0.0,
        # SLSQP solutions are numerically equivalent but differ at about 2e-6
        # under the current SciPy/NumPy stack versus the historical run.
        "weights": 1e-5,
        "predictions_and_corrections": 1e-6,
    }
    violations = {
        key: value
        for key, value in maxima.items()
        if value > tolerances[key]
    }
    if violations:
        raise ValueError(
            f"S0 componentized reproduction exceeds numerical tolerances: {violations}"
        )
    return {
        "historical_path": str(historical_prediction_path.resolve()),
        "new_path": str(new_prediction_path.resolve()),
        "compared_numeric_columns": len(numeric_columns),
        "maximum_absolute_differences": maxima,
        "tolerances": tolerances,
        "interpretation": "Same PCA allocation and alpha/gamma selections; sub-1e-5 SLSQP weight drift and sub-1e-6 prediction drift are numerical compatibility effects.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _install_scipy_19_spearman_compatibility()
    preparation = prepare_preregistrations(args)
    if args.prepare_only:
        print(json.dumps(preparation, ensure_ascii=False, indent=2))
        return preparation

    participants = json.loads(args.cohorts.read_text(encoding="utf-8"))["eeg_eligible"]
    base_cache = Path(preparation["stages"]["s0_legacy"]["cache_path"])
    base_hashes = _base_block_hashes(
        base_cache,
        participants=participants,
        mask_manifest=args.mask_manifest,
    )
    component_cache = ComponentizedModalityBankCache(
        root=args.root / "fusion/component_cache/reducers",
        legacy_composite_dir=args.legacy_component_dir,
        base_block_hashes=base_hashes,
        source_sha256=file_sha256(ROOT / "src/fusion/frozen_compression_v2.py"),
    )
    original_loader = fusion_runner._load_or_fit_modality_bank
    fusion_runner._load_or_fit_modality_bank = component_cache.load_or_fit_bank

    records: list[dict[str, Any]] = []
    try:
        for stage in STAGES:
            stage_prereg = preparation["stages"][stage]
            cache = Path(stage_prereg["cache_path"])
            cache_sha = stage_prereg["cache_sha256"]
            fusion_runner.EXPECTED_CACHE_SHA256 = cache_sha
            for seed in SEEDS:
                output_dir = args.root / "fusion/runs" / stage / f"seed_{seed}"
                result_path = output_dir / f"modality_expert_simplex5_full_s{seed}_results.json"
                if result_path.is_file():
                    if not args.resume:
                        raise FileExistsError(f"Fusion result already exists: {result_path}")
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    records.append(
                        {
                            "stage": stage,
                            "seed": seed,
                            "status": "reused",
                            "result_path": str(result_path.resolve()),
                            "macro_mae": result["metrics"]["macro_mae"],
                        }
                    )
                    if stage == "s0_legacy" and seed == SEEDS[0]:
                        reproduction = _validate_s0_reproduction(
                            new_prediction_path=(
                                output_dir
                                / f"modality_expert_simplex5_full_s{seed}_predictions.csv"
                            ),
                            historical_prediction_path=(
                                args.historical_runs
                                / f"seed_{seed}"
                                / f"modality_expert_simplex5_full_s{seed}_predictions.csv"
                            ),
                        )
                        _write_json(
                            args.root / "fusion/s0_historical_reproduction.json",
                            reproduction,
                        )
                        print("S0 historical reproduction: within frozen numeric tolerances", flush=True)
                    continue
                if output_dir.exists() and any(output_dir.iterdir()):
                    raise FileExistsError(f"Partial nonempty output requires inspection: {output_dir}")
                run_args = SimpleNamespace(
                    method="modality_expert_simplex5",
                    variant="full",
                    seed=seed,
                    preregistration=Path(stage_prereg["path"]),
                    embedding_cache=cache,
                    labels=args.labels,
                    windows=args.windows,
                    split_manifest=args.split_manifest,
                    mask_manifest=args.mask_manifest,
                    cohorts=args.cohorts,
                    output_dir=output_dir,
                    compression_cache_dir=args.root / "fusion/component_cache/composite" / stage,
                )
                started = time.perf_counter()
                print(f"Fusion start: stage={stage} seed={seed}", flush=True)
                result = fusion_runner.run(run_args)
                elapsed = time.perf_counter() - started
                record = {
                    "stage": stage,
                    "seed": seed,
                    "status": "completed",
                    "runtime_seconds": elapsed,
                    "result_path": result["result_path"],
                    "prediction_path": result["prediction_path"],
                    "macro_mae": result["metrics"]["macro_mae"],
                    "delta_vs_condition_macro": result["delta_vs_condition"]["macro"],
                }
                records.append(record)
                print(
                    f"Fusion done: stage={stage} seed={seed} "
                    f"macro_mae={record['macro_mae']:.9f} time={elapsed:.1f}s",
                    flush=True,
                )
                if stage == "s0_legacy" and seed == SEEDS[0]:
                    reproduction = _validate_s0_reproduction(
                        new_prediction_path=Path(result["prediction_path"]),
                        historical_prediction_path=(
                            args.historical_runs
                            / f"seed_{seed}"
                            / f"modality_expert_simplex5_full_s{seed}_predictions.csv"
                        ),
                    )
                    _write_json(args.root / "fusion/s0_historical_reproduction.json", reproduction)
                    print("S0 historical reproduction: within frozen numeric tolerances", flush=True)
    finally:
        fusion_runner._load_or_fit_modality_bank = original_loader

    result = {
        "schema_version": "relax_embedding_ladder_fusion_execution_v1",
        "preparation": preparation,
        "records": records,
        "completed_runs": len(records),
        "expected_runs": len(STAGES) * len(SEEDS),
    }
    if len(records) != len(STAGES) * len(SEEDS):
        raise ValueError("Fusion execution matrix is incomplete")
    _write_json(args.root / "fusion/fusion_execution_manifest.json", result)
    print(json.dumps({
        "completed_runs": len(records),
        "manifest": str((args.root / "fusion/fusion_execution_manifest.json").resolve()),
    }, ensure_ascii=False, indent=2))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preregistration-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--legacy-component-dir", type=Path, default=DEFAULT_LEGACY_COMPONENTS)
    parser.add_argument("--historical-runs", type=Path, default=DEFAULT_HISTORICAL_RUNS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_CONTRACT / "condition_labels.csv")
    parser.add_argument("--windows", type=Path, default=DEFAULT_CONTRACT / "windows.csv")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_CONTRACT / "split_manifest.csv")
    parser.add_argument(
        "--mask-manifest",
        type=Path,
        default=DEFAULT_CONTRACT / "common_valid_window_masks.csv",
    )
    parser.add_argument("--cohorts", type=Path, default=DEFAULT_CONTRACT / "cohorts.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.manifest = args.manifest or args.root / "embedding_ladder_manifest.json"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
