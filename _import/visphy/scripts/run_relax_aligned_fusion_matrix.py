"""Run and validate the 5 x 2 x 3 aligned Project B neural fusion matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_relax_foundation_probe.py"
FUSIONS = ("early", "mid", "qformer", "healnet", "mm_lego")
SEEDS = (20260705, 20260706, 20260707)
MASKS = {
    "full": ("eeg", "ecg", "eye", "head", "video"),
    "no_eeg": ("ecg", "eye", "head", "video"),
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _validate_run(
    result_path: Path,
    prediction_path: Path,
    *,
    fusion: str,
    mask_name: str,
    modalities: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path)
    expected_model = f"corrected_{fusion}_{mask_name}"
    if payload.get("model") != expected_model or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Wrong result identity in {result_path}")
    if payload.get("modalities") != list(modalities):
        raise ValueError(f"Wrong modality mask in {result_path}")
    folds = payload.get("folds", [])
    if len(folds) != 15 or any(
        int(fold.get("n_train_participants", -1)) != 13
        or int(fold.get("n_validation_participants", -1)) != 1
        or int(fold.get("n_test_participants", -1)) != 1
        for fold in folds
    ):
        raise ValueError(f"Fold protocol is not 13/1/1 in {result_path}")
    if not payload.get("runtime", {}).get("cuda_used") or any(
        not fold.get("cuda_used") for fold in folds
    ):
        raise ValueError(f"CUDA was not recorded for every fold in {result_path}")
    if len(predictions) != 135 or predictions.duplicated(["participant_id", "condition"]).any():
        raise ValueError(f"Prediction coverage is not 135 unique conditions in {prediction_path}")
    if set(predictions["seed"].astype(int)) != {seed}:
        raise ValueError(f"Wrong prediction seed in {prediction_path}")
    if set(predictions["model_variant"].astype(str)) != {f"{fusion}_{mask_name}"}:
        raise ValueError(f"Wrong prediction model variant in {prediction_path}")
    if set(predictions["modalities"].astype(str)) != {"+".join(modalities)}:
        raise ValueError(f"Wrong prediction modalities in {prediction_path}")
    cuda_values = predictions["cuda_used"].astype(str).str.lower().isin({"true", "1", "1.0"})
    if not cuda_values.all():
        raise ValueError(f"A prediction row is not marked CUDA in {prediction_path}")
    if predictions["fold_index"].nunique() != 15 or predictions["participant_id"].nunique() != 15:
        raise ValueError(f"Fold/participant coverage is incomplete in {prediction_path}")
    return {
        "model": expected_model,
        "seed": seed,
        "rows": len(predictions),
        "folds": len(folds),
        "participants": int(predictions["participant_id"].nunique()),
        "cuda_used": True,
        "result_path": str(result_path.resolve()),
        "prediction_path": str(prediction_path.resolve()),
    }


def _command(
    args: argparse.Namespace,
    *,
    fusion: str,
    mask_name: str,
    modalities: tuple[str, ...],
    seed: int,
    output_dir: Path,
) -> list[str]:
    contract = args.contract_dir
    return [
        sys.executable,
        str(RUNNER),
        "--embedding-cache", str(args.embedding_cache),
        "--cohorts", str(contract / "cohorts.json"),
        "--cohort", "all_135",
        "--modalities", *modalities,
        "--fusion", fusion,
        "--baseline", "none",
        "--target", "original",
        "--condition-control", "none",
        "--pool", "sequence",
        "--device", "cuda",
        "--require-cuda",
        "--seed", str(seed),
        "--batch-size", "16",
        "--max-epochs", "40",
        "--patience", "8",
        "--lr", "0.001",
        "--weight-decay", "0.0001",
        "--d-common", "64",
        "--dropout", "0.1",
        "--fusion-depth", "1",
        "--fusion-heads", "2",
        "--cross-attn-freq", "1",
        "--dim-head", "16",
        "--latent-dim", "32",
        "--latent-channels", "8",
        "--lego-mode", "merge-sum",
        "--run-tag", "cross_project_aligned_20260716",
        "--split-manifest", str(contract / "split_manifest.csv"),
        "--mask-manifest", str(contract / "shared_modality_masks.csv"),
        "--labels", str(contract / "condition_labels.csv"),
        "--windows", str(contract / "windows.csv"),
        "--strict",
        "--output-dir", str(output_dir),
        "--run-name", f"{fusion}_{mask_name}_s{seed}",
    ]


def _run_one(
    args: argparse.Namespace,
    *,
    fusion: str,
    mask_name: str,
    modalities: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    run_name = f"{fusion}_{mask_name}_s{seed}"
    output_dir = args.output_root / f"{fusion}_{mask_name}" / f"seed_{seed}"
    result_path = output_dir / f"{run_name}_results.json"
    prediction_path = output_dir / f"{run_name}_predictions.csv"
    command = _command(
        args,
        fusion=fusion,
        mask_name=mask_name,
        modalities=modalities,
        seed=seed,
        output_dir=output_dir,
    )
    print(f"START {run_name}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, check=False)
    record: dict[str, Any] = {
        "run_name": run_name,
        "command": command,
        "returncode": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
    }
    try:
        if completed.returncode != 0:
            raise RuntimeError(f"runner exited {completed.returncode}")
        record.update(
            _validate_run(
                result_path,
                prediction_path,
                fusion=fusion,
                mask_name=mask_name,
                modalities=modalities,
                seed=seed,
            )
        )
        record["ok"] = True
        print(f"PASS  {run_name}", flush=True)
    except Exception as error:
        record.update(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        print(f"FAIL  {run_name}: {error}", flush=True)
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    status_path = args.output_root / "new_fusion_matrix_status.json"
    status: dict[str, Any] = {
        "schema_version": "relax_aligned_fusion_matrix_v1",
        "ok": False,
        "contract_dir": str(args.contract_dir.resolve()),
        "embedding_cache": str(args.embedding_cache.resolve()),
        "architectures": list(args.architectures),
        "seeds": list(args.seeds),
        "parallel_seeds": args.parallel_seeds,
        "runs": [],
    }
    failures = 0
    for fusion in args.architectures:
        for mask_name, modalities in MASKS.items():
            with ThreadPoolExecutor(max_workers=args.parallel_seeds) as executor:
                futures = [
                    executor.submit(
                        _run_one,
                        args,
                        fusion=fusion,
                        mask_name=mask_name,
                        modalities=modalities,
                        seed=seed,
                    )
                    for seed in args.seeds
                ]
                for future in as_completed(futures):
                    record = future.result()
                    failures += int(not record.get("ok"))
                    status["runs"].append(record)
                    _write_json(status_path, status)
    status["ok"] = failures == 0
    status["successful_runs"] = sum(bool(run.get("ok")) for run in status["runs"])
    status["failed_runs"] = failures
    _write_json(status_path, status)
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", choices=FUSIONS, default=list(FUSIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--parallel-seeds", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args(argv)
    if tuple(args.seeds) != SEEDS:
        parser.error(f"Formal matrix requires seeds {SEEDS}")
    if tuple(args.architectures) != FUSIONS:
        parser.error(f"Formal matrix requires architectures {FUSIONS}")
    for name in (
        "cohorts.json",
        "split_manifest.csv",
        "shared_modality_masks.csv",
        "condition_labels.csv",
        "windows.csv",
    ):
        if not (args.contract_dir / name).is_file():
            parser.error(f"Contract file is missing: {args.contract_dir / name}")
    if not args.embedding_cache.is_file():
        parser.error(f"Embedding cache is missing: {args.embedding_cache}")
    return args


def main(argv: list[str] | None = None) -> int:
    status = run(parse_args(argv))
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
