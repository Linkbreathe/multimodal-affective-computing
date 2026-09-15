"""Run and validate the 6 x 6 x 3 EEG-eligible modality-ablation matrix."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_relax_foundation_probe.py"
ARCHITECTURES = ("late", "early", "mid", "qformer", "healnet", "mm_lego")
SEEDS = (20260705, 20260706, 20260707)
PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
CONFIGURATIONS = {
    "full": ("eeg", "ecg", "eye", "head", "video"),
    "no_eeg": ("ecg", "eye", "head", "video"),
    "no_ecg": ("eeg", "eye", "head", "video"),
    "no_eye": ("eeg", "ecg", "head", "video"),
    "no_head": ("eeg", "ecg", "eye", "video"),
    "no_video": ("eeg", "ecg", "eye", "head"),
}
TARGETS = ("relaxation", "discomfort")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except BrokenPipeError:
        pass


def _as_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "1.0"})


def _load_protocol(contract_dir: Path) -> dict[str, Any]:
    contract_path = contract_dir / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "eeg_eligible_modality_ablation_contract_v1":
        raise ValueError("Wrong EEG-eligible ablation contract schema")
    if tuple(contract.get("participants", ())) != PARTICIPANTS:
        raise ValueError("Contract participants differ from the frozen nine-person cohort")
    if tuple(contract.get("architectures", ())) != ARCHITECTURES:
        raise ValueError("Contract architectures differ from the requested six")
    if contract.get("modality_configurations") != {
        name: list(modalities) for name, modalities in CONFIGURATIONS.items()
    }:
        raise ValueError("Contract modality configurations differ from the requested six")
    if tuple(contract.get("seeds", ())) != SEEDS:
        raise ValueError("Contract seeds differ from the requested three")
    for record in contract["files"].values():
        path = contract_dir / record["path"]
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Contract file failed hash validation: {path}")
    return contract


def _expected_fold_windows(contract_dir: Path) -> dict[int, int]:
    masks = pd.read_csv(contract_dir / "common_valid_window_masks.csv")
    masks["common_valid"] = _as_true(masks["common_valid"])
    per_participant = masks.groupby("participant_id")["common_valid"].sum().astype(int).to_dict()
    splits = pd.read_csv(contract_dir / "split_manifest.csv")
    return {
        int(fold_index): int(
            sum(
                per_participant[str(participant)]
                for participant in group.loc[group["role"] == "train", "participant_id"]
            )
        )
        for fold_index, group in splits.groupby("fold_index", sort=True)
    }


def _validate_run(
    result_path: Path,
    prediction_path: Path,
    *,
    contract_dir: Path,
    contract: dict[str, Any],
    expected_fold_windows: dict[int, int],
    embedding_cache_sha256: str,
    architecture: str,
    configuration: str,
    modalities: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path)
    expected_model = f"corrected_{architecture}_{configuration}"
    expected_variant = f"{architecture}_{configuration}"
    if payload.get("model") != expected_model or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Wrong result identity in {result_path}")
    if payload.get("cohort") != "eeg_eligible" or payload.get("modalities") != list(modalities):
        raise ValueError(f"Wrong cohort or modality configuration in {result_path}")
    if payload.get("split_protocol") != "shared_7_train_1_validation_1_test":
        raise ValueError(f"Wrong split protocol in {result_path}")
    if int(payload.get("participant_count", -1)) != 9 or int(payload.get("observation_count", -1)) != 81:
        raise ValueError(f"Wrong cohort/observation count in {result_path}")
    training = payload.get("training", {})
    expected_training = {
        "batch_size": 16,
        "max_epochs": 40,
        "patience": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "d_common": 64,
    }
    if training != expected_training:
        raise ValueError(f"Wrong training settings in {result_path}: {training}")
    fusion = payload.get("fusion_config", {})
    if (
        fusion.get("fusion") != architecture
        or fusion.get("d_common") != 64
        or fusion.get("dropout") != 0.1
        or fusion.get("fusion_depth") != 1
        or fusion.get("fusion_heads") != 2
        or fusion.get("cross_attn_freq") != 1
        or fusion.get("latent_channels") != 8
        or fusion.get("latent_dim") != 32
        or fusion.get("dim_head") != 16
        or fusion.get("lego_mode") != "merge-sum"
    ):
        raise ValueError(f"Wrong fusion settings in {result_path}")

    expected_input_hashes = {
        "labels": contract["files"]["labels"]["sha256"],
        "windows": contract["files"]["windows"]["sha256"],
        "split_manifest": contract["files"]["split_manifest"]["sha256"],
        "mask_manifest": contract["files"]["common_masks"]["sha256"],
        "cohorts": contract["files"]["cohorts"]["sha256"],
        "embedding_cache": embedding_cache_sha256,
    }
    for name, expected_hash in expected_input_hashes.items():
        if payload.get("inputs", {}).get(name, {}).get("sha256") != expected_hash:
            raise ValueError(f"Wrong {name} hash in {result_path}")

    folds = payload.get("folds", [])
    split_frame = pd.read_csv(contract_dir / "split_manifest.csv")
    expected_fold_identity = {
        int(fold_index): (
            str(group["test_participant"].iloc[0]),
            str(group["validation_participant"].iloc[0]),
        )
        for fold_index, group in split_frame.groupby("fold_index", sort=True)
    }
    if len(folds) != 9:
        raise ValueError(f"Expected nine folds in {result_path}")
    for fold in folds:
        fold_index = int(fold.get("fold_index", -1))
        expected_test, expected_validation = expected_fold_identity[fold_index]
        if (
            int(fold.get("n_train_participants", -1)) != 7
            or int(fold.get("n_validation_participants", -1)) != 1
            or int(fold.get("n_test_participants", -1)) != 1
            or str(fold.get("test_participant")) != expected_test
            or str(fold.get("validation_participant")) != expected_validation
            or not fold.get("cuda_used")
        ):
            raise ValueError(f"Invalid fold {fold_index} metadata in {result_path}")
        counts = fold.get("mask_valid_windows_train", {})
        if set(counts) != set(modalities) or set(map(int, counts.values())) != {
            expected_fold_windows[fold_index]
        }:
            raise ValueError(f"Fold {fold_index} does not use the frozen common-window policy")
    if not payload.get("runtime", {}).get("cuda_used"):
        raise ValueError(f"Top-level runtime is not CUDA in {result_path}")

    labels = pd.read_csv(contract_dir / "condition_labels.csv")
    if len(predictions) != 81 or predictions.duplicated(["participant_id", "condition"]).any():
        raise ValueError(f"Prediction coverage is not 81 unique observations in {prediction_path}")
    merged = predictions.merge(
        labels[["participant_id", "condition", *TARGETS]],
        on=["participant_id", "condition"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"Prediction keys differ from the contract in {prediction_path}")
    for target in TARGETS:
        if not (
            pd.to_numeric(merged[f"{target}_true"], errors="raise")
            - pd.to_numeric(merged[target], errors="raise")
        ).abs().le(1e-6).all():
            raise ValueError(f"Prediction truths differ for {target} in {prediction_path}")
        values = pd.to_numeric(merged[f"{target}_pred"], errors="raise")
        if values.isna().any() or not values.between(0.0, 1.0).all():
            raise ValueError(f"Invalid predictions for {target} in {prediction_path}")
    if set(predictions["seed"].astype(int)) != {seed}:
        raise ValueError(f"Wrong seed in {prediction_path}")
    if set(predictions["model_variant"].astype(str)) != {expected_variant}:
        raise ValueError(f"Wrong model variant in {prediction_path}")
    if set(predictions["modalities"].astype(str)) != {"+".join(modalities)}:
        raise ValueError(f"Wrong modalities in {prediction_path}")
    if set(predictions["split_protocol"].astype(str)) != {"shared_7_train_1_validation_1_test"}:
        raise ValueError(f"Wrong prediction split protocol in {prediction_path}")
    if not _as_true(predictions["cuda_used"]).all():
        raise ValueError(f"A prediction row is not CUDA in {prediction_path}")
    if predictions["fold_index"].nunique() != 9 or predictions["participant_id"].nunique() != 9:
        raise ValueError(f"Fold/participant coverage is incomplete in {prediction_path}")
    return {
        "model": expected_model,
        "architecture": architecture,
        "configuration": configuration,
        "modalities": list(modalities),
        "seed": seed,
        "rows": len(predictions),
        "folds": len(folds),
        "participants": int(predictions["participant_id"].nunique()),
        "cuda_used": True,
        "result_path": str(result_path.resolve()),
        "prediction_path": str(prediction_path.resolve()),
        "result_sha256": file_sha256(result_path),
        "prediction_sha256": file_sha256(prediction_path),
    }


def _command(
    args: argparse.Namespace,
    *,
    architecture: str,
    configuration: str,
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
        "--cohort", "eeg_eligible",
        "--modalities", *modalities,
        "--fusion", architecture,
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
        "--run-tag", "eeg_eligible_modality_ablation_20260716",
        "--split-manifest", str(contract / "split_manifest.csv"),
        "--mask-manifest", str(contract / "common_valid_window_masks.csv"),
        "--labels", str(contract / "condition_labels.csv"),
        "--windows", str(contract / "windows.csv"),
        "--strict",
        "--output-dir", str(output_dir),
        "--run-name", f"eeg9_{architecture}_{configuration}_s{seed}",
    ]


def _run_one(
    args: argparse.Namespace,
    *,
    contract: dict[str, Any],
    expected_fold_windows: dict[int, int],
    embedding_cache_sha256: str,
    architecture: str,
    configuration: str,
    modalities: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    run_name = f"eeg9_{architecture}_{configuration}_s{seed}"
    output_dir = args.output_root / architecture / configuration / f"seed_{seed}"
    result_path = output_dir / f"{run_name}_results.json"
    prediction_path = output_dir / f"{run_name}_predictions.csv"
    validation_kwargs = {
        "contract_dir": args.contract_dir,
        "contract": contract,
        "expected_fold_windows": expected_fold_windows,
        "embedding_cache_sha256": embedding_cache_sha256,
        "architecture": architecture,
        "configuration": configuration,
        "modalities": modalities,
        "seed": seed,
    }
    if args.resume and result_path.is_file() and prediction_path.is_file():
        try:
            record = _validate_run(result_path, prediction_path, **validation_kwargs)
            record.update({"run_name": run_name, "ok": True, "resumed": True})
            _safe_print(f"SKIP  {run_name} (already valid)")
            return record
        except Exception as error:
            resume_error = f"{type(error).__name__}: {error}"
    else:
        resume_error = None

    output_dir.mkdir(parents=True, exist_ok=True)
    command = _command(
        args,
        architecture=architecture,
        configuration=configuration,
        modalities=modalities,
        seed=seed,
        output_dir=output_dir,
    )
    log_path = output_dir / "runner.log"
    _safe_print(f"START {run_name}")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    record: dict[str, Any] = {
        "run_name": run_name,
        "command": command,
        "returncode": completed.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "runner_log": str(log_path.resolve()),
        "resumed": False,
    }
    if resume_error is not None:
        record["resume_validation_error"] = resume_error
    try:
        if completed.returncode != 0:
            raise RuntimeError(f"runner exited {completed.returncode}")
        record.update(_validate_run(result_path, prediction_path, **validation_kwargs))
        record["ok"] = True
        _safe_print(f"PASS  {run_name}")
    except Exception as error:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        record.update(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "runner_log_tail": tail,
            }
        )
        _safe_print(f"FAIL  {run_name}: {error}")
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = _load_protocol(args.contract_dir)
    expected_fold_windows = _expected_fold_windows(args.contract_dir)
    embedding_cache_sha256 = file_sha256(args.embedding_cache)
    status_path = args.output_root / "eeg_eligible_ablation_matrix_status.json"
    status: dict[str, Any] = {
        "schema_version": "eeg_eligible_modality_ablation_matrix_v1",
        "ok": False,
        "contract_path": str((args.contract_dir / "contract.json").resolve()),
        "contract_sha256": file_sha256(args.contract_dir / "contract.json"),
        "embedding_cache": str(args.embedding_cache.resolve()),
        "embedding_cache_sha256": embedding_cache_sha256,
        "architectures": list(ARCHITECTURES),
        "configurations": {name: list(values) for name, values in CONFIGURATIONS.items()},
        "seeds": list(SEEDS),
        "expected_runs": 108,
        "resume": args.resume,
        "runs": [],
    }
    for architecture in ARCHITECTURES:
        for configuration, modalities in CONFIGURATIONS.items():
            for seed in SEEDS:
                record = _run_one(
                    args,
                    contract=contract,
                    expected_fold_windows=expected_fold_windows,
                    embedding_cache_sha256=embedding_cache_sha256,
                    architecture=architecture,
                    configuration=configuration,
                    modalities=modalities,
                    seed=seed,
                )
                status["runs"].append(record)
                status["successful_runs"] = sum(bool(run.get("ok")) for run in status["runs"])
                status["failed_runs"] = sum(not bool(run.get("ok")) for run in status["runs"])
                _write_json(status_path, status)
                if not record.get("ok"):
                    status["stopped_after_failure"] = record["run_name"]
                    _write_json(status_path, status)
                    return status
    status["ok"] = len(status["runs"]) == 108 and status["failed_runs"] == 0
    _write_json(status_path, status)
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not args.embedding_cache.is_file():
        parser.error(f"Embedding cache is missing: {args.embedding_cache}")
    return args


def main(argv: list[str] | None = None) -> int:
    status = run(parse_args(argv))
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
