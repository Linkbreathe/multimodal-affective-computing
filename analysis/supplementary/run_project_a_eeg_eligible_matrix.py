"""Run the 2 x 5 x 3 Project A EEG-eligible ablation matrix."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

from mac.evaluation.alignment import validate_alignment_contract


MODELS = ("classical", "dcnn")
VARIANTS = ("full", "no_eeg", "no_ecg", "no_eye", "no_head")
SEEDS = (20260705, 20260706, 20260707)
PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
MODALITIES = ("eeg", "ecg", "eye", "head")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _modalities(variant: str) -> tuple[str, ...]:
    if variant == "full":
        return MODALITIES
    removed = variant.removeprefix("no_")
    return tuple(modality for modality in MODALITIES if modality != removed)


def _csv_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_csv_option(value: str, allowed: tuple[Any, ...], converter=str) -> tuple[Any, ...]:
    requested = tuple(converter(item.strip()) for item in value.split(",") if item.strip())
    unknown = sorted(set(requested) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported selections: {unknown}; allowed={list(allowed)}")
    return requested


def _config_payload(
    *,
    model: str,
    variant: str,
    seed: int,
    contract_dir: Path,
    feature_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    contract = json.loads((contract_dir / "contract.json").read_text(encoding="utf-8"))
    run_id = f"eeg9-project-a-{model}-{variant}-s{seed}"
    payload: dict[str, Any] = {
        "extends": str((ROOT / "configs" / "base.yaml").resolve()).replace("\\", "/"),
        "run": {
            "id": run_id,
            "mode": "research",
            "output_root": str(output_root.resolve()).replace("\\", "/"),
        },
        "experiment": {"research_only": True},
        "alignment": {
            "enabled": True,
            "require_cuda": True,
            "contract_dir": str(contract_dir.resolve()).replace("\\", "/"),
            "split_manifest": str(
                (contract_dir / contract["files"]["split_manifest"]["path"]).resolve()
            ).replace("\\", "/"),
            "modality_masks": str(
                (contract_dir / contract["files"]["common_masks"]["path"]).resolve()
            ).replace("\\", "/"),
            "labels": str(
                (contract_dir / contract["files"]["labels"]["path"]).resolve()
            ).replace("\\", "/"),
            "windows": str(
                (contract_dir / contract["files"]["windows"]["path"]).resolve()
            ).replace("\\", "/"),
            "window_features": str(feature_path.resolve()).replace("\\", "/"),
        },
        "modeling": {
            "random_seed": int(seed),
            "runtime_backend": "classical" if model == "classical" else "dcnn",
            "condition_variant": variant,
            "dcnn": {
                "device": "cuda",
                "variants": [variant],
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 16,
                "max_epochs": 40,
                "early_stopping_patience": 8,
            },
        },
    }
    return payload


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _run_paths(run_root: Path, model: str) -> dict[str, Path]:
    if model == "classical":
        return {
            "predictions": run_root / "predictions" / "condition_level_lopo_predictions.csv",
            "metrics": run_root / "metrics" / "condition_level_lopo_metrics.json",
            "alignment_manifest": run_root / "manifests" / "classical_alignment_manifest.json",
            "model": run_root / "models" / "state_model.joblib",
        }
    return {
        "predictions": run_root / "predictions" / "dcnn_condition_lopo_predictions.csv",
        "metrics": run_root / "metrics" / "dcnn_condition_lopo_metrics.json",
        "alignment_manifest": run_root / "manifests" / "dcnn_alignment_manifest.json",
        "model": run_root / "models" / "dcnn_state_PLACEHOLDER.pt",
    }


def _validate_completed_run(
    run_root: Path,
    model: str,
    variant: str,
    seed: int,
    feature_hash: str,
) -> dict[str, Any]:
    paths = _run_paths(run_root, model)
    if model == "dcnn":
        paths["model"] = run_root / "models" / f"dcnn_state_{variant}.pt"
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Run {run_root.name} is missing outputs: {missing}")

    frame = pd.read_csv(paths["predictions"])
    if len(frame) != 81 or frame.duplicated(["participant_id", "condition"]).any():
        raise ValueError(f"Run {run_root.name} does not contain 81 unique OOF predictions")
    if set(frame["participant_id"].astype(str)) != set(PARTICIPANTS):
        raise ValueError(f"Run {run_root.name} has the wrong participant cohort")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {seed}:
        raise ValueError(f"Run {run_root.name} has the wrong seed")
    if set(frame["model_variant"].astype(str)) != {variant}:
        raise ValueError(f"Run {run_root.name} has the wrong model variant")
    if set(frame["modalities"].astype(str)) != {"+".join(_modalities(variant))}:
        raise ValueError(f"Run {run_root.name} has the wrong modality list")
    if set(frame["split_protocol"].astype(str)) != {"shared_7_train_1_validation_1_test"}:
        raise ValueError(f"Run {run_root.name} has the wrong fold protocol")
    if set(pd.to_numeric(frame["fold_index"], errors="raise").astype(int)) != set(range(1, 10)):
        raise ValueError(f"Run {run_root.name} does not cover all nine folds")
    for target in ("relaxation", "discomfort"):
        values = pd.to_numeric(frame[f"pred_{target}"], errors="raise")
        if values.isna().any() or not values.between(0.0, 1.0).all():
            raise ValueError(f"Run {run_root.name} has invalid {target} predictions")
    cuda = _csv_true(frame["cuda_used"])
    if model == "dcnn" and not cuda.all():
        raise ValueError(f"Run {run_root.name} did not use CUDA for every prediction")
    if model == "classical" and cuda.any():
        raise ValueError(f"Run {run_root.name} unexpectedly records CUDA")

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    variant_metrics = metrics["variants"][variant]["metrics"] if model == "dcnn" else metrics["metrics"]
    folds = metrics["variants"][variant]["folds"] if model == "dcnn" else metrics["folds"]
    if len(folds) != 9:
        raise ValueError(f"Run {run_root.name} metrics do not contain nine folds")
    for fold in folds:
        if (
            int(fold["n_train_participants"]) != 7
            or int(fold["n_validation_participants"]) != 1
            or int(fold["n_test_participants"]) != 1
        ):
            raise ValueError(f"Run {run_root.name} has a non-7/1/1 fold")
        if model == "dcnn" and not bool(fold.get("cuda_used")):
            raise ValueError(f"Run {run_root.name} has a non-CUDA DCNN fold")
    if variant_metrics.get("model_variant") != variant:
        raise ValueError(f"Run {run_root.name} metrics have the wrong model variant")

    manifest = json.loads(paths["alignment_manifest"].read_text(encoding="utf-8"))
    if manifest.get("sources", {}).get("window_features", {}).get("sha256") != feature_hash:
        raise ValueError(f"Run {run_root.name} used the wrong Project A feature table")
    if len(manifest.get("folds", [])) != 9:
        raise ValueError(f"Run {run_root.name} alignment manifest does not contain nine folds")
    runtime = manifest.get("runtime", {})
    if model == "dcnn" and not runtime.get("cuda_used"):
        raise ValueError(f"Run {run_root.name} alignment manifest does not confirm CUDA")

    return {
        "prediction_rows": len(frame),
        "participants": int(frame["participant_id"].nunique()),
        "folds": int(frame["fold_index"].nunique()),
        "cuda_used": bool(model == "dcnn"),
        "device": str(frame["device"].iloc[0]),
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "sha256": {name: file_sha256(path) for name, path in paths.items()},
    }


def _tee_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(process.wait())


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_alignment_contract(args.contract_dir)
    if contract.get("schema_version") != "eeg_eligible_modality_ablation_contract_v1":
        raise ValueError("The matrix requires the frozen nine-participant ablation contract")
    if tuple(contract.get("participants", ())) != PARTICIPANTS:
        raise ValueError("The contract cohort differs from the frozen nine participants")
    if tuple(contract.get("seeds", ())) != SEEDS:
        raise ValueError("The contract seeds differ from the formal seed set")

    input_manifest_path = args.inputs_dir / "project_a_input_manifest.json"
    feature_path = args.inputs_dir / "project_a_window_features_common.csv"
    if not input_manifest_path.is_file() or not feature_path.is_file():
        raise FileNotFoundError(
            "Build Project A common-window inputs before running the matrix"
        )
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    feature_hash = file_sha256(feature_path)
    if input_manifest["outputs"]["window_features"]["sha256"] != feature_hash:
        raise ValueError("Project A common-window feature hash differs from its manifest")

    models = _parse_csv_option(args.models, MODELS)
    variants = _parse_csv_option(args.variants, VARIANTS)
    seeds = _parse_csv_option(args.seeds, SEEDS, int)
    if "dcnn" in models:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Formal Project A DCNN runs require CUDA, but CUDA is unavailable")
        cuda_runtime = {
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cuda_device_name": torch.cuda.get_device_name(0),
        }
    else:
        cuda_runtime = {}

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.config_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root.parent / "eeg9_project_a_matrix_status.json"
    code_files = [
        ROOT / "src" / "mac" / "evaluation" / "alignment.py",
        ROOT / "src" / "mac" / "training" / "condition_train.py",
        ROOT / "src" / "mac" / "models" / "dcnn.py",
        Path(__file__).resolve(),
    ]
    status: dict[str, Any] = {
        "schema_version": "project_a_eeg_eligible_matrix_status_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(ROOT.resolve()),
        "python": str(Path(sys.executable).resolve()),
        "contract": {"path": str((args.contract_dir / "contract.json").resolve()), "sha256": file_sha256(args.contract_dir / "contract.json")},
        "input_manifest": {"path": str(input_manifest_path.resolve()), "sha256": file_sha256(input_manifest_path)},
        "window_features_sha256": feature_hash,
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in code_files},
        "models": list(models),
        "variants": list(variants),
        "seeds": list(seeds),
        "expected_runs": len(models) * len(variants) * len(seeds),
        "cuda_preflight": cuda_runtime,
        "runs": [],
        "ok": False,
    }
    _write_json(status_path, status)

    for model in models:
        for variant in variants:
            for seed in seeds:
                run_id = f"eeg9-project-a-{model}-{variant}-s{seed}"
                run_root = args.output_root / run_id
                config_path = args.config_root / model / variant / f"seed_{seed}.yaml"
                payload = _config_payload(
                    model=model,
                    variant=variant,
                    seed=seed,
                    contract_dir=args.contract_dir,
                    feature_path=feature_path,
                    output_root=args.output_root,
                )
                _write_config(config_path, payload)
                command = [
                    sys.executable,
                    "-m",
                    "mac.cli",
                    "--experiment",
                    str(config_path.resolve()),
                    "train-state" if model == "classical" else "train-dcnn-state",
                ]
                record: dict[str, Any] = {
                    "run_id": run_id,
                    "model": model,
                    "variant": variant,
                    "modalities": list(_modalities(variant)),
                    "seed": seed,
                    "config": str(config_path.resolve()),
                    "config_sha256": file_sha256(config_path),
                    "command": subprocess.list2cmdline(command),
                    "run_root": str(run_root.resolve()),
                    "log": str((run_root / "logs" / "runner.log").resolve()),
                }
                print(f"\n=== {run_id} ===", flush=True)
                try:
                    if args.resume:
                        try:
                            validation = _validate_completed_run(
                                run_root, model, variant, seed, feature_hash
                            )
                        except (FileNotFoundError, ValueError, KeyError):
                            validation = None
                        if validation is not None:
                            record.update({"status": "reused", "elapsed_seconds": 0.0, **validation})
                            status["runs"].append(record)
                            _write_json(status_path, status)
                            print(f"Reused validated run: {run_id}", flush=True)
                            continue
                    if args.dry_run:
                        record.update({"status": "planned", "elapsed_seconds": 0.0})
                        status["runs"].append(record)
                        _write_json(status_path, status)
                        print(record["command"], flush=True)
                        continue
                    started = time.perf_counter()
                    return_code = _tee_command(command, run_root / "logs" / "runner.log")
                    elapsed = time.perf_counter() - started
                    if return_code != 0:
                        raise RuntimeError(f"Command exited with code {return_code}")
                    validation = _validate_completed_run(
                        run_root, model, variant, seed, feature_hash
                    )
                    record.update(
                        {"status": "ok", "return_code": return_code, "elapsed_seconds": elapsed, **validation}
                    )
                    status["runs"].append(record)
                    _write_json(status_path, status)
                except Exception as error:
                    record.update({"status": "failed", "error": repr(error)})
                    status["runs"].append(record)
                    status["failed_runs"] = sum(
                        item["status"] == "failed" for item in status["runs"]
                    )
                    _write_json(status_path, status)
                    _write_json(run_root / "hard_failure.json", record)
                    raise

    completed = [item for item in status["runs"] if item["status"] in {"ok", "reused"}]
    status.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "successful_runs": len(completed),
            "failed_runs": sum(item["status"] == "failed" for item in status["runs"]),
            "planned_runs": sum(item["status"] == "planned" for item in status["runs"]),
            "ok": not args.dry_run and len(completed) == status["expected_runs"],
        }
    )
    _write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base = ROOT / "artifacts" / "cross_project_alignment_2026-07-16" / "eeg_eligible_ablation"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=base / "contract")
    parser.add_argument("--inputs-dir", type=Path, default=base / "project_a" / "inputs")
    parser.add_argument("--output-root", type=Path, default=base / "project_a" / "runs")
    parser.add_argument("--config-root", type=Path, default=base / "project_a" / "configs")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("contract_dir", "inputs_dir", "output_root", "config_root"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
