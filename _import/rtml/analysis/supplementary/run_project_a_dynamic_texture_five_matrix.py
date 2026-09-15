"""Run the formal 2-family x 2-video-variant x 3-seed Project A matrix."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

from real_time_ml.config import load_config_layers
from real_time_ml.evaluation.alignment import file_sha256, validate_alignment_contract
from real_time_ml.experiments.dynamic_texture_five import (
    run_classical_dynamic_texture,
    run_paired_dcnn_dynamic_texture,
)
from real_time_ml.utils import write_json


MODELS = ("classical", "1dcnn")
VARIANTS = ("full_five", "no_video")
SEEDS = (20260705, 20260706, 20260707)
PARTICIPANTS = ("P003", "P004", "P007", "P008", "P009", "P011", "P012", "P013", "P015")
DEFAULT_OUTPUT = Path(
    r"C:\Users\linki\amaster\foundation model result\91_Project_A_Dynamic_Texture_Five_Modality_Comparison_20260719"
)


class Tee(io.TextIOBase):
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _parse(value: str, allowed: tuple[Any, ...], converter: Any = str) -> tuple[Any, ...]:
    selected = tuple(converter(item.strip()) for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unsupported matrix selections: {unknown}; allowed={list(allowed)}")
    return selected


def _run_id(model: str, variant: str, seed: int) -> str:
    return f"project-a-dynamic-texture-{model}-{variant}-s{seed}"


def _config_payload(
    *,
    output_root: Path,
    contract_dir: Path,
    features: Path,
    model: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    contract = json.loads((contract_dir / "contract.json").read_text(encoding="utf-8"))
    files = contract["files"]
    return {
        "extends": str((ROOT / "configs" / "base.yaml").resolve()).replace("\\", "/"),
        "run": {
            "id": _run_id(model, variant, seed),
            "mode": "research",
            "output_root": str((output_root / "runs").resolve()).replace("\\", "/"),
        },
        "paths": {"reports_root": str((output_root / "report").resolve()).replace("\\", "/")},
        "experiment": {"research_only": True, "name": "dynamic_texture_five_modality"},
        "alignment": {
            "enabled": True,
            "require_cuda": model == "1dcnn",
            "contract_dir": str(contract_dir.resolve()).replace("\\", "/"),
            "split_manifest": str(
                (contract_dir / files["split_manifest"]["path"]).resolve()
            ).replace("\\", "/"),
            "modality_masks": str((contract_dir / files["common_masks"]["path"]).resolve()).replace(
                "\\", "/"
            ),
            "labels": str((contract_dir / files["labels"]["path"]).resolve()).replace("\\", "/"),
            "windows": str((contract_dir / files["windows"]["path"]).resolve()).replace("\\", "/"),
            "window_features": str(features.resolve()).replace("\\", "/"),
            "zero_common_window_observations": [{"participant_id": "P004", "condition": "C6"}],
        },
        "dynamic_texture": {
            "version": "dynamic_texture_v1",
            "descriptor_count": 12,
            "frames_per_window": 16,
            "analysis_shape": [16, 112, 112],
            "rgb_training_tensor_allowed": False,
            "frozen_common_mask": True,
        },
        "modeling": {
            "random_seed": int(seed),
            "runtime_backend": "classical" if model == "classical" else "dcnn",
            "condition_variant": variant,
            "dcnn": {
                "device": "cuda",
                "variants": [variant],
                "sequence_length": 8,
                "conv_channels": [16, 32],
                "kernel_sizes": [3, 3],
                "pool_sizes": [2, 2],
                "mlp_hidden": 64,
                "dropout": 0.30,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 16,
                "max_epochs": 40,
                "early_stopping_patience": 8,
                "min_non_missing_fraction": 0.40,
            },
        },
    }


def _write_resolved_config(
    output_root: Path,
    payload: dict[str, Any],
    model: str,
    variant: str,
    seed: int,
) -> tuple[Path, Any]:
    directory = output_root / "configs" / model / variant / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    yaml_path = directory / "experiment.yaml"
    yaml_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    config = load_config_layers(experiment=yaml_path)
    write_json(directory / "resolved_config.json", config.data)
    return yaml_path, config


def _validate_run(
    run_root: Path,
    *,
    model: str,
    variant: str,
    seed: int,
    feature_hash: str,
) -> dict[str, Any]:
    paths = {
        "predictions": run_root / "predictions" / "oof_predictions.csv",
        "metrics": run_root / "metrics" / "metrics.json",
        "manifest": run_root / "manifests" / "run_manifest.json",
    }
    if model == "1dcnn":
        paths["gating_audit"] = run_root / "manifests" / "gating_audit.csv"
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"{run_root.name} is missing outputs: {missing}")
    frame = pd.read_csv(paths["predictions"])
    if len(frame) != 81 or frame.duplicated(["participant_id", "condition"]).any():
        raise ValueError(f"{run_root.name} does not contain 81 unique OOF rows")
    if set(frame["participant_id"].astype(str)) != set(PARTICIPANTS):
        raise ValueError(f"{run_root.name} has the wrong participant cohort")
    if set(frame["variant"].astype(str)) != {variant}:
        raise ValueError(f"{run_root.name} has the wrong variant")
    if set(pd.to_numeric(frame["seed"], errors="raise").astype(int)) != {seed}:
        raise ValueError(f"{run_root.name} has the wrong seed")
    if set(pd.to_numeric(frame["fold_index"], errors="raise").astype(int)) != set(range(1, 10)):
        raise ValueError(f"{run_root.name} does not cover all nine folds")
    if set(frame["split_protocol"].astype(str)) != {"shared_7_train_1_validation_1_test"}:
        raise ValueError(f"{run_root.name} has the wrong split protocol")
    for target in ("relaxation", "discomfort"):
        prediction = pd.to_numeric(frame[f"pred_{target}"], errors="raise")
        if prediction.isna().any() or not prediction.between(0.0, 1.0).all():
            raise ValueError(f"{run_root.name} has invalid {target} predictions")
    zero = (
        frame["p004_c6_condition_only_fallback"].astype(str).str.lower().isin({"true", "1", "1.0"})
    )
    if int(zero.sum()) != 1:
        raise ValueError(f"{run_root.name} lacks the unique P004/C6 fallback row")
    for target in ("relaxation", "discomfort"):
        if not frame.loc[zero, f"pred_{target}"].equals(
            frame.loc[zero, f"condition_only_{target}"]
        ):
            raise ValueError(f"{run_root.name} P004/C6 fallback is not exact")
    cuda = frame["cuda_used"].astype(str).str.lower().isin({"true", "1", "1.0"})
    if model == "1dcnn" and not cuda.all():
        raise ValueError(f"{run_root.name} contains a non-CUDA prediction")
    if model == "classical" and cuda.any():
        raise ValueError(f"{run_root.name} unexpectedly records CUDA")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    if manifest["sources"]["window_features"]["sha256"] != feature_hash:
        raise ValueError(f"{run_root.name} used the wrong feature hash")
    if len(manifest["folds"]) != 9:
        raise ValueError(f"{run_root.name} manifest lacks nine folds")
    if model == "1dcnn":
        fold_gradients = [float(fold["maximum_video_gradient"]) for fold in metrics["folds"]]
        if len(fold_gradients) != 9:
            raise ValueError(f"{run_root.name} metrics lack nine fold gradients")
        if variant == "full_five" and not all(value > 0.0 for value in fold_gradients):
            raise ValueError(f"{run_root.name} lacks a nonzero Video gradient in every fold")
        if variant == "no_video" and not all(value == 0.0 for value in fold_gradients):
            raise ValueError(f"{run_root.name} has a nonzero gated Video gradient")
        gate = pd.read_csv(paths["gating_audit"])
        if len(gate) != 9:
            raise ValueError(f"{run_root.name} gating audit lacks nine folds")
        exact_zero = [
            column
            for column in gate.columns
            if column.endswith("max_abs")
            and (
                "no_video" in column or "embedding" in column or "video_encoder_gradient" in column
            )
        ]
        if exact_zero and not (gate[exact_zero].astype(float) == 0.0).all().all():
            raise ValueError(f"{run_root.name} failed an exact-zero gate assertion")
        if not gate["trained_prediction_parameter_invariant_exact"].astype(bool).all():
            raise ValueError(f"{run_root.name} prediction depends on Video encoder parameters")
    return {
        "prediction_rows": len(frame),
        "folds": int(frame["fold_index"].nunique()),
        "participants": int(frame["participant_id"].nunique()),
        "device": str(frame["device"].iloc[0]),
        "cuda_used": bool(model == "1dcnn"),
        "outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()
        },
    }


def _record_log(run_root: Path, text: str) -> None:
    path = run_root / "logs" / "runner.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    contract = validate_alignment_contract(args.contract_dir)
    if tuple(contract.get("participants", ())) != PARTICIPANTS:
        raise ValueError("Contract cohort differs from the frozen nine participants")
    if tuple(contract.get("seeds", ())) != SEEDS:
        raise ValueError("Contract seeds differ from the formal three-seed set")
    feature_manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    feature_hash = file_sha256(args.window_features)
    expected_hash = feature_manifest["outputs"]["merged_window_features"]["sha256"]
    if feature_hash != expected_hash:
        raise ValueError("Merged dynamic-texture feature hash differs from its manifest")
    if (
        feature_manifest["source_windows"] != 567
        or feature_manifest["common_valid_windows"] != 545
        or feature_manifest["masked_windows"] != 22
        or feature_manifest["descriptor_count"] != 12
    ):
        raise ValueError("Dynamic-texture feature manifest violates the frozen data assertions")
    models = _parse(args.models, MODELS)
    variants = _parse(args.variants, VARIANTS)
    seeds = _parse(args.seeds, SEEDS, int)
    if "1dcnn" in models:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Formal 1D-CNN matrix requires CUDA")
        cuda = {
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "cuda_device_name": torch.cuda.get_device_name(0),
        }
    else:
        cuda = {}
    status_path = output_root / "provenance" / "project_a_matrix_status.json"
    status: dict[str, Any] = {
        "schema_version": "project_a_dynamic_texture_matrix_status_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "working_directory": str(ROOT.resolve()),
        "models": list(models),
        "variants": list(variants),
        "seeds": list(seeds),
        "expected_runs": len(models) * len(variants) * len(seeds),
        "feature_path": str(args.window_features),
        "feature_sha256": feature_hash,
        "contract_path": str((args.contract_dir / "contract.json").resolve()),
        "contract_sha256": file_sha256(args.contract_dir / "contract.json"),
        "cuda_preflight": cuda,
        "code_sha256": {
            "dynamic_texture.py": file_sha256(
                ROOT / "src" / "real_time_ml" / "features" / "dynamic_texture.py"
            ),
            "dynamic_texture_five.py": file_sha256(
                ROOT / "src" / "real_time_ml" / "experiments" / "dynamic_texture_five.py"
            ),
            "matrix.py": file_sha256(Path(__file__).resolve()),
        },
        "runs": [],
        "ok": False,
    }
    write_json(status_path, status)
    configs: dict[tuple[str, str, int], Any] = {}
    for model in models:
        for variant in variants:
            for seed in seeds:
                payload = _config_payload(
                    output_root=output_root,
                    contract_dir=args.contract_dir,
                    features=args.window_features,
                    model=model,
                    variant=variant,
                    seed=seed,
                )
                config_path, config = _write_resolved_config(
                    output_root, payload, model, variant, seed
                )
                configs[(model, variant, seed)] = config
                status.setdefault("configs", []).append(
                    {
                        "model": model,
                        "variant": variant,
                        "seed": seed,
                        "path": str(config_path.resolve()),
                        "sha256": file_sha256(config_path),
                    }
                )
    write_json(status_path, status)

    if "classical" in models:
        for variant in variants:
            for seed in seeds:
                run_id = _run_id("classical", variant, seed)
                run_root = output_root / "runs" / run_id
                record: dict[str, Any] = {
                    "run_id": run_id,
                    "model": "classical",
                    "variant": variant,
                    "seed": seed,
                    "run_root": str(run_root.resolve()),
                }
                try:
                    if args.resume:
                        try:
                            validation = _validate_run(
                                run_root,
                                model="classical",
                                variant=variant,
                                seed=seed,
                                feature_hash=feature_hash,
                            )
                        except (FileNotFoundError, ValueError, KeyError):
                            validation = None
                        if validation is not None:
                            record.update(status="reused", elapsed_seconds=0.0, **validation)
                            status["runs"].append(record)
                            write_json(status_path, status)
                            continue
                    started = time.perf_counter()
                    log_buffer = io.StringIO()
                    with (
                        redirect_stdout(Tee(sys.stdout, log_buffer)),
                        redirect_stderr(Tee(sys.stderr, log_buffer)),
                    ):
                        run_classical_dynamic_texture(
                            configs[("classical", variant, seed)],
                            window_features=args.window_features,
                            contract_dir=args.contract_dir,
                            run_root=run_root,
                            variant=variant,
                            seed=seed,
                        )
                    _record_log(run_root, log_buffer.getvalue())
                    validation = _validate_run(
                        run_root,
                        model="classical",
                        variant=variant,
                        seed=seed,
                        feature_hash=feature_hash,
                    )
                    record.update(
                        status="ok",
                        elapsed_seconds=time.perf_counter() - started,
                        **validation,
                    )
                    status["runs"].append(record)
                    write_json(status_path, status)
                except Exception as error:
                    record.update(status="failed", error=repr(error))
                    status["runs"].append(record)
                    write_json(status_path, status)
                    write_json(run_root / "hard_failure.json", record)
                    raise

    if "1dcnn" in models:
        if set(variants) != set(VARIANTS):
            raise ValueError("Formal DCNN execution requires both paired video variants")
        for seed in seeds:
            roots = {
                variant: output_root / "runs" / _run_id("1dcnn", variant, seed)
                for variant in VARIANTS
            }
            reusable: dict[str, dict[str, Any]] = {}
            if args.resume:
                for variant in VARIANTS:
                    try:
                        reusable[variant] = _validate_run(
                            roots[variant],
                            model="1dcnn",
                            variant=variant,
                            seed=seed,
                            feature_hash=feature_hash,
                        )
                    except (FileNotFoundError, ValueError, KeyError):
                        reusable = {}
                        break
            if len(reusable) == 2:
                for variant in VARIANTS:
                    status["runs"].append(
                        {
                            "run_id": roots[variant].name,
                            "model": "1dcnn",
                            "variant": variant,
                            "seed": seed,
                            "run_root": str(roots[variant]),
                            "status": "reused",
                            "elapsed_seconds": 0.0,
                            **reusable[variant],
                        }
                    )
                write_json(status_path, status)
                continue
            started = time.perf_counter()
            log_buffer = io.StringIO()
            try:
                with (
                    redirect_stdout(Tee(sys.stdout, log_buffer)),
                    redirect_stderr(Tee(sys.stderr, log_buffer)),
                ):
                    run_paired_dcnn_dynamic_texture(
                        configs[("1dcnn", "full_five", seed)],
                        window_features=args.window_features,
                        contract_dir=args.contract_dir,
                        run_roots=roots,
                        seed=seed,
                    )
                elapsed = time.perf_counter() - started
                for variant in VARIANTS:
                    _record_log(roots[variant], log_buffer.getvalue())
                    validation = _validate_run(
                        roots[variant],
                        model="1dcnn",
                        variant=variant,
                        seed=seed,
                        feature_hash=feature_hash,
                    )
                    status["runs"].append(
                        {
                            "run_id": roots[variant].name,
                            "model": "1dcnn",
                            "variant": variant,
                            "seed": seed,
                            "run_root": str(roots[variant]),
                            "status": "ok",
                            "elapsed_seconds": elapsed,
                            **validation,
                        }
                    )
                write_json(status_path, status)
            except Exception as error:
                for variant in VARIANTS:
                    record = {
                        "run_id": roots[variant].name,
                        "model": "1dcnn",
                        "variant": variant,
                        "seed": seed,
                        "run_root": str(roots[variant]),
                        "status": "failed",
                        "error": repr(error),
                    }
                    status["runs"].append(record)
                    write_json(roots[variant] / "hard_failure.json", record)
                write_json(status_path, status)
                raise

    complete = [row for row in status["runs"] if row["status"] in {"ok", "reused"}]
    status.update(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "successful_runs": len(complete),
            "failed_runs": sum(row["status"] == "failed" for row in status["runs"]),
            "ok": len(complete) == status["expected_runs"],
        }
    )
    write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    output = DEFAULT_OUTPUT
    contract = (
        ROOT
        / "artifacts"
        / "cross_project_alignment_2026-07-16"
        / "eeg_eligible_ablation"
        / "contract"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=output)
    parser.add_argument("--contract-dir", type=Path, default=contract)
    parser.add_argument(
        "--window-features",
        type=Path,
        default=output / "features" / "project_a_window_features_dynamic_texture.csv",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=output / "features" / "dynamic_texture_manifest.json",
    )
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for name in ("output_root", "contract_dir", "window_features", "feature_manifest"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
