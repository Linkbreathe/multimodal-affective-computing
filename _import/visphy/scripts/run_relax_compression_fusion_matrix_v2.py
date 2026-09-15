"""Execute the frozen July18 five-modality matrix and its fixed ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from run_relax_compression_fusion_v2 import (
    ABLATION_VARIANTS,
    ALLOWED_SEEDS,
    DEFAULT_CACHE,
    DEFAULT_CONTRACT_DIR,
    DEFAULT_OUTPUT_ROOT,
    METHODS,
    PRIMARY_METHOD,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREREGISTRATION = DEFAULT_OUTPUT_ROOT / "preregistration/method_preregistration.json"
DEFAULT_NO_EEG_ROOT = ROOT / "artifacts/relax/condition_anchor_residual_20260717/runs/no_eeg_raw_dual"
DEFAULT_HEALNET_ROOT = Path(
    "/mnt/c/Users/linki/amaster/data_collection_v3/analysis/real_time_inference/"
    "artifacts/cross_project_alignment_2026-07-16/eeg_eligible_ablation/runs/healnet/no_eeg"
)


def _run(command: list[str], log_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n[stdout]\n" + process.stdout + "\n[stderr]\n" + process.stderr,
        encoding="utf-8",
    )
    record = {
        "command": command,
        "returncode": process.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "log": str(log_path),
    }
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with code {process.returncode}; see {log_path}")
    return record


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--historical-no-eeg-root", type=Path, default=DEFAULT_NO_EEG_ROOT)
    parser.add_argument("--historical-healnet-root", type=Path, default=DEFAULT_HEALNET_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "matrix_status.json"
    # Run modality-wise first so its reusable outer-fold PCA banks are cached
    # before PLS, GCCA, and expert runs.  This changes no method definition.
    ordered_methods = (
        "modality_rank_alloc_pca12",
        "joint_block_balanced_pca12",
        "targetwise_modality_pls1",
        "linear_gcca_shared_private",
        PRIMARY_METHOD,
    )
    jobs = [(method, "full", seed) for method in ordered_methods for seed in ALLOWED_SEEDS]
    jobs.extend((PRIMARY_METHOD, variant, seed) for variant in ABLATION_VARIANTS for seed in ALLOWED_SEEDS)
    status: dict[str, Any] = {
        "schema_version": "relax_foundation_compression_fusion_matrix_v2",
        "python": sys.executable,
        "methods": list(METHODS),
        "primary": PRIMARY_METHOD,
        "ablation_variants": list(ABLATION_VARIANTS),
        "seeds": list(ALLOWED_SEEDS),
        "expected_full_runs": len(METHODS) * len(ALLOWED_SEEDS),
        "expected_ablation_runs": len(ABLATION_VARIANTS) * len(ALLOWED_SEEDS),
        "expected_runs": len(jobs),
        "records": [],
        "evaluation": None,
        "complete": False,
    }
    _write_status(status_path, status)
    if not args.evaluate_only:
        for method, variant, seed in jobs:
            run_dir = args.output_root / "runs" / method / variant / f"seed_{seed}"
            result_path = run_dir / f"{method}_{variant}_s{seed}_results.json"
            if result_path.is_file() and not args.force:
                record = {
                    "candidate": method,
                    "variant": variant,
                    "seed": seed,
                    "status": "skipped_existing",
                    "result": str(result_path),
                }
            else:
                command = [
                    sys.executable,
                    str(ROOT / "scripts/run_relax_compression_fusion_v2.py"),
                    "--method", method,
                    "--variant", variant,
                    "--seed", str(seed),
                    "--preregistration", str(args.preregistration),
                    "--embedding-cache", str(args.embedding_cache),
                    "--labels", str(args.contract_dir / "condition_labels.csv"),
                    "--windows", str(args.contract_dir / "windows.csv"),
                    "--split-manifest", str(args.contract_dir / "split_manifest.csv"),
                    "--mask-manifest", str(args.contract_dir / "common_valid_window_masks.csv"),
                    "--cohorts", str(args.contract_dir / "cohorts.json"),
                    "--compression-cache-dir", str(args.output_root / "compression_cache"),
                    "--output-dir", str(run_dir),
                ]
                record = {
                    "candidate": method,
                    "variant": variant,
                    "seed": seed,
                    "status": "completed",
                    **_run(command, run_dir / "runner.log"),
                }
            status["records"].append(record)
            _write_status(status_path, status)

    if not args.skip_evaluation:
        evaluation_dir = args.output_root / "evaluation"
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_relax_compression_fusion_v2.py"),
            "--root", str(args.output_root),
            "--preregistration", str(args.preregistration),
            "--output-dir", str(evaluation_dir),
            "--no-eeg-root", str(args.historical_no_eeg_root),
            "--healnet-root", str(args.historical_healnet_root),
        ]
        status["evaluation"] = _run(command, evaluation_dir / "evaluator.log")
    status["complete"] = True
    _write_status(status_path, status)
    print(json.dumps({"status": str(status_path), "evaluation": status["evaluation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
