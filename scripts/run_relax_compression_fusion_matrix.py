"""Execute the preregistered six-candidate, three-seed compression/fusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_relax_compression_fusion import ALLOWED_SEEDS, METHODS
from evaluate_relax_compression_fusion import DEFAULT_HEALNET_ROOT, DEFAULT_NO_EEG_ROOT


ROOT = Path(__file__).resolve().parent.parent


def _run(command: list[str], log_path: Path) -> dict[str, object]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=ROOT / "artifacts/relax/aligned_20260716/condition_embeddings.pt",
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT
        / "artifacts/relax/foundation_compression_fusion_20260717/preregistration/method_preregistration.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/relax/foundation_compression_fusion_20260717",
    )
    parser.add_argument(
        "--historical-no-eeg-root", type=Path, default=DEFAULT_NO_EEG_ROOT
    )
    parser.add_argument(
        "--historical-healnet-root", type=Path, default=DEFAULT_HEALNET_ROOT
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    if not args.evaluate_only:
        for method in METHODS:
            for seed in ALLOWED_SEEDS:
                run_dir = args.output_root / "runs" / method / f"seed_{seed}"
                result_path = run_dir / f"{method}_s{seed}_results.json"
                if result_path.is_file() and not args.force:
                    records.append(
                        {
                            "candidate": method,
                            "seed": seed,
                            "status": "skipped_existing",
                            "result": str(result_path),
                        }
                    )
                    continue
                command = [
                    sys.executable,
                    str(ROOT / "scripts/run_relax_compression_fusion.py"),
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--preregistration",
                    str(args.preregistration),
                    "--embedding-cache",
                    str(args.embedding_cache),
                    "--labels",
                    str(args.contract_dir / "condition_labels.csv"),
                    "--windows",
                    str(args.contract_dir / "windows.csv"),
                    "--split-manifest",
                    str(args.contract_dir / "split_manifest.csv"),
                    "--mask-manifest",
                    str(args.contract_dir / "common_valid_window_masks.csv"),
                    "--cohorts",
                    str(args.contract_dir / "cohorts.json"),
                    "--output-dir",
                    str(run_dir),
                ]
                record = _run(command, run_dir / "runner.log")
                records.append(
                    {"candidate": method, "seed": seed, "status": "completed", **record}
                )

    evaluation: dict[str, object] | None = None
    if not args.skip_evaluation:
        evaluation_dir = args.output_root / "evaluation"
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_relax_compression_fusion.py"),
            "--runs-dir",
            str(args.output_root / "runs"),
            "--labels",
            str(args.contract_dir / "condition_labels.csv"),
            "--split-manifest",
            str(args.contract_dir / "split_manifest.csv"),
            "--preregistration",
            str(args.preregistration),
            "--historical-no-eeg-root",
            str(args.historical_no_eeg_root),
            "--historical-healnet-root",
            str(args.historical_healnet_root),
            "--output-dir",
            str(evaluation_dir),
            "--force",
        ]
        evaluation = _run(command, evaluation_dir / "evaluator.log")

    status = {
        "schema_version": "relax_foundation_compression_fusion_matrix_v1",
        "python": sys.executable,
        "methods": list(METHODS),
        "seeds": list(ALLOWED_SEEDS),
        "expected_runs": len(METHODS) * len(ALLOWED_SEEDS),
        "records": records,
        "evaluation": evaluation,
    }
    status_path = args.output_root / "matrix_status.json"
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": str(status_path), "evaluation": evaluation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
