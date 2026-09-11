"""Execute the complete three-seed condition-anchor family and its evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_relax_condition_anchor_probe import ALLOWED_SEEDS, CANDIDATES


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
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/relax/condition_anchor_residual_20260717",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    if not args.evaluate_only:
        for candidate in CANDIDATES:
            for seed in ALLOWED_SEEDS:
                run_dir = args.output_root / "runs" / candidate / f"seed_{seed}"
                result_path = run_dir / f"{candidate}_s{seed}_results.json"
                if result_path.is_file() and not args.force:
                    records.append(
                        {
                            "candidate": candidate,
                            "seed": seed,
                            "status": "skipped_existing",
                            "result": str(result_path),
                        }
                    )
                    continue
                command = [
                    sys.executable,
                    str(ROOT / "scripts/run_relax_condition_anchor_probe.py"),
                    "--candidate",
                    candidate,
                    "--seed",
                    str(seed),
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
                records.append({"candidate": candidate, "seed": seed, "status": "completed", **record})

    evaluation_dir = args.output_root / "evaluation"
    evaluation_command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_relax_condition_anchor.py"),
        "--runs-dir",
        str(args.output_root / "runs"),
        "--labels",
        str(args.contract_dir / "condition_labels.csv"),
        "--split-manifest",
        str(args.contract_dir / "split_manifest.csv"),
        "--output-dir",
        str(evaluation_dir),
    ]
    evaluation = _run(evaluation_command, evaluation_dir / "evaluator.log")
    status = {
        "schema_version": "relax_condition_anchor_matrix_v1",
        "python": sys.executable,
        "candidate_count": len(CANDIDATES),
        "seeds": list(ALLOWED_SEEDS),
        "expected_runs": len(CANDIDATES) * len(ALLOWED_SEEDS),
        "records": records,
        "evaluation": evaluation,
    }
    status_path = args.output_root / "matrix_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": str(status_path), "evaluation": str(evaluation_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
