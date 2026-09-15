#!/usr/bin/env python
"""Phase 0/1 strict audit and cohort freezing for Relax foundation probes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.relax_foundation import (  # noqa: E402
    PARTICIPANTS,
    RelaxHardFailure,
    build_cohorts,
    build_qc_manifest,
    validate_phase0_inputs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relax-run-dir", required=True, help="Relax-Model run/artifacts directory.")
    parser.add_argument("--raw-root", default=None, help="Raw Relax data root for source-file and head-field checks.")
    parser.add_argument("--labels-root", default=None, help="Directory containing Painting Reflection workbook.")
    parser.add_argument("--eeg-weights", default="brain-bzh/reve-large")
    parser.add_argument("--eeg-pos-bank", default="brain-bzh/reve-positions")
    parser.add_argument("--participants", nargs="*", default=list(PARTICIPANTS))
    parser.add_argument("--output-dir", default="artifacts/relax", help="Where audit manifests are written.")
    parser.add_argument("--write-cohorts", action="store_true", help="Write locked cohorts.json.")
    parser.add_argument("--high-quality-min-count", type=int, default=4)
    parser.add_argument("--high-quality-max-count", type=int, default=6)
    parser.add_argument("--strict", action="store_true", help="Kept for explicit CLI parity; audits are always strict.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        phase0 = validate_phase0_inputs(
            args.relax_run_dir,
            raw_root=args.raw_root,
            labels_root=args.labels_root,
            eeg_weights=args.eeg_weights,
            eeg_pos_bank=args.eeg_pos_bank,
            participants=args.participants,
            require_reve=True,
        )
        write_json(out / "phase0_audit.json", phase0)
        qc_manifest = build_qc_manifest(args.relax_run_dir, participants=args.participants)
        write_json(out / "qc_manifest.json", qc_manifest)
        if args.write_cohorts:
            cohorts_path = out / "cohorts.json"
            if cohorts_path.exists():
                raise RelaxHardFailure(
                    f"{cohorts_path} already exists. Refusing to rewrite locked cohorts."
                )
            cohorts = build_cohorts(
                qc_manifest,
                high_quality_min_count=args.high_quality_min_count,
                high_quality_max_count=args.high_quality_max_count,
            )
            write_json(cohorts_path, cohorts)
        return 0
    except RelaxHardFailure as error:
        failure = {
            "status": "failed",
            "error": str(error),
            "relax_run_dir": str(args.relax_run_dir),
            "raw_root": str(args.raw_root) if args.raw_root else None,
            "labels_root": str(args.labels_root) if args.labels_root else None,
        }
        write_json(out / "phase0_hard_failures.json", failure)
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
