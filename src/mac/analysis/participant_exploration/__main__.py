from __future__ import annotations

import argparse
from pathlib import Path

from .extraction import extract
from .sources import audit
from .statistics import analyze


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post hoc nine-participant multimodal analysis; no plots or model training."
    )
    parser.add_argument("stage", choices=["audit", "extract", "analyze", "report", "verify", "run"])
    parser.add_argument("--raw-root", type=Path, default=Path("data/waterlilies"))
    parser.add_argument(
        "--thesis-root", type=Path, default=Path("/mnt/c/Users/linki/amaster/msc-thesis-log")
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("/mnt/c/Users/linki/Downloads/tables-20260621T182109Z-3-001/tables"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/supplementary/participant_multimodal_20261025"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--participants",
        nargs="+",
        help="Extraction smoke check only; final statistics always require the full cohort.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.stage in ["audit", "run"]:
        audit(args.raw_root, args.thesis_root, args.labels_root, output, args.workers)
    if args.stage in ["extract", "run"]:
        if args.stage == "run" and args.participants:
            parser.error("--participants is only supported by extract")
        extract(output, args.participants)
    if args.stage in ["analyze", "run"]:
        analyze(output)
    if args.stage in ["report", "run"]:
        from .reporting import report

        report(output)
    if args.stage in ["verify", "run"]:
        from .verification import verify

        verify(output)


if __name__ == "__main__":
    main()
