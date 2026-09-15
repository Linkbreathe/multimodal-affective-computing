"""Run the locked Windows-side RQ2 representation track."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from real_time_ml.config import load_config  # noqa: E402
from real_time_ml.windows_rq2_representations import run_windows_rq2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-root",
        default=r"C:\Users\linki\Wei\Models\rq2_shared",
        help="Shared Windows/WSL hand-off root",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "project.yaml"),
        help="Windows project configuration",
    )
    parser.add_argument(
        "--run-id",
        default="windows_rq2_fmq9_20260822",
        help="Stable Windows run identifier",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    result = run_windows_rq2(config, arguments.shared_root, arguments.run_id)
    print(f"WINDOWS_DONE: {result['bundle'].shared_root / 'windows_results' / 'WINDOWS_DONE.json'}")
    print(f"contract_hash: {result['bundle'].contract_hash}")
    print(f"representation_index_rows: {len(result['outputs']['index'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
