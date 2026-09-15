"""Analyze and compare all experiment results."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked as `python scripts/analyze_results.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from mac.reporting.registry import ResultsRegistry


def main() -> None:
    registry = ResultsRegistry()
    results = registry.get_all()

    if not results:
        print("No experiments found in registry.")
        return

    df = pd.DataFrame(results)

    print("\n=== Experiment Comparison ===\n")
    display_cols = [c for c in ["experiment_name", "fusion_type", "weighted_f1", "ccc"] if c in df.columns]
    print(df[display_cols].to_string(index=False))

    if "weighted_f1" in df.columns:
        print("\n=== Best by Weighted F1 ===")
        best = df.loc[df["weighted_f1"].idxmax()]
        print(f"  {best['experiment_name']}: F1={best['weighted_f1']:.4f}")

    if "ccc" in df.columns:
        print("\n=== Best by CCC ===")
        best_ccc = df.loc[df["ccc"].idxmax()]
        print(f"  {best_ccc['experiment_name']}: CCC={best_ccc['ccc']:.4f}")


if __name__ == "__main__":
    main()
