"""Auto-generate experiment reports."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def generate_report(
    experiment_name: str,
    config: dict,
    fold_results: list[dict[str, float]],
    report_dir: str,
) -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(report_dir) / f"{experiment_name}_{timestamp}.md"

    metrics = {}
    for key in fold_results[0]:
        vals = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
        if vals:
            metrics[key] = {"mean": np.mean(vals), "std": np.std(vals)}

    lines = [
        f"# Experiment Report: {experiment_name}",
        f"\n**Date:** {datetime.now().isoformat()}",
        f"\n## Aggregate Results\n",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} |")

    report = "\n".join(lines)
    path.write_text(report)
    return str(path)
