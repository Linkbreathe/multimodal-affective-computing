"""Auto-generate experiment reports."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def generate_report(
    experiment_name: str,
    config: dict,
    fold_results: list[dict[str, Any]],
    report_dir: str,
    run_metadata: dict[str, Any] | None = None,
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

    if any("test_subject" in r for r in fold_results):
        lines.extend([
            "\n## Per-Fold Results\n",
            "| Test | Val | Train | N Train | N Val | N Test | Weighted F1 | CCC |",
            "|------|-----|-------|---------|-------|--------|-------------|-----|",
        ])
        for result in fold_results:
            lines.append(
                "| "
                f"{_format_cell(result.get('test_subject'))} | "
                f"{_format_cell(result.get('val_subject'))} | "
                f"{_format_cell(result.get('train_subjects'))} | "
                f"{_format_cell(result.get('n_train'))} | "
                f"{_format_cell(result.get('n_val'))} | "
                f"{_format_cell(result.get('n_test'))} | "
                f"{_format_metric(result.get('weighted_f1'))} | "
                f"{_format_metric(result.get('ccc'))} |"
            )

    if run_metadata:
        lines.extend([
            "\n## Run Metadata\n",
            "| Key | Value |",
            "|-----|-------|",
        ])
        for key, value in run_metadata.items():
            encoded = json.dumps(value, sort_keys=True, default=str)
            lines.append(f"| {key} | `{encoded}` |")

    report = "\n".join(lines)
    path.write_text(report)
    return str(path)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)
    return str(value)


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return ""
