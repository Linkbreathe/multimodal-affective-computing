"""2x2 ablation study: (4s vs 10s) x (all channels vs TP7+TP8) using EEGPT linear probe.

Runs LOSO classification for each of 4 conditions, generates comparison tables
and paired t-tests for statistical significance.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import ttest_rel

from auxiliary.benchmarks.seedv.scripts.run_seedv_linear_probe import run_linear_probe_loso

import torch

log = logging.getLogger(__name__)

CONDITIONS = {
    "4s_all":  "data/embeddings/seedv/base/4s_all",
    "10s_all": "data/embeddings/seedv/base/10s_all",
    "4s_tp":   "data/embeddings/seedv/base/4s_tp",
    "10s_tp":  "data/embeddings/seedv/base/10s_tp",
}

METRICS = ["accuracy", "balanced_accuracy", "weighted_f1", "macro_f1", "cohen_kappa"]


def run_ablation(
    config: dict,
    output_dir: str | Path,
    device: torch.device,
    conditions: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Run all ablation conditions and collect per-fold results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if conditions is None:
        conditions = list(CONDITIONS.keys())

    all_results: dict[str, list[dict]] = {}

    for cond_name in conditions:
        emb_dir = CONDITIONS[cond_name]
        cond_output = output_dir / cond_name

        print(f"\n{'='*60}")
        print(f"Condition: {cond_name} ({emb_dir})")
        print(f"{'='*60}")

        results = run_linear_probe_loso(config, emb_dir, cond_output, device)
        all_results[cond_name] = results

    return all_results


def generate_report(
    all_results: dict[str, list[dict]],
    output_dir: Path,
) -> str:
    """Generate a 2x2 comparison report with t-tests."""
    lines: list[str] = []

    lines.append("=" * 70)
    lines.append("EEGPT Linear Probe — 2x2 Ablation Report")
    lines.append("(Window Length) x (Channel Configuration)")
    lines.append("=" * 70)
    lines.append("")

    # ------------------------------------------------------------------
    # Per-condition summary table
    # ------------------------------------------------------------------
    lines.append("Per-Condition Summary (mean +/- std across LOSO folds):")
    lines.append("-" * 70)
    header = f"{'Condition':>12}"
    for m in METRICS:
        header += f"  {m:>18}"
    lines.append(header)
    lines.append("-" * 70)

    summaries: dict[str, dict[str, tuple[float, float]]] = {}
    for cond, results in all_results.items():
        summaries[cond] = {}
        row = f"{cond:>12}"
        for m in METRICS:
            vals = [r[m] for r in results]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            summaries[cond][m] = (mean, std)
            row += f"  {mean:.4f}+/-{std:.4f}"
        lines.append(row)

    lines.append("")

    # ------------------------------------------------------------------
    # 2x2 tables per metric
    # ------------------------------------------------------------------
    for m in METRICS:
        lines.append(f"2x2 Table — {m}:")
        lines.append(f"{'':>15} {'All Channels':>20} {'TP7+TP8':>20}")

        for window in ["4s", "10s"]:
            all_key = f"{window}_all"
            tp_key = f"{window}_tp"
            all_val = summaries.get(all_key, {}).get(m, (float("nan"), float("nan")))
            tp_val = summaries.get(tp_key, {}).get(m, (float("nan"), float("nan")))
            lines.append(
                f"{window:>15} {all_val[0]:.4f}+/-{all_val[1]:.4f}"
                f"   {tp_val[0]:.4f}+/-{tp_val[1]:.4f}"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Paired t-tests
    # ------------------------------------------------------------------
    lines.append("Paired t-tests (across LOSO folds):")
    lines.append("-" * 70)

    comparisons = [
        ("Window: 4s vs 10s (all channels)", "4s_all", "10s_all"),
        ("Window: 4s vs 10s (TP only)", "4s_tp", "10s_tp"),
        ("Channels: All vs TP (4s)", "4s_all", "4s_tp"),
        ("Channels: All vs TP (10s)", "10s_all", "10s_tp"),
    ]

    for desc, cond_a, cond_b in comparisons:
        if cond_a not in all_results or cond_b not in all_results:
            continue

        lines.append(f"\n  {desc}:")
        results_a = all_results[cond_a]
        results_b = all_results[cond_b]

        # Align by subject
        subjects_a = {r["test_subject"]: r for r in results_a}
        subjects_b = {r["test_subject"]: r for r in results_b}
        common = sorted(set(subjects_a) & set(subjects_b))

        for m in METRICS:
            vals_a = np.array([subjects_a[s][m] for s in common])
            vals_b = np.array([subjects_b[s][m] for s in common])
            diff = vals_a - vals_b

            if np.std(diff) == 0:
                lines.append(f"    {m:>25}: diff={np.mean(diff):.4f}, t=N/A, p=N/A (no variance)")
            else:
                t_stat, p_val = ttest_rel(vals_a, vals_b)
                sig = "*" if p_val < 0.05 else ""
                sig += "*" if p_val < 0.01 else ""
                sig += "*" if p_val < 0.001 else ""
                lines.append(
                    f"    {m:>25}: diff={np.mean(diff):+.4f}, "
                    f"t={t_stat:.3f}, p={p_val:.4f} {sig}"
                )

    report = "\n".join(lines)

    # Save
    report_path = output_dir / "ablation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\nReport saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="2x2 ablation: (4s vs 10s) x (all vs TP) with EEGPT linear probe.",
    )
    parser.add_argument(
        "--config",
        default="auxiliary/benchmarks/seedv/configs/seedv_linear_probe.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--output_dir", default="logs/seedv_probe_ablation",
        help="Root output directory for ablation results.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda/cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--conditions", nargs="*", default=None,
        choices=list(CONDITIONS.keys()),
        help="Subset of conditions to run (default: all 4).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    if config.get("seed"):
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

    all_results = run_ablation(config, args.output_dir, device, args.conditions)
    generate_report(all_results, Path(args.output_dir))


if __name__ == "__main__":
    main()
