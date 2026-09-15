"""SEED-V EEGPT 2x2 Ablation Study: Window Length x Channel Configuration.

Orchestrates preprocessing, embedding extraction, LOSO classification,
and statistical comparison across 4 conditions:

  A) 4s_all  -- 4-second windows, all 58 channels
  B) 10s_all -- 10-second windows, all 58 channels
  C) 4s_tp   -- 4-second windows, TP7+TP8 only
  D) 10s_tp  -- 10-second windows, TP7+TP8 only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so local imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_seedv_embeddings import extract_embeddings  # noqa: E402
from scripts.run_seedv_experiment import load_embeddings, run_loso  # noqa: E402
from src.data.seedv_preprocessing import preprocess_seedv  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------
CONDITIONS = [
    {"name": "4s_all", "window_sec": 4, "channels": None},
    {"name": "10s_all", "window_sec": 10, "channels": None},
    {"name": "4s_tp", "window_sec": 4, "channels": ["TP7", "TP8"]},
    {"name": "10s_tp", "window_sec": 10, "channels": ["TP7", "TP8"]},
]

SAMPLING_RATE = 256


def _make_config(condition_name: str) -> dict:
    """Build a run_loso-compatible config dict for a given condition."""
    return {
        "embeddings_dir": f"data/embeddings/seedv/base/{condition_name}",
        "num_classes": 5,
        "emotions": ["Disgust", "Fear", "Sad", "Neutral", "Happy"],
        "fusion": {"d_common": 256, "dropout": 0.1},
        "training": {
            "batch_size": 64,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "max_epochs": 100,
            "patience": 10,
        },
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Phase 1 -- Preprocessing
# ---------------------------------------------------------------------------
def phase_preprocess(
    window_secs: list[int], data_dir: str | None = None, skip: bool = False,
) -> None:
    """Ensure preprocessed data exists for each unique window length."""
    for ws in sorted(set(window_secs)):
        out_dir = Path(f"data/preprocessed/seedv_preprocessed_{ws}s")
        manifest = out_dir / "manifest.csv"
        if manifest.exists():
            log.info("Preprocessed data already exists: %s", manifest)
            continue
        if skip:
            log.warning(
                "Preprocessed data missing for %ds and --skip_preprocess set; skipping.",
                ws,
            )
            continue
        if data_dir and not Path(data_dir).is_dir():
            raise FileNotFoundError(
                f"Raw SEED-V data directory not found: {data_dir}. "
                "Pass --data_dir pointing to the SEED-V root."
            )
        kwargs: dict = {"window_sec": ws, "output_dir": str(out_dir)}
        if data_dir:
            kwargs["data_dir"] = data_dir
        log.info("Preprocessing SEED-V with window_sec=%d -> %s", ws, out_dir)
        preprocess_seedv(**kwargs)


# ---------------------------------------------------------------------------
# Phase 2 -- Embedding extraction
# ---------------------------------------------------------------------------
def phase_extract(conditions: list[dict], skip: bool = False) -> None:
    """Extract EEGPT embeddings for each condition."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cond in conditions:
        name = cond["name"]
        ws = cond["window_sec"]
        channels = cond["channels"]
        window_samples = ws * SAMPLING_RATE

        emb_dir = Path(f"data/embeddings/seedv/base/{name}")
        emb_manifest = emb_dir / "manifest.csv"
        if emb_manifest.exists():
            log.info("Embeddings already exist: %s", emb_manifest)
            continue
        if skip:
            log.warning(
                "Embeddings missing for %s and --skip_extract set; skipping.", name
            )
            continue

        preproc_manifest = Path(f"data/preprocessed/seedv_preprocessed_{ws}s/manifest.csv")
        if not preproc_manifest.exists():
            raise FileNotFoundError(
                f"Preprocessed manifest not found: {preproc_manifest}. "
                "Run without --skip_preprocess first."
            )

        log.info(
            "Extracting embeddings for condition '%s' (window=%ds, channels=%s)",
            name,
            ws,
            channels or "all",
        )
        extract_embeddings(
            manifest_path=str(preproc_manifest),
            output_dir=str(emb_dir),
            device=device,
            channels=channels,
            window_samples=window_samples,
        )


# ---------------------------------------------------------------------------
# Phase 3 -- LOSO classification
# ---------------------------------------------------------------------------
def phase_classify(conditions: list[dict]) -> dict[str, list[dict]]:
    """Run LOSO for each condition, return per-condition fold results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results: dict[str, list[dict]] = {}

    for cond in conditions:
        name = cond["name"]
        log.info("Running LOSO classification for condition '%s'", name)

        # Reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        config = _make_config(name)
        fold_results = run_loso(config, device)
        all_results[name] = fold_results

    return all_results


# ---------------------------------------------------------------------------
# Phase 4 -- Report generation
# ---------------------------------------------------------------------------
def _fold_metric(results: list[dict], metric: str) -> np.ndarray:
    """Extract a per-fold metric array."""
    return np.array([r[metric] for r in results])


def phase_report(all_results: dict[str, list[dict]]) -> None:
    """Generate the comparison report and per-condition CSVs."""
    out_dir = Path("logs/seedv_eegpt_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-condition LOSO CSVs ---
    cond_stats: dict[str, dict] = {}
    for name, folds in all_results.items():
        df = pd.DataFrame(
            [
                {
                    "subject": r["test_subject"],
                    "accuracy": r["accuracy"],
                    "f1_weighted": r["f1_weighted"],
                    "f1_macro": r["f1_macro"],
                    "n_test": r["n_test"],
                }
                for r in folds
            ]
        )
        df.to_csv(out_dir / f"{name}_loso.csv", index=False)

        accs = _fold_metric(folds, "accuracy")
        f1ws = _fold_metric(folds, "f1_weighted")
        n_total = sum(r["n_test"] for r in folds)
        cond_stats[name] = {
            "acc_mean": accs.mean(),
            "acc_std": accs.std(),
            "f1w_mean": f1ws.mean(),
            "f1w_std": f1ws.std(),
            "n_segments": n_total,
            "accs": accs,
            "f1ws": f1ws,
        }

    # --- Build report text ---
    lines: list[str] = []
    lines.append("SEED-V EEGPT Ablation Study: Window Length x Channel Configuration")
    lines.append("=" * 67)
    lines.append("")
    lines.append("Condition Results:")

    for name in ["4s_all", "10s_all", "4s_tp", "10s_tp"]:
        if name not in cond_stats:
            continue
        s = cond_stats[name]
        lines.append(
            f"  {name + ':':10s} Acc={s['acc_mean']:.4f}\u00b1{s['acc_std']:.4f}  "
            f"F1_w={s['f1w_mean']:.4f}\u00b1{s['f1w_std']:.4f}  "
            f"({s['n_segments']} segments)"
        )

    # --- 2x2 Accuracy table ---
    lines.append("")
    lines.append("2x2 Accuracy Table:")
    lines.append(f"{'':14s}{'4s':>10s}{'10s':>12s}")
    for ch_label, sfx in [("All ch", "_all"), ("TP7/8", "_tp")]:
        row = f"{ch_label:14s}"
        for ws_pfx in ["4s", "10s"]:
            key = f"{ws_pfx}{sfx}"
            if key in cond_stats:
                row += f"{cond_stats[key]['acc_mean']:>10.4f}  "
            else:
                row += f"{'N/A':>10s}  "
        lines.append(row)

    # --- 2x2 F1 table ---
    lines.append("")
    lines.append("2x2 F1 Weighted Table:")
    lines.append(f"{'':14s}{'4s':>10s}{'10s':>12s}")
    for ch_label, sfx in [("All ch", "_all"), ("TP7/8", "_tp")]:
        row = f"{ch_label:14s}"
        for ws_pfx in ["4s", "10s"]:
            key = f"{ws_pfx}{sfx}"
            if key in cond_stats:
                row += f"{cond_stats[key]['f1w_mean']:>10.4f}  "
            else:
                row += f"{'N/A':>10s}  "
        lines.append(row)

    # --- Statistical tests ---
    lines.append("")
    lines.append("Statistical Tests (paired t-test, 16 LOSO folds):")

    comparisons = [
        ("4s vs 10s (all ch)", "4s_all", "10s_all"),
        ("4s vs 10s (TP7/8)", "4s_tp", "10s_tp"),
        ("All vs TP7/8 (4s)", "4s_all", "4s_tp"),
        ("All vs TP7/8 (10s)", "10s_all", "10s_tp"),
    ]
    for label, a, b in comparisons:
        if a in cond_stats and b in cond_stats:
            t_stat, p_val = ttest_rel(cond_stats[a]["accs"], cond_stats[b]["accs"])
            lines.append(f"  {label + ':':25s} t={t_stat:.2f}, p={p_val:.4f}")
        else:
            lines.append(f"  {label + ':':25s} N/A (missing condition)")

    report_text = "\n".join(lines) + "\n"

    # --- Write report ---
    report_path = out_dir / "ablation_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    log.info("Ablation report saved to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEED-V EEGPT 2x2 Ablation: Window Length x Channel Configuration"
    )
    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip preprocessing even if data is missing.",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Skip embedding extraction even if data is missing.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Run only these conditions (e.g., 4s_all 10s_tp). Default: all 4.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to raw SEED-V root (containing EEG_raw/). "
             "Defaults to /mnt/c/Users/Public/Data/SEED-V/SEED-V if omitted.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (cuda / cpu). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Filter conditions if requested
    valid_names = {c["name"] for c in CONDITIONS}
    if args.conditions:
        for c in args.conditions:
            if c not in valid_names:
                parser.error(f"Unknown condition '{c}'. Valid: {sorted(valid_names)}")
        conditions = [c for c in CONDITIONS if c["name"] in args.conditions]
    else:
        conditions = CONDITIONS

    log.info(
        "Ablation conditions: %s", [c["name"] for c in conditions]
    )

    # Phase 1: Preprocessing
    window_secs = [c["window_sec"] for c in conditions]
    phase_preprocess(window_secs, data_dir=args.data_dir, skip=args.skip_preprocess)

    # Phase 2: Embedding extraction
    phase_extract(conditions, skip=args.skip_extract)

    # Phase 3: LOSO classification
    all_results = phase_classify(conditions)

    # Phase 4: Report
    phase_report(all_results)


if __name__ == "__main__":
    main()
