"""SEED-V 5-class emotion classification using EEGPT official linear probe (LOSO).

Uses the official EEGPT two-layer LinearWithConstraint probe architecture,
adapted for pre-extracted mean-pooled embeddings [B, 2048].

LOSO protocol: 16 folds, each with 14 train / 1 val / 1 test subjects.
Training recipe matches official EEGPT: AdamW + OneCycleLR + CrossEntropyLoss.

Note: EEGPT embeddings require z-score normalization per feature dimension.
The frozen encoder produces near-constant output vectors (cosine sim ~1.0
across all samples), but per-dimension variation after normalization may
carry discriminative signal.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.eegpt_linear_probe import EEGPTLinearProbe

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (reuses logic from run_seedv_experiment.py)
# ---------------------------------------------------------------------------

def load_embeddings(
    embeddings_dir: str | Path,
) -> dict[int, dict[str, torch.Tensor]]:
    """Load all embeddings and group by subject.

    Returns
    -------
    dict : ``{subject_id: {"embeddings": Tensor[N, 2048], "labels": Tensor[N]}}``
    """
    embeddings_dir = Path(embeddings_dir) / "eegpt_eeg"
    subject_data: dict[int, dict[str, torch.Tensor]] = {}

    for subject_dir in sorted(embeddings_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject_id = int(subject_dir.name)
        embs: list[torch.Tensor] = []
        labels: list[int] = []
        for pt_file in sorted(subject_dir.glob("segment_*.pt")):
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            embs.append(data["embedding"])
            labels.append(data["label"])
        if embs:
            subject_data[subject_id] = {
                "embeddings": torch.stack(embs),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    return subject_data


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------

def train_one_fold(
    train_embs: torch.Tensor,
    train_labels: torch.Tensor,
    val_embs: torch.Tensor,
    val_labels: torch.Tensor,
    config: dict,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Train one LOSO fold with validation-based early stopping.

    Returns the best model and validation metrics at the best epoch.
    """
    probe_cfg = config.get("probe", {})
    train_cfg = config["training"]

    model = EEGPTLinearProbe(
        num_classes=config.get("num_classes", 5),
        input_dim=probe_cfg.get("input_dim", 2048),
        bottleneck_dim=probe_cfg.get("bottleneck_dim", 16),
        dropout=probe_cfg.get("dropout", 0.5),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    batch_size = train_cfg.get("batch_size", 64)
    max_epochs = train_cfg.get("max_epochs", 100)
    patience = train_cfg.get("patience", 15)

    train_ds = TensorDataset(train_embs.to(device), train_labels.to(device))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=train_cfg["lr"],
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=train_cfg.get("pct_start", 0.2),
    )

    criterion = nn.CrossEntropyLoss()

    # Validation data on device
    val_embs_dev = val_embs.to(device)
    val_labels_np = val_labels.numpy()

    best_score = -1.0
    patience_counter = 0
    best_state: dict = {}

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        for batch_embs, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_embs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_logits = model(val_embs_dev)
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()

        val_bacc = balanced_accuracy_score(val_labels_np, val_preds)

        if val_bacc > best_score:
            best_score = val_bacc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best model
    model.load_state_dict(best_state)
    model.to(device)
    return model, {"best_val_bacc": best_score, "stopped_epoch": epoch + 1}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    embs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    num_classes: int = 5,
) -> dict:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    with torch.no_grad():
        logits = model(embs.to(device))
        preds = logits.argmax(dim=-1).cpu().numpy()

    true = labels.numpy()
    return {
        "accuracy": float(accuracy_score(true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(true, preds)),
        "weighted_f1": float(f1_score(true, preds, average="weighted")),
        "macro_f1": float(f1_score(true, preds, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(true, preds)),
        "confusion_matrix": confusion_matrix(true, preds, labels=list(range(num_classes))),
        "preds": preds,
        "true": true,
    }


# ---------------------------------------------------------------------------
# LOSO
# ---------------------------------------------------------------------------

def run_linear_probe_loso(
    config: dict,
    embeddings_dir: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> list[dict]:
    """Run Leave-One-Subject-Out evaluation with the EEGPT linear probe.

    Parameters
    ----------
    config : dict
        Full config (from YAML).
    embeddings_dir : str | Path
        Directory containing eegpt_eeg/{subject_id}/segment_*.pt files.
    output_dir : str | Path
        Where to save per-fold results and summary.
    device : torch.device
        Compute device.

    Returns
    -------
    list[dict] : per-fold results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_data = load_embeddings(embeddings_dir)
    subjects = sorted(subject_data.keys())
    num_classes = config.get("num_classes", 5)

    print(f"Loaded {len(subjects)} subjects: {subjects}")
    total = sum(d["embeddings"].shape[0] for d in subject_data.values())
    print(f"Total samples: {total}")

    results: list[dict] = []

    for i, test_subject in enumerate(tqdm(subjects, desc="LOSO folds")):
        # Seed per fold for reproducibility
        seed = config.get("seed", 42) + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Validation subject: next in sorted order, wrapping around
        val_subject = subjects[(i + 1) % len(subjects)]

        train_subjects = [s for s in subjects if s != test_subject and s != val_subject]

        train_embs = torch.cat([subject_data[s]["embeddings"] for s in train_subjects])
        train_labels = torch.cat([subject_data[s]["labels"] for s in train_subjects])
        val_embs = subject_data[val_subject]["embeddings"]
        val_labels = subject_data[val_subject]["labels"]
        test_embs = subject_data[test_subject]["embeddings"]
        test_labels = subject_data[test_subject]["labels"]

        # Z-score normalization (fit on train, apply to val/test)
        train_mean = train_embs.mean(dim=0, keepdim=True)
        train_std = train_embs.std(dim=0, keepdim=True).clamp(min=1e-8)
        train_embs = (train_embs - train_mean) / train_std
        val_embs = (val_embs - train_mean) / train_std
        test_embs = (test_embs - train_mean) / train_std

        # Train
        model, train_info = train_one_fold(
            train_embs, train_labels, val_embs, val_labels, config, device,
        )

        # Evaluate on test subject
        fold_metrics = evaluate(model, test_embs, test_labels, device, num_classes)
        fold_metrics["test_subject"] = test_subject
        fold_metrics["val_subject"] = val_subject
        fold_metrics["n_train"] = len(train_labels)
        fold_metrics["n_val"] = len(val_labels)
        fold_metrics["n_test"] = len(test_labels)
        fold_metrics["stopped_epoch"] = train_info["stopped_epoch"]
        fold_metrics["best_val_bacc"] = train_info["best_val_bacc"]
        results.append(fold_metrics)

        print(
            f"  Subject {test_subject}: "
            f"acc={fold_metrics['accuracy']:.4f}, "
            f"bacc={fold_metrics['balanced_accuracy']:.4f}, "
            f"F1_w={fold_metrics['weighted_f1']:.4f}, "
            f"kappa={fold_metrics['cohen_kappa']:.4f}, "
            f"epoch={train_info['stopped_epoch']}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    metrics_keys = ["accuracy", "balanced_accuracy", "weighted_f1", "macro_f1", "cohen_kappa"]
    summary: dict[str, tuple[float, float]] = {}
    for key in metrics_keys:
        vals = [r[key] for r in results]
        summary[key] = (float(np.mean(vals)), float(np.std(vals)))

    print("\n" + "=" * 60)
    print("LOSO Results Summary (EEGPT Linear Probe)")
    print("=" * 60)
    for key, (mean, std) in summary.items():
        print(f"  {key:25s}: {mean:.4f} +/- {std:.4f}")

    # Per-class from aggregated confusion matrix
    total_cm = sum(r["confusion_matrix"] for r in results)
    emotions = config.get("emotions", ["Disgust", "Fear", "Sad", "Neutral", "Happy"])
    print(f"\nAggregated Confusion Matrix:")
    print(f"{'':>10}", end="")
    for e in emotions:
        print(f"{e:>10}", end="")
    print()
    for i_row, e in enumerate(emotions):
        print(f"{e:>10}", end="")
        for j_col in range(len(emotions)):
            print(f"{total_cm[i_row][j_col]:>10}", end="")
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    # Per-fold CSV
    fold_df = pd.DataFrame([
        {
            "subject": r["test_subject"],
            "val_subject": r["val_subject"],
            "accuracy": r["accuracy"],
            "balanced_accuracy": r["balanced_accuracy"],
            "weighted_f1": r["weighted_f1"],
            "macro_f1": r["macro_f1"],
            "cohen_kappa": r["cohen_kappa"],
            "n_train": r["n_train"],
            "n_val": r["n_val"],
            "n_test": r["n_test"],
            "stopped_epoch": r["stopped_epoch"],
            "best_val_bacc": r["best_val_bacc"],
        }
        for r in results
    ])
    fold_df.to_csv(output_dir / "loso_results.csv", index=False)

    # Summary text
    with open(output_dir / "summary.txt", "w") as f:
        f.write("SEED-V EEGPT Linear Probe — LOSO Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Embeddings: {embeddings_dir}\n")
        f.write(f"Subjects: {len(subjects)}\n")
        f.write(f"Total samples: {total}\n\n")
        for key, (mean, std) in summary.items():
            f.write(f"{key:25s}: {mean:.4f} +/- {std:.4f}\n")
        f.write(f"\nAggregated Confusion Matrix:\n")
        f.write(f"{'':>10}")
        for e in emotions:
            f.write(f"{e:>10}")
        f.write("\n")
        for i_row, e in enumerate(emotions):
            f.write(f"{e:>10}")
            for j_col in range(len(emotions)):
                f.write(f"{total_cm[i_row][j_col]:>10}")
            f.write("\n")

    # Save confusion matrix as numpy
    np.save(output_dir / "confusion_matrix.npy", total_cm)

    print(f"\nResults saved to {output_dir}/")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SEED-V emotion classification with EEGPT official linear probe (LOSO)."
    )
    parser.add_argument(
        "--config", default="configs/seedv_linear_probe.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--embeddings_dir", default="data/embeddings_seedv",
        help="Directory with extracted EEGPT embeddings "
             "(expects {condition}/eegpt_eeg/ subdirectories).",
    )
    parser.add_argument(
        "--output_dir", default="logs/seedv_probe",
        help="Where to save results.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda/cpu). Auto-detected if omitted.",
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

    run_linear_probe_loso(config, args.embeddings_dir, args.output_dir, device)


if __name__ == "__main__":
    main()
