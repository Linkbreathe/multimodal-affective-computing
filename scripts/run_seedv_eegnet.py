"""SEED-V 5-class emotion recognition — EEGNet baseline (LOSO).

EEGNet (Lawhern et al., 2018) is the standard compact CNN baseline for EEG
classification. This script trains EEGNet end-to-end directly on raw
preprocessed EEG segments from the SEED-V dataset.

Data format: each .pt file contains {"eeg": Tensor[C, T], "label": int}.
  - Actual shape from preprocessing: [62, 1024] (62 channels, 1024 samples).
  - The model is parameterized to accept any (C, T).
  - EEGNet input format: [B, 1, C, T].

LOSO protocol: 16 folds, 14 train / 1 val / 1 test (same rotation as other
scripts in this repo). Validation subject is (test_subject_idx + 1) % N.

Training: AdamW + OneCycleLR + CrossEntropyLoss + AMP + early stopping on
validation balanced_accuracy.

Usage:
    python scripts/run_seedv_eegnet.py
    python scripts/run_seedv_eegnet.py --max-epochs 50 --batch-size 64 --folds 2
    python scripts/run_seedv_eegnet.py --manifest data/seedv_preprocessed/manifest.csv
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

log = logging.getLogger(__name__)

# Emotion label mapping (SEED-V)
EMOTIONS = ["Disgust", "Fear", "Sad", "Neutral", "Happy"]
NUM_CLASSES = 5


# ---------------------------------------------------------------------------
# EEGNet model
# ---------------------------------------------------------------------------

class DepthwiseConv2d(nn.Module):
    """Depthwise (channel-wise) convolution — groups == in_channels."""

    def __init__(
        self,
        in_channels: int,
        depth_multiplier: int,
        kernel_size: tuple[int, int],
        bias: bool = False,
        max_norm: float | None = 1.0,
    ) -> None:
        super().__init__()
        out_channels = in_channels * depth_multiplier
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
        )
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                norm = self.conv.weight.norm(2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
                self.conv.weight.data = self.conv.weight.data * (
                    self.max_norm / norm.clamp(min=self.max_norm)
                )
        return self.conv(x)


class SeparableConv2d(nn.Module):
    """Separable convolution = depthwise + pointwise."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=(0, kernel_size[1] // 2),
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class EEGNet(nn.Module):
    """EEGNet (Lawhern et al., 2018) for EEG-based emotion classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (C). Default 62 for SEED-V.
    n_times : int
        Number of time samples (T). Default 1024.
    n_classes : int
        Number of output classes. Default 5 for SEED-V.
    F1 : int
        Number of temporal filters in Block 1. Default 8.
    D : int
        Depth multiplier for the depthwise conv. Default 2.
    F2 : int
        Number of pointwise filters in Block 2. Default 16 (= F1 * D).
    kernel_length : int
        Length of temporal filter in Block 1 (half the sampling rate).
        Default 64 (suitable for 128 Hz data; use 32 for 256 Hz, etc.).
    dropout : float
        Dropout rate. Default 0.5.
    """

    def __init__(
        self,
        n_channels: int = 62,
        n_times: int = 1024,
        n_classes: int = 5,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times

        # --- Block 1: temporal + depthwise spatial convolution ---
        # Input: [B, 1, C, T]
        # After Conv2d(1, F1, (1, kernel_length), padding same): [B, F1, C, T]
        self.block1 = nn.Sequential(
            # Temporal filter — same-padding along time axis
            nn.Conv2d(
                1, F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
            # Depthwise spatial filter — collapses channel dim from C → 1
            DepthwiseConv2d(F1, D, kernel_size=(n_channels, 1), max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            # Pool over time by 4 → T/4
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # After block1: [B, F1*D, 1, T//4]

        # --- Block 2: separable convolution ---
        self.block2 = nn.Sequential(
            SeparableConv2d(F1 * D, F2, kernel_size=(1, 16)),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            # Pool over time by 8 → T/32
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # After block2: [B, F2, 1, T//32]
        self._flat_dim = self._compute_flat_dim(n_channels, n_times, F1, D, F2)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_dim, n_classes),
        )

        self._init_weights()

    def _compute_flat_dim(
        self, n_channels: int, n_times: int, F1: int, D: int, F2: int
    ) -> int:
        """Infer flattened dimension with a dry run."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.block1(x)
            x = self.block2(x)
            return int(x.numel())

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor[B, 1, C, T]

        Returns
        -------
        logits : Tensor[B, n_classes]
        """
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EEGSegmentDataset(Dataset):
    """Loads SEED-V preprocessed .pt segments on the fly.

    Each file contains {"eeg": Tensor[C, T], "label": int, ...}.
    The tensor is reshaped to [1, C, T] for EEGNet input.
    """

    def __init__(
        self,
        file_paths: list[Path],
        labels: list[int],
        root: Path | None = None,
    ) -> None:
        self.file_paths = file_paths
        self.labels = labels
        self.root = root  # if paths are relative, resolve against root

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fp = self.file_paths[idx]
        if self.root is not None and not fp.is_absolute():
            fp = self.root / fp
        item = torch.load(fp, map_location="cpu", weights_only=False)
        eeg = item["eeg"].float()          # [C, T]
        eeg = eeg.unsqueeze(0)             # [1, C, T]  — EEGNet input
        return eeg, self.labels[idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str | Path) -> dict[int, dict]:
    """Load preprocessed segment manifest and group by subject.

    Returns
    -------
    dict : ``{subject_id: {"files": [Path, ...], "labels": [int, ...]}}``
    """
    df = pd.read_csv(manifest_path)
    subjects: dict[int, dict] = {}
    for _, row in df.iterrows():
        sid = int(row["subject"])
        if sid not in subjects:
            subjects[sid] = {"files": [], "labels": []}
        subjects[sid]["files"].append(Path(row["file_path"]))
        subjects[sid]["labels"].append(int(row["emotion_label"]))
    return subjects


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def build_model(args: argparse.Namespace, n_channels: int, n_times: int) -> EEGNet:
    return EEGNet(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=NUM_CLASSES,
        F1=args.F1,
        D=args.D,
        F2=args.F2,
        kernel_length=args.kernel_length,
        dropout=args.dropout,
    )


def train_one_fold(
    train_files: list[Path],
    train_labels: list[int],
    val_files: list[Path],
    val_labels: list[int],
    args: argparse.Namespace,
    device: torch.device,
    root: Path,
) -> tuple[EEGNet, dict]:
    """Train EEGNet on one LOSO fold with early stopping on val balanced_acc.

    Returns the best model (by val balanced_accuracy) and training info.
    """
    # ---- Detect EEG shape from first file ----
    first = train_files[0] if train_files[0].is_absolute() else root / train_files[0]
    sample = torch.load(first, map_location="cpu", weights_only=False)
    n_channels, n_times = sample["eeg"].shape  # [C, T]

    model = build_model(args, n_channels, n_times).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"EEGNet params: {n_params:,}  |  input shape: [1, {n_channels}, {n_times}]")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_ds = EEGSegmentDataset(train_files, train_labels, root=root)
    val_ds = EEGSegmentDataset(val_files, val_labels, root=root)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_bacc = -1.0
    patience_counter = 0
    best_state: dict = {}
    stopped_epoch = args.max_epochs

    for epoch in range(args.max_epochs):
        # ---- Train ----
        model.train()
        for eeg_batch, label_batch in train_loader:
            eeg_batch = eeg_batch.to(device, non_blocking=True)
            label_batch = torch.tensor(label_batch, dtype=torch.long, device=device) \
                if not isinstance(label_batch, torch.Tensor) \
                else label_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(eeg_batch)
                loss = criterion(logits, label_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only step scheduler when the optimizer actually stepped
            # (scaler skips the step when gradients are inf/nan)
            if scaler.get_scale() >= scale_before:
                scheduler.step()

        # ---- Validate ----
        model.eval()
        val_preds: list[np.ndarray] = []
        val_true: list[np.ndarray] = []
        with torch.no_grad():
            for eeg_batch, label_batch in val_loader:
                eeg_batch = eeg_batch.to(device, non_blocking=True)
                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(eeg_batch)
                preds = logits.argmax(dim=-1).cpu().numpy()
                val_preds.append(preds)
                labels_np = (
                    label_batch.numpy()
                    if isinstance(label_batch, torch.Tensor)
                    else np.array(label_batch)
                )
                val_true.append(labels_np)

        all_preds = np.concatenate(val_preds)
        all_true = np.concatenate(val_true)
        val_bacc = balanced_accuracy_score(all_true, all_preds)

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                stopped_epoch = epoch + 1
                log.info(f"  Early stop at epoch {stopped_epoch}")
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.to(device)

    return model, {
        "best_val_bacc": best_val_bacc,
        "stopped_epoch": stopped_epoch,
        "n_params": n_params,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: EEGNet,
    files: list[Path],
    labels: list[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    root: Path,
) -> dict:
    """Evaluate EEGNet and return comprehensive metrics."""
    ds = EEGSegmentDataset(files, labels, root=root)
    loader = DataLoader(
        ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    all_preds: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        for eeg_batch, label_batch in loader:
            eeg_batch = eeg_batch.to(device, non_blocking=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(eeg_batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            labels_np = (
                label_batch.numpy()
                if isinstance(label_batch, torch.Tensor)
                else np.array(label_batch)
            )
            all_true.append(labels_np)

    preds_np = np.concatenate(all_preds)
    true_np = np.concatenate(all_true)

    return {
        "accuracy": float(accuracy_score(true_np, preds_np)),
        "balanced_accuracy": float(balanced_accuracy_score(true_np, preds_np)),
        "weighted_f1": float(f1_score(true_np, preds_np, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(true_np, preds_np, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(true_np, preds_np)),
        "confusion_matrix": confusion_matrix(
            true_np, preds_np, labels=list(range(NUM_CLASSES))
        ),
        "preds": preds_np,
        "true": true_np,
    }


# ---------------------------------------------------------------------------
# LOSO loop
# ---------------------------------------------------------------------------

def run_loso(
    args: argparse.Namespace,
    subject_data: dict[int, dict],
    output_dir: Path,
    device: torch.device,
    root: Path,
) -> list[dict]:
    """Run Leave-One-Subject-Out cross-validation with EEGNet.

    Each fold: test = subject i, val = subject (i+1)%N, train = remaining 14.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted(subject_data.keys())
    n_subjects = len(subjects)

    print(f"Subjects: {subjects}")
    print(f"Total segments: {sum(len(d['labels']) for d in subject_data.values())}")
    print(f"EEGNet params: F1={args.F1}, D={args.D}, F2={args.F2}, "
          f"kernel_length={args.kernel_length}, dropout={args.dropout}")
    print()

    folds_to_run = args.folds if args.folds > 0 else n_subjects

    results: list[dict] = []

    for fold_idx, test_subject in enumerate(tqdm(subjects[:folds_to_run], desc="LOSO folds")):
        # Reproducibility
        seed = args.seed + fold_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        val_subject = subjects[(fold_idx + 1) % n_subjects]
        train_subjects = [s for s in subjects if s != test_subject and s != val_subject]

        train_files: list[Path] = []
        train_labels: list[int] = []
        for s in train_subjects:
            train_files.extend(subject_data[s]["files"])
            train_labels.extend(subject_data[s]["labels"])

        val_files = subject_data[val_subject]["files"]
        val_labels = subject_data[val_subject]["labels"]
        test_files = subject_data[test_subject]["files"]
        test_labels = subject_data[test_subject]["labels"]

        print(f"\nFold {fold_idx + 1}/{folds_to_run} — "
              f"test={test_subject}, val={val_subject}, "
              f"train_n={len(train_labels)}, val_n={len(val_labels)}, "
              f"test_n={len(test_labels)}")

        model, train_info = train_one_fold(
            train_files, train_labels,
            val_files, val_labels,
            args, device, root,
        )

        fold_metrics = evaluate(
            model, test_files, test_labels,
            device, args.batch_size, args.num_workers, root,
        )
        fold_metrics["test_subject"] = test_subject
        fold_metrics["val_subject"] = val_subject
        fold_metrics["n_train"] = len(train_labels)
        fold_metrics["n_val"] = len(val_labels)
        fold_metrics["n_test"] = len(test_labels)
        fold_metrics["stopped_epoch"] = train_info["stopped_epoch"]
        fold_metrics["best_val_bacc"] = train_info["best_val_bacc"]
        fold_metrics["n_params"] = train_info["n_params"]
        results.append(fold_metrics)

        print(
            f"  Subject {test_subject}: "
            f"acc={fold_metrics['accuracy']:.4f}, "
            f"bacc={fold_metrics['balanced_accuracy']:.4f}, "
            f"F1_w={fold_metrics['weighted_f1']:.4f}, "
            f"F1_mac={fold_metrics['macro_f1']:.4f}, "
            f"kappa={fold_metrics['cohen_kappa']:.4f}, "
            f"epoch={train_info['stopped_epoch']}, "
            f"val_bacc={train_info['best_val_bacc']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Summary & saving
# ---------------------------------------------------------------------------

def summarize_and_save(
    results: list[dict],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Print summary statistics and save all result files."""
    metric_keys = [
        "accuracy", "balanced_accuracy", "weighted_f1", "macro_f1", "cohen_kappa"
    ]

    summary: dict[str, tuple[float, float]] = {}
    for key in metric_keys:
        vals = [r[key] for r in results]
        summary[key] = (float(np.mean(vals)), float(np.std(vals)))

    print("\n" + "=" * 65)
    print("LOSO Results Summary — EEGNet Baseline (SEED-V)")
    print("=" * 65)
    for key, (mean, std) in summary.items():
        print(f"  {key:25s}: {mean:.4f} +/- {std:.4f}")

    total_cm = sum(r["confusion_matrix"] for r in results)
    print(f"\nAggregated Confusion Matrix (rows=true, cols=pred):")
    header = f"{'':>10}" + "".join(f"{e:>10}" for e in EMOTIONS)
    print(header)
    for i_row, emo in enumerate(EMOTIONS):
        row_str = f"{emo:>10}" + "".join(f"{total_cm[i_row][j]:>10}" for j in range(NUM_CLASSES))
        print(row_str)

    # Per-fold CSV
    fold_df = pd.DataFrame([
        {
            "test_subject": r["test_subject"],
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
    fold_csv = output_dir / "loso_results.csv"
    fold_df.to_csv(fold_csv, index=False)
    print(f"\nPer-fold results saved to: {fold_csv}")

    # Summary text
    summary_txt = output_dir / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("SEED-V EEGNet Baseline — LOSO Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: EEGNet (Lawhern et al., 2018)\n")
        f.write(f"  F1={args.F1}, D={args.D}, F2={args.F2}, "
                f"kernel_length={args.kernel_length}, dropout={args.dropout}\n")
        if results:
            f.write(f"  n_params={results[0]['n_params']:,}\n")
        f.write(f"\nTraining:\n")
        f.write(f"  max_epochs={args.max_epochs}, patience={args.patience}, "
                f"lr={args.lr}, batch_size={args.batch_size}\n\n")
        f.write(f"Folds evaluated: {len(results)}\n\n")
        for key, (mean, std) in summary.items():
            f.write(f"{key:25s}: {mean:.4f} +/- {std:.4f}\n")
        f.write(f"\nAggregated Confusion Matrix (rows=true, cols=pred):\n")
        f.write(header + "\n")
        for i_row, emo in enumerate(EMOTIONS):
            row_str = (
                f"{emo:>10}"
                + "".join(f"{total_cm[i_row][j]:>10}" for j in range(NUM_CLASSES))
            )
            f.write(row_str + "\n")
    print(f"Summary saved to: {summary_txt}")

    # Confusion matrix numpy
    cm_path = output_dir / "confusion_matrix.npy"
    np.save(cm_path, total_cm)
    print(f"Confusion matrix saved to: {cm_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EEGNet baseline for SEED-V 5-class emotion recognition (LOSO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--manifest",
        default="data/seedv_preprocessed/manifest.csv",
        help="Path to manifest CSV (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/seedv_eegnet",
        help="Directory for result files.",
    )

    # LOSO
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help="Number of LOSO folds to run (0 = all 16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (incremented per fold).",
    )

    # EEGNet architecture
    parser.add_argument("--F1", type=int, default=8, help="Temporal filters in Block 1.")
    parser.add_argument("--D", type=int, default=2, help="Depth multiplier for depthwise conv.")
    parser.add_argument("--F2", type=int, default=16, help="Filters in Block 2 (default=F1*D).")
    parser.add_argument(
        "--kernel-length",
        type=int,
        default=64,
        dest="kernel_length",
        help="Temporal kernel length (half of sampling rate in samples).",
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")

    # Training
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without val improvement).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for OneCycleLR.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2, dest="weight_decay", help="AdamW weight decay."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        dest="num_workers",
        help="DataLoader worker processes.",
    )

    # Device
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (cuda/cpu). Auto-detected if omitted.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve paths relative to repo root (script lives in scripts/)
    repo_root = Path(__file__).resolve().parents[1]

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Manifest: {manifest_path}")
    print(f"Output:   {output_dir}")
    print()

    # Global seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load manifest
    subject_data = load_manifest(manifest_path)
    # Root for resolving relative .pt paths (manifest paths are relative to repo root)
    data_root = repo_root

    # Run LOSO
    results = run_loso(args, subject_data, output_dir, device, root=data_root)

    if results:
        summarize_and_save(results, output_dir, args)


if __name__ == "__main__":
    main()
