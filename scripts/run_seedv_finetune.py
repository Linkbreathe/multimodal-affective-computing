"""SEED-V 5-class emotion recognition with EEGPT full fine-tuning (LOSO).

End-to-end fine-tuning of the EEGPT encoder + avg-pool classification head
on raw preprocessed EEG segments. Matches EEG-FM-Bench training recipe:
  - OneCycleLR scheduler (step-level)
  - Layer-wise learning rates via get_layer_groups()
  - Warmup epochs (encoder frozen)
  - Mixed precision (AMP)
  - Gradient clipping
  - Label smoothing

LOSO protocol: 16 folds, each with 14 train / 1 val / 1 test subjects.

Usage:
    python scripts/run_seedv_finetune.py --config configs/seedv_finetune.yaml
    python scripts/run_seedv_finetune.py --config configs/seedv_finetune.yaml --max-epochs 2 --folds 1
"""
from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.encoders.eegpt import EEGPTEncoder
from src.models.eegpt_finetune_head import AvgPoolClassificationHead

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentDataset(Dataset):
    """Loads preprocessed .pt segments on the fly."""

    def __init__(self, file_paths: list[Path], labels: list[int]) -> None:
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = torch.load(self.file_paths[idx], map_location="cpu", weights_only=False)
        return item["eeg"].float(), self.labels[idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str | Path) -> dict[int, dict]:
    """Load preprocessed segment manifest and group by subject.

    Returns: {subject_id: {"files": [Path, ...], "labels": [int, ...]}}
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
# Single-fold training
# ---------------------------------------------------------------------------

def build_encoder(config: dict, device: torch.device) -> EEGPTEncoder:
    """Create EEGPT encoder with fine-tuning settings."""
    enc_cfg = config.get("encoder", {})
    weights_path = config.get(
        "eegpt_weights", "weights/eegpt/eegpt_mcae_58chs_4s_large4E.ckpt"
    )
    encoder = EEGPTEncoder(
        finetune=True,
        patch_stride=enc_cfg.get("patch_stride", 32),
        drop_rate=enc_cfg.get("drop_rate", 0.1),
        attn_drop_rate=enc_cfg.get("attn_drop_rate", 0.1),
        drop_path_rate=enc_cfg.get("drop_path_rate", 0.1),
        window_samples=enc_cfg.get("window_samples", 2560),
        weights_path=weights_path,
    )
    return encoder.to(device)


def build_head(config: dict, device: torch.device) -> AvgPoolClassificationHead:
    """Create classification head."""
    head_cfg = config.get("head", {})
    head = AvgPoolClassificationHead(
        input_dim=2048,
        num_classes=config.get("num_classes", 5),
        hidden_dims=head_cfg.get("hidden_dims", [128]),
        dropout=head_cfg.get("dropout", 0.3),
    )
    return head.to(device)


def train_one_fold(
    encoder: EEGPTEncoder,
    head: AvgPoolClassificationHead,
    train_files: list[Path],
    train_labels: list[int],
    val_files: list[Path],
    val_labels: list[int],
    config: dict,
    device: torch.device,
) -> tuple[EEGPTEncoder, AvgPoolClassificationHead, dict]:
    """Train one LOSO fold end-to-end. Returns best encoder, head, and info."""

    train_cfg = config["training"]
    ft_cfg = config.get("finetune", {})

    batch_size = train_cfg.get("batch_size", 32)
    max_epochs = train_cfg.get("max_epochs", 50)
    patience = train_cfg.get("patience", 15)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    max_lr = train_cfg["max_lr"]
    encoder_lr_scale = train_cfg.get("encoder_lr_scale", 0.1)
    gradient_clip = train_cfg.get("gradient_clip", 3.0)
    use_amp = train_cfg.get("use_amp", True)
    label_smoothing = train_cfg.get("label_smoothing", 0.1)
    lr_schedule = train_cfg.get("lr_schedule", "onecycle")
    layer_wise_lr = ft_cfg.get("layer_wise_lr", True)
    unfreeze_from_block = ft_cfg.get("unfreeze_from_block")

    # Deep copy to avoid cross-fold contamination
    encoder = copy.deepcopy(encoder)
    head = copy.deepcopy(head)

    # Freeze encoder initially (will unfreeze after warmup)
    encoder.freeze()

    # Data loaders
    train_ds = SegmentDataset(train_files, train_labels)
    val_ds = SegmentDataset(val_files, val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Build optimizer param groups
    head_params = list(head.parameters())
    param_groups = [{"params": head_params, "lr": max_lr}]

    # Encoder param groups (initially have lr set, but grad disabled during warmup)
    if layer_wise_lr:
        for group in encoder.get_layer_groups():
            param_groups.append({
                "params": group["params"],
                "lr": max_lr * encoder_lr_scale * group["lr_scale"],
            })
    else:
        param_groups.append({
            "params": list(encoder.parameters()),
            "lr": max_lr * encoder_lr_scale,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_steps = steps_per_epoch * max_epochs

    if lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in param_groups],
            total_steps=total_steps,
            pct_start=train_cfg.get("pct_start", 0.2),
        )
    elif lr_schedule == "cosine":
        warmup_steps = steps_per_epoch * warmup_epochs
        warm_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps,
        )
        cos_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm_sched, cos_sched], milestones=[warmup_steps],
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    patience_counter = 0
    best_encoder_state: dict = {}
    best_head_state: dict = {}
    best_epoch = 0

    for epoch in range(max_epochs):
        # --- Warmup / unfreeze logic ---
        if epoch < warmup_epochs:
            # Encoder frozen, only head trains
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
        elif epoch == warmup_epochs:
            # Unfreeze encoder
            if unfreeze_from_block is not None:
                encoder.unfreeze(from_layer=unfreeze_from_block)
            else:
                encoder.unfreeze()
            log.info("Epoch %d: encoder unfrozen (from_block=%s)", epoch, unfreeze_from_block)

        # --- Train ---
        head.train()
        if epoch >= warmup_epochs:
            # Keep BatchNorm layers frozen if present without disabling
            # EEGPT's dropout/drop_path regularization.
            encoder.selective_train()

        epoch_loss = 0.0
        n_correct = 0
        n_total = 0

        for batch_eeg, batch_labels in train_loader:
            batch_eeg = batch_eeg.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                features = encoder.forward_features(batch_eeg)  # [B, n_patches, 2048]
                logits = head(features)                          # [B, num_classes]
                loss = criterion(logits, batch_labels)

            scaler.scale(loss).backward()

            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"] if p.requires_grad],
                    gradient_clip,
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item() * batch_eeg.size(0)
            preds = logits.argmax(dim=-1)
            n_correct += (preds == batch_labels).sum().item()
            n_total += batch_eeg.size(0)

        train_loss = epoch_loss / max(n_total, 1)
        train_acc = n_correct / max(n_total, 1)

        # --- Validate ---
        encoder.eval()
        head.eval()
        val_preds_all = []
        val_labels_all = []

        with torch.no_grad():
            for batch_eeg, batch_labels in val_loader:
                batch_eeg = batch_eeg.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    features = encoder.forward_features(batch_eeg)
                    logits = head(features)
                val_preds_all.append(logits.argmax(dim=-1).cpu())
                val_labels_all.append(batch_labels)

        val_preds = torch.cat(val_preds_all).numpy()
        val_true = torch.cat(val_labels_all).numpy()
        val_bacc = balanced_accuracy_score(val_true, val_preds)

        # Early stopping
        if val_bacc > best_score:
            best_score = val_bacc
            patience_counter = 0
            best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best
    encoder.load_state_dict(best_encoder_state)
    head.load_state_dict(best_head_state)
    encoder.to(device)
    head.to(device)

    return encoder, head, {"best_val_bacc": best_score, "stopped_epoch": best_epoch}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    encoder: EEGPTEncoder,
    head: AvgPoolClassificationHead,
    files: list[Path],
    labels: list[int],
    device: torch.device,
    num_classes: int = 5,
    batch_size: int = 32,
    use_amp: bool = True,
) -> dict:
    """Evaluate encoder+head on a set of segments."""
    encoder.eval()
    head.eval()
    ds = SegmentDataset(files, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds = []
    with torch.no_grad():
        for batch_eeg, _ in loader:
            batch_eeg = batch_eeg.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                features = encoder.forward_features(batch_eeg)
                logits = head(features)
            all_preds.append(logits.argmax(dim=-1).cpu())

    preds = torch.cat(all_preds).numpy()
    true = np.array(labels)

    return {
        "accuracy": float(accuracy_score(true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(true, preds)),
        "weighted_f1": float(f1_score(true, preds, average="weighted")),
        "macro_f1": float(f1_score(true, preds, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(true, preds)),
        "confusion_matrix": confusion_matrix(true, preds, labels=list(range(num_classes))),
    }


# ---------------------------------------------------------------------------
# LOSO
# ---------------------------------------------------------------------------

def run_finetune_loso(
    config: dict,
    manifest_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    max_folds: int | None = None,
) -> list[dict]:
    """Run Leave-One-Subject-Out fine-tuning evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_data = load_manifest(manifest_path)
    subjects = sorted(subject_data.keys())
    num_classes = config.get("num_classes", 5)
    use_amp = config["training"].get("use_amp", True)
    batch_size = config["training"].get("batch_size", 32)

    print(f"Loaded {len(subjects)} subjects: {subjects}")
    total = sum(len(d["labels"]) for d in subject_data.values())
    print(f"Total samples: {total}")

    # Build template models once (deep-copied per fold)
    encoder_template = build_encoder(config, device)
    head_template = build_head(config, device)

    results: list[dict] = []
    fold_subjects = subjects[:max_folds] if max_folds else subjects

    for i, test_subject in enumerate(tqdm(fold_subjects, desc="LOSO folds")):
        seed = config.get("seed", 42) + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        val_subject = subjects[(i + 1) % len(subjects)]
        train_subjects = [s for s in subjects if s != test_subject and s != val_subject]

        # Gather files and labels
        train_files = [f for s in train_subjects for f in subject_data[s]["files"]]
        train_labels = [l for s in train_subjects for l in subject_data[s]["labels"]]
        val_files = subject_data[val_subject]["files"]
        val_labels = subject_data[val_subject]["labels"]
        test_files = subject_data[test_subject]["files"]
        test_labels = subject_data[test_subject]["labels"]

        # Train
        encoder, head, train_info = train_one_fold(
            encoder_template, head_template,
            train_files, train_labels,
            val_files, val_labels,
            config, device,
        )

        # Evaluate on test subject
        fold_metrics = evaluate(
            encoder, head, test_files, test_labels,
            device, num_classes, batch_size, use_amp,
        )
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

        # Free GPU memory between folds
        del encoder, head
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    metrics_keys = ["accuracy", "balanced_accuracy", "weighted_f1", "macro_f1", "cohen_kappa"]
    summary: dict[str, tuple[float, float]] = {}
    for key in metrics_keys:
        vals = [r[key] for r in results]
        summary[key] = (float(np.mean(vals)), float(np.std(vals)))

    print("\n" + "=" * 60)
    print("LOSO Results Summary (EEGPT Fine-Tuning)")
    print("=" * 60)
    for key, (mean, std) in summary.items():
        print(f"  {key:25s}: {mean:.4f} +/- {std:.4f}")

    emotions = config.get("emotions", ["Disgust", "Fear", "Sad", "Neutral", "Happy"])
    total_cm = sum(r["confusion_matrix"] for r in results)
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

    with open(output_dir / "summary.txt", "w") as f:
        f.write("SEED-V EEGPT Fine-Tuning — LOSO Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Manifest: {manifest_path}\n")
        f.write(f"Subjects: {len(subjects)}\n")
        f.write(f"Total samples: {total}\n\n")
        for key, (mean, std) in summary.items():
            f.write(f"{key:25s}: {mean:.4f} +/- {std:.4f}\n")

    np.save(output_dir / "confusion_matrix.npy", total_cm)

    print(f"\nResults saved to {output_dir}/")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SEED-V emotion classification with EEGPT full fine-tuning (LOSO)."
    )
    parser.add_argument(
        "--config", default="configs/seedv_finetune.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--manifest", default="data/seedv_preprocessed/manifest.csv",
        help="Path to preprocessed segment manifest CSV.",
    )
    parser.add_argument(
        "--output_dir", default="logs/seedv_finetune",
        help="Where to save results.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda/cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override max_epochs from config (for quick testing).",
    )
    parser.add_argument(
        "--folds", type=int, default=None,
        help="Run only the first N LOSO folds (for quick testing).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    if config.get("seed"):
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

    run_finetune_loso(
        config, args.manifest, args.output_dir, device,
        max_folds=args.folds,
    )


if __name__ == "__main__":
    main()
