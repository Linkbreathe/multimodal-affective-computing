"""SEED-V 5-class emotion recognition with EEGPT LoRA fine-tuning (LOSO).

This mirrors the existing full fine-tuning path but freezes the pretrained
EEGPT backbone and trains only injected LoRA weights plus the classification
head. Best checkpoints are saved as LoRA weights only.
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from auxiliary.benchmarks.seedv.scripts.run_seedv_finetune import (
    SegmentDataset,
    build_encoder as build_base_encoder,
    build_head,
    evaluate,
    load_manifest,
)

log = logging.getLogger(__name__)


def build_encoder(config: dict, device: torch.device):
    """Create EEGPT encoder and inject LoRA after pretrained weights load."""
    lora_cfg = config.get("lora", {})
    if not lora_cfg.get("use_lora", False):
        raise ValueError(
            "auxiliary/benchmarks/seedv/configs/seedv_lora.yaml must set lora.use_lora=true"
        )

    encoder = build_base_encoder(config, device)
    injected_modules = encoder.inject_lora(lora_cfg)
    if not injected_modules:
        raise ValueError("LoRA injection did not match any EEGPT target modules")
    return encoder


def _count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _count_named_parameters(model: nn.Module, needle: str) -> int:
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if needle in name
    )


def train_one_fold(
    encoder,
    head,
    train_files: list[Path],
    train_labels: list[int],
    val_files: list[Path],
    val_labels: list[int],
    config: dict,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict]:
    """Train one LOSO fold with frozen EEGPT base + trainable LoRA + head."""
    train_cfg = config["training"]
    lora_cfg = config["lora"]

    batch_size = train_cfg.get("batch_size", 32)
    max_epochs = train_cfg.get("max_epochs", 50)
    patience = train_cfg.get("patience", 15)
    max_lr = train_cfg["max_lr"]
    lora_lr_scale = lora_cfg.get("lora_lr_scale", 1.0)
    gradient_clip = train_cfg.get("gradient_clip", 3.0)
    use_amp = train_cfg.get("use_amp", True)
    label_smoothing = train_cfg.get("label_smoothing", 0.1)
    lr_schedule = train_cfg.get("lr_schedule", "onecycle")

    encoder = copy.deepcopy(encoder)
    head = copy.deepcopy(head)
    encoder.freeze()

    train_ds = SegmentDataset(train_files, train_labels)
    val_ds = SegmentDataset(val_files, val_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    head_params = list(head.parameters())
    lora_params = [
        param for name, param in encoder.named_parameters() if "lora_" in name and param.requires_grad
    ]
    param_groups = [
        {"params": head_params, "lr": max_lr},
        {"params": lora_params, "lr": max_lr * lora_lr_scale},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_steps = steps_per_epoch * max_epochs

    if lr_schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in param_groups],
            total_steps=total_steps,
            pct_start=train_cfg.get("pct_start", 0.2),
        )
    elif lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    patience_counter = 0
    best_lora_state: dict[str, torch.Tensor] = {}
    best_head_state: dict[str, torch.Tensor] = {}
    best_epoch = 0

    for epoch in range(max_epochs):
        encoder.selective_train()
        head.train()

        epoch_loss = 0.0
        n_correct = 0
        n_total = 0

        for batch_eeg, batch_labels in train_loader:
            batch_eeg = batch_eeg.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                features = encoder.forward_features(batch_eeg)
                logits = head(features)
                loss = criterion(logits, batch_labels)

            scaler.scale(loss).backward()

            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for group in param_groups for param in group["params"] if param.requires_grad],
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
        log.info(
            "Epoch %d/%d: train_loss=%.4f train_acc=%.4f",
            epoch + 1,
            max_epochs,
            train_loss,
            train_acc,
        )

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
        log.info("Epoch %d/%d: val_bacc=%.4f", epoch + 1, max_epochs, val_bacc)

        if val_bacc > best_score:
            best_score = val_bacc
            patience_counter = 0
            best_lora_state = {
                key: value.cpu().clone()
                for key, value in encoder.get_lora_state_dict().items()
            }
            best_head_state = {
                key: value.cpu().clone()
                for key, value in head.state_dict().items()
            }
            best_epoch = epoch + 1
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    encoder.load_lora_state_dict(best_lora_state)
    head.load_state_dict(best_head_state)
    encoder.to(device)
    head.to(device)

    return encoder, head, {
        "best_val_bacc": best_score,
        "stopped_epoch": best_epoch,
        "best_lora_state": best_lora_state,
    }


def run_finetune_loso(
    config: dict,
    manifest_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    max_folds: int | None = None,
) -> list[dict]:
    """Run Leave-One-Subject-Out LoRA fine-tuning evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_ckpt_dir = output_dir / "lora_checkpoints"
    lora_ckpt_dir.mkdir(exist_ok=True)

    subject_data = load_manifest(manifest_path)
    subjects = sorted(subject_data.keys())
    num_classes = config.get("num_classes", 5)
    use_amp = config["training"].get("use_amp", True)
    batch_size = config["training"].get("batch_size", 32)

    print(f"Loaded {len(subjects)} subjects: {subjects}")
    total = sum(len(data["labels"]) for data in subject_data.values())
    print(f"Total samples: {total}")

    encoder_template = build_encoder(config, device)
    head_template = build_head(config, device)

    total_params = _count_parameters(encoder_template) + _count_parameters(head_template)
    lora_params = _count_named_parameters(encoder_template, "lora_")
    head_params = _count_parameters(head_template)
    print(
        "Parameter summary: "
        f"total={total_params:,}, "
        f"trainable_lora={lora_params:,}, "
        f"trainable_head={head_params:,}, "
        f"trainable_total={lora_params + head_params:,}"
    )

    results: list[dict] = []
    fold_subjects = subjects[:max_folds] if max_folds else subjects

    for i, test_subject in enumerate(tqdm(fold_subjects, desc="LOSO folds")):
        seed = config.get("seed", 42) + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        val_subject = subjects[(i + 1) % len(subjects)]
        train_subjects = [s for s in subjects if s != test_subject and s != val_subject]

        train_files = [file for s in train_subjects for file in subject_data[s]["files"]]
        train_labels = [label for s in train_subjects for label in subject_data[s]["labels"]]
        val_files = subject_data[val_subject]["files"]
        val_labels = subject_data[val_subject]["labels"]
        test_files = subject_data[test_subject]["files"]
        test_labels = subject_data[test_subject]["labels"]

        encoder, head, train_info = train_one_fold(
            encoder_template,
            head_template,
            train_files,
            train_labels,
            val_files,
            val_labels,
            config,
            device,
        )

        lora_path = lora_ckpt_dir / f"subject_{int(test_subject):02d}_best_lora.pt"
        torch.save(
            {
                "lora_state_dict": train_info["best_lora_state"],
                "head_state_dict": {
                    k: v.cpu().clone() for k, v in head.state_dict().items()
                },
                "best_val_bacc": train_info["best_val_bacc"],
                "stopped_epoch": train_info["stopped_epoch"],
                "lora_config": config.get("lora", {}),
                "test_subject": test_subject,
            },
            lora_path,
        )

        fold_metrics = evaluate(
            encoder,
            head,
            test_files,
            test_labels,
            device,
            num_classes,
            batch_size,
            use_amp,
        )
        fold_metrics["test_subject"] = test_subject
        fold_metrics["val_subject"] = val_subject
        fold_metrics["n_train"] = len(train_labels)
        fold_metrics["n_val"] = len(val_labels)
        fold_metrics["n_test"] = len(test_labels)
        fold_metrics["stopped_epoch"] = train_info["stopped_epoch"]
        fold_metrics["best_val_bacc"] = train_info["best_val_bacc"]
        fold_metrics["lora_checkpoint"] = str(lora_path)
        results.append(fold_metrics)

        print(
            f"  Subject {test_subject}: "
            f"acc={fold_metrics['accuracy']:.4f}, "
            f"bacc={fold_metrics['balanced_accuracy']:.4f}, "
            f"F1_w={fold_metrics['weighted_f1']:.4f}, "
            f"kappa={fold_metrics['cohen_kappa']:.4f}, "
            f"epoch={train_info['stopped_epoch']}"
        )

        del encoder, head
        torch.cuda.empty_cache()

    metrics_keys = ["accuracy", "balanced_accuracy", "weighted_f1", "macro_f1", "cohen_kappa"]
    summary: dict[str, tuple[float, float]] = {}
    for key in metrics_keys:
        values = [result[key] for result in results]
        summary[key] = (float(np.mean(values)), float(np.std(values)))

    print("\n" + "=" * 60)
    print("LOSO Results Summary (EEGPT LoRA Fine-Tuning)")
    print("=" * 60)
    for key, (mean, std) in summary.items():
        print(f"  {key:25s}: {mean:.4f} +/- {std:.4f}")

    emotions = config.get("emotions", ["Disgust", "Fear", "Sad", "Neutral", "Happy"])
    total_cm = sum(result["confusion_matrix"] for result in results)
    print("\nAggregated Confusion Matrix:")
    print(f"{'':>10}", end="")
    for emotion in emotions:
        print(f"{emotion:>10}", end="")
    print()
    for row_idx, emotion in enumerate(emotions):
        print(f"{emotion:>10}", end="")
        for col_idx in range(len(emotions)):
            print(f"{total_cm[row_idx][col_idx]:>10}", end="")
        print()

    fold_df = pd.DataFrame(
        [
            {
                "subject": result["test_subject"],
                "val_subject": result["val_subject"],
                "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "weighted_f1": result["weighted_f1"],
                "macro_f1": result["macro_f1"],
                "cohen_kappa": result["cohen_kappa"],
                "n_train": result["n_train"],
                "n_val": result["n_val"],
                "n_test": result["n_test"],
                "stopped_epoch": result["stopped_epoch"],
                "best_val_bacc": result["best_val_bacc"],
                "lora_checkpoint": result["lora_checkpoint"],
            }
            for result in results
        ]
    )
    fold_df.to_csv(output_dir / "loso_results.csv", index=False)

    with open(output_dir / "summary.txt", "w") as handle:
        handle.write("SEED-V EEGPT LoRA Fine-Tuning — LOSO Results\n")
        handle.write("=" * 50 + "\n\n")
        handle.write(f"Manifest: {manifest_path}\n")
        handle.write(f"Subjects: {len(subjects)}\n")
        handle.write(f"Total samples: {total}\n\n")
        for key, (mean, std) in summary.items():
            handle.write(f"{key:25s}: {mean:.4f} +/- {std:.4f}\n")

    np.save(output_dir / "confusion_matrix.npy", total_cm)

    print(f"\nResults saved to {output_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SEED-V emotion classification with EEGPT LoRA fine-tuning (LOSO)."
    )
    parser.add_argument(
        "--config",
        default="auxiliary/benchmarks/seedv/configs/seedv_lora.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--manifest",
        default="data/preprocessed/seedv_preprocessed/manifest.csv",
        help="Path to preprocessed segment manifest CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default="logs/seedv_lora_finetune",
        help="Where to save results.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs from config (for quick testing).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        help="Run only the first N LOSO folds (for quick testing).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as handle:
        config = yaml.safe_load(handle)

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
        config,
        args.manifest,
        args.output_dir,
        device,
        max_folds=args.folds,
    )


if __name__ == "__main__":
    main()
