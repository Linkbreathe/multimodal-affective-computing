"""SEED-V emotion classification with REVE — official two-stage fine-tuning (LOSO).

Ports the official REVE downstream protocol (reve_eeg/src/downstream_tasks/train_core.py)
into our repo, adapted for SEED-V LOSO evaluation:

  Stage 1 — Linear Probing:
    - Encoder frozen, only `linear_head` + `cls_query_token` trainable.
    - Mixup + AMP + ReduceLROnPlateau on val balanced accuracy.

  Stage 2 — LoRA Fine-Tuning:
    - Inject LoRA into REVE attention modules (to_qkv + to_out).
    - Backbone frozen, LoRA + classification head trainable.
    - Same optimizer/scheduler/mixup recipe.

Online resamples 256 Hz preprocessed segments to 200 Hz (REVE's native rate).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
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

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from mac.encoders.reve import ReveEncoder
from mac.models.reve_classifier import (
    ReveClassifier,
    freeze_for_linear_probe,
    unfreeze_all,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset (with online resample 256 -> 200 Hz)
# ---------------------------------------------------------------------------

class SegmentDataset(Dataset):
    """Loads preprocessed .pt segments and resamples to REVE's native rate."""

    def __init__(
        self,
        file_paths: list[Path],
        labels: list[int],
        src_sr: int = 256,
        tgt_sr: int = 200,
    ) -> None:
        self.file_paths = file_paths
        self.labels = labels
        self.src_sr = src_sr
        self.tgt_sr = tgt_sr

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = torch.load(self.file_paths[idx], map_location="cpu", weights_only=False)
        eeg = item["eeg"].float()  # [60, 2560] at 256 Hz
        if self.src_sr != self.tgt_sr:
            eeg = AF.resample(eeg, self.src_sr, self.tgt_sr)  # [60, 2000] at 200 Hz
        return eeg, int(self.labels[idx])


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str | Path) -> dict[int, dict]:
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
# Builders
# ---------------------------------------------------------------------------

def build_classifier(config: dict, dropout: float, device: torch.device) -> ReveClassifier:
    encoder = ReveEncoder(
        weights_path=config["reve_weights"],
        pos_bank_path=config["reve_pos_bank"],
        finetune=False,  # frozen by default; will be unfrozen for stage 2
    )
    classifier = ReveClassifier(
        encoder=encoder,
        n_classes=config.get("num_classes", 5),
        dropout=dropout,
        pooling=config.get("classifier", {}).get("pooling", "last"),
    )
    return classifier.to(device)


# ---------------------------------------------------------------------------
# Training utilities (mirrors official train_core.py)
# ---------------------------------------------------------------------------

def mixup_loss(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    use_mixup: bool,
) -> torch.Tensor:
    if use_mixup:
        mm = random.random()
        perm = torch.randperm(data.size(0), device=data.device)
        mixed = mm * data + (1 - mm) * data[perm]
        output = model(mixed)
        loss = mm * F.cross_entropy(output, target) + (1 - mm) * F.cross_entropy(output, target[perm])
    else:
        output = model(data)
        loss = F.cross_entropy(output, target)
    return loss


def exponential_warmup_lambda(total_steps: int):
    def fn(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        return min(1.0, (10 ** (step / total_steps) - 1) / 9) if step < total_steps else 1.0
    return fn


def evaluate(
    model: ReveClassifier,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    num_classes: int,
) -> dict:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for eeg, labels in loader:
            eeg = eeg.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(eeg)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels)

    preds = torch.cat(all_preds).numpy()
    true = torch.cat(all_labels).numpy()
    return {
        "accuracy": float(accuracy_score(true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(true, preds)),
        "weighted_f1": float(f1_score(true, preds, average="weighted")),
        "macro_f1": float(f1_score(true, preds, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(true, preds)),
        "confusion_matrix": confusion_matrix(true, preds, labels=list(range(num_classes))),
    }


def train_stage(
    model: ReveClassifier,
    stage_cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
    num_classes: int,
    stage_name: str,
    ckpt_path: Path | None = None,
    epoch_log_path: Path | None = None,
) -> tuple[float, int, dict]:
    """Run one of the two training stages. Returns (best_val_bacc, best_epoch, best_state).

    If ``ckpt_path`` exists, training resumes from that checkpoint. The checkpoint
    is overwritten after each epoch and removed once the stage completes.
    Per-epoch metrics are appended to ``epoch_log_path`` (JSONL).
    """
    n_epochs = stage_cfg["n_epochs"]
    patience = stage_cfg.get("patience", 10)
    warmup_epochs = stage_cfg.get("warmup_epochs", 0)
    use_mixup = stage_cfg.get("mixup", True)
    lr = stage_cfg["lr"]
    weight_decay = stage_cfg.get("weight_decay", 0.01)
    sched_cfg = stage_cfg.get("scheduler", {})

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log.info("[%s] trainable params: %d", stage_name, n_trainable)

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    warmup_steps = max(1, warmup_epochs * len(train_loader))
    warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=exponential_warmup_lambda(warmup_steps),
    )
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "max"),
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 6),
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_bacc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    patience_counter = 0
    global_step = 0
    start_epoch = 0

    # ---- Resume support ----
    if ckpt_path is not None and ckpt_path.exists():
        try:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if ck.get("stage_name") == stage_name and ck.get("done", False) is False:
                model.load_state_dict(ck["model_state"])
                optimizer.load_state_dict(ck["optimizer_state"])
                scaler.load_state_dict(ck["scaler_state"])
                warmup_sched.load_state_dict(ck["warmup_state"])
                plateau_sched.load_state_dict(ck["plateau_state"])
                start_epoch = int(ck["next_epoch"])
                best_bacc = float(ck["best_bacc"])
                best_epoch = int(ck["best_epoch"])
                best_state = {k: v.clone() for k, v in ck["best_state"].items()}
                patience_counter = int(ck["patience_counter"])
                global_step = int(ck.get("global_step", 0))
                log.info(
                    "[%s] resumed from %s at epoch %d (best_bacc=%.4f@%d)",
                    stage_name, ckpt_path, start_epoch, best_bacc, best_epoch,
                )
        except Exception as exc:
            log.warning("[%s] failed to resume from %s: %s — starting fresh",
                        stage_name, ckpt_path, exc)

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        in_warmup = epoch < warmup_epochs

        for eeg, labels in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = mixup_loss(model, eeg, labels, use_mixup)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skip_lr = scaler.get_scale() != prev_scale
            if not skip_lr and in_warmup:
                warmup_sched.step()

            epoch_loss += loss.item() * eeg.size(0)
            n_seen += eeg.size(0)
            global_step += 1

        train_loss = epoch_loss / max(n_seen, 1)

        # Validate
        metrics = evaluate(model, val_loader, device, use_amp, num_classes)
        val_bacc = metrics["balanced_accuracy"]
        plateau_sched.step(val_bacc)

        log.info(
            "[%s] epoch %d/%d  train_loss=%.4f  val_bacc=%.4f  lr=%.2e",
            stage_name, epoch + 1, n_epochs, train_loss, val_bacc, optimizer.param_groups[0]["lr"],
        )

        improved = val_bacc > best_bacc
        if improved:
            best_bacc = val_bacc
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        # ---- Per-epoch JSONL log (for live monitoring) ----
        if epoch_log_path is not None:
            try:
                epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(epoch_log_path, "a") as fh:
                    fh.write(json.dumps({
                        "stage": stage_name,
                        "epoch": epoch + 1,
                        "n_epochs": n_epochs,
                        "train_loss": train_loss,
                        "val_bacc": val_bacc,
                        "lr": optimizer.param_groups[0]["lr"],
                        "best_bacc": best_bacc,
                        "best_epoch": best_epoch,
                    }) + "\n")
            except Exception as exc:
                log.warning("[%s] failed to write epoch log: %s", stage_name, exc)

        # ---- Per-epoch resume checkpoint ----
        if ckpt_path is not None:
            try:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "stage_name": stage_name,
                    "next_epoch": epoch + 1,
                    "n_epochs": n_epochs,
                    "best_bacc": best_bacc,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "patience_counter": patience_counter,
                    "global_step": global_step,
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "warmup_state": warmup_sched.state_dict(),
                    "plateau_state": plateau_sched.state_dict(),
                    "done": False,
                }, ckpt_path)
            except Exception as exc:
                log.warning("[%s] failed to write resume checkpoint: %s", stage_name, exc)

        if patience_counter >= patience:
            log.info("[%s] early stopping at epoch %d", stage_name, epoch + 1)
            break

    # Mark stage done so resume won't re-enter
    if ckpt_path is not None and ckpt_path.exists():
        try:
            ckpt_path.unlink()
        except Exception:
            pass

    return best_bacc, best_epoch, best_state


# ---------------------------------------------------------------------------
# Per-fold two-stage training
# ---------------------------------------------------------------------------

def train_one_fold(
    config: dict,
    train_files: list[Path],
    train_labels: list[int],
    val_files: list[Path],
    val_labels: list[int],
    device: torch.device,
    fold_ckpt_dir: Path | None = None,
    fold_log_path: Path | None = None,
    skip_lp: bool = False,
) -> tuple[ReveClassifier, dict]:
    train_cfg = config["training"]
    lp_cfg = config["linear_probing"]
    ft_cfg = config["fine_tuning"]
    lora_cfg = config["lora"]
    data_cfg = config.get("data", {})

    src_sr = data_cfg.get("src_sr", 256)
    tgt_sr = data_cfg.get("tgt_sr", 200)
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = train_cfg.get("num_workers", 2)
    use_amp = train_cfg.get("use_amp", True)
    grad_clip = train_cfg.get("gradient_clip", 2.0)
    num_classes = config.get("num_classes", 5)

    # Datasets
    train_ds = SegmentDataset(train_files, train_labels, src_sr=src_sr, tgt_sr=tgt_sr)
    val_ds = SegmentDataset(val_files, val_labels, src_sr=src_sr, tgt_sr=tgt_sr)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    lp_ckpt = (fold_ckpt_dir / "stage_lp.pt") if fold_ckpt_dir is not None else None
    ft_ckpt = (fold_ckpt_dir / "stage_ft.pt") if fold_ckpt_dir is not None else None

    # ----- Stage 1: Linear Probing -----
    model = build_classifier(config, dropout=lp_cfg.get("dropout", 0.05), device=device)

    if skip_lp:
        log.info("Stage 1 (LP) skipped — going straight to LoRA fine-tuning")
        info: dict = {"lp_best_val_bacc": float("nan"), "lp_best_epoch": 0}
    else:
        log.info("=" * 60)
        log.info("Stage 1 — Linear Probing")
        log.info("=" * 60)
        freeze_for_linear_probe(model)

        lp_bacc, lp_epoch, lp_state = train_stage(
            model, lp_cfg, train_loader, val_loader,
            device, use_amp, grad_clip, num_classes, stage_name="LP",
            ckpt_path=lp_ckpt, epoch_log_path=fold_log_path,
        )
        model.load_state_dict(lp_state)

        info = {
            "lp_best_val_bacc": lp_bacc,
            "lp_best_epoch": lp_epoch,
        }

    # ----- Stage 2: LoRA Fine-Tuning -----
    if not ft_cfg.get("enabled", True):
        return model, info

    log.info("=" * 60)
    log.info("Stage 2 — LoRA Fine-Tuning")
    log.info("=" * 60)
    unfreeze_all(model)
    injected = model.encoder.inject_lora(lora_cfg)
    if not injected:
        raise RuntimeError("LoRA injection matched no REVE modules")

    # After inject_lora, encoder.freeze() is called inside inject_lora — only LoRA params
    # in the encoder remain trainable. Re-enable head + cls_query_token for FT.
    for p in model.linear_head.parameters():
        p.requires_grad = True
    model.cls_query_token.requires_grad = True

    # Drop the old LP head dropout state and apply FT dropout
    for module in model.linear_head.modules():
        if isinstance(module, nn.Dropout):
            module.p = ft_cfg.get("dropout", 0.1)

    ft_bacc, ft_epoch, ft_state = train_stage(
        model, ft_cfg, train_loader, val_loader,
        device, use_amp, grad_clip, num_classes, stage_name="FT",
        ckpt_path=ft_ckpt, epoch_log_path=fold_log_path,
    )
    model.load_state_dict(ft_state)

    info["ft_best_val_bacc"] = ft_bacc
    info["ft_best_epoch"] = ft_epoch
    return model, info


# ---------------------------------------------------------------------------
# LOSO orchestration
# ---------------------------------------------------------------------------

def run_loso(
    config: dict,
    manifest_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    max_folds: int | None = None,
    resume: bool = True,
    skip_lp: bool = False,
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    fold_state_dir = output_dir / "fold_state"
    fold_state_dir.mkdir(exist_ok=True)
    epoch_log_dir = output_dir / "epoch_logs"
    epoch_log_dir.mkdir(exist_ok=True)
    fold_done_dir = output_dir / "fold_done"
    fold_done_dir.mkdir(exist_ok=True)

    subject_data = load_manifest(manifest_path)
    subjects = sorted(subject_data.keys())
    num_classes = config.get("num_classes", 5)
    use_amp = config["training"].get("use_amp", True)
    batch_size = config["training"].get("batch_size", 16)
    data_cfg = config.get("data", {})
    src_sr = data_cfg.get("src_sr", 256)
    tgt_sr = data_cfg.get("tgt_sr", 200)

    print(f"Loaded {len(subjects)} subjects: {subjects}")
    total = sum(len(d["labels"]) for d in subject_data.values())
    print(f"Total samples: {total}")

    results: list[dict] = []
    fold_subjects = subjects[:max_folds] if max_folds else subjects

    for i, test_subject in enumerate(tqdm(fold_subjects, desc="LOSO folds")):
        seed = config.get("seed", 42) + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        val_subject = subjects[(i + 1) % len(subjects)]
        train_subjects = [s for s in subjects if s != test_subject and s != val_subject]

        # ---- Resume: skip folds whose final result already exists ----
        done_path = fold_done_dir / f"subject_{int(test_subject):02d}.json"
        if resume and done_path.exists():
            try:
                with open(done_path) as fh:
                    cached = json.load(fh)
                cm_path = fold_done_dir / f"subject_{int(test_subject):02d}_cm.npy"
                cached["confusion_matrix"] = (
                    np.load(cm_path) if cm_path.exists()
                    else np.zeros((num_classes, num_classes), dtype=int)
                )
                results.append(cached)
                log.info("Fold %d: subject %d already complete — loaded cached result", i + 1, test_subject)
                continue
            except Exception as exc:
                log.warning("Failed to load cached fold %d: %s — re-running", i + 1, exc)

        train_files = [f for s in train_subjects for f in subject_data[s]["files"]]
        train_labels = [l for s in train_subjects for l in subject_data[s]["labels"]]
        val_files = subject_data[val_subject]["files"]
        val_labels = subject_data[val_subject]["labels"]
        test_files = subject_data[test_subject]["files"]
        test_labels = subject_data[test_subject]["labels"]

        log.info("Fold %d: test_subject=%d  val_subject=%d", i + 1, test_subject, val_subject)

        fold_ckpt_dir = fold_state_dir / f"subject_{int(test_subject):02d}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
        fold_log_path = epoch_log_dir / f"subject_{int(test_subject):02d}.jsonl"

        model, info = train_one_fold(
            config, train_files, train_labels,
            val_files, val_labels, device,
            fold_ckpt_dir=fold_ckpt_dir,
            fold_log_path=fold_log_path,
            skip_lp=skip_lp,
        )

        # Evaluate on test
        test_ds = SegmentDataset(test_files, test_labels, src_sr=src_sr, tgt_sr=tgt_sr)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
        )
        fold_metrics = evaluate(model, test_loader, device, use_amp, num_classes)
        fold_metrics["test_subject"] = test_subject
        fold_metrics["val_subject"] = val_subject
        fold_metrics["n_train"] = len(train_labels)
        fold_metrics["n_val"] = len(val_labels)
        fold_metrics["n_test"] = len(test_labels)
        fold_metrics.update(info)
        results.append(fold_metrics)

        ckpt_path = ckpt_dir / f"subject_{int(test_subject):02d}.pt"
        torch.save(
            {
                "head_state": {k: v.cpu() for k, v in model.linear_head.state_dict().items()},
                "cls_query_token": model.cls_query_token.detach().cpu(),
                "lora_state": model.encoder.get_lora_state_dict() if hasattr(model.encoder, "get_lora_state_dict") else {},
                "info": info,
                "test_subject": test_subject,
            },
            ckpt_path,
        )

        # ---- Mark fold complete (for resume) ----
        cm_arr = fold_metrics.pop("confusion_matrix")
        np.save(fold_done_dir / f"subject_{int(test_subject):02d}_cm.npy", cm_arr)
        with open(fold_done_dir / f"subject_{int(test_subject):02d}.json", "w") as fh:
            json.dump({k: (float(v) if isinstance(v, (np.floating, float, int)) else v)
                       for k, v in fold_metrics.items()}, fh, indent=2)
        fold_metrics["confusion_matrix"] = cm_arr
        # Clean per-stage resume state for this fold
        for f in fold_ckpt_dir.glob("*.pt"):
            try:
                f.unlink()
            except Exception:
                pass

        print(
            f"  Subject {test_subject}: "
            f"acc={fold_metrics['accuracy']:.4f}  "
            f"bacc={fold_metrics['balanced_accuracy']:.4f}  "
            f"F1_w={fold_metrics['weighted_f1']:.4f}  "
            f"kappa={fold_metrics['cohen_kappa']:.4f}  "
            f"lp_val={info.get('lp_best_val_bacc', float('nan')):.4f}  "
            f"ft_val={info.get('ft_best_val_bacc', float('nan')):.4f}"
        )

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    metric_keys = ["accuracy", "balanced_accuracy", "weighted_f1", "macro_f1", "cohen_kappa"]
    summary = {k: (float(np.mean([r[k] for r in results])), float(np.std([r[k] for r in results])))
               for k in metric_keys}

    print("\n" + "=" * 60)
    print("LOSO Results Summary (REVE Official Two-Stage Fine-Tuning)")
    print("=" * 60)
    for key, (mean, std) in summary.items():
        print(f"  {key:25s}: {mean:.4f} +/- {std:.4f}")

    emotions = config.get("emotions", ["Disgust", "Fear", "Sad", "Neutral", "Happy"])
    total_cm = sum(r["confusion_matrix"] for r in results)
    print("\nAggregated Confusion Matrix:")
    print(f"{'':>10}", end="")
    for e in emotions:
        print(f"{e:>10}", end="")
    print()
    for i_row, e in enumerate(emotions):
        print(f"{e:>10}", end="")
        for j_col in range(len(emotions)):
            print(f"{total_cm[i_row][j_col]:>10}", end="")
        print()

    fold_df = pd.DataFrame(
        [
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
                "lp_best_val_bacc": r.get("lp_best_val_bacc"),
                "lp_best_epoch": r.get("lp_best_epoch"),
                "ft_best_val_bacc": r.get("ft_best_val_bacc"),
                "ft_best_epoch": r.get("ft_best_epoch"),
            }
            for r in results
        ]
    )
    fold_df.to_csv(output_dir / "loso_results.csv", index=False)

    with open(output_dir / "summary.txt", "w") as fh:
        fh.write("SEED-V REVE Official Two-Stage Fine-Tuning — LOSO Results\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Manifest: {manifest_path}\n")
        fh.write(f"Subjects: {len(subjects)}\n")
        fh.write(f"Total samples: {total}\n\n")
        for key, (mean, std) in summary.items():
            fh.write(f"{key:25s}: {mean:.4f} +/- {std:.4f}\n")

    np.save(output_dir / "confusion_matrix.npy", total_cm)
    print(f"\nResults saved to {output_dir}/")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SEED-V REVE official two-stage fine-tuning (LOSO)."
    )
    parser.add_argument(
        "--config",
        default="Auxiliary/benchmarks/seedv/configs/seedv_reve_official.yaml",
    )
    parser.add_argument("--manifest", default="data/preprocessed/seedv_preprocessed/manifest.csv")
    parser.add_argument("--output_dir", default="logs/seedv_reve_official")
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds", type=int, default=None,
                        help="Run only the first N LOSO folds (for testing).")
    parser.add_argument("--lp-epochs", type=int, default=None)
    parser.add_argument("--ft-epochs", type=int, default=None)
    parser.add_argument("--skip-lp", action="store_true",
                        help="Skip Stage 1 (linear probing) — go straight to LoRA fine-tuning. "
                             "Use for the LoRA-only control (B2).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume from existing fold/stage checkpoints.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.lp_epochs is not None:
        config["linear_probing"]["n_epochs"] = args.lp_epochs
    if args.ft_epochs is not None:
        config["fine_tuning"]["n_epochs"] = args.ft_epochs

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    if config.get("seed"):
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

    run_loso(
        config, args.manifest, args.output_dir, device,
        max_folds=args.folds,
        resume=not args.no_resume,
        skip_lp=args.skip_lp,
    )


if __name__ == "__main__":
    main()
