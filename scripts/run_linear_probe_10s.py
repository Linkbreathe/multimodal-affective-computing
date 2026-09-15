"""Pure video → nn.Linear(768, 9) → emotion probe with LOSO evaluation.

No fusion framework, no MLP, no multi-task head. Matches the training
recipe from the reference repo's probe_common.py (AdamW lr=1e-3 wd=1e-4,
batch 256, patience 20, model selection on val macro F1).

Usage:
    conda run -n visphy python scripts/run_linear_probe_10s.py --name linear_probe_video
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from mac.evaluation.metrics import compute_class_weights

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("linear_probe")

NUM_CLASSES = 9
EMBED_DIM = 768


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_video_embeddings(
    embeddings_dir: str,
    manifest: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Load and pool video embeddings by subject.

    Each .pt file contains {"embedding": [T, 768], "label": int, ...}.
    We mean-pool [T, 768] → [768] and pair with the integer label.
    """
    emb_path = Path(embeddings_dir)
    data_by_subject: dict[str, list[dict]] = {}

    for subj_id, subj_rows in manifest.groupby("subject"):
        subj = str(subj_id).zfill(3)
        samples = []

        for _, row in subj_rows.iterrows():
            seg_idx = int(row["segment_path"].split("/")[-1].replace(".p", ""))
            path = emb_path / subj / f"segment_{seg_idx:04d}.pt"
            if not path.exists():
                continue

            data = torch.load(path, weights_only=False)
            emb = data["embedding"]
            if emb.dim() == 2:
                emb = emb.mean(dim=0)  # [T, 768] → [768]
            elif emb.dim() == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0).mean(dim=0)
            label = int(data["label"])

            samples.append({"embedding": emb, "label": label})

        if samples:
            data_by_subject[subj] = samples

    return data_by_subject


def _stack_subjects(
    data_by_subject: dict[str, list[dict]],
    subject_ids: list[str] | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack embeddings and labels for one or more subjects."""
    if isinstance(subject_ids, str):
        subject_ids = [subject_ids]
    embs, labels = [], []
    for subj in subject_ids:
        for s in data_by_subject.get(subj, []):
            embs.append(s["embedding"])
            labels.append(s["label"])
    return torch.stack(embs), torch.tensor(labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Probe training (one fold)
# ---------------------------------------------------------------------------

def train_probe_fold(
    train_embs: torch.Tensor,
    train_labels: torch.Tensor,
    val_embs: torch.Tensor,
    val_labels: torch.Tensor,
    device: str,
    seed: int,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 20,
) -> nn.Linear:
    """Train nn.Linear(768, 9) with CE + class weights. Return best checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    probe = nn.Linear(EMBED_DIM, NUM_CLASSES).to(device)

    cw = compute_class_weights(train_labels.numpy(), NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(train_embs, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    val_embs_dev = val_embs.to(device)
    val_labels_np = val_labels.numpy()

    best_f1 = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        probe.train()
        for emb_batch, label_batch in train_loader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)
            logits = probe(emb_batch)
            loss = loss_fn(logits, label_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_preds = probe(val_embs_dev).argmax(dim=-1).cpu().numpy()
        val_macro = float(f1_score(val_labels_np, val_preds, average="macro", zero_division=0))

        if val_macro > best_f1:
            best_f1 = val_macro
            best_state = copy.deepcopy(probe.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_probe(
    probe: nn.Linear,
    embs: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> dict[str, float]:
    """Evaluate probe, return macro F1 and weighted F1."""
    probe.eval()
    with torch.no_grad():
        preds = probe(embs.to(device)).argmax(dim=-1).cpu().numpy()
    y_true = labels.numpy()
    return {
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# LOSO
# ---------------------------------------------------------------------------

def run_loso(
    data_by_subject: dict[str, list[dict]],
    device: str,
    seed: int = 42,
) -> list[dict]:
    """Run 40-fold LOSO with a fresh linear probe per fold."""
    subject_ids = sorted(data_by_subject.keys())
    fold_results = []

    for i, test_subj in enumerate(subject_ids):
        val_subj = subject_ids[(i + 1) % len(subject_ids)]
        train_subjs = [s for s in subject_ids if s not in (test_subj, val_subj)]

        train_embs, train_labels = _stack_subjects(data_by_subject, train_subjs)
        val_embs, val_labels = _stack_subjects(data_by_subject, val_subj)
        test_embs, test_labels = _stack_subjects(data_by_subject, test_subj)

        if len(train_embs) == 0 or len(test_embs) == 0:
            log.warning(f"Skipping fold {test_subj}: insufficient data")
            continue

        probe = train_probe_fold(
            train_embs, train_labels,
            val_embs, val_labels,
            device, seed + i,
        )
        metrics = evaluate_probe(probe, test_embs, test_labels, device)
        metrics["test_subject"] = test_subj
        fold_results.append(metrics)

        log.info(
            f"Fold {i+1}/{len(subject_ids)}: test={test_subj} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pure video linear probe with LOSO")
    parser.add_argument("--embeddings-dir", default="data/embeddings/egoemotion/10s/video_mae_v2")
    parser.add_argument("--data-dir", default="data/datasets/egoemotion_raw")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None)
    parser.add_argument("--fixed-split", action="store_true",
                        help="Use fixed train/val/test split instead of LOSO")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or f"linear_probe_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log.info(f"Experiment: {name}")
    log.info(f"Device: {device}")
    log.info(f"Model: nn.Linear({EMBED_DIM}, {NUM_CLASSES}) — {EMBED_DIM * NUM_CLASSES + NUM_CLASSES} params")

    # Load manifest
    data_dir = Path(args.data_dir)
    manifest = pd.read_csv(data_dir / "ce_hardlabel_manifests" / "dataset_manifest.csv")
    log.info(f"Manifest: {len(manifest)} segments, {manifest['subject'].nunique()} subjects")

    # Load embeddings
    data_by_subject = load_video_embeddings(args.embeddings_dir, manifest)
    total = sum(len(v) for v in data_by_subject.values())
    log.info(f"Loaded: {total} pooled [768] embeddings across {len(data_by_subject)} subjects")

    if total == 0:
        log.error("No data loaded. Check embeddings directory.")
        return

    if args.fixed_split:
        # Fixed train/val/test split (matches old repo protocol)
        splits_dir = data_dir / "ce_hardlabel_manifests" / "splits"
        split_subjects: dict[str, set[str]] = {}
        for split_name in ("train", "val", "test"):
            csv_path = splits_dir / f"{split_name}.csv"
            if not csv_path.exists():
                log.error(f"Missing {csv_path} for --fixed-split")
                return
            df = pd.read_csv(csv_path)
            split_subjects[split_name] = {str(int(s)).zfill(3) for s in df["subject_id"].unique()}

        train_embs, train_labels = _stack_subjects(data_by_subject, sorted(split_subjects["train"]))
        val_embs, val_labels = _stack_subjects(data_by_subject, sorted(split_subjects["val"]))
        test_embs, test_labels = _stack_subjects(data_by_subject, sorted(split_subjects["test"]))

        log.info(f"Fixed split: train={len(train_labels)}, val={len(val_labels)}, test={len(test_labels)}")

        probe = train_probe_fold(train_embs, train_labels, val_embs, val_labels, device, args.seed)
        test_metrics = evaluate_probe(probe, test_embs, test_labels, device)

        log.info(f"\n{'='*60}")
        log.info(f"Fixed-Split Linear Probe Results:")
        log.info(f"  Macro F1:    {test_metrics['macro_f1']:.4f}")
        log.info(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
        log.info(f"{'='*60}")
        return

    # Run LOSO
    fold_results = run_loso(data_by_subject, device, args.seed)

    if not fold_results:
        log.error("No fold results.")
        return

    # Report
    macro_scores = [r["macro_f1"] for r in fold_results]
    weighted_scores = [r["weighted_f1"] for r in fold_results]

    log.info(f"\n{'='*60}")
    log.info(f"Linear Probe LOSO Results ({len(fold_results)} folds):")
    log.info(f"  Macro F1:    {np.mean(macro_scores):.4f} +/- {np.std(macro_scores):.4f}")
    log.info(f"  Weighted F1: {np.mean(weighted_scores):.4f} +/- {np.std(weighted_scores):.4f}")
    log.info(f"  ---")
    log.info(f"  Phase 1 MLP baseline (weighted F1): 0.2279")
    log.info(f"  Paper baseline (Classical, All):     0.46")
    log.info(f"{'='*60}")

    # Save results
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, "w") as f:
        f.write(f"# {name}\n\n")
        f.write(f"Model: nn.Linear({EMBED_DIM}, {NUM_CLASSES})\n")
        f.write(f"Folds: {len(fold_results)}\n\n")
        f.write(f"| Subject | Macro F1 | Weighted F1 |\n")
        f.write(f"|---------|----------|-------------|\n")
        for r in fold_results:
            f.write(f"| {r['test_subject']} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |\n")
        f.write(f"\n**Macro F1:** {np.mean(macro_scores):.4f} +/- {np.std(macro_scores):.4f}\n")
        f.write(f"**Weighted F1:** {np.mean(weighted_scores):.4f} +/- {np.std(weighted_scores):.4f}\n")
    log.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
