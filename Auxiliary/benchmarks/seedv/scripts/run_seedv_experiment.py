"""SEED-V 5-class emotion classification with EEGPT embeddings (LOSO)."""

import argparse
import logging
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


class EmotionClassifier(nn.Module):
    """Simple MLP classifier for emotion recognition."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_embeddings(
    embeddings_dir: str | Path,
    manifest_path: str | Path | None = None,
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


def train_one_fold(
    train_embs: torch.Tensor,
    train_labels: torch.Tensor,
    val_embs: torch.Tensor,
    val_labels: torch.Tensor,
    test_embs: torch.Tensor,
    test_labels: torch.Tensor,
    config: dict,
    device: torch.device,
) -> dict:
    """Train and evaluate one LOSO fold.

    Early stopping is performed on the *validation* loss, not training loss.
    The model checkpoint with the lowest validation loss is restored before
    final evaluation on the test set.
    """
    input_dim = train_embs.shape[1]
    num_classes = config.get("num_classes", 5)
    hidden_dim = config.get("fusion", {}).get("d_common", 256)
    dropout = config.get("fusion", {}).get("dropout", 0.1)

    model = EmotionClassifier(input_dim, hidden_dim, num_classes, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.01),
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["max_epochs"]
    )

    # DataLoaders
    train_ds = TensorDataset(train_embs.to(device), train_labels.to(device))
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    val_embs_dev = val_embs.to(device)
    val_labels_dev = val_labels.to(device)

    # Training with early stopping on validation loss
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"].get("patience", 10)
    best_state: dict = {}

    for epoch in range(config["training"]["max_epochs"]):
        model.train()
        for batch_embs, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_embs)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_embs_dev), val_labels_dev).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate on test set using best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(test_embs.to(device))
        preds = test_logits.argmax(dim=-1).cpu().numpy()

    true = test_labels.numpy()
    acc = accuracy_score(true, preds)
    bal_acc = balanced_accuracy_score(true, preds)
    kappa = cohen_kappa_score(true, preds)
    f1_w = f1_score(true, preds, average="weighted")
    f1_m = f1_score(true, preds, average="macro")
    cm = confusion_matrix(true, preds, labels=list(range(num_classes)))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "kappa": kappa,
        "f1_weighted": f1_w,
        "f1_macro": f1_m,
        "confusion_matrix": cm,
        "preds": preds,
        "true": true,
        "best_val_loss": best_val_loss,
    }


def run_loso(config: dict, device: torch.device) -> list[dict]:
    """Run Leave-One-Subject-Out cross-validation."""
    embeddings_dir = config["embeddings_dir"]
    subject_data = load_embeddings(embeddings_dir)
    subjects = sorted(subject_data.keys())

    print(f"Loaded {len(subjects)} subjects: {subjects}")
    print(
        f"Total samples: {sum(d['embeddings'].shape[0] for d in subject_data.values())}"
    )

    results: list[dict] = []

    for fold_idx, test_subject in enumerate(tqdm(subjects, desc="LOSO folds")):
        # Validation subject: the subject immediately after the test subject
        # (wrapping around), giving a 14 train / 1 val / 1 test split.
        val_subject = subjects[(fold_idx + 1) % len(subjects)]
        train_subjects = [s for s in subjects if s not in (test_subject, val_subject)]

        train_embs = torch.cat([subject_data[s]["embeddings"] for s in train_subjects])
        train_labels = torch.cat([subject_data[s]["labels"] for s in train_subjects])
        val_embs = subject_data[val_subject]["embeddings"]
        val_labels = subject_data[val_subject]["labels"]
        test_embs = subject_data[test_subject]["embeddings"]
        test_labels = subject_data[test_subject]["labels"]

        fold_result = train_one_fold(
            train_embs,
            train_labels,
            val_embs,
            val_labels,
            test_embs,
            test_labels,
            config,
            device,
        )
        fold_result["test_subject"] = test_subject
        fold_result["val_subject"] = val_subject
        fold_result["n_train"] = len(train_labels)
        fold_result["n_val"] = len(val_labels)
        fold_result["n_test"] = len(test_labels)
        results.append(fold_result)

        print(
            f"  Subject {test_subject} (val={val_subject}): "
            f"bal_acc={fold_result['balanced_accuracy']:.4f}, "
            f"kappa={fold_result['kappa']:.4f}, "
            f"acc={fold_result['accuracy']:.4f}, "
            f"F1_w={fold_result['f1_weighted']:.4f}, "
            f"F1_m={fold_result['f1_macro']:.4f}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    mean_bal_acc = np.mean([r["balanced_accuracy"] for r in results])
    std_bal_acc = np.std([r["balanced_accuracy"] for r in results])
    mean_kappa = np.mean([r["kappa"] for r in results])
    std_kappa = np.std([r["kappa"] for r in results])
    mean_acc = np.mean([r["accuracy"] for r in results])
    std_acc = np.std([r["accuracy"] for r in results])
    mean_f1w = np.mean([r["f1_weighted"] for r in results])
    std_f1w = np.std([r["f1_weighted"] for r in results])
    mean_f1m = np.mean([r["f1_macro"] for r in results])
    std_f1m = np.std([r["f1_macro"] for r in results])

    print("\n" + "=" * 60)
    print("LOSO Results Summary  (14 train / 1 val / 1 test)")
    print("=" * 60)
    print(f"Balanced Accuracy: {mean_bal_acc:.4f} +/- {std_bal_acc:.4f}  [PRIMARY]")
    print(f"Cohen Kappa:       {mean_kappa:.4f} +/- {std_kappa:.4f}")
    print(f"Accuracy:          {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"F1 Weighted:       {mean_f1w:.4f} +/- {std_f1w:.4f}")
    print(f"F1 Macro:          {mean_f1m:.4f} +/- {std_f1m:.4f}")

    # Per-subject breakdown
    emotions = config.get("emotions", ["Disgust", "Fear", "Sad", "Neutral", "Happy"])
    print("\nPer-Subject Breakdown:")
    header = f"{'Subj':>6} {'ValSubj':>8} {'BalAcc':>8} {'Kappa':>7} {'Acc':>7} {'F1_w':>7} {'F1_m':>7} {'N_test':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['test_subject']:>6} {r['val_subject']:>8} "
            f"{r['balanced_accuracy']:>8.4f} {r['kappa']:>7.4f} "
            f"{r['accuracy']:>7.4f} {r['f1_weighted']:>7.4f} "
            f"{r['f1_macro']:>7.4f} {r['n_test']:>7}"
        )

    # Per-class from aggregated confusion matrix
    total_cm = sum(r["confusion_matrix"] for r in results)
    print("\nAggregated Confusion Matrix:")
    print(f"{'':>10}", end="")
    for e in emotions:
        print(f"{e:>10}", end="")
    print()
    for i, e in enumerate(emotions):
        print(f"{e:>10}", end="")
        for j in range(len(emotions)):
            print(f"{total_cm[i][j]:>10}", end="")
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_dir = Path("logs/seedv_eegpt")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold CSV
    fold_df = pd.DataFrame(
        [
            {
                "test_subject": r["test_subject"],
                "val_subject": r["val_subject"],
                "balanced_accuracy": r["balanced_accuracy"],
                "kappa": r["kappa"],
                "accuracy": r["accuracy"],
                "f1_weighted": r["f1_weighted"],
                "f1_macro": r["f1_macro"],
                "n_train": r["n_train"],
                "n_val": r["n_val"],
                "n_test": r["n_test"],
                "best_val_loss": r["best_val_loss"],
            }
            for r in results
        ]
    )
    fold_df.to_csv(results_dir / "loso_results.csv", index=False)

    # Summary text
    with open(results_dir / "summary.txt", "w") as f:
        f.write("SEED-V EEGPT LOSO Results  (14 train / 1 val / 1 test)\n")
        f.write("=" * 54 + "\n")
        f.write(f"Balanced Accuracy: {mean_bal_acc:.4f} +/- {std_bal_acc:.4f}  [PRIMARY]\n")
        f.write(f"Cohen Kappa:       {mean_kappa:.4f} +/- {std_kappa:.4f}\n")
        f.write(f"Accuracy:          {mean_acc:.4f} +/- {std_acc:.4f}\n")
        f.write(f"F1 Weighted:       {mean_f1w:.4f} +/- {std_f1w:.4f}\n")
        f.write(f"F1 Macro:          {mean_f1m:.4f} +/- {std_f1m:.4f}\n")
        f.write("\nPer-Subject Breakdown:\n")
        f.write(fold_df.to_string(index=False))
        f.write("\n")

    print(f"\nResults saved to {results_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SEED-V emotion classification with EEGPT"
    )
    parser.add_argument(
        "--config",
        default="Auxiliary/benchmarks/seedv/configs/seedv_base.yaml",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (cuda/cpu). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config.get("seed"):
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

    run_loso(config, device)


if __name__ == "__main__":
    main()
