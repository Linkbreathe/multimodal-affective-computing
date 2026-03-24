"""FusionTrainer: LOSO cross-validation loop with multi-task training."""
from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.tasks.heads import MultiTaskHead
from src.tasks.losses import MultiTaskLoss
from src.fusion.base import BaseFusionModule
from src.trainer.early_stopping import EarlyStopping
from src.utils.metrics import (
    weighted_f1_score,
    concordance_correlation_coefficient,
    compute_class_weights,
)

log = logging.getLogger("fusion")


class FusionTrainer:
    """Trains a fusion model with LOSO cross-validation."""

    def __init__(
        self,
        fusion_model: BaseFusionModule,
        task_head: MultiTaskHead,
        config: dict[str, Any],
        device: str | None = None,
    ) -> None:
        self.fusion_model = fusion_model
        self.task_head = task_head
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._last_fusion: nn.Module | None = None
        self._last_head: nn.Module | None = None

    def train_fold(
        self,
        train_data: list[dict],
        val_data: list[dict],
        fold_name: str = "fold",
        tb_logger: Any | None = None,
    ) -> dict[str, float]:
        """Train one LOSO fold. Returns val metrics from best checkpoint."""
        cfg = self.config["training"]

        # Fresh model copies for this fold
        fusion_model = copy.deepcopy(self.fusion_model).to(self.device)
        task_head = copy.deepcopy(self.task_head).to(self.device)
        all_params = list(fusion_model.parameters()) + list(task_head.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["max_epochs"]
        )

        # Class weights from training data
        train_labels = np.array(
            [d["labels"]["emotion_label"].item() for d in train_data]
        )
        class_weights = compute_class_weights(train_labels, num_classes=9).to(
            self.device
        )

        loss_fn = MultiTaskLoss(
            ce_weight=class_weights,
            lambda_ce=self.config["loss_weights"]["ce"],
            lambda_kl=self.config["loss_weights"]["kl"],
            lambda_vad=self.config["loss_weights"]["vad"],
        ).to(self.device)

        early_stop = EarlyStopping(patience=cfg["patience"], mode="max")
        best_state = None

        train_loader = self._make_loader(train_data, cfg["batch_size"], shuffle=True)
        val_loader = self._make_loader(val_data, cfg["batch_size"], shuffle=False)

        for epoch in range(cfg["max_epochs"]):
            # Train
            fusion_model.train()
            task_head.train()
            epoch_loss = 0.0
            for batch in train_loader:
                embeddings, modality_ids, labels = self._unpack_batch(batch)
                fused = fusion_model(embeddings, modality_ids)
                outputs = task_head(fused)
                loss, breakdown = loss_fn(outputs, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # Validate
            val_metrics = self._evaluate(fusion_model, task_head, val_loader, loss_fn)

            if tb_logger:
                tb_logger.log_scalars(
                    f"{fold_name}/train",
                    {"loss": epoch_loss / max(len(train_loader), 1)},
                    epoch,
                )
                tb_logger.log_scalars(f"{fold_name}/val", val_metrics, epoch)

            # Early stopping
            if early_stop.step(val_metrics["weighted_f1"]):
                log.info(f"{fold_name} early stopping at epoch {epoch}")
                break

            if early_stop.counter == 0:
                best_state = {
                    "fusion": {
                        k: v.cpu().clone()
                        for k, v in fusion_model.state_dict().items()
                    },
                    "head": {
                        k: v.cpu().clone()
                        for k, v in task_head.state_dict().items()
                    },
                }

        # Restore best
        if best_state:
            fusion_model.load_state_dict(best_state["fusion"])
            task_head.load_state_dict(best_state["head"])

        self._last_fusion = fusion_model
        self._last_head = task_head
        return self._evaluate(fusion_model, task_head, val_loader, loss_fn)

    @torch.no_grad()
    def _evaluate(
        self,
        fusion_model: nn.Module,
        task_head: nn.Module,
        loader: DataLoader,
        loss_fn: MultiTaskLoss,
    ) -> dict[str, float]:
        fusion_model.eval()
        task_head.eval()
        all_preds, all_labels = [], []
        all_vad_pred, all_vad_true = [], []
        total_loss = 0.0

        for batch in loader:
            embeddings, modality_ids, labels = self._unpack_batch(batch)
            fused = fusion_model(embeddings, modality_ids)
            outputs = task_head(fused)
            loss, _ = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = outputs["emotion_logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels["emotion_label"].cpu().numpy())
            all_vad_pred.append(outputs["vad_pred"].cpu().numpy())
            all_vad_true.append(labels["vad"].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_vad_pred = (
            np.concatenate(all_vad_pred) if all_vad_pred else np.zeros((0, 3))
        )
        all_vad_true = (
            np.concatenate(all_vad_true) if all_vad_true else np.zeros((0, 3))
        )

        f1 = weighted_f1_score(all_labels, all_preds)
        ccc_vals = (
            [
                concordance_correlation_coefficient(
                    all_vad_true[:, i], all_vad_pred[:, i]
                )
                for i in range(min(3, all_vad_true.shape[1]))
            ]
            if len(all_vad_true) > 0
            else [0.0, 0.0, 0.0]
        )

        return {
            "weighted_f1": f1,
            "loss": total_loss / max(len(loader), 1),
            "ccc": float(np.mean(ccc_vals)),
            "ccc_valence": ccc_vals[0] if ccc_vals else 0.0,
            "ccc_arousal": ccc_vals[1] if len(ccc_vals) > 1 else 0.0,
            "ccc_dominance": ccc_vals[2] if len(ccc_vals) > 2 else 0.0,
        }

    def _unpack_batch(
        self, batch: dict
    ) -> tuple[list[torch.Tensor], list[str], dict[str, torch.Tensor]]:
        embeddings = [
            batch["embeddings"][m].to(self.device) for m in batch["modality_ids"]
        ]
        labels = {k: v.to(self.device) for k, v in batch["labels"].items()}
        return embeddings, batch["modality_ids"], labels

    def _make_loader(
        self, data: list[dict], batch_size: int, shuffle: bool
    ) -> DataLoader:

        class _ListDS(Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        def collate(batch):
            modality_ids = batch[0]["modality_ids"]
            embeddings = {}
            for m in modality_ids:
                tensors = [
                    b["embeddings"][b["modality_ids"].index(m)] for b in batch
                ]
                embeddings[m] = torch.stack(tensors)
            labels = {}
            for k in batch[0]["labels"]:
                vals = [b["labels"][k] for b in batch]
                labels[k] = (
                    torch.stack(vals)
                    if isinstance(vals[0], torch.Tensor)
                    else torch.tensor(vals)
                )
            return {
                "embeddings": embeddings,
                "modality_ids": modality_ids,
                "labels": labels,
            }

        return DataLoader(
            _ListDS(data),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=0,
        )

    def run_loso(
        self,
        all_data: dict[str, list[dict]],
        subject_ids: list[str],
        tb_base_dir: str | None = None,
    ) -> list[dict[str, float]]:
        """Run full LOSO cross-validation."""
        fold_results = []
        for i, test_subj in enumerate(subject_ids):
            seed = self.config["seed"] + i
            torch.manual_seed(seed)
            np.random.seed(seed)

            val_subj = subject_ids[(i + 1) % len(subject_ids)]
            train_subjects = [
                s for s in subject_ids if s not in (test_subj, val_subj)
            ]

            train_data = [
                s for subj in train_subjects for s in all_data.get(subj, [])
            ]
            val_data = all_data.get(val_subj, [])
            test_data = all_data.get(test_subj, [])

            if not train_data or not test_data:
                log.warning(f"Skipping fold {test_subj}: insufficient data")
                continue

            log.info(
                f"Fold {i+1}/{len(subject_ids)}: test={test_subj}, val={val_subj}, "
                f"train={len(train_data)}, val_size={len(val_data)}, "
                f"test_size={len(test_data)}"
            )

            self.train_fold(train_data, val_data, fold_name=f"fold_{test_subj}")

            # Evaluate on test subject using best model from this fold
            test_loader = self._make_loader(
                test_data, self.config["training"]["batch_size"], shuffle=False
            )
            train_labels = np.array(
                [d["labels"]["emotion_label"].item() for d in train_data]
            )
            class_weights = compute_class_weights(train_labels, 9).to(self.device)
            loss_fn = MultiTaskLoss(ce_weight=class_weights).to(self.device)
            metrics = self._evaluate(
                self._last_fusion, self._last_head, test_loader, loss_fn
            )
            metrics["test_subject"] = test_subj
            fold_results.append(metrics)
            log.info(
                f"Fold {test_subj}: F1={metrics['weighted_f1']:.4f}, "
                f"CCC={metrics['ccc']:.4f}"
            )

        return fold_results
