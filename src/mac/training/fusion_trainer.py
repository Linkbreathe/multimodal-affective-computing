"""FusionTrainer: LOSO cross-validation loop with multi-task training."""
from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mac.tasks.heads import MultiTaskHead
from mac.tasks.losses import DirichletKLLoss, MultiTaskLoss
from mac.fusion.base import BaseFusionModule
from mac.fusion.cggm import CGGMModule
from mac.training.early_stopping import EarlyStopping
from mac.evaluation.metrics import (
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
        encoder_param_groups: list[dict] | None = None,
    ) -> None:
        self.fusion_model = fusion_model
        self.task_head = task_head
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._last_fusion: nn.Module | None = None
        self._last_head: nn.Module | None = None

        # Encoder fine-tuning configuration
        self.encoder_param_groups = encoder_param_groups or []
        ft_cfg = config.get("finetune", {})
        self.finetune_enabled = ft_cfg.get("enabled", False) and bool(self.encoder_param_groups)
        self.gradient_clip = ft_cfg.get("gradient_clip", 0.0)
        self.warmup_epochs = ft_cfg.get("warmup_epochs", 0)

        # CGGM configuration
        cggm_cfg = config.get("cggm", {})
        self.cggm_enabled = cggm_cfg.get("enabled", False)
        self.cggm_lambda_mod = cggm_cfg.get("lambda_mod", 0.1)
        self.cggm_hidden = cggm_cfg.get("classifier_hidden", 128)

        # TMC evidential fusion configuration
        tmc_cfg = config.get("tmc", {})
        self.tmc_enabled = tmc_cfg.get("enabled", False)
        self.tmc_annealing_epochs = tmc_cfg.get("annealing_epochs", 10)
        self.tmc_num_classes = tmc_cfg.get("num_classes", 9)

        # Distillation configuration
        distill_cfg = config.get("distill", {})
        self.distill_enabled = distill_cfg.get("enabled", False)
        self.distill_lambda = distill_cfg.get("lambda_distill", 1.0)

    def train_fold(
        self,
        train_data: list[dict],
        val_data: list[dict],
        fold_name: str = "fold",
        tb_logger: Any | None = None,
    ) -> dict[str, float]:
        """Train one LOSO fold. Returns val metrics from best checkpoint."""
        cfg = self.config["training"]

        # Fresh model copies for this fold (includes encoder if present)
        fusion_model = copy.deepcopy(self.fusion_model).to(self.device)
        task_head = copy.deepcopy(self.task_head).to(self.device)

        # Build optimizer param groups
        if self.finetune_enabled:
            # Downstream params (projector + fusion + head) — exclude encoder params
            encoder_param_ids = set()
            for enc in getattr(fusion_model, "encoders", {}).values():
                for p in enc.parameters():
                    encoder_param_ids.add(id(p))

            downstream_params = [
                p for p in fusion_model.parameters()
                if id(p) not in encoder_param_ids
            ] + list(task_head.parameters())

            # Re-resolve encoder param groups from the deepcopied model
            encoder_groups = []
            for enc_name, enc in fusion_model.encoders.items():
                for group in enc.get_layer_groups():
                    # Find the matching original group to get the LR
                    lr = cfg["lr"]
                    for orig_g in self.encoder_param_groups:
                        if orig_g["name"] == f"{enc_name}_{group['name']}":
                            lr = orig_g["lr"]
                            break
                    encoder_groups.append({
                        "params": group["params"],
                        "lr": lr,
                    })

            param_groups = [{"params": downstream_params, "lr": cfg["lr"]}]
            param_groups.extend(encoder_groups)

            all_params = downstream_params
            for g in encoder_groups:
                all_params = all_params + g["params"]

            log.info(
                f"{fold_name} fine-tuning: {len(encoder_groups)} encoder param groups, "
                f"warmup={self.warmup_epochs} epochs, grad_clip={self.gradient_clip}"
            )
        else:
            all_params = list(fusion_model.parameters()) + list(task_head.parameters())
            param_groups = [{"params": all_params, "lr": cfg["lr"]}]

        # CGGM: per-modality classifiers for gradient modulation
        cggm = None
        if self.cggm_enabled:
            modality_ids = fusion_model.modality_names
            d_common = fusion_model.d_common
            cggm = CGGMModule(
                modality_ids=modality_ids,
                d_common=d_common,
                num_classes=9,
                d_hidden=self.cggm_hidden,
            ).to(self.device)
            param_groups.append({"params": list(cggm.parameters()), "lr": cfg["lr"]})
            all_params = all_params + list(cggm.parameters())
            log.info(
                f"{fold_name} CGGM enabled: {sum(p.numel() for p in cggm.parameters()):,} "
                f"classifier params, lambda_mod={self.cggm_lambda_mod}"
            )

        optimizer = torch.optim.AdamW(param_groups)
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

        # TMC: Dirichlet KL loss for evidential regularization
        dkl_loss = DirichletKLLoss() if self.tmc_enabled else None

        train_loader = self._make_loader(train_data, cfg["batch_size"], shuffle=True)
        val_loader = self._make_loader(val_data, cfg["batch_size"], shuffle=False)

        for epoch in range(cfg["max_epochs"]):
            # Train mode — use selective_train for encoders (keeps BN in eval)
            fusion_model.train()
            if self.finetune_enabled and hasattr(fusion_model, "encoders"):
                for enc in fusion_model.encoders.values():
                    enc.selective_train()
                # Warmup: freeze encoder params for the first N epochs
                encoder_frozen_this_epoch = epoch < self.warmup_epochs
                if encoder_frozen_this_epoch:
                    for enc in fusion_model.encoders.values():
                        for p in enc.parameters():
                            p.requires_grad = False
                else:
                    # Re-enable gradients on unfrozen encoder params
                    for enc_name, enc in fusion_model.encoders.items():
                        ft_cfg = self.config.get("finetune", {})
                        from_block = ft_cfg.get("unfreeze_from_block")
                        if from_block is not None:
                            enc.unfreeze(from_layer=from_block)
                        else:
                            enc.unfreeze()
                        enc.selective_train()

            task_head.train()
            if cggm is not None:
                cggm.train()
            epoch_loss = 0.0
            for batch in train_loader:
                embeddings, modality_ids, labels, masks = self._unpack_batch(batch)
                raw_signals = self._unpack_raw_signals(batch)

                if cggm is not None and hasattr(fusion_model, "project_and_pool"):
                    # CGGM path: split projection from fusion so we can
                    # hook gradients on the projected embeddings.
                    proj_list, proj_masks = fusion_model.project_and_pool(
                        embeddings, modality_ids, masks
                    )

                    # Compute per-modality classifier losses and gradient scales
                    scales, mod_loss = cggm.compute_scales(
                        proj_list, modality_ids, labels["emotion_label"]
                    )

                    # Register gradient scaling hooks
                    hooks = CGGMModule.register_hooks(proj_list, modality_ids, scales)

                    # Forward through fusion + task head
                    fused = fusion_model.fusion(proj_list, modality_ids, proj_masks)
                    outputs = task_head(fused)
                    loss, breakdown = loss_fn(outputs, labels)

                    # Add weighted per-modality classifier loss
                    total_loss = loss + self.cggm_lambda_mod * mod_loss

                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.gradient_clip)
                    optimizer.step()

                    # Clean up hooks
                    for h in hooks:
                        h.remove()

                    epoch_loss += loss.item()
                else:
                    # Standard path (no CGGM) — passes raw_signals for encoder forward
                    fused = self._forward_fusion(
                        fusion_model, embeddings, modality_ids, masks, raw_signals
                    )
                    outputs = task_head(fused)
                    loss, breakdown = loss_fn(outputs, labels)

                    # TMC: add per-modality Dirichlet KL with annealing
                    if self.tmc_enabled and dkl_loss is not None and hasattr(fusion_model, 'fusion'):
                        tmc_mod = fusion_model.fusion
                        if hasattr(tmc_mod, 'last_alphas') and tmc_mod.last_alphas:
                            lambda_t = min(1.0, epoch / max(self.tmc_annealing_epochs, 1))
                            dkl_total = torch.tensor(0.0, device=loss.device)
                            for mod_id, alpha in tmc_mod.last_alphas.items():
                                dkl_total = dkl_total + dkl_loss(
                                    alpha, labels["emotion_label"], self.tmc_num_classes
                                )
                            loss = loss + lambda_t * dkl_total

                    # Distillation: add soft-target KL loss from fusion module
                    if self.distill_enabled and hasattr(fusion_model, 'fusion'):
                        d_mod = fusion_model.fusion
                        if hasattr(d_mod, 'last_distill_loss') and d_mod.last_distill_loss is not None:
                            loss = loss + self.distill_lambda * d_mod.last_distill_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(all_params, max_norm=self.gradient_clip)
                    optimizer.step()
                    epoch_loss += loss.item()

            scheduler.step()

            # Validate (no CGGM hooks — standard forward)
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

        # TMC reliability tracking
        tmc_uncertainties: dict[str, list[float]] = {}
        tmc_combined_uncert: list[float] = []

        for batch in loader:
            embeddings, modality_ids, labels, masks = self._unpack_batch(batch)
            raw_signals = self._unpack_raw_signals(batch)
            fused = self._forward_fusion(
                fusion_model, embeddings, modality_ids, masks, raw_signals
            )
            outputs = task_head(fused)
            loss, _ = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = outputs["emotion_logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels["emotion_label"].cpu().numpy())
            all_vad_pred.append(outputs["vad_pred"].cpu().numpy())
            all_vad_true.append(labels["vad"].cpu().numpy())

            # Collect TMC per-modality uncertainties
            tmc_mod = getattr(fusion_model, 'fusion', None)
            if tmc_mod is not None and hasattr(tmc_mod, 'last_uncertainties') and tmc_mod.last_uncertainties:
                for mod_id, u in tmc_mod.last_uncertainties.items():
                    tmc_uncertainties.setdefault(mod_id, []).extend(
                        u.squeeze(-1).cpu().numpy().tolist()
                    )
                if tmc_mod.last_combined_uncertainty is not None:
                    tmc_combined_uncert.extend(
                        tmc_mod.last_combined_uncertainty.squeeze(-1).cpu().numpy().tolist()
                    )

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

        metrics: dict[str, float] = {
            "weighted_f1": f1,
            "loss": total_loss / max(len(loader), 1),
            "ccc": float(np.mean(ccc_vals)),
            "ccc_valence": ccc_vals[0] if ccc_vals else 0.0,
            "ccc_arousal": ccc_vals[1] if len(ccc_vals) > 1 else 0.0,
            "ccc_dominance": ccc_vals[2] if len(ccc_vals) > 2 else 0.0,
        }

        # Log TMC reliability scores
        if tmc_uncertainties:
            for mod_id, vals in tmc_uncertainties.items():
                mean_u = float(np.mean(vals))
                metrics[f"uncertainty_{mod_id}"] = mean_u
                log.info(f"  TMC uncertainty [{mod_id}]: {mean_u:.4f} (reliability={1-mean_u:.4f})")
            if tmc_combined_uncert:
                mean_cu = float(np.mean(tmc_combined_uncert))
                metrics["uncertainty_combined"] = mean_cu
                log.info(f"  TMC combined uncertainty: {mean_cu:.4f}")

        # Log distillation cosine similarities
        d_mod = getattr(fusion_model, 'fusion', None)
        if d_mod is not None and hasattr(d_mod, 'last_cosine_sims') and d_mod.last_cosine_sims:
            for mod_id, sim in d_mod.last_cosine_sims.items():
                metrics[f"cosine_sim_{mod_id}_video"] = sim
                log.info(f"  Distill cosine_sim [{mod_id}<->video]: {sim:.4f}")

        return metrics

    @staticmethod
    def _forward_fusion(
        fusion_model: nn.Module,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor | None],
        raw_signals: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if raw_signals is None:
            return fusion_model(embeddings, modality_ids, masks)
        return fusion_model(embeddings, modality_ids, masks, raw_signals=raw_signals)

    def _unpack_batch(
        self, batch: dict
    ) -> tuple[list[torch.Tensor], list[str], dict[str, torch.Tensor], list[torch.Tensor | None]]:
        embeddings = [
            batch["embeddings"][m].to(self.device) for m in batch["modality_ids"]
        ]
        masks_dict = batch.get("masks", {})
        masks = [
            masks_dict[m].to(self.device) if masks_dict.get(m) is not None else None
            for m in batch["modality_ids"]
        ]
        labels = {k: v.to(self.device) for k, v in batch["labels"].items()}
        return embeddings, batch["modality_ids"], labels, masks

    def _unpack_raw_signals(self, batch: dict) -> dict[str, torch.Tensor] | None:
        """Extract raw signals from batch, if present."""
        raw = batch.get("raw_signals")
        if not raw:
            return None
        return {k: v.to(self.device) for k, v in raw.items()}

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
            masks = {}
            for m in modality_ids:
                tensors = [
                    b["embeddings"][b["modality_ids"].index(m)] for b in batch
                ]
                if tensors[0].dim() <= 1:
                    # 0D or 1D (pooled vectors): simple stack → (B,) or (B, D)
                    embeddings[m] = torch.stack(tensors)
                    masks[m] = None
                elif all(t.shape[0] == tensors[0].shape[0] for t in tensors):
                    # 2D with uniform sequence length: simple stack → (B, T, D)
                    embeddings[m] = torch.stack(tensors)
                    masks[m] = None
                else:
                    # 2D with variable sequence length: pad + mask
                    max_t = max(t.shape[0] for t in tensors)
                    d = tensors[0].shape[-1]
                    padded = torch.zeros(len(tensors), max_t, d)
                    mask = torch.zeros(len(tensors), max_t, dtype=torch.bool)
                    for i, t in enumerate(tensors):
                        padded[i, : t.shape[0]] = t
                        mask[i, : t.shape[0]] = True
                    embeddings[m] = padded
                    masks[m] = mask
            labels = {}
            for k in batch[0]["labels"]:
                vals = [b["labels"][k] for b in batch]
                labels[k] = (
                    torch.stack(vals)
                    if isinstance(vals[0], torch.Tensor)
                    else torch.tensor(vals)
                )

            # Collate raw signals if present
            raw_signals = {}
            if batch[0].get("raw_signals"):
                for sig_name in batch[0]["raw_signals"]:
                    tensors = [b["raw_signals"][sig_name] for b in batch
                               if sig_name in b.get("raw_signals", {})]
                    if tensors:
                        raw_signals[sig_name] = torch.stack(tensors)

            result = {
                "embeddings": embeddings,
                "modality_ids": modality_ids,
                "labels": labels,
                "masks": masks,
            }
            if raw_signals:
                result["raw_signals"] = raw_signals
            return result

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
    ) -> list[dict[str, Any]]:
        """Run full LOSO cross-validation."""
        self._validate_loso_inputs(all_data, subject_ids)
        fold_results = []
        for i, test_subj in enumerate(subject_ids):
            seed = self.config["seed"] + i
            torch.manual_seed(seed)
            np.random.seed(seed)

            val_subj = subject_ids[(i + 1) % len(subject_ids)]
            train_subjects = [
                s for s in subject_ids if s not in (test_subj, val_subj)
            ]
            self._validate_loso_split(train_subjects, val_subj, test_subj)

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
            metrics["val_subject"] = val_subj
            metrics["train_subjects"] = list(train_subjects)
            metrics["n_train"] = len(train_data)
            metrics["n_val"] = len(val_data)
            metrics["n_test"] = len(test_data)
            fold_results.append(metrics)
            log.info(
                f"Fold {test_subj}: F1={metrics['weighted_f1']:.4f}, "
                f"CCC={metrics['ccc']:.4f}"
            )

        return fold_results

    @staticmethod
    def _validate_loso_inputs(
        all_data: dict[str, list[dict]],
        subject_ids: list[str],
    ) -> None:
        if len(subject_ids) < 3:
            raise ValueError("LOSO with held-out validation requires at least 3 subjects")

        duplicates = sorted({s for s in subject_ids if subject_ids.count(s) > 1})
        if duplicates:
            raise ValueError(f"Duplicate subject_ids are not allowed: {duplicates}")

        missing = [s for s in subject_ids if not all_data.get(s)]
        if missing:
            raise ValueError(f"subject_ids with no data: {missing}")

    @staticmethod
    def _validate_loso_split(
        train_subjects: list[str],
        val_subject: str,
        test_subject: str,
    ) -> None:
        train_set = set(train_subjects)
        if len(train_set) != len(train_subjects):
            raise ValueError(f"Duplicate train subjects in LOSO split: {train_subjects}")
        if val_subject == test_subject:
            raise ValueError(f"LOSO leakage: val_subject equals test_subject {test_subject}")
        if test_subject in train_set:
            raise ValueError(f"LOSO leakage: test_subject {test_subject} is in train_subjects")
        if val_subject in train_set:
            raise ValueError(f"LOSO leakage: val_subject {val_subject} is in train_subjects")
