"""Fine-tune PPG encoder end-to-end with frozen pre-extracted video/eye embeddings.

Loads raw PPG segments from ppg_ear_125hz.npy and runs the PPG encoder
inside the training loop WITH gradients on selectively unfrozen layers.
Video and eye-tracking modalities use pre-extracted embeddings as before.

Usage:
    conda run -n visphy python scripts/run_finetune_ppg.py \
        --fusion_config auxiliary/benchmarks/egoemotion/configs/finetune/ppg_only.yaml
    conda run -n visphy python scripts/run_finetune_ppg.py \
        --fusion_config auxiliary/benchmarks/egoemotion/configs/finetune/video_ppg_finetune.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from mac.encoders.registry import ModalityRegistry
from mac.fusion.projector import ModalityProjector
from mac.tasks.heads import MultiTaskHead
from mac.training.fusion_trainer import FusionTrainer
from mac.config.simple import load_config, merge_configs, config_hash
from mac.utils.logging_setup import setup_logging
from mac.reporting.registry import ResultsRegistry
from mac.reporting.experiment import generate_report

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fusion")

CHUNK_LEN_SEC = 10
FS_PPG = 125
CHUNK_SAMPLES_PPG = CHUNK_LEN_SEC * FS_PPG  # 1250

from mac.data.embedding_shapes import normalize_loaded_embedding
from mac.fusion.factory import ProjectedFusion


# ---------------------------------------------------------------------------
# Model building — reuse existing fusion factory
# ---------------------------------------------------------------------------

class HybridProjectedFusion(ProjectedFusion):
    """Wraps PPG encoder + projector + fusion for end-to-end fine-tuning.

    Pre-extracted modalities (video, eye) pass through the projector as before.
    Raw modalities (ppg) are first encoded, then projected.
    The encoder is a submodule so it participates in deepcopy and state_dict.
    """

    def __init__(
        self,
        projector: ModalityProjector,
        fusion,
        modality_names: list[str],
        encoders: dict[str, torch.nn.Module] | None = None,
    ):
        super().__init__(projector, fusion, modality_names)

        # Register encoders as submodules for deepcopy/state_dict
        self.encoders = torch.nn.ModuleDict(encoders or {})

    def forward(self, embeddings, modality_ids, masks=None, raw_signals=None):
        """Forward pass with optional raw signal encoding.

        Args:
            embeddings: List of pre-extracted embedding tensors.
            modality_ids: List of modality names.
            masks: Optional masks for variable-length sequences.
            raw_signals: Dict of {modality_name: raw_tensor} for encoder pass.
        """
        if raw_signals:
            # Run raw signals through their encoders (WITH gradients)
            emb_list = list(embeddings)
            for i, mod_id in enumerate(modality_ids):
                if mod_id in raw_signals and mod_id in self.encoders:
                    encoded = self.encoders[mod_id](raw_signals[mod_id])
                    emb_list[i] = encoded
            embeddings = emb_list

        proj_list, masks = self.project_and_pool(embeddings, modality_ids, masks)
        return self.fusion(proj_list, modality_ids, masks)


def build_finetune_model(cfg: dict, enabled: list[str], raw_modalities: list[str], device: str):
    """Build a HybridProjectedFusion with encoders for raw modalities.

    Returns (model, encoder_param_groups, d_out).
    """
    from mac.fusion.factory import build_fusion_model as _build_core

    registry = ModalityRegistry(cfg["modalities"])
    projector, fusion = _build_core(cfg, registry)

    # Build encoders for raw modalities
    encoders = {}
    for mod_name in raw_modalities:
        if mod_name not in enabled:
            continue
        encoder_cls = registry.get_encoder_class(mod_name)
        encoder = encoder_cls()
        # Unfreeze specified layers
        ft_cfg = cfg.get("finetune", {})
        from_block = ft_cfg.get("unfreeze_from_block")
        if from_block is not None:
            encoder.unfreeze(from_layer=from_block)
        else:
            encoder.unfreeze()
        encoders[mod_name] = encoder

    model = HybridProjectedFusion(projector, fusion, enabled, encoders)

    # Build encoder param groups with layer-wise LR
    encoder_base_lr = cfg.get("finetune", {}).get("encoder_base_lr", 1e-5)
    encoder_param_groups = []
    for mod_name, encoder in encoders.items():
        for group in encoder.get_layer_groups():
            encoder_param_groups.append({
                "name": f"{mod_name}_{group['name']}",
                "params": group["params"],
                "lr": encoder_base_lr * group["lr_scale"],
            })

    return model, encoder_param_groups, fusion.d_out


# ---------------------------------------------------------------------------
# Data loading — hybrid: pre-extracted embeddings + raw PPG
# ---------------------------------------------------------------------------

def load_hybrid_data_by_subject(
    embeddings_dir: str,
    data_dir: str,
    encoder_dir_names: list[str],
    config_mod_names: list[str],
    raw_modalities: list[str],
    manifest: pd.DataFrame,
    pool_clips: bool = False,
) -> dict[str, list[dict]]:
    """Load data with pre-extracted embeddings for frozen modalities
    and raw signal references for fine-tuned modalities.

    For raw modalities: stores segment boundaries so the trainer can
    load raw signals on-the-fly. For pre-extracted: loads .pt as before.
    """
    emb_path = Path(embeddings_dir)
    raw_path = Path(data_dir)
    data_by_subject: dict[str, list[dict]] = {}

    # Determine which modalities are raw vs pre-extracted
    raw_set = set(raw_modalities)

    # Build mapping: config_mod_name -> encoder_dir_name
    mod_to_dir = dict(zip(config_mod_names, encoder_dir_names))

    # Pre-extracted modalities (need .pt files on disk)
    preextracted_mods = [m for m in config_mod_names if m not in raw_set]
    preextracted_dirs = [mod_to_dir[m] for m in preextracted_mods]

    for subj_id, subj_rows in manifest.groupby("subject"):
        subj = str(subj_id).zfill(3)
        samples = []

        # For raw PPG: load the full signal once per subject
        ppg_signal = None
        if "ppg" in raw_set:
            ppg_path = raw_path / subj / "ppg_ear_125hz.npy"
            if ppg_path.exists():
                ppg_signal = np.load(ppg_path)
            else:
                log.warning(f"PPG not found for {subj}: {ppg_path}")

        for _, row in subj_rows.iterrows():
            global_seq = int(row["global_seq"])

            # Check pre-extracted embeddings exist
            all_exist = all(
                (emb_path / enc / subj / f"segment_{global_seq:04d}.pt").exists()
                for enc in preextracted_dirs
            )
            if not all_exist:
                continue

            # Load pre-extracted embeddings
            emb_list = []
            for mod_name in config_mod_names:
                if mod_name in raw_set:
                    # Placeholder — will be replaced by encoder output in forward()
                    emb_list.append(torch.zeros(1))
                else:
                    enc_dir = mod_to_dir[mod_name]
                    data = torch.load(
                        emb_path / enc_dir / subj / f"segment_{global_seq:04d}.pt",
                        weights_only=False,
                    )
                    emb = normalize_loaded_embedding(data["embedding"], enc_dir)
                    if pool_clips and enc_dir == "video_mae_v2" and emb.dim() == 2:
                        emb = emb.mean(dim=0)
                    emb_list.append(emb)

            # Load raw PPG segment
            raw_signals = {}
            if "ppg" in raw_set and ppg_signal is not None:
                start_90hz = int(row["start_90hz"])
                start_125 = int(start_90hz * FS_PPG / 90)
                end_125 = start_125 + CHUNK_SAMPLES_PPG

                if end_125 > len(ppg_signal):
                    continue

                chunk = ppg_signal[start_125:end_125].copy()
                # Per-segment z-score normalization (matches extraction pipeline)
                std = chunk.std()
                if std > 0:
                    chunk = (chunk - chunk.mean()) / std
                # [1, T] channels-first for Conv1d
                raw_signals["ppg"] = torch.tensor(
                    chunk, dtype=torch.float32
                ).unsqueeze(0)

            # Parse labels
            soft_label = torch.tensor(
                [float(v) for v in str(row["soft_label"]).split(",")],
                dtype=torch.float32,
            )
            vad = torch.tensor([
                float(row["valence"]),
                float(row["arousal"]),
                float(row["dominance"]),
            ], dtype=torch.float32)

            samples.append({
                "embeddings": emb_list,
                "modality_ids": config_mod_names,
                "raw_signals": raw_signals,
                "labels": {
                    "emotion_label": torch.tensor(int(row["emotion_label"]), dtype=torch.long),
                    "soft_label": soft_label,
                    "vad": vad,
                },
            })

        if samples:
            data_by_subject[subj] = samples

    return data_by_subject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PPG encoder with fusion")
    parser.add_argument(
        "--config",
        default="auxiliary/benchmarks/egoemotion/configs/egoemotion.yaml",
    )
    parser.add_argument("--fusion_config", required=True, help="Path to finetune config")
    parser.add_argument("--embeddings_dir", default="data/embeddings/egoemotion/10s_task_aware")
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--pool-clips", action="store_true",
                        help="Mean-pool video clip sequences to single [768] vectors")
    args = parser.parse_args()

    cfg = merge_configs(load_config(args.config), load_config(args.fusion_config))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"finetune_{cfg.get('fusion_type', 'exp')}_{timestamp}"

    logger = setup_logging(cfg["logging"]["log_dir"], name)
    logger.info(f"Experiment: {name} (PPG fine-tuning)")
    logger.info(f"Config hash: {config_hash(cfg)}")
    logger.info(f"Device: {device}")

    # Finetune config
    ft_cfg = cfg.get("finetune", {})
    raw_modalities = ft_cfg.get("raw_modalities", [])
    logger.info(f"Raw (fine-tuned) modalities: {raw_modalities}")
    logger.info(f"Unfreeze from block: {ft_cfg.get('unfreeze_from_block', 'all')}")
    logger.info(f"Encoder base LR: {ft_cfg.get('encoder_base_lr', 1e-5)}")
    logger.info(f"Warmup epochs: {ft_cfg.get('warmup_epochs', 0)}")

    # Load manifest
    manifest_path = Path(args.embeddings_dir) / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"subject": str})

    # Verify manifest has start_90hz column
    if "start_90hz" not in manifest.columns:
        logger.error(
            "Manifest missing start_90hz/end_90hz columns. "
            "Re-run scripts/segment_and_extract_10s.py to regenerate."
        )
        return

    logger.info(f"Manifest: {len(manifest)} segments across {manifest['subject'].nunique()} subjects")

    # Setup modalities
    registry = ModalityRegistry(cfg["modalities"])
    enabled = registry.get_enabled_modalities()
    encoder_dirs = [registry.get_embedding_dir_name(m) for m in enabled]
    logger.info(f"Enabled modalities: {enabled}")

    # Build model with encoders
    model, encoder_param_groups, d_out = build_finetune_model(
        cfg, enabled, raw_modalities, device,
    )
    task_head = MultiTaskHead(d_fused=d_out, num_emotions=9, num_vad=3)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Fusion type: {cfg.get('fusion_type')}, d_out={d_out}")
    logger.info(f"Total params: {total_params:,} (trainable: {trainable_params:,})")
    logger.info(f"Head params: {sum(p.numel() for p in task_head.parameters()):,}")
    for pg in encoder_param_groups:
        n = sum(p.numel() for p in pg["params"])
        logger.info(f"  Encoder group '{pg['name']}': {n:,} params, lr={pg['lr']:.2e}")

    # Load hybrid data
    data_by_subject = load_hybrid_data_by_subject(
        args.embeddings_dir, cfg["data_dir"],
        encoder_dirs, enabled, raw_modalities, manifest,
        pool_clips=args.pool_clips,
    )
    total_samples = sum(len(v) for v in data_by_subject.values())
    logger.info(f"Loaded: {total_samples} samples across {len(data_by_subject)} subjects")

    if total_samples == 0:
        logger.error("No data! Check embeddings and raw data paths.")
        return

    # Run LOSO
    logger.info("Starting LOSO cross-validation with PPG fine-tuning...")
    trainer_config = {
        "training": cfg["training"],
        "loss_weights": cfg["loss_weights"],
        "seed": cfg["seed"],
        "finetune": ft_cfg,
    }

    trainer = FusionTrainer(
        fusion_model=model,
        task_head=task_head,
        config=trainer_config,
        device=device,
        encoder_param_groups=encoder_param_groups,
    )

    fold_results = trainer.run_loso(
        all_data=data_by_subject,
        subject_ids=list(data_by_subject.keys()),
    )

    if not fold_results:
        logger.error("No fold results! Check data and labels.")
        return

    # Report
    f1_scores = [r["weighted_f1"] for r in fold_results]
    ccc_scores = [r["ccc"] for r in fold_results]
    logger.info(f"\n{'='*50}")
    logger.info(f"PPG Fine-Tune LOSO Results ({len(fold_results)} folds):")
    logger.info(f"  Weighted F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    logger.info(f"  CCC:         {np.mean(ccc_scores):.4f} +/- {np.std(ccc_scores):.4f}")
    logger.info(f"{'='*50}")

    # Save report
    report_path = generate_report(
        experiment_name=name,
        config=cfg,
        fold_results=fold_results,
        report_dir=cfg["logging"]["report_dir"],
    )
    logger.info(f"Report saved: {report_path}")

    # Update registry
    results_registry = ResultsRegistry()
    results_registry.add(
        experiment_name=name,
        fusion_type=f"finetune_{cfg.get('fusion_type', 'unknown')}",
        metrics={
            "weighted_f1": float(np.mean(f1_scores)),
            "weighted_f1_std": float(np.std(f1_scores)),
            "ccc": float(np.mean(ccc_scores)),
            "ccc_std": float(np.std(ccc_scores)),
        },
        config_hash=config_hash(cfg),
    )
    logger.info("Results registered.")


if __name__ == "__main__":
    main()
