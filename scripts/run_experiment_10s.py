"""Run fusion experiments with task-aware 10s segments.

Uses task-aware segmentation (per-task chunking, no inter-task gaps).
Labels come from the task-aware manifest CSV.

Usage:
    conda run -n visphy python scripts/run_experiment_10s.py --fusion_config configs/fusion/early.yaml
    conda run -n visphy python scripts/run_experiment_10s.py --fusion_config configs/fusion/mid.yaml --name my_10s_exp
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.encoders.registry import ModalityRegistry
from src.fusion.projector import ModalityProjector
from src.tasks.heads import MultiTaskHead
from src.trainer.fusion_trainer import FusionTrainer
from src.utils.config import load_config, merge_configs, config_hash
from src.utils.logging_setup import setup_logging
from src.utils.registry import ResultsRegistry
from src.utils.reporting import generate_report

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fusion")

EMOTIONS = [
    "Amused", "Content", "Excited", "Awe", "Neutral",
    "Fear", "Sad", "Disgust", "Anger",
]

from scripts.run_experiment import normalize_loaded_embedding


# ---------------------------------------------------------------------------
# Model building: reuse the same ProjectedFusion from run_experiment.py
# ---------------------------------------------------------------------------

class ProjectedFusion(torch.nn.Module):
    """Wraps projector + fusion into a single module for the trainer."""

    def __init__(self, projector: ModalityProjector, fusion, modality_names: list[str]):
        super().__init__()
        self.projector = projector
        self.fusion = fusion
        self.modality_names = modality_names
        self.d_common = fusion.d_common
        self.d_out = fusion.d_out

    def forward(self, embeddings, modality_ids, masks=None):
        proj_dict = {}
        for emb, mod_id in zip(embeddings, modality_ids):
            proj_dict[mod_id] = emb
        projected = self.projector(proj_dict)
        proj_list = [projected[m] for m in modality_ids]

        # If the fusion module does not support sequence inputs, pool any
        # 3D (B, T, D) projected tensors to 2D (B, D) with mask-aware mean.
        if not getattr(self.fusion, "supports_sequence_input", False):
            pooled = []
            mask_list = masks if masks else [None] * len(proj_list)
            for proj, mask in zip(proj_list, mask_list):
                if proj.dim() == 3:
                    if mask is not None:
                        mask_f = mask.unsqueeze(-1).float()
                        proj = (proj * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                    else:
                        proj = proj.mean(dim=1)
                pooled.append(proj)
            proj_list = pooled
            masks = None

        return self.fusion(proj_list, modality_ids, masks)


def build_fusion_model(cfg: dict, enabled: list[str]) -> tuple:
    """Build a fusion model + projector from config. Returns (ProjectedFusion, d_out)."""
    from scripts.run_experiment import build_fusion_model as _build_core

    registry = ModalityRegistry(cfg["modalities"])
    projector, fusion = _build_core(cfg, registry)
    projected = ProjectedFusion(projector, fusion, enabled)
    return projected, fusion.d_out


# ---------------------------------------------------------------------------
# Data loading from task-aware embeddings + manifest
# ---------------------------------------------------------------------------

def load_10s_data_by_subject(
    embeddings_dir: str,
    encoder_dir_names: list[str],
    config_mod_names: list[str],
    manifest: pd.DataFrame,
    pool_clips: bool = False,
) -> dict[str, list[dict]]:
    """Load task-aware 10s embeddings organized by subject for LOSO.

    Labels (hard, soft, VAD) come directly from the task-aware manifest.
    """
    emb_path = Path(embeddings_dir)
    data_by_subject: dict[str, list[dict]] = {}

    for subj_id, subj_rows in manifest.groupby("subject"):
        subj = str(subj_id).zfill(3)
        samples = []

        for _, row in subj_rows.iterrows():
            global_seq = int(row["global_seq"])

            # Check all modalities exist
            all_exist = all(
                (emb_path / enc / subj / f"segment_{global_seq:04d}.pt").exists()
                for enc in encoder_dir_names
            )
            if not all_exist:
                continue

            emb_list = []
            for enc in encoder_dir_names:
                data = torch.load(
                    emb_path / enc / subj / f"segment_{global_seq:04d}.pt",
                    weights_only=False,
                )
                emb = normalize_loaded_embedding(data["embedding"], enc)
                if pool_clips and enc == "video_mae_v2" and emb.dim() == 2:
                    emb = emb.mean(dim=0)
                emb_list.append(emb)

            # Parse soft label from manifest (comma-separated string)
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
    parser = argparse.ArgumentParser(description="Run task-aware 10s-segment fusion experiments")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fusion_config", required=True, help="Path to fusion method config")
    parser.add_argument("--embeddings_dir", default="data/embeddings_10s_task_aware")
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--pool-clips", action="store_true",
                        help="Mean-pool video clip sequences to single [768] vectors")
    args = parser.parse_args()

    cfg = merge_configs(load_config(args.config), load_config(args.fusion_config))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"{cfg.get('fusion_type', 'exp')}_10s_{timestamp}"

    logger = setup_logging(cfg["logging"]["log_dir"], name)
    logger.info(f"Experiment: {name} (task-aware 10s segments)")
    logger.info(f"Config hash: {config_hash(cfg)}")
    logger.info(f"Device: {device}")

    # Load task-aware manifest
    manifest_path = Path(args.embeddings_dir) / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"subject": str})
    logger.info(f"Manifest: {len(manifest)} segments across {manifest['subject'].nunique()} subjects")

    # Setup modalities
    registry = ModalityRegistry(cfg["modalities"])
    enabled = registry.get_enabled_modalities()
    encoder_dirs = [registry.get_embedding_dir_name(m) for m in enabled]
    logger.info(f"Enabled modalities: {enabled}")

    # Build model
    projected_fusion, d_out = build_fusion_model(cfg, enabled)
    task_head = MultiTaskHead(d_fused=d_out, num_emotions=9, num_vad=3)

    logger.info(f"Fusion type: {cfg.get('fusion_type')}, d_out={d_out}")
    logger.info(f"Fusion params: {sum(p.numel() for p in projected_fusion.parameters()):,}")
    logger.info(f"Head params: {sum(p.numel() for p in task_head.parameters()):,}")

    # Load data
    data_by_subject = load_10s_data_by_subject(
        args.embeddings_dir, encoder_dirs, enabled, manifest,
        pool_clips=args.pool_clips,
    )
    total_samples = sum(len(v) for v in data_by_subject.values())
    logger.info(f"Loaded: {total_samples} samples across {len(data_by_subject)} subjects")

    if total_samples == 0:
        logger.error("No data! Run scripts/segment_and_extract_10s.py first.")
        return

    # Run LOSO
    logger.info("Starting LOSO cross-validation...")
    trainer = FusionTrainer(
        fusion_model=projected_fusion,
        task_head=task_head,
        config={
            "training": cfg["training"],
            "loss_weights": cfg["loss_weights"],
            "seed": cfg["seed"],
        },
        device=device,
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
    logger.info(f"Task-Aware 10s-Segment LOSO Results ({len(fold_results)} folds):")
    logger.info(f"  Weighted F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    logger.info(f"  CCC:         {np.mean(ccc_scores):.4f} +/- {np.std(ccc_scores):.4f}")
    logger.info(f"  Paper baseline (Classical, All): 0.46")
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
    registry = ResultsRegistry()
    registry.add(
        experiment_name=name,
        fusion_type=f"{cfg.get('fusion_type', 'unknown')}_10s_task_aware",
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
