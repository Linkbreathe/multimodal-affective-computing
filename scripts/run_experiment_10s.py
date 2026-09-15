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

from src.data.egoemotion import Ego10sLoadReport, load_egoemotion_10s_by_subject
from src.encoders.registry import ModalityRegistry
from src.fusion.projector import ModalityProjector
from src.tasks.heads import MultiTaskHead, TMCTaskHead
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

from src.data.embedding_shapes import normalize_loaded_embedding
from src.fusion.factory import ProjectedFusion


# ---------------------------------------------------------------------------
# Model building: shared ProjectedFusion from src.fusion.factory
# ---------------------------------------------------------------------------

def build_fusion_model(cfg: dict, enabled: list[str]) -> tuple:
    """Build a fusion model + projector from config. Returns (ProjectedFusion, d_out)."""
    from src.fusion.factory import build_fusion_model as _build_core

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
    data_dir_or_pool_clips: object | None = None,
    *,
    pool_clips: bool = False,
    max_missing_fraction: float = 0.01,
    require_manifest_hash: bool = False,
    return_report: bool = False,
) -> dict[str, list[dict]] | tuple[dict[str, list[dict]], Ego10sLoadReport]:
    """Load task-aware 10s embeddings organized by subject for LOSO.

    Labels (hard, soft, VAD) come directly from the task-aware manifest.
    """
    if isinstance(data_dir_or_pool_clips, bool):
        pool_clips = data_dir_or_pool_clips

    data_by_subject, report = load_egoemotion_10s_by_subject(
        embeddings_dir=embeddings_dir,
        encoder_dir_names=encoder_dir_names,
        config_mod_names=config_mod_names,
        manifest=manifest,
        normalize_embedding=normalize_loaded_embedding,
        pool_clips=pool_clips,
        max_missing_fraction=max_missing_fraction,
        require_manifest_hash=require_manifest_hash,
    )
    return (data_by_subject, report) if return_report else data_by_subject


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run task-aware 10s-segment fusion experiments")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fusion_config", required=True, help="Path to fusion method config")
    parser.add_argument("--embeddings_dir", default="data/embeddings/egoemotion/10s_task_aware")
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--pool-clips", action="store_true",
                        help="Mean-pool video clip sequences to single [768] vectors")
    parser.add_argument("--max-missing-fraction", type=float, default=0.01,
                        help="Maximum manifest-row fraction allowed to miss any enabled modality")
    parser.add_argument("--require-manifest-hash", action="store_true",
                        help="Require each embedding payload to match the manifest content hash")
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
    if cfg.get("fusion_type") == "tmc":
        task_head = TMCTaskHead(num_emotions=9, num_vad=3)
    else:
        task_head = MultiTaskHead(d_fused=d_out, num_emotions=9, num_vad=3)

    logger.info(f"Fusion type: {cfg.get('fusion_type')}, d_out={d_out}")
    logger.info(f"Fusion params: {sum(p.numel() for p in projected_fusion.parameters()):,}")
    logger.info(f"Head params: {sum(p.numel() for p in task_head.parameters()):,}")

    # Load data
    data_by_subject, load_report = load_10s_data_by_subject(
        args.embeddings_dir, encoder_dirs, enabled, manifest,
        pool_clips=args.pool_clips,
        max_missing_fraction=args.max_missing_fraction,
        require_manifest_hash=args.require_manifest_hash,
        return_report=True,
    )
    total_samples = sum(len(v) for v in data_by_subject.values())
    logger.info(f"Loaded: {total_samples} samples across {len(data_by_subject)} subjects")
    logger.info(
        "Loader report: manifest_rows=%d, loaded_rows=%d, dropped_rows=%d, "
        "subjects=%d/%d, missing_fraction=%.4f, missing_by_encoder=%s",
        load_report.manifest_rows,
        load_report.loaded_rows,
        load_report.dropped_rows,
        load_report.subjects_loaded,
        load_report.subjects_in_manifest,
        load_report.missing_fraction,
        load_report.missing_by_encoder,
    )

    if total_samples == 0:
        logger.error("No data! Run scripts/segment_and_extract_10s.py first.")
        return

    # Run LOSO
    logger.info("Starting LOSO cross-validation...")
    trainer_config = {
        "training": cfg["training"],
        "loss_weights": cfg["loss_weights"],
        "seed": cfg["seed"],
    }
    if "cggm" in cfg:
        trainer_config["cggm"] = cfg["cggm"]
        logger.info(f"CGGM gradient modulation: {cfg['cggm']}")
    if cfg.get("fusion_type") == "tmc":
        tmc_cfg = cfg["fusion"].get("tmc", {})
        trainer_config["tmc"] = {
            "enabled": True,
            "annealing_epochs": tmc_cfg.get("annealing_epochs", 10),
            "num_classes": tmc_cfg.get("num_classes", 9),
        }
        logger.info(f"TMC evidential fusion: annealing_epochs={tmc_cfg.get('annealing_epochs', 10)}")
    if cfg.get("fusion_type") == "distill_late":
        dcfg = cfg["fusion"].get("distill", {})
        trainer_config["distill"] = {
            "enabled": True,
            "lambda_distill": dcfg.get("lambda_distill", 1.0),
        }
        logger.info(f"Distillation: tau={dcfg.get('tau', 3.0)}, lambda={dcfg.get('lambda_distill', 1.0)}")

    trainer = FusionTrainer(
        fusion_model=projected_fusion,
        task_head=task_head,
        config=trainer_config,
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
        run_metadata={"egoemotion_10s_loader": load_report.as_dict()},
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
