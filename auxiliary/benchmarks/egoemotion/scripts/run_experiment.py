"""CLI for running fusion experiments end-to-end.

Usage:
    conda run -n visphy python scripts/run_experiment.py --fusion_config configs/fusion/early.yaml
    conda run -n visphy python scripts/run_experiment.py --fusion_config configs/fusion/perceiver_io.yaml --name my_experiment
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch

from mac.data.label_builder import build_label_mapping
from mac.data.segments import SegmentExtractor
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

# Re-export for historical script imports and serialized model references.
from mac.data.embedding_shapes import normalize_loaded_embedding
from mac.fusion.factory import ProjectedFusion, build_fusion_model


def load_data_by_subject(
    embeddings_dir: str,
    embedding_dir_names: list[str],
    config_modality_names: list[str],
    subject_ids: list[str],
    labels: dict,
) -> dict[str, list[dict]]:
    """Load cached embeddings organized by subject for LOSO.

    Args:
        embeddings_dir: Root dir for cached embeddings
        embedding_dir_names: Dir names in cache (e.g., ['patchtst_eye', 'papagei_ppg'])
        config_modality_names: Config modality names (e.g., ['eye_tracking', 'ppg'])
        subject_ids: List of subject IDs to load
        labels: Label mapping from build_label_mapping
    """
    embeddings_path = Path(embeddings_dir)
    data_by_subject: dict[str, list[dict]] = {}

    for subj in subject_ids:
        samples = []
        first_mod = embedding_dir_names[0]
        mod_dir = embeddings_path / first_mod / subj
        if not mod_dir.exists():
            continue

        seg_files = sorted(mod_dir.glob("segment_*.pt"))
        for f in seg_files:
            seg_idx = int(f.stem.split("_")[1])

            all_exist = all(
                (embeddings_path / m / subj / f.name).exists()
                for m in embedding_dir_names
            )
            if not all_exist:
                continue

            emb_list = []
            for emb_dir in embedding_dir_names:
                path = embeddings_path / emb_dir / subj / f"segment_{seg_idx:04d}.pt"
                data = torch.load(path, weights_only=False)
                emb_list.append(normalize_loaded_embedding(data["embedding"], emb_dir))

            key = f"{subj}_{seg_idx:04d}"
            if key not in labels:
                continue

            samples.append({
                "embeddings": emb_list,
                "modality_ids": config_modality_names,  # Use config names for projector
                "labels": labels[key],
            })

        if samples:
            data_by_subject[subj] = samples

    return data_by_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="auxiliary/benchmarks/egoemotion/configs/egoemotion.yaml",
    )
    parser.add_argument("--fusion_config", required=True, help="Path to fusion method config")
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Load and merge configs
    base_cfg = load_config(args.config)
    fusion_cfg = load_config(args.fusion_config)
    cfg = merge_configs(base_cfg, fusion_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"{cfg.get('fusion_type', 'experiment')}_{timestamp}"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logger = setup_logging(cfg["logging"]["log_dir"], name)
    logger.info(f"Starting experiment: {name}")
    logger.info(f"Config hash: {config_hash(cfg)}")
    logger.info(f"Device: {device}")
    logger.info(f"Fusion type: {cfg.get('fusion_type', 'unknown')}")

    # Setup modality registry
    registry = ModalityRegistry(cfg["modalities"])
    enabled = registry.get_enabled_modalities()
    logger.info(f"Enabled modalities: {enabled}")

    # Build models
    projector, fusion = build_fusion_model(cfg, registry)
    projected_fusion = ProjectedFusion(projector, fusion, enabled)
    task_head = MultiTaskHead(d_fused=fusion.d_out, num_emotions=9, num_vad=3)

    logger.info(f"Fusion params: {sum(p.numel() for p in projected_fusion.parameters()):,}")
    logger.info(f"Head params: {sum(p.numel() for p in task_head.parameters()):,}")

    # Build label mapping
    logger.info("Building label mapping...")
    labels = build_label_mapping(
        data_dir=cfg["data_dir"],
        task_times_path=f"{cfg['data_dir']}/task_times.npy",
    )
    logger.info(f"Labels: {len(labels)} segments")

    # Get subject list
    extractor = SegmentExtractor(
        data_dir=cfg["data_dir"],
        task_times_path=f"{cfg['data_dir']}/task_times.npy",
    )
    subject_ids = extractor.get_subject_ids()
    logger.info(f"Subjects: {len(subject_ids)}")

    # Map config modality names to embedding dir names
    embedding_dir_names = [registry.get_embedding_dir_name(m) for m in enabled]

    # Load data by subject
    logger.info("Loading cached embeddings...")
    data_by_subject = load_data_by_subject(
        embeddings_dir=cfg["embeddings_dir"],
        embedding_dir_names=embedding_dir_names,
        config_modality_names=enabled,
        subject_ids=subject_ids,
        labels=labels,
    )
    logger.info(f"Loaded data for {len(data_by_subject)} subjects, "
                f"{sum(len(v) for v in data_by_subject.values())} total samples")

    if not data_by_subject:
        logger.error("No data loaded! Run scripts/extract_embeddings.py first.")
        return

    # Create trainer
    trainer_config = {
        "training": cfg["training"],
        "loss_weights": cfg["loss_weights"],
        "seed": cfg["seed"],
    }
    if "cggm" in cfg:
        trainer_config["cggm"] = cfg["cggm"]
        logger.info(f"CGGM gradient modulation: {cfg['cggm']}")

    trainer = FusionTrainer(
        fusion_model=projected_fusion,
        task_head=task_head,
        config=trainer_config,
        device=device,
    )

    # Run LOSO
    logger.info("Starting LOSO cross-validation...")
    fold_results = trainer.run_loso(
        all_data=data_by_subject,
        subject_ids=list(data_by_subject.keys()),
    )

    if not fold_results:
        logger.error("No fold results! Check data and labels.")
        return

    # Aggregate results
    f1_scores = [r["weighted_f1"] for r in fold_results]
    ccc_scores = [r["ccc"] for r in fold_results]
    logger.info(f"\n{'='*50}")
    logger.info(f"LOSO Results ({len(fold_results)} folds):")
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
    registry_db = ResultsRegistry()
    registry_db.add(
        experiment_name=name,
        fusion_type=cfg.get("fusion_type", "unknown"),
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
