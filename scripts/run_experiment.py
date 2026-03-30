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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data.label_builder import build_label_mapping
from src.data.segments import SegmentExtractor
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

SEQUENCE_ENCODER_DIRS = {"video_mae_v2", "patchtst_eye"}


def normalize_loaded_embedding(
    embedding: torch.Tensor,
    encoder_dir_name: str,
) -> torch.Tensor:
    """Normalize cached embeddings without collapsing singleton sequences.

    Sequence encoders may legitimately emit a single token with shape [1, D].
    Keep that 2D shape intact so batching can still pad/stack sequence inputs.
    Pooled encoders such as PPG should continue to load as [D].
    """
    if embedding.dim() >= 3 and embedding.shape[0] == 1:
        return embedding.squeeze(0)

    if (
        encoder_dir_name not in SEQUENCE_ENCODER_DIRS
        and embedding.dim() == 2
        and embedding.shape[0] == 1
    ):
        return embedding.squeeze(0)

    return embedding


def build_fusion_model(cfg: dict, registry: ModalityRegistry):
    """Build a fusion model + projector from config."""
    fusion_type = cfg.get("fusion_type", "early")
    d_common = cfg["fusion"]["d_common"]
    dropout = cfg["fusion"].get("dropout", 0.1)
    enabled = registry.get_enabled_modalities()
    embed_dims = registry.get_all_embed_dims()

    # Build projector
    projector = ModalityProjector(embed_dims=embed_dims, d_common=d_common)

    # Build fusion module
    if fusion_type == "early":
        from src.fusion.early import EarlyFusion
        fusion = EarlyFusion(d_common=d_common, num_modalities=len(enabled), dropout=dropout)

    elif fusion_type == "mid":
        from src.fusion.mid import MidFusion
        fusion = MidFusion(d_common=d_common, modality_ids=enabled, dropout=dropout)

    elif fusion_type == "late":
        from src.fusion.late import LateFusion
        mode = cfg["fusion"].get("mode", "weighted")
        fusion = LateFusion(d_common=d_common, num_modalities=len(enabled), mode=mode, dropout=dropout)

    elif fusion_type == "perceiver_io":
        from src.fusion.perceiver_io import PerceiverIOFusion
        pcfg = cfg["fusion"].get("perceiver", {})
        fusion = PerceiverIOFusion(
            d_common=d_common,
            n_latents=pcfg.get("n_latents", 32),
            d_latent=pcfg.get("d_latent", d_common),
            n_layers=pcfg.get("n_layers", 2),
            n_heads=pcfg.get("n_heads", 4),
            dropout=pcfg.get("dropout", dropout),
        )

    elif fusion_type == "qformer":
        from src.fusion.qformer import QFormerFusion
        qcfg = cfg["fusion"].get("qformer", {})
        fusion = QFormerFusion(
            d_common=d_common,
            n_queries=qcfg.get("n_queries", 16),
            d_query=qcfg.get("d_query", d_common),
            n_layers=qcfg.get("n_layers", 6),
            n_heads=qcfg.get("n_heads", 4),
            cross_attn_freq=qcfg.get("cross_attn_freq", 2),
            dropout=qcfg.get("dropout", dropout),
        )

    elif fusion_type == "healnet":
        from src.fusion.healnet import HEALNetFusion
        hcfg = cfg["fusion"].get("healnet", {})
        fusion = HEALNetFusion(
            d_common=d_common,
            memory_size=hcfg.get("memory_size", 16),
            n_layers=hcfg.get("n_layers", 2),
            n_heads=hcfg.get("n_heads", 4),
            num_modalities=len(enabled),
            dropout=hcfg.get("dropout", dropout),
        )

    elif fusion_type == "multimodal_lego":
        from src.fusion.multimodal_lego import MultimodalLegoFusion
        lcfg = cfg["fusion"].get("lego", {})
        fusion = MultimodalLegoFusion(
            d_common=d_common,
            modality_ids=enabled,
            mode=lcfg.get("mode", "merge-sum"),
            latent_channels=lcfg.get("latent_channels", 64),
            latent_dim=lcfg.get("latent_dim", 64),
            depth=lcfg.get("depth", 2),
            heads=lcfg.get("heads", 8),
            dim_head=lcfg.get("dim_head", 64),
            attn_dropout=lcfg.get("attn_dropout", 0.0),
            ff_dropout=lcfg.get("ff_dropout", 0.0),
            frequency_domain=lcfg.get("frequency_domain", True),
            fourier_dim=lcfg.get("fourier_dim", 1),
            track_imaginary=lcfg.get("track_imaginary", True),
            normalise=lcfg.get("normalise", True),
            alpha=lcfg.get("alpha", 0.5),
        )

    elif fusion_type == "bottleneck":
        from src.fusion.bottleneck import BottleneckFusion
        bcfg = cfg["fusion"].get("bottleneck", {})
        fusion = BottleneckFusion(
            d_common=d_common,
            num_modalities=len(enabled),
            bottleneck_ratio=bcfg.get("bottleneck_ratio", 0.5),
            dropout=bcfg.get("dropout", dropout),
        )

    elif fusion_type == "tmc":
        from src.fusion.tmc import TMCFusion
        tcfg = cfg["fusion"].get("tmc", {})
        fusion = TMCFusion(
            d_common=d_common,
            num_classes=tcfg.get("num_classes", 9),
            num_modalities=len(enabled),
            dropout=cfg["fusion"].get("dropout", dropout),
        )

    elif fusion_type == "distill_late":
        from src.fusion.distill_late import DistillLateFusion, EnrichedModalityProjector
        dcfg = cfg["fusion"].get("distill", {})
        enriched_mods = dcfg.get("enriched_modalities", [])
        # Override projector with enriched version
        projector = EnrichedModalityProjector(
            embed_dims=embed_dims,
            d_common=d_common,
            enriched_modalities=enriched_mods,
        )
        fusion = DistillLateFusion(
            d_common=d_common,
            num_modalities=len(enabled),
            num_classes=dcfg.get("num_classes", 9),
            tau=dcfg.get("tau", 3.0),
            enriched_modalities=enriched_mods,
            teacher_modality=dcfg.get("teacher_modality", "video"),
            dropout=cfg["fusion"].get("dropout", dropout),
        )
        fusion.register_modality_classifiers(enabled)

    elif fusion_type == "enriched_late":
        from src.fusion.distill_late import EnrichedModalityProjector
        from src.fusion.late import LateFusion
        ecfg = cfg["fusion"].get("enriched", {})
        enriched_mods = ecfg.get("enriched_modalities", [])
        proj_dropout = ecfg.get("projection_dropout", 0.0)
        projector = EnrichedModalityProjector(
            embed_dims=embed_dims,
            d_common=d_common,
            enriched_modalities=enriched_mods,
            dropout=proj_dropout,
        )
        fusion = LateFusion(
            d_common=d_common,
            num_modalities=len(enabled),
            mode=cfg["fusion"].get("mode", "weighted"),
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    return projector, fusion


class ProjectedFusion(torch.nn.Module):
    """Wraps projector + fusion into a single module for the trainer."""

    def __init__(self, projector: ModalityProjector, fusion, modality_names: list[str]):
        super().__init__()
        self.projector = projector
        self.fusion = fusion
        self.modality_names = modality_names
        # Match BaseFusionModule interface
        self.d_common = fusion.d_common
        self.d_out = fusion.d_out

    def project_and_pool(self, embeddings, modality_ids, masks=None):
        """Project raw embeddings and optionally pool sequences.

        Returns (proj_list, masks) where proj_list contains the projected
        (and pooled, if the fusion module does not support sequences) tensors.
        Useful for inserting gradient hooks between projection and fusion.
        """
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
                        # mask: (B, T) bool — True = valid token
                        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
                        proj = (proj * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                    else:
                        proj = proj.mean(dim=1)
                pooled.append(proj)
            proj_list = pooled
            masks = None

        return proj_list, masks

    def forward(self, embeddings, modality_ids, masks=None, raw_signals=None):
        proj_list, masks = self.project_and_pool(embeddings, modality_ids, masks)
        return self.fusion(proj_list, modality_ids, masks)


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
    parser.add_argument("--config", default="configs/base.yaml")
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
