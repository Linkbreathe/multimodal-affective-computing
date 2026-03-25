"""Run fusion experiments with paper-matched 10s segments.

Uses the exact same 2,678 segments from the paper's manifest.
Results are directly comparable to the paper's 0.46 F1 baseline.

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
# Data loading from 10s embeddings + manifest labels
# ---------------------------------------------------------------------------

def _load_kl_soft_labels(data_dir: Path) -> dict[tuple[int, int], np.ndarray]:
    """Load per-segment soft labels from KL manifest. Key=(subject, seg_idx)."""
    kl_path = data_dir / "kl_softlabel_manifests" / "dataset_manifest.csv"
    if not kl_path.exists():
        return {}
    kl = pd.read_csv(kl_path)
    result = {}
    for _, row in kl.iterrows():
        subj = int(row["subject"])
        seg_idx = int(row["segment_path"].split("/")[-1].replace(".p", ""))
        soft = np.array([float(row.get(e, 0.0)) for e in EMOTIONS], dtype=np.float32)
        total = soft.sum()
        if total > 0:
            soft /= total
        result[(subj, seg_idx)] = soft
    return result


def _load_vad_scores(data_dir: Path) -> dict[tuple[int, int], np.ndarray]:
    """Load per-segment VAD scores from VAD manifest. Key=(subject, seg_idx)."""
    vad_path = data_dir / "vad_binary_quadrant_manifests" / "dataset_manifest.csv"
    if not vad_path.exists():
        return {}
    vad = pd.read_csv(vad_path)
    result = {}
    for _, row in vad.iterrows():
        subj = int(row["subject"])
        seg_idx = int(row["segment_path"].split("/")[-1].replace(".p", ""))
        scores = np.array([
            float(row.get("valence_score", 0.0)),
            float(row.get("arousal_score", 0.0)),
            float(row.get("dominance_score", 0.0)),
        ], dtype=np.float32)
        result[(subj, seg_idx)] = scores
    return result


def load_10s_data_by_subject(
    embeddings_dir: str,
    encoder_dir_names: list[str],
    config_mod_names: list[str],
    manifest: pd.DataFrame,
    data_dir: Path,
    pool_clips: bool = False,
) -> dict[str, list[dict]]:
    """Load 10s-segmented embeddings organized by subject for LOSO.

    Uses actual soft labels from KL manifest and VAD scores from VAD manifest
    rather than one-hot approximations.
    """
    emb_path = Path(embeddings_dir)
    kl_labels = _load_kl_soft_labels(data_dir)
    vad_scores = _load_vad_scores(data_dir)

    data_by_subject: dict[str, list[dict]] = {}

    for subj_id, subj_rows in manifest.groupby("subject"):
        subj = str(subj_id).zfill(3)
        samples = []

        for _, row in subj_rows.iterrows():
            seg_idx = int(row["segment_path"].split("/")[-1].replace(".p", ""))

            # Check all modalities exist
            all_exist = all(
                (emb_path / enc / subj / f"segment_{seg_idx:04d}.pt").exists()
                for enc in encoder_dir_names
            )
            if not all_exist:
                continue

            emb_list = []
            for enc in encoder_dir_names:
                data = torch.load(
                    emb_path / enc / subj / f"segment_{seg_idx:04d}.pt",
                    weights_only=False,
                )
                # Step 1: normalize (strips batch-dim artifact)
                emb = normalize_loaded_embedding(data["embedding"], enc)
                # Step 2: pool video clips (video-only, after normalize)
                if pool_clips and enc == "video_mae_v2" and emb.dim() == 2:
                    emb = emb.mean(dim=0)  # [T, 768] -> [768]
                emb_list.append(emb)

            # Soft label from KL manifest (fall back to one-hot if missing)
            kl_key = (int(subj_id), seg_idx)
            if kl_key in kl_labels:
                soft_label = torch.tensor(kl_labels[kl_key], dtype=torch.float32)
            else:
                soft_label = torch.zeros(9, dtype=torch.float32)
                soft_label[int(row["label"])] = 1.0

            # VAD from VAD manifest (fall back to zeros if missing)
            vad_key = (int(subj_id), seg_idx)
            vad = torch.tensor(vad_scores.get(vad_key, np.zeros(3, dtype=np.float32)), dtype=torch.float32)

            samples.append({
                "embeddings": emb_list,
                "modality_ids": config_mod_names,
                "labels": {
                    "emotion_label": torch.tensor(int(row["label"]), dtype=torch.long),
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
    parser = argparse.ArgumentParser(description="Run 10s-segment fusion experiments")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fusion_config", required=True, help="Path to fusion method config")
    parser.add_argument("--embeddings_dir", default="data/embeddings_10s")
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
    logger.info(f"Experiment: {name} (10s segments, paper-matched)")
    logger.info(f"Config hash: {config_hash(cfg)}")
    logger.info(f"Device: {device}")

    # Load manifest
    data_dir = Path(cfg["data_dir"])
    manifest = pd.read_csv(data_dir / "ce_hardlabel_manifests" / "dataset_manifest.csv")
    logger.info(f"Manifest: {len(manifest)} segments across {manifest['subject'].nunique()} subjects")

    # Setup modalities
    enabled = [m for m, c in cfg["modalities"].items() if c.get("enabled", False)]
    encoder_map = {"video": "video_mae_v2", "eye_tracking": "patchtst_eye", "ppg": "papagei_ppg"}
    encoder_dirs = [encoder_map[m] for m in enabled]
    logger.info(f"Enabled modalities: {enabled}")

    # Build model
    projected_fusion, d_out = build_fusion_model(cfg, enabled)
    task_head = MultiTaskHead(d_fused=d_out, num_emotions=9, num_vad=3)

    logger.info(f"Fusion type: {cfg.get('fusion_type')}, d_out={d_out}")
    logger.info(f"Fusion params: {sum(p.numel() for p in projected_fusion.parameters()):,}")
    logger.info(f"Head params: {sum(p.numel() for p in task_head.parameters()):,}")

    # Load data
    data_by_subject = load_10s_data_by_subject(
        args.embeddings_dir, encoder_dirs, enabled, manifest, data_dir,
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
    logger.info(f"10s-Segment LOSO Results ({len(fold_results)} folds):")
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
        fusion_type=f"{cfg.get('fusion_type', 'unknown')}_10s",
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
