"""Run MM-Lego experiments with pre-trained blocks (correct paradigm).

Usage:
  # First pre-train blocks:
  python scripts/pretrain_lego_blocks.py

  # Then run merge (zero-shot) or fuse (light fine-tuning):
  python scripts/run_lego_experiment.py --mode merge-sum
  python scripts/run_lego_experiment.py --mode fuse-stack --tune_epochs 20
"""
import sys, argparse, logging
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mac.data.segments import SegmentExtractor
from mac.data.label_builder import build_label_mapping
from mac.fusion.multimodal_lego import LegoBlock, MultimodalLegoFusion
from mac.config.simple import load_config, merge_configs, config_hash
from mac.evaluation.metrics import weighted_f1_score, compute_class_weights
from mac.utils.logging_setup import setup_logging
from mac.reporting.experiment import generate_report
from mac.reporting.registry import ResultsRegistry

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fusion")

LEGO_GLOBAL_PRETRAIN_WARNING = (
    "MM-Lego blocks from pretrain_lego_blocks.py are produced by global supervised "
    "pretraining across all subjects. Reusing them inside LOSO can leak held-out "
    "subject label information. Pass --allow-global-pretrained-blocks only for a "
    "clearly labeled ablation, not for a clean LOSO result."
)


def require_lego_pretrain_leakage_acknowledgement(allow_global_pretrained_blocks: bool) -> None:
    if not allow_global_pretrained_blocks:
        raise RuntimeError(LEGO_GLOBAL_PRETRAIN_WARNING)


class MultimodalDataset(Dataset):
    """Loads embeddings for all modalities with labels."""
    def __init__(self, embeddings_dir, encoder_dir_names, subject_ids, labels):
        self.samples = []
        emb_path = Path(embeddings_dir)
        first_mod = encoder_dir_names[0]
        for subj in subject_ids:
            subj_dir = emb_path / first_mod / subj
            if not subj_dir.exists():
                continue
            for f in sorted(subj_dir.glob("segment_*.pt")):
                seg_idx = int(f.stem.split("_")[1])
                key = f"{subj}_{seg_idx:04d}"
                if key not in labels:
                    continue
                all_exist = all(
                    (emb_path / m / subj / f.name).exists() for m in encoder_dir_names
                )
                if not all_exist:
                    continue
                embs = []
                for m in encoder_dir_names:
                    data = torch.load(emb_path / m / subj / f"segment_{seg_idx:04d}.pt", weights_only=False)
                    emb = data["embedding"]
                    if emb.dim() == 2:
                        emb = emb.mean(dim=0)
                    embs.append(emb)
                self.samples.append((embs, labels[key]["emotion_label"], subj))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        embs, label, subj = self.samples[idx]
        return embs, label, subj


def load_pretrained_blocks(save_dir, modality_names, cfg, device):
    """Load pre-trained LegoBlocks from disk."""
    blocks = {}
    heads = {}
    lcfg = cfg["fusion"].get("lego", {})

    modality_dims = {"video": 768, "eye_tracking": 128, "ppg": 512}

    for mod in modality_names:
        path = Path(save_dir) / f"lego_block_{mod}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Pre-trained block not found: {path}. Run pretrain_lego_blocks.py first.")

        ckpt = torch.load(path, map_location=device, weights_only=False)

        block = LegoBlock(
            input_dim=modality_dims[mod],
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
        )
        block.load_state_dict(ckpt["block_state_dict"])
        blocks[mod] = block.to(device)

        head = nn.Sequential(
            nn.LayerNorm(lcfg.get("latent_dim", 64)),
            nn.Linear(lcfg.get("latent_dim", 64), 9),
        )
        head.load_state_dict(ckpt["head_state_dict"])
        heads[mod] = head.to(device)

        log.info(f"Loaded pre-trained block for {mod} (val_loss={ckpt['val_loss']:.4f}, acc={ckpt.get('val_acc', 0):.3f})")

    return blocks, heads


def slerp_weights(w1, w2, alpha=0.5):
    """Spherical linear interpolation between two weight tensors."""
    w1_flat = w1.flatten().float()
    w2_flat = w2.flatten().float()
    # Normalize
    w1_norm = w1_flat / (w1_flat.norm() + 1e-8)
    w2_norm = w2_flat / (w2_flat.norm() + 1e-8)
    dot = torch.clamp(torch.dot(w1_norm, w2_norm), -1.0, 1.0)
    theta = torch.acos(dot)
    if theta.abs() < 1e-6:
        return (alpha * w1 + (1 - alpha) * w2)
    sin_theta = torch.sin(theta)
    result = (torch.sin((1 - alpha) * theta) / sin_theta) * w1_flat + \
             (torch.sin(alpha * theta) / sin_theta) * w2_flat
    return result.reshape(w1.shape)


def merge_heads_slerp(heads, alpha=0.5):
    """Merge multiple classifier heads using pairwise SLERP."""
    head_list = list(heads.values())
    if len(head_list) == 1:
        return head_list[0]

    # Start with first head, iteratively SLERP with remaining
    merged = nn.Sequential(
        nn.LayerNorm(head_list[0][0].normalized_shape[0]),
        nn.Linear(head_list[0][1].in_features, head_list[0][1].out_features),
    )

    # Average all heads' weights
    with torch.no_grad():
        for i, (name, param) in enumerate(merged.named_parameters()):
            all_params = [list(h.parameters())[i] for h in head_list]
            # Iterative SLERP
            result = all_params[0].data.clone()
            for j in range(1, len(all_params)):
                result = slerp_weights(result, all_params[j].data, alpha)
            param.copy_(result)

    return merged


def run_merge_experiment(mode, blocks, heads, data_by_subject, subject_ids, cfg, device):
    """Run merge experiment (zero-shot, no training)."""
    lcfg = cfg["fusion"].get("lego", {})

    # Freeze all blocks
    for block in blocks.values():
        block.eval()
        for p in block.parameters():
            p.requires_grad = False

    # Merge classifier heads via SLERP
    merged_head = merge_heads_slerp(heads, alpha=lcfg.get("alpha", 0.5)).to(device)
    merged_head.eval()

    merge_method = mode.split("-")[1]  # "sum", "product", "mean", "harmonic"

    fold_results = []
    modality_names = list(blocks.keys())

    for i, test_subj in enumerate(subject_ids):
        test_data = data_by_subject.get(test_subj, [])
        if not test_data:
            continue

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for embs, label, _ in test_data:
                # Run each block independently, get complex latents for Fourier merge
                latents = []
                for j, mod in enumerate(modality_names):
                    x = embs[j].unsqueeze(0).unsqueeze(1).to(device)  # [1, 1, D]
                    l = blocks[mod](x, return_complex=True)
                    latents.append(l)

                # Merge latents in Fourier domain
                if merge_method == "sum":
                    merged = sum(latents)
                elif merge_method == "product":
                    merged = torch.stack(latents).prod(dim=0)
                elif merge_method == "mean":
                    merged = torch.stack(latents).mean(dim=0)
                elif merge_method == "harmonic":
                    alpha_val = lcfg.get("alpha", 0.5)
                    # Pairwise harmonic
                    merged = latents[0]
                    for l in latents[1:]:
                        mag1, mag2 = merged.abs(), l.abs()
                        phase1 = torch.angle(merged) if merged.is_complex() else torch.zeros_like(merged)
                        phase2 = torch.angle(l) if l.is_complex() else torch.zeros_like(l)
                        mag = 2 * alpha_val * (1 - alpha_val) * mag1 * mag2 / (alpha_val * mag2 + (1 - alpha_val) * mag1 + 1e-8)
                        phase = (phase1 + phase2) / 2
                        if merged.is_complex():
                            merged = mag * torch.exp(1j * phase)
                        else:
                            merged = mag

                # Pool and classify
                if merged.dim() == 3:
                    pooled = merged.mean(dim=1)
                else:
                    pooled = merged
                if pooled.is_complex():
                    pooled = pooled.real

                logits = merged_head(pooled)
                pred = logits.argmax(dim=-1).item()
                all_preds.append(pred)
                all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)

        f1 = weighted_f1_score(np.array(all_labels), np.array(all_preds))
        fold_results.append({"weighted_f1": f1, "test_subject": test_subj})

        if (i + 1) % 10 == 0:
            log.info(f"  Fold {i+1}/{len(subject_ids)}: F1={f1:.4f}")

    return fold_results


def run_fuse_experiment(mode, blocks, heads, data_by_subject, subject_ids, cfg, device, tune_epochs=20):
    """Run fuse experiment with light fine-tuning."""
    lcfg = cfg["fusion"].get("lego", {})
    modality_names = list(blocks.keys())
    modality_dims = {"video": 768, "eye_tracking": 128, "ppg": 512}

    # Determine the fuse sub-mode (stack or weave)
    fuse_mode = mode  # "fuse-stack" or "fuse-weave"

    fold_results = []
    for i, test_subj in enumerate(subject_ids):
        seed = cfg.get("seed", 42) + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        val_subj = subject_ids[(i + 1) % len(subject_ids)]
        train_subjects = [s for s in subject_ids if s not in (test_subj, val_subj)]

        train_data = [s for subj in train_subjects for s in data_by_subject.get(subj, [])]
        val_data = data_by_subject.get(val_subj, [])
        test_data = data_by_subject.get(test_subj, [])

        if not train_data or not test_data:
            continue

        # Create independent LegoBlocks per modality with pre-trained weights
        # (not using MultimodalLegoFusion which expects d_common; our blocks have
        #  per-modality input_dim matching the raw encoder dims)
        fold_blocks = nn.ModuleDict()
        for mod in modality_names:
            block = LegoBlock(
                input_dim=modality_dims[mod],
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
            )
            block.load_state_dict(blocks[mod].state_dict())
            fold_blocks[mod] = block
        fold_blocks = fold_blocks.to(device)

        # Merged head for classification
        merged_head = merge_heads_slerp(heads, alpha=lcfg.get("alpha", 0.5)).to(device)

        # Fine-tune with low learning rate
        all_params = list(fold_blocks.parameters()) + list(merged_head.parameters())
        optimizer = torch.optim.Adam(all_params, lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        # Train
        fold_blocks.train()
        merged_head.train()
        for epoch in range(tune_epochs):
            epoch_loss = 0
            np.random.shuffle(train_data)
            for batch_start in range(0, len(train_data), 32):
                batch = train_data[batch_start:batch_start+32]
                batch_embs = [[] for _ in modality_names]
                batch_labels = []
                for embs, label, _ in batch:
                    for j in range(len(modality_names)):
                        batch_embs[j].append(embs[j])
                    batch_labels.append(label)

                batch_embs = [torch.stack(e).to(device) for e in batch_embs]
                batch_labels = torch.stack(batch_labels).to(device) if isinstance(batch_labels[0], torch.Tensor) else torch.tensor(batch_labels).to(device)

                # Run fuse forward pass
                fused = _fuse_forward(fold_blocks, batch_embs, modality_names, fuse_mode)
                if fused.is_complex():
                    fused = fused.real
                if fused.dim() == 3:
                    fused = fused.mean(dim=1)
                logits = merged_head(fused)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Evaluate on test
        fold_blocks.eval()
        merged_head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embs, label, _ in test_data:
                emb_tensors = [e.unsqueeze(0).to(device) for e in embs]
                fused = _fuse_forward(fold_blocks, emb_tensors, modality_names, fuse_mode)
                if fused.is_complex():
                    fused = fused.real
                if fused.dim() == 3:
                    fused = fused.mean(dim=1)
                logits = merged_head(fused)
                all_preds.append(logits.argmax(-1).item())
                all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)

        f1 = weighted_f1_score(np.array(all_labels), np.array(all_preds))
        fold_results.append({"weighted_f1": f1, "test_subject": test_subj})

        if (i + 1) % 10 == 0:
            log.info(f"  Fold {i+1}/{len(subject_ids)}: F1={f1:.4f}")

    return fold_results


def _fuse_forward(blocks, embeddings, modality_names, mode):
    """Run fuse-stack or fuse-weave with per-modality LegoBlocks.

    Args:
        blocks: nn.ModuleDict of LegoBlock per modality
        embeddings: list of tensors [B, D_mod] per modality
        modality_names: list of modality name strings
        mode: "fuse-stack" or "fuse-weave"
    Returns:
        latent: [B, latent_channels, latent_dim]
    """
    # Ensure 3D: [B, 1, D]
    embs = []
    for emb in embeddings:
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        embs.append(emb)

    if mode == "fuse-stack":
        # Sequential: pass latent through each block
        latent = None
        for emb, mod in zip(embs, modality_names):
            block = blocks[mod]
            if latent is None:
                latent = block(emb, return_complex=False)
            else:
                latent = block(emb, latent_override=latent, return_complex=False)
        return latent

    elif mode == "fuse-weave":
        # Interleave single-depth passes
        first_block = blocks[modality_names[0]]
        B = embs[0].shape[0]
        from einops import repeat
        latent = repeat(first_block.latent, "n d -> b n d", b=B)
        depth = first_block.depth

        for d in range(depth):
            for emb, mod in zip(embs, modality_names):
                block = blocks[mod]
                cross_attn = block.cross_attns[d]
                cross_ff = block.cross_ffs[d]

                if block.frequency_domain:
                    if block.normalise:
                        latent_n = latent / (latent.norm(dim=block.fourier_dim, keepdim=True) + 1e-8)
                    else:
                        latent_n = latent
                    l_real = torch.fft.fft(latent_n, dim=block.fourier_dim).real
                    x_real = torch.fft.fft(emb, dim=block.fourier_dim).real
                else:
                    l_real = latent
                    x_real = emb

                latent = cross_attn(l_real, context=x_real) + l_real
                latent = cross_ff(latent) + latent

        return latent

    else:
        raise ValueError(f"Unknown fuse mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="auxiliary/benchmarks/egoemotion/configs/egoemotion.yaml",
    )
    parser.add_argument("--fusion_config", default="configs/fusion/multimodal_lego.yaml")
    parser.add_argument("--mode", required=True,
                       choices=["merge-sum", "merge-product", "merge-mean", "merge-harmonic",
                               "fuse-stack", "fuse-weave"])
    parser.add_argument("--blocks_dir", default="checkpoints/lego_blocks")
    parser.add_argument("--tune_epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--allow-global-pretrained-blocks",
        action="store_true",
        help="Acknowledge that globally supervised pretrained Lego blocks are a leaky ablation.",
    )
    args = parser.parse_args()
    require_lego_pretrain_leakage_acknowledgement(args.allow_global_pretrained_blocks)

    cfg = merge_configs(load_config(args.config), load_config(args.fusion_config))
    device = args.device if torch.cuda.is_available() else "cpu"
    name = args.name or f"lego_{args.mode}_correct"

    logger = setup_logging(cfg["logging"]["log_dir"], name)
    logger.info(f"MM-Lego experiment: {args.mode} (correct paradigm)")

    # Load labels and subjects
    labels = build_label_mapping(cfg["data_dir"], f"{cfg['data_dir']}/task_times.npy")
    extractor = SegmentExtractor(cfg["data_dir"], f"{cfg['data_dir']}/task_times.npy")
    subject_ids = extractor.get_subject_ids()

    # Load pre-trained blocks
    enabled = [m for m, c in cfg["modalities"].items() if c.get("enabled", False)]
    blocks, heads = load_pretrained_blocks(args.blocks_dir, enabled, cfg, device)

    # Build data by subject
    encoder_map = {"video": "video_mae_v2", "eye_tracking": "patchtst_eye", "ppg": "papagei_ppg"}
    encoder_dirs = [encoder_map[m] for m in enabled]

    data_by_subject = {}
    for subj in subject_ids:
        ds = MultimodalDataset(cfg["embeddings_dir"], encoder_dirs, [subj], labels)
        if len(ds) > 0:
            data_by_subject[subj] = list(ds)

    logger.info(f"Data: {sum(len(v) for v in data_by_subject.values())} samples across {len(data_by_subject)} subjects")

    # Run experiment
    if args.mode.startswith("merge"):
        logger.info(f"Running MERGE experiment (zero-shot, no training)")
        fold_results = run_merge_experiment(args.mode, blocks, heads, data_by_subject, subject_ids, cfg, device)
    else:
        logger.info(f"Running FUSE experiment (light fine-tuning, {args.tune_epochs} epochs)")
        fold_results = run_fuse_experiment(args.mode, blocks, heads, data_by_subject, subject_ids, cfg, device, args.tune_epochs)

    # Report
    f1_scores = [r["weighted_f1"] for r in fold_results]
    logger.info(f"\n{'='*50}")
    logger.info(f"MM-Lego {args.mode} Results ({len(fold_results)} folds):")
    logger.info(f"  Weighted F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
    logger.info(f"{'='*50}")

    # Save to registry
    registry = ResultsRegistry()
    registry.add(
        experiment_name=name,
        fusion_type=f"lego_{args.mode}_correct",
        metrics={"weighted_f1": float(np.mean(f1_scores)), "weighted_f1_std": float(np.std(f1_scores))},
        config_hash=config_hash(cfg),
    )


if __name__ == "__main__":
    main()
