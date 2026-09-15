"""Pre-train LegoBlock for each modality independently."""
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
from mac.fusion.multimodal_lego import LegoBlock
from mac.config.simple import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


class UnimodalDataset(Dataset):
    """Load cached embeddings for a single modality with labels."""
    def __init__(self, embeddings_dir, encoder_dir_name, subject_ids, labels):
        self.samples = []
        emb_path = Path(embeddings_dir) / encoder_dir_name
        for subj in subject_ids:
            subj_dir = emb_path / subj
            if not subj_dir.exists():
                continue
            for f in sorted(subj_dir.glob("segment_*.pt")):
                seg_idx = int(f.stem.split("_")[1])
                key = f"{subj}_{seg_idx:04d}"
                if key in labels:
                    data = torch.load(f, weights_only=False)
                    emb = data["embedding"]
                    if emb.dim() == 2:
                        emb = emb.mean(dim=0)
                    self.samples.append((emb, labels[key]["emotion_label"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LegoBlockWithHead(nn.Module):
    """LegoBlock + classification head for unimodal pre-training."""
    def __init__(self, input_dim, latent_channels, latent_dim, depth, heads, dim_head, num_classes=9, **kwargs):
        super().__init__()
        self.block = LegoBlock(
            input_dim=input_dim,
            latent_channels=latent_channels,
            latent_dim=latent_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            **kwargs,
        )
        # Classification head (will be used for PlugHeads later)
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x):
        # x: [B, D] -> unsqueeze to [B, 1, D] for cross-attention
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # LegoBlock.forward uses return_complex, not return_embeddings
        latent = self.block(x, return_complex=False)
        # Mean pool latent channels: [B, latent_channels, latent_dim] -> [B, latent_dim]
        if latent.dim() == 3:
            pooled = latent.mean(dim=1)
        else:
            pooled = latent
        # Take real part if complex (Fourier domain)
        if pooled.is_complex():
            pooled = pooled.real
        logits = self.head(pooled)
        return logits, latent


def pretrain_one_modality(
    modality_name, encoder_dir_name, input_dim,
    all_subject_ids, labels, cfg, device, save_dir,
):
    """Pre-train a LegoBlock for one modality."""
    save_path = Path(save_dir) / f"lego_block_{modality_name}.pt"
    if save_path.exists():
        log.info(f"[{modality_name}] Pre-trained block found at {save_path}, skipping")
        return

    log.info(f"[{modality_name}] Pre-training LegoBlock (input_dim={input_dim})")

    # Create dataset
    dataset = UnimodalDataset(
        cfg["embeddings_dir"], encoder_dir_name, all_subject_ids, labels
    )
    log.info(f"[{modality_name}] Dataset: {len(dataset)} samples")

    if len(dataset) == 0:
        log.warning(f"[{modality_name}] No samples found, skipping")
        return

    # Split 85% train, 15% val
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42))
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    lcfg = cfg["fusion"].get("lego", {})
    model = LegoBlockWithHead(
        input_dim=input_dim,
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
        num_classes=9,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 10

    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0.0
        for emb, label in train_loader:
            emb, label = emb.to(device), label.to(device)
            logits, _ = model(emb)
            loss = criterion(logits, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for emb, label in val_loader:
                emb, label = emb.to(device), label.to(device)
                logits, _ = model(emb)
                val_loss += criterion(logits, label).item()
                correct += (logits.argmax(-1) == label).sum().item()
                total += len(label)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        acc = correct / max(total, 1)
        log.info(f"[{modality_name}] Epoch {epoch+1}/50 — train_loss={train_loss/len(train_loader):.4f}, val_loss={avg_val_loss:.4f}, acc={acc:.3f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "block_state_dict": model.block.state_dict(),
                "head_state_dict": model.head.state_dict(),
                "input_dim": input_dim,
                "modality": modality_name,
                "val_loss": best_val_loss,
                "val_acc": acc,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"[{modality_name}] Early stopping at epoch {epoch+1}")
                break

    log.info(f"[{modality_name}] Pre-training complete. Best val_loss={best_val_loss:.4f}, saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="auxiliary/benchmarks/egoemotion/configs/egoemotion.yaml",
    )
    parser.add_argument("--fusion_config", default="configs/fusion/multimodal_lego.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="checkpoints/lego_blocks")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fcfg = load_config(args.fusion_config)
    from mac.config.simple import merge_configs
    cfg = merge_configs(cfg, fcfg)

    device = args.device if torch.cuda.is_available() else "cpu"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Build labels
    labels = build_label_mapping(cfg["data_dir"], f"{cfg['data_dir']}/task_times.npy")
    log.info(f"Labels: {len(labels)} segments")

    extractor = SegmentExtractor(cfg["data_dir"], f"{cfg['data_dir']}/task_times.npy")
    all_subjects = extractor.get_subject_ids()

    # Modality -> (encoder_dir_name, input_dim)
    modality_info = {
        "video": ("video_mae_v2", 768),
        "eye_tracking": ("patchtst_eye", 128),
        "ppg": ("papagei_ppg", 512),
    }

    for mod_name, (enc_dir, input_dim) in modality_info.items():
        if cfg["modalities"][mod_name].get("enabled", False):
            pretrain_one_modality(
                mod_name, enc_dir, input_dim,
                all_subjects, labels, cfg, device, args.save_dir,
            )

    log.info("All blocks pre-trained.")


if __name__ == "__main__":
    main()
