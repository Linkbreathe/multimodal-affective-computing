"""Fine-tune VideoMAEv2-Base on EgoEmotion 9-class emotion classification.

Reads frames directly from pov.mp4 (no pre-cut clips). After training,
extracts [768] embeddings from the fine-tuned model for all segments.

Usage:
    conda run -n visphy python scripts/finetune_videomae_10s.py --device cuda
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.metrics import compute_class_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("finetune")

EMOTIONS = [
    "Amused", "Content", "Excited", "Awe", "Neutral",
    "Fear", "Sad", "Disgust", "Anger",
]
EMOTION_TO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}
NUM_CLASSES = 9
EMBED_DIM = 768
CHUNK_SAMPLES_90HZ = 900
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset: reads frames from pov.mp4
# ---------------------------------------------------------------------------

class VideoSegmentDataset(Dataset):
    """Loads 16-frame clips from pov.mp4 for each 10s segment."""

    def __init__(
        self,
        split_csv: Path,
        data_dir: Path,
        task_times: dict,
        num_frames: int = 16,
        sampling_rate: int = 6,
        is_train: bool = True,
    ):
        self.data_dir = data_dir
        self.task_times = task_times
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.is_train = is_train

        df = pd.read_csv(split_csv)
        self.samples = []
        for _, row in df.iterrows():
            subj = str(int(row["subject_id"])).zfill(3)
            seg_idx = int(row["segments"])
            label = EMOTION_TO_LABEL[row["emotion"]]
            self.samples.append((subj, seg_idx, label))

        # Pre-open video captures per subject (closed in __del__)
        self._caps: dict[str, tuple[cv2.VideoCapture, float, int]] = {}

    def _get_cap(self, subj: str):
        if subj not in self._caps:
            path = str(self.data_dir / subj / "pov.mp4")
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            session_a_start = int(self.task_times[subj]["session_A"][0])
            self._caps[subj] = (cap, fps, session_a_start)
        return self._caps[subj]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subj, seg_idx, label = self.samples[idx]
        cap, fps, session_a_start_90 = self._get_cap(subj)

        # Compute absolute frame range for this 10s segment
        abs_start_90 = session_a_start_90 + seg_idx * CHUNK_SAMPLES_90HZ
        start_sec = abs_start_90 / 90.0
        end_sec = start_sec + 10.0
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        total_frames = end_frame - start_frame

        # Sample 16 frames with temporal stride
        span = self.num_frames * self.sampling_rate  # 96 frames
        max_start = max(0, total_frames - span)
        if self.is_train and max_start > 0:
            offset = np.random.randint(0, max_start)
        else:
            offset = max_start // 2  # center

        indices = [offset + i * self.sampling_rate for i in range(self.num_frames)]
        # Clamp to available range
        indices = [min(i, total_frames - 1) for i in indices]

        # Read frames
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_buf = {}
        needed = set(indices)
        for fi in range(max(indices) + 1):
            ret, frame = cap.read()
            if not ret:
                break
            if fi in needed:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buf[fi] = frame

        for i in indices:
            if i in frame_buf:
                frames.append(frame_buf[i])
            elif frames:
                frames.append(frames[-1])  # repeat last
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        # Normalize
        clip = np.stack(frames).astype(np.float32) / 255.0
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        clip = torch.tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2)  # [3, 16, 224, 224]

        return clip, label

    def __del__(self):
        for cap, _, _ in self._caps.values():
            cap.release()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_videomae_for_finetune(num_classes: int = 9, device: str = "cpu"):
    """Load VideoMAEv2-Base with classification head, bypassing meta tensor bug."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    config = AutoConfig.from_pretrained(
        "OpenGVLab/VideoMAEv2-Base", trust_remote_code=True,
    )
    config.model_config["num_classes"] = num_classes

    model_class = get_class_from_dynamic_module(
        "modeling_videomaev2.VideoMAEv2",
        "OpenGVLab/VideoMAEv2-Base",
        trust_remote_code=True,
    )
    model = model_class(config)

    # Load pretrained backbone (strict=False: head is randomly initialized)
    weights_path = hf_hub_download("OpenGVLab/VideoMAEv2-Base", "model.safetensors")
    sd = load_file(weights_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log.info(f"Loaded pretrained weights: {len(missing)} missing (head), {len(unexpected)} unexpected")

    return model.to(device)


# ---------------------------------------------------------------------------
# Layer decay parameter groups
# ---------------------------------------------------------------------------

def _get_layer_id(name: str, depth: int = 12) -> int:
    """Map parameter name to layer index for layer decay."""
    if "patch_embed" in name or "pos_embed" in name or "cls_token" in name:
        return 0
    if "blocks." in name:
        block_id = int(name.split("blocks.")[1].split(".")[0])
        return block_id + 1
    # head, fc_norm, etc.
    return depth + 1


def build_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    layer_decay: float,
    depth: int = 12,
) -> list[dict]:
    """Build parameter groups with layer-wise learning rate decay."""
    num_layers = depth + 2  # patch_embed(0) + blocks(1..12) + head(13)
    lr_scales = {i: layer_decay ** (num_layers - 1 - i) for i in range(num_layers)}

    groups: dict[int, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _get_layer_id(name, depth)
        if layer_id not in groups:
            groups[layer_id] = {
                "params": [],
                "lr": lr * lr_scales[layer_id],
                "weight_decay": weight_decay if "bias" not in name and "norm" not in name else 0.0,
            }
        groups[layer_id]["params"].append(param)

    # Split no-decay params properly
    final_groups = []
    for layer_id in sorted(groups.keys()):
        g = groups[layer_id]
        # All params in this layer group share the same lr
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if _get_layer_id(name, depth) != layer_id:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        layer_lr = lr * lr_scales[layer_id]
        if decay_params:
            final_groups.append({"params": decay_params, "lr": layer_lr, "weight_decay": weight_decay})
        if no_decay_params:
            final_groups.append({"params": no_decay_params, "lr": layer_lr, "weight_decay": 0.0})

    total_params = sum(len(g["params"]) for g in final_groups)
    log.info(f"Layer decay: {len(final_groups)} param groups, {total_params} params, "
             f"lr range [{lr * lr_scales[0]:.6f}, {lr * lr_scales[num_layers-1]:.6f}]")
    return final_groups


# ---------------------------------------------------------------------------
# Warmup + cosine schedule
# ---------------------------------------------------------------------------

def cosine_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Linear warmup then cosine decay to 0."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, scheduler, loss_fn, device,
    grad_accum_steps=4, max_grad_norm=5.0,
):
    model.train()
    total_loss, n_samples = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (clips, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        clips = clips.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device) if not isinstance(labels, torch.Tensor) else labels.to(device)

        with torch.cuda.amp.autocast(enabled=device != "cpu", dtype=torch.bfloat16):
            logits = model(clips)
        loss = loss_fn(logits.float(), labels) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps * clips.size(0)
        n_samples += clips.size(0)

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for clips, labels in tqdm(loader, desc="Eval", leave=False):
        clips = clips.to(device)
        with torch.cuda.amp.autocast(enabled=device != "cpu", dtype=torch.bfloat16):
            logits = model(clips)
        preds = logits.float().argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.numpy())
        else:
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    weighted = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    return {"macro_f1": macro, "weighted_f1": weighted}


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model, manifest, task_times, data_dir, output_dir, device,
):
    """Extract [768] embeddings from fine-tuned model for all segments."""
    model.eval()
    output_dir = Path(output_dir)
    extracted = 0

    for subj_id, subj_rows in tqdm(manifest.groupby("subject"), desc="Extracting"):
        subj = str(subj_id).zfill(3)
        video_path = str(data_dir / subj / "pov.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning(f"Cannot open {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        session_a_start_90 = int(task_times[subj]["session_A"][0])

        for _, row in subj_rows.iterrows():
            seg_idx = int(row["segment_path"].split("/")[-1].replace(".p", ""))

            abs_start_90 = session_a_start_90 + seg_idx * CHUNK_SAMPLES_90HZ
            start_sec = abs_start_90 / 90.0
            start_frame = int(start_sec * fps)
            total_frames = int(10.0 * fps)

            # Center-sample 16 frames with stride 6
            span = 16 * 6
            offset = max(0, (total_frames - span) // 2)
            indices = [offset + i * 6 for i in range(16)]
            indices = [min(i, total_frames - 1) for i in indices]

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_buf = {}
            for fi in range(max(indices) + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                if fi in set(indices):
                    frame_buf[fi] = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)

            frames = []
            for i in indices:
                if i in frame_buf:
                    frames.append(frame_buf[i])
                elif frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

            clip = np.stack(frames).astype(np.float32) / 255.0
            clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
            clip = torch.tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(enabled=device != "cpu", dtype=torch.bfloat16):
                emb = model.extract_features(clip).float().cpu()  # [1, 768]

            save_path = output_dir / subj / f"segment_{seg_idx:04d}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embedding": emb.squeeze(0),  # [768]
                "segment_idx": seg_idx,
                "subject": subj,
                "label": int(row["label"]),
                "emotion": row["emotion"],
            }, save_path)
            extracted += 1

        cap.release()

    log.info(f"Extracted {extracted} embeddings to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAEv2 on EgoEmotion")
    parser.add_argument("--data-dir", default="data/egoemotion_raw")
    parser.add_argument("--device", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or f"videomae_ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    data_dir = Path(args.data_dir)
    splits_dir = data_dir / "ce_hardlabel_manifests" / "splits"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log.info(f"Experiment: {name}")
    log.info(f"Device: {device}")

    # Load task_times for video frame alignment
    task_times = np.load(data_dir / "task_times.npy", allow_pickle=True).item()

    # Datasets
    train_ds = VideoSegmentDataset(splits_dir / "train.csv", data_dir, task_times, is_train=True)
    val_ds = VideoSegmentDataset(splits_dir / "val.csv", data_dir, task_times, is_train=False)
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Compute class weights from training labels
    train_labels = np.array([s[2] for s in train_ds.samples])
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)

    # Model
    model = load_videomae_for_finetune(NUM_CLASSES, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {trainable:,} / {total_params:,} trainable params")

    # Optimizer with layer decay
    param_groups = build_param_groups(model, args.lr, args.weight_decay, args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    # Scheduler
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    scheduler = cosine_with_warmup(optimizer, args.warmup_epochs, args.epochs, steps_per_epoch)

    # Loss
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_f1 = -1.0
    best_state = None
    no_improve = 0
    ckpt_dir = Path("checkpoints") / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            grad_accum_steps=args.grad_accum,
        )
        val_metrics = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[-1]["lr"]

        log.info(
            f"Epoch {epoch:02d}/{args.epochs} | loss={train_loss:.4f} | "
            f"val_macro={val_metrics['macro_f1']:.4f} val_weighted={val_metrics['weighted_f1']:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({"model_state_dict": best_state, "epoch": epoch, "val": val_metrics},
                       ckpt_dir / "best.pt")
            log.info(f"  -> New best: macro_f1={best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    log.info(f"Best val macro F1: {best_f1:.4f}")

    # Extract embeddings for ALL segments (not just train/val)
    log.info("Extracting embeddings from fine-tuned model...")
    full_manifest = pd.read_csv(data_dir / "ce_hardlabel_manifests" / "dataset_manifest.csv")
    emb_output_dir = Path("data/embeddings_10s_finetuned/video_mae_v2")
    extract_embeddings(model, full_manifest, task_times, data_dir, emb_output_dir, device)

    log.info(f"Done. Embeddings at {emb_output_dir}")
    log.info(f"Next: conda run -n visphy python scripts/run_linear_probe_10s.py "
             f"--embeddings-dir {emb_output_dir} --name linear_probe_finetuned")


if __name__ == "__main__":
    main()
