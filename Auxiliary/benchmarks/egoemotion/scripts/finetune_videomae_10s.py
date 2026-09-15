"""Fine-tune VideoMAEv2-Base on EgoEmotion 9-class emotion classification.

Reads frames directly from pov.mp4 (no pre-cut clips).  Uses consecutive
16-frame clips — the same temporal strategy as the frozen extraction pipeline.
After training, extracts [num_clips, 768] embeddings from the fine-tuned model.

Training augmentation: RandomResizedCrop + horizontal flip + color jitter.
Validation / extraction: center crop (official HuggingFace pipeline).

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

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mac.data.video_transforms import (
    CLIP_LEN,
    make_consecutive_clips,
    normalize_clip,
    preprocess_frame,
    read_frames,
)
from mac.evaluation.metrics import compute_class_weights

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


# ---------------------------------------------------------------------------
# Spatial augmentation (applied consistently across all frames in a clip)
# ---------------------------------------------------------------------------

def _random_resized_crop(
    frames_bgr: list[np.ndarray],
    crop_size: int = 224,
    scale: tuple[float, float] = (0.5, 1.0),
    ratio: tuple[float, float] = (0.75, 1.333),
) -> list[np.ndarray]:
    """RandomResizedCrop with identical parameters for every frame."""
    h, w = frames_bgr[0].shape[:2]
    area = h * w

    for _ in range(10):
        target_area = area * np.random.uniform(*scale)
        aspect = np.exp(np.random.uniform(np.log(ratio[0]), np.log(ratio[1])))
        new_w = int(round(np.sqrt(target_area * aspect)))
        new_h = int(round(np.sqrt(target_area / aspect)))
        if 0 < new_w <= w and 0 < new_h <= h:
            x = np.random.randint(0, w - new_w + 1)
            y = np.random.randint(0, h - new_h + 1)
            return [
                cv2.resize(f[y : y + new_h, x : x + new_w], (crop_size, crop_size))
                for f in frames_bgr
            ]

    # Fallback: center crop the largest square
    short = min(h, w)
    y, x = (h - short) // 2, (w - short) // 2
    return [
        cv2.resize(f[y : y + short, x : x + short], (crop_size, crop_size))
        for f in frames_bgr
    ]


def _random_horizontal_flip(
    frames: list[np.ndarray], p: float = 0.5,
) -> list[np.ndarray]:
    if np.random.random() < p:
        return [np.ascontiguousarray(f[:, ::-1]) for f in frames]
    return frames


def _color_jitter(
    frames: list[np.ndarray],
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
) -> list[np.ndarray]:
    """Consistent color jitter across all frames (operates on BGR uint8)."""
    b_factor = 1.0 + np.random.uniform(-brightness, brightness)
    c_factor = 1.0 + np.random.uniform(-contrast, contrast)
    s_factor = 1.0 + np.random.uniform(-saturation, saturation)

    result = []
    for f in frames:
        out = np.clip(f.astype(np.float32) * b_factor, 0, 255)              # brightness
        out = np.clip((out - out.mean()) * c_factor + out.mean(), 0, 255)    # contrast
        out = out.astype(np.uint8)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)            # saturation
        result.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoSegmentDataset(Dataset):
    """One consecutive 16-frame clip per 10s segment.

    Training:   random clip + RandomResizedCrop + flip + colour jitter.
    Validation: center clip + center crop (matches frozen extraction).
    """

    def __init__(
        self,
        split_csv: Path,
        data_dir: Path,
        task_times: dict,
        clip_len: int = CLIP_LEN,
        is_train: bool = True,
    ):
        self.data_dir = data_dir
        self.task_times = task_times
        self.clip_len = clip_len
        self.is_train = is_train

        df = pd.read_csv(split_csv)
        self.samples = []
        for _, row in df.iterrows():
            subj = str(int(row["subject_id"])).zfill(3)
            seg_idx = int(row["segments"])
            label = EMOTION_TO_LABEL[row["emotion"]]
            self.samples.append((subj, seg_idx, label))

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

    def _read_raw_frames(self, subj: str, seg_idx: int) -> list[np.ndarray]:
        """Read raw BGR frames for a 10s segment (timestamp-based seeking)."""
        cap, fps, session_a_start_90 = self._get_cap(subj)
        start_sec = (session_a_start_90 + seg_idx * CHUNK_SAMPLES_90HZ) / 90.0
        expected = int(10.0 * fps)

        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        frames = []
        for _ in range(expected):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def __getitem__(self, idx):
        subj, seg_idx, label = self.samples[idx]
        raw_frames = self._read_raw_frames(subj, seg_idx)

        n_clips = len(raw_frames) // self.clip_len
        if n_clips == 0:
            while len(raw_frames) < self.clip_len:
                raw_frames.append(
                    raw_frames[-1] if raw_frames
                    else np.zeros((224, 224, 3), dtype=np.uint8)
                )
            n_clips = 1

        if self.is_train:
            clip_idx = np.random.randint(0, n_clips)
        else:
            clip_idx = n_clips // 2

        start = clip_idx * self.clip_len
        clip_bgr = raw_frames[start : start + self.clip_len]

        if self.is_train:
            clip = self._train_preprocess(clip_bgr)
        else:
            clip = self._val_preprocess(clip_bgr)

        return clip, label

    def _train_preprocess(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        frames_bgr = _color_jitter(frames_bgr)
        frames_bgr = _random_resized_crop(frames_bgr)
        frames_bgr = _random_horizontal_flip(frames_bgr)
        frames_rgb = np.stack(
            [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr],
        )
        return normalize_clip(frames_rgb)

    def _val_preprocess(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        frames_rgb = np.stack([preprocess_frame(f) for f in frames_bgr])
        return normalize_clip(frames_rgb)

    def __del__(self):
        for cap, _, _ in self._caps.values():
            cap.release()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_videomae_for_finetune(num_classes: int = 9, device: str = "cpu"):
    """Load VideoMAEv2-Base with a fresh classification head."""
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

    weights_path = hf_hub_download("OpenGVLab/VideoMAEv2-Base", "model.safetensors")
    sd = load_file(weights_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log.info(f"Loaded pretrained weights: {len(missing)} missing (head), {len(unexpected)} unexpected")

    return model.to(device)


# ---------------------------------------------------------------------------
# Layer-wise LR decay
# ---------------------------------------------------------------------------

def _get_layer_id(name: str, depth: int = 12) -> int:
    if "patch_embed" in name or "pos_embed" in name or "cls_token" in name:
        return 0
    if "blocks." in name:
        return int(name.split("blocks.")[1].split(".")[0]) + 1
    return depth + 1


def build_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    layer_decay: float,
    depth: int = 12,
) -> list[dict]:
    """Parameter groups with layer-wise LR decay and proper decay/no-decay split."""
    num_layers = depth + 2
    lr_scales = {i: layer_decay ** (num_layers - 1 - i) for i in range(num_layers)}

    param_map: dict[tuple[int, bool], list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _get_layer_id(name, depth)
        has_decay = "bias" not in name and "norm" not in name
        param_map.setdefault((layer_id, has_decay), []).append(param)

    groups = []
    for (layer_id, has_decay), params in sorted(param_map.items()):
        groups.append({
            "params": params,
            "lr": lr * lr_scales[layer_id],
            "weight_decay": weight_decay if has_decay else 0.0,
        })

    total_params = sum(len(g["params"]) for g in groups)
    log.info(
        f"Layer decay: {len(groups)} groups, {total_params} params, "
        f"lr range [{lr * lr_scales[0]:.6f}, {lr * lr_scales[num_layers - 1]:.6f}]"
    )
    return groups


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def cosine_with_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training / evaluation
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
        labels = (
            torch.tensor(labels, dtype=torch.long, device=device)
            if not isinstance(labels, torch.Tensor)
            else labels.to(device)
        )

        with torch.amp.autocast("cuda", enabled=device != "cpu", dtype=torch.bfloat16):
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
        with torch.amp.autocast("cuda", enabled=device != "cpu", dtype=torch.bfloat16):
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
# Post-training embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, manifest_df, task_times, data_dir, output_dir, device):
    """Extract [num_clips, 768] embeddings using consecutive 16-frame clips.

    Uses the same center-crop pipeline as frozen extraction so that frozen
    and fine-tuned embeddings are directly comparable.
    """
    model.eval()
    output_dir = Path(output_dir)
    extracted = 0

    for subj_id, subj_rows in tqdm(manifest_df.groupby("subject_id"), desc="Extracting"):
        subj = str(int(subj_id)).zfill(3)
        video_path = str(data_dir / subj / "pov.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning(f"Cannot open {video_path}")
            continue
        session_a_start_90 = int(task_times[subj]["session_A"][0])

        for _, row in subj_rows.iterrows():
            seg_idx = int(row["segments"])

            start_sec = (session_a_start_90 + seg_idx * CHUNK_SAMPLES_90HZ) / 90.0
            frames_arr = read_frames(cap, start_sec, 10.0)

            if len(frames_arr) < CLIP_LEN:
                log.warning(f"Video {subj}/seg_{seg_idx}: only {len(frames_arr)} frames, skipping")
                continue

            clips = make_consecutive_clips(frames_arr)
            if not clips:
                continue

            embeddings = []
            for clip in clips:
                clip_t = clip.unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", enabled=device != "cpu", dtype=torch.bfloat16):
                    emb = model.extract_features(clip_t).float().cpu()
                embeddings.append(emb)
            emb = torch.cat(embeddings, dim=0)  # [num_clips, 768]

            save_path = output_dir / subj / f"segment_{seg_idx:04d}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embedding": emb,
                "segment_idx": seg_idx,
                "subject": subj,
                "label": EMOTION_TO_LABEL[row["emotion"]],
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
    parser.add_argument("--data-dir", default="data/datasets/egoemotion_raw")
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
    parser.add_argument("--label-smoothing", type=float, default=0.1)
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

    task_times = np.load(data_dir / "task_times.npy", allow_pickle=True).item()

    train_ds = VideoSegmentDataset(
        splits_dir / "train.csv", data_dir, task_times, is_train=True,
    )
    val_ds = VideoSegmentDataset(
        splits_dir / "val.csv", data_dir, task_times, is_train=False,
    )
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    train_labels = np.array([s[2] for s in train_ds.samples])
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)

    model = load_videomae_for_finetune(NUM_CLASSES, device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {trainable:,} / {total_params:,} trainable params")

    param_groups = build_param_groups(model, args.lr, args.weight_decay, args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    scheduler = cosine_with_warmup(
        optimizer, args.warmup_epochs, args.epochs, optimizer_steps_per_epoch,
    )

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing,
    )

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
            f"val_macro={val_metrics['macro_f1']:.4f} "
            f"val_weighted={val_metrics['weighted_f1']:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(
                {"model_state_dict": best_state, "epoch": epoch, "val": val_metrics},
                ckpt_dir / "best.pt",
            )
            log.info(f"  -> New best: macro_f1={best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info(f"Best val macro F1: {best_f1:.4f}")

    # Extract embeddings for ALL segments (combine all split CSVs)
    log.info("Extracting embeddings from fine-tuned model...")
    all_splits = pd.concat(
        [pd.read_csv(f) for f in sorted(splits_dir.glob("*.csv"))],
        ignore_index=True,
    ).drop_duplicates(subset=["subject_id", "segments"])
    emb_output_dir = Path("data/embeddings/egoemotion/10s_finetuned/video_mae_v2")
    extract_embeddings(model, all_splits, task_times, data_dir, emb_output_dir, device)

    log.info(f"Done. Embeddings at {emb_output_dir}")
    log.info(
        f"Next: conda run -n visphy python scripts/run_linear_probe_10s.py "
        f"--embeddings-dir {emb_output_dir} --name linear_probe_finetuned"
    )


if __name__ == "__main__":
    main()
