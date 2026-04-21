"""Contrastive pre-training for the InceptionTime gaze encoder."""
from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.encoders.inceptiontime import InceptionTimeGazeEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


class GazeWindowDataset(Dataset):
    """All-subject gaze windows with 10s duration and 50% overlap."""

    def __init__(
        self,
        data_dir: str,
        window_size: int = 900,
        overlap: float = 0.5,
    ) -> None:
        self.window_size = window_size
        self.step = max(1, int(window_size * (1.0 - overlap)))
        self.segments: list[np.ndarray] = []

        data_path = Path(data_dir)
        for subj_dir in sorted(data_path.iterdir()):
            if not subj_dir.is_dir() or not subj_dir.name.isdigit():
                continue
            gaze_path = subj_dir / "gaze_90fps.npy"
            if not gaze_path.exists():
                continue

            gaze = np.load(gaze_path)
            if gaze.ndim == 1:
                gaze = gaze[:, np.newaxis]
            gaze = np.asarray(gaze[:, :2], dtype=np.float32)
            if len(gaze) < window_size:
                continue

            for start in range(0, len(gaze) - window_size + 1, self.step):
                window = np.ascontiguousarray(gaze[start : start + window_size].T)
                self.segments.append(window)

        log.info(
            "Loaded %d gaze windows from %s (window=%d, step=%d)",
            len(self.segments),
            data_dir,
            self.window_size,
            self.step,
        )

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.segments[idx])


class GazeAugmenter:
    """Gaze-specific augmentations for contrastive view generation."""

    def __init__(
        self,
        seq_len: int = 900,
        jitter_std: float = 0.005,
        scale_range: tuple[float, float] = (0.8, 1.2),
        rotation_deg: float = 15.0,
        crop_range: tuple[float, float] = (0.7, 1.0),
        warp_knots: int = 4,
    ) -> None:
        self.seq_len = seq_len
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.rotation_rad = math.radians(rotation_deg)
        self.crop_range = crop_range
        self.warp_knots = warp_knots
        self.ops = [
            self.jitter,
            self.scale,
            self.time_warp,
            self.rotate_2d,
            self.crop_and_resize,
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        num_ops = random.randint(2, 3)
        for op in random.sample(self.ops, k=num_ops):
            out = op(out)
        return out.contiguous()

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.jitter_std

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        factor = random.uniform(*self.scale_range)
        return x * factor

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        signal = x.detach().cpu().numpy()
        target_time = np.arange(self.seq_len, dtype=np.float32)
        knot_time = np.linspace(0, self.seq_len - 1, num=self.warp_knots, dtype=np.float32)
        local_speed = np.random.uniform(0.8, 1.2, size=self.warp_knots).astype(np.float32)
        warped_time = np.interp(target_time, knot_time, local_speed)
        warped_time = np.cumsum(warped_time)
        warped_time = (
            (warped_time - warped_time[0])
            / max(warped_time[-1] - warped_time[0], 1e-6)
            * (self.seq_len - 1)
        )
        warped = np.stack(
            [np.interp(target_time, warped_time, signal[idx]) for idx in range(signal.shape[0])],
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(warped).to(device=x.device, dtype=x.dtype)

    def rotate_2d(self, x: torch.Tensor) -> torch.Tensor:
        theta = random.uniform(-self.rotation_rad, self.rotation_rad)
        rotation = x.new_tensor(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ]
        )
        return rotation @ x

    def crop_and_resize(self, x: torch.Tensor) -> torch.Tensor:
        crop_fraction = random.uniform(*self.crop_range)
        crop_len = max(2, int(self.seq_len * crop_fraction))
        start = random.randint(0, self.seq_len - crop_len)
        cropped = x[:, start : start + crop_len].unsqueeze(0)
        resized = F.interpolate(
            cropped,
            size=self.seq_len,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(0)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard NT-Xent loss across positive pairs in a batch."""
    if z_a.shape[0] < 2:
        raise ValueError("NT-Xent requires batch_size >= 2")

    z = torch.cat([z_a, z_b], dim=0)
    z = F.normalize(z, dim=1)
    logits = z @ z.T / temperature
    batch_size = z_a.shape[0]

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(mask, float("-inf"))

    targets = torch.arange(batch_size, device=logits.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(logits, targets)


def build_views(batch: torch.Tensor, augmenter: GazeAugmenter) -> tuple[torch.Tensor, torch.Tensor]:
    view_a = torch.stack([augmenter(segment) for segment in batch], dim=0)
    view_b = torch.stack([augmenter(segment) for segment in batch], dim=0)
    return view_a, view_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/datasets/egoemotion_raw")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--save_path",
        default="checkpoints/inceptiontime_gaze_pretrained.pt",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = GazeWindowDataset(args.data_dir, window_size=900, overlap=0.5)
    if len(dataset) == 0:
        raise RuntimeError(f"No gaze windows found in {args.data_dir}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    augmenter = GazeAugmenter(seq_len=900)
    encoder = InceptionTimeGazeEncoder().to(device)
    projection_head = ProjectionHead().to(device)
    optimizer = AdamW(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stagnant_epochs = 0

    for epoch in range(args.epochs):
        encoder.train()
        projection_head.train()
        running_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False):
            view_a, view_b = build_views(batch, augmenter)
            view_a = view_a.to(device, non_blocking=True)
            view_b = view_b.to(device, non_blocking=True)

            z_a = projection_head(encoder(view_a))
            z_b = projection_head(encoder(view_b))
            loss = nt_xent_loss(z_a, z_b, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / max(len(loader), 1)
        log.info(
            "Epoch %d/%d | loss=%.4f | lr=%.6f",
            epoch + 1,
            args.epochs,
            avg_loss,
            scheduler.get_last_lr()[0],
        )

        if avg_loss + 1e-6 < best_loss:
            best_loss = avg_loss
            stagnant_epochs = 0
            torch.save(encoder.state_dict(), args.save_path)
            log.info("Saved best encoder checkpoint to %s", args.save_path)
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= args.patience:
                log.info(
                    "Early stopping after %d stagnant epochs. Best loss: %.4f",
                    stagnant_epochs,
                    best_loss,
                )
                break

    log.info("Pre-training complete. Best contrastive loss: %.4f", best_loss)


if __name__ == "__main__":
    main()
