"""SEED-V 4-channel headband linear probe with FROZEN REVE.

Establishes a lower-bound benchmark for SEED-V 5-class emotion classification
using a consumer-grade EEG headband layout (M1, TP9, M2, TP10) on top of the
frozen pretrained REVE encoder. Pure linear probe — no LoRA, no unfreezing.

Channel mapping (documented honestly):
    REVE-side : ["M1",  "TP9", "M2",  "TP10"]   # headband coordinates
    SEED-V    : ["FT7", "TP7", "FT8", "TP8"]    # nearest sensors actually sliced

REVE's patch embedding is channel-agnostic (a single shared Linear); channel
identity comes entirely from 3D position embeddings. So we can instantiate
ReveEncoder with the headband coordinate list while feeding it data sliced
from the nearest SEED-V sensors. The position bank already contains M1/M2/TP9/
TP10 coordinates — see src/encoders/reve_pos_bank.py.

This script reuses the LOSO/training/optimizer/eval machinery from
scripts/run_seedv_reve_official_finetune.py and only overrides:
  1. The dataset to slice 4 channels in [FT7, TP7, FT8, TP8] order
  2. The classifier builder to pass the headband channel list to ReveEncoder
The config sets fine_tuning.enabled=false so only Stage 1 (linear probing) runs.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.encoders.reve import ReveEncoder
from src.models.reve_classifier import ReveClassifier

# Reuse everything from the official two-stage script
from scripts import run_seedv_reve_official_finetune as base

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel-subset dataset (overrides base.SegmentDataset)
# ---------------------------------------------------------------------------

class HeadbandSegmentDataset(base.SegmentDataset):
    """SegmentDataset that slices a fixed 4-channel subset before resampling."""

    def __init__(
        self,
        file_paths,
        labels,
        channel_indices: list[int],
        src_sr: int = 256,
        tgt_sr: int = 200,
    ) -> None:
        super().__init__(file_paths, labels, src_sr=src_sr, tgt_sr=tgt_sr)
        self.channel_indices = torch.tensor(channel_indices, dtype=torch.long)

    def __getitem__(self, idx):
        item = torch.load(self.file_paths[idx], map_location="cpu", weights_only=False)
        eeg = item["eeg"].float()                       # [60, 2560] @ 256 Hz
        eeg = eeg.index_select(0, self.channel_indices) # [4, 2560]
        if self.src_sr != self.tgt_sr:
            eeg = AF.resample(eeg, self.src_sr, self.tgt_sr)  # [4, 2000] @ 200 Hz
        return eeg, int(self.labels[idx])


def _resolve_channel_indices(data_channels: list[str]) -> list[int]:
    canonical = [c.upper() for c in ReveEncoder.SEEDV_60_CHANNELS]
    out = []
    for ch in data_channels:
        ch_u = ch.upper()
        if ch_u not in canonical:
            raise ValueError(
                f"data_channel {ch!r} is not in the SEED-V 60-channel layout. "
                f"Available: {canonical}"
            )
        out.append(canonical.index(ch_u))
    return out


# ---------------------------------------------------------------------------
# Headband classifier builder
# ---------------------------------------------------------------------------

def make_headband_builder(reve_channels: list[str]):
    def build(config: dict, dropout: float, device: torch.device) -> ReveClassifier:
        encoder = ReveEncoder(
            channels=reve_channels,
            weights_path=config["reve_weights"],
            pos_bank_path=config["reve_pos_bank"],
            finetune=False,  # frozen
        )
        classifier = ReveClassifier(
            encoder=encoder,
            n_classes=config.get("num_classes", 5),
            dropout=dropout,
            pooling=config.get("classifier", {}).get("pooling", "last"),
        )
        return classifier.to(device)
    return build


# ---------------------------------------------------------------------------
# Sanity check (run once at startup)
# ---------------------------------------------------------------------------

def sanity_check(reve_channels: list[str], config: dict, device: torch.device) -> None:
    log.info("Sanity check: instantiating frozen ReveEncoder with %s", reve_channels)
    enc = ReveEncoder(
        channels=reve_channels,
        weights_path=config["reve_weights"],
        pos_bank_path=config["reve_pos_bank"],
        finetune=False,
    ).to(device)
    enc.eval()
    assert enc._pos_template.shape[0] == len(reve_channels), (
        f"Position template channel count {enc._pos_template.shape[0]} "
        f"!= requested {len(reve_channels)}"
    )
    with torch.no_grad():
        x = torch.randn(2, len(reve_channels), 2000, device=device)
        emb = enc(x)
    assert emb.shape == (2, 512), f"Unexpected encoder output shape {tuple(emb.shape)}"
    n_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    log.info("Sanity check passed: encoder out=%s, encoder trainable params=%d",
             tuple(emb.shape), n_trainable)
    if n_trainable != 0:
        raise RuntimeError(
            f"Frozen encoder has {n_trainable} trainable params — expected 0"
        )
    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SEED-V 4-channel headband linear probe with frozen REVE (LOSO)."
    )
    parser.add_argument("--config", default="configs/seedv_reve_headband.yaml")
    parser.add_argument("--manifest", default="data/seedv_preprocessed/manifest.csv")
    parser.add_argument("--output_dir", default="logs/seedv_reve_headband")
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds", type=int, default=None,
                        help="Run only the first N LOSO folds (smoke test).")
    parser.add_argument("--lp-epochs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    headband_cfg = config.get("headband")
    if not headband_cfg:
        raise ValueError(f"{args.config} is missing the required `headband:` section")
    reve_channels = headband_cfg["reve_channels"]
    data_channels = headband_cfg["data_channels"]
    if len(reve_channels) != len(data_channels):
        raise ValueError("headband.reve_channels and headband.data_channels must have equal length")
    channel_indices = _resolve_channel_indices(data_channels)

    log.info("Headband layout (REVE coords):    %s", reve_channels)
    log.info("Sourced from SEED-V channels:     %s", data_channels)
    log.info("SEED-V channel indices in [0,60): %s", channel_indices)

    if config.get("fine_tuning", {}).get("enabled", True):
        raise ValueError("fine_tuning.enabled must be false for the linear-probe script")

    if args.lp_epochs is not None:
        config["linear_probing"]["n_epochs"] = args.lp_epochs

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    if config.get("seed"):
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

    sanity_check(reve_channels, config, device)

    # Monkey-patch the base module:
    #  - SegmentDataset → HeadbandSegmentDataset (with channel slicing)
    #  - build_classifier → headband variant (4-ch ReveEncoder)
    original_dataset = base.SegmentDataset
    original_builder = base.build_classifier

    def dataset_factory(file_paths, labels, src_sr=256, tgt_sr=200):
        return HeadbandSegmentDataset(
            file_paths, labels,
            channel_indices=channel_indices,
            src_sr=src_sr, tgt_sr=tgt_sr,
        )

    base.SegmentDataset = dataset_factory  # type: ignore[assignment]
    base.build_classifier = make_headband_builder(reve_channels)  # type: ignore[assignment]
    try:
        base.run_loso(
            config, args.manifest, args.output_dir, device,
            max_folds=args.folds,
            resume=not args.no_resume,
            skip_lp=False,
        )
    finally:
        base.SegmentDataset = original_dataset  # type: ignore[assignment]
        base.build_classifier = original_builder  # type: ignore[assignment]

    # Drop a small marker file documenting the channel mapping next to the results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(out_dir / "channel_mapping.json", "w") as fh:
        _json.dump({
            "frozen_backbone": True,
            "linear_probe_only": True,
            "reve_channels": reve_channels,
            "data_channels": data_channels,
            "seedv_channel_indices": channel_indices,
            "note": (
                "SEED-V has no M1/M2/TP9/TP10 sensors; the four nearest SEED-V "
                "channels (FT7/TP7/FT8/TP8) are sliced and fed to REVE under the "
                "headband coordinate names. This is a SEED-V proxy for a "
                "consumer headband, not a real headband measurement."
            ),
        }, fh, indent=2)


if __name__ == "__main__":
    main()
