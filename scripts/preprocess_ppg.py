#!/usr/bin/env python3
"""Preprocess raw PPG signals using Papagei-style filtering and resample to 125 Hz.

Creates ppg_{sensor}_125hz.npy files that the extraction pipeline expects.

Usage:
    python scripts/preprocess_ppg.py                       # ear only, local + ref
    python scripts/preprocess_ppg.py --signals both        # ear + nose
    python scripts/preprocess_ppg.py --dry-run             # preview without writing
    python scripts/preprocess_ppg.py --data-dir data/datasets/egoemotion_raw --signals ear
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ppg_preprocessing import (
    compute_segment_quality,
    detect_native_sample_rate,
    preprocess_ppg,
)

log = logging.getLogger(__name__)


def process_subject(
    subject_dir: Path,
    sensor: str,
    force: bool,
    dry_run: bool,
) -> dict | None:
    """Process one subject's PPG signal. Returns metadata or None on skip."""
    raw_file = f"ppg_{sensor}.npy"
    ref_90fps = f"ppg_{sensor}_90fps.npy"
    out_file = f"ppg_{sensor}_125hz.npy"
    quality_file = f"ppg_{sensor}_125hz_quality.json"

    raw_path = subject_dir / raw_file
    ref_path = subject_dir / ref_90fps
    out_path = subject_dir / out_file

    if not raw_path.exists():
        log.warning(f"  {subject_dir.name}: {raw_file} not found, skipping")
        return None

    if out_path.exists() and not force:
        log.info(f"  {subject_dir.name}: {out_file} already exists (use --force to overwrite)")
        return None

    # Detect native sample rate
    if ref_path.exists():
        raw_len = np.load(raw_path).shape[0]
        ref_len = np.load(ref_path).shape[0]
        fs_native = detect_native_sample_rate(raw_len, ref_len)
    else:
        fs_native = 256
        log.warning(f"  {subject_dir.name}: {ref_90fps} not found, assuming {fs_native} Hz")

    raw_signal = np.load(raw_path)
    if raw_signal.ndim > 1:
        raw_signal = raw_signal.squeeze()

    log.info(
        f"  {subject_dir.name}: {raw_file} "
        f"({len(raw_signal)} samples, {fs_native} Hz, "
        f"{len(raw_signal)/fs_native:.1f}s)"
    )

    if dry_run:
        expected_len = int(len(raw_signal) * 125 / fs_native)
        return {"subject": subject_dir.name, "fs_native": fs_native, "expected_samples_125hz": expected_len}

    preprocessed, metadata = preprocess_ppg(raw_signal, fs_native, fs_target=125)
    quality = compute_segment_quality(preprocessed, fs=125, segment_s=10.0)

    metadata["subject"] = subject_dir.name
    metadata["sensor"] = sensor
    metadata["segments_total"] = int(len(quality))
    metadata["segments_good"] = int(quality.sum())
    metadata["segments_bad"] = int((~quality).sum())

    np.save(out_path, preprocessed)
    with open(subject_dir / quality_file, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(
        f"  {subject_dir.name}: saved {out_file} "
        f"({len(preprocessed)} samples, {metadata['duration_s']}s, "
        f"{metadata['segments_good']}/{metadata['segments_total']} good segments)"
    )
    return metadata


def find_subject_dirs(data_dir: Path) -> list[Path]:
    """Find subject directories (numeric names)."""
    if not data_dir.exists():
        return []
    return sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml", help="Config file for data paths")
    parser.add_argument("--data-dir", help="Override data directory (default: from config)")
    parser.add_argument("--signals", choices=["ear", "nose", "both"], default="ear")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolve data directories
    cfg_path = Path(args.config)
    data_dirs: list[Path] = []
    if args.data_dir:
        data_dirs.append(Path(args.data_dir))
    elif cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        data_dirs.append(Path(cfg["data_dir"]))
        ref = cfg.get("ref_data_dir")
        if ref:
            data_dirs.append(Path(ref))
    else:
        data_dirs.append(Path("data/datasets/egoemotion_raw"))

    sensors = ["ear", "nose"] if args.signals == "both" else [args.signals]

    total_processed = 0
    for data_dir in data_dirs:
        subjects = find_subject_dirs(data_dir)
        if not subjects:
            log.warning(f"No subjects found in {data_dir}")
            continue
        log.info(f"Processing {len(subjects)} subjects in {data_dir}")
        for sensor in sensors:
            log.info(f"Sensor: ppg_{sensor}")
            for subj_dir in subjects:
                result = process_subject(subj_dir, sensor, args.force, args.dry_run)
                if result is not None:
                    total_processed += 1

    action = "would process" if args.dry_run else "processed"
    log.info(f"Done: {action} {total_processed} files")


if __name__ == "__main__":
    main()
