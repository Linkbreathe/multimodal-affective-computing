"""Extract 10s-chunked embeddings for all modalities with temporal alignment.

Convenience wrapper around segment_and_extract_10s.py.  Both scripts use the
paper's manifest (ce_hardlabel_manifests/dataset_manifest.csv) which defines
the exact 2,678 non-overlapping 10-second segments across 40 subjects.

Temporal alignment (reference clock = 90 Hz from task_times.npy):
    Eye tracking (90 Hz): gaze_90fps[start:start+900], pupils_90fps[start:start+900]  -> 900 samples, 4ch
    PPG          (125 Hz): ppg_ear_125hz[int(start*125/90) : +1250]                   -> 1250 samples, 1ch
    Video        (10 FPS): pov.mp4 frames at [start/90, (start+900)/90] sec           -> ~100 frames -> 16-frame clips

Output layout:
    data/embeddings/egoemotion/10s/
    ├── video_mae_v2/{subject}/segment_{seg_idx:04d}.pt
    ├── patchtst_eye/{subject}/segment_{seg_idx:04d}.pt
    └── papagei_ppg/{subject}/segment_{seg_idx:04d}.pt

Each .pt file stores: {embedding: Tensor, segment_idx: int, subject: str, label: int, emotion: str}

Usage:
    conda run -n visphy python scripts/extract_embeddings_10s.py --encoder all
    conda run -n visphy python scripts/extract_embeddings_10s.py --encoder papagei_ppg
    conda run -n visphy python scripts/extract_embeddings_10s.py --encoder patchtst_eye
    conda run -n visphy python scripts/extract_embeddings_10s.py --encoder video_mae_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Re-export main from the canonical extraction script
from scripts.segment_and_extract_10s import main  # noqa: E402

if __name__ == "__main__":
    main()
