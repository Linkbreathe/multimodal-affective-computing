# Multimodal Fusion for Emotional State Recognition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multimodal fusion system that fuses egocentric video, eye tracking, and PPG embeddings from frozen encoders to recognize emotional states, with systematic exploration from baseline to advanced fusion architectures.

**Architecture:** Frozen encoders (VideoMAE V2, PatchTST, Papagei) extract embeddings cached to disk. A detachable modality registry feeds variable-length embedding lists into swappable fusion modules. FusionTrainer runs LOSO evaluation with multi-task heads (discrete emotion CE, soft label KL, continuous affect CCC). TensorBoard logging and auto-reporting throughout.

**Tech Stack:** Python 3.10, PyTorch 2.10+cu128, transformers 4.49, timm 1.0.25, tensorboard, scikit-learn, decord, opencv, PyYAML. Conda env: `visphy`. GPU: RTX 5080 16GB.

**Spec:** `docs/superpowers/specs/2026-03-24-multimodal-fusion-design.md`

---

## File Structure

```
real-time-vis-physio-fusion/
├── configs/
│   ├── base.yaml                      # Global defaults (seed, batch_size, D_common, paths)
│   └── fusion/
│       ├── early.yaml
│       ├── mid.yaml
│       ├── late.yaml
│       ├── perceiver_io.yaml
│       ├── qformer.yaml
│       ├── healnet.yaml
│       └── multimodal_lego.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── segments.py                # Segment extraction from task_times.npy + labels
│   │   ├── dataset.py                 # EmbeddingDataset: loads cached .pt embeddings
│   │   └── collate.py                 # Variable-length collation with padding/masking
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseEncoder ABC
│   │   ├── registry.py                # ModalityRegistry: config-driven encoder lookup
│   │   ├── video_mae.py               # VideoMAE V2 frozen wrapper
│   │   ├── patchtst.py                # PatchTST architecture + self-supervised pre-training
│   │   ├── papagei.py                 # Papagei frozen wrapper
│   │   └── extract.py                 # Embedding extraction + caching pipeline
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── base.py                    # BaseFusionModule ABC
│   │   ├── projector.py               # ModalityProjector: per-modality linear to D_common
│   │   ├── early.py                   # EarlyFusion
│   │   ├── mid.py                     # MidFusion
│   │   ├── late.py                    # LateFusion
│   │   ├── perceiver_io.py            # PerceiverIOFusion
│   │   ├── qformer.py                 # QFormerFusion
│   │   ├── healnet.py                 # HEALNetFusion
│   │   └── multimodal_lego.py         # MultimodalLegoFusion
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── heads.py                   # MultiTaskHead: emotion + soft_label + VAD
│   │   └── losses.py                  # WeightedCE, KLDiv, CCCLoss, MultiTaskLoss
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── fusion_trainer.py          # FusionTrainer: LOSO loop, train/val/test per fold
│   │   └── early_stopping.py          # EarlyStopping: patience-based on val F1
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Load + merge YAML configs
│       ├── logging_setup.py           # System logging to file + console
│       ├── tb.py                      # TensorBoard writer helpers
│       ├── metrics.py                 # weighted_f1, ccc, class_weights
│       ├── reporting.py               # Auto-generate markdown experiment reports
│       └── registry.py                # ResultsRegistry: JSON experiment tracker
├── scripts/
│   ├── extract_embeddings.py          # CLI: run encoder embedding extraction
│   ├── pretrain_patchtst.py           # CLI: self-supervised PatchTST pre-training
│   └── run_experiment.py              # CLI: run a fusion experiment end-to-end
├── tests/
│   ├── __init__.py
│   ├── test_segments.py
│   ├── test_dataset.py
│   ├── test_collate.py
│   ├── test_encoders.py
│   ├── test_fusion_modules.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_trainer.py
│   └── test_config.py
├── data/
│   ├── egoemotion_raw/                # Existing raw data
│   └── embeddings/                    # Cached encoder outputs (generated)
├── runs/                              # TensorBoard logs (generated)
├── logs/                              # System logs (generated)
├── reports/                           # Experiment reports (generated)
└── checkpoints/                       # Model checkpoints (generated)
```

---

## Phase 0: Reference Data Setup

### Task 1: Copy Reference Files and Validate Data

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/__init__.py`

- [ ] **Step 1: Copy reference files from Windows drive**

```bash
cp /mnt/c/Users/Public/Data/egoEMOTION/egoEMOTION/task_times.npy \
   /home/link/Wei/Models/core/real-time-vis-physio-fusion/data/datasets/egoemotion_raw/
cp /mnt/c/Users/Public/Data/egoEMOTION/egoEMOTION/personality_questionnaire_results.csv \
   /home/link/Wei/Models/core/real-time-vis-physio-fusion/data/datasets/egoemotion_raw/
```

- [ ] **Step 2: Validate task_times.npy loads correctly**

```bash
conda run -n visphy python -c "
import numpy as np
tt = np.load('data/datasets/egoemotion_raw/task_times.npy', allow_pickle=True).item()
print(f'Subjects: {len(tt)}')
print(f'Sample keys for 005: {list(tt[\"005\"].keys())[:5]}')
print(f'Sample segment: {tt[\"005\"][\"video_Neutral\"]}')
"
```
Expected: Prints subject count (40+), task keys, and a `[start, end]` pair.

- [ ] **Step 3: Validate per-subject data files exist for all 40 subjects**

```bash
conda run -n visphy python -c "
import os, numpy as np
data_dir = 'data/datasets/egoemotion_raw'
subjects = sorted([d for d in os.listdir(data_dir) if d.isdigit()])
print(f'Found {len(subjects)} subjects')
required = ['gaze_90fps.npy', 'pupils_90fps.npy', 'ppg_ear_125hz.npy', 'pov.mp4']
for s in subjects:
    missing = [f for f in required if not os.path.exists(os.path.join(data_dir, s, f))]
    if missing:
        print(f'{s} missing: {missing}')
print('Validation complete')
# Check PPG shape
ppg = np.load(os.path.join(data_dir, '005', 'ppg_ear_125hz.npy'))
print(f'PPG shape for 005: {ppg.shape}')
gaze = np.load(os.path.join(data_dir, '005', 'gaze_90fps.npy'))
print(f'Gaze shape for 005: {gaze.shape}')
pupils = np.load(os.path.join(data_dir, '005', 'pupils_90fps.npy'))
print(f'Pupils shape for 005: {pupils.shape}')
"
```
Expected: 40 subjects, no missing files, PPG shape `(N,)`, gaze `(N, 2)`, pupils `(N, 2)`.

- [ ] **Step 4: Create package init files**

```bash
mkdir -p src/data src/encoders src/fusion src/tasks src/trainer src/utils tests scripts
touch src/__init__.py src/data/__init__.py src/encoders/__init__.py src/fusion/__init__.py \
      src/tasks/__init__.py src/trainer/__init__.py src/utils/__init__.py tests/__init__.py
```

- [ ] **Step 5: Create .gitignore**

Create `.gitignore`:
```
data/
embeddings/
runs/
logs/
reports/
checkpoints/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore src/ tests/
git commit -m "feat: scaffold project structure with package init files"
```

---

## Phase 1: Data Pipeline & Encoders

### Task 2: Segment Extraction

**Files:**
- Create: `src/data/segments.py`
- Create: `tests/test_segments.py`

- [ ] **Step 1: Write failing test for segment extraction**

```python
# tests/test_segments.py
import pytest
import numpy as np
from src.data.segments import SegmentExtractor

@pytest.fixture
def extractor():
    return SegmentExtractor(
        data_dir="data/datasets/egoemotion_raw",
        task_times_path="data/datasets/egoemotion_raw/task_times.npy",
    )

def test_loads_task_times(extractor):
    assert len(extractor.task_times) > 0
    assert "005" in extractor.task_times

def test_get_subject_ids(extractor):
    subjects = extractor.get_subject_ids()
    assert len(subjects) == 40
    assert "005" in subjects
    assert "020" not in subjects  # known missing

def test_get_segments_for_subject(extractor):
    segments = extractor.get_segments("005")
    assert len(segments) > 0
    seg = segments[0]
    assert "task_name" in seg
    assert "start_idx" in seg
    assert "end_idx" in seg
    assert "subject_id" in seg
    assert seg["end_idx"] > seg["start_idx"]

def test_get_segment_data_gaze(extractor):
    segments = extractor.get_segments("005")
    seg = segments[0]
    gaze = extractor.load_signal("005", "gaze_90fps.npy", seg["start_idx"], seg["end_idx"])
    assert gaze.ndim == 2
    assert gaze.shape[1] == 2  # x, y

def test_get_segment_data_ppg(extractor):
    segments = extractor.get_segments("005")
    seg = segments[0]
    # PPG is at 125Hz, need to convert 90Hz indices
    ppg = extractor.load_ppg_segment("005", seg["start_idx"], seg["end_idx"])
    assert ppg.ndim == 2
    assert ppg.shape[1] == 1  # reshaped from (N,)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_segments.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.segments'`

- [ ] **Step 3: Implement SegmentExtractor**

```python
# src/data/segments.py
"""Segment extraction from task_times.npy aligned to 90Hz eye-tracker timeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


class SegmentExtractor:
    """Extracts temporal segments per subject using task_times.npy."""

    # Mapping from 90Hz index space to other sampling rates
    SAMPLE_RATES = {
        "gaze_90fps.npy": 90,
        "pupils_90fps.npy": 90,
        "ppg_ear_125hz.npy": 125,
        "ppg_nose_125hz.npy": 125,
    }

    def __init__(self, data_dir: str, task_times_path: str) -> None:
        self.data_dir = Path(data_dir)
        self.task_times: dict[str, dict[str, list[int]]] = np.load(
            task_times_path, allow_pickle=True
        ).item()
        # Filter to subjects that exist in data_dir
        existing = {
            d.name
            for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        }
        self.subject_ids = sorted(existing & set(self.task_times.keys()))

    def get_subject_ids(self) -> list[str]:
        return list(self.subject_ids)

    def get_segments(self, subject_id: str) -> list[dict[str, Any]]:
        """Return list of segment dicts for a subject.

        Each segment has: task_name, start_idx, end_idx (in 90Hz space),
        subject_id. Indices are shifted by session_A start for alignment.
        """
        subject_tasks = self.task_times[subject_id]
        session_a_start = subject_tasks["session_A"][0]
        segments = []
        for task_name, (start, end) in subject_tasks.items():
            if task_name in ("session_A", "session_B"):
                continue  # skip session-level entries
            segments.append(
                {
                    "task_name": task_name,
                    "start_idx": start - session_a_start,
                    "end_idx": end - session_a_start,
                    "subject_id": subject_id,
                }
            )
        return segments

    def load_signal(
        self, subject_id: str, filename: str, start_90hz: int, end_90hz: int
    ) -> np.ndarray:
        """Load a signal segment, converting 90Hz indices to native rate."""
        filepath = self.data_dir / subject_id / filename
        data = np.load(filepath)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        native_rate = self.SAMPLE_RATES.get(filename, 90)
        if native_rate != 90:
            start = int(start_90hz * native_rate / 90)
            end = int(end_90hz * native_rate / 90)
        else:
            start, end = start_90hz, end_90hz

        return data[start:end]

    def load_ppg_segment(
        self, subject_id: str, start_90hz: int, end_90hz: int
    ) -> np.ndarray:
        """Load ear PPG segment at 125Hz, reshaped to (N, 1)."""
        return self.load_signal(subject_id, "ppg_ear_125hz.npy", start_90hz, end_90hz)

    def load_eye_tracking_segment(
        self, subject_id: str, start_90hz: int, end_90hz: int
    ) -> np.ndarray:
        """Load gaze + pupils as (N, 4): [gaze_x, gaze_y, pupil_L, pupil_R]."""
        gaze = self.load_signal(subject_id, "gaze_90fps.npy", start_90hz, end_90hz)
        pupils = self.load_signal(subject_id, "pupils_90fps.npy", start_90hz, end_90hz)
        min_len = min(len(gaze), len(pupils))
        return np.concatenate([gaze[:min_len], pupils[:min_len]], axis=1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n visphy python -m pytest tests/test_segments.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/segments.py tests/test_segments.py
git commit -m "feat: segment extraction from task_times.npy with rate conversion"
```

---

### Task 3: Label Loading

**Files:**
- Modify: `src/data/segments.py`
- Create: `tests/test_segments.py` (add tests)

- [ ] **Step 1: Write failing test for label loading**

```python
# Add to tests/test_segments.py
from src.data.segments import LabelLoader

@pytest.fixture
def label_loader():
    return LabelLoader(data_dir="data/datasets/egoemotion_raw")

def test_load_hard_labels(label_loader):
    manifest = label_loader.load_ce_manifest()
    assert len(manifest) > 0
    row = manifest.iloc[0]
    assert "subject" in manifest.columns
    assert "emotion" in manifest.columns
    assert "label" in manifest.columns

def test_load_soft_labels(label_loader):
    manifest = label_loader.load_kl_manifest()
    emotions = ["Amused", "Content", "Excited", "Awe", "Neutral", "Fear", "Sad", "Disgust", "Anger"]
    for e in emotions:
        assert e in manifest.columns

def test_load_vad_labels(label_loader):
    manifest = label_loader.load_vad_manifest()
    assert "valence_score" in manifest.columns
    assert "arousal_score" in manifest.columns
    assert "dominance_score" in manifest.columns
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_segments.py::test_load_hard_labels -v
```
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement LabelLoader**

```python
# Add to src/data/segments.py
import pandas as pd


class LabelLoader:
    """Loads emotion labels from manifest CSVs."""

    EMOTIONS = [
        "Amused", "Content", "Excited", "Awe", "Neutral",
        "Fear", "Sad", "Disgust", "Anger",
    ]

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        # Cache loaded DataFrames to avoid repeated CSV reads
        self._ce_cache: pd.DataFrame | None = None
        self._kl_cache: pd.DataFrame | None = None
        self._vad_cache: pd.DataFrame | None = None

    def load_ce_manifest(self) -> pd.DataFrame:
        if self._ce_cache is None:
            path = self.data_dir / "ce_hardlabel_manifests" / "dataset_manifest.csv"
            self._ce_cache = pd.read_csv(path)
        return self._ce_cache

    def load_kl_manifest(self) -> pd.DataFrame:
        if self._kl_cache is None:
            path = self.data_dir / "kl_softlabel_manifests" / "dataset_manifest.csv"
            self._kl_cache = pd.read_csv(path)
        return self._kl_cache

    def load_vad_manifest(self) -> pd.DataFrame:
        if self._vad_cache is None:
            path = self.data_dir / "vad_binary_quadrant_manifests" / "dataset_manifest.csv"
            self._vad_cache = pd.read_csv(path)
        return self._vad_cache

    def get_segment_labels(
        self, subject_id: str, task_name: str
    ) -> dict[str, Any] | None:
        """Get all label types for a segment. Returns None if not found.

        Matches by subject_id AND task_name in segment path.
        Returns emotion_label, soft_label (9-dim), and vad (3-dim).
        """
        ce = self.load_ce_manifest()
        kl = self.load_kl_manifest()
        vad = self.load_vad_manifest()

        # Match on subject and task_name in segment path
        subj_str = str(subject_id).zfill(3)
        ce_mask = (
            (ce["subject"].astype(str).str.zfill(3) == subj_str)
            & ce["segment_path"].str.contains(task_name, na=False)
        )
        if ce_mask.sum() == 0:
            return None

        ce_row = ce[ce_mask].iloc[0]

        # Soft labels from KL manifest (same row order)
        kl_mask = (
            (kl["subject"].astype(str).str.zfill(3) == subj_str)
            & kl["segment_path"].str.contains(task_name, na=False)
        )
        soft_label = np.zeros(9, dtype=np.float32)
        if kl_mask.sum() > 0:
            kl_row = kl[kl_mask].iloc[0]
            soft_label = np.array(
                [kl_row[e] for e in self.EMOTIONS], dtype=np.float32
            )
            # Normalize to probability distribution
            total = soft_label.sum()
            if total > 0:
                soft_label /= total

        # VAD from VAD manifest
        vad_scores = np.zeros(3, dtype=np.float32)
        vad_mask = vad["subject_id"].astype(str).str.zfill(3) == subj_str
        if "segments" in vad.columns:
            vad_mask = vad_mask & vad["segments"].str.contains(task_name, na=False)
        if vad_mask.sum() > 0:
            vad_row = vad[vad_mask].iloc[0]
            vad_scores = np.array([
                float(vad_row.get("valence_score", 0)),
                float(vad_row.get("arousal_score", 0)),
                float(vad_row.get("dominance_score", 0)),
            ], dtype=np.float32)

        return {
            "emotion_label": int(ce_row["label"]),
            "emotion_name": ce_row["emotion"],
            "soft_label": soft_label,
            "vad": vad_scores,
        }
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_segments.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/segments.py tests/test_segments.py
git commit -m "feat: label loading from manifest CSVs (CE, KL, VAD)"
```

---

### Task 4: Config System

**Files:**
- Create: `src/utils/config.py`
- Create: `configs/base.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
import pytest
from src.utils.config import load_config, merge_configs

def test_load_base_config():
    cfg = load_config("configs/base.yaml")
    assert "seed" in cfg
    assert "data_dir" in cfg
    assert "modalities" in cfg

def test_merge_overrides():
    base = {"seed": 42, "model": {"lr": 1e-4}}
    override = {"model": {"lr": 1e-3, "dropout": 0.1}}
    merged = merge_configs(base, override)
    assert merged["seed"] == 42
    assert merged["model"]["lr"] == 1e-3
    assert merged["model"]["dropout"] == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_config.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement config system**

```python
# src/utils/config.py
"""YAML config loading with deep merge support."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def config_hash(cfg: dict) -> str:
    """Deterministic hash of config for experiment tracking."""
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
```

Create `configs/base.yaml`:
```yaml
# Base configuration for all experiments
seed: 42
data_dir: "data/datasets/egoemotion_raw"
embeddings_dir: "data/embeddings/egoemotion/papagei"
ref_data_dir: "/mnt/c/Users/Public/Data/egoEMOTION/egoEMOTION"

# Modality registry
modalities:
  video:
    encoder: VideoMAEV2
    embed_dim: 768
    enabled: true
  eye_tracking:
    encoder: PatchTST
    embed_dim: 128
    enabled: true
  ppg:
    encoder: Papagei
    embed_dim: 768
    enabled: true

# Fusion defaults
fusion:
  d_common: 256
  dropout: 0.1

# Training defaults
training:
  batch_size: 64
  lr: 1e-4
  max_epochs: 100
  patience: 10
  num_workers: 4

# Multi-task loss weights
loss_weights:
  ce: 1.0
  kl: 1.0
  vad: 1.0

# Evaluation
eval:
  primary_metric: "weighted_f1"
  loso: true

# Logging
logging:
  tensorboard_dir: "runs"
  log_dir: "logs"
  report_dir: "reports"
  checkpoint_dir: "checkpoints"
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_config.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/config.py configs/base.yaml tests/test_config.py
git commit -m "feat: YAML config system with deep merge and hashing"
```

---

### Task 5: Metrics Module

**Files:**
- Create: `src/utils/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics.py
import pytest
import numpy as np
import torch
from src.utils.metrics import weighted_f1_score, concordance_correlation_coefficient, compute_class_weights

def test_weighted_f1_perfect():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    assert weighted_f1_score(y_true, y_pred) == 1.0

def test_weighted_f1_random():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([1, 2, 0, 2, 0, 1])
    f1 = weighted_f1_score(y_true, y_pred)
    assert 0.0 <= f1 <= 1.0

def test_ccc_perfect():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ccc = concordance_correlation_coefficient(x, x)
    assert abs(ccc - 1.0) < 1e-6

def test_ccc_anticorrelated():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ccc = concordance_correlation_coefficient(x, y)
    assert ccc < 0

def test_class_weights():
    labels = np.array([0, 0, 0, 1, 2])
    weights = compute_class_weights(labels, num_classes=3)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (3,)
    assert weights[0] < weights[1]  # class 0 more frequent, lower weight
    assert weights[0] < weights[2]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_metrics.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement metrics**

```python
# src/utils/metrics.py
"""Evaluation metrics: weighted F1, CCC, class weights."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score


def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def concordance_correlation_coefficient(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Lin's Concordance Correlation Coefficient."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return 0.0
    return float(2 * covariance / denominator)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse frequency class weights for weighted CE loss."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_metrics.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/metrics.py tests/test_metrics.py
git commit -m "feat: metrics module (weighted F1, CCC, class weights)"
```

---

### Task 6: Loss Functions

**Files:**
- Create: `src/tasks/losses.py`
- Create: `tests/test_losses.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_losses.py
import pytest
import torch
from src.tasks.losses import WeightedCELoss, SoftLabelKLLoss, CCCLoss, MultiTaskLoss

def test_weighted_ce_loss():
    weights = torch.tensor([1.0, 2.0, 1.5])
    loss_fn = WeightedCELoss(weight=weights)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    loss = loss_fn(logits, targets)
    assert loss.shape == ()
    assert loss.item() > 0

def test_kl_loss():
    loss_fn = SoftLabelKLLoss()
    logits = torch.randn(4, 9)
    soft_targets = torch.softmax(torch.randn(4, 9), dim=-1)
    loss = loss_fn(logits, soft_targets)
    assert loss.shape == ()
    assert loss.item() >= 0

def test_ccc_loss():
    loss_fn = CCCLoss()
    pred = torch.randn(4, 3)
    target = torch.randn(4, 3)
    loss = loss_fn(pred, target)
    assert loss.shape == ()

def test_ccc_loss_perfect():
    loss_fn = CCCLoss()
    x = torch.tensor([[1.0, 2.0, 3.0]])
    loss = loss_fn(x, x)
    assert loss.item() < 0.01  # 1 - CCC ≈ 0

def test_multitask_loss():
    weights = torch.ones(9)
    mt_loss = MultiTaskLoss(
        ce_weight=weights,
        lambda_ce=1.0,
        lambda_kl=1.0,
        lambda_vad=1.0,
    )
    outputs = {
        "emotion_logits": torch.randn(4, 9),
        "soft_logits": torch.randn(4, 9),
        "vad_pred": torch.randn(4, 3),
    }
    targets = {
        "emotion_label": torch.tensor([0, 1, 2, 3]),
        "soft_label": torch.softmax(torch.randn(4, 9), dim=-1),
        "vad": torch.randn(4, 3),
    }
    loss, breakdown = mt_loss(outputs, targets)
    assert loss.shape == ()
    assert "ce" in breakdown
    assert "kl" in breakdown
    assert "vad" in breakdown
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_losses.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement losses**

```python
# src/tasks/losses.py
"""Multi-task loss functions: weighted CE, KL divergence, CCC loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCELoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class SoftLabelKLLoss(nn.Module):
    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")


class CCCLoss(nn.Module):
    """1 - CCC as a loss (minimized when CCC = 1)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute CCC per dimension, average
        mean_pred = pred.mean(dim=0)
        mean_target = target.mean(dim=0)
        var_pred = pred.var(dim=0)
        var_target = target.var(dim=0)
        covar = ((pred - mean_pred) * (target - mean_target)).mean(dim=0)
        denom = var_pred + var_target + (mean_pred - mean_target) ** 2
        ccc = 2 * covar / (denom + 1e-8)
        return 1.0 - ccc.mean()


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        ce_weight: torch.Tensor | None = None,
        lambda_ce: float = 1.0,
        lambda_kl: float = 1.0,
        lambda_vad: float = 1.0,
    ) -> None:
        super().__init__()
        self.ce_loss = WeightedCELoss(weight=ce_weight)
        self.kl_loss = SoftLabelKLLoss()
        self.ccc_loss = CCCLoss()
        self.lambda_ce = lambda_ce
        self.lambda_kl = lambda_kl
        self.lambda_vad = lambda_vad

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ce = self.ce_loss(outputs["emotion_logits"], targets["emotion_label"])
        kl = self.kl_loss(outputs["soft_logits"], targets["soft_label"])
        vad = self.ccc_loss(outputs["vad_pred"], targets["vad"])
        total = self.lambda_ce * ce + self.lambda_kl * kl + self.lambda_vad * vad
        breakdown = {"ce": ce.item(), "kl": kl.item(), "vad": vad.item()}
        return total, breakdown
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_losses.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/losses.py tests/test_losses.py
git commit -m "feat: multi-task losses (weighted CE, KL div, CCC)"
```

---

### Task 7: Base Encoder and Modality Registry

**Files:**
- Create: `src/encoders/base.py`
- Create: `src/encoders/registry.py`
- Create: `tests/test_encoders.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_encoders.py
import pytest
import torch
from src.encoders.base import BaseEncoder
from src.encoders.registry import ModalityRegistry

def test_base_encoder_is_abstract():
    with pytest.raises(TypeError):
        BaseEncoder()

def test_registry_from_config():
    config = {
        "video": {"encoder": "VideoMAEV2", "embed_dim": 768, "enabled": True},
        "ppg": {"encoder": "Papagei", "embed_dim": 768, "enabled": True},
        "eye_tracking": {"encoder": "PatchTST", "embed_dim": 128, "enabled": False},
    }
    registry = ModalityRegistry(config)
    enabled = registry.get_enabled_modalities()
    assert "video" in enabled
    assert "ppg" in enabled
    assert "eye_tracking" not in enabled

def test_registry_embed_dims():
    config = {
        "video": {"encoder": "VideoMAEV2", "embed_dim": 768, "enabled": True},
        "ppg": {"encoder": "Papagei", "embed_dim": 768, "enabled": True},
    }
    registry = ModalityRegistry(config)
    assert registry.get_embed_dim("video") == 768
    assert registry.get_embed_dim("ppg") == 768
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_encoders.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement base encoder and registry**

```python
# src/encoders/base.py
"""Abstract base class for frozen encoders."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Base class for all frozen modality encoders."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embeddings. Output shape varies by encoder."""
        ...

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
```

```python
# src/encoders/registry.py
"""Config-driven modality registry."""
from __future__ import annotations

from typing import Any


class ModalityRegistry:
    """Manages which modalities are enabled and their configurations."""

    def __init__(self, modality_config: dict[str, dict[str, Any]]) -> None:
        self._config = modality_config

    def get_enabled_modalities(self) -> list[str]:
        return [k for k, v in self._config.items() if v.get("enabled", False)]

    def get_embed_dim(self, modality: str) -> int:
        return self._config[modality]["embed_dim"]

    def get_encoder_name(self, modality: str) -> str:
        return self._config[modality]["encoder"]

    def get_all_embed_dims(self) -> dict[str, int]:
        return {m: self.get_embed_dim(m) for m in self.get_enabled_modalities()}
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_encoders.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/encoders/base.py src/encoders/registry.py tests/test_encoders.py
git commit -m "feat: base encoder ABC and modality registry"
```

---

### Task 8: VideoMAE V2 Encoder Wrapper

**Files:**
- Create: `src/encoders/video_mae.py`
- Add to: `tests/test_encoders.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_encoders.py
from src.encoders.video_mae import VideoMAEV2Encoder

def test_video_mae_loads():
    encoder = VideoMAEV2Encoder()
    assert encoder.embed_dim == 768

def test_video_mae_forward_shape():
    encoder = VideoMAEV2Encoder()
    encoder.freeze()
    # 16-frame clip at 224x224
    dummy = torch.randn(1, 3, 16, 224, 224)
    with torch.no_grad():
        out = encoder(dummy)
    assert out.shape == (1, 768)  # CLS token or mean-pooled
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_encoders.py::test_video_mae_loads -v
```
Expected: FAIL

- [ ] **Step 3: Implement VideoMAE V2 wrapper**

```python
# src/encoders/video_mae.py
"""VideoMAE V2 frozen encoder wrapper."""
from __future__ import annotations

import torch
from transformers import VideoMAEModel

from src.encoders.base import BaseEncoder


class VideoMAEV2Encoder(BaseEncoder):
    """Wraps HuggingFace VideoMAE V2 for embedding extraction.

    Input: [B, 3, 16, 224, 224] (16-frame clips)
    Output: [B, 768] (CLS token embedding)
    """

    MODEL_NAME = "MCG-NJU/videomae-base"

    def __init__(self) -> None:
        super().__init__(embed_dim=768)
        self.model = VideoMAEModel.from_pretrained(self.MODEL_NAME)
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> VideoMAE expects [B, T, C, H, W] or pixel_values
        if x.dim() == 5 and x.shape[1] == 3:
            x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        outputs = self.model(pixel_values=x)
        # Use CLS token (first token of last hidden state)
        return outputs.last_hidden_state[:, 0, :]
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_encoders.py::test_video_mae_loads tests/test_encoders.py::test_video_mae_forward_shape -v
```
Expected: PASS (may need to download model weights on first run).

- [ ] **Step 5: Commit**

```bash
git add src/encoders/video_mae.py tests/test_encoders.py
git commit -m "feat: VideoMAE V2 frozen encoder wrapper"
```

---

### Task 9: PatchTST Encoder (Architecture + Pre-Training)

**Files:**
- Create: `src/encoders/patchtst.py`
- Create: `scripts/pretrain_patchtst.py`
- Add to: `tests/test_encoders.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_encoders.py
from src.encoders.patchtst import PatchTSTEncoder

def test_patchtst_forward_shape():
    encoder = PatchTSTEncoder(
        num_channels=4,
        patch_len=45,
        stride=22,
        d_model=128,
        n_heads=4,
        n_layers=3,
        seq_len=900,  # 10 seconds at 90Hz
    )
    x = torch.randn(2, 900, 4)  # [B, T, C]
    out = encoder(x)
    # Should return [B, num_patches, d_model]
    expected_patches = (900 - 45) // 22 + 1  # ~39
    assert out.shape[0] == 2
    assert out.shape[2] == 128
    assert out.shape[1] == expected_patches

def test_patchtst_mean_pool():
    encoder = PatchTSTEncoder(
        num_channels=4, patch_len=45, stride=22,
        d_model=128, n_heads=4, n_layers=3, seq_len=900,
    )
    x = torch.randn(2, 900, 4)
    out = encoder(x)
    pooled = out.mean(dim=1)  # [B, 128]
    assert pooled.shape == (2, 128)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_encoders.py::test_patchtst_forward_shape -v
```
Expected: FAIL

- [ ] **Step 3: Implement PatchTST encoder**

```python
# src/encoders/patchtst.py
"""PatchTST encoder for eye tracking (gaze + pupils).

Architecture: channel-independent patching → Transformer encoder.
Pre-training: self-supervised masked patch prediction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.encoders.base import BaseEncoder


class PatchTSTEncoder(BaseEncoder):
    """PatchTST for multivariate time series.

    Input: [B, T, C] (batch, time, channels)
    Output: [B, num_patches, d_model]
    """

    def __init__(
        self,
        num_channels: int = 4,
        patch_len: int = 45,
        stride: int = 22,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        seq_len: int = 900,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim=d_model)
        self.patch_len = patch_len
        self.stride = stride
        self.num_channels = num_channels
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Channel-independent: project each channel's patch independently
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.channel_embed = nn.Embedding(num_channels, d_model)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(d_model)

    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B*C, num_patches, patch_len]"""
        B, T, C = x.shape
        # Unfold along time dimension
        patches = x.unfold(1, self.patch_len, self.stride)  # [B, num_patches, C, patch_len]
        num_patches = patches.shape[1]
        # Channel independent: treat each channel as a separate sample
        patches = patches.permute(0, 2, 1, 3)  # [B, C, num_patches, patch_len]
        patches = patches.reshape(B * C, num_patches, self.patch_len)
        return patches, B, C, num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches, B, C, num_patches = self._create_patches(x)
        # Project patches
        h = self.patch_proj(patches)  # [B*C, num_patches, d_model]

        # Add positional embedding (truncate or pad if needed)
        pos = self.pos_embed[:, :num_patches, :]
        h = h + pos

        # Transformer
        h = self.transformer(h)
        h = self.norm(h)

        # Average across channels: reshape back to [B, C, num_patches, d_model]
        h = h.view(B, C, num_patches, -1)
        h = h.mean(dim=1)  # [B, num_patches, d_model]
        return h


class PatchTSTPreTrainer:
    """Self-supervised masked patch prediction for PatchTST pre-training."""

    def __init__(
        self,
        encoder: PatchTSTEncoder,
        mask_ratio: float = 0.4,
        lr: float = 1e-3,
        device: str = "cuda",
    ) -> None:
        self.encoder = encoder.to(device)
        self.mask_ratio = mask_ratio
        self.device = device
        # Learnable mask token (replaces masked patch embeddings)
        # Note: mask_token is a plain tensor with requires_grad=True (not nn.Module)
        self.mask_token = torch.randn(1, 1, encoder.embed_dim, device=device) * 0.02
        self.mask_token.requires_grad_(True)
        # Prediction head for masked patches
        self.pred_head = nn.Linear(
            encoder.embed_dim, encoder.patch_len
        ).to(device)
        params = list(encoder.parameters()) + list(self.pred_head.parameters()) + [self.mask_token]
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    def train_step(self, x: torch.Tensor) -> float:
        """One pre-training step. x: [B, T, C]."""
        self.encoder.train()
        x = x.to(self.device)

        patches, B, C, num_patches = self.encoder._create_patches(x)
        # Create mask
        num_mask = int(num_patches * self.mask_ratio)
        mask_indices = torch.rand(B * C, num_patches, device=self.device).argsort(dim=1)[:, :num_mask]

        # Get original patch values for targets
        targets = torch.gather(
            patches, 1,
            mask_indices.unsqueeze(-1).expand(-1, -1, self.encoder.patch_len)
        )

        # Forward through encoder — replace masked patches with [MASK] token
        h = self.encoder.patch_proj(patches)
        # Apply mask: replace selected positions with learnable mask token
        mask_expand = mask_indices.unsqueeze(-1).expand(-1, -1, self.encoder.embed_dim)
        mask_token = self.mask_token.expand(B * C, num_mask, -1)
        h.scatter_(1, mask_expand, mask_token)

        pos = self.encoder.pos_embed[:, :num_patches, :]
        h = h + pos
        h = self.encoder.transformer(h)
        h = self.encoder.norm(h)

        # Predict original values of masked patches
        masked_h = torch.gather(
            h, 1,
            mask_indices.unsqueeze(-1).expand(-1, -1, self.encoder.embed_dim)
        )
        pred = self.pred_head(masked_h)

        loss = nn.functional.mse_loss(pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

- [ ] **Step 4: Create pre-training script**

```python
# scripts/pretrain_patchtst.py
"""Self-supervised PatchTST pre-training on all eye tracking data."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.encoders.patchtst import PatchTSTEncoder, PatchTSTPreTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


class EyeTrackingDataset(Dataset):
    """Loads full eye tracking recordings, returns random chunks."""

    def __init__(self, data_dir: str, chunk_len: int = 900) -> None:
        self.chunk_len = chunk_len
        self.segments: list[np.ndarray] = []
        data_path = Path(data_dir)
        for subj_dir in sorted(data_path.iterdir()):
            if not subj_dir.is_dir() or not subj_dir.name.isdigit():
                continue
            gaze_path = subj_dir / "gaze_90fps.npy"
            pupil_path = subj_dir / "pupils_90fps.npy"
            if not gaze_path.exists() or not pupil_path.exists():
                continue
            gaze = np.load(gaze_path)  # (N, 2)
            pupils = np.load(pupil_path)  # (N, 2)
            min_len = min(len(gaze), len(pupils))
            combined = np.concatenate([gaze[:min_len], pupils[:min_len]], axis=1)  # (N, 4)
            # Chunk into fixed-length segments
            for start in range(0, min_len - chunk_len, chunk_len // 2):
                self.segments.append(combined[start : start + chunk_len])
        log.info(f"Loaded {len(self.segments)} chunks from {data_dir}")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.segments[idx], dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/datasets/egoemotion_raw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", default="checkpoints/patchtst_pretrained.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = EyeTrackingDataset(args.data_dir, chunk_len=900)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    encoder = PatchTSTEncoder(
        num_channels=4, patch_len=45, stride=22,
        d_model=128, n_heads=4, n_layers=3, seq_len=900,
    )
    trainer = PatchTSTPreTrainer(encoder, mask_ratio=0.4, lr=args.lr, device=device)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in loader:
            epoch_loss += trainer.train_step(batch)
        avg_loss = epoch_loss / len(loader)
        log.info(f"Epoch {epoch+1}/{args.epochs} — loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), args.save_path)
            log.info(f"Saved best model (loss={best_loss:.4f})")

    log.info(f"Pre-training complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_encoders.py::test_patchtst_forward_shape tests/test_encoders.py::test_patchtst_mean_pool -v
```
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/encoders/patchtst.py scripts/pretrain_patchtst.py tests/test_encoders.py
git commit -m "feat: PatchTST encoder with self-supervised pre-training"
```

---

### Task 10: Papagei Encoder Wrapper

**Files:**
- Create: `src/encoders/papagei.py`
- Add to: `tests/test_encoders.py`

- [ ] **Step 1: Research Papagei installation and API**

```bash
conda run -n visphy pip install papagei-foundation  # or clone from GitHub
# Verify: conda run -n visphy python -c "import papagei; print(papagei.__version__)"
```

Note: The exact package name and API may differ. Check https://arxiv.org/pdf/2410.20542 and the associated GitHub repo. Adapt the wrapper below to the actual API.

- [ ] **Step 2: Write failing test**

```python
# Add to tests/test_encoders.py
from src.encoders.papagei import PapageiEncoder

def test_papagei_loads():
    encoder = PapageiEncoder()
    assert encoder.embed_dim == 768

def test_papagei_forward_shape():
    encoder = PapageiEncoder()
    encoder.freeze()
    # 10 seconds of PPG at 125Hz
    x = torch.randn(2, 1250, 1)  # [B, T, 1]
    with torch.no_grad():
        out = encoder(x)
    assert out.shape[0] == 2
    assert out.shape[-1] == 768
```

- [ ] **Step 3: Implement Papagei wrapper**

```python
# src/encoders/papagei.py
"""Papagei frozen encoder wrapper for PPG signals.

NOTE: Adapt this to the actual Papagei API after installation.
The import path and model loading may differ from below.
"""
from __future__ import annotations

import torch

from src.encoders.base import BaseEncoder


class PapageiEncoder(BaseEncoder):
    """Wraps Papagei foundation model for PPG embedding extraction.

    Input: [B, T, 1] PPG signal at 125Hz
    Output: [B, 768] or [B, T', 768]

    TODO: Update import and model loading after verifying Papagei package.
    """

    def __init__(self) -> None:
        super().__init__(embed_dim=768)
        # Placeholder — replace with actual Papagei model loading
        # from papagei import PapageiModel
        # self.model = PapageiModel.from_pretrained("papagei-base")
        self._placeholder = True
        self.freeze()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "_placeholder") and self._placeholder:
            # Placeholder: return random embeddings with correct shape
            B = x.shape[0]
            return torch.randn(B, self.embed_dim, device=x.device)
        # Real implementation:
        # outputs = self.model(x)
        # return outputs.last_hidden_state.mean(dim=1)  # or CLS token
        raise NotImplementedError("Replace with actual Papagei forward pass")
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_encoders.py::test_papagei_loads tests/test_encoders.py::test_papagei_forward_shape -v
```
Expected: PASS (using placeholder). Will be updated when actual Papagei is installed.

- [ ] **Step 5: Commit**

```bash
git add src/encoders/papagei.py tests/test_encoders.py
git commit -m "feat: Papagei encoder wrapper (placeholder, pending installation)"
```

---

### Task 11: Embedding Extraction Pipeline

**Files:**
- Create: `src/encoders/extract.py`
- Create: `scripts/extract_embeddings.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_encoders.py
import tempfile
from src.encoders.extract import EmbeddingExtractor

def test_embedding_cache_structure(tmp_path):
    """Test that extraction creates correct directory structure."""
    extractor = EmbeddingExtractor(
        data_dir="data/datasets/egoemotion_raw",
        output_dir=str(tmp_path / "embeddings"),
        task_times_path="data/datasets/egoemotion_raw/task_times.npy",
    )
    # Just test the path generation
    path = extractor.get_cache_path("video_mae_v2", "005", 0)
    assert "video_mae_v2" in str(path)
    assert "005" in str(path)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_encoders.py::test_embedding_cache_structure -v
```
Expected: FAIL

- [ ] **Step 3: Implement embedding extractor**

```python
# src/encoders/extract.py
"""Embedding extraction and caching pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.data.segments import SegmentExtractor

log = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extracts and caches embeddings from frozen encoders."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        task_times_path: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.segment_extractor = SegmentExtractor(
            data_dir=data_dir,
            task_times_path=task_times_path,
        )

    def get_cache_path(
        self, encoder_name: str, subject_id: str, segment_idx: int
    ) -> Path:
        return self.output_dir / encoder_name / subject_id / f"segment_{segment_idx:04d}.pt"

    def is_cached(
        self, encoder_name: str, subject_id: str, segment_idx: int
    ) -> bool:
        return self.get_cache_path(encoder_name, subject_id, segment_idx).exists()

    def save_embedding(
        self,
        encoder_name: str,
        subject_id: str,
        segment_idx: int,
        embedding: torch.Tensor,
        metadata: dict[str, Any],
        config_hash: str = "",
    ) -> None:
        path = self.get_cache_path(encoder_name, subject_id, segment_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "embedding": embedding,
            "metadata": metadata,
            "config_hash": config_hash,
        }, path)

    def validate_cache(self, encoder_name: str, expected_hash: str) -> bool:
        """Check if cached embeddings match the expected config hash.

        Returns True if cache is valid, False if stale/missing.
        """
        encoder_dir = self.output_dir / encoder_name
        if not encoder_dir.exists():
            return False
        # Check first available embedding
        for pt_file in encoder_dir.rglob("*.pt"):
            data = torch.load(pt_file, weights_only=False)
            stored_hash = data.get("config_hash", "")
            return stored_hash == expected_hash
        return False

    def load_embedding(
        self, encoder_name: str, subject_id: str, segment_idx: int
    ) -> dict[str, Any]:
        path = self.get_cache_path(encoder_name, subject_id, segment_idx)
        return torch.load(path, weights_only=False)

    def extract_all(
        self,
        encoder_name: str,
        encode_fn: callable,
        device: str = "cuda",
        config_hash: str = "",
    ) -> None:
        """Extract embeddings for all subjects and segments.

        Args:
            encoder_name: Name for cache directory (e.g., 'video_mae_v2')
            encode_fn: Function(subject_id, segment) -> torch.Tensor
            device: Torch device
            config_hash: Hash of encoder config for cache invalidation
        """
        # Validate existing cache
        if self.validate_cache(encoder_name, config_hash):
            log.info(f"Cache valid for {encoder_name} (hash={config_hash})")
        else:
            log.info(f"Cache invalid/missing for {encoder_name}, re-extracting")

        subjects = self.segment_extractor.get_subject_ids()
        for subj in tqdm(subjects, desc=f"Extracting {encoder_name}"):
            segments = self.segment_extractor.get_segments(subj)
            for idx, seg in enumerate(segments):
                if self.is_cached(encoder_name, subj, idx):
                    continue
                try:
                    embedding = encode_fn(subj, seg)
                    self.save_embedding(
                        encoder_name, subj, idx,
                        embedding.cpu(),
                        metadata=seg,
                        config_hash=config_hash,
                    )
                except Exception as e:
                    log.warning(f"Failed {encoder_name}/{subj}/seg_{idx}: {e}")
```

- [ ] **Step 4: Create extraction CLI script**

```python
# scripts/extract_embeddings.py
"""CLI for extracting and caching encoder embeddings."""
from __future__ import annotations

import argparse
import logging

import torch

from src.encoders.extract import EmbeddingExtractor
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--encoder", choices=["video_mae_v2", "patchtst_eye", "papagei_ppg", "all"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    extractor = EmbeddingExtractor(
        data_dir=cfg["data_dir"],
        output_dir=cfg["embeddings_dir"],
        task_times_path=f"{cfg['data_dir']}/task_times.npy",
    )

    device = args.device if torch.cuda.is_available() else "cpu"
    encoders_to_run = (
        ["video_mae_v2", "patchtst_eye", "papagei_ppg"]
        if args.encoder == "all"
        else [args.encoder]
    )

    for enc_name in encoders_to_run:
        log.info(f"Extracting: {enc_name}")
        if enc_name == "video_mae_v2":
            _extract_video(extractor, device)
        elif enc_name == "patchtst_eye":
            _extract_eye_tracking(extractor, cfg, device)
        elif enc_name == "papagei_ppg":
            _extract_ppg(extractor, device)


def _extract_video(extractor: EmbeddingExtractor, device: str) -> None:
    from src.encoders.video_mae import VideoMAEV2Encoder
    import cv2
    import torch

    encoder = VideoMAEV2Encoder().to(device)

    def encode_fn(subject_id: str, segment: dict) -> torch.Tensor:
        video_path = f"{extractor.segment_extractor.data_dir}/{subject_id}/pov.mp4"
        start_sec = segment["start_idx"] / 90.0
        end_sec = segment["end_idx"] / 90.0
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)  # should be ~10
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) < 16:
            # Pad with last frame
            while len(frames) < 16:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        # Group into 16-frame non-overlapping clips
        import numpy as np
        # ImageNet normalization constants
        MEAN = np.array([0.485, 0.456, 0.406])
        STD = np.array([0.229, 0.224, 0.225])

        frames_arr = np.stack(frames)  # [N, 224, 224, 3]
        clips = []
        for i in range(0, len(frames_arr) - 15, 16):
            clip = frames_arr[i : i + 16]  # [16, 224, 224, 3]
            clip = clip.astype(np.float32) / 255.0
            clip = (clip - MEAN) / STD  # ImageNet normalization
            clip = torch.tensor(clip, dtype=torch.float32).permute(3, 0, 1, 2)  # [3, 16, 224, 224]
            clips.append(clip)

        if not clips:
            clips.append(torch.zeros(3, 16, 224, 224))

        clip_batch = torch.stack(clips).to(device)  # [num_clips, 3, 16, 224, 224]
        embeddings = []
        with torch.no_grad():
            for clip in clip_batch:
                emb = encoder(clip.unsqueeze(0))  # [1, 768]
                embeddings.append(emb)
        return torch.cat(embeddings, dim=0)  # [num_clips, 768]

    extractor.extract_all("video_mae_v2", encode_fn, device)


def _extract_eye_tracking(
    extractor: EmbeddingExtractor, cfg: dict, device: str
) -> None:
    from src.encoders.patchtst import PatchTSTEncoder

    encoder = PatchTSTEncoder(
        num_channels=4, patch_len=45, stride=22,
        d_model=128, n_heads=4, n_layers=3, seq_len=900,
    ).to(device)

    # Load pre-trained weights if available
    import os
    pretrained_path = "checkpoints/patchtst_pretrained.pt"
    if os.path.exists(pretrained_path):
        encoder.load_state_dict(torch.load(pretrained_path, weights_only=True))
        log.info(f"Loaded pre-trained PatchTST from {pretrained_path}")
    encoder.freeze()

    def encode_fn(subject_id: str, segment: dict) -> torch.Tensor:
        eye_data = extractor.segment_extractor.load_eye_tracking_segment(
            subject_id, segment["start_idx"], segment["end_idx"]
        )
        # Process in chunks of seq_len=900
        x = torch.tensor(eye_data, dtype=torch.float32)
        seq_len = 900
        chunks = []
        for i in range(0, len(x) - seq_len + 1, seq_len // 2):
            chunks.append(x[i : i + seq_len])
        if not chunks:
            # Pad short segments
            padded = torch.zeros(seq_len, 4)
            padded[: len(x)] = x
            chunks.append(padded)

        batch = torch.stack(chunks).to(device)  # [num_chunks, 900, 4]
        with torch.no_grad():
            embeddings = encoder(batch)  # [num_chunks, num_patches, 128]
        # Flatten: [total_patches, 128]
        return embeddings.reshape(-1, 128)

    extractor.extract_all("patchtst_eye", encode_fn, device)


def _extract_ppg(extractor: EmbeddingExtractor, device: str) -> None:
    from src.encoders.papagei import PapageiEncoder

    encoder = PapageiEncoder().to(device)

    def encode_fn(subject_id: str, segment: dict) -> torch.Tensor:
        ppg = extractor.segment_extractor.load_ppg_segment(
            subject_id, segment["start_idx"], segment["end_idx"]
        )
        x = torch.tensor(ppg, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T, 1]
        with torch.no_grad():
            return encoder(x)  # [1, 768]

    extractor.extract_all("papagei_ppg", encode_fn, device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_encoders.py::test_embedding_cache_structure -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/encoders/extract.py scripts/extract_embeddings.py tests/test_encoders.py
git commit -m "feat: embedding extraction and caching pipeline with CLI"
```

---

### Task 11b: Build Segment-to-Label Mapping

**Files:**
- Create: `src/data/label_builder.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_segments.py
from src.data.label_builder import build_label_mapping

def test_build_label_mapping():
    mapping = build_label_mapping(
        data_dir="data/datasets/egoemotion_raw",
        task_times_path="data/datasets/egoemotion_raw/task_times.npy",
    )
    assert len(mapping) > 0
    # Keys are "{subject_id}_{segment_idx}"
    sample_key = list(mapping.keys())[0]
    assert "_" in sample_key
    sample = mapping[sample_key]
    assert "emotion_label" in sample
    assert "soft_label" in sample
    assert "vad" in sample
    assert sample["soft_label"].shape == (9,)
    assert sample["vad"].shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_segments.py::test_build_label_mapping -v
```
Expected: FAIL

- [ ] **Step 3: Implement label builder**

```python
# src/data/label_builder.py
"""Build segment-to-label mapping for EmbeddingDataset."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from src.data.segments import SegmentExtractor, LabelLoader

log = logging.getLogger(__name__)


def build_label_mapping(
    data_dir: str,
    task_times_path: str,
) -> dict[str, dict[str, Any]]:
    """Build a mapping from '{subject_id}_{segment_idx}' to label tensors.

    Returns dict with keys like '005_0000' containing:
        - emotion_label: int tensor
        - soft_label: float tensor [9]
        - vad: float tensor [3]
    """
    extractor = SegmentExtractor(data_dir=data_dir, task_times_path=task_times_path)
    loader = LabelLoader(data_dir=data_dir)
    mapping: dict[str, dict[str, Any]] = {}

    for subject_id in extractor.get_subject_ids():
        segments = extractor.get_segments(subject_id)
        for idx, seg in enumerate(segments):
            labels = loader.get_segment_labels(subject_id, seg["task_name"])
            if labels is None:
                log.debug(f"No labels for {subject_id}/{seg['task_name']}, skipping")
                continue
            key = f"{subject_id}_{idx:04d}"
            mapping[key] = {
                "emotion_label": torch.tensor(labels["emotion_label"], dtype=torch.long),
                "soft_label": torch.tensor(labels["soft_label"], dtype=torch.float32),
                "vad": torch.tensor(labels["vad"], dtype=torch.float32),
            }

    log.info(f"Built label mapping: {len(mapping)} segments with labels")
    return mapping
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_segments.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/label_builder.py tests/test_segments.py
git commit -m "feat: segment-to-label mapping builder for EmbeddingDataset"
```

---

## Phase 2: FusionTrainer Infrastructure

### Task 12: Embedding Dataset and Collation

**Files:**
- Create: `src/data/dataset.py`
- Create: `src/data/collate.py`
- Create: `tests/test_dataset.py`
- Create: `tests/test_collate.py`

- [ ] **Step 1: Write failing test for dataset**

```python
# tests/test_dataset.py
import pytest
import torch
import tempfile
from pathlib import Path
from src.data.dataset import EmbeddingDataset

@pytest.fixture
def mock_embeddings(tmp_path):
    """Create mock cached embeddings for 2 subjects, 3 segments each."""
    modalities = {
        "video_mae_v2": 768,
        "patchtst_eye": 128,
        "papagei_ppg": 768,
    }
    for mod, dim in modalities.items():
        for subj in ["005", "006"]:
            for seg_idx in range(3):
                path = tmp_path / mod / subj / f"segment_{seg_idx:04d}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                n_tokens = 5 if mod != "papagei_ppg" else 1
                torch.save({
                    "embedding": torch.randn(n_tokens, dim),
                    "metadata": {
                        "subject_id": subj,
                        "task_name": f"video_task_{seg_idx}",
                        "start_idx": seg_idx * 1000,
                        "end_idx": (seg_idx + 1) * 1000,
                    },
                }, path)
    return tmp_path

def test_dataset_loads(mock_embeddings):
    ds = EmbeddingDataset(
        embeddings_dir=str(mock_embeddings),
        modalities=["video_mae_v2", "patchtst_eye", "papagei_ppg"],
        subject_ids=["005", "006"],
    )
    assert len(ds) == 6  # 2 subjects * 3 segments

def test_dataset_getitem(mock_embeddings):
    ds = EmbeddingDataset(
        embeddings_dir=str(mock_embeddings),
        modalities=["video_mae_v2", "patchtst_eye", "papagei_ppg"],
        subject_ids=["005"],
    )
    sample = ds[0]
    assert "embeddings" in sample
    assert "subject_id" in sample
    assert len(sample["embeddings"]) == 3  # 3 modalities

def test_dataset_with_labels(mock_embeddings):
    """Test that labels are correctly looked up by zero-padded key."""
    labels = {
        "005_0000": {
            "emotion_label": torch.tensor(2, dtype=torch.long),
            "soft_label": torch.softmax(torch.randn(9), dim=0),
            "vad": torch.randn(3),
        },
        "005_0001": {
            "emotion_label": torch.tensor(5, dtype=torch.long),
            "soft_label": torch.softmax(torch.randn(9), dim=0),
            "vad": torch.randn(3),
        },
    }
    ds = EmbeddingDataset(
        embeddings_dir=str(mock_embeddings),
        modalities=["video_mae_v2", "patchtst_eye", "papagei_ppg"],
        subject_ids=["005"],
        labels=labels,
    )
    sample = ds[0]
    assert "labels" in sample
    assert sample["labels"]["emotion_label"].item() == 2
    assert sample["labels"]["soft_label"].shape == (9,)
    assert sample["labels"]["vad"].shape == (3,)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_dataset.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement EmbeddingDataset**

```python
# src/data/dataset.py
"""PyTorch Dataset for cached encoder embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Loads pre-extracted embeddings from disk.

    Each sample contains embeddings from all enabled modalities for one segment.
    """

    def __init__(
        self,
        embeddings_dir: str,
        modalities: list[str],
        subject_ids: list[str],
        labels: dict[str, dict] | None = None,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.modalities = modalities
        self.labels = labels or {}

        # Build index: list of (subject_id, segment_idx) pairs
        # Use first modality to enumerate segments
        self.samples: list[tuple[str, int]] = []
        first_mod = modalities[0]
        for subj in subject_ids:
            mod_dir = self.embeddings_dir / first_mod / subj
            if not mod_dir.exists():
                continue
            seg_files = sorted(mod_dir.glob("segment_*.pt"))
            for f in seg_files:
                idx = int(f.stem.split("_")[1])
                # Verify all modalities have this segment
                all_exist = all(
                    (self.embeddings_dir / m / subj / f.name).exists()
                    for m in modalities
                )
                if all_exist:
                    self.samples.append((subj, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        subject_id, seg_idx = self.samples[index]
        embeddings = {}
        metadata = {}
        for mod in self.modalities:
            path = (
                self.embeddings_dir / mod / subject_id / f"segment_{seg_idx:04d}.pt"
            )
            data = torch.load(path, weights_only=False)
            embeddings[mod] = data["embedding"]
            metadata = data.get("metadata", metadata)

        sample = {
            "embeddings": embeddings,
            "subject_id": subject_id,
            "segment_idx": seg_idx,
            "metadata": metadata,
        }

        # Add labels if available
        key = f"{subject_id}_{seg_idx:04d}"
        if key in self.labels:
            sample["labels"] = self.labels[key]

        return sample
```

- [ ] **Step 4: Implement collation with padding**

```python
# src/data/collate.py
"""Variable-length collation for multi-modal embeddings."""
from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_embeddings(
    batch: list[dict[str, Any]], pool: bool = False
) -> dict[str, Any]:
    """Collate batch of embedding samples.

    Args:
        batch: List of samples from EmbeddingDataset.
        pool: If True, mean-pool each modality to single vector.
              If False, pad sequences and create attention masks.
    """
    modalities = list(batch[0]["embeddings"].keys())
    result: dict[str, Any] = {
        "subject_ids": [s["subject_id"] for s in batch],
        "segment_indices": [s["segment_idx"] for s in batch],
    }

    if pool:
        # Mean-pool each modality -> [B, D]
        for mod in modalities:
            tensors = [s["embeddings"][mod].mean(dim=0) for s in batch]
            result[f"{mod}_emb"] = torch.stack(tensors)
    else:
        # Pad sequences -> [B, T_max, D] + masks
        for mod in modalities:
            seqs = [s["embeddings"][mod] for s in batch]
            padded = pad_sequence(seqs, batch_first=True)
            lengths = torch.tensor([s.shape[0] for s in seqs])
            mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
            result[f"{mod}_emb"] = padded
            result[f"{mod}_mask"] = mask

    # Collate labels if present
    if "labels" in batch[0] and batch[0]["labels"] is not None:
        label_keys = batch[0]["labels"].keys()
        for key in label_keys:
            vals = [s["labels"][key] for s in batch]
            if isinstance(vals[0], (int, float)):
                result[key] = torch.tensor(vals)
            elif isinstance(vals[0], torch.Tensor):
                result[key] = torch.stack(vals)

    return result
```

- [ ] **Step 5: Write and run collation test**

```python
# tests/test_collate.py
import pytest
import torch
from src.data.collate import collate_embeddings

def test_collate_pooled():
    batch = [
        {"embeddings": {"vid": torch.randn(5, 768), "ppg": torch.randn(1, 768)},
         "subject_id": "005", "segment_idx": 0},
        {"embeddings": {"vid": torch.randn(3, 768), "ppg": torch.randn(1, 768)},
         "subject_id": "006", "segment_idx": 1},
    ]
    out = collate_embeddings(batch, pool=True)
    assert out["vid_emb"].shape == (2, 768)
    assert out["ppg_emb"].shape == (2, 768)

def test_collate_padded():
    batch = [
        {"embeddings": {"vid": torch.randn(5, 768), "ppg": torch.randn(1, 768)},
         "subject_id": "005", "segment_idx": 0},
        {"embeddings": {"vid": torch.randn(3, 768), "ppg": torch.randn(1, 768)},
         "subject_id": "006", "segment_idx": 1},
    ]
    out = collate_embeddings(batch, pool=False)
    assert out["vid_emb"].shape == (2, 5, 768)  # padded to max length
    assert out["vid_mask"].shape == (2, 5)
    assert out["vid_mask"][0].all()  # first sample has all 5
    assert out["vid_mask"][1, :3].all()  # second has 3
    assert not out["vid_mask"][1, 3:].any()  # rest masked
```

- [ ] **Step 6: Run all tests**

```bash
conda run -n visphy python -m pytest tests/test_dataset.py tests/test_collate.py -v
```
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add src/data/dataset.py src/data/collate.py tests/test_dataset.py tests/test_collate.py
git commit -m "feat: embedding dataset and variable-length collation"
```

---

### Task 13: Multi-Task Heads

**Files:**
- Create: `src/tasks/heads.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_losses.py
from src.tasks.heads import MultiTaskHead

def test_multitask_head_shapes():
    head = MultiTaskHead(d_fused=256, num_emotions=9, num_vad=3)
    x = torch.randn(4, 256)
    outputs = head(x)
    assert outputs["emotion_logits"].shape == (4, 9)
    assert outputs["soft_logits"].shape == (4, 9)
    assert outputs["vad_pred"].shape == (4, 3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_losses.py::test_multitask_head_shapes -v
```
Expected: FAIL

- [ ] **Step 3: Implement multi-task heads**

```python
# src/tasks/heads.py
"""Multi-task prediction heads for emotion recognition."""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """Three task heads sharing a fusion backbone output.

    - Discrete emotion: [B, 9] logits for CE loss
    - Soft emotion: [B, 9] logits for KL loss
    - VAD: [B, 3] continuous predictions
    """

    def __init__(
        self,
        d_fused: int = 256,
        num_emotions: int = 9,
        num_vad: int = 3,
    ) -> None:
        super().__init__()
        self.emotion_head = nn.Linear(d_fused, num_emotions)
        self.soft_head = nn.Linear(d_fused, num_emotions)
        self.vad_head = nn.Linear(d_fused, num_vad)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "emotion_logits": self.emotion_head(x),
            "soft_logits": self.soft_head(x),
            "vad_pred": self.vad_head(x),
        }
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_losses.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tasks/heads.py tests/test_losses.py
git commit -m "feat: multi-task prediction heads (emotion, soft label, VAD)"
```

---

### Task 14: Base Fusion Module and Modality Projector

**Files:**
- Create: `src/fusion/base.py`
- Create: `src/fusion/projector.py`
- Create: `tests/test_fusion_modules.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_fusion_modules.py
import pytest
import torch
from src.fusion.base import BaseFusionModule
from src.fusion.projector import ModalityProjector

def test_base_fusion_is_abstract():
    with pytest.raises(TypeError):
        BaseFusionModule(d_common=256)

def test_projector():
    embed_dims = {"video": 768, "eye": 128, "ppg": 768}
    proj = ModalityProjector(embed_dims=embed_dims, d_common=256)
    inputs = {
        "video": torch.randn(4, 768),
        "eye": torch.randn(4, 128),
        "ppg": torch.randn(4, 768),
    }
    outputs = proj(inputs)
    for mod in embed_dims:
        assert outputs[mod].shape == (4, 256)

def test_projector_sequential():
    """Test projector with sequence inputs [B, T, D]."""
    embed_dims = {"video": 768, "eye": 128}
    proj = ModalityProjector(embed_dims=embed_dims, d_common=256)
    inputs = {
        "video": torch.randn(4, 10, 768),
        "eye": torch.randn(4, 20, 128),
    }
    outputs = proj(inputs)
    assert outputs["video"].shape == (4, 10, 256)
    assert outputs["eye"].shape == (4, 20, 256)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement base fusion and projector**

```python
# src/fusion/base.py
"""Abstract base class for fusion modules."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFusionModule(ABC, nn.Module):
    """Base class for all fusion layers.

    Interface: forward(embeddings, modality_ids, masks=None) -> [B, d_out]
    """

    def __init__(self, d_common: int, d_out: int | None = None) -> None:
        super().__init__()
        self.d_common = d_common
        self.d_out = d_out or d_common

    @abstractmethod
    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Fuse multi-modal embeddings.

        Args:
            embeddings: List of [B, D_common] (pooled) or [B, T_i, D_common] (sequential)
            modality_ids: List of modality names (same order as embeddings)
            masks: Optional list of [B, T_i] boolean masks (True = valid)

        Returns:
            [B, d_out] fused representation
        """
        ...
```

```python
# src/fusion/projector.py
"""Per-modality linear projection to common dimension."""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """Projects each modality's embeddings to a shared dimension."""

    def __init__(
        self, embed_dims: dict[str, int], d_common: int
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {mod: nn.Linear(dim, d_common) for mod, dim in embed_dims.items()}
        )

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {mod: self.projectors[mod](x) for mod, x in inputs.items()}
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fusion/base.py src/fusion/projector.py tests/test_fusion_modules.py
git commit -m "feat: base fusion module ABC and modality projector"
```

---

### Task 15: Logging, TensorBoard, and Reporting Utilities

**Files:**
- Create: `src/utils/logging_setup.py`
- Create: `src/utils/tb.py`
- Create: `src/utils/reporting.py`
- Create: `src/utils/registry.py`

- [ ] **Step 1: Implement utilities (these are infrastructure, tested via integration)**

```python
# src/utils/logging_setup.py
"""System logging setup."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{experiment_name}.log"
    logger = logging.getLogger("fusion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
```

```python
# src/utils/tb.py
"""TensorBoard logging helpers."""
from __future__ import annotations

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir: str) -> None:
        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, tag: str, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self.writer.add_scalar(f"{tag}/{k}", v, step)

    def log_hparams(self, hparams: dict, metrics: dict) -> None:
        self.writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        self.writer.close()
```

```python
# src/utils/reporting.py
"""Auto-generate experiment reports."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def generate_report(
    experiment_name: str,
    config: dict,
    fold_results: list[dict[str, float]],
    report_dir: str,
) -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(report_dir) / f"{experiment_name}_{timestamp}.md"

    import numpy as np

    # Aggregate metrics
    metrics = {}
    for key in fold_results[0]:
        vals = [r[key] for r in fold_results]
        metrics[key] = {"mean": np.mean(vals), "std": np.std(vals)}

    lines = [
        f"# Experiment Report: {experiment_name}",
        f"\n**Date:** {datetime.now().isoformat()}",
        f"\n## Configuration\n",
        f"```yaml\n{config}\n```\n",
        f"\n## Aggregate Results\n",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v['mean']:.4f} | {v['std']:.4f} |")

    lines.extend([
        f"\n## Per-Fold Results\n",
        "| Fold | " + " | ".join(fold_results[0].keys()) + " |",
        "|------|" + "|".join(["------"] * len(fold_results[0])) + "|",
    ])
    for i, r in enumerate(fold_results):
        vals = " | ".join(f"{v:.4f}" for v in r.values())
        lines.append(f"| {i} | {vals} |")

    report = "\n".join(lines)
    path.write_text(report)
    return str(path)
```

```python
# src/utils/registry.py
"""Results registry for cross-experiment comparison."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResultsRegistry:
    def __init__(self, path: str = "reports/results_registry.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = []

    def add(
        self,
        experiment_name: str,
        fusion_type: str,
        metrics: dict[str, float],
        config_hash: str,
    ) -> None:
        entry = {
            "experiment_name": experiment_name,
            "fusion_type": fusion_type,
            "timestamp": datetime.now().isoformat(),
            "config_hash": config_hash,
            **metrics,
        }
        self.data.append(entry)
        self.path.write_text(json.dumps(self.data, indent=2))

    def get_all(self) -> list[dict[str, Any]]:
        return self.data
```

- [ ] **Step 2: Commit**

```bash
git add src/utils/logging_setup.py src/utils/tb.py src/utils/reporting.py src/utils/registry.py
git commit -m "feat: logging, TensorBoard, reporting, and results registry utilities"
```

---

### Task 16: Early Stopping

**Files:**
- Create: `src/trainer/early_stopping.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_trainer.py
import pytest
from src.trainer.early_stopping import EarlyStopping

def test_early_stopping_improves():
    es = EarlyStopping(patience=3, mode="max")
    assert not es.step(0.5)
    assert not es.step(0.6)
    assert not es.step(0.7)
    assert es.best_score == 0.7

def test_early_stopping_triggers():
    es = EarlyStopping(patience=3, mode="max")
    es.step(0.5)       # best=0.5, counter=0
    es.step(0.4)       # counter=1
    assert not es.step(0.3)  # counter=2, not yet (2 < 3)
    assert es.step(0.3)      # counter=3 >= 3, should stop

def test_early_stopping_min_mode():
    es = EarlyStopping(patience=3, mode="min")
    es.step(0.5)       # best=0.5
    es.step(0.4)       # improvement, counter=0
    es.step(0.5)       # counter=1
    assert not es.step(0.6)  # counter=2, not yet (2 < 3)
    assert es.step(0.7)      # counter=3 >= 3, should stop
    assert es.step(0.7)  # 3rd worse -> stop
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_trainer.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement early stopping**

```python
# src/trainer/early_stopping.py
"""Early stopping based on validation metric."""
from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )
        if improved:
            self.best_score = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_trainer.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trainer/early_stopping.py tests/test_trainer.py
git commit -m "feat: early stopping with patience and min/max modes"
```

---

### Task 17: FusionTrainer (LOSO Loop)

**Files:**
- Create: `src/trainer/fusion_trainer.py`
- Create: `scripts/run_experiment.py`
- Add to: `tests/test_trainer.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_trainer.py
import torch
import torch.nn as nn
from src.trainer.fusion_trainer import FusionTrainer
from src.fusion.base import BaseFusionModule
from src.tasks.heads import MultiTaskHead

class DummyFusion(BaseFusionModule):
    def __init__(self):
        super().__init__(d_common=32, d_out=32)
        self.fc = nn.Linear(32, 32)

    def forward(self, embeddings, modality_ids, masks=None):
        # Simple: mean all embeddings
        stacked = torch.stack(embeddings, dim=0).mean(dim=0)
        return self.fc(stacked)

def test_trainer_single_fold():
    """Test that trainer can run a single fold with mock data."""
    fusion = DummyFusion()
    head = MultiTaskHead(d_fused=32, num_emotions=9)
    trainer = FusionTrainer(
        fusion_model=fusion,
        task_head=head,
        config={
            "training": {"batch_size": 4, "lr": 1e-3, "max_epochs": 2, "patience": 5, "num_workers": 0},
            "loss_weights": {"ce": 1.0, "kl": 1.0, "vad": 1.0},
            "seed": 42,
        },
    )

    # Create tiny mock dataset
    train_data = [
        {
            "embeddings": [torch.randn(32), torch.randn(32)],
            "modality_ids": ["video", "ppg"],
            "labels": {
                "emotion_label": torch.tensor(0),
                "soft_label": torch.softmax(torch.randn(9), dim=0),
                "vad": torch.randn(3),
            },
        }
        for _ in range(8)
    ]
    val_data = train_data[:2]

    metrics = trainer.train_fold(train_data, val_data, fold_name="test_fold")
    assert "weighted_f1" in metrics
    assert "ccc" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_trainer.py::test_trainer_single_fold -v
```
Expected: FAIL

- [ ] **Step 3: Implement FusionTrainer**

```python
# src/trainer/fusion_trainer.py
"""FusionTrainer: LOSO cross-validation loop with multi-task training."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.heads import MultiTaskHead
from src.tasks.losses import MultiTaskLoss
from src.fusion.base import BaseFusionModule
from src.trainer.early_stopping import EarlyStopping
from src.utils.metrics import weighted_f1_score, concordance_correlation_coefficient, compute_class_weights

log = logging.getLogger("fusion")


class FusionTrainer:
    """Trains a fusion model with LOSO cross-validation."""

    def __init__(
        self,
        fusion_model: BaseFusionModule,
        task_head: MultiTaskHead,
        config: dict[str, Any],
        device: str | None = None,
    ) -> None:
        self.fusion_model = fusion_model
        self.task_head = task_head
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train_fold(
        self,
        train_data: list[dict],
        val_data: list[dict],
        fold_name: str = "fold",
        tb_logger: Any | None = None,
    ) -> dict[str, float]:
        """Train one LOSO fold. Returns test metrics on val_data."""
        cfg = self.config["training"]

        # Reset model weights to fresh initialization for each fold
        import copy
        fusion_model = copy.deepcopy(self.fusion_model).to(self.device)
        task_head = copy.deepcopy(self.task_head).to(self.device)
        all_params = list(fusion_model.parameters()) + list(task_head.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["max_epochs"]
        )

        # Compute class weights from training data
        train_labels = np.array([d["labels"]["emotion_label"].item() for d in train_data])
        class_weights = compute_class_weights(train_labels, num_classes=9).to(self.device)

        loss_fn = MultiTaskLoss(
            ce_weight=class_weights,
            lambda_ce=self.config["loss_weights"]["ce"],
            lambda_kl=self.config["loss_weights"]["kl"],
            lambda_vad=self.config["loss_weights"]["vad"],
        ).to(self.device)

        early_stop = EarlyStopping(patience=cfg["patience"], mode="max")
        best_state = None

        train_loader = self._make_loader(train_data, cfg["batch_size"], shuffle=True)
        val_loader = self._make_loader(val_data, cfg["batch_size"], shuffle=False)

        for epoch in range(cfg["max_epochs"]):
            # Train
            fusion_model.train()
            task_head.train()
            epoch_loss = 0.0
            for batch in train_loader:
                embeddings, modality_ids, labels = self._unpack_batch(batch)
                fused = fusion_model(embeddings, modality_ids)
                outputs = task_head(fused)
                loss, breakdown = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # Validate
            val_metrics = self._evaluate(fusion_model, task_head, val_loader, loss_fn)

            if tb_logger:
                tb_logger.log_scalars(
                    f"{fold_name}/train", {"loss": epoch_loss / len(train_loader)}, epoch
                )
                tb_logger.log_scalars(f"{fold_name}/val", val_metrics, epoch)

            # Early stopping on weighted F1
            if early_stop.step(val_metrics["weighted_f1"]):
                log.info(f"{fold_name} early stopping at epoch {epoch}")
                break

            if early_stop.counter == 0:
                best_state = {
                    "fusion": {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()},
                    "head": {k: v.cpu().clone() for k, v in task_head.state_dict().items()},
                }

        # Restore best model
        if best_state:
            fusion_model.load_state_dict(best_state["fusion"])
            task_head.load_state_dict(best_state["head"])

        # Final evaluation and return models for test evaluation
        self._last_fusion = fusion_model
        self._last_head = task_head
        return self._evaluate(fusion_model, task_head, val_loader, loss_fn)

    @torch.no_grad()
    def _evaluate(
        self, fusion_model: nn.Module, task_head: nn.Module,
        loader: DataLoader, loss_fn: MultiTaskLoss,
    ) -> dict[str, float]:
        fusion_model.eval()
        task_head.eval()
        all_preds, all_labels = [], []
        all_vad_pred, all_vad_true = [], []
        total_loss = 0.0

        for batch in loader:
            embeddings, modality_ids, labels = self._unpack_batch(batch)
            fused = fusion_model(embeddings, modality_ids)
            outputs = task_head(fused)
            loss, _ = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = outputs["emotion_logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels["emotion_label"].cpu().numpy())
            all_vad_pred.append(outputs["vad_pred"].cpu().numpy())
            all_vad_true.append(labels["vad"].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_vad_pred = np.concatenate(all_vad_pred)
        all_vad_true = np.concatenate(all_vad_true)

        f1 = weighted_f1_score(all_labels, all_preds)
        ccc_vals = [
            concordance_correlation_coefficient(all_vad_true[:, i], all_vad_pred[:, i])
            for i in range(3)
        ]

        return {
            "weighted_f1": f1,
            "loss": total_loss / max(len(loader), 1),
            "ccc": float(np.mean(ccc_vals)),
            "ccc_valence": ccc_vals[0],
            "ccc_arousal": ccc_vals[1],
            "ccc_dominance": ccc_vals[2],
        }

    def _unpack_batch(
        self, batch: dict
    ) -> tuple[list[torch.Tensor], list[str], dict[str, torch.Tensor]]:
        """Unpack a collated batch into fusion inputs."""
        embeddings = [batch["embeddings"][m].to(self.device) for m in batch["modality_ids"]]
        labels = {k: v.to(self.device) for k, v in batch["labels"].items()}
        return embeddings, batch["modality_ids"], labels

    def _make_loader(
        self, data: list[dict], batch_size: int, shuffle: bool
    ) -> DataLoader:
        """Create DataLoader from list of sample dicts."""
        from torch.utils.data import Dataset as _DS

        class _ListDS(_DS):
            def __init__(self, items: list):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, idx):
                return self.items[idx]

        def collate(batch):
            modality_ids = batch[0]["modality_ids"]
            embeddings = {}
            for m in modality_ids:
                embeddings[m] = torch.stack([b["embeddings"][b["modality_ids"].index(m)] for b in batch])
            labels = {}
            for k in batch[0]["labels"]:
                vals = [b["labels"][k] for b in batch]
                labels[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else torch.tensor(vals)
            return {"embeddings": embeddings, "modality_ids": modality_ids, "labels": labels}

        return DataLoader(
            _ListDS(data), batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
            num_workers=0,
        )

    def run_loso(
        self,
        all_data: dict[str, list[dict]],
        subject_ids: list[str],
        tb_base_dir: str | None = None,
    ) -> list[dict[str, float]]:
        """Run full LOSO cross-validation.

        Args:
            all_data: Dict mapping subject_id -> list of samples
            subject_ids: List of all subject IDs
            tb_base_dir: TensorBoard log directory
        """
        fold_results = []
        for i, test_subj in enumerate(subject_ids):
            seed = self.config["seed"] + i
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Split: test=1 subject, val=next subject, train=rest
            val_subj = subject_ids[(i + 1) % len(subject_ids)]
            train_subjects = [s for s in subject_ids if s not in (test_subj, val_subj)]

            train_data = [s for subj in train_subjects for s in all_data.get(subj, [])]
            val_data = all_data.get(val_subj, [])
            test_data = all_data.get(test_subj, [])

            if not train_data or not test_data:
                log.warning(f"Skipping fold {test_subj}: insufficient data")
                continue

            log.info(
                f"Fold {i+1}/{len(subject_ids)}: test={test_subj}, val={val_subj}, "
                f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}"
            )

            # Train on train, early stop on val
            self.train_fold(train_data, val_data, fold_name=f"fold_{test_subj}")

            # Evaluate on test subject using the best model from this fold
            test_loader = self._make_loader(test_data, self.config["training"]["batch_size"], shuffle=False)
            train_labels = np.array([d["labels"]["emotion_label"].item() for d in train_data])
            class_weights = compute_class_weights(train_labels, 9).to(self.device)
            loss_fn = MultiTaskLoss(ce_weight=class_weights).to(self.device)
            # Use the trained models from train_fold (stored as _last_fusion/_last_head)
            metrics = self._evaluate(self._last_fusion, self._last_head, test_loader, loss_fn)
            metrics["test_subject"] = test_subj
            fold_results.append(metrics)
            log.info(f"Fold {test_subj}: F1={metrics['weighted_f1']:.4f}, CCC={metrics['ccc']:.4f}")

        return fold_results
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_trainer.py -v
```
Expected: All PASS.

- [ ] **Step 5: Create experiment runner script**

```python
# scripts/run_experiment.py
"""CLI for running fusion experiments."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime

import torch

from src.utils.config import load_config, merge_configs, config_hash
from src.utils.logging_setup import setup_logging
from src.utils.reporting import generate_report
from src.utils.registry import ResultsRegistry

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fusion_config", required=True, help="Path to fusion method config")
    parser.add_argument("--name", default=None, help="Experiment name")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    fusion_cfg = load_config(args.fusion_config)
    cfg = merge_configs(base_cfg, fusion_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"{fusion_cfg.get('fusion_type', 'experiment')}_{timestamp}"

    logger = setup_logging(cfg["logging"]["log_dir"], name)
    logger.info(f"Starting experiment: {name}")
    logger.info(f"Config hash: {config_hash(cfg)}")

    # TODO: Build fusion model, dataset, and run LOSO
    # This will be completed when fusion models are implemented
    logger.info("Experiment runner ready. Fusion models to be added in Phase 3.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/trainer/fusion_trainer.py scripts/run_experiment.py tests/test_trainer.py
git commit -m "feat: FusionTrainer with LOSO loop, early stopping, and experiment CLI"
```

---

**>>> HUMAN REVIEW GATE: Phase 2 complete. Verify infrastructure works end-to-end before proceeding. <<<**

---

## Phase 3: Baseline Fusion

### Task 18: Early Fusion

**Files:**
- Create: `src/fusion/early.py`
- Create: `configs/fusion/early.yaml`
- Add to: `tests/test_fusion_modules.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_fusion_modules.py
from src.fusion.early import EarlyFusion

def test_early_fusion_shape():
    fusion = EarlyFusion(d_common=256, num_modalities=3, dropout=0.1)
    embeddings = [torch.randn(4, 256) for _ in range(3)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

def test_early_fusion_variable_modalities():
    fusion = EarlyFusion(d_common=256, num_modalities=2, dropout=0.1)
    embeddings = [torch.randn(4, 256) for _ in range(2)]
    out = fusion(embeddings, modality_ids=["video", "ppg"])
    assert out.shape == (4, 256)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py::test_early_fusion_shape -v
```
Expected: FAIL

- [ ] **Step 3: Implement Early Fusion**

```python
# src/fusion/early.py
"""Early fusion: concatenate + MLP."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class EarlyFusion(BaseFusionModule):
    """Concatenate all modality embeddings, pass through shared MLP."""

    def __init__(
        self,
        d_common: int = 256,
        num_modalities: int = 3,
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_out or d_common)
        d_in = d_common * num_modalities
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_common * 2),
            nn.BatchNorm1d(d_common * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_common * 2, d_common),
            nn.BatchNorm1d(d_common),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_common, self.d_out),
        )

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        concat = torch.cat(embeddings, dim=-1)  # [B, D_common * num_mod]
        return self.mlp(concat)
```

Create `configs/fusion/early.yaml`:
```yaml
fusion_type: early
fusion:
  d_common: 256
  dropout: 0.1
training:
  lr: 1e-4
  max_epochs: 100
```

- [ ] **Step 4: Run tests**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py -v
```
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fusion/early.py configs/fusion/early.yaml tests/test_fusion_modules.py
git commit -m "feat: early fusion (concatenate + MLP)"
```

---

### Task 19: Mid Fusion

**Files:**
- Create: `src/fusion/mid.py`
- Create: `configs/fusion/mid.yaml`
- Add to: `tests/test_fusion_modules.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_fusion_modules.py
from src.fusion.mid import MidFusion

def test_mid_fusion_shape():
    fusion = MidFusion(d_common=256, modality_ids=["video", "eye", "ppg"], dropout=0.1)
    embeddings = [torch.randn(4, 256) for _ in range(3)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)
```

- [ ] **Step 2: Implement Mid Fusion**

```python
# src/fusion/mid.py
"""Mid fusion: per-modality MLP + concatenate + shared MLP."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class MidFusion(BaseFusionModule):
    """Per-modality refinement then concatenation and cross-modal MLP."""

    def __init__(
        self,
        d_common: int = 256,
        modality_ids: list[str] | None = None,
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        modality_ids = modality_ids or ["video", "eye_tracking", "ppg"]
        super().__init__(d_common=d_common, d_out=d_out or d_common)
        self.modality_mlps = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(d_common, d_common),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for mod in modality_ids
        })
        d_in = d_common * len(modality_ids)
        self.cross_modal = nn.Sequential(
            nn.Linear(d_in, d_common * 2),
            nn.BatchNorm1d(d_common * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_common * 2, self.d_out),
        )

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        refined = [self.modality_mlps[m](e) for m, e in zip(modality_ids, embeddings)]
        concat = torch.cat(refined, dim=-1)
        return self.cross_modal(concat)
```

Create `configs/fusion/mid.yaml`:
```yaml
fusion_type: mid
fusion:
  d_common: 256
  dropout: 0.1
training:
  lr: 1e-4
  max_epochs: 100
```

- [ ] **Step 3: Run tests and commit**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py -v
git add src/fusion/mid.py configs/fusion/mid.yaml tests/test_fusion_modules.py
git commit -m "feat: mid fusion (per-modality MLP + cross-modal MLP)"
```

---

### Task 20: Late Fusion

**Files:**
- Create: `src/fusion/late.py`
- Create: `configs/fusion/late.yaml`
- Add to: `tests/test_fusion_modules.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_fusion_modules.py
from src.fusion.late import LateFusion

def test_late_fusion_avg():
    fusion = LateFusion(d_common=256, num_modalities=3, mode="average")
    embeddings = [torch.randn(4, 256) for _ in range(3)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

def test_late_fusion_weighted():
    fusion = LateFusion(d_common=256, num_modalities=3, mode="weighted")
    embeddings = [torch.randn(4, 256) for _ in range(3)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)
```

- [ ] **Step 2: Implement Late Fusion**

```python
# src/fusion/late.py
"""Late fusion: per-modality full branches + decision-level fusion."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class LateFusion(BaseFusionModule):
    """Per-modality MLP branches fused at decision level."""

    def __init__(
        self,
        d_common: int = 256,
        num_modalities: int = 3,
        mode: str = "average",
        dropout: float = 0.1,
        d_out: int | None = None,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_out or d_common)
        self.mode = mode
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_common, d_common),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_common, self.d_out),
            )
            for _ in range(num_modalities)
        ])
        if mode == "weighted":
            self.weights = nn.Parameter(torch.ones(num_modalities))

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        branch_outs = [branch(emb) for branch, emb in zip(self.branches, embeddings)]
        stacked = torch.stack(branch_outs, dim=0)  # [num_mod, B, D]

        if self.mode == "average":
            return stacked.mean(dim=0)
        elif self.mode == "weighted":
            w = torch.softmax(self.weights, dim=0).view(-1, 1, 1)
            return (stacked * w).sum(dim=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
```

Create `configs/fusion/late.yaml`:
```yaml
fusion_type: late
fusion:
  d_common: 256
  dropout: 0.1
  mode: "weighted"  # or "average"
training:
  lr: 1e-4
  max_epochs: 100
```

- [ ] **Step 3: Run tests and commit**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py -v
git add src/fusion/late.py configs/fusion/late.yaml tests/test_fusion_modules.py
git commit -m "feat: late fusion (average and weighted decision-level)"
```

---

**>>> HUMAN REVIEW GATE: Phase 3 complete. Run baseline experiments, review results before proceeding to advanced architectures. <<<**

---

## Phase 4a: Perceiver IO + Q-Former

### Task 21: Perceiver IO Fusion

**Files:**
- Create: `src/fusion/perceiver_io.py`
- Create: `configs/fusion/perceiver_io.yaml`
- Add to: `tests/test_fusion_modules.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_fusion_modules.py
from src.fusion.perceiver_io import PerceiverIOFusion

def test_perceiver_io_pooled():
    fusion = PerceiverIOFusion(d_common=256, n_latents=32, n_layers=2, n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

def test_perceiver_io_sequential():
    fusion = PerceiverIOFusion(d_common=256, n_latents=32, n_layers=2, n_heads=4)
    embeddings = [torch.randn(4, 10, 256), torch.randn(4, 20, 256), torch.randn(4, 1, 256)]
    masks = [torch.ones(4, 10, dtype=torch.bool), torch.ones(4, 20, dtype=torch.bool), torch.ones(4, 1, dtype=torch.bool)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"], masks=masks)
    assert out.shape == (4, 256)
```

- [ ] **Step 2: Implement Perceiver IO**

```python
# src/fusion/perceiver_io.py
"""Perceiver IO fusion: learnable latent array with cross-attention."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class PerceiverIOFusion(BaseFusionModule):
    """Perceiver IO: latent array cross-attends to all modality inputs."""

    def __init__(
        self,
        d_common: int = 256,
        n_latents: int = 32,
        d_latent: int | None = None,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        d_latent = d_latent or d_common
        super().__init__(d_common=d_common, d_out=d_latent)
        self.n_latents = n_latents

        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_latent) * 0.02)

        # Learnable modality type embeddings
        self.modality_embed = nn.Embedding(16, d_common)  # up to 16 modalities

        # Cross-attention + self-attention layers
        self.cross_attn_layers = nn.ModuleList()
        self.self_attn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(d_latent, n_heads, dropout=dropout, batch_first=True, kdim=d_common, vdim=d_common)
            )
            self.self_attn_layers.append(
                nn.TransformerEncoderLayer(d_latent, n_heads, dim_feedforward=d_latent * 4, dropout=dropout, batch_first=True, norm_first=True)
            )

        self.cross_norms = nn.ModuleList([nn.LayerNorm(d_latent) for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_latent, d_latent)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]

        # Ensure all embeddings are 3D [B, T, D]
        processed = []
        all_masks = []
        for i, emb in enumerate(embeddings):
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)  # [B, 1, D]
            # Add modality type embedding
            emb = emb + self.modality_embed.weight[i].unsqueeze(0).unsqueeze(0)
            processed.append(emb)
            if masks is not None:
                m = masks[i]
                if m.dim() == 1:
                    m = m.unsqueeze(1)
                all_masks.append(m)

        # Concatenate all modality inputs
        kv = torch.cat(processed, dim=1)  # [B, T_total, D_common]
        if all_masks:
            kv_mask = torch.cat(all_masks, dim=1)  # [B, T_total]
            key_padding_mask = ~kv_mask
        else:
            key_padding_mask = None

        # Expand latents for batch
        latents = self.latents.expand(B, -1, -1)

        # Iterative cross-attention + self-attention
        for cross_attn, self_attn, norm in zip(
            self.cross_attn_layers, self.self_attn_layers, self.cross_norms
        ):
            # Cross-attention: latents query the input
            residual = latents
            latents_normed = norm(latents)
            attended, _ = cross_attn(
                latents_normed, kv, kv,
                key_padding_mask=key_padding_mask,
            )
            latents = residual + attended
            # Self-attention among latents
            latents = self_attn(latents)

        # Output: mean pool latents
        out = latents.mean(dim=1)  # [B, d_latent]
        return self.output_proj(out)
```

Create `configs/fusion/perceiver_io.yaml`:
```yaml
fusion_type: perceiver_io
fusion:
  d_common: 256
  perceiver:
    n_latents: 32
    d_latent: 256
    n_layers: 2
    n_heads: 4
    dropout: 0.1
training:
  lr: 1e-4
  max_epochs: 100
```

- [ ] **Step 3: Run tests and commit**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py -v
git add src/fusion/perceiver_io.py configs/fusion/perceiver_io.yaml tests/test_fusion_modules.py
git commit -m "feat: Perceiver IO fusion with learnable latent cross-attention"
```

---

### Task 22: Q-Former Fusion

**Files:**
- Create: `src/fusion/qformer.py`
- Create: `configs/fusion/qformer.yaml`
- Add to: `tests/test_fusion_modules.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_fusion_modules.py
from src.fusion.qformer import QFormerFusion

def test_qformer_pooled():
    fusion = QFormerFusion(d_common=256, n_queries=16, n_layers=2, n_heads=4)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

def test_qformer_sequential():
    fusion = QFormerFusion(d_common=256, n_queries=16, n_layers=2, n_heads=4)
    embeddings = [torch.randn(4, 10, 256), torch.randn(4, 20, 256)]
    masks = [torch.ones(4, 10, dtype=torch.bool), torch.ones(4, 20, dtype=torch.bool)]
    out = fusion(embeddings, modality_ids=["video", "eye"], masks=masks)
    assert out.shape == (4, 256)
```

- [ ] **Step 2: Implement Q-Former**

```python
# src/fusion/qformer.py
"""Q-Former fusion: learnable query tokens with cross-attention to modalities."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class QFormerFusion(BaseFusionModule):
    """Q-Former: learnable queries extract task-relevant info via cross-attention."""

    def __init__(
        self,
        d_common: int = 256,
        n_queries: int = 16,
        d_query: int | None = None,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        d_query = d_query or d_common
        super().__init__(d_common=d_common, d_out=d_query)

        self.queries = nn.Parameter(torch.randn(1, n_queries, d_query) * 0.02)
        self.modality_embed = nn.Embedding(16, d_common)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn": nn.TransformerEncoderLayer(
                    d_query, n_heads, dim_feedforward=d_query * 4,
                    dropout=dropout, batch_first=True, norm_first=True,
                ),
                "cross_attn": nn.MultiheadAttention(
                    d_query, n_heads, dropout=dropout, batch_first=True,
                    kdim=d_common, vdim=d_common,
                ),
                "cross_norm": nn.LayerNorm(d_query),
            }))

        self.output_proj = nn.Linear(d_query, d_query)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]

        # Prepare KV from all modalities
        processed = []
        for i, emb in enumerate(embeddings):
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            emb = emb + self.modality_embed.weight[i]
            processed.append(emb)

        kv = torch.cat(processed, dim=1)  # [B, T_total, D]
        queries = self.queries.expand(B, -1, -1)

        for layer in self.layers:
            # Self-attention among queries
            queries = layer["self_attn"](queries)
            # Cross-attention: queries attend to modality inputs
            residual = queries
            q_normed = layer["cross_norm"](queries)
            attended, _ = layer["cross_attn"](q_normed, kv, kv)
            queries = residual + attended

        # Pool queries -> single vector
        out = queries.mean(dim=1)  # [B, d_query]
        return self.output_proj(out)
```

Create `configs/fusion/qformer.yaml`:
```yaml
fusion_type: qformer
fusion:
  d_common: 256
  qformer:
    n_queries: 16
    d_query: 256
    n_layers: 2
    n_heads: 4
    dropout: 0.1
training:
  lr: 1e-4
  max_epochs: 100
```

- [ ] **Step 3: Run tests and commit**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py -v
git add src/fusion/qformer.py configs/fusion/qformer.yaml tests/test_fusion_modules.py
git commit -m "feat: Q-Former fusion with learnable query cross-attention"
```

---

**>>> HUMAN REVIEW GATE: Phase 4a complete. Review Perceiver IO and Q-Former results. <<<**

---

## Phase 4b: HEALNet + Multimodal Lego

### Task 23: HEALNet Fusion

**Files:**
- Create: `src/fusion/healnet.py`
- Create: `configs/fusion/healnet.yaml`
- Add to: `tests/test_fusion_modules.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_fusion_modules.py
from src.fusion.healnet import HEALNetFusion

def test_healnet_pooled():
    fusion = HEALNetFusion(d_common=256, memory_size=16, n_layers=2, n_heads=4, num_modalities=3)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

def test_healnet_missing_modality():
    """HEALNet should handle fewer modalities than expected."""
    fusion = HEALNetFusion(d_common=256, memory_size=16, n_layers=2, n_heads=4, num_modalities=3)
    embeddings = [torch.randn(4, 256), torch.randn(4, 256)]  # only 2
    out = fusion(embeddings, modality_ids=["video", "ppg"])
    assert out.shape == (4, 256)
```

- [ ] **Step 2: Implement HEALNet**

```python
# src/fusion/healnet.py
"""HEALNet: Hybrid Early-fusion Attention Learning Network."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class HEALNetFusion(BaseFusionModule):
    """Iterative early fusion with shared memory updated by each modality."""

    def __init__(
        self,
        d_common: int = 256,
        memory_size: int = 16,
        n_layers: int = 2,
        n_heads: int = 4,
        num_modalities: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_common)
        self.memory_size = memory_size

        # Learnable shared memory
        self.init_memory = nn.Parameter(torch.randn(1, memory_size, d_common) * 0.02)

        # Per-layer cross-attention (shared across modalities)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    d_common, n_heads, dropout=dropout, batch_first=True,
                ),
                "norm_q": nn.LayerNorm(d_common),
                "norm_kv": nn.LayerNorm(d_common),
                "ffn": nn.Sequential(
                    nn.Linear(d_common, d_common * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_common * 4, d_common),
                    nn.Dropout(dropout),
                ),
                "norm_ffn": nn.LayerNorm(d_common),
            }))

        self.output_proj = nn.Linear(d_common, d_common)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]
        memory = self.init_memory.expand(B, -1, -1)

        # Ensure all embeddings are 3D
        processed = []
        for emb in embeddings:
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            processed.append(emb)

        for layer in self.layers:
            # Update memory with each modality sequentially
            for emb in processed:
                residual = memory
                q = layer["norm_q"](memory)
                kv = layer["norm_kv"](emb)
                attended, _ = layer["cross_attn"](q, kv, kv)
                memory = residual + attended

            # FFN on updated memory
            residual = memory
            memory = residual + layer["ffn"](layer["norm_ffn"](memory))

        # Pool memory -> single vector
        out = memory.mean(dim=1)  # [B, d_common]
        return self.output_proj(out)
```

Create `configs/fusion/healnet.yaml`:
```yaml
fusion_type: healnet
fusion:
  d_common: 256
  healnet:
    memory_size: 16
    n_layers: 2
    n_heads: 4
    dropout: 0.1
training:
  lr: 1e-4
  max_epochs: 100
```

- [ ] **Step 3: Run tests and commit**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py -v
git add src/fusion/healnet.py configs/fusion/healnet.yaml tests/test_fusion_modules.py
git commit -m "feat: HEALNet fusion with iterative shared memory"
```

---

### Task 24: Multimodal Lego Fusion

**Files:**
- Create: `src/fusion/multimodal_lego.py`
- Create: `configs/fusion/multimodal_lego.yaml`
- Add to: `tests/test_fusion_modules.py`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_fusion_modules.py
from src.fusion.multimodal_lego import MultimodalLegoFusion

def test_lego_topology_a():
    fusion = MultimodalLegoFusion(
        d_common=256, modality_ids=["video", "eye", "ppg"],
        topology="pairwise", n_heads=4,
    )
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

def test_lego_topology_b():
    fusion = MultimodalLegoFusion(
        d_common=256, modality_ids=["video", "eye", "ppg"],
        topology="hierarchical", n_heads=4,
    )
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)

def test_lego_topology_c():
    fusion = MultimodalLegoFusion(
        d_common=256, modality_ids=["video", "eye", "ppg"],
        topology="gated", n_heads=4,
    )
    embeddings = [torch.randn(4, 256), torch.randn(4, 256), torch.randn(4, 256)]
    out = fusion(embeddings, modality_ids=["video", "eye", "ppg"])
    assert out.shape == (4, 256)
```

- [ ] **Step 2: Implement Multimodal Lego**

```python
# src/fusion/multimodal_lego.py
"""Multimodal Lego: composable atomic fusion blocks."""
from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn

from src.fusion.base import BaseFusionModule


class CrossAttnBlock(nn.Module):
    """Cross-attention between two modalities."""
    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        residual = query
        q = self.norm(query)
        out, _ = self.attn(q, key_value, key_value)
        return (residual + out).squeeze(1)


class GatedFusionBlock(nn.Module):
    """Sigmoid-gated element-wise fusion."""
    def __init__(self, d: int, n_inputs: int):
        super().__init__()
        self.gate = nn.Linear(d * n_inputs, n_inputs)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(inputs, dim=-1)
        gates = torch.sigmoid(self.gate(concat))  # [B, n_inputs]
        stacked = torch.stack(inputs, dim=-1)  # [B, D, n_inputs]
        gated = (stacked * gates.unsqueeze(1)).sum(dim=-1)  # [B, D]
        return gated


class MultimodalLegoFusion(BaseFusionModule):
    """Composable fusion with configurable topology."""

    def __init__(
        self,
        d_common: int = 256,
        modality_ids: list[str] | None = None,
        topology: str = "pairwise",
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_common)
        modality_ids = modality_ids or ["video", "eye_tracking", "ppg"]
        self.topology = topology
        self.mod_ids = modality_ids
        n_mod = len(modality_ids)

        if topology == "pairwise":
            # All-pairs cross-attention → concat → MLP
            self.pair_blocks = nn.ModuleDict()
            for a, b in combinations(range(n_mod), 2):
                self.pair_blocks[f"{a}_{b}"] = CrossAttnBlock(d_common, n_heads, dropout)
                self.pair_blocks[f"{b}_{a}"] = CrossAttnBlock(d_common, n_heads, dropout)
            n_outputs = n_mod * (n_mod - 1)
            self.output_mlp = nn.Sequential(
                nn.Linear(d_common * n_outputs, d_common),
                nn.ReLU(),
                nn.Linear(d_common, d_common),
            )

        elif topology == "hierarchical":
            # Fuse (eye+ppg) first, then fuse with video
            self.stage1 = CrossAttnBlock(d_common, n_heads, dropout)
            self.stage2 = CrossAttnBlock(d_common, n_heads, dropout)
            self.output_mlp = nn.Sequential(
                nn.Linear(d_common * 2, d_common),
                nn.ReLU(),
                nn.Linear(d_common, d_common),
            )

        elif topology == "gated":
            # Self-attention per modality → all-pairs cross-attention → gated fusion
            self.self_attns = nn.ModuleList([
                nn.TransformerEncoderLayer(d_common, n_heads, d_common * 4, dropout, batch_first=True, norm_first=True)
                for _ in range(n_mod)
            ])
            self.cross_blocks = nn.ModuleDict()
            for a, b in combinations(range(n_mod), 2):
                self.cross_blocks[f"{a}_{b}"] = CrossAttnBlock(d_common, n_heads, dropout)
            self.gate = GatedFusionBlock(d_common, n_mod)
            self.output_mlp = nn.Linear(d_common, d_common)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Ensure 2D
        embs = [e.squeeze(1) if e.dim() == 3 and e.shape[1] == 1 else e for e in embeddings]
        if embs[0].dim() == 3:
            embs = [e.mean(dim=1) for e in embs]

        if self.topology == "pairwise":
            outputs = []
            for (a, b) in combinations(range(len(embs)), 2):
                outputs.append(self.pair_blocks[f"{a}_{b}"](embs[a], embs[b]))
                outputs.append(self.pair_blocks[f"{b}_{a}"](embs[b], embs[a]))
            return self.output_mlp(torch.cat(outputs, dim=-1))

        elif self.topology == "hierarchical":
            # Assume order: video, eye, ppg -> fuse eye+ppg first
            physio_fused = self.stage1(embs[1], embs[2])  # eye attends to ppg
            combined = self.stage2(embs[0], physio_fused)  # video attends to physio
            return self.output_mlp(torch.cat([combined, physio_fused], dim=-1))

        elif self.topology == "gated":
            # Self-attention (treat as 1-token sequence)
            refined = [sa(e.unsqueeze(1)).squeeze(1) for sa, e in zip(self.self_attns, embs)]
            # Cross-attention pairs
            for (a, b) in combinations(range(len(refined)), 2):
                crossed = self.cross_blocks[f"{a}_{b}"](refined[a], refined[b])
                refined[a] = refined[a] + crossed
            # Gated fusion
            gated = self.gate(refined)
            return self.output_mlp(gated)

        raise ValueError(f"Unknown topology: {self.topology}")
```

Create `configs/fusion/multimodal_lego.yaml`:
```yaml
fusion_type: multimodal_lego
fusion:
  d_common: 256
  lego:
    topology: "pairwise"  # "pairwise", "hierarchical", "gated"
    n_heads: 4
    dropout: 0.1
training:
  lr: 1e-4
  max_epochs: 100
```

- [ ] **Step 3: Run tests and commit**

```bash
conda run -n visphy python -m pytest tests/test_fusion_modules.py -v
git add src/fusion/multimodal_lego.py configs/fusion/multimodal_lego.yaml tests/test_fusion_modules.py
git commit -m "feat: Multimodal Lego with pairwise, hierarchical, and gated topologies"
```

---

**>>> HUMAN REVIEW GATE: Phase 4b complete. Review all advanced architecture results. Proceed to hybrid optimization. <<<**

---

## Phase 5: Analysis & Hybrid Optimization

### Task 25: Results Analysis and Comparison

**Files:**
- Create: `scripts/analyze_results.py`

- [ ] **Step 1: Implement analysis script**

```python
# scripts/analyze_results.py
"""Analyze and compare all experiment results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.registry import ResultsRegistry


def main() -> None:
    registry = ResultsRegistry()
    results = registry.get_all()

    if not results:
        print("No experiments found in registry.")
        return

    df = pd.DataFrame(results)
    print("\n=== Experiment Comparison ===\n")
    print(df[["experiment_name", "fusion_type", "weighted_f1", "ccc"]].to_string(index=False))

    print("\n=== Best by Weighted F1 ===")
    best = df.loc[df["weighted_f1"].idxmax()]
    print(f"  {best['experiment_name']}: F1={best['weighted_f1']:.4f}")

    print("\n=== Best by CCC ===")
    best_ccc = df.loc[df["ccc"].idxmax()]
    print(f"  {best_ccc['experiment_name']}: CCC={best_ccc['ccc']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/analyze_results.py
git commit -m "feat: results analysis and comparison script"
```

---

### Task 26: Hybrid Architecture (Data-Driven)

This task is intentionally under-specified because it depends on Phase 3-4 results.

- [ ] **Step 1: Run `scripts/analyze_results.py` to identify winning modules**
- [ ] **Step 2: Create `src/fusion/hybrid.py` combining best components**
- [ ] **Step 3: Create `configs/fusion/hybrid.yaml`**
- [ ] **Step 4: Add tests for hybrid architecture**
- [ ] **Step 5: Run LOSO evaluation**
- [ ] **Step 6: Iterate based on results (up to 3 hybrid variants)**
- [ ] **Step 7: Commit best hybrid**

```bash
git commit -m "feat: hybrid fusion architecture (data-driven combination)"
```

---

**>>> HUMAN REVIEW GATE: Phase 5 complete. Final review of SOTA results and experiment reports. <<<**

---

## Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|-----------------|
| 0 | 1 | Reference files copied, data validated |
| 1 | 2-11 | Segments, labels, configs, metrics, losses, encoders, embedding cache |
| 2 | 12-17 | Dataset, collation, multi-task heads, fusion base, FusionTrainer |
| 3 | 18-20 | Early, Mid, Late fusion + baseline results |
| 4a | 21-22 | Perceiver IO, Q-Former |
| 4b | 23-24 | HEALNet, Multimodal Lego |
| 5 | 25-26 | Analysis, hybrid optimization, SOTA push |

Total: **26 tasks**, ~130 steps, 5 human review gates.
