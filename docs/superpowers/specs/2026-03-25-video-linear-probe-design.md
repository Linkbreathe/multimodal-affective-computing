# Pure Video Linear Probe — Design Spec

## Goal

Build a direct `video embedding → nn.Linear(768, 9) → emotion prediction`
pipeline with LOSO evaluation. No fusion framework, no MLP, no multi-task
head. Reproduces the old repo's probe architecture (`probe_common.py:329`)
under the current repo's evaluation protocol (LOSO).

---

## 1. Input Data Format

### Embedding files

**Location:** `data/embeddings/egoemotion/10s/video_mae_v2/{subject}/segment_{idx:04d}.pt`

**Verified contents** (from `005/segment_0005.pt`):

```python
{
    "embedding": torch.float32, shape [6, 768],  # 6 clips × 768-dim
    "segment_idx": 5,                             # int
    "subject": "005",                             # str
    "label": 4,                                   # int, 0-8
    "emotion": "Neutral",                         # str
}
```

**Confirmed:** All segments have exactly 6 clips (uniform `[6, 768]` shape
across subjects 005, 010, 025). No variable-length padding needed.

### Manifest

**Location:** `data/datasets/egoemotion_raw/ce_hardlabel_manifests/dataset_manifest.csv`

**Columns used:** `subject` (int), `segment_path` (str, segment index
extracted from filename), `label` (int, 0-8).

**Total:** 2,678 segments across 40 subjects.

**Label distribution (imbalanced):**

| Label | Emotion | Count |
|-------|---------|-------|
| 0 | Amused | 682 |
| 1 | Content | 390 |
| 2 | Excited | 115 |
| 3 | Awe | 103 |
| 4 | Neutral | 449 |
| 5 | Fear | 232 |
| 6 | Sad | 338 |
| 7 | Disgust | 185 |
| 8 | Anger | 184 |

### Embedding reduction

Each `.pt` file stores `[6, 768]`. Reduce to one sample-level vector:

```python
emb = torch.load(path, weights_only=False)["embedding"]  # [6, 768]
emb = emb.mean(dim=0)                                     # [768]
```

This is a simple mean over the 6 clip embeddings. The result is one `[768]`
vector per 10s segment, paired with one integer label `0-8`.

### Label pairing

Each `.pt` file contains its own `"label"` field. Use that directly —
no manifest lookup needed during loading. The manifest is only used to
enumerate which (subject, segment_idx) pairs exist.

---

## 2. Script Structure

### New file: `scripts/run_linear_probe_10s.py`

**Imports — what is used:**

```python
import torch, torch.nn as nn, torch.optim, numpy as np, pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import f1_score               # macro F1
from src.utils.metrics import compute_class_weights  # reuse existing
```

**Imports — what is NOT used:**

No import from `src/fusion/`, `src/tasks/`, `src/trainer/`,
`scripts/run_experiment.py`. The script is fully standalone.

### Functions

#### `load_video_embeddings(embeddings_dir, manifest) -> dict[str, list[dict]]`

```
Input:
    embeddings_dir: str  — path to data/embeddings/egoemotion/10s/video_mae_v2
    manifest: pd.DataFrame  — loaded from dataset_manifest.csv

Output:
    dict mapping subject_id (str, e.g. "005") to list of
    {"embedding": Tensor[768], "label": int}

Logic:
    for each (subject, segment_idx) in manifest:
        path = embeddings_dir / subject / f"segment_{seg_idx:04d}.pt"
        data = torch.load(path)
        emb = data["embedding"].mean(dim=0)   # [6, 768] → [768]
        label = int(data["label"])
        append to subject's list
```

#### `macro_f1_score(y_true, y_pred) -> float`

```
Wraps sklearn: f1_score(y_true, y_pred, average="macro", zero_division=0)
```

This matches the old repo's `compute_macro_f1` semantics (unweighted mean
of per-class F1). Using sklearn instead of hand-rolling for correctness.

#### `train_probe_fold(train_embs, train_labels, val_embs, val_labels, device, seed) -> nn.Linear`

```
Input:
    train_embs: Tensor[N_train, 768]
    train_labels: Tensor[N_train] (long)
    val_embs: Tensor[N_val, 768]
    val_labels: Tensor[N_val] (long)
    device: str
    seed: int

Output:
    nn.Linear(768, 9) — best checkpoint by val macro F1

Training recipe (matches old repo probe_common.py):
    probe = nn.Linear(768, 9)
    class_weights = compute_class_weights(train_labels.numpy(), 9)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 256
    max_epochs = 100
    patience = 20
    no scheduler

Loop:
    for epoch in range(max_epochs):
        # --- train ---
        probe.train()
        for emb_batch, label_batch in train_loader:  # shuffle=True
            logits = probe(emb_batch)                 # [B, 9]
            loss = loss_fn(logits, label_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # --- validate ---
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_embs.to(device))
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()
        val_macro = macro_f1_score(val_labels.numpy(), val_preds)

        # --- checkpoint selection ---
        if val_macro > best_f1:
            best_f1 = val_macro
            best_state = copy.deepcopy(probe.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    probe.load_state_dict(best_state)
    return probe
```

#### `evaluate_probe(probe, embs, labels, device) -> dict`

```
Input:
    probe: nn.Linear(768, 9)
    embs: Tensor[N, 768]
    labels: Tensor[N] (long)

Output:
    {"macro_f1": float, "weighted_f1": float}

Logic:
    probe.eval()
    with torch.no_grad():
        preds = probe(embs.to(device)).argmax(dim=-1).cpu().numpy()
    y_true = labels.numpy()
    return {
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, preds, average="weighted", zero_division=0),
    }
```

#### `run_loso(data_by_subject, device, seed) -> list[dict]`

```
Input:
    data_by_subject: dict[str, list[dict]]  — from load_video_embeddings
    device: str
    seed: int

Output:
    list of 40 dicts, one per fold:
    {"test_subject": str, "macro_f1": float, "weighted_f1": float}

Logic:
    subject_ids = sorted(data_by_subject.keys())
    for i, test_subj in enumerate(subject_ids):
        torch.manual_seed(seed + i)

        val_subj = subject_ids[(i + 1) % len(subject_ids)]
        train_subjs = [s for s in subject_ids if s not in (test_subj, val_subj)]

        train_embs, train_labels = stack_subjects(train_subjs)
        val_embs, val_labels = stack_subjects(val_subj)
        test_embs, test_labels = stack_subjects(test_subj)

        probe = train_probe_fold(
            train_embs, train_labels, val_embs, val_labels,
            device, seed + i,
        )
        metrics = evaluate_probe(probe, test_embs, test_labels, device)
        metrics["test_subject"] = test_subj
        fold_results.append(metrics)

    return fold_results
```

#### `main()`

```
CLI args:
    --embeddings-dir  default="data/embeddings/egoemotion/10s/video_mae_v2"
    --data-dir        default="data/datasets/egoemotion_raw"
    --device          default=None (auto-detect)
    --seed            default=42
    --name            default=None

Flow:
    1. Load manifest CSV
    2. Load and pool all video embeddings
    3. Run LOSO
    4. Print results table (per-fold + mean ± std)
    5. Save report to reports/
```

---

## 3. What Is Controlled vs Uncontrolled

### Controlled in this experiment (matched to old repo)

| Setting | Value | Old repo source |
|---------|-------|-----------------|
| Probe model | `nn.Linear(768, 9)` | `probe_common.py:329` |
| Loss | `CrossEntropyLoss(weight=class_weights)` | `probe_common.py:332-337` |
| Class weight formula | `N / (C * count_c)` | `probe_common.py:140-144` matches `metrics.py:28-33` |
| Optimizer | `AdamW(lr=1e-3, wd=1e-4)` | `probe_common.py:338-342` |
| Batch size | 256 | `probe_common.py:38` |
| Max epochs | 100 | `probe_common.py:37` |
| Patience | 20 | `probe_common.py:41` |
| Model selection | Val **macro F1** | `probe_common.py:381` |
| Scheduler | None | No scheduler in `probe_common.py` |
| Embedding shape | `[768]` global-pooled | `export_video_embeddings.py:167` produces `[768]` |
| No fusion | Confirmed | `probe_common.py` has no fusion imports |

### Intentionally different (with justification)

| Setting | Old repo | This experiment | Why |
|---------|----------|-----------------|-----|
| Eval protocol | Fixed single split | LOSO 40-fold | Current repo standard; more rigorous |
| Reported metrics | Macro F1 only | Macro F1 **and** weighted F1 | Enables comparison to both repos |
| LOSO fold assignment | N/A | val = next subject (circular) | Matches `fusion_trainer.py:286` |
| Seed per fold | N/A | `seed + fold_index` | Matches `fusion_trainer.py:282-284` |

### Uncontrolled (remaining differences)

| Factor | Old repo | This experiment | Status |
|--------|----------|-----------------|--------|
| **Encoder state** | Fine-tuned on EgoEmotion | Frozen self-supervised | **The variable under test** |
| **Pretrained checkpoint** | `vit_b_k710_dl_from_giant.pth` | `OpenGVLab/VideoMAEv2-Base` | Different starting point |
| **Pooling mechanism** | `forward_features()` with learned LayerNorm | `mean(dim=0)` over 6 clip embeddings | Different pooling |
| **Clip sampling** | 16 frames, stride 6, from full 10s | 6 non-overlapping 16-frame clips | Different temporal coverage |

---

## 4. Interpretation Guide

| Linear probe result | Comparison to Phase 1 MLP (0.2279) | Meaning |
|---------------------|-------------------------------------|---------|
| ≈ 0.22 | Same | Frozen features are the bottleneck; architecture irrelevant |
| > 0.22 significantly | Higher | MLP was harmful (overfitting or poor optimization) |
| < 0.22 | Lower | MLP extracted some useful nonlinear structure |

Regardless of outcome: the linear probe macro F1 on LOSO minus the old
repo's macro F1 on fixed split gives an **upper bound** on the feature
quality gap (because LOSO is strictly harder than fixed split).

---

## 5. Files

| File | Change |
|------|--------|
| `scripts/run_linear_probe_10s.py` | **New.** ~150 lines, fully standalone. |
| `tests/test_linear_probe.py` | **New.** Synthetic-data test for train_probe_fold. |

---

## 6. Executable Checklist

```bash
# 1. Implement the script (see structure above)

# 2. Run the test
conda run -n visphy python -m pytest tests/test_linear_probe.py -v

# 3. Run the experiment
conda run -n visphy python scripts/run_linear_probe_10s.py \
    --name linear_probe_video_only --device cuda

# 4. Expected output format:
#   Fold 005: macro_f1=X.XXXX, weighted_f1=X.XXXX
#   ...
#   LOSO Results (40 folds):
#     Macro F1:    X.XXXX +/- X.XXXX
#     Weighted F1: X.XXXX +/- X.XXXX
#     Phase 1 MLP baseline: 0.2279 weighted F1
#     Old repo reference:   reported macro F1 on fixed split
```

---

## Out of Scope

- Fine-tuning VideoMAE on EgoEmotion
- Any fusion module (`src/fusion/`)
- Multi-task loss (KL, CCC/VAD)
- Sequence-aware architectures
- The `src/trainer/` or `src/tasks/` modules
