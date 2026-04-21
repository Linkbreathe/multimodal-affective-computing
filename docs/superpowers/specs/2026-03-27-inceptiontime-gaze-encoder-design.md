# InceptionTime Gaze Encoder Design

**Date:** 2026-03-27
**Status:** Approved
**Scope:** Replace PatchTST eye-tracking encoder with InceptionTime for gaze-movement-only emotion recognition

## Problem

The current eye-tracking encoder (PatchTST) processes a 4-channel signal combining gaze (yaw, pitch) and pupil (left diameter, right diameter) data. Two issues motivate a redesign:

1. **Pupil data is being dropped** — the encoder should use only gaze movement (2 channels), which carries the movement dynamics relevant to emotion.
2. **Channel independence is wrong for gaze** — PatchTST processes each channel independently and averages at the end. But yaw and pitch jointly define a 2D trajectory — saccade direction, scan path shape, and fixation location are inherently joint yaw-pitch phenomena. A diagonal saccade is indistinguishable from a horizontal one under channel-independent processing.

## Constraints

- **Input:** Raw yaw/pitch in radians, 2 channels, 90 Hz, 10s segments → `[B, 2, 900]`
- **Output:** Fixed-length embedding `[B, 128]` (single vector per segment)
- **Training:** Self-supervised contrastive pre-training → freeze → extract embeddings
- **Interface:** Must implement `BaseEncoder(ABC, nn.Module)` with `forward(x) -> Tensor` and `freeze()`
- **No pupil data, no hand-crafted features** — raw signal learning end-to-end

## Architecture: InceptionTime

### Why InceptionTime

Gaze emotion recognition is fundamentally a **multi-scale temporal pattern recognition** problem:

| Gaze Event | Duration | Samples at 90Hz | Emotional Signal |
|------------|----------|-----------------|------------------|
| Saccade | 20-80ms | 2-7 | Frequency → arousal; amplitude → exploration breadth |
| Short fixation | 200-300ms | 18-27 | Attention capture, orienting response |
| Long fixation | 400-600ms | 36-54 | Deep processing, rumination (sadness), freeze (fear) |
| Scan path pattern | 1-10s | 90-900 | Exploration strategy — hypervigilant (fear) vs. withdrawn (sadness) vs. active (excitement) |

InceptionTime captures all these scales simultaneously through parallel multi-scale convolution branches in every module, without requiring the depth that single-kernel architectures (ResNet1D) need to build up receptive field.

### Model Structure

```
Input [B, 2, 900]
  │
  ├── Residual Block 1
  │     InceptionModule → InceptionModule
  │     Shortcut: Conv1D(1×1) for dim matching
  │
  ├── Residual Block 2
  │     InceptionModule → InceptionModule
  │     Shortcut: Conv1D(1×1)
  │
  ├── Residual Block 3
  │     InceptionModule → InceptionModule
  │     Shortcut: Conv1D(1×1)
  │
  ├── Global Average Pooling (over time)
  │
  └── Output [B, 128]
```

**6 Inception modules** total, grouped into 3 residual blocks of 2.

### Inception Module Detail

Each module has 4 parallel branches:

| Branch | Operation | Kernel Size | Gaze Timescale | What It Captures |
|--------|-----------|-------------|----------------|------------------|
| A | Bottleneck → Conv1D | 9 (~100ms) | Saccade scale | Velocity spikes, onset/offset detection |
| B | Bottleneck → Conv1D | 19 (~210ms) | Short fixation | Fixation detection, brief dwells |
| C | Bottleneck → Conv1D | 39 (~430ms) | Long fixation / scan transition | Fixation clusters, gaze shifts between regions |
| D | MaxPool(3) → Conv1D(1×1) | 3 | Local extrema | Peak velocities, direction reversals |

- **Bottleneck:** Conv1D(1×1) reducing input channels → `nb_filters // 4` (parameter control)
- **Each conv branch:** 32 filters, padding='same', stride=1
- **Concatenation:** 4 branches × 32 filters = **128 output channels**
- **Post-concat:** BatchNorm → ReLU
- **Residual shortcut:** Every 2 modules, Conv1D(1×1) for dimension matching when needed

### Key Design Decisions

- **Kernel sizes 9/19/39** — deliberately aligned to gaze event timescales at 90 Hz, not arbitrary
- **Joint channel processing** — every convolution operates on both yaw and pitch simultaneously, learning saccade direction, trajectory curvature, and 2D scan path shape
- **embed_dim = 128** — proportional to signal complexity (2 channels). Video encoder outputs 768 (for RGB×16 frames), PPG outputs 512 (complex cardiovascular dynamics). 128 is appropriate for 2D gaze movement
- **No dropout in conv layers** — BatchNorm provides sufficient regularization during pre-training
- **~130K parameters** — lightweight, appropriate for the dataset size (2,678 labeled segments, ~35K unlabeled segments for pre-training)

## Self-Supervised Pre-Training

### Objective: Contrastive Learning (NT-Xent / InfoNCE)

Contrastive learning produces discriminative features — it forces the encoder to distinguish between different movement patterns, which transfers better to emotion classification than reconstruction objectives.

### Training Loop

1. Sample a gaze segment `[2, 900]`
2. Create two augmented views → `view_a`, `view_b`
3. Encode both → `z_a, z_b ∈ R^128`
4. Project through MLP head → `p_a, p_b ∈ R^64` (discarded after pre-training)
5. NT-Xent loss: pull same-segment pairs together, push different-segment pairs apart

### Gaze-Specific Augmentations

| Augmentation | Parameters | What It Does | Why It's Valid for Gaze |
|-------------|------------|-------------|------------------------|
| Jittering | σ = 0.005 rad (~0.3°) | Add Gaussian noise | Simulates ET sensor noise; preserves movement patterns |
| Scaling | factor ∈ [0.8, 1.2] | Multiply signal by random factor | Simulates individual differences in gaze range; preserves temporal dynamics |
| Time warping | cubic spline, 4 knots | Smooth non-linear time distortion | Saccade speed varies person-to-person; trajectory shape is what matters |
| 2D rotation | θ ∈ [-15°, +15°] | Rotate [yaw, pitch] vector by angle θ | Head orientation varies across subjects; trajectory shape preserved, only coordinate frame shifts. **Gaze-specific** — physically meaningful transformation |
| Crop & resize | 70-100% sub-segment | Random crop, resample to 900 | Forces model to recognize patterns at different temporal scales |

Each view applies 2-3 randomly selected augmentations in sequence.

### Pre-Training Data

All available gaze data — not just labeled task segments. Includes inter-task periods (calibration, questionnaires), since no labels are needed.

- 40 subjects × ~1 hour each at 90 Hz
- 10s windows with 50% overlap → **~35,000 segments**
- 13× more data than the 2,678 labeled segments

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 100 | Early stopping on contrastive loss plateau |
| Batch size | 256 | Contrastive learning benefits from large batches (more negatives) |
| Optimizer | AdamW | |
| Learning rate | 1e-3 | |
| Weight decay | 1e-4 | |
| LR schedule | Cosine annealing | |
| Temperature τ | 0.07 | Standard NT-Xent temperature |
| Projection head | Linear(128→64) + ReLU + Linear(64→64) | Discarded after pre-training |
| Checkpoint path | `checkpoints/inceptiontime_gaze_pretrained.pt` | |

## Integration with Existing Pipeline

### Encoder Registration

- New file: `src/encoders/inceptiontime.py` → `InceptionTimeGazeEncoder(BaseEncoder)`
- Registered in `ModalityRegistry` as the `eye_tracking` encoder
- Config: `configs/base.yaml` → `eye_tracking.encoder: inceptiontime`, `eye_tracking.embed_dim: 128`
- PatchTST remains in the codebase as an alternative (switch is config-driven)

### Data Loading Changes

| Component | Before | After |
|-----------|--------|-------|
| `segments.py:load_eye_tracking_segment()` | Always loads gaze + pupils → `[T, 4]` | Config-driven: loads gaze-only `[T, 2]` when `encoder=inceptiontime`, loads gaze+pupils `[T, 4]` when `encoder=patchtst` |
| `segment_and_extract_10s.py` | Concatenates 4 channels | Reads encoder config to determine channel loading; transposes to channels-first `[2, 900]` for InceptionTime |
| Input tensor | `[B, 900, 4]` (time-first) | `[B, 2, 900]` (channels-first) for InceptionTime; `[B, 900, 4]` preserved for PatchTST |

### Fusion Compatibility

Output is `[B, 128]` (pooled). `ModalityProjector` maps to `[B, 256]` (d_common).

- **Early/Mid/Late fusion:** Expect `[B, D]` — works directly
- **Perceiver IO, QFormer, HEALNet, MM-Lego:** Expect `[B, T, D]` — `ProjectedFusion` unsqueezes pooled inputs to `[B, 1, D]` when `supports_sequence_input=True`

### New Scripts

- `scripts/pretrain_inceptiontime.py` — contrastive pre-training on all gaze data
- No changes to `scripts/run_experiment_10s.py` — it reads from embedding cache and is encoder-agnostic

### Embedding Cache

- Path: `data/embeddings/egoemotion/10s/inceptiontime/{subject_id}/segment_{idx:04d}.pt`
- Format: `{"embedding": [128], "metadata": dict, "config_hash": str}`
- Existing PatchTST embeddings remain untouched in `data/embeddings/egoemotion/10s/patchtst/`

## Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| `src/encoders/inceptiontime.py` | Create | InceptionTimeGazeEncoder class |
| `scripts/pretrain_inceptiontime.py` | Create | Contrastive pre-training script |
| `src/data/segments.py` | Modify | `load_eye_tracking_segment()` → gaze-only loading |
| `scripts/segment_and_extract_10s.py` | Modify | 2-channel gaze extraction, channels-first |
| `configs/base.yaml` | Modify | `eye_tracking.encoder: inceptiontime` |
| `src/encoders/registry.py` | Modify | Register InceptionTime encoder |
| `tests/test_encoders.py` | Modify | Add InceptionTime encoder tests |
