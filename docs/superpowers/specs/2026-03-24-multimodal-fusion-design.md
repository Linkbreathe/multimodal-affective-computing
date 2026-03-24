# Multimodal Fusion for Emotional State Recognition — Design Spec

**Date:** 2026-03-24
**Status:** Approved
**Branch:** ege/baselines

---

## 1. Objective

Design and optimize a multimodal fusion layer for emotional state recognition by fusing egocentric video, eye tracking, and PPG signals through frozen encoders. The system must support multiple prediction tasks simultaneously, allow modalities to be attached/detached without code changes, and systematically explore fusion architectures from baselines to SOTA.

## 2. Dataset

**EgoEmotion** (NeurIPS 2025, ETH Zurich)
- 40 subjects (IDs 005–046, excluding 001–004, 020, 041)
- 50+ hours of recordings from Meta Project Aria glasses
- Raw data: `/home/link/Wei/Models/core/real-time-vis-physio-fusion/data/egoemotion_raw/`
- Reference data: `/mnt/c/Users/Public/Data/egoEMOTION/egoEMOTION/`
- Reference codebase: `/home/link/Wei/Models/core/egoEMOTION/`

### Per-Subject Data

| Stream | File | Sampling Rate | Shape |
|--------|------|---------------|-------|
| POV video | `pov.mp4` | 10 FPS | `(N, 1408, 1408)` |
| Gaze | `gaze_90fps.npy` | 90 Hz | `(N, 2)` |
| Pupils | `pupils_90fps.npy` | 90 Hz | `(N, 2)` |
| PPG (ear) | `ppg_ear_125hz.npy` | 125 Hz | `(N, 1)` |
| PPG (ear, resampled) | `ppg_ear_90fps.npy` | 90 Hz | `(N, 1)` |
| Session annotations | `Session_A_*.csv`, `Session_B_*.csv` | — | — |

### Segment Definition

- `task_times.npy` (from reference data) defines segment boundaries in the 90Hz eye-tracker timeline
- Each segment corresponds to one emotion-elicitation video or one naturalistic task
- Configurable `chunk_len`: full task segments initially, sub-chunking optional later

### Label Schemes (Multi-Task)

| Task | Labels | Loss | Primary? |
|------|--------|------|----------|
| Discrete emotions | 9 classes (Amused, Content, Excited, Awe, Neutral, Fear, Sad, Disgust, Anger) | Cross-entropy | Yes |
| Soft emotion distribution | Probability over 9 emotions | KL divergence | No |
| Continuous affect | Valence, Arousal, Dominance | MSE / CCC | No |

### Evaluation Protocol

- **Leave-One-Subject-Out (LOSO):** 40 folds, each subject held out once
- **Primary metric:** Weighted F1 on 9-class discrete emotions
- **Secondary metrics:** KL divergence (soft labels), CCC (continuous affect)
- **Target:** Weighted F1 > 0.46 (paper's best with all modalities, classical ML)

### Pre-Existing Splits

Manifest CSVs exist under `ce_hardlabel_manifests/`, `kl_softlabel_manifests/`, `vad_binary_quadrant_manifests/` with train/val/test splits (2,129 / 280 / 272 samples). These are available for reference but LOSO is the primary evaluation protocol.

## 3. Frozen Encoders

### 3.1 Video — VideoMAE V2

- **Reference:** https://arxiv.org/abs/2303.16727
- **Input:** 16-frame clips at 224x224, extracted from `pov.mp4` at segment boundaries
- **Output:** `[num_clips, 768]` per segment
- **Notes:** POV video is 10 FPS. For a segment of duration T seconds, extract T*10 frames, group into 16-frame clips with stride. Pre-extract and cache all embeddings.

### 3.2 Eye Tracking — PatchTST (Self-Supervised Pre-Training)

- **Reference:** Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- **Input:** 4 channels — gaze_x, gaze_y, pupil_left, pupil_right at 90Hz
- **Output:** `[T', 128]` per segment (T' = number of patches), mean-poolable to `[1, 128]`

**Architecture:**
- Patch length: 45 samples (0.5s at 90Hz)
- Stride: 22 samples (50% overlap)
- Embedding dim: 128
- Transformer layers: 3
- Attention heads: 4
- Channel independence: true

**Pre-training:**
- Self-supervised masked patch prediction (40% masking ratio)
- Train on ALL subjects' full eye tracking recordings (not just labeled segments)
- Pre-train once, freeze, then use identically to other encoders

### 3.3 PPG — Papagei

- **Reference:** https://arxiv.org/pdf/2410.20542
- **Input:** Single-channel PPG from ear (`ppg_ear_125hz.npy`) at 125Hz
- **Output:** `[1, 768]` or `[T', 768]` per segment
- **Notes:** Ear PPG chosen over nose for lower motion artifact in egocentric settings. Nose PPG has documented quality issues for several subjects (009, 011, 013, 017, 028, 034).

## 4. Detachable Modality System

Config-driven modality registry. Any modality can be added or removed without changing fusion layer code.

```yaml
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
```

**Design principles:**
- Fusion layers receive `List[Tensor]` of variable length — no fixed position assumptions
- Learned modality-type embeddings added so fusion layers distinguish modality origin
- Each modality projected to `D_common` (default 256) before fusion
- Adding a new modality (e.g., EEG) = adding a registry entry + encoder wrapper. Zero fusion code changes.
- Ablation studies: disable any modality via config to measure individual contribution

## 5. Embedding Cache

One-time pre-extraction of all frozen encoder outputs, cached to disk.

```
data/embeddings/
├── video_mae_v2/
│   ├── {subject_id}/segment_{idx}.pt
│   └── ...
├── patchtst_eye/
│   ├── {subject_id}/segment_{idx}.pt
│   └── ...
└── papagei_ppg/
    ├── {subject_id}/segment_{idx}.pt
    └── ...
```

- Format: `.pt` (PyTorch tensors)
- Each file stores the embedding tensor + metadata (segment boundaries, subject, label)
- Cache invalidation: hash of encoder config + data path; rebuild if changed
- Estimated total cache size: ~2-5 GB (embeddings are much smaller than raw data)

## 6. Shared Infrastructure — FusionTrainer

### 6.1 FusionTrainer Class

Central experiment runner:
- Takes a fusion model, modality config, and task config
- Handles LOSO loop: 40 folds, each subject as held-out test
- Per-fold: train on 39 subjects, evaluate on 1, collect predictions
- Aggregate across folds: mean/std for all metrics
- Deterministic seeding: `seed = global_seed + fold_index`

### 6.2 Multi-Task Heads

```
Fusion Output [B, D_fused]
    ├── Discrete Emotion Head → [B, 9]    (CE loss, primary)
    ├── Soft Label Head        → [B, 9]    (KL divergence loss)
    └── VAD Head               → [B, 3]    (MSE / CCC loss)
```

- Shared fusion backbone, separate linear projection heads
- Multi-task loss: `L = λ₁·L_CE + λ₂·L_KL + λ₃·L_VAD`
- λ weights configurable, default equal weighting
- Model selection based on discrete emotion weighted F1 (primary)

### 6.3 TensorBoard Logging

```
runs/
├── {experiment_name}_{timestamp}/
│   ├── fold_{subject_id}/
│   │   ├── train/loss, lr, per_task_loss
│   │   └── val/loss, f1, kl, ccc
│   ├── aggregate/
│   │   ├── mean_f1, std_f1
│   │   ├── mean_ccc, std_ccc
│   │   └── per_class_f1
│   └── hparams
```

### 6.4 System Logging

- Python `logging` module → file + console
- Logs: config dump at start, per-epoch metrics, fold summaries, timing, GPU memory
- Saved to `logs/{experiment_name}_{timestamp}.log`

### 6.5 Experiment Reporting

Auto-generated Markdown report at pipeline completion:
- Config summary, dataset stats
- Per-fold results table
- Aggregate metrics with confidence intervals
- Best/worst subject analysis
- Comparison table against previous experiments
- Saved to `reports/{experiment_name}_{timestamp}.md`

### 6.6 Results Registry

JSON file tracking all completed experiments:
- Fields: experiment name, fusion type, primary F1, CCC, KL, timestamp, config hash
- Enables automatic cross-experiment comparison tables

## 7. Fusion Architectures

### 7.1 Baselines (Phase 3)

All operate on cached embeddings projected to `D_common` (default 256).

**Early Fusion:**
- Concatenate projected embeddings: `[B, D_common * num_modalities]`
- Shared MLP (2-3 layers, ReLU, dropout, batch norm)
- → Multi-task heads

**Mid Fusion:**
- Per-modality MLP (1-2 layers) after projection
- Concatenate refined embeddings: `[B, D_common * num_modalities]`
- Shared cross-modal MLP (2-3 layers)
- → Multi-task heads

**Late Fusion:**
- Per-modality full MLP branch → per-modality task predictions
- Decision-level fusion: (a) simple averaging, (b) learned weighted averaging
- Both variants evaluated

**Shared hyperparameter search:**
- Learning rate: {1e-3, 1e-4}
- Dropout: {0.1, 0.3}
- D_common: {128, 256}

### 7.2 Advanced Architectures (Phase 4)

All implement a common interface: `forward(List[Tensor], modality_ids) → Tensor[B, D_fused]`

**Perceiver IO** (Phase 4a)
- Reference: https://arxiv.org/abs/2107.14795
- Learnable latent array `[N_latent, D_latent]` cross-attends to concatenated modality embeddings
- Repeated cross-attention → self-attention blocks (L layers)
- Output decoded via output cross-attention to task dimensions
- Naturally handles variable modality count/length
- Hyperparams: `N_latent` {32, 64}, `D_latent` {256, 512}, `L` {2, 4}

**Q-Former** (Phase 4a)
- Reference: https://arxiv.org/abs/2301.12597
- Learnable query tokens cross-attend to all modality embeddings
- Interleaved cross-attention and self-attention layers
- Output: `[B, N_query, D_query]` → pooled → multi-task heads
- Query bottleneck forces task-relevant feature compression
- Hyperparams: `N_query` {8, 16, 32}, `D_query` {256}, layers {2, 4}

**HEALNet** (Phase 4b)
- Reference: https://arxiv.org/abs/2311.09115
- Shared memory `[B, S, D]` updated iteratively by each modality via cross-attention
- Modalities contribute sequentially per layer, permutation-invariant via shared parameters
- Handles missing modalities by skipping their update step
- Hyperparams: `S` {16, 32}, `D` {256}, `L` {2, 4}

**Multimodal Lego** (Phase 4b)
- Reference: https://arxiv.org/abs/2405.19950
- Library of atomic fusion operations: self-attention, cross-attention, pooling, gating
- Architecture defined as configurable sequence of blocks
- Supports systematic topology search
- Enables recombination of winning blocks from other architectures

## 8. Hybrid Optimization (Phase 5)

### 8.1 Analysis

- Cross-architecture comparison matrix (F1, CCC, KL)
- Per-emotion breakdown: which fusion method excels at which class
- Attention visualization: modality contribution analysis
- Ablation: best architectures re-run with one modality removed

### 8.2 Module Extraction

Identify winning components:
- Best cross-attention mechanism
- Best modality interaction pattern
- Best pooling/readout strategy

### 8.3 Hybrid Construction

- Combine best modules into 2-3 custom fusion candidates
- Evaluate with same LOSO protocol
- Iterate on layer counts, dimensions, dropout

### 8.4 SOTA Push

- Deep hyperparameter search on best hybrid
- Learning rate scheduling (cosine, warmup)
- Multi-task loss weighting (uncertainty weighting, GradNorm)
- Data augmentation: time-series jittering, temporal crop variations
- Ensemble: top-K models averaged

### 8.5 Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| Weighted F1 (discrete emotions) | > 0.46 | > 0.55 |
| CCC (continuous affect) | > 0.75 | > 0.80 |
| Worst-class F1 improvement | Measurable | > 0.30 |

## 9. Sub-Agent Architecture

Hybrid orchestration: Master Agent coordinates, sub-agents handle mechanical tasks, human reviews at checkpoints.

| Agent | Responsibility | Active Phase |
|-------|---------------|--------------|
| Data Engineer | Encoder integration, embedding extraction, caching, data pipeline | 1–2 |
| Infra Architect | FusionTrainer, multi-task heads, LOSO loop, TensorBoard, reporting | 2 |
| Baseline Builder | Early/Mid/Late fusion implementations | 3 |
| Advanced Fusion A | Perceiver IO + Q-Former | 4a |
| Advanced Fusion B | HEALNet + Multimodal Lego | 4b |
| Optimizer | Result analysis, module extraction, hybrid architecture, SOTA push | 5 |
| QA/Review | Code review, experiment validation, report generation | 1–6 |

Maximum 3-4 agents active concurrently.

## 10. Project Structure

```
real-time-vis-physio-fusion/
├── configs/                  # YAML experiment configs
│   ├── base.yaml
│   └── fusion/
│       ├── early.yaml
│       ├── mid.yaml
│       ├── late.yaml
│       ├── perceiver_io.yaml
│       ├── qformer.yaml
│       ├── healnet.yaml
│       └── multimodal_lego.yaml
├── src/
│   ├── data/                 # Data loading, segment extraction, caching
│   ├── encoders/             # Frozen encoder wrappers (VideoMAE, PatchTST, Papagei)
│   ├── fusion/               # Fusion layer implementations
│   ├── tasks/                # Multi-task heads and losses
│   ├── trainer/              # FusionTrainer, LOSO loop
│   └── utils/                # Logging, reporting, registry
├── data/                     # Raw data + cached embeddings
│   ├── egoemotion_raw/
│   └── embeddings/
├── runs/                     # TensorBoard logs
├── logs/                     # System logs
├── reports/                  # Auto-generated experiment reports
├── checkpoints/              # Model checkpoints per experiment/fold
└── docs/
    └── superpowers/specs/    # Design documents
```

## 11. Environment

- **Conda environment:** `visphy` (Python 3.10.20)
- **GPU:** Single consumer GPU, 24GB VRAM
- **Key packages:** PyTorch 2.10+cu128, transformers 4.49, timm 1.0.25, tensorboard 2.20, decord 0.6.0, scikit-learn 1.1.3, opencv 4.11
- **Additional installs:** Only if Papagei requires custom dependencies not in `visphy`

## 12. Phased Execution Plan

| Phase | Description | Parallelism | Checkpoint |
|-------|-------------|-------------|------------|
| 1 | Encoder integration + embedding cache | Sequential | All embeddings cached, verified |
| 2 | FusionTrainer infrastructure | Sequential | LOSO loop running, TensorBoard logging confirmed |
| 3 | Baseline fusion (Early, Mid, Late) | Sequential | Baseline F1 numbers, comparison table |
| 4a | Perceiver IO + Q-Former | Parallel pair | Results for both, added to registry |
| 4b | HEALNet + Multimodal Lego | Parallel pair | Results for both, added to registry |
| 5 | Analysis + hybrid optimization | Sequential | Best hybrid architecture, SOTA results |

Human review gates after Phases 2, 3, 4a, 4b, and 5.
