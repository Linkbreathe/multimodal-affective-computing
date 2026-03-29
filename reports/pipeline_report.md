# Multimodal Emotion Recognition Pipeline Report
## Task-Aware Segmentation with InceptionTime + Papagei + VideoMAE V2

**Date:** 2026-03-29
**Branch:** ege/training-data-pipeline
**Dataset:** egoEMOTION (28 subjects, 16 tasks each)

---

## 1. Data Sources

### 1.1 Raw Data

The egoEMOTION dataset contains synchronized egocentric recordings from smart glasses and wearable sensors during two experimental sessions:

| Source | Modality | Native Rate | File |
|---|---|---|---|
| Tobii Pro Glasses 3 | Gaze direction (x, y) | 90 Hz | `gaze_90fps.npy` |
| Tobii Pro Glasses 3 | Egocentric video | ~10 FPS | `pov.mp4` |
| Corsano CardioWatch | PPG (ear) | 128 Hz | `ppg_ear_125hz.npy` (resampled) |
| Self-report | 9-class emotion + VAD | Per task | `Session_A_{id}.csv`, `Session_B_{id}.csv` |

**Session structure per subject:**
- **Session A:** 9 video-watching tasks (Amused, Content, Excited, Awe, Neutral, Fear, Sad, Disgust, Anger) with self-reported emotion ratings after each
- **Session B:** 7 interactive tasks (TryNotToLaugh, SadLetter, FlappyBird, Slenderman, Jellybean, Painting, Jenga) with self-reported emotion ratings after each
- **Inter-task periods:** Calibration, questionnaire filling, transitions (excluded from analysis)

### 1.2 Label Space

**Discrete emotions (9-class):** Amused (0), Content (1), Excited (2), Awe (3), Neutral (4), Fear (5), Sad (6), Disgust (7), Anger (8)

**Continuous affect (VAD):** Valence, Arousal, Dominance (self-reported per task)

**Label assignment:** One label per task. All 10s chunks within a task inherit that task's label. For Session B tasks with multiple rounds (e.g., Jenga has 2), emotion scores and VAD are averaged.

---

## 2. Segmentation: From Fixed-Window to Task-Aware

### 2.1 The Problem with the Original Approach

The original pipeline tiled fixed 10s windows across the entire recording timeline (session_A_start to session_B_end), ignoring task boundaries. This caused:

- **Label contamination:** Windows falling on calibration/questionnaire periods received emotion labels they shouldn't have
- **Cross-task leakage:** Windows straddling two tasks mixed distinct emotional contexts under a single label
- **Inflated segment count:** ~424 segments per subject vs ~227 after correction

### 2.2 The Fix: Official egoEMOTION Task-Aware Segmentation

Following the official egoEMOTION repo (`libs/features/extraction.py`), the corrected pipeline:

1. **Splits by task first** using `task_times.npy` for per-task time boundaries
2. **Chunks within each task** into non-overlapping 10s windows (900 samples at 90 Hz)
3. **Discards incomplete trailing chunks** (integer division: a 25s task yields 2 chunks, 5s discarded)
4. **Excludes inter-task gaps** entirely (calibration, questionnaires, transitions)

### 2.3 Segmentation Results

| Metric | Value |
|---|---|
| Subjects | 28 (subset of 40 official) |
| Tasks per subject | 16 (9 Session A + 7 Session B) |
| Total tasks | 448 |
| Total 10s chunks | 6,663 |
| Discarded trailing time | 1,733s (~29 min total) |
| Excluded inter-task time | ~46.9% of recording per subject |
| Validation | PASSED (0 overlaps, 0 boundary violations, all chunks = 900 samples) |

**Emotion class distribution:**

| Emotion | Count | % | Note |
|---|---|---|---|
| Amused | 1,876 | 28.2% | Most frequent (TryNotToLaugh + video) |
| Fear | 1,028 | 15.4% | Slenderman is longest task |
| Content | 863 | 13.0% | |
| Neutral | 832 | 12.5% | |
| Sad | 655 | 9.8% | |
| Excited | 469 | 7.0% | |
| Anger | 420 | 6.3% | |
| Disgust | 379 | 5.7% | |
| Awe | 141 | 2.1% | Rarest (13.3x fewer than Amused) |

**Class imbalance is severe.** Addressed via inverse-frequency class weights in the CE loss.

### 2.4 Impact on Performance

| Segmentation | Weighted F1 | CCC |
|---|---|---|
| Fixed-window (Phase 3, Mar 25) | 0.13 - 0.23 | ~0.0 |
| Task-aware (Phase 4, Mar 28-29) | 0.63 - 0.67 | 0.14 - 0.40 |
| **Improvement** | **+3x to +5x** | **Recovered from zero** |

**This was the single largest improvement in the entire pipeline.** Correcting the segmentation to respect task boundaries and exclude non-task data improved F1 by 3-5x and recovered CCC from near-zero.

---

## 3. Encoder Pipeline

### 3.1 Eye Tracking: InceptionTime (Gaze-Only)

| Property | Value |
|---|---|
| Input | `[B, 2, 900]` channels-first (gaze_x, gaze_y) |
| Output | `[B, 128]` global average pooled |
| Architecture | 3 InceptionResidualBlocks, multi-scale kernels (9, 19, 39), bottleneck |
| Pre-training | Contrastive (NT-Xent, T=0.07), 100 epochs on 23,621 windows |
| Pre-training loss | 5.2 (epoch 1) -> 0.035 (epoch 100) |
| Representation quality | Same-task cosine similarity = 0.971, different-task = 0.873 |

**Design choice:** Gaze-only (2-channel) was chosen over gaze+pupils (4-channel PatchTST) to simplify the input and reduce dependency on pupil quality, which varies across subjects (4 subjects have excluded/degraded pupil data in the official dataset).

### 3.2 PPG: Papagei

| Property | Value |
|---|---|
| Input | `[B, 1250, 1]` single-channel PPG at 125 Hz, z-score normalized |
| Output | `[B, 512]` projection head embeddings |
| Architecture | Foundation model (frozen), from `papagei-foundation-model` |
| Weights | `weights/papagei/papagei_s.pt` (pre-trained, no fine-tuning) |

### 3.3 Video: VideoMAE V2

| Property | Value |
|---|---|
| Input | `[B, 3, 16, 224, 224]` 16-frame clips, ImageNet normalized |
| Output | `[N, 768]` per clip, N = ~6 clips per 10s segment |
| Architecture | ViT-Base from `OpenGVLab/VideoMAEv2-Base` (frozen) |
| Preprocessing | Resize shortest edge to 224, center crop, BGR->RGB, ImageNet normalize |

**Embedding dimensions summary:**

| Modality | Raw Dim | After Projection (d_common=256) |
|---|---|---|
| Eye tracking | 128 | 256 |
| PPG | 512 | 256 |
| Video | 768 | 256 |

---

## 4. Fusion Methods and Results

### 4.1 Training Setup

- **Evaluation:** Leave-One-Subject-Out (LOSO) with 28 folds
- **Per fold:** Train on 26 subjects, validate on 1, test on 1
- **Optimizer:** AdamW (lr=1e-4), CosineAnnealing scheduler
- **Early stopping:** Patience=10, monitored on validation weighted F1
- **Losses:** CE (classification) + KL (soft labels) + CCC (VAD regression), all weighted 1.0
- **Class weights:** Inverse-frequency weighting for the CE loss
- **Batch size:** 64

### 4.2 Results (28-fold LOSO, all methods)

| # | Fusion Method | Params | F1 | F1 Std | CCC | CCC_V | CCC_A | CCC_D |
|---|---|---|---|---|---|---|---|---|
| 1 | **Multimodal LEGO** | ~1.2M | **0.667** | 0.129 | 0.144 | 0.303 | 0.023 | 0.107 |
| 2 | Late | ~0.6M | 0.651 | 0.126 | 0.393 | 0.700 | 0.130 | 0.304 |
| 3 | Mid | ~0.4M | 0.649 | 0.129 | 0.349 | 0.729 | 0.090 | 0.227 |
| 4 | Perceiver IO | ~3.2M | 0.647 | 0.135 | 0.369 | 0.712 | 0.107 | 0.241 |
| 5 | **Q-Former** | ~7.5M | 0.646 | 0.128 | **0.396** | **0.743** | **0.138** | **0.308** |
| 6 | HEALNet | ~9.9M | 0.635 | 0.123 | 0.364 | 0.710 | 0.112 | 0.254 |
| 7 | Early | ~0.3M | 0.641 | 0.127 | 0.337 | 0.691 | 0.062 | 0.258 |

**Paper baseline (classical ML, all modalities): F1 = 0.46**

All 7 methods beat the baseline by 37-45%.

### 4.3 Per-Fold Analysis

**Hardest subjects (consistently low F1 across all methods):**
- S015: F1 = 0.33-0.39 (worst in every method)
- S016: F1 = 0.39-0.50
- S007: F1 = 0.43-0.51

**Easiest subjects (consistently high F1):**
- S031: F1 = 0.82-0.93 (best in 6/7 methods)
- S018: F1 = 0.83-0.89
- S027: F1 = 0.81-0.89

This extreme subject variability (F1 std ~0.12-0.14) suggests that individual expression patterns dominate classification difficulty, more so than the fusion architecture choice.

---

## 5. Problems Encountered

### 5.1 Segmentation Bug (Critical, Fixed)

**Problem:** Original pipeline tiled 10s windows across the full recording, including calibration and questionnaire periods. Labels were assigned by positional index from manifest CSVs.

**Impact:** F1 = 0.13-0.23, CCC ~0.0. The model was training on noise-labeled segments.

**Fix:** Task-aware segmentation following the official egoEMOTION approach.

### 5.2 Video Coordinate Bug (Critical, Fixed)

**Problem:** The video extraction used absolute timestamps (`abs_start_90hz / 90.0`) to seek into `pov.mp4`, but the video file starts at `session_A[0]` (shifted coordinates), not at recording start.

**Impact:** Would have extracted frames from wrong time positions.

**Fix:** Changed to shifted coordinates (`start_90hz / 90.0`), consistent with how the physio files are indexed.

### 5.3 BatchNorm Singleton Crash (Moderate, Identified)

**Problem:** `nn.BatchNorm1d` in early.py (2 instances) and mid.py (1 instance) crashes when batch_size=1 during training. This occurs on certain LOSO folds where the last training batch has exactly 1 sample.

**Impact:** Early and mid fusion crashed at fold 014 in some runs. The issue is non-deterministic (depends on batch shuffling and dataset sizes per fold).

**Fix:** Replace `nn.BatchNorm1d(N)` with `nn.LayerNorm(N)` at 3 locations. LayerNorm operates per-sample, not per-batch.

### 5.4 Codex CLI Sandbox GPU Access (Operational)

**Problem:** The Codex CLI `workspace-write` sandbox blocks GPU access (`device_count=0` inside sandbox) and restricts DataLoader multiprocessing (PermissionError on socket operations).

**Impact:** Pre-training and extraction had to run outside the Codex sandbox with direct bash execution.

**Workaround:** Used `--sandbox danger-full-access` for fusion experiments, which restored GPU access.

### 5.5 Class Imbalance

**Problem:** 13.3x ratio between most common (Amused, 28.2%) and rarest (Awe, 2.1%) class. Interactive tasks (Slenderman, Jenga, TryNotToLaugh) produce far more 10s chunks than video-watching tasks.

**Mitigation:** Inverse-frequency class weights in the CE loss. Not fully resolved -- Awe likely remains underrepresented.

---

## 6. Insights and Analysis

### 6.1 What Improved the Outcome Most

**Ranked by impact:**

1. **Task-aware segmentation (+3-5x F1):** The single most important fix. Without respecting task boundaries, the model was training on mislabeled data from calibration/questionnaire periods. This explains why the Phase 3 experiments (Mar 25) showed near-random performance despite correct encoders and fusion architectures.

2. **Multi-task learning (CE + KL + CCC):** The three-loss formulation forces the shared representation to capture both discrete emotion categories and continuous affect dimensions. The separate soft-label head (KL loss) acts as an implicit regularizer on the shared backbone.

3. **Frozen pre-trained encoders:** Using strong foundation models (VideoMAE V2, Papagei) without fine-tuning avoids overfitting on the small dataset (6,663 segments). The InceptionTime encoder, pre-trained with contrastive learning on all subjects' gaze data, learned meaningful temporal patterns (same-task cosine similarity 0.971 vs different-task 0.873).

4. **LOSO with proper validation split:** Using a separate validation subject (not included in test) for early stopping prevents information leakage and gives honest generalization estimates.

### 6.2 Fusion Architecture Insights

**F1 is remarkably stable across architectures.** The range is only 0.635-0.667 (5% spread) despite architectures varying from a 3-layer MLP (Early, ~0.3M params) to a 6-layer transformer (Q-Former, ~7.5M params). This suggests the bottleneck is in the frozen encoder representations, not the fusion module.

**CCC separates the methods more clearly.** Q-Former achieves the best CCC (0.396), while Multimodal LEGO collapses to 0.144. The Fourier-domain processing in LEGO appears incompatible with the continuous VAD regression task -- its complex-valued latent transformations may distort the linear relationships that CCC measures.

**Late fusion is surprisingly competitive.** With the simplest architecture (per-modality MLPs, averaged), it ranks #2 in F1 and #2 in CCC. This supports the interpretation that the pre-trained encoders already capture most of the relevant information, and the fusion layer mainly needs to combine, not deeply integrate, the modality representations.

**Attention-based methods show no clear advantage over MLPs for F1.** HEALNet (9.9M params, cross-attention) and Perceiver IO (3.2M params, latent attention) both slightly underperform Mid fusion (0.4M params, MLP) on F1. The added modeling capacity doesn't translate to better discrete classification on this dataset size.

### 6.3 VAD Prediction Asymmetry

| VAD Dimension | Best CCC | Method |
|---|---|---|
| Valence | 0.743 | Q-Former |
| Dominance | 0.308 | Q-Former |
| Arousal | 0.138 | Q-Former |

**Valence is well-predicted (CCC ~0.7) while arousal is nearly unpredictable (CCC ~0.1).** This is a known phenomenon in affective computing: valence correlates strongly with facial expression and physiological signals (PPG, gaze patterns), while arousal is more transient and harder to capture from 10s windows. The egocentric perspective (no frontal face camera) further limits arousal-related visual cues.

### 6.4 Subject-Level Variability

The F1 standard deviation across LOSO folds (~0.12-0.14) is larger than the F1 difference between the best and worst fusion methods (~0.03). **The identity of the test subject matters more than the choice of fusion architecture.** This points to:

- Strong individual differences in emotional expression patterns
- Some subjects may have noisier or less discriminative physiological signals
- The 16-task design creates limited label diversity per subject (each emotion appears in only 1-2 tasks)

### 6.5 Recommendations for Next Steps

1. **Address arousal prediction:** Consider longer windows (20s as in the official DL pipeline) or temporal modeling across consecutive segments
2. **Handle class imbalance more aggressively:** Oversample rare classes (Awe, Disgust) or use focal loss
3. **Encoder fine-tuning:** Selective unfreezing of top layers of VideoMAE or Papagei could improve task-specific representations
4. **Cross-subject adaptation:** Subject-specific calibration or domain adaptation to reduce the 0.12+ F1 variability across folds
5. **Expand to full 40 subjects:** Currently using 28 of the 40 available subjects

---

## 7. Reproduction Commands

```bash
# Step 0: Copy resampled data
for d in data/egoemotion_raw/0*/; do
  subj=$(basename $d)
  cp /mnt/c/.../egoEMOTION/$subj/gaze_90fps.npy $d/
  cp /mnt/c/.../egoEMOTION/$subj/ppg_ear_125hz.npy $d/
done

# Step 1: Pre-train InceptionTime
conda run -n visphy python scripts/pretrain_inceptiontime.py \
  --data_dir data/egoemotion_raw --epochs 100 --batch_size 256 --patience 10

# Step 2: Task-aware segmentation + embedding extraction
conda run -n visphy python scripts/segment_and_extract_10s.py --encoder all \
  --output_dir data/embeddings_10s_task_aware

# Step 3: Run fusion experiments
for cfg in early mid late healnet perceiver_io qformer multimodal_lego; do
  conda run -n visphy python scripts/run_experiment_10s.py \
    --fusion_config configs/fusion/${cfg}.yaml \
    --embeddings_dir data/embeddings_10s_task_aware --pool-clips
done

# Step 4: Run ablation study (19 experiments)
for cfg in configs/ablation/*.yaml; do
  name="ablation_$(basename $cfg .yaml)"
  conda run -n visphy python scripts/run_experiment_10s.py \
    --fusion_config $cfg --name $name \
    --embeddings_dir data/embeddings_10s_task_aware --pool-clips
done
```

---

## 8. Ablation Study

A systematic 19-experiment ablation across 4 axes using late fusion as the base method (F1=0.651, CCC=0.393). All experiments use full 28-fold LOSO on 6,663 segments.

### 8.1 Modality Contribution

| Config | Modalities | F1 | CCC | Delta F1 | Delta CCC | Interpretation |
|---|---|---|---|---|---|---|
| m5_video_ppg | V+P | 0.648 | 0.381 | -0.003 | -0.013 | ~= reference |
| m1_video_only | V | 0.645 | 0.395 | -0.006 | +0.002 | Video alone matches full system |
| m4_video_eye | V+E | 0.645 | 0.394 | -0.006 | +0.001 | Adding eye = no gain |
| **Reference** | **V+E+P** | **0.651** | **0.393** | **--** | **--** | |
| m6_eye_ppg | E+P | 0.218 | 0.076 | -0.433 | -0.317 | Physio-only is near-chance |
| m2_eye_only | E | 0.197 | 0.039 | -0.454 | -0.354 | Gaze alone fails |
| m3_ppg_only | P | 0.140 | 0.031 | -0.511 | -0.362 | PPG alone fails |

**Finding:** Video is the sole performance driver. Video-only F1=0.645 is within noise of the 3-modality system (0.651). Eye-tracking and PPG contribute essentially nothing to the current pipeline. The late fusion correctly learns to near-zero-weight the non-video branches.

### 8.2 Loss Function Impact

| Config | Loss | F1 | CCC | Delta F1 | Delta CCC | Interpretation |
|---|---|---|---|---|---|---|
| l2_ce_kl | CE+KL | 0.646 | -0.005 | -0.005 | -0.398 | KL without VAD = CCC collapse |
| l1_ce_only | CE | 0.645 | 0.011 | -0.006 | -0.382 | CE alone = CCC collapse |
| **Reference** | **CE+KL+VAD** | **0.651** | **0.393** | **--** | **--** | |
| l3_ce_vad | CE+VAD | 0.642 | 0.394 | -0.009 | +0.001 | VAD alone fully recovers CCC |

**Finding:** The VAD regression loss is essential for CCC and the KL soft-label loss is expendable. Without VAD, CCC collapses to ~0 regardless of KL. CE+VAD alone fully recovers reference CCC (0.394 vs 0.393). The KL term adds nothing measurable to either F1 or CCC.

### 8.3 Architecture Sensitivity

| Config | d_common | Dropout | F1 | CCC | Delta F1 | Delta CCC |
|---|---|---|---|---|---|---|
| **Reference** | **256** | **0.1** | **0.651** | **0.393** | **--** | **--** |
| a6_dropout_03 | 256 | 0.3 | 0.645 | 0.387 | -0.006 | -0.007 |
| a8_large_reg | 512 | 0.3 | 0.640 | 0.395 | -0.011 | +0.002 |
| a7_compact_reg | 128 | 0.3 | 0.639 | 0.390 | -0.012 | -0.004 |
| a4_dropout_0 | 256 | 0.0 | 0.635 | 0.369 | -0.016 | -0.025 |
| a1_dcommon_128 | 128 | 0.1 | 0.635 | 0.372 | -0.016 | -0.022 |
| a3_dcommon_512 | 512 | 0.1 | 0.631 | 0.389 | -0.020 | -0.005 |

**Finding:** Architecture is remarkably insensitive. All 6 variants span only 0.014 F1 and 0.026 CCC. The default d_common=256 with dropout=0.1 is near-optimal. Removing dropout entirely hurts more than any dimension change, confirming that regularization matters more than capacity.

### 8.4 Training Dynamics

| Config | Patience | LR | F1 | CCC | Delta F1 | Delta CCC |
|---|---|---|---|---|---|---|
| t1_patience_5 | 5 | 1e-4 | 0.646 | 0.368 | -0.005 | -0.025 |
| **Reference** | **10** | **1e-4** | **0.651** | **0.393** | **--** | **--** |
| t3_patience_20 | 20 | 1e-4 | 0.636 | 0.394 | -0.015 | +0.001 |
| t4_lr_1e3 | 10 | 1e-3 | 0.632 | **0.410** | -0.019 | **+0.017** |
| t6_lr_1e5 | 10 | 1e-5 | 0.567 | 0.135 | -0.084 | -0.258 |

**Finding:** Learning rate is the most sensitive hyperparameter. lr=1e-5 is catastrophic (-0.084 F1, -0.258 CCC). lr=1e-3 achieves the study's best CCC (0.410) at minor F1 cost. The default lr=1e-4 with patience=10 is a solid balanced choice.

### 8.5 Ablation Summary

The five key takeaways from the ablation study:

1. **The system is effectively video-only.** Video-only performance (F1=0.645) is within noise of the full trimodal system (F1=0.651). The current InceptionTime gaze encoder and Papagei PPG encoder add no measurable value.

2. **VAD regression loss is the sole driver of CCC.** Without VAD supervision, CCC collapses to ~0. The KL soft-label loss contributes nothing. This implies the multi-task setup's value lies entirely in the continuous affect regression objective, not the soft-label regularization.

3. **Architecture barely matters.** The fusion head is not the bottleneck. All d_common/dropout variants produce F1 within a 0.014 range — dwarfed by the 0.13+ standard deviation across LOSO folds.

4. **Learning rate is the most impactful training knob.** lr=1e-3 achieves the best CCC in the entire study (0.410), suggesting the default 1e-4 may be slightly conservative for CCC optimization.

5. **The clear next priority is stronger physiological encoders.** Before exploring more fusion architectures, the gaze and PPG representations need fundamental improvement to justify the multimodal design.
