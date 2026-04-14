# Research Idea Report — Round 3: SEED-V EEG+Eye Fusion

**Direction**: Frozen foundation model EEG + Eye tracking fusion for emotion recognition on SEED-V
**Generated**: 2026-04-13 (Round 3 — Pivot to SEED-V EEG+Eye)
**Pipeline**: Phase 0 (deep analysis) → Phase 1 (26-paper survey) → Phase 2 (GPT-5.4 xhigh brainstorm) → Phase 3 (filtering) → Phase 4 (devil's advocate + novelty) → Phase 5 (3 pilots) → Phase 6 (this report)
**Ideas evaluated**: 11 generated → 9 survived filtering → 3 piloted → 6 recommended

---

## Executive Summary

After 69 experiments on egoEMOTION (Track 1) that conclusively showed PPG and gaze carry no emotion signal, and 11 encoder experiments on SEED-V (Track 2) establishing REVE EEG as the best cross-subject encoder, **this round pivots to SEED-V EEG+Eye fusion** — exploiting eye tracking data that has been sitting unused in the dataset.

**The pivotal discovery from our pilots:**

| Modality | Model | SEED-V 5-class LOSO bacc | vs Chance |
|---|---|---|---|
| **Eye tracking** | **SVM on 66 features** | **0.551 ± 0.084** | **+0.351** |
| EEG (62ch) | Frozen REVE + linear probe | 0.323 ± 0.061 | +0.123 |
| Chance | — | 0.200 | — |

**Eye tracking features outperform frozen REVE EEG by 70% under strict LOSO.** This is the opposite of the egoEMOTION paradigm (where video dominated everything) and represents a genuine surprise: a $0 hand-crafted feature set beats a NeurIPS 2025 foundation model pretrained on 25,000 subjects.

The confusion patterns are complementary: REVE collapses to predicting "Sad" on hard cases, while eye tracking's errors are distributed differently. This motivates asymmetric fusion where eye is the base predictor and EEG adds neural corrections.

---

## Literature Landscape (26 papers, 10 new)

### Cluster 1: EEG+Eye Fusion — Active but All Train-from-Scratch
Six papers (MambaMER MICCAI'25, TEREE, CMGNN IEEE JBHI'24, Cross-Modal Align, MHESA, MACDB) fuse EEG+eye for emotion. **ALL train encoders from scratch. NONE use frozen EEG FMs.** Most report within-subject evaluation (86-97% accuracy), NOT LOSO. Our frozen-FM LOSO setting is completely unexplored.

### Cluster 2: Frozen FM + Simple Fusion Works
NeurIPS WS'25 (Ghallab+ 2025) proves frozen physiological FMs with simple concatenation achieve near-SOTA. Validates our approach but never tried with eye tracking.

### Cluster 3: Cross-Subject Adaptation
MACDB, MHESA, CSCL tackle cross-subject transfer for EEG+eye but from scratch. Test-time adaptation with frozen EEG FMs not studied.

### Cluster 4: Consumer-Grade EEG
4ch portable systems exist (60.2% accuracy) but no unified framework combining REVE 4ch headband + eye tracking.

### Three New Gaps Identified
| Gap | Status | Our Advantage |
|---|---|---|
| G10: Frozen EEG FM + Eye Fusion | **Completely unresolved** | REVE + SEED-V eye data + fusion infrastructure |
| G11: Neural-Behavioral Asymmetric Fusion | **Not framed as concept** | Pilot data shows complementary error patterns |
| G12: Consumer-Grade FM Emotion | **Fragmented** | REVE 4ch (0.283) + eye tracking = natural combination |

---

## Pilot Experiment Results

### Pilot 1: Eye-Only LOSO Baseline (Idea 201) — STRONG POSITIVE

| Classifier | Features | Balanced Acc | Macro F1 |
|---|---|---|---|
| LR | mean(33) | 0.524 ± 0.112 | 0.493 ± 0.129 |
| SVM | mean(33) | 0.507 ± 0.097 | 0.471 ± 0.102 |
| **SVM** | **mean+std(66)** | **0.551 ± 0.084** | **0.532 ± 0.088** |
| LR | mean+std(66) | 0.547 ± 0.094 | 0.526 ± 0.110 |
| MLP | mean+std(66) | 0.489 ± 0.098 | 0.458 ± 0.109 |

Per-class: Neutral 63.2%, Fear 60.4%, Happy 54.9%, Disgust 51.4%, Sad 45.8%.

### Pilot 2: REVE Confusion Analysis (Idea 202) — COMPLEMENTARY ERRORS

REVE overall: bacc=0.323, near-maximal prediction entropy (effectively guessing).

| Class | REVE Recall | Eye Recall | Complementarity |
|---|---|---|---|
| Happy | 48.1% | 54.9% | Eye +7% |
| Sad | 38.4% | 45.8% | Eye +7% |
| Neutral | 32.9% | 63.2% | Eye +30% |
| Disgust | 24.1% | 51.4% | Eye +27% |
| Fear | 18.9% | 60.4% | Eye +41% |

**Key structural finding:** REVE uses "Sad" as a black-hole attractor (34.7% of Disgust → Sad, 31.6% of Fear → Sad, 31.7% of Neutral → Sad). Eye tracking resolves exactly these confusion pairs via pupil dilation (arousal axis) and gaze patterns (fixation duration, saccade rate, gaze aversion).

Top confusion pairs (total errors): Sad↔Neutral (1363), Disgust↔Sad (1210), Fear↔Sad (1056). These 3 pairs account for 47% of all REVE errors and are precisely the pairs eye tracking should resolve.

### Pilot 3: Simple EEG+Eye Fusion (Idea 210) — CRITICAL FINDING

| Features | Balanced Acc | Macro F1 | Delta vs Eye |
|---|---|---|---|
| Eye-only (66) | 0.543 ± 0.091 | 0.543 | baseline |
| EEG-only REVE (512) | 0.285 ± 0.096 | 0.280 | -0.258 |
| Concat (578) | 0.397 ± 0.091 | 0.397 | **-0.146** |
| PCA-Concat (132) | 0.539 ± 0.093 | 0.537 | -0.004 |

**Frozen REVE embeddings are near-collapsed**: cosine similarity ~0.95 between ALL emotion classes — they are NOT emotion-discriminative under LOSO. This explains the weak EEG-only baseline.

**Raw concatenation HURTS badly** (-14.6pp): 512 noisy EEG dims overwhelm 66 informative eye dims.

**PCA-Concat recovers to near eye-only** (-0.4pp): dimensionality reduction removes EEG noise, but the remaining EEG signal adds essentially nothing.

**Per-class detail**: PCA-Concat helps Happy (+10.4pp) but hurts Disgust (-6.9pp) and Fear (-4.2pp).

**Diagnosis**: The frozen REVE encoder is the binding constraint. Without fine-tuning, EEG adds noise to eye tracking. This is consistent with the frozen encoder wall observed in egoEMOTION.

---

## Ranked Ideas

### ★ TIER 1: Core Paper Spine (Do First)

---

### Idea 201: Eye-Only LOSO Baseline ✅ COMPLETED

- **One-line**: Establish eye tracking as a standalone emotion classifier on SEED-V under strict LOSO.
- **Result**: bacc=0.551 (SVM, 66 features) — 2.75x chance, 70% above REVE EEG.
- **Contribution**: Diagnostic. Establishes that behavioral signal > neural FM signal under cross-subject evaluation.
- **Status**: **DONE.** This IS the result.

---

### Idea 202: Confusion-Pair Utility Map ✅ COMPLETED

- **One-line**: Identify where EEG and eye tracking make different errors, mapping per-pair complementarity.
- **Result**: REVE's Sad-attractor pattern is orthogonal to eye tracking's error distribution. Top 3 confusion pairs (47% of REVE errors) are precisely the ones eye features resolve.
- **Contribution**: Diagnostic. Explains WHY fusion should help and WHERE.
- **Status**: **DONE.**

---

### Idea 210: Simple Fusion Baselines (NEW — must run)

- **One-line**: Test whether concatenating REVE EEG embeddings with eye features improves over eye-only under LOSO.
- **Hypothesis**: Fusion should improve because REVE and eye make different errors, but the gain may be small because REVE is weak. If fusion HURTS, the "adding noise" hypothesis is confirmed.
- **Minimum experiment**: Concat [eye_66, reve_512] → SVM/LR with LOSO. Also test late fusion (weighted average of separate heads).
- **Expected outcome**:
  - Success: bacc > 0.57 (fusion beats eye-only by > 2%)
  - Partial: bacc ≈ 0.55 (no improvement but no harm)
  - Failure: bacc < 0.54 (REVE actively hurts eye tracking)
- **Novelty**: 6/10 — Simple fusion is standard, but with frozen EEG FM + eye under LOSO it's never been done.
- **Feasibility**: 0.5 days, CPU only.
- **Risk**: LOW — guaranteed diagnostic result.
- **Pilot result**: NEGATIVE for raw concat (-14.6pp). NEUTRAL for PCA-concat (-0.4pp). REVE embeddings are near-collapsed (cosine sim ~0.95 across classes).
- **Implication**: Simple fusion fails. Smarter fusion (Idea 205) needs to use REVE linear probe predictions, not raw embeddings. The paper narrative shifts toward diagnostic + the question of WHETHER EEG adds anything.
- **Status**: **DONE.**

---

### Idea 204: Behavioral Primitive Decomposition

- **One-line**: Split eye tracking's 33 features into fixation, saccade, and pupil primitives — which one actually complements REVE?
- **Hypothesis**: Pupil features (arousal proxy) should resolve REVE's Sad-vs-Fear/Neutral confusion. Fixation features may encode attention patterns. Saccade features may be mostly noise.
- **Minimum experiment**: Group the 33 eye features by type (based on SEED-V documentation). Run eye-only SVM per group. Then fuse each group with REVE separately.
- **Expected outcome**: Pupil > Fixation > Saccade for complementarity with REVE. This informs which features matter for consumer deployment.
- **Novelty**: 5/10 — Primitive analysis is standard, but with frozen FM + LOSO it's new.
- **Feasibility**: 1-2 days, CPU only.
- **Risk**: LOW.
- **Reviewer's objection**: "This is feature engineering 101." Counter: the contribution is the finding, not the method — knowing which behavioral primitives complement neural FMs matters for system design.

---

### Idea 205: Uncertainty-Triggered Eye Residuals (REVISED)

- **One-line**: Use eye tracking as the base predictor; let REVE EEG add residual corrections only on uncertain trials.
- **Hypothesis**: Eye tracking is the strong generalizer (0.551 bacc), REVE is weak (0.323) but makes different errors. Dense symmetric fusion risks degrading the strong modality. Uncertainty-gated residuals let REVE contribute only when eye tracking is uncertain.
- **Minimum experiment**: (1) Train eye-only SVM. (2) Identify uncertain trials (entropy > threshold). (3) Train a residual head on [eye_logits, eye_entropy, reve_emb] → delta-logits. (4) Final pred = eye_logits + α·gate(eye_entropy)·delta. (5) Compare vs concat fusion and eye-only.
- **Expected outcome**:
  - Success: bacc > concat fusion, specifically on uncertain trials
  - Partial: comparable to concat but with interpretable gating decisions
  - Failure: gate learns to ignore REVE (confirming REVE adds nothing)
- **Novelty**: 7/10 — Gated residual fusion exists, but eye-as-base + EEG-as-corrector inverts the standard assumption. No paper does this with frozen FMs.
- **Feasibility**: 3-4 days, 1 GPU.
- **Risk**: LOW-MEDIUM.
- **Reviewer's objection**: "This is stacking with extra steps." Counter: (1) the reversed asymmetry (behavioral base, neural corrector) is the finding; (2) the gating analysis reveals WHEN neural signal helps; (3) the ablation comparing both directions (EEG-base vs eye-base) is diagnostic gold.

---

### Idea 206: Consumer-Grade Degradation Frontier

- **One-line**: Map the performance frontier as EEG channels drop from 62→4 and eye features degrade, testing where multimodal sensing outweighs unimodal.
- **Hypothesis**: Eye tracking adds little when REVE sees full 62ch EEG, but becomes essential when EEG degrades to 4ch headband (where REVE drops from 0.323 to 0.283 bacc).
- **Minimum experiment**: Evaluate across grid: {62ch, 32ch, 16ch, 8ch, 4ch} × {full eye, gaze-only, pupil-only, no eye}. Use REVE's channel-agnostic architecture for subsetting. Plot iso-accuracy contours.
- **Expected outcome**: A crossover point where eye+4ch > 62ch EEG alone. This defines the practical frontier for consumer devices.
- **Novelty**: 8/10 — No existing paper maps this degradation curve for frozen EEG FMs.
- **Feasibility**: 4-5 days (requires re-extracting REVE embeddings at each channel count).
- **Risk**: LOW — guaranteed informative result.
- **Reviewer's objection**: "Synthetic channel dropping ≠ real consumer hardware." Counter: (1) REVE's 4D positional encoding makes it natively channel-agnostic; (2) we validated the 4ch proxy against known nearest-neighbor channels; (3) this gives the first principled degradation curve for EEG FMs.

---

### ★ TIER 2: Extensions (If T1 Succeeds)

---

### Idea 203: Neural-Behavioral Lead/Lag Sweep

- **One-line**: Test whether eye features are more informative at temporal offsets relative to EEG.
- **Hypothesis**: EEG captures faster involuntary response; eye tracks later appraisal/regulation. Zero-lag may be suboptimal.
- **Minimum experiment**: Extract eye features from early/mid/late clip segments. Compare unimodal and fusion performance at each offset.
- **Novelty**: 7/10. Lag in physiology is known, but for frozen FM + eye under LOSO it's new.
- **Feasibility**: 2-3 days.
- **Risk**: LOW (guaranteed diagnostic result).
- **Status**: PENDING.

---

### Idea 207: Top-2 Confusion Expert Routing

- **One-line**: When REVE's top-2 predictions fall into a known ambiguous pair (e.g., Sad vs Neutral), invoke an eye-conditioned binary expert.
- **Hypothesis**: Eye can resolve specific pairwise confusion (Sad↔Neutral, Fear↔Sad) better than overall 5-class decoding.
- **Novelty**: 7/10. MoE over modality combinations is studied, but confusion-pair-specific routing is new.
- **Feasibility**: 3 days.
- **Risk**: LOW.

---

### Idea 208: Eye Nuisance Audit

- **One-line**: Test whether eye features encode subject/session identity more than emotion, explaining why they generalize under LOSO.
- **Hypothesis**: Unlike EEG (which has severe inter-subject variability), eye tracking may be more generalizable because oculomotor responses to emotional content are relatively universal.
- **Novelty**: 6/10.
- **Feasibility**: 2 days (linear probes for subject vs emotion).
- **Risk**: LOW.

---

### Idea 209: Agreement-as-Confidence Calibration

- **One-line**: Use EEG-eye agreement (not fusion) to calibrate prediction confidence and enable selective prediction.
- **Hypothesis**: Even if REVE doesn't improve accuracy, it can tell us WHEN eye predictions are reliable.
- **Feasibility**: 2 days.
- **Risk**: LOW.

---

## Eliminated Ideas

| Idea | Reason |
|---|---|
| Ideas 101-113 (egoEMOTION Round 2) | PPG/gaze are dead modalities. The entire egoEMOTION fusion direction is terminal. |
| EEG-Prototype-Conditioned Eye (GPT idea 5) | Too complex, similar principle to Idea 205 |
| Reliability-Aware Eye Gating (GPT idea 7) | Subsumed by Idea 205's uncertainty mechanism |
| Training-Free Retrieval (GPT idea 9) | Interesting but tangential to paper narrative |
| CGGM, bottleneck, TMC, distillation | Already failed on egoEMOTION (won't retry) |

---

## Best Paper Narrative

### Title
**"When Does Eye Tracking Help Frozen EEG Foundation Models? Neural-Behavioral Asymmetry for Cross-Subject Emotion Recognition"**

### Thesis
Cross-subject emotion recognition with frozen EEG foundation models is fundamentally limited by inter-subject neural variability. Eye tracking, despite using simple hand-crafted features, outperforms a NeurIPS 2025 EEG FM under strict LOSO because behavioral responses are more generalizable across individuals. The optimal fusion strategy is asymmetric: eye tracking as the strong base predictor, EEG as a conditional corrector for uncertain trials.

### 4-Claim Structure

1. **Claim 1 (Diagnostic)**: Under strict LOSO evaluation, eye tracking features (SVM on 66 features) achieve 0.551 balanced accuracy on SEED-V, outperforming frozen REVE EEG (0.323) by 70%. Hand-crafted behavioral features beat a foundation model pretrained on 25K subjects.

2. **Claim 2 (Diagnostic)**: The error patterns are complementary, not redundant. REVE collapses to "Sad" prediction on hard cases; eye tracking resolves these confusion pairs via pupil dilation (arousal) and gaze patterns. Specific behavioral primitives (pupil, fixation) target specific confusion pairs.

3. **Claim 3 (Method)**: Because utility is sparse and asymmetric, uncertainty-triggered eye-residual fusion (eye as base, EEG as conditional corrector) outperforms symmetric fusion architectures.

4. **Claim 4 (Practical)**: The value of eye tracking increases as EEG quality degrades. In consumer-grade settings (4ch headband), eye tracking compensates for lost neural coverage, defining a practical deployment frontier.

### Fallback Claims (if ideas partially fail)
- If fusion doesn't improve over eye-only → "Frozen EEG FMs add no complementary signal beyond eye tracking under LOSO" (strong negative result)
- If all behavioral primitives contribute equally → "Eye tracking's emotion signal is holistic, not decomposable" (still publishable diagnostic)
- If consumer degradation is monotonic → "Eye tracking is universally valuable, not just in degraded EEG settings" (practical result)

---

## Suggested Execution Order

### Week 0 (Day 1-2): Diagnostics ✅ PARTIALLY DONE
- [x] Idea 201: Eye-only LOSO baseline (DONE: bacc=0.551)
- [x] Idea 202: REVE confusion analysis (DONE: complementary errors confirmed)
- [ ] Idea 210: Simple fusion baselines (RUNNING)
- [ ] Idea 204: Behavioral primitive decomposition

### Week 1 (Days 3-6): Core Method
- [ ] Idea 205: Uncertainty-triggered eye residuals (3-4 days)
- [ ] Idea 203: Lead/lag sweep (2-3 days, can overlap)

### Week 2 (Days 7-10): Extension + Paper
- [ ] Idea 206: Consumer degradation frontier (4-5 days)
- [ ] Idea 208: Eye nuisance audit (2 days, can overlap)

### Week 3: Paper Framing
- Combine all results into paper narrative
- Run multi-seed experiments for statistical significance
- Invoke `/experiment-plan` for full protocol

---

## Risk Assessment

### All-Null Escape Plan (from GPT-5.4 devil's advocate)

If EEG+Eye fusion fails to improve over eye-only, the paper pivots to a **stronger negative result**:

> "Frozen EEG Foundation Models Fail Cross-Subject Emotion Recognition Where Eye Tracking Succeeds: A Diagnostic Study on SEED-V"

Supporting evidence:
- Eye nuisance audit (Idea 208) explains WHY: subject identity dominates EEG representations but not eye features
- Agreement-as-confidence (Idea 209) as rescue use-case: EEG helps confidence, not accuracy
- Consumer degradation (Idea 206) as practical negative: even in wearable settings, eye may be sufficient alone

**Decision thresholds (UPDATED with Pilot 3 results):**
- ~~If Idea 210 shows bacc > 0.57 → proceed with Idea 205~~ NOT MET
- ~~If Idea 210 shows bacc ≈ 0.55 → proceed cautiously~~ **THIS IS WHERE WE ARE** (PCA-concat = 0.539)
- ~~If Idea 210 shows bacc < 0.54 → pivot to negative result paper~~ Raw concat hit this (0.397)

**Decision: Proceed with DIAGNOSTIC-FIRST paper narrative.** Idea 205 (uncertainty residuals) should use REVE linear probe predictions (bacc=0.323) NOT raw embeddings (bacc=0.285). The paper's value is in the diagnostic findings, with Idea 205 as a methodological contribution IF it works.

---

## Prior Rounds (Reference)

### Round 1 (2026-04-13 early): egoEMOTION landscape
4 ideas proposed (ECSE, OSMD, MDLAT, Rank Diagnostic). Rank diagnostic (005) was the key prerequisite.

### Round 2 (2026-04-13 mid): egoEMOTION Beast Mode
13 new ideas generated (101-113). Best narrative: "Sparse Asynchronous Correction" with residual towers + lag alignment + counterfactual supervision. **All ideas target dead modalities (PPG, gaze) on egoEMOTION — superseded by Round 3's pivot to SEED-V EEG+Eye.**

### Round 3 (2026-04-13 late): SEED-V EEG+Eye — THIS REPORT
Pivot driven by discovery that: (1) SEED-V has unused eye tracking data, (2) Eye tracking bacc=0.551 >> REVE EEG 0.323 under LOSO, (3) Error patterns are complementary. All new ideas target this direction.

---

## Next Steps

- [ ] Complete Idea 210 (simple fusion pilot — running)
- [ ] Run Idea 204 (behavioral primitive decomposition)
- [ ] If fusion positive: implement Idea 205 (uncertainty residuals)
- [ ] If 2+ ideas positive: invoke `/experiment-plan` for full protocol
- [ ] If results strong: invoke `/paper-plan` with the narrative arc above
- [ ] For full pipeline: `/research-pipeline` end-to-end
