# Query Pack

> Compressed summary for /idea-creator. Auto-generated, max 8000 chars. Updated: 2026-04-13 (round 3)

**Project direction:** Quad-modal emotion recognition (EEG+video+audio+PPG) fusing egocentric video with physiological signals using frozen pretrained foundation encoders (EEGPT, VideoMAE, emotion2vec, PaPaGei/Pulse-PPG) and learnable fusion heads on SEED-V.

**Top gaps (ranked):**
- G2: Frozen encoder representation alignment — no work aligns heterogeneous frozen FM embeddings
- G9: Egocentric video + physiological fusion — completely unexplored
- G6: Dimensional CCC with frozen encoders — most works report accuracy, not CCC
- G8: Modality collapse — rank bottleneck entangles noisy/predictive features (ICML'25)
- G5: Quad-modal FM fusion — HEALNet/FuseMoE handle N-modal but not frozen FMs
- G7: Missing modality robustness — 5 recent papers address it, but not for 4 frozen FMs

**Paper clusters:**
1. *Shared-Latent Bottleneck Fusion* (HEALNet NeurIPS'24): Iterative cross-attention updates a shared latent array per modality. Handles missing modalities by skipping layers. Most architecturally promising for our 4-modal frozen setup.
2. *MoE-Based Fusion* (FuseMoE NeurIPS'24, I²MoE ICML'25): Experts specialize in modality combinations. Laplace gating prevents collapse. Fleximodal handles arbitrary missing modalities. I²MoE adds interpretable interaction types.
3. *Information-Theoretic Fusion* (CyIN NeurIPS'25): Cyclic IB purifies task-relevant features and reconstructs missing modalities. Elegant theory but computational cost.
4. *Physio Foundation Models* (PhysioOmni 2025): Decoupled shared/private codebooks for modality-invariant/specific features. SOTA on emotion+sleep+motor with missing modalities.
5. *Modality Collapse Analysis* (ICML'25): Rank bottleneck in fusion head causes collapse. Fix: cross-modal distillation or basis reallocation. May explain our TMC failure.
6. *Contrastive Alignment* (ECG-EEG'25, MICCAI'25): Align representations across modalities; learnable tokens for missing modalities.

**Failed ideas (anti-repetition):**
- CGGM gradient modulation: useless with frozen encoders
- Bottleneck fusion: destroys features (but Pulse-PPG > PaPaGei by +0.064 CCC_V)
- TMC evidential fusion: Dirichlet KL causes evidence collapse, CCC=0.000 — likely modality collapse (G8)
- Soft-target distillation: cosine~0 between modalities, no transfer possible
- Enriched projections: best CCC_V=0.717 but trades off F1 — structural tradeoff, possibly rank bottleneck (G8)

**Top papers (26):** HEALNet (NeurIPS'24), FuseMoE (NeurIPS'24), CyIN (NeurIPS'25), Modality-Collapse (ICML'25), I²MoE (ICML'25), PhysioOmni (2025), MICCAI-Contrastive-Dropout (2025), REVE (NeurIPS'25), EEG-FM-Bench (2026), EA-FUSION (BSPC'26), EmotionTFN (Sensors'25), CrossAttn-Physio (2025), Contrastive-ECG-EEG (2025), EEG-Visual-MixedAttn (2025), EEG-Audio-Video (2024), EEG2Video (NeurIPS'24), MambaMER (MICCAI'25), TEREE (Intelligence-Based Medicine'25), CMGNN (IEEE JBHI'24), CrossModal-Align (arXiv'25), MHESA (Expert Systems'24), MACDB (KBS'25), Simple-Fusion (NeurIPS'25 WS), SEED-VII (IEEE TAC'25), Portable-4ch (MDPI'25), NeuroGaze (2025)

**Active chains:**
- HEALNet bottleneck → adapt for frozen encoder outputs → shared latent captures cross-modal emotion → test CCC
- FuseMoE routing → experts per modality-combination → handle missing/noisy channels → test F1+CCC
- Modality collapse analysis → diagnose our TMC/enriched failures → apply basis reallocation → fix F1/CCC tradeoff
- PhysioOmni codebook → shared emotional codebook + private per-modality → lightweight layer on frozen embeddings
- Contrastive alignment → align frozen embeddings in shared space → then apply HEALNet/FuseMoE → two-stage fusion

**Proposed ideas — Round 1 (4):**
- idea:001 ECSE — SupCon projection to extract emotion subspace from frozen embeddings (LOW risk, 1 week)
- idea:002 OSMD — Orthogonal subspace regularization to prevent modality collapse (MEDIUM risk, 3-4 days)
- idea:003 MDLAT — Modality dropout + learnable tokens + confidence gating (LOW risk, 1 week)
- idea:005 Rank Diagnostic — SVD analysis of emotion info in frozen embeddings (LOW risk, 1 day, DO FIRST)

**Proposed ideas — Round 2 Beast Mode (13 new, top 4 shown):**
- idea:101 Video-Residual Correction Towers — asymmetric fusion, video as base, others as conditional correctors (LOW risk, 3-4 days) ★ T1
- idea:102 Lag-Aware Physiological Alignment — temporal offset sweep for PPG/gaze relative to video (LOW risk, 2-3 days) ★ T1
- idea:103 Counterfactual Marginal-Utility Supervision — leave-one-out delta-loss as fusion weight supervision (MEDIUM risk, 4-5 days) ★ T1
- idea:104 Dual-Head Pareto Decoder — separate F1/CCC decoders to diagnose tradeoff (LOW risk, 3-4 days) ★ T2
- Also: 105 cross-dataset EEG anchor, 106 PID-sparse routing, 107 rank-budget reallocation, 108 disagreement-as-signal, 109 transition-target, 110 batch remixing, 111 sensor-on-demand, 112 EEG-queried video, 113 subject calibration

**Piloted ideas — Round 3 SEED-V EEG+Eye (6 new):**
- idea:201 Eye-Only LOSO Baseline — bacc=0.551 (SVM, 66 features), establishes eye floor (piloted, POSITIVE)
- idea:202 Confusion-Pair Utility Map — REVE Sad-attractor + eye resolves top 3 confusions, confirms complementarity (piloted, POSITIVE)
- idea:210 Simple EEG+Eye Concat Fusion — Ghallab-style concat baseline (piloted, PENDING)
- idea:204 Behavioral Primitive Decomposition — fixation/saccade/pupil decomposition for modality-aware fusion (proposed, LOW risk, 2 days)
- idea:205 Uncertainty-Triggered Eye Residuals — eye base + EEG corrector gated on eye entropy (proposed, LOW-MEDIUM risk, 3-4 days) **★ CORE METHOD**
- idea:206 Consumer-Grade Degradation Frontier — 62ch→4ch + eye degradation grid (proposed, LOW risk, 4-5 days)

**Best paper narratives:**
1. "Sparse Asynchronous Correction" (EgoEmotion) — Ideas 005 + 102 + 101 + 103. Thesis: frozen fusion fails because it assumes dense, synchronous, symmetric modality contribution.
2. "Uncertainty-Gated Asymmetric Fusion" (SEED-V) — Ideas 201 + 202 + 205 + 206. Thesis: eye tracking provides stable base; EEG adds value only at decision boundaries where eye uncertainty is high. Consumer-grade viability via degradation frontier.

**Open unknowns:**
- Does PPG recover signal at non-zero temporal offsets? (102)
- Do residual correction towers extract sparse utility from video's error cases? (101)
- Does counterfactual supervision beat confidence gating? (103)
- Is the F1/CCC tradeoff in the representation or the readout head? (104)
- Does cross-modal disagreement (EEG vs video) encode meaningful mixed emotion? (108)
- Does simple concat beat both unimodal baselines on SEED-V? (210)
- Does uncertainty gating outperform symmetric EEG+eye fusion? (205)
- What is the minimum viable EEG channel count when compensated by eye tracking? (206)
- Do behavioral primitives (fixation/saccade/pupil) carry distinct emotion information? (204)
