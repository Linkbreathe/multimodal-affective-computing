# Research Wiki Index

> Auto-generated categorical index. Last updated: 2026-04-13 (26 papers, 16 ideas)

## Papers

### EEG Foundation Models
- [REVE](papers/elouahidi2025_reve.md) — Universal EEG FM with 4D positional encoding, NeurIPS 2025
- [EEG-FM-Bench](papers/xiong2026_eeg_fm_bench.md) — Unified benchmark for 7 EEG-FMs, 2026

### General Multimodal Fusion (Top Venues)
- [HEALNet](papers/hemker2024_healnet.md) — Shared latent bottleneck + iterative cross-attention, NeurIPS 2024
- [FuseMoE](papers/han2024_fusemoe.md) — MoE with Laplace gating for fleximodal fusion, NeurIPS 2024
- [CyIN](papers/lin2025_cyin.md) — Cyclic information bottleneck for incomplete multimodal, NeurIPS 2025
- [Modality Collapse](papers/chen2025_modality_collapse.md) — Root cause analysis + fix via rank/distillation, ICML 2025
- [I²MoE](papers/raina2025_i2moe.md) — Interpretable interaction-aware MoE fusion, ICML 2025
- [Contrastive Modality Dropout](papers/miccai2025_contrastive_modality_dropout.md) — Learnable tokens + contrastive for missing modality, MICCAI 2025

### Multimodal Physiological Foundation Models
- [PhysioOmni](papers/fu2025_physioomni.md) — Decoupled tokenizer for multi-physio FM with missing modality, 2025

### Physiological Signal Fusion
- [EmotionTFN](papers/emotiontfn2025.md) — Multi-scale temporal fusion for EEG+PPG+GSR, Sensors 2025
- [Cross-Attention Physio](papers/crossattn_physio2025.md) — Dual-branch cross-attention with missing-modality robustness, 2025
- [Contrastive ECG-EEG](papers/contrastive_ecg_eeg2025.md) — Contrastive alignment of ECG with EEG features, 2025

### EEG + Video/Face Fusion
- [EA-FUSION](papers/ea_fusion2026.md) — Quality-aware EEG+face fusion with CMEF, BSPC 2026
- [EEG-Visual Mixed Attention](papers/eeg_visual_mixed_attn2025.md) — Mixed attention for EEG-visual alignment, 2025

### EEG + Eye Movement Fusion
- [MambaMER](papers/ping2025_mambamer.md) — EEG-guided Mamba for asymmetric EEG+eye fusion, MICCAI 2025
- [TEREE](papers/esmi2025_teree.md) — Triple EEG representation + Bayesian spurious correlation minimization, Intelligence-Based Medicine 2025
- [CMGNN](papers/fu2024_cmgnn.md) — Cross-modal guiding neural network, EEG guides eye features, IEEE JBHI 2024
- [Cross-Modal Alignment](papers/crossmodal_align2025.md) — PCA-RFE + temporal cross-modal attention, arXiv 2025
- [MHESA](papers/mhesa2024.md) — Homogeneous encoding space alignment for EEG+eye, Expert Systems with Applications 2024
- [MACDB](papers/macdb2025.md) — Multi-level alignment + consistent decision boundaries for cross-subject, Knowledge-Based Systems 2025

### Multi-Modal (3+) Fusion
- [EEG-Audio-Video](papers/eeg_audio_video2024.md) — Three-modal transformer fusion, arXiv 2024
- [Simple Fusion](papers/ghallab2025_simplefusion.md) — Foundation model embeddings + concatenation achieves near-SOTA, NeurIPS 2025 Workshop

### Datasets
- [SEED-VII](papers/jiang2025_seedvii.md) — 7-emotion EEG+eye dataset with continuous labels + MAET transformer, IEEE TAC 2025

### Portable / Consumer EEG
- [Portable 4ch](papers/portable4ch2025.md) — $35 4-channel EEG with self-supervised learning, 60.2% accuracy, MDPI Mathematics 2025
- [NeuroGaze](papers/neurogaze2025.md) — Consumer Muse S2 + webcam gaze BCI for VR, 2025

### Brain-Video Decoding
- [EEG2Video](papers/liu2024_eeg2video.md) — Decoding visual perception from EEG, NeurIPS 2024

## Ideas

### Round 1 — Frozen Encoder Alignment (EgoEmotion)
- [001: ECSE](ideas/001_ecse.md) — Supervised contrastive subspace extraction for frozen FM embeddings (proposed)
- [002: OSMD](ideas/002_osmd.md) — Orthogonal subspace modality disentanglement to prevent collapse (proposed)
- [003: MDLAT](ideas/003_mdlat.md) — Modality dropout + learnable absence tokens + confidence gating (proposed)
- [005: Rank Diagnostic](ideas/005_rank_diagnostic.md) — SVD analysis of where emotion lives in frozen embeddings (proposed)

### Round 2 — Sparse Asynchronous Correction (EgoEmotion)
- [101: Residual Towers](ideas/101_residual_towers.md) — Video-residual correction towers, video as base predictor (proposed)
- [102: Lag Alignment](ideas/102_lag_alignment.md) — Lag-aware physiological temporal alignment (proposed)
- [103: Counterfactual Utility](ideas/103_counterfactual_utility.md) — Leave-one-out delta-loss as fusion weight supervision (proposed)
- [104: Pareto Decoder](ideas/104_pareto_decoder.md) — Dual-head Pareto decoder for F1/CCC tradeoff diagnosis (proposed)

### Round 3 — SEED-V EEG+Eye Fusion
- [201: Eye-Only Baseline](ideas/201_eye_only_baseline.md) — Eye-only LOSO baseline, bacc=0.551 (piloted, positive)
- [202: Confusion Utility](ideas/202_confusion_utility.md) — Confusion-pair utility map, REVE Sad-attractor + eye complementarity (piloted, positive)
- [204: Primitive Decomposition](ideas/204_primitive_decomposition.md) — Behavioral primitive decomposition of eye tracking (proposed)
- [205: Uncertainty Residuals](ideas/205_uncertainty_residuals.md) — Uncertainty-triggered eye residuals, eye base + EEG corrector (proposed) **CORE METHOD**
- [206: Consumer Degradation](ideas/206_consumer_degradation.md) — Consumer-grade degradation frontier, 62ch->4ch + eye grid (proposed)
- [210: Simple Fusion](ideas/210_simple_fusion.md) — Simple EEG+eye concat fusion baseline (piloted, pending)

## Experiments

_No experiments yet._

## Claims

_No claims yet._
