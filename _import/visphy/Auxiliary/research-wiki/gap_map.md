# Gap Map

> Field gaps with stable IDs. Referenced by papers, ideas, and claims.

## G1: Cross-Setup EEG Generalization
**Status:** Partially addressed (REVE)
EEG foundation models struggle with varying electrode configurations across datasets. REVE's 4D positional encoding addresses this but hasn't been validated in multimodal fusion contexts.

## G2: Frozen Encoder Representation Alignment
**Status:** Unresolved
No existing work studies how to align heterogeneous frozen foundation model embeddings (EEG, video, audio, PPG) for multimodal fusion. PhysioOmni's shared/private codebooks and contrastive ECG-EEG alignment are closest but operate on trainable encoders.

## G3: Quality-Aware Multimodal Fusion
**Status:** Partially addressed (EA-FUSION)
Signal quality varies across modalities (e.g., PPG motion artifacts, video occlusion), but most fusion methods treat all modalities equally. EA-FUSION addresses this for face video but not for physiological signals broadly.

## G4: Multi-Scale Temporal Dynamics in Cross-Modal Fusion
**Status:** Partially addressed (EmotionTFN)
Different modalities operate at different temporal scales (EEG: ms, PPG: seconds, video: frames). EmotionTFN uses hierarchical temporal attention but trains from scratch.

## G5: Quad-Modal Foundation Model Fusion
**Status:** Partially addressed (HEALNet, FuseMoE)
No published work fuses EEG + video + audio + PPG all using pretrained foundation model encoders. HEALNet's shared latent bottleneck and FuseMoE's MoE routing both handle N-modal fusion but were not designed for frozen FM embeddings specifically.

## G6: Dimensional Emotion Prediction with Frozen Encoders
**Status:** Unresolved
Most fusion works report classification accuracy, not valence/arousal CCC. Our project needs CCC optimization with frozen encoders — an underexplored combination.

## G7: Missing Modality Robustness
**Status:** Actively researched (HEALNet, FuseMoE, CyIN, PhysioOmni, MICCAI contrastive dropout)
Multiple recent top-venue papers address this. HEALNet skips layers, FuseMoE routes around, CyIN reconstructs via cyclic translation, MICCAI uses learnable tokens. All validated on 2-3 modalities; 4-modal case underexplored.

## G8: Modality Collapse in Multimodal Fusion
**Status:** Partially addressed (ICML 2025 analysis)
Root cause identified: rank bottleneck in fusion head entangles noisy and predictive features. Fixes: cross-modal distillation and basis reallocation. Not yet studied with frozen encoders.

## G9: Egocentric Video + Physiological Fusion
**Status:** Unresolved
No published work specifically fuses egocentric video (first-person perspective) with physiological signals for emotion recognition. Existing EEG+video works use third-person stimuli videos, not egocentric view.

## G10: Asymmetric Modality Guidance in Multi-Modal Fusion
**Status:** Partially addressed (MambaMER, CMGNN)
Multiple EEG+eye papers (MambaMER, CMGNN) show that asymmetric fusion -- where EEG guides secondary modality feature extraction -- outperforms symmetric fusion. However, this has only been demonstrated for 2-modal (EEG+eye) settings. No work extends asymmetric guidance to 3+ modalities or uses it with frozen foundation model encoders.

## G11: Evaluation Protocol Inconsistency in EEG+Eye Fusion
**Status:** Unresolved
Papers report widely varying accuracies on SEED-V (60%-98%) but use different evaluation protocols (within-subject vs. LOSO vs. multi-to-one transfer). MambaMER's 86.95% is likely within-subject; TEREE's 97.7% uses multi-to-one. Direct comparison is impossible without standardized protocols.

## G12: Simple Fusion Baseline with Foundation Models
**Status:** Partially addressed (Ghallab 2025)
Ghallab et al. show simple concatenation of FM embeddings achieves near-SOTA, but only for EEG+ECG. Whether simple fusion remains competitive with 4+ modalities including non-physiological signals (video, audio) is unknown. This establishes a critical baseline that complex fusion methods must beat.

## G13: Consumer-Grade to Research-Grade EEG Transfer
**Status:** Unresolved
Portable 4ch systems achieve 60.2% while research-grade systems exceed 85%. No published work demonstrates successful transfer or distillation from research-grade EEG models to consumer hardware (Muse S2, 4ch systems) for emotion recognition.
