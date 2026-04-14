---
type: idea
node_id: idea:003
title: "Modality Dropout with Learnable Absence Tokens + Confidence Gating"
stage: proposed
outcome: unknown
based_on: [paper:miccai2025_contrastive_modality_dropout, paper:ea_fusion2026, paper:han2024_fusemoe]
target_gaps: [G7, G3, G9]
risk: LOW
effort: 1 week
created_at: 2026-04-13
---

# Force the model to extract value from every modality via dropout training with learnable absence tokens and confidence-gated inference.

## Hypothesis
Video dominates because the fusion head learns to rely on it exclusively. Modality dropout forces the model to find complementary info in physio signals. Confidence gating ensures dynamic per-sample modality weighting at inference.

## Proposed Method
- Training: Drop each modality with p=0.3, replace with learnable embedding
- Add per-modality auxiliary classifiers for confidence estimation
- Inference: confidence-weighted fusion w_m = softmax(conf_m / tau)
- Ablation: (a) baseline, (b) dropout only, (c) dropout + tokens, (d) full

## Expected Outcome
- Success: PPG/gaze contribute when video is degraded; better robustness
- Failure: Even with dropout, physio signals unused → confirms signal-level failure

## Failure Notes
(to be filled after pilot)
