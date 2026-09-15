---
type: idea
node_id: idea:101
title: "Video-Residual Correction Towers"
stage: proposed
outcome: unknown
based_on: [paper:han2024_fusemoe, paper:chen2025_modality_collapse]
target_gaps: [G2, G8, G9]
risk: LOW
effort: 3-4 days
created_at: 2026-04-13
---

# Make video the base predictor; train EEG/gaze/PPG residual heads to correct video's mistakes.

## Hypothesis
Extra modalities have sparse utility concentrated on video's error cases. Symmetric fusion dilutes this sparse signal. Residual heads that only fire when they can correct video extract maximum value from weak modalities.

## Proposed Method
1. Freeze trained video-only classifier (F1=0.646)
2. Train residual head: [video_logits, video_entropy, eeg_emb, gaze_emb, ppg_emb] → delta-logits
3. Final prediction = video_logits + alpha * delta
4. Evaluate overall F1/CCC AND F1 on video-error subset

## Expected Outcome
- Success: F1 > 0.655 AND improvement on video-error cases
- Partial: Overall F1 unchanged but error-case F1 improves → sparse utility confirmed
- Failure: Near-zero corrections → modalities genuinely carry no complementary info

## Failure Notes
(to be filled after pilot)

## Reusable Components
- Asymmetric fusion module (base + corrector pattern)
- Per-sample error-case analysis pipeline
