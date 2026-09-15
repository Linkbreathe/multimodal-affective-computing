---
type: idea
node_id: idea:205
title: "Uncertainty-Triggered Eye Residuals"
stage: proposed
outcome: unknown
based_on: [paper:ping2025_mambamer, paper:fu2024_cmgnn]
target_gaps: [G10, G11, G12]
risk: LOW-MEDIUM
effort: 3-4 days
created_at: 2026-04-13
---

# Eye tracking as base predictor; EEG fires residual corrections only when eye uncertainty is high.

## Hypothesis
Eye tracking provides a stable, low-noise base signal for emotion recognition. EEG carries complementary information but is noisy and subject-variable. An uncertainty-gated residual architecture lets EEG correct eye predictions only when eye confidence is low, avoiding EEG noise injection on easy samples. This is the asymmetric analog of idea:101 (video-residual towers) adapted for the EEG+eye modality pair.

## Proposed Method
1. Train eye-only base classifier (from idea:201 baseline, bacc=0.551)
2. Compute eye prediction entropy as uncertainty signal
3. Train EEG residual head: [eye_logits, eye_entropy, eeg_embedding] -> delta_logits
4. Final prediction = eye_logits + gate(eye_entropy) * delta_logits
5. Gate = sigmoid(a * entropy + b), learned end-to-end
6. LOSO evaluation on SEED-V

## Expected Outcome
- Success: Fused bacc > 0.551 (eye-only) AND improvement concentrated on high-entropy samples
- Partial: Overall bacc unchanged but high-entropy subset improves → sparse utility confirmed
- Failure: Near-zero corrections → EEG adds no complementary info beyond eye tracking

## Failure Notes
(to be filled after pilot)

## Reusable Components
- Uncertainty-gated residual fusion module (generalizes to any base+corrector pair)
- Per-sample uncertainty analysis pipeline
- Entropy-based gating mechanism
