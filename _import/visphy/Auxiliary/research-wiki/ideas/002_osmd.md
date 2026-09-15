---
type: idea
node_id: idea:002
title: "Orthogonal Subspace Modality Disentanglement (OSMD)"
stage: proposed
outcome: unknown
based_on: [paper:chen2025_modality_collapse]
target_gaps: [G8, G2]
risk: MEDIUM
effort: 3-4 days
created_at: 2026-04-13
---

# Enforce orthogonal per-modality subspaces to prevent modality collapse and resolve the F1/CCC tradeoff.

## Hypothesis
The F1/CCC tradeoff and TMC collapse are symptoms of modality collapse: rank bottleneck in the fusion head allows noisy PPG/gaze features to entangle with predictive video features. Orthogonality regularization prevents this.

## Proposed Method
- Add L_orth = sum_{i!=j} ||P_i^T P_j||_F^2 to loss
- P_i = projection matrix for modality i
- Sweep lambda in {0, 0.01, 0.1, 1.0}
- Measure effective rank per modality per emotion class

## Expected Outcome
- Success: F1 AND CCC both improve (tradeoff resolved)
- Partial: One improves, other stays same
- Failure: No change → problem isn't collapse but genuine lack of complementary info

## Failure Notes
(to be filled after pilot)
