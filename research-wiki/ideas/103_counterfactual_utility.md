---
type: idea
node_id: idea:103
title: "Counterfactual Marginal-Utility Supervision"
stage: proposed
outcome: unknown
based_on: [paper:chen2025_modality_collapse, paper:raina2025_i2moe]
target_gaps: [G2, G5, G8]
risk: MEDIUM
effort: 4-5 days
created_at: 2026-04-13
---

# Supervise fusion weights with each modality's leave-one-out delta-loss rather than confidence.

## Hypothesis
Confidence gating weights modalities by certainty, but certainty ≠ utility. Counterfactual supervision teaches fusion to weight by actual causal contribution.

## Proposed Method
1. Precompute per-sample Δ_m = L(without_m) - L(with_all) from reference model
2. Train utility predictor g(emb_m) → predicted Δ_m
3. Use predicted utility as fusion weights: w_m = softmax(g(emb_m)/τ)
4. Compare: uniform, confidence, oracle Δ, learned Δ

## Expected Outcome
- Success: Learned utility weights improve F1 beyond residual towers
- Partial: Oracle Δ shows signal but learned predictor too noisy
- Failure: No modality ever has positive Δ → definitively closes "better fusion" hypothesis

## Failure Notes
(to be filled after pilot)

## Reusable Components
- Counterfactual utility computation pipeline
- Modality contribution analysis framework
