---
type: idea
node_id: idea:005
title: "Effective Rank Diagnostic — Where Does Emotion Live in Frozen Embeddings?"
stage: proposed
outcome: unknown
based_on: [paper:chen2025_modality_collapse, paper:xiong2026_eeg_fm_bench]
target_gaps: [G2, G8]
risk: LOW
effort: 1 day
created_at: 2026-04-13
---

# Compute SVD-based effective rank of frozen embeddings per emotion class per modality to diagnose WHERE emotion information lives and WHY fusion fails.

## Hypothesis
PPG embeddings will show near-zero effective rank difference across emotion classes. Video embeddings will show high discriminability concentrated in a small subspace. Cross-modality alignment will be near-zero.

## Proposed Method
- SVD of [B_class x D] embedding matrix per emotion class per modality
- Compute: effective rank, explained variance ratio, Fisher discriminant ratio per dimension
- Cross-modality: principal angles between top-k subspaces
- No GPU needed — CPU SVD on cached .pt embeddings

## Expected Outcome
Quantitative evidence for/against each hypothesis about fusion failure. Informs which of Ideas 001-003 to pursue.

## Failure Notes
(to be filled after analysis)
