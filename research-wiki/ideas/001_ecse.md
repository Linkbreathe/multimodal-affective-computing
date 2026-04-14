---
type: idea
node_id: idea:001
title: "Emotion-Supervised Contrastive Subspace Extraction (ECSE)"
stage: proposed
outcome: unknown
based_on: [paper:chen2025_modality_collapse, paper:hemker2024_healnet]
target_gaps: [G2, G8]
risk: LOW
effort: 1 week
created_at: 2026-04-13
---

# Replace linear per-modality projectors with SupCon-trained projection heads to concentrate emotion-relevant features from frozen FM embeddings before fusion.

## Hypothesis
Frozen FM embeddings contain emotion-discriminative information diluted across hundreds of general-purpose dimensions. A supervised contrastive projection head can concentrate this into a compact subspace, making downstream fusion more effective than with linear projection.

## Proposed Method
- Stage 1: Train per-modality 2-layer MLP projectors with SupCon loss (same-emotion samples cluster)
- Stage 2: Freeze projectors, train fusion head with task loss (CE + KL + CCC)
- Compare against baseline nn.Linear projection

## Expected Outcome
- Success: F1 > 0.660 (> +0.01 over best baseline)
- Failure: No improvement → frozen embeddings genuinely lack extractable emotion info
- Either result is publishable

## Failure Notes
(to be filled after pilot)

## Reusable Components
- SupCon projection module (modality-agnostic, reusable)
- Per-modality emotion discriminability metrics
