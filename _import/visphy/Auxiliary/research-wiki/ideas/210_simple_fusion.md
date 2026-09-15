---
type: idea
node_id: idea:210
title: "Simple EEG+Eye Concatenation Fusion Baseline"
stage: piloted
outcome: pending
based_on: [paper:ghallab2025_simplefusion]
target_gaps: [G10, G12]
risk: LOW
effort: 0.5 days
created_at: 2026-04-13
---

# Concatenate EEG and eye embeddings as the simplest fusion baseline.

## Hypothesis
Following Ghallab et al. (2025), simple concatenation of foundation model embeddings may be surprisingly competitive. This establishes the floor that any complex fusion method must beat.

## Proposed Method
1. Concatenate REVE EEG embeddings + eye features (or eye embeddings)
2. Train SVM / linear classifier on concatenated representation
3. LOSO evaluation on SEED-V

## Expected Outcome
- Success: Concat fusion > both unimodal baselines → confirms complementarity
- Failure: Concat fusion <= best unimodal → complementarity is not linearly accessible

## Pilot Results
(pending — experiment queued)

## Reusable Components
- Concat fusion pipeline for SEED-V
- Baseline comparison framework
