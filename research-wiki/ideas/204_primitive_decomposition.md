---
type: idea
node_id: idea:204
title: "Behavioral Primitive Decomposition"
stage: proposed
outcome: unknown
based_on: [paper:fu2024_cmgnn]
target_gaps: [G10, G11]
risk: LOW
effort: 2 days
created_at: 2026-04-13
---

# Decompose eye tracking into behavioral primitives (fixation, saccade, pupil) for modality-aware fusion.

## Hypothesis
Raw eye features conflate distinct behavioral signals (fixation patterns, saccade dynamics, pupil dilation) that carry different emotion information. Decomposing into primitives allows the fusion model to weight each behavioral channel independently, similar to how CMGNN uses EEG to guide eye features.

## Proposed Method
1. Segment eye tracking into three primitive streams: fixation, saccade, pupil
2. Extract per-primitive feature sets
3. Fuse primitives with EEG using per-primitive attention weights
4. Compare against monolithic 66-feature eye baseline (idea:201)

## Expected Outcome
- Success: Primitive-aware fusion > monolithic eye fusion → decomposition captures structure
- Failure: No improvement → primitives are redundant at this granularity

## Failure Notes
(to be filled after pilot)

## Reusable Components
- Behavioral primitive segmentation pipeline
- Per-primitive feature extraction module
