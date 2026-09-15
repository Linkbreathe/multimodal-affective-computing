---
type: idea
node_id: idea:102
title: "Lag-Aware Physiological Alignment"
stage: proposed
outcome: unknown
based_on: [paper:elouahidi2025_reve, paper:fu2025_physioomni]
target_gaps: [G4, G9]
risk: LOW
effort: 2-3 days
created_at: 2026-04-13
---

# Sweep/learn temporal offsets for PPG and gaze relative to video to test autonomic response lag hypothesis.

## Hypothesis
PPG autonomic response lags event-bearing video by 1-5 seconds. The 10s segments align all modalities to the same window, but PPG emotion response peaks seconds after the visual stimulus.

## Proposed Method
1. Extract PPG/gaze embeddings at offsets [-5s, -3s, -1s, 0s, +1s, +3s, +5s]
2. Evaluate unimodal PPG/gaze F1 at each offset
3. Use best-offset embeddings in residual fusion (Idea 101)
4. Optionally learn per-subject offsets

## Expected Outcome
- Success: PPG F1 improves at non-zero offset → temporal misalignment is root cause
- Failure: No offset helps → PPG truly carries no emotion signal

## Failure Notes
(to be filled after pilot)

## Reusable Components
- Temporal offset sweep pipeline
- Per-subject learned offset module
