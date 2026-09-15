---
type: idea
node_id: idea:206
title: "Consumer-Grade Degradation Frontier"
stage: proposed
outcome: unknown
based_on: [paper:portable4ch2025, paper:neurogaze2025]
target_gaps: [G12]
risk: LOW
effort: 4-5 days
created_at: 2026-04-13
---

# Map the degradation frontier from research-grade to consumer-grade hardware (62ch->4ch EEG + eye tracking degradation grid).

## Hypothesis
Fusion with eye tracking may compensate for EEG channel reduction, maintaining acceptable accuracy with consumer hardware. The degradation frontier reveals the minimum viable hardware configuration.

## Proposed Method
1. Simulate channel reduction: 62ch -> 32ch -> 16ch -> 8ch -> 4ch EEG (channel subsets matching consumer layouts)
2. Simulate eye tracking degradation: full eye tracker -> webcam-grade (reduced temporal resolution, no pupil diameter)
3. Build 5x3 grid: [5 EEG configs] x [3 eye configs]
4. Evaluate each cell with best fusion method (from ideas 205/210)
5. Plot iso-accuracy contours to identify minimum viable configuration

## Expected Outcome
- Best case: 4ch EEG + full eye tracking matches 62ch EEG-only performance
- Expected: Graceful degradation curve with eye tracking compensating ~50% of EEG channel loss
- Worst case: Steep cliff at low channel counts → consumer hardware fundamentally insufficient

## Failure Notes
(to be filled after pilot)

## Reusable Components
- Channel subsetting simulation pipeline
- Eye degradation simulation module
- Hardware-accuracy frontier visualization
