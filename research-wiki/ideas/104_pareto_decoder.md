---
type: idea
node_id: idea:104
title: "Dual-Head Pareto Decoder for F1/CCC"
stage: proposed
outcome: unknown
based_on: [paper:chen2025_modality_collapse]
target_gaps: [G6, G8]
risk: LOW
effort: 3-4 days
created_at: 2026-04-13
---

# Separate task-specific decoders for F1 and CCC to diagnose whether the tradeoff is in representation or readout.

## Hypothesis
F1/CCC tradeoff is a conflicting-objective problem, not a representation problem. Separate decoders will reveal whether the Pareto frontier improves.

## Proposed Method
1. Shared frozen features → two heads (class-F1, V/A-CCC)
2. Independent loss weighting per head
3. Trace Pareto frontier vs. single multi-task head

## Expected Outcome
- Pareto frontier expands → confirms objective conflict, not representation limit
- No change → confirms representation is the bottleneck
