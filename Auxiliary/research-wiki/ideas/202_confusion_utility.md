---
type: idea
node_id: idea:202
title: "Confusion-Pair Utility Map"
stage: piloted
outcome: positive
based_on: [paper:elouahidi2025_reve]
target_gaps: [G10, G11]
risk: LOW
effort: 0.5 days
created_at: 2026-04-13
---

# Map per-class confusion pairs to identify where each modality adds value.

## Hypothesis
REVE (EEG) and eye tracking have complementary error profiles. REVE exhibits a Sad-attractor pattern (over-predicts Sad), while eye tracking resolves different confusion pairs. A confusion-pair utility map reveals the fusion opportunity.

## Proposed Method
1. Build per-class confusion matrices for REVE-only and eye-only classifiers
2. Identify top-3 confusion pairs for each modality
3. Check if eye tracking resolves REVE's top confusions and vice versa

## Pilot Results
- REVE's Sad-attractor pattern confirmed: Sad is over-predicted, absorbing Neutral and Fear
- Eye tracking resolves the top 3 REVE confusion pairs
- Complementarity is asymmetric: EEG benefits more from eye correction than reverse
- Establishes clear fusion motivation

## Reusable Components
- Confusion-pair analysis pipeline
- Per-modality utility scoring framework
