---
type: idea
node_id: idea:201
title: "Eye-Only LOSO Baseline on SEED-V"
stage: piloted
outcome: positive
based_on: []
target_gaps: [G10, G11]
risk: LOW
effort: 0.5 days
created_at: 2026-04-13
---

# Eye-only LOSO baseline to establish eye-tracking floor before fusion.

## Hypothesis
Eye tracking alone carries substantial emotion-discriminative signal on SEED-V. Establishing a clean LOSO baseline is necessary before any EEG+eye fusion claims are meaningful.

## Proposed Method
1. Extract 66 eye features (fixation, saccade, pupil statistics) per trial
2. SVM classifier with LOSO cross-validation on SEED-V (5 emotions)
3. Report balanced accuracy per fold and overall

## Pilot Results
- **bacc = 0.551** (SVM, 66 features, LOSO)
- Confirms eye tracking carries non-trivial signal above chance (0.20)
- Provides clean comparison anchor for fusion experiments

## Reusable Components
- Eye feature extraction pipeline (66 features)
- LOSO evaluation harness for SEED-V
