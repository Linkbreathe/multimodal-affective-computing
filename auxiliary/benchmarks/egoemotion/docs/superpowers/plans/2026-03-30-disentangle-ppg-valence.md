# PPG Valence Disentanglement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run four 28-fold LOSO ablations that isolate Pulse-PPG vs video contributions to valence CCC and determine whether the prior 3-modality enriched result was genuinely PPG-driven.

**Architecture:** Add four ablation YAMLs under `configs/ablation/` that switch modality enablement and projection type between linear late fusion and enriched distill-late projection. Execute the existing `scripts/run_experiment_10s.py` pipeline sequentially with task-aware cached embeddings, then extract aggregate `weighted_f1`, `ccc`, `ccc_valence`, `ccc_arousal`, and `ccc_dominance` from each generated report.

**Tech Stack:** YAML configs, Python experiment runner, Conda env `visphy`, task-aware embedding cache, Markdown reporting.

---

### Task 1: Add disentanglement ablation configs

**Files:**
- Create: `configs/ablation/ppg_only_linear.yaml`
- Create: `configs/ablation/ppg_only_enriched.yaml`
- Create: `configs/ablation/video_only_ref.yaml`
- Create: `configs/ablation/video_ppg_enriched.yaml`

- [ ] **Step 1: Add the four YAML configs**
- [ ] **Step 2: Verify each YAML deep-merges correctly with `configs/base.yaml`**
- [ ] **Step 3: Confirm `pulseppg_ppg` and `video_mae_v2` embeddings exist for all 28 subjects**

### Task 2: Execute sequential LOSO runs

**Files:**
- Read: `scripts/run_experiment_10s.py`
- Read: `reports/*.md`
- Read: `/tmp/disentangle_*.log`

- [ ] **Step 1: Run `ppg_only_linear` and capture its `/tmp` log**
- [ ] **Step 2: Parse aggregate `weighted_f1`, `ccc`, `ccc_valence`, `ccc_arousal`, `ccc_dominance` from the generated report**
- [ ] **Step 3: Repeat for `ppg_only_enriched`**
- [ ] **Step 4: Repeat for `video_only_ref`**
- [ ] **Step 5: Repeat for `video_ppg_enriched`**

### Task 3: Summarize and answer the scientific question

**Files:**
- Read: `reports/pipeline_report.md`
- Read: generated `reports/disentangle_*.md`

- [ ] **Step 1: Build a comparison table with modalities, projection type, and all requested metrics**
- [ ] **Step 2: Compare against the prior enriched 3-modality `CCC_Valence=0.717` claim**
- [ ] **Step 3: State whether the valence effect survives in PPG-only enriched form or is still video-dominated**
