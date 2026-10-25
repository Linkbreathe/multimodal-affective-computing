# Change history

## 2026-10-25 — Organize individual results and prepare GitHub delivery

- Added a Chinese reading index and a complete nine-participant narrative/table report under
  `docs/supplementary/participant_multimodal_20261025/`. Each participant retains all nine
  condition ratings, actual presentation order, eleven raw feature series, both outcome
  associations, paired-condition lists, and non-estimable/missingness reasons.
- Added a detailed P007/P009 comparison with all eleven feature correlations for both outcomes,
  condition-level examples, and a standalone thesis case paragraph. Explicitly marked this
  as an illustrative post hoc case, not representative sampling or an additional validation.
- Added `scripts/summarize_participant_exploration.py`, a standard-library-only renderer of
  frozen CSV results. It does not re-extract features, refit models, recalculate associations,
  select features, or create plots. Existing completed-run outputs and manifests are unchanged.
- At the author's request, explicitly version the seven core result tables and selected
  definitions/reports from the otherwise ignored artifact directory for GitHub delivery.
  The earlier general artifact-ignore policy remains intact; this is a curated exception.
- Do not stage raw `data/waterlilies`, questionnaire/thesis sources, video frames, extraction
  caches, beat/channel audits, local configuration, or absolute source-path inventories.
  Versioned condition tables contain pseudonymous participant-level measurements, not a claim
  of full anonymization. Historical verification metadata is not proof that a fresh clone can
  access the locally held raw inputs.
- Preserve the dedicated `feat/participant-multimodal-exploration` branch; publish with a normal
  push to the existing `origin`, without modifying the default branch or force-pushing.
- Verification: 16 targeted tests pass (13 analysis tests and 3 delivery/report tests).
  The generated report exactly matches the frozen CSV rendering, includes all 198 associations
  and 39 non-estimable entries, and the P007/P009 case's 44 correlation cells and 36 rating
  cells agree with the source tables. Ruff and whitespace checks pass. Raw-input integrity
  results from the original completed run remain preserved, not relabeled as a new extraction.

## 2026-10-25 — Post hoc participant-level multimodal analysis

Branch: `feat/participant-multimodal-exploration`.

### Purpose and protected inputs

This analysis was conducted after the main prediction analyses and is post hoc and exploratory.
It describes condition responses and feature–rating associations within the thesis's fixed
FMQ-9 cohort. It is not a retrospective justification for model, feature, or experiment selection.
No models are trained, no thesis text is edited or compiled, and no plots, image exports,
plotting scripts, or plotting dependencies are added.

The author approved the available thesis checkout at
`C:\Users\linki\amaster\msc-thesis-log` in place of the unavailable D: path.
The current cohort records, questionnaire, and event order agree for all 81 observations.
The data under `data/waterlilies` remain user-owned read-only inputs and are not staged in Git.

### Architectural decisions

- Added `mac.analysis.participant_exploration` as an isolated offline package, with separate
  computational specification, provenance/alignment, physiological extraction, behavioral
  extraction, optical flow, statistical summaries, reporting, and verification modules.
- Extracted `discover_sources(raw_root, participants)` from `mac.data.index.build_index`.
  Discovery is now reusable without writing manifests. The existing `build_index` API and
  its manifest-writing behavior remain compatible.
- Added a module CLI with `audit`, `extract`, `analyze`, `report`, `verify`, and `run` stages.
  Per-participant caches permit interrupted extraction to resume. Completed runs are protected
  by a completion marker; new analyses require a new output directory.
- Frozen feature rules and a configuration hash are persisted before association calculation.
  Source and consumed-frame hashes support provenance checks. Analysis never imports training
  or plotting workflows. Reporting emits UTF-8 text and CSV only.
- Kept generated research outputs under
  `artifacts/supplementary/participant_multimodal_20261025/`, outside existing model artifacts.
  Generated outputs follow the repository's existing artifact-ignore policy.

### Measurement decisions and corrections

- Retained exactly eleven core features, with per-feature availability. Signal windows are
  extraction inputs, never additional independent ratings. No common-five-modality exclusion
  is applied to this supplement.
- Preserved linked-mastoid TP9/TP10 and actual handcrafted detrending/causal 1–45 Hz filtering.
  The author explicitly chose to preserve the implementation despite the unused configured
  50 Hz notch. This discrepancy is documented, not silently corrected in the thesis.
- EEG powers are recomputed from jointly valid spans, with independent TP9/TP10 diagnostics,
  fixed band boundaries, equal channel aggregation, and natural-log analysis. No band ratios
  or correlation-driven channel selection are introduced.
- Recomputed time-weighted mean heart rate instead of the legacy reciprocal-median-RR feature.
  Used the existing NeuroKit cleaning/detector family, with explicit parameters and no peak
  insertion. RMSSD uses the fixed 5–65 s interval and original RR adjacency, excluding interrupted
  recordings. It is not an average of short-window RMSSD or a cross-condition rolling history.
- Eye velocity/proportion use elapsed-time weights and availability flags. Source provenance
  cannot distinguish eye tracking from head-forward fallback; the low-speed I-VT proportion is
  an operational proxy and can be constant at this logging rate.
- Head speed uses shortest quaternion rotations; dispersion is weighted RMS geodesic distance
  from the chordal mean rotation, avoiding Euler wrapping and quaternion-sign artifacts.
- Optical flow uses fixed 2 Hz sampling, 320×180 grayscale frames, and Farnebäck parameters.
  Mean and population temporal SD use pixels/actual second with explicit gap rules.
- Retained P003's documented P001 identity alias and existing ISO-video-timestamp fallback;
  explicitly recorded P011's backup XDF and P015's clock residual warnings. Counter rollover
  is distinguished from missing physiological samples.
- The RTX 5080 was detected, but the installed OpenCV build has no CUDA Farnebäck support.
  Extraction uses up to four CPU processes; no algorithm is changed merely to use the GPU.

### Statistics and delivery

- Exported the seven requested tables, a complete core-table data dictionary, variable units,
  source/alignment/channel/beat/frame audits, frozen parameters, execution instructions,
  Chinese report, and a separate Chinese thesis-passage draft.
- Used within-person variable means, sample SD, raw/analysis-scale descriptions, and Spearman
  correlations with at least six valid paired conditions. Constant inputs remain non-estimable.
  Across-person correlation summaries are descriptive distributions, not population correlations.
- Retained all 198 feature–outcome–participant combinations: 159 estimable and 39 non-estimable.
  EEG has 78 valid conditions per band; RMSSD has 78. Other core features have 81 valid values.
  Missing EEG affects P004/C2, C6, C7; missing RMSSD affects P004/C8, P007/C1, P011/C2.
- Retained all weak/inconsistent results. Extreme raw observations are flagged for inspection
  without exclusion or recomputation. Existing model MAEs are read from thesis tables only.

### Verification

Mathematical tests cover counter rollover, gaps, RR adjacency, frequency power units,
quaternion wrapping/sign, elapsed-time weighting, constants, tied ranks, missing values,
timestamp fallback, joint-channel requirements, read-only discovery, and the existing index API.
Real-data verification checks table dimensions, independent SciPy correlations, centered sums,
direction counts, per-feature missingness, original-file hashes, consumed frame hashes, and
absence of graphical outputs. Final machine-readable results are in `verification.json`.

Unresolved measurement limits include absent gaze-source flags, no manual R-peak ground truth,
electrode identity relying on acquisition documentation rather than generic XDF labels alone,
residual EEG contamination, and only one session with at most nine conditions per person.
