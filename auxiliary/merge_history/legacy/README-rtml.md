# Relax-Model

**Multimodal research tools for estimating relaxation and discomfort in virtual reality, comparing representations, and studying adaptive visual experiences.**

Relax-Model connects recorded physiological signals, eye tracking, head motion, and first-person video with participant-reported experience. It provides a Python pipeline for data preparation, feature extraction, participant-level model evaluation, recorded replay, and Unity integration.

This repository contains both research experiments and runtime implementations. Their presence demonstrates implemented workflows; it does not establish that a model generalizes to new users or that adaptive control improves relaxation.

[Chinese overview](README_zh.md) · [Input contracts](data/contracts/input_tables.md) · [Label contract](data/contracts/labels.md) · [Unity Shadow protocol](integrations/unity/PROTOCOL.md) · [Auxiliary research notes](Auxiliary/research/)

## Research purpose

The project investigates how an immersive system could respond to a person's experience using measurements collected during a VR session. The experimental setting uses nine visual conditions in a 3 × 3 stimulus grid, with questionnaire ratings associated with each participant and condition.

The code supports four related lines of investigation:

1. **Measurement and characterization:** describe how physiology, gaze, head movement, and visual context vary across conditions, including signal quality, missingness, and baseline effects.
2. **State estimation:** predict self-reported relaxation and discomfort, and test whether measured signals add information beyond condition identity and previous ratings.
3. **Representation and modality comparison:** compare handcrafted features, temporal neural encoders, visual representations, and modality ablations under explicit participant splits and shared data masks.
4. **Adaptive interaction:** examine how state estimates, uncertainty, signal availability, and control rules translate into recommendations or experimental Unity control commands.

Video and stimulus parameters describe what a participant sees. Predictive performance from these inputs alone is not sufficient evidence that a model measures the participant's internal state. Likewise, offline prediction, replay behavior, and a running controller are distinct from prospective evidence of benefit.

## Intended users

| User group | Main use |
| --- | --- |
| Affective computing and physiological computing researchers | Inspect signal processing, construct features, and evaluate associations with reported experience. |
| VR / XR and human–computer interaction researchers | Study the relationship between visual conditions, user experience, and adaptation policies. |
| Machine learning researchers | Compare classical models, temporal encoders, fusion methods, and missing-modality behavior without mixing participant splits. |
| Unity developers and experiment operators | Connect tracked sensors, verify model compatibility, inspect readiness, and run the appropriate inference or control service. |
| Research collaborators and reviewers | Trace an analysis from its data contract through model inputs, predictions, metrics, and provenance. |

Participants are the people whose sessions and questionnaire responses are analyzed. The software is operated by researchers and developers; it is not a participant-facing consumer application or a clinical assessment tool.

## Three execution paths

| Path | Entry points | Behavior and scope |
| --- | --- | --- |
| Offline research | `rtml` extraction / training / evaluation commands; `analysis/` scripts | Produces features, models, comparisons, and reports from recorded data. Several experiments require previously generated contracts or embeddings. |
| Shadow inference | `rtml replay`, `rtml serve` | Emits state predictions and recommendations for logging or display. The Shadow protocol requires `shadow=true`; the supplied Shadow bridge does not automatically apply condition changes. |
| Adaptive Control | `rtml adaptive-model`, `rtml adaptive-control` | Separate experimental service with a model registry, Unity control profile, readiness checks, session state, and control policy. It can issue control commands and must not be confused with the Shadow bridge. |

Research-only visual and fusion checkpoints are not automatically promoted to runtime backends. The Adaptive Control service has its own configuration and model compatibility checks; it does not inherit every policy restriction of the Shadow path.

## Data and supervision

### Main condition-level study

The main analysis contract covers **15 participants × 9 conditions = 135 participant–condition observations**. Signals are divided into complete, non-overlapping **10-second windows within each condition**. A condition change resets the window origin.

The supervised targets come from questionnaire ratings:

```text
relaxation = (rating - 1) / 6
discomfort = (rating - 1) / 6
```

Window-level features provide a trajectory within a condition. Repeating a condition rating over its windows does not create additional independent questionnaire observations. The main condition-level workflow aggregates windows before fitting models; temporal experiments preserve window sequences while retaining participant–condition targets.

Raw recordings and questionnaires are external inputs. They are not distributed with this repository, and generated models, caches, and most analysis outputs are ignored by Git. A fresh clone supports source inspection and data-independent tests; reproducing study results also requires the matching source data, configuration, and experiment artifacts.

### Windows RQ2 representation track

The dedicated [Windows RQ2 representation module](src/real_time_ml/windows_rq2_representations.py) validates a separate, locked **FMQ-9** contract: 9 participants, 81 condition labels, 567 source windows, and 545 common-valid windows. It preserves the `P004/C6` condition key even when no common-valid window is available, rather than silently removing that observation.

This track builds Windows-side representations and handoff metadata. `WINDOWS_DONE.json` marks completion of that track only. Downstream fusion, pretrained representations in a separate environment, and combined statistical conclusions are outside the completion claim of this marker. These cohort counts are contract expectations, not a report of a new run.

## Modalities and processing

| Modality | Source and processing | Main implementation |
| --- | --- | --- |
| EEG | Four raw electrode columns ordered `M2, TP9, TP10, M1`; configurable mastoid reference; signal features from TP9 / TP10; quality and coverage gating. | [physio.py](src/real_time_ml/features/physio.py) |
| ECG | Bipolar ECG, R-peak / RR measurements, short-window heart-rate features, and longer-history HRV when sufficient history is available. | [physio.py](src/real_time_ml/features/physio.py) |
| Eye tracking | Gaze direction and velocity-based fixation / saccade summaries. | [eye.py](src/real_time_ml/features/eye.py) |
| Head motion | Unity HMD pose and movement summaries. | [head.py](src/real_time_ml/features/head.py) |
| First-person video | Handcrafted visual features, frozen VideoMAE2 embeddings, and research-only dynamic texture descriptors. | [features/](src/real_time_ml/features/) |

Recorded physiology extraction and live services use `StreamingPhysioProcessor`. The default EEG configuration uses linked-mastoid referencing. The continuous MNE workflow is a separate audit / quality-control path, not the source of a second set of training features.

EEG availability is participant- and window-dependent. Inspect the configured exclusions and coverage thresholds before interpreting an EEG comparison. Some filtering options belong to offline analyses: a configured `notch_hz` value alone does not mean a notch filter is applied by the streaming processor.

## Models and evaluation

The classical condition-level workflow compares Ridge, ElasticNet, SVR, Random Forest, Extra Trees, and histogram gradient boosting. It models residuals around a condition-only baseline, with separate relaxation and discomfort targets and a high-discomfort classification component.

The repository also contains temporal 1D-CNN models, visual encoder comparisons, multimodal fusion experiments, and personalization / warm-start analyses. These are separate experimental choices, not interchangeable evidence for a single deployed model.

Evaluation design includes:

- **Leave-one-participant-out (LOPO) evaluation:** the test participant is held out from training.
- **Training-side model selection:** native workflows select features and models within the non-test participants; aligned experiments use their declared training / validation / test manifests.
- **Baseline comparisons:** condition-only and history-based predictions help determine whether additional sensors provide useful information.
- **Multiple outcomes:** prediction error, ranking / correlation, and high-discomfort recall, precision, and false negatives answer different questions.
- **Explicit modality and cohort definitions:** missing-modality comparisons must retain the intended label keys, splits, and availability masks.

The Shadow deployment gate can force `hold` when model evidence is insufficient. Missing inputs, poor quality, timing problems, and uncertainty also affect recommendations. Passing software tests or model compatibility checks does not establish prospective controller efficacy.

## Installation

The package targets **Python 3.11**. Run the following from the repository root in PowerShell:

```powershell
conda env create -f environment.yml
conda activate rtml-p002-p016
rtml --help
```

Alternatively, in an existing Python 3.11 environment:

```powershell
python -m pip install -e ".[dev]"
```

Optional dependency groups in [pyproject.toml](pyproject.toml) include `ecg`, `head`, `clip`, and `dcnn`. Neural experiments require a compatible PyTorch installation; GPU-specific experiments additionally require a working CUDA environment. The core feature / classical workflow and source-level tests do not require a complete visual encoder setup.

The separate [VideoMAE2 environment file](environment-videomae2.yml) documents the visual research environment. Model weights, raw data, and Unity project assets are not installed by the Python package.

## Configuration

New experiments use a layered configuration:

```text
configs/project.yaml          Legacy operational defaults
        ↓
configs/base.yaml             Shared experiment and protocol defaults
        ↓
configs/experiments/*.yaml    Run ID and experiment settings
        ↓
configs/local.yaml            Local data roots and device settings
```

Create the local file only if it does not already exist:

```powershell
if (-not (Test-Path configs/local.yaml)) {
    Copy-Item configs/local.example.yaml configs/local.yaml
}
```

Edit `paths.raw_root`, `paths.labels_root`, and device settings for your workstation. `configs/local.yaml` is ignored by Git. The legacy configuration and some historical research configurations contain workstation-specific paths; review them before reuse.

Use a distinct `run.id` for a new experiment. Normal layered runs place machine outputs under `artifacts/runs/<run_id>/`; some specialized research configurations explicitly select another output root. Global CLI options such as `--experiment` and `--local-config` come **before** the subcommand.

## Offline workflow

### 1. Prepare data and features

After configuring local paths, use the same run configuration throughout a workflow:

```powershell
$runArgs = @(
    "--experiment", "configs/experiments/runtime-classical.yaml",
    "--local-config", "configs/local.yaml"
)

rtml @runArgs index
rtml @runArgs preprocess
rtml @runArgs extract-features --no-video
```

`--no-video` skips video extraction for the physiological / behavioral workflow. For a small data smoke run, supported commands accept `--participants`; use a separate run ID for this subset, since stage tables can be overwritten and full-cohort training expects its complete label contract.

### 2. Train, evaluate, and summarize

```powershell
rtml @runArgs train-state
rtml @runArgs evaluate
rtml @runArgs report
```

The layered summary writer reads the selected run's normalized outputs and writes `reports/<run_id>_summary_zh.md`. Generated reports may still be in Chinese even though this project overview is in English.

For the legacy pipeline, `rtml --config <project-config.yaml> run-all` orchestrates its stages. Inspect the selected configuration and existing output locations first; `--force` requests recomputation instead of cached stage reuse.

### 3. Select an explicit research experiment

| Task | Command or source |
| --- | --- |
| Temporal condition model | `rtml train-dcnn-state` |
| Handcrafted video features / comparison | `rtml extract-handcrafted-video`, `rtml train-video-ml` |
| Frozen visual representations | `rtml extract-videomae2`, `rtml train-videomae2-dcnn` |
| Minimal fusion benchmarks | `rtml benchmark-minimal-fusion`, `rtml benchmark-minimal-fusion-dcnn` |
| Dynamic texture extraction | `rtml extract-dynamic-texture --help` |
| Shared-split and modality ablations | [analysis/supplementary/](analysis/supplementary/) |
| Windows RQ2 handoff | `python analysis/supplementary/run_windows_rq2.py --help` |

These entries are a source map, not a single sequential recipe. Supply the appropriate experiment configuration and required upstream artifacts. In particular, dynamic texture and aligned comparisons depend on declared contracts / masks; the RQ2 launcher requires its prepared shared root.

## Unity and live operation

### Shadow prediction and replay

```powershell
rtml replay --help
rtml serve --help
```

With matching data, models, and configuration available, `replay` processes recorded input and `serve` accepts live input. The default Shadow transport uses EEG / ECG over LSL and Unity messages over local UDP:

| Direction | Default endpoint |
| --- | --- |
| Unity → Python | `127.0.0.1:5055` |
| Python → Unity | `127.0.0.1:5056` |

The Shadow protocol emits a `StatePrediction` and a `ConditionRecommendation` for each cycle, including schema, timestamps, cycle index, and window bounds. See the [protocol](integrations/unity/PROTOCOL.md) and [C# Shadow bridge](integrations/unity/RtmlShadowUdpBridge.cs).

### Experimental Adaptive Control

```powershell
rtml adaptive-model list
rtml adaptive-model verify --bundle <registered-bundle-id>
rtml adaptive-control --help
```

[configs/adaptive-control.yaml](configs/adaptive-control.yaml) defines the model registry, Unity profile path, networking, and policy settings. The checked-in profile path refers to an external Unity project and must match the operator's installation. Use a compatible registered bundle and the corresponding Unity control integration.

The local [launcher](launch-adaptive-control.cmd) starts the configured service. In the matching Unity project, operators use the Adaptive Control configuration / control panel and readiness checks before starting a session. The service tracks physiology and head / eye availability and writes status and decision logs.

Shadow and Adaptive Control use the same default UDP ports: run the intended service for the session, or configure separate ports. A working connection or readiness result verifies operational prerequisites; it is not a validation of model accuracy or human benefit.

## Repository map

```text
configs/                         Shared, legacy, experiment, and control settings
data/contracts/                  Input, label, feature, and output contracts
src/real_time_ml/
  README.md                      Package namespace and module map
  cli.py                         rtml command entry point
  data/                          Source indexing and table / video I/O
  preprocessing/                 Boundaries, windows, and audit processing
  features/                      Physiological, behavioral, and visual features
  modeling/                      Classical, neural, and fusion models
  evaluation/                    Metrics and alignment contracts
  experiments/                   Experiment-specific orchestration
  realtime/                      Shadow clock, buffers, engine, and service
  adaptive_control/              Separate experimental control runtime
  reporting/                     Run summaries
  windows_rq2_representations.py Windows representation handoff
analysis/
  adaptive_offline/              Offline measurement and control simulations
  decision_reanalysis/           Recorded decision reanalysis
  idiographic/                   Participant-specific analyses
  phase_baseline/                Phase / baseline analysis
  supplementary/                Comparisons, ablations, figures, and reports
integrations/unity/              Shadow bridge and message protocol
tests/                           Unit, integration, and slow tests
  Auxiliary/                       Research plans and historical review notes
artifacts/                       Local generated outputs (mostly ignored)
```

## Outputs and reproducibility

Depending on the workflow, generated outputs include source manifests and hashes, condition labels and boundaries, window tables, features, model checkpoints, out-of-fold predictions, metrics, and runtime logs. See the [output contract](data/contracts/outputs.md).

For a reproducible comparison, retain the exact configuration, run ID, cohort keys, split manifest, feature / representation definition, model seed, software environment, and input hashes. An aggregate score without these does not identify the experiment that produced it.

Historical reports and plans describe particular analysis snapshots. Consult the corresponding run artifacts before reusing numerical findings. This README makes no claim that a fresh clone includes all assets needed to reproduce every historical report.

## Tests

Run the data-independent test selection:

```powershell
python -m pytest -m "not integration and not slow"
```

Tests cover data normalization, label handling, streaming features, condition-level data, temporal models, shared splits, dynamic texture contracts, RQ2 validation, message schemas, and control behavior.

With the required source data and dependencies available, additional selections are:

```powershell
python -m pytest -m integration
python -m pytest -m slow
```

Integration tests read study data; slow tests may train models or perform more expensive work. Passing these checks supports implementation correctness within their coverage, not physiological validity, generalization, or causal efficacy.

## Contributing and extending

- Keep source recordings read-only and generated outputs outside versioned source directories.
- Preserve participant–condition supervision and the declared cohort; make exclusions explicit.
- Fit preprocessing, feature selection, and tuning only on the permitted training / validation data.
- Add a distinct configuration and run ID for a new comparison.
- Keep research-only representations separate from runtime model selection.
- Update relevant contracts and tests when changing features, splits, or Unity messages.
- Describe evidence at its actual level: implementation, offline evaluation, recorded replay, or prospective participant study.
