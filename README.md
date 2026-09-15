# Multimodal Affective Computing

An end-to-end research codebase for studying affective and relaxation-related state from
physiology, gaze, head motion and first-person video. The active thesis workflow covers
recorded-session ingestion, signal alignment and quality control, 10-second windows,
handcrafted and pretrained representations, multimodal fusion, participant-independent
evaluation, recorded replay and non-interventional Shadow inference.

The repository consolidates the reusable implementation of
`real-time-vis-physio-fusion` and `Relax-Model` into one installable Python package,
`mac`. The code is organised by pipeline stage rather than by source repository.
EgoEmotion, SEED-V, the retired Adaptive Control service and merge-history material remain
available under `Auxiliary/`, but they are not part of the thesis-facing execution path.

> This repository implements research workflows; it does not by itself establish that a
> model generalises to unseen people, that a learned representation measures an internal
> affective state, or that an adaptive intervention benefits a participant. Claims must be
> tied to the dataset, cohort, target, split contract and evaluation protocol that produced
> them.

[中文说明](README_zh.md) · [Thesis scope](docs/thesis-scope.md) ·
[Code and protocol map](docs/code-map.md) ·
[Input contracts](data/contracts/input_tables.md) ·
[Output contract](data/contracts/outputs.md) ·
[Unity Shadow protocol](integrations/unity/PROTOCOL.md) ·
[Auxiliary material](Auxiliary/README.md)

## 1. Purpose and research questions

No single modality gives a complete or consistently reliable view of affective state.
Physiology carries autonomic and neural measurements, gaze and head motion describe
behaviour, and first-person video provides context. They also differ in sampling rate,
latency, missingness, noise and model requirements. This project keeps those differences
explicit instead of forcing every modality through one preprocessing recipe.

The RELAX thesis workflow addresses five connected questions:

| Research question | Implementation in this repository | Main evidence produced |
| --- | --- | --- |
| How do signals change across experimental conditions? | Alignment, quality coverage, handcrafted features, condition aggregation | QC tables, descriptive statistics, figures and condition summaries |
| Which modalities contribute useful information? | Single-modality, ablation and matched-mask multimodal comparisons | Fold-level metrics, out-of-fold predictions and paired comparisons |
| Do pretrained representations transfer to this task? | EEGPT, REVE, NeuroRVQ, ECGFounder, VideoMAE v2, PaPaGei and related wrappers | Frozen embeddings, projection/compression studies, fine-tuning and LoRA results |
| Does performance generalise to unseen participants? | LOPO/LOSO splits, explicit manifests and participant-level grouping | Per-fold predictions, aggregate metrics and uncertainty estimates |
| Can estimates support a safe runtime workflow? | Chronological recorded replay and Shadow-only inference | Timestamped state predictions, recommendations, safety-gate decisions and logs |

The primary unit of analysis is the **participant-condition**, not an independently sampled
10-second segment. Windows inherit condition labels for feature construction, but correlated
windows from the same participant-condition must not be split across training and test data.

## 2. End-to-end data flow

```text
Raw recordings + questionnaire tables + marker streams + Unity/video logs
        |
        v
Session indexing and timestamp/marker alignment                 mac.data
        |
        v
Complete 10 s windows, channel contracts and quality coverage   mac.preprocessing
        |
        +--> handcrafted EEG/ECG/eye/head/video features         mac.features
        |
        +--> frozen or adapted pretrained representations        mac.encoders
        |
        v
Single-modality models and multimodal fusion                     mac.models / mac.fusion
        |
        v
Participant-independent training and evaluation                 mac.training / mac.evaluation
        |
        +--> metrics, predictions, manifests and reports          mac.reporting
        |
        +--> chronological recorded replay                        mac.adaptive.offline
        |
        +--> non-interventional Shadow runtime                    mac.realtime / Unity bridge
```

Raw recordings and questionnaire exports are external inputs and are not committed. Model
weights, large embedding caches and most generated artifacts are also external or ignored.
A fresh clone supports source inspection and data-independent checks; reproducing a result
requires the corresponding data, weights, configuration, manifests and commit.

## 3. Repository structure

```text
src/mac/                 Active reusable package
scripts/                 Thesis experiment runners, cache builders and audits
analysis/                Statistical analyses, supplementary comparisons and figures
configs/                 RELAX runtime, experiment and fusion configuration
data/contracts/          Versioned input, label, feature and output contracts
integrations/unity/      Shadow UDP protocol and Unity C# bridge
tests/                   Active package and thesis-workflow tests
docs/                    Current scope, architecture and protocol documentation
artifacts/               Models/caches/results when present; mostly generated or external
reports/                 Human-readable report outputs when present
weights/                 Local pretrained weights; not installed automatically
Auxiliary/
  benchmarks/            EgoEmotion and SEED-V comparison material
  adaptive_control/      Retired experimental closed-loop service and dedicated tests
  merge_history/         Old path maps, branch history, validation records and legacy snapshots
  research*/ plan/       Historical research notes and planning material
```

The boundary is intentional: active thesis code may import `mac`, while active code must
not depend on an auxiliary benchmark runner, archived report or retired control service.
Auxiliary experiments may reuse public `mac` components.

### 3.1 The `mac` package

| Module | What it does | Typical inputs | Typical outputs |
| --- | --- | --- | --- |
| `mac.data` | Session indexing, XDF/CSV/XLSX I/O, label parsing, timestamp alignment, video indexes, windows and condition aggregation | Raw session folders, marker streams, questionnaire tables | Source manifests, aligned boundaries, window tables |
| `mac.preprocessing` | Marker/window preprocessing, MNE QC and modality-specific ECG/PPG/RELAX/SEED-V signal preparation | Raw or aligned signals plus sampling/channel contracts | Filtered/resampled arrays, coverage and QC records |
| `mac.features` | Handcrafted EEG/ECG, eye, head and visual descriptors; dynamic texture; official VideoMAE2 extraction | Complete windows, video frames or retained MP4 | Window-level feature tables and visual embeddings |
| `mac.encoders` | Wrappers and registries for pretrained physiological and video encoders | Tensors following model-specific sampling and shape contracts | Frozen or trainable embeddings |
| `mac.fusion` | Early/mid/late fusion, bottleneck, HEALNet, Perceiver IO, Q-Former, TMC, CGGM, MM-Lego, distillation and compression | Per-modality embeddings and availability masks | Joint representations or predictions |
| `mac.models` | Classical condition models, temporal 1D-CNN, visual models, EEG heads, LoRA and feature-group validation | Feature tables or encoded sequences | Estimators, checkpoints and predictions |
| `mac.tasks` | Task heads, losses, relaxation regression and condition-control targets | Fused representations and labels | Losses and task predictions |
| `mac.training` | Condition/state/policy/video training, LOSO trainer and early stopping | Config, split manifests, features or embeddings | Model bundles, fold outputs and training summaries |
| `mac.evaluation` | LOPO contracts, alignment validation, metrics, paired comparisons and safety gates | Held-out predictions and labels | Metrics, uncertainty and deployment/hold decisions |
| `mac.experiments` | Research-only orchestration separated from runtime training | Locked experiment configuration | Comparison matrices and audit artifacts |
| `mac.adaptive.offline` | Frozen-prefix inference and chronological recorded replay; no live intervention | Recorded sessions and frozen models | Replay decisions and temporal metrics |
| `mac.realtime` | Ten-second clock, buffers, inference engine, replay, serve and Shadow recommendation policy | Live or replayed signal windows | State/recommendation messages and JSONL/Parquet logs |
| `mac.reporting` | Run summaries, model/video reports and result registry | Manifests, predictions and metrics | Human-readable reports and indexed result metadata |
| `mac.config` | Layered RELAX configuration and the separate flat fusion-YAML loader | Versioned YAML plus untracked local paths | Validated `ProjectConfig` or flat experiment dictionaries |
| `mac.utils` | Atomic writes, hashing, logging and TensorBoard helpers | Files and runtime metadata | Reproducible writes, hashes and logs |

`src/real_time_ml/` and `src/src/` are compatibility shims for pre-merge imports. New code
must import `mac`. The historical mapping is retained only in
[`Auxiliary/merge_history/`](Auxiliary/merge_history/README.md).

### 3.2 Modalities and active contracts

| Modality | Active RELAX contract | Processing/output |
| --- | --- | --- |
| EEG | Raw 500 Hz columns `[M2, TP9, TP10, M1]`; M1/M2 are references | Linked-mastoid or configured reference, TP9/TP10 signal features, 1-45 Hz feature bands, QC and coverage gating |
| ECG | Bipolar signal from configured columns `[7, 8]` | Filtering, peak/RR/HR measurements, quality checks and longer-history HRV when enough history exists |
| Eye | Timestamped gaze vectors/events | Gaze direction, velocity, fixation/saccade and availability summaries |
| Head | Unity HMD pose and motion | Position/orientation/movement descriptors and coverage |
| Video | Timestamped Unity frame index and optional retained MP4 | Handcrafted visual features, 16-frame VideoMAE2 clips, optional research descriptors |
| PPG | Reusable encoder/preprocessing support, primarily auxiliary benchmarks | PaPaGei/Pulse-PPG-compatible preprocessing and embeddings |

Model-specific representation pipelines may resample or normalise a signal differently from
the real-time handcrafted path. Those pipelines are alternatives with explicit consumers;
they are not interchangeable preprocessing aliases.

## 4. Protocols that must remain separate

Several paths use similar names but answer different questions:

| Protocol | Unit/cohort | Cache or input contract | Entry points |
| --- | --- | --- | --- |
| RELAX condition-level | 15 participants x 9 conditions = 135 participant-condition observations | Versioned raw/label contracts, complete non-overlapping 10 s windows | `mac index`, `preprocess`, `extract-features`, `train-state`, `evaluate` |
| Windows RQ2 | Locked FMQ-9 handoff: 9 participants, 81 condition labels, 567 source windows, 545 common-valid windows | Explicit shared-root completion and common-valid masks | `scripts/run_rq2_wsl.py`, `run_rq2_modality_ablation.py` |
| RELAX foundation sample cache | Samples contain participant, condition, labels, modality windows and masks | Foundation Dataset contract | `scripts/relax_foundation/` |
| RELAX aligned cache | Parallel participant-condition arrays plus explicit split/label/window/mask manifests | Aligned condition-embedding contract | `scripts/build_relax_alignment_cache.py`, `run_relax_foundation_probe.py` |
| EgoEmotion / SEED-V | Dataset-specific labels, splits and sampling units | Independent benchmark caches | `Auxiliary/benchmarks/` |

Do not exchange caches between the foundation and aligned runners, compare metrics from
different cohorts as if they shared a denominator, or treat repeated condition labels across
windows as independent observations. See [the code map](docs/code-map.md) for the detailed
entry-point boundary.

## 5. Installation

### 5.1 Classical/runtime environment

Python 3.11 is required. The most reproducible starting point is the checked-in Conda
environment:

```powershell
git clone https://github.com/Linkbreathe/multimodal-affective-computing.git
Set-Location multimodal-affective-computing
git switch thesis-core-auxiliary

conda env create -f environment.yml
conda activate mac
mac --help
```

For an existing Python 3.11 environment:

```powershell
python -m pip install -e ".[dev]"
```

The core package intentionally excludes PyTorch so indexing, handcrafted features,
classical models, CLI discovery and source compilation can be used without a GPU stack.
The complete test collection imports neural modules during collection and therefore
requires the `dl` extra even when slow/external tests are filtered out.

### 5.2 Deep-learning and representation environment

Install the PyTorch build appropriate for the host and CUDA version first, then install the
research extras:

```powershell
python -m pip install -e ".[dl,viz,ecg,head]"
```

| Extra | Adds |
| --- | --- |
| `dl` | PyTorch ecosystem, transformers, timm, einops, safetensors and HDF5 support |
| `viz` | matplotlib, seaborn, tqdm and TensorBoard |
| `ecg` | NeuroKit2 ECG analysis |
| `head` | AHRS utilities |
| `dev` | pytest, coverage and Ruff |

The EgoEmotion VideoMAE environment at
`Auxiliary/benchmarks/egoemotion/environment-videomae2.yml` is intentionally separate: it
pins `timm` 0.4.12, whereas the shared encoder stack uses `timm` 1.x.

### 5.3 External resources

These are not downloaded by the package:

- RELAX recordings, marker streams, Unity eye/head/video logs and questionnaire tables;
- pretrained checkpoints for EEGPT, REVE, NeuroRVQ, ECGFounder and VideoMAE variants;
- sibling checkouts `papagei-foundation-model/` and `pulseppg/` when those encoders are used;
- the external VideoMAE2 repository/checkpoint configured under `features.video.videomae2`;
- the Unity project containing the matching Shadow bridge scene and configuration.

Record checkpoint hashes and external repository commits in every reproducible run.

## 6. Reproducing the thesis workflow

Reproduction is staged because source-only verification, classical RELAX experiments,
pretrained representation experiments and Unity replay have different prerequisites.

### Step 1 — Record the software identity

```powershell
git status --short --branch
git rev-parse HEAD
python --version
python -m pip freeze | Out-File artifacts-environment.txt
```

Use a clean checkout at a recorded commit. Do not rely only on a branch name, because branch
heads can move.

### Step 2 — Run source-only verification

```powershell
python -m compileall -q src scripts analysis tests Auxiliary/benchmarks Auxiliary/adaptive_control
python -m mac --help
git diff --check
```

After installing the `dl` development stack, run the data-independent test selection:

```powershell
python -m pytest -m "not external and not integration and not slow"
```

`external` requires weights, external model code or participant data; `integration` reads
participant sources; `slow` trains models or performs expensive processing. Passing this
selection does not reproduce paper metrics.

### Step 3 — Configure local data roots

Never edit versioned experiment YAML with workstation paths. Create the ignored local layer:

```powershell
Copy-Item configs/local.example.yaml configs/local.yaml
```

Edit only `configs/local.yaml`:

```yaml
paths:
  raw_root: "D:/path/to/relax-recordings"
  labels_root: "D:/path/to/questionnaire-tables"

hardware:
  dcnn_device: cuda
```

Validate source columns, labels and expected outputs against:

- [`data/contracts/input_tables.md`](data/contracts/input_tables.md)
- [`data/contracts/labels.md`](data/contracts/labels.md)
- [`data/contracts/features.md`](data/contracts/features.md)
- [`data/contracts/outputs.md`](data/contracts/outputs.md)

### Step 4 — Reproduce the condition-level classical baseline

Use explicit stages so each artifact can be inspected and hashed:

```powershell
$runArgs = @(
  "--experiment", "configs/experiments/runtime-classical.yaml",
  "--local-config", "configs/local.yaml"
)

mac @runArgs index
mac @runArgs preprocess
mac @runArgs extract-features --no-video
mac @runArgs train-state
mac @runArgs evaluate
mac @runArgs report
```

The stages produce source/manifests, aligned windows, feature tables, model bundles,
out-of-fold predictions, metrics and a final report under the configured artifact/run
directories. The human-readable summary follows
`reports/<run_id>_summary_zh.md`; machine outputs follow the directory contract documented
in `data/contracts/outputs.md`.

Before accepting a result, verify:

- every outer fold holds out a participant, not independently sampled windows;
- preprocessing and feature selection are fitted only on permitted training data;
- the participant list, EEG-disabled list, target transform and condition order match the
  declared protocol;
- each result records config, seed, split manifest, input hashes and software commit;
- safety/deployment gates report `hold` unless their declared criteria are satisfied.

### Step 5 — Add video processing when required

```powershell
mac @runArgs build-video-mp4
mac @runArgs extract-handcrafted-video
mac @runArgs extract-videomae2
mac @runArgs train-video-ml
mac @runArgs report-video-fusion
```

The configured VideoMAE2 path uses retained MP4, 16 frames per window and a separately
managed external repository/checkpoint. Review the script and local paths first, then run
`powershell -File scripts/setup_videomae2.ps1` on Windows if setup is needed.

### Step 6 — Reproduce aligned pretrained-representation experiments

First build or obtain an aligned cache using the same encoder checkpoints, preprocessing,
cohort and masks. Then run the manifest-locked probe, for example:

```bash
python scripts/run_relax_foundation_probe.py \
  --embedding-cache /path/to/aligned/condition_embeddings.pt \
  --cohorts /path/to/cohorts.json --cohort all_135 \
  --split-manifest /path/to/splits.csv --mask-manifest /path/to/masks.csv \
  --labels /path/to/labels.csv --windows /path/to/windows.csv \
  --modalities eeg ecg eye head video --fusion healnet \
  --seed 20260705 --device cuda --require-cuda --strict \
  --output-dir artifacts/relax/my_aligned_run
```

Use `scripts/build_relax_alignment_cache.py --help` for cache construction options. The
earlier runner under `scripts/relax_foundation/run_relax_foundation_probe.py` consumes the
foundation sample cache and must not be pointed at an aligned cache.

### Step 7 — Reproduce recorded replay or Shadow inference

```powershell
mac @runArgs replay --output artifacts/realtime/replay.jsonl
mac @runArgs serve --max-cycles 1
```

The production-style `serve` path expects the configured LSL physiology stream and Unity
UDP messages. Default transport is Unity to Python at `127.0.0.1:5055`, and Python to Unity
at `127.0.0.1:5056`. Keep `policy.shadow: true`; this repository does not present the active
thesis runtime as a validated closed-loop intervention.

See the [wire protocol](integrations/unity/PROTOCOL.md) and
[C# bridge](integrations/unity/RtmlShadowUdpBridge.cs). The retired experimental Adaptive
Control service is isolated under `Auxiliary/adaptive_control/` and is reproduced separately.

## 7. Configuration model

The active RELAX workflow uses layered configuration:

```text
configs/project.yaml                 Legacy complete operational defaults
        |
configs/base.yaml                    Thesis protocol and versioned safe defaults
        |
configs/experiments/<experiment>.yaml
        |
configs/local.yaml                   Untracked paths and hardware only
```

Global options such as `--experiment` and `--local-config` must appear before the CLI
subcommand. Fusion runners also retain a separate flat-YAML loader in
`mac.config.simple`; it is deliberately not re-exported as `mac.config.load_config`.

Shared fusion templates live in `configs/fusion/`. EgoEmotion and SEED-V configurations
live with their auxiliary benchmarks rather than in the active `configs/` root.

## 8. Reproducibility record

Every reported experiment should retain the following information:

| Category | Required record |
| --- | --- |
| Code | Commit hash, clean/dirty state, runner and complete command |
| Environment | Python, package lock/export, OS, CPU/GPU and CUDA versions |
| Data | Dataset version, participant/cohort list, exclusions, input hashes and target definition |
| Protocol | Unit of analysis, window policy, preprocessing, masks and cache schema |
| Models | Encoder architecture, external repository commit, checkpoint hash and trainable/frozen layers |
| Evaluation | Outer/inner split manifests, seeds, metrics, per-fold predictions and aggregation rule |
| Outputs | Run ID, config snapshot, manifests, logs, model cards, metrics and report path |

Scores are comparable only when target, cohort, valid-window mask, split policy and metric
aggregation match. A high window count does not increase the number of independent
participant-condition labels.

## 9. Auxiliary and historical material

- `Auxiliary/benchmarks/egoemotion/` and `Auxiliary/benchmarks/seedv/` preserve independent
  benchmark runners, configurations, tests and historical reports.
- `Auxiliary/adaptive_control/` preserves the retired Unity/UDP control experiment outside
  the active `mac` CLI.
- `Auxiliary/merge_history/` preserves old-to-new path maps, branch/consolidation records
  and original README/environment snapshots. These documents describe historical states
  and are not current reproduction instructions.
- `Auxiliary/research-wiki/`, `Auxiliary/research/` and `Auxiliary/plan/` contain research
  notes and planning material rather than runtime inputs.

## 10. Development rules

- Put reusable implementations in the appropriate `src/mac/` pipeline stage.
- Put experiment orchestration in `scripts/` or `analysis/`, and version parameters in
  `configs/`.
- Keep raw recordings read-only and workstation paths in ignored local configuration.
- Keep generated outputs out of source directories and attach manifests/hashes to runs.
- Preserve participant grouping in every split and fit data-dependent transforms on
  training data only.
- Keep research-only checkpoints out of runtime backend selection until they pass the
  declared evaluation and safety gates.
- Document sampling rate, units, channel order, expected tensor shape, checkpoint identity
  and missing-modality behaviour for every new encoder or feature family.
- Describe evidence at its actual level: implementation, offline evaluation, recorded
  replay, Shadow deployment or prospective participant study.
