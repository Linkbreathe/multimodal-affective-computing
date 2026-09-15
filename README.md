# Multimodal Affective Computing

**One pipeline from recorded sensors to a running adaptive system: session ingest, feature and representation extraction, multimodal fusion, participant-independent evaluation, recorded replay, and live inference.**

This repository is the merger of two previously separate codebases that were already
exchanging data through the filesystem and machine-specific paths:

| Merged from | Called in the code | Supplies |
| --- | --- | --- |
| `real-time-vis-physio-fusion` | Project B | Pretrained encoders, 16 fusion architectures, LOSO evaluation over EgoEmotion, SEED-V and RELAX |
| `Relax-Model` | Project A | Session indexing, handcrafted features, classical and temporal models, Shadow inference, Adaptive Control, Unity integration |

They are now one installable package, `mac`, organised by **what each module does**
rather than by which project it came from. Both Git histories are preserved; see
[the merge map](docs/MERGE-MAP.md) to translate any path or command written before the merge.

The presence of an implemented workflow demonstrates that it runs. It does not establish
that a model generalises to new participants, that a representation measures an internal
state, or that adaptive control improves anything for a person.

[中文总览](README_zh.md) · [合并对照表](docs/MERGE-MAP.md) · [代码地图](docs/code-map.md) · [输入契约](data/contracts/input_tables.md) · [Unity Shadow 协议](integrations/unity/PROTOCOL.md)

## Research purpose

Emotion and relaxation are hard to infer from any single sensor. Video carries visual
context, gaze and head motion describe behaviour, and PPG, ECG and EEG measure physiology.
These sources differ in timing, noise, availability and representation requirements.

The merged codebase supports five related lines of investigation:

1. **Measurement and characterisation** — how physiology, gaze, head movement and visual
   context vary across conditions, including signal quality, missingness and baseline effects.
2. **Modality contribution** — which signals carry usable information, and whether combining
   them beats single modalities under one evaluation protocol.
3. **Representation transfer** — how well frozen pretrained representations transfer to
   affective targets, and when projection, compression, fine-tuning or LoRA adaptation help.
4. **Participant generalisation** — whether models predict for participants excluded from
   training, under leave-one-participant-out or explicit split manifests.
5. **Adaptation** — how state estimates, uncertainty and availability translate into
   recommendations, in recorded replay and in the experimental live control service.

These are objectives, not claims. Interpret every result within its dataset, target
definition, cohort and evaluation protocol.

## The package

```text
src/mac/
  data/            Session indexing and I/O, label parsing, alignment, windows,
                   dataset and cache protocols for EgoEmotion / SEED-V / RELAX
  preprocessing/   Streaming-compatible pipeline and quality control;
                   per-modality preprocessing for PPG, ECG, SEED-V, RELAX physio
  features/        Handcrafted physiological, eye, head and video features;
                   dynamic texture descriptors; frozen VideoMAE v2 embeddings
  encoders/        Pretrained encoder wrappers: EEGPT, REVE, PaPaGei, Pulse-PPG,
                   ECGFounder, VideoMAE v2, InceptionTime, PatchTST, NeuroRVQ
  fusion/          Early / mid / late, bottleneck, HEALNet, Perceiver IO, Q-Former,
                   TMC, CGGM, MM-Lego, distillation, frozen compression,
                   and the minimal Ridge / 1D-CNN fusion benchmarks
  models/          EEG heads, LoRA, head-motion CNN, classical condition models,
                   temporal 1D-CNN, visual models
  tasks/           Task heads, losses, relaxation regression, condition controls
  training/        LOSO trainer and early stopping; condition, state, policy and
                   video training entry points
  evaluation/      Metrics, participant-fold contracts, LOPO, safety gates
  experiments/     Research-only experiment orchestration
  adaptive/
    offline/       Frozen-prefix inference and chronological recorded replay
    control/       Live control runtime (separate from offline replay by design)
  realtime/        Shadow clock, engine, serve, replay, recommendation policy
  reporting/       Run summaries, experiment reports, results registry
  config/          Layered configuration (base -> experiment -> local) and the
                   flat YAML loader used by the fusion runners
  utils/           Atomic writes, hashing, logging, TensorBoard helpers
  cli.py           The `mac` command
```

Supporting trees at the repository root: `scripts/` (84 runners, extraction, audits and
reports), `analysis/` (56 offline analyses, ablations, figures), `configs/`, `tests/` (65
modules), `integrations/unity/`, `data/contracts/`, `docs/`, `Auxiliary/`.

`src/real_time_ml/` and `src/src/` are generated compatibility shims that alias the old
import paths onto `mac`. They are scheduled for removal.

## Installation

Python 3.11.

```powershell
conda env create -f environment.yml
conda activate mac
mac --help
```

Or into an existing 3.11 environment:

```powershell
python -m pip install -e ".[dev]"
```

The core install deliberately excludes PyTorch, so the classical workflow and the
data-independent tests install without a GPU stack. For the pretrained-encoder and fusion
work, install a platform-appropriate PyTorch build first, then:

```powershell
python -m pip install -e ".[dl,viz,ecg,head]"
```

| Extra | Scope |
| --- | --- |
| `dl` | torch, torchvision, einops, transformers, huggingface-hub, safetensors, timm, h5py |
| `viz` | matplotlib, seaborn, tqdm, tensorboard |
| `ecg` / `head` | neurokit2 / ahrs |
| `dev` | pytest, pytest-cov, ruff |

`environment-videomae2.yml` stays a **separate** environment: it pins `timm` 0.4.12, which
cannot coexist with the `timm` 1.x the encoder stack needs. `requirements.txt`,
`requirements-extraction.txt` and `requirements-dev.txt` carry the pinned versions that
were validated for the fusion research on Linux.

## Configuration

Two configuration systems coexist, because the two halves used different ones and neither
was reduced to the other in this first merged version.

```text
configs/project.yaml              Legacy operational defaults          -+
       |                                                               |  layered:
configs/base.yaml                 Shared experiment/protocol defaults  |  runtime,
       |                                                               |  classical,
configs/experiments/*.yaml        Run ID and experiment settings       |  analysis
       |                                                               |
configs/local.yaml                Local data roots and device          -+

configs/egoemotion.yaml           Flat EgoEmotion fusion config        -+  flat:
configs/seedv_*.yaml              SEED-V protocols                      |  fusion
configs/fusion/ ablation/ finetune/   Fusion, ablation, fine-tuning    -+  research
```

`configs/local.yaml` is ignored by Git; copy it from `configs/local.example.yaml` and set
`paths.raw_root`, `paths.labels_root` and device settings. Global CLI options such as
`--experiment` and `--local-config` come **before** the subcommand.

> `configs/egoemotion.yaml` was `configs/base.yaml` before the merge. It was renamed because
> the layered system needs its own `configs/base.yaml` at that exact location.

## Execution paths

| Path | Entry points | Scope |
| --- | --- | --- |
| Offline research | `mac` extraction / training / evaluation commands; `scripts/`; `analysis/` | Features, models, comparisons and reports from recorded data |
| Shadow inference | `mac replay`, `mac serve` | State predictions and recommendations for logging or display; requires `shadow=true` |
| Adaptive Control | `mac adaptive-model`, `mac adaptive-control` | Separate experimental service with its own registry, readiness checks and control policy; it **can** issue commands and must not be confused with the Shadow bridge |

Research-only visual and fusion checkpoints are not automatically promoted to runtime
backends. Shadow and Adaptive Control default to the same UDP ports: run one per session,
or configure separate ports.

### Offline, condition-level workflow

```powershell
$runArgs = @("--experiment","configs/experiments/runtime-classical.yaml","--local-config","configs/local.yaml")
mac @runArgs index
mac @runArgs preprocess
mac @runArgs extract-features --no-video
mac @runArgs train-state
mac @runArgs evaluate
mac @runArgs report
```

### EgoEmotion 10-second segment fusion

```bash
python scripts/run_experiment_10s.py \
  --config configs/egoemotion.yaml --fusion_config configs/fusion/early.yaml \
  --embeddings_dir data/embeddings/egoemotion/10s_task_aware \
  --name ego_10s_early --device cuda
```

### SEED-V

```bash
python scripts/run_seedv_experiment.py --config configs/seedv_base.yaml
```

### RELAX aligned protocol

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

The earlier foundation protocol (`scripts/relax_foundation/run_relax_foundation_probe.py`)
expects a different cache format and is **not** interchangeable with the aligned runner
despite the similar name. See [the code map](docs/code-map.md) for protocol boundaries.

### Unity

```powershell
mac replay --help
mac serve --help
mac adaptive-model list
mac adaptive-control --help
```

Default Shadow transport: Unity to Python `127.0.0.1:5055`, Python to Unity `127.0.0.1:5056`.
See [the protocol](integrations/unity/PROTOCOL.md) and the [C# bridge](integrations/unity/RtmlShadowUdpBridge.cs).

## Data and supervision

**Condition-level RELAX study:** 15 participants x 9 conditions = 135 participant-condition
observations, split into complete non-overlapping 10-second windows within each condition.
Targets come from questionnaire ratings, `(rating - 1) / 6`. Repeating a condition rating
across its windows does not create additional independent observations.

**Windows RQ2 track:** a separate locked FMQ-9 contract — 9 participants, 81 condition
labels, 567 source windows, 545 common-valid windows — validated by
[`mac.windows_rq2_representations`](src/mac/windows_rq2_representations.py). `WINDOWS_DONE.json`
marks completion of that track only.

**EgoEmotion and SEED-V** have their own labels, sampling units, cache formats and splits.
Task-level and 10-second caches are not interchangeable.

Raw recordings, questionnaires and most generated outputs are external and Git-ignored. A
fresh clone supports source inspection and data-independent tests; reproducing results also
needs the matching source data, configuration and artifacts.

Prepare pretrained weights separately: VideoMAE v2 (`OpenGVLab/VideoMAEv2-Base`), PaPaGei /
Pulse-PPG (expected in sibling checkouts `papagei-foundation-model/` and `pulseppg/`), EEGPT,
REVE, ECGFounder, NeuroRVQ.

## Evaluation and reproducibility

Record, for every experiment: commit, runner, full command and configuration; dataset
version, target definition, cohort and window policy; participant splits; encoder and
checkpoint identities, preprocessing, cache format and masks; seeds, protocol hashes,
package versions and hardware; per-fold outputs and the aggregation used.

LOSO / LOPO hold out one participant per fold. Segments must follow participant splits —
splitting correlated segments independently answers a different question. Compare fusion
against single-modality and condition-only baselines under matching rules. Scores from
different targets, cohorts or protocols are not comparable.

## Tests

```powershell
python -m pytest -m "not external and not integration and not slow"
```

Marker meanings: `integration` reads participant source data, `slow` trains models or does
expensive work, `external` needs pretrained weights, external model code or real
participant data.

Recorded state of the merged suite on the development machine, against the two pre-merge
baselines:

| | collected | passed | failed | skipped | collection errors |
| --- | --- | --- | --- | --- | --- |
| `Relax-Model` before merge | 96 | 93 | 3 | 0 | 0 |
| `real-time-vis-physio-fusion` before merge | 198 | 185 | 0 | 13 | 7 |
| **sum** | **294** | **278** | **3** | **13** | **7** |
| **merged** | **294** | **278** | **3** | **13** | **7** |

The 3 failures and 7 collection errors are the same ones, for the same reasons, as before
the merge: the failures read a generated cross-project contract that was never versioned,
and the collection errors are `einops` and `huggingface_hub` missing from that environment.
Installing the `dl` extra removes the collection errors.

## Contributing

- Put reusable implementations in `src/mac/` at the pipeline stage they belong to,
  experiment orchestration in `scripts/` or `analysis/`, and parameters in `configs/`.
- Keep source recordings read-only and generated outputs outside versioned source trees.
- Preserve participant-condition supervision and the declared cohort; make exclusions explicit.
- Fit preprocessing, feature selection and tuning only on permitted training data.
- Use a distinct configuration and run ID for a new comparison.
- Keep research-only representations out of runtime model selection, and keep
  `adaptive/offline/` distinct from `adaptive/control/`.
- For a new encoder, document expected shapes, temporal sampling, channel order, units,
  weights and missing-modality behaviour.
- Describe evidence at its actual level: implementation, offline evaluation, recorded
  replay, or prospective participant study.
