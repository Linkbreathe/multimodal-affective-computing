# Multimodal Affective Computing

**One pipeline from recorded sensors to a running adaptive system: session ingest, feature and representation extraction, multimodal fusion, participant-independent evaluation, recorded replay, and live inference.**

This repository is the merger of two previously separate codebases that were already
exchanging data through the filesystem and machine-specific paths:

| Merged from | Called in the code | Supplies |
| --- | --- | --- |
| `real-time-vis-physio-fusion` | Project B | Reusable pretrained encoders and fusion architectures; EgoEmotion/SEED-V comparisons are auxiliary |
| `Relax-Model` | Project A | Thesis-facing session indexing, handcrafted features, classical and temporal models, Shadow inference, Unity integration |

They are now one installable package, `mac`, organised by **what each module does**
rather than by which project it came from. Both Git histories are preserved; see
[the merge map](docs/MERGE-MAP.md) to translate any path or command written before the merge.

The presence of an implemented workflow demonstrates that it runs. It does not establish
that a model generalises to new participants, that a representation measures an internal
state, or that adaptive control improves anything for a person.

[中文总览](README_zh.md) · [合并对照表](docs/MERGE-MAP.md) · [论文代码范围](docs/thesis-scope.md) · [代码地图](docs/code-map.md) · [输入契约](data/contracts/input_tables.md) · [Unity Shadow 协议](integrations/unity/PROTOCOL.md)

## Thesis purpose

Emotion and relaxation are hard to infer from any single sensor. Video carries visual
context, gaze and head motion describe behaviour, and PPG, ECG and EEG measure physiology.
These sources differ in timing, noise, availability and representation requirements.

The primary workflow is the RELAX-based thesis pipeline:

1. **Measurement and characterisation** — how physiology, gaze, head movement and visual
   context vary across conditions, including signal quality, missingness and baseline effects.
2. **Modality contribution** — which signals carry usable information, and whether combining
   them beats single modalities under one evaluation protocol.
3. **Representation transfer** — how well frozen pretrained representations transfer to
   affective targets, and when projection, compression, fine-tuning or LoRA adaptation help.
4. **Participant generalisation** — whether models predict for participants excluded from
   training, under leave-one-participant-out or explicit split manifests.
5. **Adaptation** — how state estimates, uncertainty and availability translate into
   recommendations, in recorded replay and the non-interventional Shadow runtime.

EgoEmotion and SEED-V remain available as auxiliary benchmarks under
`Auxiliary/benchmarks/`; they are not part of the thesis runtime surface. These are
objectives, not claims. Interpret every result within its dataset, target definition,
cohort and evaluation protocol.

## The package

```text
src/mac/
  data/            RELAX session indexing and I/O, label parsing, alignment, windows,
                   condition data and cache protocols; compatibility adapters for benchmarks
  preprocessing/   Streaming-compatible pipeline and quality control;
                   per-modality preprocessing for PPG, ECG and RELAX physiology
  features/        Handcrafted physiological, eye, head and video features;
                   dynamic texture descriptors; frozen VideoMAE v2 embeddings
  encoders/        Reusable pretrained encoder wrappers: EEGPT, REVE, PaPaGei, Pulse-PPG,
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
  realtime/        Shadow clock, engine, serve, replay, recommendation policy
  reporting/       Run summaries, experiment reports, results registry
  config/          Layered configuration (base -> experiment -> local) and the
                   flat YAML loader used by the fusion runners
  utils/           Atomic writes, hashing, logging, TensorBoard helpers
  cli.py           The `mac` command
```

Supporting trees at the repository root: `scripts/` and `analysis/` for thesis workflows,
`configs/` for runtime configuration, `tests/` for the active suite,
`integrations/unity/`, `data/contracts/`, `docs/`, and `Auxiliary/` for historical and
auxiliary benchmark material. See [论文代码范围](docs/thesis-scope.md).

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

`Auxiliary/benchmarks/egoemotion/environment-videomae2.yml` stays a **separate** benchmark
environment: it pins `timm` 0.4.12, which cannot coexist with the `timm` 1.x the encoder
stack needs. `requirements.txt`,
`requirements-extraction.txt` and `requirements-dev.txt` carry the pinned versions that
were validated for the fusion research on Linux.

## Configuration

Two configuration systems coexist, because the two halves used different ones and neither
was reduced to the other in this first merged version.

```text
configs/fusion/                  Shared fusion model templates
configs/project.yaml              RELAX operational defaults          -+  layered:
configs/base.yaml                 RELAX protocol defaults              |  runtime,
configs/experiments/*.yaml        Thesis experiment settings            |  classical
                                                                          -+  analysis

Auxiliary/benchmarks/egoemotion/configs/  EgoEmotion benchmark configs
Auxiliary/benchmarks/seedv/configs/       SEED-V benchmark configs
```

`configs/local.yaml` is ignored by Git; copy it from `configs/local.example.yaml` and set
`paths.raw_root`, `paths.labels_root` and device settings. Global CLI options such as
`--experiment` and `--local-config` come **before** the subcommand.

The benchmark configurations are intentionally outside the thesis-facing `configs/`
surface. The layered RELAX system uses `project.yaml -> base.yaml -> experiments/ ->
local.yaml`.

## Execution paths

| Path | Entry points | Scope |
| --- | --- | --- |
| Offline research | `mac` extraction / training / evaluation commands; `scripts/`; `analysis/` | Features, models, comparisons and reports from recorded data |
| Shadow inference | `mac replay`, `mac serve` | State predictions and recommendations for logging or display; requires `shadow=true` |

Research-only visual and fusion checkpoints are not automatically promoted to runtime
backends. The retired Adaptive Control experiment is archived under
[`Auxiliary/adaptive_control/`](Auxiliary/adaptive_control/); it is not part of the thesis
execution surface.

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

EgoEmotion and SEED-V benchmark commands are documented in
[`Auxiliary/benchmarks/README.md`](Auxiliary/benchmarks/README.md); they are deliberately
not listed as thesis execution paths.

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

**Auxiliary EgoEmotion and SEED-V benchmarks** have their own labels, sampling units,
cache formats and splits. Task-level and 10-second caches are not interchangeable with
the RELAX condition-level pipeline.

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

Recorded state of the merged suite on the development machine (Windows, Python 3.11),
against the two pre-merge baselines:

| | collected | passed | failed | skipped | collection errors |
| --- | --- | --- | --- | --- | --- |
| `Relax-Model` before merge | 96 | 93 | 3 | 0 | 0 |
| `real-time-vis-physio-fusion` before merge | 198 | 185 | 0 | 13 | 7 |
| **sum** | **294** | **278** | **3** | **13** | **7** |
| **merged, one test file per process** | **294** | **278** | **3** | **13** | **7** |
| merged, all in one process | 294 | 277 | 4 | 13 | 7 |

With one process per test file the merged suite matches the pre-merge baselines exactly.
Running everything in a single process costs **one extra failure**, which is a property of
that process, not of the code:

- **3 failures, unchanged from before the merge** — they read
  `artifacts/cross_project_alignment_2026-07-16/.../contract.json`, a generated artifact
  that was never versioned. They fail identically in the original `Relax-Model` checkout.
- **7 collection errors, unchanged from before the merge** — `einops` and `huggingface_hub`
  absent from that environment. Installing the `dl` extra removes them.
- **1 new failure, a Windows flake, not a code defect** —
  `test_cross_project_alignment.py::test_validation_ranking_uses_dedicated_validation_rows`
  raises `OSError: GetModuleFileNameEx failed` from inside `threadpoolctl` (3.6.0) while it
  enumerates loaded DLLs, not from any assertion. It passes when run alone, and passes when
  only the RTML-origin test files run in this repository. It appears because the merged
  suite now loads both dependency stacks into a single process, which makes that
  enumeration race more likely. Giving each test file its own process removes it — that is
  the first merged row above, measured by running `pytest <file>` for all 63 files. The
  convenient way is `pip install pytest-xdist` then `pytest -n 4 --dist loadfile`.

Static verification that does not depend on the environment:

```powershell
# every module in the package and both shim trees imports without error
python -c "import pkgutil,importlib,mac; [importlib.import_module(m.name) for m in pkgutil.walk_packages(mac.__path__,'mac.')]"
python -m compileall -q src scripts analysis tests Auxiliary/benchmarks Auxiliary/adaptive_control
```

The active `scripts/` and `analysis/` entry points are the thesis workflow. Dataset-specific
benchmark entry points and the retired Adaptive Control runtime are kept under `Auxiliary/`
and are validated separately when their external datasets, model dependencies or Unity
installation are available.

## Contributing

- Put reusable implementations in `src/mac/` at the pipeline stage they belong to,
  experiment orchestration in `scripts/` or `analysis/`, and parameters in `configs/`.
- Keep source recordings read-only and generated outputs outside versioned source trees.
- Preserve participant-condition supervision and the declared cohort; make exclusions explicit.
- Fit preprocessing, feature selection and tuning only on permitted training data.
- Use a distinct configuration and run ID for a new comparison.
- Keep research-only representations out of runtime model selection, and keep
  `adaptive/offline/` distinct from the archived `Auxiliary/adaptive_control/` runtime.
- For a new encoder, document expected shapes, temporal sampling, channel order, units,
  weights and missing-modality behaviour.
- Describe evidence at its actual level: implementation, offline evaluation, recorded
  replay, or prospective participant study.
