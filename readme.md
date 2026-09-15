# Real-time Visual–Physiological Fusion

A research codebase for studying emotion and relaxation through visual, behavioral, and physiological signals. It combines pretrained feature extraction, multimodal fusion, subject-independent evaluation, and offline adaptive replay across EgoEmotion, SEED-V, and RELAX experiments.

The long-term motivation is to understand how complementary sensors can support responsive, personalized affective systems. The implemented adaptive workflows replay recorded sessions offline; they do not establish the effectiveness of a live closed-loop system.

## Research purpose

Emotion and relaxation are difficult to infer from a single sensor. Video provides visual context, gaze and head motion describe behavior, and PPG, ECG, and EEG provide physiological measurements. These sources differ in timing, noise, availability, and representation requirements.

This project investigates the following questions:

1. **Modality contribution:** Which signals carry useful information, and does combining them improve prediction over individual modalities under the same evaluation protocol?
2. **Transfer from pretrained models:** How well do frozen representations transfer to affective tasks, and when are projection, compression, fine-tuning, or LoRA adaptation useful?
3. **Fusion design:** How do early, intermediate, and late fusion compare with attention-based approaches, HEALNet, and MM-Lego workflows?
4. **Participant generalization:** Can models predict targets for participants excluded from training, using leave-one-subject-out (LOSO) evaluation or explicit participant split manifests?
5. **Condition effects and representation quality:** How much predictive information reflects physiology or behavior, and how much relates to experimental conditions or dataset structure?
6. **Offline adaptation:** How do frozen-model prefix predictions and adaptive controllers behave when replaying recorded sessions?

These are research objectives, not claims that every fusion method improves performance. Interpret results within their dataset, target definition, cohort, and evaluation protocol.

## Intended users

| User | Typical use |
| --- | --- |
| Affective computing and multimodal learning researchers | Compare representations, fusion architectures, and modality ablations. |
| Physiological signal and EEG researchers | Integrate encoders and inspect preprocessing, channels, units, and representation quality. |
| Human–computer interaction and relaxation researchers | Study recorded behavior and physiology, and explore offline adaptive replay. |
| Graduate students and research engineers | Reproduce experiments or extend shared data, model, and training components. |
| Collaborators and reviewers | Trace configurations, input manifests, audits, predictions, and reports. |

Running experiments assumes familiarity with Python, PyTorch, command-line tools, and the selected dataset. Participants are the subjects of the recorded studies; the repository itself is primarily a researcher-facing tool.

## Research tracks

The repository combines several research stages. Their labels, sampling units, cache formats, and splits differ. Choose a protocol before preparing inputs.

| Track | Entry point | Main inputs |
| --- | --- | --- |
| EgoEmotion task-level fusion | `scripts/run_experiment.py` | `configs/base.yaml`, fusion configuration, task embeddings |
| EgoEmotion 10-second fusion | `scripts/run_experiment_10s.py` | Task-aware segment manifest and embeddings |
| PPG adaptation / MM-Lego | `scripts/run_finetune_ppg.py` / `scripts/run_lego_experiment.py` | Fine-tuning configuration / pretrained Lego weights |
| SEED-V EEG and eye tracking | `scripts/run_seedv_experiment.py`, `scripts/run_seedv_reve_experiment.py`, `scripts/run_seedv_eeg_eye_fusion.py` | Encoder-specific preprocessing, embeddings, and configurations |
| RELAX foundation probes | `scripts/relax_foundation/run_relax_foundation_probe.py` | Earlier `samples` cache |
| RELAX aligned fusion | `scripts/run_relax_foundation_probe.py` | Aligned cache, labels, windows, masks, and splits |
| RELAX frozen-feature compression | `scripts/run_relax_compression_fusion_v2.py` | Five-modality protocol, preregistration, fixed cohorts and seeds |
| RQ2 audit and pipeline | `scripts/run_rq2_wsl.py` | External hand-off and aligned caches |
| HEALNet prefix inference / replay | `scripts/run_healnet_prefix_inference.py` / `scripts/run_healnet_adaptive_replay.py` | Recorded P009 sessions and frozen model |

Model availability depends on the runner. See the [code map and protocol boundaries](docs/code-map.md) for extraction, ablation, audit, and reporting entry points. Supporting consolidation documents are currently in Chinese.

## Experiment workflow

1. **Define the experiment:** Select the dataset, target, participant cohort, sample duration, and evaluation split.
2. **Prepare data:** Check timestamps, channels, signal units, valid windows, and modality availability.
3. **Prepare representations:** Extract features or load the cache expected by the selected encoder and runner.
4. **Train and evaluate:** Apply the protocol's splits, masks, seeds, and fusion configuration.
5. **Audit and report:** Preserve configurations, input identities, predictions, and evaluation outputs.

Cached-feature training and raw-signal extraction have different resource requirements. Some tracks also require external encoder source repositories, pretrained checkpoints, or a hand-off from another project.

## Installation

Run commands from the repository root. The consolidation was validated on Linux with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Alternatively:

```bash
conda env create -f environment.yml
conda activate visphy
```

| Dependency file | Scope |
| --- | --- |
| `requirements.txt` | Core cached-feature training and evaluation |
| `requirements-extraction.txt` | Core dependencies plus signal processing and feature extraction |
| `requirements-dev.txt` | Extraction dependencies plus pytest |

For CUDA, install a platform-compatible PyTorch 2.10.0 build first. The validation environment used `2.10.0+cu128`. Dependency versions were checked in an existing environment; a complete fresh installation has not yet been validated. [Historical environment exports](docs/environments/) are retained for reference.

## Data and pretrained models

Prepare the datasets and weights needed for your selected track separately. Local datasets and newly generated caches and outputs are generally ignored by Git. Some PaPaGei weights, reports, and `code.zip` already exist in Git history; ignore rules do not remove historical files.

- **VideoMAE V2:** Uses `OpenGVLab/VideoMAEv2-Base` remote model code and safetensors weights. Prepare a download or an existing Hugging Face cache.
- **PaPaGei / Pulse-PPG:** Default source locations are sibling repositories named `papagei-foundation-model/` and `pulseppg/`.
- **EEGPT, REVE, ECGFounder, and eye-tracking encoders:** Prepare the corresponding weights. Paths are defined by configurations, constructor arguments, and CLI options.
- **NeuroRVQ:** The adapter accepts an upstream source directory, checkpoint, modality, and channel list.

Some historical runners retain Windows/WSL paths, fixed participants, seeds, or checksums. Inspect the selected configuration and CLI before running. Replacing a cache or cohort changes the experiment and must be recorded.

## Repository layout

```text
configs/                Dataset, fusion, fine-tuning, and ablation configurations
src/
  data/                 Loading, segmentation, preprocessing, and cache protocols
  encoders/             Pretrained encoder wrappers and registration
  fusion/               Fusion models, shared factory, and feature compression
  models/               EEG heads, LoRA components, and head-motion CNN
  tasks/                Task heads, losses, and condition controls
  trainer/              LOSO training and early stopping
  adaptive/             Prefix inference, controllers, and offline replay
  utils/                Configuration, metrics, logging, and reporting
scripts/                Runners, extraction, audits, and reports
  relax_foundation/     Earlier RELAX foundation protocol
tests/                  Standard test suite
data/
  datasets/             Raw datasets, including EgoEmotion and RELAX
  preprocessed/         Preprocessed signals, including SEED-V
  embeddings/           EgoEmotion and SEED-V caches
weights/                External pretrained weights
checkpoints/            Models trained by this project
artifacts/              RELAX caches, protocol artifacts, and experiment outputs
wsl_results/            RQ2 WSL outputs
combined/               RQ2 combined reports
```

## Running experiments

These examples require prepared inputs. Paths beginning with `/path/to/` are placeholders. Run each entry point with `--help` to inspect its supported options.

### EgoEmotion: 10-second segment fusion

Prepare task-aware segments and embeddings with `scripts/segment_and_extract_10s.py`, then update local paths in `configs/base.yaml`.

```bash
python scripts/run_experiment_10s.py \
  --config configs/base.yaml --fusion_config configs/fusion/early.yaml \
  --embeddings_dir data/embeddings/egoemotion/10s_task_aware \
  --name ego_10s_early --device cuda
```

Task-level caches and 10-second segment caches are not interchangeable. PPG fine-tuning and MM-Lego have separate entry points. The base configuration specifies encoders, projection width, training parameters, loss weights, and output directories.

### SEED-V

```bash
python scripts/run_seedv_experiment.py --config configs/seedv_base.yaml
```

Check preprocessing, embedding, and checkpoint paths, channel order, and signal units first. `extract_seedv_embeddings.py` and `extract_seedv_reve_embeddings.py` serve different encoders. Use the corresponding `run_seedv_*` scripts for LoRA, fine-tuning, fusion, and ablations.

### RELAX: aligned protocol

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

Build aligned caches with `scripts/build_relax_alignment_cache.py`. Labels, windows, splits, and masks must match the cache identity.

### RELAX: earlier foundation protocol

```bash
python scripts/relax_foundation/run_relax_foundation_probe.py \
  --embedding-cache /path/to/sample_cache.pt \
  --cohorts /path/to/cohorts.json --cohort all_135 \
  --modalities eeg ecg eye head video --fusion early \
  --output-dir artifacts/relax/foundation_run
```

This runner expects a top-level `samples` list and `embedding_dims`. The aligned runner expects parallel participant/condition arrays, targets, embeddings, and masks. The formats and runners are incompatible despite their similar names.

Compression v1/v2, condition-anchor probes, and embedding-ladder experiments also have distinct protocols. Consult the [code map](docs/code-map.md) before reusing artifacts.

### RQ2: audit the hand-off first

```bash
python scripts/run_rq2_wsl.py \
  --shared-root /path/to/rq2_shared \
  --cache-root artifacts/relax/aligned_20260716 \
  --output-root wsl_results --audit-only
```

The audit checks the contract, labels and windows, folds, anchors, Windows completion markers, and cache identity. Failure produces an error report and exits without rebuilding inputs. Remove `--audit-only` to run the subsequent pipeline after preparing the hand-off. The full pipeline also writes reports into `combined/`.

### HEALNet offline replay

Inspect required session and model inputs:

```bash
python scripts/run_healnet_prefix_inference.py --help
python scripts/run_healnet_adaptive_replay.py --help
```

These workflows study adaptive decisions using recorded prefixes and frozen predictions. Replay metrics describe behavior on historical data. Live sensing latency and prospective participant outcomes require separate evaluation.

## Evaluation and reproducibility

Record the following for each experiment:

- Git commit, runner, complete command, and configuration.
- Dataset version, target definition, cohort, and window policy.
- Participant splits for training, validation, and testing.
- Encoder source and checkpoint identities, preprocessing, cache format, and masks.
- Seeds, required protocol hashes, package versions, and hardware.
- Per-fold outputs and the aggregation method used in reports.

LOSO holds out a participant for testing in each fold. Segments must follow participant splits; independently splitting correlated segments answers a different generalization question. Use the selected protocol's leakage audits and split checks. Compare fusion with single-modality and condition baselines using matching evaluation rules.

Metrics and report locations vary by track. For example, the EgoEmotion base configuration selects weighted F1 as its primary metric. Scores from different targets, cohorts, or protocols are not directly comparable.

## Tests and validation status

Run tests that do not require real participant data or external pretrained resources:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 python -m pytest -q -m "not external"
```

With the required PaPaGei source/weights, ECGFounder weights, and real ECG available:

```bash
python -m pytest -q
```

`pytest.ini` collects the standard suite from `tests/`. Temporary checks in artifact directories are outside that suite. Tests marked `external` require local resources.

The recorded consolidation validation on **2026-09-11** reported **261 passing tests**, successful source compilation, and successful `--help` checks for 11 entry points. This is a historical result. Full LOSO training, GPU feature extraction, the real external RQ2 pipeline, and online control were outside that validation. See the [validation record](docs/consolidation-validation.md) for details.

## Development and maintenance

Place reusable implementations in `src/`, experiment orchestration and reporting in `scripts/`, and parameters in `configs/`. Standard training, 10-second training, and PPG fine-tuning share fusion construction and pooling through `src/fusion/factory.py`.

For new encoders, document expected shapes, temporal sampling, channel order, units, weights, and missing-modality behavior. Preserve existing cache contracts and use a distinct entry point when changing experimental protocols. Add focused validation for substantive behavior changes.

`main` is the integration branch for the consolidated research code. Earlier stages remain accessible through Git history after obsolete branch references are removed. Branch cleanup does not require deleting local datasets or uncommitted work.

## Further documentation

- [Code map and protocol boundaries](docs/code-map.md)
- [Branch consolidation history](docs/branch-history.md)
- [Consolidation validation](docs/consolidation-validation.md)
- [Historical environments](docs/environments/)
- [Research wiki](research-wiki/), [reports](reports/), and [design notes](docs/superpowers/)

Historical reports describe the research state when written. Start current runs from this README, the code map, and the selected runner's actual arguments.
