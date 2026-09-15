# `mac` package map

One package, organised by pipeline stage rather than by which of the two merged
repositories a module came from. The thesis-facing workflow is the RELAX
condition/state and real-time pipeline. EgoEmotion and SEED-V runners are kept
under `auxiliary/benchmarks/`; this package retains only the reusable adapters
and model components they share with RELAX. See [the archived merge map](../../auxiliary/merge_history/MERGE-MAP.md)
for the old-to-new import table, and [the code map](../../docs/code-map.md) for protocol boundaries.

## Pipeline stages

| Subpackage | Responsibility | Merged from |
| --- | --- | --- |
| `data` | RELAX source indexing, labels, tables, video metadata, alignment and window construction; compatibility adapters for benchmark caches | both |
| `preprocessing` | Marker alignment, condition boundaries, windows and MNE quality audits; PPG, ECG and RELAX physiology preprocessing | both |
| `features` | Handcrafted EEG/ECG, eye, head and video features; dynamic-texture descriptors; frozen VideoMAE v2 embeddings | RTML |
| `encoders` | Pretrained encoder wrappers and the config-driven modality registry | VisPhy |
| `fusion` | Fusion architectures, the shared factory, projection and mask-aware pooling, frozen-feature compression, and the minimal Ridge / 1D-CNN benchmarks | both |
| `models` | EEG heads, LoRA, head-motion CNN; condition-level classical models, temporal 1D-CNN, visual models; feature-group and validation helpers | both |
| `tasks` | Task heads, losses, relaxation regression and condition controls | VisPhy |
| `training` | LOSO trainer and early stopping; condition, state, policy and video training entry points | both |
| `evaluation` | Metrics, participant-fold contracts, LOPO, dynamic-texture statistics, deployment safety gates | both |
| `experiments` | Research-only experiment orchestration, isolated from runtime training | RTML |
| `adaptive.offline` | Frozen-checkpoint prefix inference and chronological, non-interventional recorded replay | VisPhy |
| `realtime` | Shadow clock, buffers, inference engine, replay and serving; the Shadow recommendation policy | RTML |
| `runtime` | Compatibility facades re-exporting `realtime`, from an earlier intra-RTML namespace migration | RTML |
| `reporting` | Run summaries, experiment reports, results registry, video and multimodal report builders | both |
| `config` | Layered configuration and run output paths, plus the flat YAML loader | both |
| `utils` | Atomic writes and hashing (`utils.io`), logging setup, TensorBoard helpers | both |

Top-level modules: `cli.py` (the `mac` command), `schema.py` (message validation),
`windows_rq2_representations.py` (the Windows RQ2 contract and handoff).

## Two things that must not be confused

**`adaptive.offline` vs the archived Adaptive Control runtime.** The first replays recorded
sessions through frozen models and never intervenes. The second is retained under
`auxiliary/adaptive_control/` for historical Unity/UDP experiments and is not part of the
active `mac` package or thesis evidence chain.

**`config.load_config` vs `config.simple.load_config`.** Two functions with the same name
and different contracts, one from each merged repository:

```python
from mac.config import load_config          # -> ProjectConfig, layered, validates the protocol
from mac.config.simple import load_config   # -> dict, flat YAML, used by the fusion runners
```

The flat one is deliberately **not** re-exported from `mac.config`, so importing the package
never gives you the wrong one by accident.

## Compatibility shims

`src/real_time_ml/` and `src/src/` alias the pre-merge import paths onto this package. Leaf
modules alias through `sys.modules`, so the old and new names yield the same objects rather
than duplicate classes. They are scheduled for removal; write new code against `mac`.

`__pycache__` directories are generated locally and ignored. The former `features.zip` under
this package was a local source snapshot containing cached bytecode; it lives in
`auxiliary/archive/` and is not part of the Python package.
