# `real_time_ml` package map

The Python package is kept together because its modules are imported by the `rtml` CLI, tests, offline analysis scripts, or the Unity services. The subpackages have these responsibilities:

- `config`: legacy and layered experiment configuration plus run output paths.
- `data`: source indexing, labels, tables, video metadata, and alignment helpers.
- `preprocessing`: marker alignment, condition boundaries, windows, and MNE quality audits.
- `features`: EEG/ECG, eye, head, video, VideoMAE2, and dynamic-texture extraction.
- `modeling`: condition-level models, temporal models, video/fusion research models, reports, and safety gates.
- `training`: stable public training entry points used by the CLI; implementations remain in `modeling` for compatibility.
- `realtime`: the implemented Shadow clock, buffers, inference engine, replay, and serving paths.
- `runtime`: compatibility facades that re-export the `realtime` implementation during namespace migration.
- `adaptive_control`: the separate experimental Adaptive Control service, model registry, policy, and readiness logic.
- `policy`: the legacy Shadow recommendation policy.
- `evaluation`, `experiments`, and `windows_rq2_representations.py`: research comparisons, alignment contracts, dynamic-texture experiments, and the Windows RQ2 handoff.
- `reporting`, `schema.py`, and `utils.py`: summaries, message validation, file writing, and shared utilities.

`__pycache__` directories are generated locally and ignored. The former `features.zip` under this package was a local source snapshot containing cached bytecode; it has been moved to `Auxiliary/archive/` and is not part of the Python package.
