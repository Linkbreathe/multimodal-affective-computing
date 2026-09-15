# Output contract

Machine artifacts use `manifests/`, `preprocessed/`, `features/`, `models/`,
`checkpoints/`, `metrics/`, `predictions/`, and `logs/` under a run directory.
The only final human-readable artifact is
`reports/<run_id>_summary_zh.md`.  All runs remain Shadow-only; a failed safety
gate yields `hold`.
