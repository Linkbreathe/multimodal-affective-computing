# Frozen legacy artifacts

Everything that was already below `artifacts/` before the layered
configuration migration is retained in place as compatibility evidence.  Do
not move, rename, or overwrite its models, 946-window features, video caches,
or reports.

New work must use an explicit `run.id` through `rtml --experiment ...` and is
written below `artifacts/runs/<run_id>/`.  Run
`python scripts/export_legacy_manifest.py` to refresh the inventory after an
explicitly reviewed legacy import.
