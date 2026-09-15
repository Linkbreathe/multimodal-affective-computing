# Merge history and legacy snapshots

This directory contains provenance material for the consolidation of
`real-time-vis-physio-fusion` and `Relax-Model` into
`multimodal-affective-computing`. It is not part of the active thesis workflow.

| Path | Purpose |
| --- | --- |
| `MERGE-MAP.md` | Old-to-new paths and import namespaces used during consolidation. |
| `branch-history.md` | Historical branch cleanup and worktree notes from VisPhy. |
| `consolidation-validation.md` | Validation results recorded before the thesis-scope reorganisation. |
| `legacy/` | Original README, environment and test-configuration snapshots retained verbatim. |

The files describe the repository at specific historical commits. Commands, paths,
dependency versions and test counts in them may not match the current branch. Use the
root `README.md`, `docs/code-map.md`, `docs/thesis-scope.md` and `data/contracts/` for
current operation and reproduction.

Both source histories remain in the Git object graph. The source remotes are not
currently configured in this checkout, and `git log --follow` does not reliably cross
the prefix rewrite introduced by `git subtree`. Query the imported source commit and its
original path when detailed provenance is required.
