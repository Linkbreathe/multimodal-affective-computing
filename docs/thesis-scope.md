# Thesis scope

This branch makes the thesis workflow the primary repository surface. The thesis
workflow is the RELAX-based multimodal relaxation/state pipeline: participant and
condition indexing, questionnaire targets, 10-second windows, physiology/eye/head/video
features, condition-level models, participant-independent evaluation, recorded replay and
non-interventional Shadow inference.

## Primary implementation

The active implementation remains in `src/mac/` and is organized by pipeline stage:

- `mac.data`, `mac.preprocessing`, `mac.features`: RELAX ingestion, alignment, quality
  control and feature extraction;
- `mac.models`, `mac.tasks`, `mac.training`, `mac.evaluation`: state/condition models,
  training, LOPO evaluation and deployment safety gates;
- `mac.adaptive.offline`, `mac.realtime`, `mac.reporting`, `mac.config`: offline replay,
  Shadow runtime, reports and layered runtime configuration;
- `scripts/relax_foundation/`, the `scripts/run_relax_*` runners, `scripts/rq2_*` and
  `analysis/`: thesis experiments, audits and execution boundaries;
- `auxiliary/integrations/unity/`: optional external Shadow protocol and Unity bridge
  material. It supports replay/deployment demonstrations but is not part of the installed
  Python package.

The encoder and fusion implementations under `src/mac/encoders/` and
`src/mac/fusion/` stay available as reusable components. Some of them are used by the
RELAX representation and alignment experiments, so moving them wholesale to an archive
would break the thesis path.

## auxiliary benchmark material

Dataset-specific benchmark material that is not part of the thesis runtime is now under:

```text
auxiliary/benchmarks/egoemotion/
  configs/ scripts/ tests/ reports/ figures/
auxiliary/benchmarks/seedv/
  configs/ scripts/ tests/
```

This includes EgoEmotion/SEED-V entry points, their experiment configurations, tests and
historical reports. The source adapters that are still imported by the shared `mac`
package remain in `src/mac/`; they are implementation support, not thesis datasets.

The experimental Adaptive Control service is archived separately under
`auxiliary/adaptive_control/` with its Unity/UDP service, model registry, configuration,
launchers and dedicated tests. It is intentionally outside the thesis-facing CLI and
evidence chain.

Some existing supplementary report generators still contain read-only readers for historical
`artifacts/realtime/adaptive_control/` logs. They are retained as evidence provenance and do
not import, start or expose the archived control service; new thesis execution does not depend
on those logs.

## Boundary rule

The thesis runtime may import `mac`, but active thesis code must not depend on an
auxiliary benchmark runner or report. auxiliary runners may import reusable `mac`
components. This keeps the thesis path installable and testable without requiring the
EgoEmotion/SEED-V datasets.

The two RELAX cache protocols also remain separate: the foundation sample cache is used
by `scripts/relax_foundation/`, while the participant-condition aligned cache is used by
the aligned RELAX runners. They are not interchangeable.
