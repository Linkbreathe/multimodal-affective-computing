# Auxiliary benchmarks

These folders contain dataset-specific experiments that support development and
comparison but are outside the thesis runtime path.

- [`egoemotion/`](egoemotion/): EgoEmotion segmentation, embedding, fusion, ablation and
  historical reporting material.
- [`seedv/`](seedv/): SEED-V EEG/eye emotion benchmarks, preprocessing entry points and
  ablations.

Both benchmark groups reuse implementations from `src/mac/`. The reusable implementations
stay in the installable package because the RELAX thesis pipeline uses parts of the encoder
and fusion stack. The benchmark folders themselves are not imported by the active runtime.
