# SEED-V benchmark material

This directory contains the SEED-V-only configurations, EEG/eye emotion runners and
their tests. The code is retained for comparison and reproduction, not as part of the
thesis runtime path.

Run commands from the repository root, for example:

```powershell
python Auxiliary/benchmarks/seedv/scripts/run_seedv_experiment.py `
  --config Auxiliary/benchmarks/seedv/configs/seedv_base.yaml
```

The runners reuse `mac.encoders` and `mac.models` implementations and require external
SEED-V data and pretrained weights.
