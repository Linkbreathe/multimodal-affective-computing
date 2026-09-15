# EgoEmotion benchmark material

This directory contains the EgoEmotion-only configurations, runners, tests, historical
reports and VAD figures moved out of the thesis-facing root.

Run commands from the repository root, for example:

```powershell
python auxiliary/benchmarks/egoemotion/scripts/run_experiment_10s.py `
  --config auxiliary/benchmarks/egoemotion/configs/egoemotion.yaml `
  --fusion_config configs/fusion/early.yaml
```

The runners reuse `mac` encoders, fusion modules and training utilities. They require the
external EgoEmotion data and, for most experiments, pretrained weights.
