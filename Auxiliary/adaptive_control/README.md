# Archived Adaptive Control

This folder contains the experimental Unity/UDP Adaptive Control runtime. It
is retained for historical reproduction and is deliberately outside the
paper's active thesis CLI.

The active thesis paths are:

- `mac realtime serve` for live Shadow inference;
- `mac replay` and `mac replay-video` for recorded replay;
- `mac adaptive.offline` for offline adaptive replay and analysis.

The archived service can be inspected from the repository root with:

```powershell
$env:PYTHONPATH = "src;."
python -m Auxiliary.adaptive_control.cli adaptive-model list
python -m Auxiliary.adaptive_control.cli adaptive-model verify --bundle realtime_multimodal_window_v1
python -m Auxiliary.adaptive_control.cli adaptive-control --bundle realtime_multimodal_window_v1
```

`start-adaptive-control.ps1` and `launch-adaptive-control.cmd` are kept here
with the service, configuration, model manifests, and its dedicated tests.
