"""Shared test fixtures for the merged multimodal affective computing suite.

Combines the egoEMOTION synthetic-data fixture (formerly
``real-time-vis-physio-fusion/tests/conftest.py``) with the source-path shim
that ``Relax-Model/tests/conftest.py`` used before the package was installable.
The shim is a no-op once ``pip install -e .`` has been run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
    sys.path.insert(0, str(_SRC / "src"))
@pytest.fixture
def ego_data_dir(tmp_path):
    root = tmp_path / "egoemotion"
    root.mkdir()
    task_times = {}
    for number in range(1, 42):
        subject = f"{number:03d}"
        task_times[subject] = {"session_A": [1000, 2800], "video_Neutral": [1090, 1180]}
        if number != 20:
            (root / subject).mkdir()
    np.save(root / "task_times.npy", task_times)
    np.save(root / "005/gaze_90fps.npy", np.arange(3600, dtype=np.float32).reshape(1800, 2))
    np.save(root / "005/ppg_ear_125hz.npy", np.arange(2500, dtype=np.float32))
    np.save(root / "005/ecg_90fps.npy", np.sin(np.arange(1800, dtype=np.float32) / 10))
    pd.DataFrame([{
        "Video Emotion": "Neutral", "Neutral": 1.0,
        "Valence": 0.5, "Arousal": 0.4, "Dominance": 0.6,
    }]).to_csv(root / "005/Session_A_005.csv", index=False)
    return root
