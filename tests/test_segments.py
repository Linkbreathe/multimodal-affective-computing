import pytest
import numpy as np
from src.data.segments import SegmentExtractor

@pytest.fixture
def extractor():
    return SegmentExtractor(
        data_dir="data/egoemotion_raw",
        task_times_path="data/egoemotion_raw/task_times.npy",
    )

def test_loads_task_times(extractor):
    assert len(extractor.task_times) > 0
    assert "005" in extractor.task_times

def test_get_subject_ids(extractor):
    subjects = extractor.get_subject_ids()
    assert len(subjects) == 40
    assert "005" in subjects
    assert "020" not in subjects

def test_get_segments_for_subject(extractor):
    segments = extractor.get_segments("005")
    assert len(segments) > 0
    seg = segments[0]
    assert "task_name" in seg
    assert "start_idx" in seg
    assert "end_idx" in seg
    assert "subject_id" in seg
    assert seg["end_idx"] > seg["start_idx"]

def test_get_segment_data_gaze(extractor):
    segments = extractor.get_segments("005")
    seg = segments[0]
    gaze = extractor.load_signal("005", "gaze_90fps.npy", seg["start_idx"], seg["end_idx"])
    assert gaze.ndim == 2
    assert gaze.shape[1] == 2

def test_get_segment_data_ppg(extractor):
    segments = extractor.get_segments("005")
    seg = segments[0]
    ppg = extractor.load_ppg_segment("005", seg["start_idx"], seg["end_idx"])
    assert ppg.ndim == 2
    assert ppg.shape[1] == 1


from src.data.segments import LabelLoader
import pandas as pd

@pytest.fixture
def label_loader():
    return LabelLoader(data_dir="data/egoemotion_raw")

def test_load_hard_labels(label_loader):
    manifest = label_loader.load_ce_manifest()
    assert len(manifest) > 0
    assert "subject" in manifest.columns
    assert "emotion" in manifest.columns
    assert "label" in manifest.columns

def test_load_soft_labels(label_loader):
    manifest = label_loader.load_kl_manifest()
    emotions = ["Amused", "Content", "Excited", "Awe", "Neutral", "Fear", "Sad", "Disgust", "Anger"]
    for e in emotions:
        assert e in manifest.columns

def test_load_vad_labels(label_loader):
    manifest = label_loader.load_vad_manifest()
    assert "valence_score" in manifest.columns
    assert "arousal_score" in manifest.columns
    assert "dominance_score" in manifest.columns
