import pytest
import numpy as np
from src.data.segments import SegmentExtractor

@pytest.fixture
def extractor(ego_data_dir):
    return SegmentExtractor(
        data_dir=str(ego_data_dir),
        task_times_path=str(ego_data_dir / "task_times.npy"),
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


def test_load_eye_tracking_segment_inceptiontime_gaze_only(extractor, monkeypatch):
    gaze = np.arange(20, dtype=np.float32).reshape(10, 2)

    def fake_load_signal(subject_id, filename, start_90hz, end_90hz):
        assert filename == "gaze_90fps.npy"
        return gaze[start_90hz:end_90hz]

    monkeypatch.setattr(extractor, "load_signal", fake_load_signal)
    eye = extractor.load_eye_tracking_segment(
        "005",
        2,
        7,
        encoder_name="inceptiontime",
    )
    assert eye.ndim == 2
    assert eye.shape[1] == 2
    np.testing.assert_array_equal(eye, gaze[2:7])


def test_load_eye_tracking_segment_patchtst_includes_pupils(extractor, monkeypatch):
    gaze = np.arange(20, dtype=np.float32).reshape(10, 2)
    pupils = np.arange(10, 30, dtype=np.float32).reshape(10, 2)

    def fake_load_signal(subject_id, filename, start_90hz, end_90hz):
        if filename == "gaze_90fps.npy":
            return gaze[start_90hz:end_90hz]
        if filename == "pupils_90fps.npy":
            return pupils[start_90hz:end_90hz]
        raise AssertionError(f"Unexpected filename {filename}")

    monkeypatch.setattr(extractor, "load_signal", fake_load_signal)
    eye = extractor.load_eye_tracking_segment(
        "005",
        1,
        6,
        encoder_name="patchtst",
    )
    assert eye.ndim == 2
    assert eye.shape[1] == 4
    np.testing.assert_array_equal(
        eye,
        np.concatenate([gaze[1:6], pupils[1:6]], axis=1),
    )

def test_get_segment_data_ppg(extractor):
    segments = extractor.get_segments("005")
    seg = segments[0]
    ppg = extractor.load_ppg_segment("005", seg["start_idx"], seg["end_idx"])
    assert ppg.ndim == 2
    assert ppg.shape[1] == 1


from src.data.segments import LabelLoader
import pandas as pd

@pytest.fixture
def label_loader(tmp_path):
    manifests = {
        "ce_hardlabel_manifests": {"subject": "005", "emotion": "Neutral", "label": 4},
        "kl_softlabel_manifests": {e: float(e == "Neutral") for e in LabelLoader.EMOTIONS},
        "vad_binary_quadrant_manifests": {
            "valence_score": 0.5, "arousal_score": 0.4, "dominance_score": 0.6,
        },
    }
    for directory, row in manifests.items():
        path = tmp_path / directory
        path.mkdir()
        pd.DataFrame([row]).to_csv(path / "dataset_manifest.csv", index=False)
    return LabelLoader(data_dir=str(tmp_path))

def test_load_hard_labels(label_loader):
    manifest = label_loader.load_ce_manifest()
    assert len(manifest) > 0
    assert "subject" in manifest.columns
    assert "emotion" in manifest.columns
    assert "label" in manifest.columns
    assert manifest.loc[0, "label"] == 4
    assert label_loader.load_ce_manifest() is manifest

def test_load_soft_labels(label_loader):
    manifest = label_loader.load_kl_manifest()
    emotions = ["Amused", "Content", "Excited", "Awe", "Neutral", "Fear", "Sad", "Disgust", "Anger"]
    for e in emotions:
        assert e in manifest.columns
    assert manifest[emotions].iloc[0].sum() == 1.0
    assert manifest.loc[0, "Neutral"] == 1.0

def test_load_vad_labels(label_loader):
    manifest = label_loader.load_vad_manifest()
    assert "valence_score" in manifest.columns
    assert "arousal_score" in manifest.columns
    assert "dominance_score" in manifest.columns
    assert manifest.iloc[0].tolist() == [0.5, 0.4, 0.6]


from src.data.label_builder import build_label_mapping

def test_build_label_mapping(ego_data_dir):
    mapping = build_label_mapping(
        data_dir=str(ego_data_dir),
        task_times_path=str(ego_data_dir / "task_times.npy"),
    )
    assert len(mapping) > 0
    sample_key = list(mapping.keys())[0]
    assert "_" in sample_key
    sample = mapping[sample_key]
    assert "emotion_label" in sample
    assert "soft_label" in sample
    assert "vad" in sample
    assert sample["soft_label"].shape == (9,)
    assert sample["vad"].shape == (3,)
