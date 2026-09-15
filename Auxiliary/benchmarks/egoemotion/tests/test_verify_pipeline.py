from pathlib import Path


def test_verify_pipeline_does_not_hardcode_legacy_28_subject_counts():
    text = Path("Auxiliary/benchmarks/egoemotion/scripts/verify_pipeline.py").read_text()

    assert "6663" not in text
    assert "expected 28" not in text
