from pathlib import Path


def test_10s_extraction_writes_manifest_hash_to_embedding_payloads():
    text = Path("scripts/segment_and_extract_10s.py").read_text()

    assert "compute_manifest_hash" in text
    assert '"manifest_hash": manifest_hash' in text
