from scripts.run_seedv_ablation_probe import CONDITIONS


def test_ablation_probe_uses_current_extraction_layout():
    assert CONDITIONS == {
        "4s_all": "data/embeddings/seedv/base/4s_all",
        "10s_all": "data/embeddings/seedv/base/10s_all",
        "4s_tp": "data/embeddings/seedv/base/4s_tp",
        "10s_tp": "data/embeddings/seedv/base/10s_tp",
    }
