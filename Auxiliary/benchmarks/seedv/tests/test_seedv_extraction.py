from pathlib import Path

import torch

from Auxiliary.benchmarks.seedv.scripts.extract_seedv_embeddings import _owned_cpu_embedding


def test_owned_cpu_embedding_breaks_shared_storage_before_save(tmp_path: Path):
    batch = torch.randn(1823, 2048)
    row_view = batch[0]

    assert row_view.untyped_storage().nbytes() > (
        row_view.numel() * row_view.element_size()
    )

    owned = _owned_cpu_embedding(row_view)

    assert owned.shape == (2048,)
    assert owned.device.type == "cpu"
    assert owned.untyped_storage().nbytes() == (
        owned.numel() * owned.element_size()
    )

    out_path = tmp_path / "embedding.pt"
    torch.save({"embedding": owned}, out_path)

    reloaded = torch.load(out_path, weights_only=False)["embedding"]
    assert reloaded.untyped_storage().nbytes() == (
        reloaded.numel() * reloaded.element_size()
    )
    assert out_path.stat().st_size < 20_000
