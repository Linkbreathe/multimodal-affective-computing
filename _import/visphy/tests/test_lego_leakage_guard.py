import pytest

from scripts.run_lego_experiment import require_lego_pretrain_leakage_acknowledgement


def test_lego_experiment_requires_explicit_global_pretrain_acknowledgement():
    with pytest.raises(RuntimeError, match="global supervised pretraining"):
        require_lego_pretrain_leakage_acknowledgement(False)

    require_lego_pretrain_leakage_acknowledgement(True)
