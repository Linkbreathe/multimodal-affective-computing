import pytest
from src.trainer.early_stopping import EarlyStopping

def test_early_stopping_improves():
    es = EarlyStopping(patience=3, mode="max")
    assert not es.step(0.5)
    assert not es.step(0.6)
    assert not es.step(0.7)
    assert es.best_score == 0.7

def test_early_stopping_triggers():
    es = EarlyStopping(patience=3, mode="max")
    es.step(0.5)
    es.step(0.4)
    assert not es.step(0.3)
    assert es.step(0.3)

def test_early_stopping_min_mode():
    es = EarlyStopping(patience=3, mode="min")
    es.step(0.5)
    es.step(0.4)
    es.step(0.5)
    assert not es.step(0.6)
    assert es.step(0.7)
