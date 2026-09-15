import pytest
from mac.config.simple import load_config, merge_configs, config_hash

def test_load_base_config():
    cfg = load_config("configs/egoemotion.yaml")
    assert "seed" in cfg
    assert "data_dir" in cfg
    assert "modalities" in cfg

def test_merge_overrides():
    base = {"seed": 42, "model": {"lr": 1e-4}}
    override = {"model": {"lr": 1e-3, "dropout": 0.1}}
    merged = merge_configs(base, override)
    assert merged["seed"] == 42
    assert merged["model"]["lr"] == 1e-3
    assert merged["model"]["dropout"] == 0.1

def test_config_hash_deterministic():
    cfg = {"a": 1, "b": {"c": 2}}
    h1 = config_hash(cfg)
    h2 = config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 12
