"""
Unit test for Configuration loading and environment variable overrides.
"""

import os
import pytest
from utils.config_parser import load_config, load_yaml, apply_env_overrides, validate_config_compatibility


def test_load_yaml_configs():
    """Verify that all YAML configuration files exist and parse successfully."""
    datasets_cfg = load_yaml("configs/datasets.yaml")
    models_cfg = load_yaml("configs/models.yaml")
    tasks_cfg = load_yaml("configs/tasks.yaml")
    benchmark_cfg = load_yaml("configs/benchmark.yaml")

    assert "adni" in datasets_cfg
    assert "oasis3" in datasets_cfg
    assert "mock_dataset" in datasets_cfg
    assert "baseline_resnet18" in models_cfg
    assert "foundation_brainiac" in models_cfg
    assert "AD_vs_CN" in tasks_cfg or "ad_vs_cn" in tasks_cfg
    assert "seeds" in benchmark_cfg
    assert "label_budgets" in benchmark_cfg


def test_load_aggregated_config():
    """Verify load_config returns full dictionary structure."""
    config = load_config("configs")
    assert "datasets" in config
    assert "models" in config
    assert "tasks" in config
    assert "benchmark" in config


def test_env_override(monkeypatch):
    """Verify environment variable overrides for secure data mounting."""
    monkeypatch.setenv("ADNI_DATA_DIR", "/custom/secure/adni/path")
    monkeypatch.setenv("OASIS3_DATA_DIR", "/custom/secure/oasis3/path")

    config = load_config("configs")
    assert config["datasets"]["adni"]["root_dir"] == "/custom/secure/adni/path"
    assert config["datasets"]["oasis3"]["root_dir"] == "/custom/secure/oasis3/path"


def test_validate_compatibility():
    """Verify model and task compatibility checking."""
    config = load_config("configs")
    assert validate_config_compatibility("baseline_resnet18", "ad_vs_cn", config) is True
    assert validate_config_compatibility("foundation_brainiac", "brain_age", config) is True
    
    with pytest.raises(ValueError):
        validate_config_compatibility("non_existent_model", "ad_vs_cn", config)
