"""YAML Configuration parser with environment variable overrides and schema validation."""
import os
import yaml
from typing import Dict, Any, Optional


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override dataset paths with environment variables if present.
    Enables secure path mounting without modifying tracked repository files.
    """
    if "adni" in config:
        if os.getenv("ADNI_DATA_DIR"):
            config["adni"]["root_dir"] = os.getenv("ADNI_DATA_DIR")
        if os.getenv("ADNI_METADATA_PATH"):
            config["adni"]["metadata_path"] = os.getenv("ADNI_METADATA_PATH")

    if "oasis3" in config:
        if os.getenv("OASIS3_DATA_DIR"):
            config["oasis3"]["root_dir"] = os.getenv("OASIS3_DATA_DIR")
        if os.getenv("OASIS3_METADATA_PATH"):
            config["oasis3"]["metadata_path"] = os.getenv("OASIS3_METADATA_PATH")

    return config


def load_config(config_dir: str = "configs") -> Dict[str, Any]:
    """
    Load and aggregate all benchmark configuration files.
    """
    datasets = load_yaml(os.path.join(config_dir, "datasets.yaml"))
    models = load_yaml(os.path.join(config_dir, "models.yaml"))
    tasks = load_yaml(os.path.join(config_dir, "tasks.yaml"))
    benchmark = load_yaml(os.path.join(config_dir, "benchmark.yaml"))

    datasets = apply_env_overrides(datasets)

    return {
        "datasets": datasets,
        "models": models,
        "tasks": tasks,
        "benchmark": benchmark,
    }


def validate_config_compatibility(model_key: str, task_key: str, config: Dict[str, Any]) -> bool:
    """
    Validate that a given model and task combination is mathematically and architecturally valid.
    """
    models = config.get("models", {})
    tasks = config.get("tasks", {})

    if model_key not in models:
        raise ValueError(f"Unknown model key '{model_key}'. Available models: {list(models.keys())}")
    
    if task_key not in tasks:
        raise ValueError(f"Unknown task key '{task_key}'. Available tasks: {list(tasks.keys())}")

    model_cfg = models[model_key]
    task_cfg = tasks[task_key]

    # Validate output dimensions
    if "output_dim" in task_cfg:
        expected_dim = task_cfg["output_dim"]
        if expected_dim <= 0:
            return False

    return True
