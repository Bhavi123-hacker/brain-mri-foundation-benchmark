"""
Unit test for Experiment Matrix Configuration Validator.
"""

from utils.config_parser import load_config
from experiments.validator import validate_experiment_matrix


def test_valid_experiment_matrix():
    """Verify that current benchmark configurations pass validation."""
    config = load_config("configs")
    is_valid, errors = validate_experiment_matrix(config)
    assert is_valid is True, f"Validation errors: {errors}"
    assert len(errors) == 0


def test_invalid_experiment_matrix():
    """Verify validator catches broken configuration."""
    broken_config = {
        "models": {},
        "datasets": {},
        "tasks": {},
        "benchmark": {"seeds": [], "label_budgets": [1.5]}
    }
    is_valid, errors = validate_experiment_matrix(broken_config)
    assert is_valid is False
    assert len(errors) > 0
