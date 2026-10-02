"""
Experiment Matrix and Configuration Validation Suite.
"""

from typing import Dict, Any, List, Tuple


def validate_experiment_matrix(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check the validity of configured models, datasets, tasks, budgets, and seeds.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    models = config.get("models", {})
    datasets = config.get("datasets", {})
    tasks = config.get("tasks", {})
    benchmark = config.get("benchmark", {})

    if not models:
        errors.append("No models defined in configuration.")
    if not datasets:
        errors.append("No datasets defined in configuration.")
    if not tasks:
        errors.append("No tasks defined in configuration.")
    if not benchmark:
        errors.append("No benchmark settings defined in configuration.")

    # Validate Seeds
    seeds = benchmark.get("seeds", [])
    if not seeds or not isinstance(seeds, list):
        errors.append("Benchmark must define a list of integer random seeds.")

    # Validate Budgets
    budgets = benchmark.get("label_budgets", [])
    if not budgets or not isinstance(budgets, list):
        errors.append("Benchmark must define a list of label budget fractions.")
    for b in budgets:
        if b <= 0.0 or b > 1.0:
            errors.append(f"Invalid label budget fraction: {b}. Must be in (0.0, 1.0].")

    # Validate Model Configurations
    for m_key, m_val in models.items():
        if "architecture" not in m_val:
            errors.append(f"Model '{m_key}' is missing 'architecture' field.")
        if "input_shape" not in m_val:
            errors.append(f"Model '{m_key}' is missing 'input_shape' field.")

    # Validate Task Configurations
    for t_key, t_val in tasks.items():
        if "task_type" not in t_val:
            errors.append(f"Task '{t_key}' is missing 'task_type' field.")
        if "target_column" not in t_val:
            errors.append(f"Task '{t_key}' is missing 'target_column' field.")

    is_valid = len(errors) == 0
    return is_valid, errors
