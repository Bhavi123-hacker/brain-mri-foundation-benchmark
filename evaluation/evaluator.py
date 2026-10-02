"""
Standardized Evaluation Runner across Benchmark Test Partitions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from evaluation.metrics import compute_classification_metrics, compute_regression_metrics
from evaluation.profiler import measure_inference_latency


def evaluate_model_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    task_type: str = "classification",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Standard evaluation protocol executed uniformly across baseline and foundation models.
    """
    model.eval()
    model.to(device)

    all_outputs = []
    all_targets = []
    sample_input = None

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            if sample_input is None:
                sample_input = inputs[:1].clone()
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    if not all_outputs:
        return {"error": "Empty test set"}

    outputs_cat = torch.cat(all_outputs, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    if task_type == "classification":
        probs = torch.sigmoid(outputs_cat)
        metrics = compute_classification_metrics(targets_cat, probs)
    elif task_type == "regression":
        metrics = compute_regression_metrics(targets_cat, outputs_cat)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Latency profiling
    if sample_input is not None:
        latency_info = measure_inference_latency(model, sample_input, device=device)
        metrics.update(latency_info)

    return metrics
