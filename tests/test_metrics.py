"""
Unit test for Medical Evaluation Metrics.
"""

import numpy as np
import torch
import pytest
from evaluation.metrics import compute_classification_metrics, compute_regression_metrics


def test_classification_metrics_perfect_prediction():
    """Verify metrics for perfect classification."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.95, 0.90])

    metrics = compute_classification_metrics(y_true, y_prob)

    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["brier_score"] < 0.05


def test_classification_metrics_torch_tensors():
    """Verify tensor compatibility."""
    y_true = torch.tensor([0, 1, 0, 1])
    y_prob = torch.tensor([0.2, 0.8, 0.3, 0.7])

    metrics = compute_classification_metrics(y_true, y_prob)
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


def test_regression_metrics():
    """Verify continuous regression metrics."""
    y_true = np.array([60.0, 70.0, 80.0, 90.0])
    y_pred = np.array([61.0, 69.0, 82.0, 88.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert np.isclose(metrics["mae"], 1.5, atol=1e-2)
    assert metrics["r2"] > 0.95
    assert metrics["pearson_r"] > 0.95
    assert "rmse" in metrics
