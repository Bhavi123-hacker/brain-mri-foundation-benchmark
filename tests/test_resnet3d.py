"""
Unit tests for 3D ResNet-18, ResNet-34, and ResNet-50 Baseline Architectures.
"""

import torch
import pytest
from models.baseline.resnet3d import (
    build_resnet18_3d,
    build_resnet34_3d,
    build_resnet50_3d,
    BasicBlock3D,
    Bottleneck3D,
)


def test_resnet18_forward_and_backward():
    """Verify ResNet-18 forward pass, output shape, and backward gradient flow."""
    model = build_resnet18_3d(output_dim=1, task_type="classification", in_channels=1)
    
    # Input tensor: batch_size=2, channels=1, D=32, H=32, W=32
    x = torch.randn(2, 1, 32, 32, 32, requires_grad=True)
    out = model(x)

    assert out.shape == (2, 1)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    param_counts = model.get_parameter_count()
    assert param_counts["total_parameters"] > 30_000_000 # ~33M params
    assert param_counts["trainable_parameters"] == param_counts["total_parameters"]


def test_resnet18_feature_extraction():
    """Verify feature extractor output dimension."""
    model = build_resnet18_3d(output_dim=1, task_type="regression")
    x = torch.randn(2, 1, 32, 32, 32)
    feats = model.extract_features(x)
    assert feats.shape == (2, 512)


def test_resnet34_and_resnet50_initialization():
    """Verify ResNet-34 and ResNet-50 instantiate and forward properly."""
    model34 = build_resnet34_3d(output_dim=1, task_type="classification")
    model50 = build_resnet50_3d(output_dim=1, task_type="classification")

    x = torch.randn(2, 1, 32, 32, 32)
    out34 = model34(x)
    out50 = model50(x)

    assert out34.shape == (2, 1)
    assert out50.shape == (2, 1)

    assert model34.get_parameter_count()["total_parameters"] > 60_000_000
    assert model50.get_parameter_count()["total_parameters"] > 40_000_000
