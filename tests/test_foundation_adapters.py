"""
Unit and Integration Tests for Foundation Model Adapters (BrainIAC, FOMO25, MONAI Swin UNETR SSL).
Verifies tensor dimensions, freeze/unfreeze mechanisms, gradient flow, and checkpoint persistence.
"""

import os
import torch
import pytest

from models import build_model
from models.foundation_brainiac.adapter import BrainIACAdapter
from models.foundation_fomo25.adapter import FOMO25Adapter
from models.foundation_swin.adapter import SwinUNETRSSLAdapter


def test_brainiac_adapter_linear_probe_and_finetune(tmp_path):
    """Test BrainIAC linear probing, freezing, gradient flow, and fine-tuning."""
    # 1. Linear Probe Mode
    model = BrainIACAdapter(output_dim=1, task_type="classification", adaptation_mode="linear_probe")
    x = torch.randn(2, 1, 32, 32, 32)
    
    # Verify parameter counts and frozen state
    counts = model.get_parameter_count()
    assert counts["trainable_parameters"] < counts["total_parameters"]
    assert counts["frozen_parameters"] > 0

    # Forward pass
    out = model(x)
    assert out.shape == (2, 1)

    # Backward pass: only head should get gradients
    loss = out.sum()
    loss.backward()
    for name, param in model.patch_embed.named_parameters():
        assert param.grad is None or (param.grad == 0).all()
    for name, param in model.head.named_parameters():
        assert param.grad is not None

    # 2. Fine-tuning Mode (Unfreeze)
    model.unfreeze_backbone()
    counts_ft = model.get_parameter_count()
    assert counts_ft["trainable_parameters"] == counts_ft["total_parameters"]
    assert counts_ft["frozen_parameters"] == 0

    # 3. Checkpoint Save and Reload
    chk_path = str(tmp_path / "brainiac_test.pt")
    torch.save(model.state_dict(), chk_path)
    reloaded_model = BrainIACAdapter(output_dim=1, task_type="classification", weights_path=chk_path)
    assert reloaded_model is not None


def test_fomo25_adapter_linear_probe_and_finetune(tmp_path):
    """Test FOMO25 adapter forward, backward, freezing, and checkpointing."""
    model = FOMO25Adapter(output_dim=1, task_type="regression", adaptation_mode="linear_probe")
    x = torch.randn(2, 1, 32, 32, 32)

    counts = model.get_parameter_count()
    assert counts["trainable_parameters"] < counts["total_parameters"]

    out = model(x)
    assert out.shape == (2, 1)

    loss = out.sum()
    loss.backward()
    for name, param in model.head.named_parameters():
        assert param.grad is not None

    model.unfreeze_backbone()
    counts_ft = model.get_parameter_count()
    assert counts_ft["frozen_parameters"] == 0

    chk_path = str(tmp_path / "fomo25_test.pt")
    torch.save(model.state_dict(), chk_path)
    reloaded = FOMO25Adapter(output_dim=1, task_type="regression", weights_path=chk_path)
    assert reloaded is not None


def test_swin_unetr_adapter_linear_probe_and_finetune(tmp_path):
    """Test MONAI Swin UNETR SSL adapter forward, backward, and metadata."""
    model = SwinUNETRSSLAdapter(output_dim=1, task_type="classification", adaptation_mode="linear_probe")
    x = torch.randn(2, 1, 32, 32, 32)

    feats = model.extract_features(x)
    assert feats.shape == (2, 768)

    out = model(x)
    assert out.shape == (2, 1)

    loss = out.sum()
    loss.backward()
    for param in model.head.parameters():
        assert param.grad is not None

    meta = model.get_model_metadata()
    assert meta["is_foundation_model"] is True
    assert meta["model_name"] == "MONAI-SwinUNETR-SSL"

    chk_path = str(tmp_path / "swin_test.pt")
    torch.save(model.state_dict(), chk_path)
    reloaded = SwinUNETRSSLAdapter(output_dim=1, task_type="classification", weights_path=chk_path)
    assert reloaded is not None


def test_unified_model_factory_dispatch():
    """Verify build_model dispatches all foundation and baseline models."""
    models_to_test = [
        ("baseline_resnet18", "classification"),
        ("baseline_resnet34", "classification"),
        ("baseline_resnet50", "classification"),
        ("foundation_brainiac", "classification"),
        ("foundation_fomo25", "regression"),
        ("foundation_swin_unetr", "classification"),
    ]

    for model_key, task_type in models_to_test:
        model = build_model(model_key=model_key, task_type=task_type, output_dim=1)
        assert model is not None
        assert model.task_type == task_type
