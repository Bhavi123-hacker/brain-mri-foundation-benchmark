import pytest
import torch
from models.v2_models import ResNet10_3D, DenseNet121_3D, MultiView25D_ResNet18, DualBranchHybridModel

def test_resnet10_3d():
    model = ResNet10_3D(output_dim=1)
    params = model.get_parameter_count()
    assert params["total_parameters"] > 10_000_000
    assert params["trainable_parameters"] == params["total_parameters"]

    x = torch.randn(2, 1, 64, 64, 64)
    feat = model.extract_features(x)
    assert feat.shape == (2, 512)
    out = model(x)
    assert out.shape == (2, 1)

def test_densenet121_3d():
    model = DenseNet121_3D(output_dim=1)
    params = model.get_parameter_count()
    assert 10_000_000 < params["total_parameters"] < 15_000_000

    x = torch.randn(2, 1, 64, 64, 64)
    feat = model.extract_features(x)
    assert feat.shape == (2, 512)
    out = model(x)
    assert out.shape == (2, 1)

def test_multiview_25d_resnet18():
    model = MultiView25D_ResNet18(pretrained=False, output_dim=1)
    params = model.get_parameter_count()
    assert params["total_parameters"] > 11_000_000

    x = torch.randn(2, 1, 64, 64, 64)
    feat = model.extract_features(x)
    assert feat.shape == (2, 256)
    out = model(x)
    assert out.shape == (2, 1)

def test_dual_branch_hybrid():
    model = DualBranchHybridModel(spatial_backbone="resnet10", num_radiomics=6, output_dim=1)
    x = torch.randn(2, 1, 64, 64, 64)
    rad = torch.randn(2, 6)
    out = model(x, rad)
    assert out.shape == (2, 1)
