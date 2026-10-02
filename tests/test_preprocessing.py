"""
Unit test for 3D Preprocessing Transformations and Tensor Reshaping.
"""

import os
import torch
import numpy as np
import nibabel as nib
import pytest
from preprocessing.transforms import get_preprocessing_transforms
from preprocessing.mock_data import create_3d_brain_phantom


@pytest.fixture
def temp_nifti_file(tmp_path):
    """Create a temporary NIfTI file for transformation testing."""
    file_path = str(tmp_path / "test_brain.nii.gz")
    vol_data = create_3d_brain_phantom(shape=(32, 32, 32), seed=123)
    affine = np.eye(4)
    nii = nib.Nifti1Image(vol_data, affine)
    nib.save(nii, file_path)
    return file_path


def test_preprocessing_transforms_pipeline(temp_nifti_file):
    """Verify that preprocessing transform produces expected tensor shape and values."""
    target_shape = (64, 64, 64)
    transform = get_preprocessing_transforms(
        spatial_shape=target_shape,
        voxel_spacing=None,
        intensity_norm="z_score",
        is_training=False
    )

    tensor = transform(temp_nifti_file)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 4 # (C, D, H, W)
    assert tensor.shape == (1, 64, 64, 64)
    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()


def test_training_augmentations(temp_nifti_file):
    """Verify training transforms with 3D augmentations run without error."""
    target_shape = (32, 32, 32)
    transform = get_preprocessing_transforms(
        spatial_shape=target_shape,
        voxel_spacing=None,
        intensity_norm="min_max",
        is_training=True
    )

    tensor = transform(temp_nifti_file)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 32, 32, 32)
