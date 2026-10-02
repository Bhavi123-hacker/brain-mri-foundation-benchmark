"""
Unit test for Dataset loading, synthetic data generation, and BrainMRIDataset class.
"""

import os
import torch
import pandas as pd
import pytest
from preprocessing.mock_data import generate_mock_dataset
from preprocessing.pipeline import BrainMRIDataset
from preprocessing.transforms import get_preprocessing_transforms


@pytest.fixture(scope="module")
def mock_dataset_csv(tmp_path_factory):
    """Fixture to generate a temporary synthetic mock dataset."""
    tmp_dir = str(tmp_path_factory.mktemp("mock_data"))
    csv_path = generate_mock_dataset(
        output_dir=tmp_dir,
        num_patients=4,
        scans_per_patient=2,
        volume_shape=(16, 16, 16),
        seed=42
    )
    return csv_path


def test_mock_data_generation(mock_dataset_csv):
    """Verify synthetic dataset creates CSV with expected columns."""
    assert os.path.exists(mock_dataset_csv)
    df = pd.read_csv(mock_dataset_csv)
    assert len(df) == 8 # 4 patients * 2 scans
    assert "PTID" in df.columns
    assert "IMAGEUID" in df.columns
    assert "DX" in df.columns
    assert "AGE" in df.columns
    assert "FILE_PATH" in df.columns
    assert bool(df["IS_SYNTHETIC"].all()) is True


def test_brain_mri_dataset_classification(mock_dataset_csv):
    """Verify BrainMRIDataset returns tensors and binary targets for classification."""
    df = pd.read_csv(mock_dataset_csv)
    transform = get_preprocessing_transforms(spatial_shape=(16, 16, 16), voxel_spacing=None)
    
    # Filter to CN and AD for test
    df_binary = df[df["DX"].isin(["CN", "AD"])].copy()
    if len(df_binary) == 0:
        df_binary = df.copy()
        df_binary["DX"] = ["CN", "AD"] * (len(df_binary) // 2)

    dataset = BrainMRIDataset(
        metadata_df=df_binary,
        target_column="DX",
        task_type="classification",
        positive_class="AD",
        negative_class="CN",
        transform=transform
    )

    assert len(dataset) > 0
    vol, target, meta = dataset[0]
    assert isinstance(vol, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (1,)
    assert target.item() in [0.0, 1.0]
    assert vol.ndim == 4 # (C, D, H, W)
    assert vol.shape[1:] == (16, 16, 16)


def test_brain_mri_dataset_regression(mock_dataset_csv):
    """Verify BrainMRIDataset returns continuous targets for brain-age regression."""
    df = pd.read_csv(mock_dataset_csv)
    transform = get_preprocessing_transforms(spatial_shape=(16, 16, 16), voxel_spacing=None)

    dataset = BrainMRIDataset(
        metadata_df=df,
        target_column="AGE",
        task_type="regression",
        transform=transform
    )

    assert len(dataset) == len(df)
    vol, target, meta = dataset[0]
    assert isinstance(vol, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert target.item() > 0.0
