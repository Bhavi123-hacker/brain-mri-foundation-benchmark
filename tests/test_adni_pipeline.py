"""
Integration tests for ADNI benchmark manifest, NIfTI loading, disk caching, and PyTorch Dataset integration.
"""

import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import pytest

from preprocessing.pipeline import BrainMRIDataset
from preprocessing.cache_manager import PreprocessedCacheManager

MANIFEST_CSV = "results/adni_cn_ad_benchmark_manifest.csv"
MANIFEST_JSON = "results/adni_cn_ad_benchmark_manifest.json"
CACHE_DIR = "data/cache/adni_cn_ad"
AMBIGUOUS_CSV = "results/adni_ambiguous_subjects.csv"


def test_adni_benchmark_manifest_integrity():
    """Verify ADNI benchmark manifest has exactly 50 subjects, balanced classes, and no ambiguous subjects."""
    assert os.path.exists(MANIFEST_CSV), f"Missing {MANIFEST_CSV}"
    assert os.path.exists(MANIFEST_JSON), f"Missing {MANIFEST_JSON}"

    df = pd.read_csv(MANIFEST_CSV)
    assert len(df) == 50, f"Expected 50 subjects, got {len(df)}"

    required_cols = [
        "subject_id",
        "nifti_path",
        "label",
        "baseline_diagnosis",
        "manufacturer",
        "field_strength",
        "voxel_spacing",
        "shape",
        "inclusion_status",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Class balance check
    cn_count = (df["baseline_diagnosis"] == "CN").sum()
    ad_count = (df["baseline_diagnosis"] == "AD").sum()
    assert cn_count == 28, f"Expected 28 CN, got {cn_count}"
    assert ad_count == 22, f"Expected 22 AD, got {ad_count}"

    label_cn = (df["label"] == 0.0).sum()
    label_ad = (df["label"] == 1.0).sum()
    assert label_cn == 28 and label_ad == 22

    # Exclusion check: ensure none of the 9 ambiguous subjects leaked in
    if os.path.exists(AMBIGUOUS_CSV):
        amb_df = pd.read_csv(AMBIGUOUS_CSV)
        amb_subjects = set(amb_df["subject_id"])
        manifest_subjects = set(df["subject_id"])
        overlap = amb_subjects.intersection(manifest_subjects)
        assert len(overlap) == 0, f"Ambiguous subjects found in manifest: {overlap}"


def test_adni_nifti_files_exist_and_valid():
    """Verify all NIfTI files exist on disk and have valid 3D geometry."""
    df = pd.read_csv(MANIFEST_CSV)
    sample_path = df.iloc[0]["nifti_path"]
    if not os.path.exists(sample_path):
        pytest.skip(f"Historical raw ADNI path {sample_path} not present on current system.")

    for _, row in df.iterrows():
        path = row["nifti_path"]
        assert os.path.exists(path), f"NIfTI missing on disk: {path}"

    nii = nib.load(sample_path)
    assert len(nii.shape) == 3, f"Expected 3D shape, got {nii.shape}"
    assert nii.affine is not None


def test_adni_preprocessed_cache_integrity():
    """Verify cached .npy and .pt tensors exist, have correct shape (1, 128, 128, 128) and no NaNs."""
    if not os.path.exists(CACHE_DIR):
        pytest.skip(f"Legacy cache directory {CACHE_DIR} not present on current system.")

    npy_files = glob.glob(os.path.join(CACHE_DIR, "*.npy"))
    pt_files = glob.glob(os.path.join(CACHE_DIR, "*.pt"))

    if len(npy_files) == 0:
        pytest.skip(f"Legacy cache files not present in {CACHE_DIR} on current system.")

    assert len(npy_files) == 50, f"Expected 50 .npy files, got {len(npy_files)}"
    assert len(pt_files) == 50, f"Expected 50 .pt files, got {len(pt_files)}"

    # Load and test sample .npy
    sample_npy = np.load(npy_files[0])
    assert sample_npy.shape == (1, 128, 128, 128)
    assert sample_npy.dtype == np.float32
    assert not np.isnan(sample_npy).any()
    assert not np.isinf(sample_npy).any()
    assert sample_npy.std() > 0

    # Test PreprocessedCacheManager retrieval
    cache_manager = PreprocessedCacheManager(cache_dir=CACHE_DIR)
    df = pd.read_csv(MANIFEST_CSV)
    sample_nifti = df.iloc[0]["nifti_path"]
    cached_tensor = cache_manager.get(sample_nifti, transform_id="adni_zscore_128")
    assert cached_tensor is not None
    assert tuple(cached_tensor.shape) == (1, 128, 128, 128)


def test_adni_dataset_integration():
    """Verify BrainMRIDataset loads from ADNI manifest and cache without errors."""
    df = pd.read_csv(MANIFEST_CSV)
    sample_path = df.iloc[0]["nifti_path"]
    npy_files = glob.glob(os.path.join(CACHE_DIR, "*.npy")) if os.path.exists(CACHE_DIR) else []
    if not os.path.exists(sample_path) and len(npy_files) == 0:
        pytest.skip("Legacy ADNI 50-subject NIfTI and cache files not present on current system.")

    cache_manager = PreprocessedCacheManager(cache_dir=CACHE_DIR)

    dataset = BrainMRIDataset(
        metadata_df=df,
        target_column="label",
        task_type="classification",
        positive_class=1.0,
        negative_class=0.0,
        file_path_col="nifti_path",
        cache_manager=cache_manager,
        transform_id="adni_zscore_128",
    )

    assert len(dataset) == 50
    volume, target, meta = dataset[0]

    assert isinstance(volume, torch.Tensor)
    assert tuple(volume.shape) == (1, 128, 128, 128)
    assert volume.dtype == torch.float32
    assert isinstance(target, torch.Tensor)
    assert target.shape == torch.Size([1])
    assert target.item() in [0.0, 1.0]
    assert "file_path" in meta
