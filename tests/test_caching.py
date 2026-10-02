"""
Unit tests for 3D Preprocessed Tensor Caching.
"""

import os
import torch
import pytest
from preprocessing.cache_manager import PreprocessedCacheManager


def test_cache_manager_operations(tmp_path):
    """Verify caching, retrieval, float16 conversion, and size tracking."""
    cache_dir = str(tmp_path / "tensor_cache")
    manager = PreprocessedCacheManager(cache_dir=cache_dir)

    dummy_path = "/path/to/subject_scan.nii.gz"
    dummy_tensor = torch.randn(1, 32, 32, 32, dtype=torch.float32)

    assert manager.has(dummy_path) is False
    assert manager.get(dummy_path) is None

    # Put tensor
    saved_path = manager.put(dummy_path, dummy_tensor, transform_id="resampled_128", dtype=torch.float16)
    assert os.path.exists(saved_path)
    assert manager.has(dummy_path, transform_id="resampled_128") is True

    # Retrieve tensor
    retrieved = manager.get(dummy_path, transform_id="resampled_128")
    assert retrieved is not None
    assert retrieved.shape == (1, 32, 32, 32)
    assert retrieved.dtype == torch.float16
    assert torch.allclose(retrieved.float(), dummy_tensor, atol=1e-2)

    # Size check
    size_mb = manager.get_cache_size_mb()
    assert size_mb > 0.0
