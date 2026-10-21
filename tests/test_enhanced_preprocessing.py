import pytest
import torch
import numpy as np
from preprocessing.enhanced_pipeline import (
    RobustPercentileZScore,
    AnatomicalROIExtractor,
    get_v2_preprocessing_pipeline,
    ScanQualityControlEngine
)

def test_robust_percentile_zscore():
    dummy = torch.randn(1, 64, 64, 64).abs()
    dummy[:, :10, :, :] = 0.0
    normer = RobustPercentileZScore(0.5, 99.5)
    out = normer(dummy)
    assert out.shape == dummy.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_anatomical_roi_extractor():
    dummy = torch.randn(1, 128, 128, 128).abs()
    extractor = AnatomicalROIExtractor()
    rois = extractor(dummy)
    assert "left_hippo" in rois
    assert "right_hippo" in rois
    assert "ventricles" in rois
    assert rois["left_hippo"].shape == (1, 30, 30, 22)
    assert rois["right_hippo"].shape == (1, 30, 30, 22)
    assert rois["ventricles"].shape == (1, 35, 40, 32)

    biomarkers = extractor.compute_morphometric_biomarkers(dummy.numpy())
    assert "ventricle_to_brain_ratio" in biomarkers
    assert "hippocampus_ratio" in biomarkers
    assert biomarkers["ventricle_to_brain_ratio"] >= 0.0

def test_v2_pipelines_shape_and_range():
    dummy = torch.randn(1, 128, 128, 128).abs()
    for p_type in ["A", "B", "C", "D"]:
        pipe = get_v2_preprocessing_pipeline(p_type, is_training=False)
        out = pipe(dummy)
        assert out.shape == (1, 128, 128, 128)
        assert not torch.isnan(out).any()

def test_scan_qc_engine():
    dummy = torch.randn(1, 128, 128, 128).abs()
    qc = ScanQualityControlEngine()
    res = qc.validate_tensor(dummy, "SUBJ_001")
    assert res["status"] == "PASS"
    assert res["foreground_ratio"] > 0.05
