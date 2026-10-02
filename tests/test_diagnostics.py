"""
Unit tests for Dataset Diagnostics and Quality Auditing.
"""

import os
import pandas as pd
import pytest
from preprocessing.diagnostics import DatasetDiagnostics
from preprocessing.mock_data import generate_mock_dataset


@pytest.fixture(scope="module")
def sample_dataset(tmp_path_factory):
    tmp_dir = str(tmp_path_factory.mktemp("diag_data"))
    csv_path = generate_mock_dataset(
        output_dir=tmp_dir,
        num_patients=12,
        scans_per_patient=2,
        volume_shape=(16, 16, 16),
        seed=42
    )
    df = pd.read_csv(csv_path)
    df["subject_id"] = df["PTID"]
    df["scan_id"] = df["IMAGEUID"]
    df["file_path"] = df["FILE_PATH"]
    df["diagnosis"] = df["DX"]
    df["age"] = df["AGE"]
    df["sex"] = df["PTGENDER"]
    df["dataset"] = "MOCK"
    df["is_valid_scan"] = True
    df["exclusion_reason"] = None
    return df


def test_clinical_report(sample_dataset):
    """Verify statistical report computation."""
    diag = DatasetDiagnostics(sample_dataset, dataset_name="MOCK_TEST")
    report = diag.generate_clinical_report()

    assert report["dataset_name"] == "MOCK_TEST"
    assert report["total_records"] == 24
    assert report["total_subjects"] == 12
    assert report["usable_scans"] == 24
    assert report["age_statistics"]["mean"] > 50.0
    assert report["task_feasibility"]["Task_1_AD_vs_CN"]["is_feasible"] is True


def test_image_header_diagnostics(sample_dataset):
    """Verify NIfTI header inspection."""
    diag = DatasetDiagnostics(sample_dataset, dataset_name="MOCK_TEST")
    header_report = diag.run_image_header_diagnostics(max_samples=4)

    assert header_report["inspected_files_count"] == 4
    assert header_report["corrupt_files_count"] == 0
    assert len(header_report["detected_shapes"]) > 0
    assert header_report["detected_shapes"][0] == [16, 16, 16]
