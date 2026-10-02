"""
Unit tests for ADNI Metadata Parser and Schema Normalization.
"""

import os
import pandas as pd
import pytest
from preprocessing.adni_parser import ADNIMetadataParser
from preprocessing.schema import validate_dataframe_schema


def test_adni_normalization():
    """Verify ADNI subject ID, diagnosis, and demographic normalization."""
    parser = ADNIMetadataParser()

    assert parser.normalize_subject_id("002_S_0295") == "ADNI_002_S_0295"
    assert parser.normalize_subject_id("ADNI_002_S_0295") == "ADNI_002_S_0295"
    assert parser.normalize_subject_id("123-S-4567") == "ADNI_123_S_4567"

    assert parser.normalize_diagnosis("CN") == "CN"
    assert parser.normalize_diagnosis("NL") == "CN"
    assert parser.normalize_diagnosis("LMCI") == "MCI"
    assert parser.normalize_diagnosis("EMCI") == "MCI"
    assert parser.normalize_diagnosis("AD") == "AD"
    assert parser.normalize_diagnosis("Dementia") == "AD"
    assert parser.normalize_diagnosis("Unknown_dx") is None

    assert parser.normalize_sex("Male") == "M"
    assert parser.normalize_sex("F") == "F"


def test_adni_parse_dataframe(tmp_path):
    """Verify parsing a simulated ADNI metadata table."""
    raw_df = pd.DataFrame([
        {"PTID": "002_S_0295", "IMAGEUID": "I1234", "DX": "CN", "AGE": 75.4, "PTGENDER": "Male", "VISCODE": "bl", "FILE_PATH": "/nonexistent/path1.nii"},
        {"PTID": "002_S_0413", "IMAGEUID": "I5678", "DX": "LMCI", "AGE": 68.2, "PTGENDER": "Female", "VISCODE": "m12", "FILE_PATH": "/nonexistent/path2.nii"},
        {"PTID": "002_S_0999", "IMAGEUID": "I9999", "DX": "AD", "AGE": 81.0, "PTGENDER": "M", "VISCODE": "bl", "FILE_PATH": "/nonexistent/path3.nii"},
    ])

    parser = ADNIMetadataParser()
    norm_df = parser.parse(metadata_df=raw_df)

    assert len(norm_df) == 3
    assert norm_df["subject_id"].tolist() == ["ADNI_002_S_0295", "ADNI_002_S_0413", "ADNI_002_S_0999"]
    assert norm_df["diagnosis"].tolist() == ["CN", "MCI", "AD"]
    assert norm_df["sex"].tolist() == ["M", "F", "M"]
    assert norm_df["dataset"].tolist() == ["ADNI", "ADNI", "ADNI"]

    # Files do not exist, so is_valid_scan must be False
    assert (norm_df["is_valid_scan"] == False).all()
    assert (norm_df["exclusion_reason"].str.contains("not found")).all()

    # Verify no schema errors
    errors = validate_dataframe_schema(norm_df)
    assert len(errors) == 0
