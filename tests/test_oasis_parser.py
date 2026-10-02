"""
Unit tests for OASIS-3 Metadata Parser and CDR Mapping.
"""

import os
import pandas as pd
import pytest
from preprocessing.oasis_parser import OASIS3MetadataParser
from preprocessing.schema import validate_dataframe_schema


def test_oasis_cdr_mapping():
    """Verify CDR to diagnosis classification mapping."""
    parser = OASIS3MetadataParser()

    assert parser.map_cdr_to_diagnosis(0.0) == "CN"
    assert parser.map_cdr_to_diagnosis("0.0") == "CN"
    assert parser.map_cdr_to_diagnosis(0.5) == "MCI"
    assert parser.map_cdr_to_diagnosis("0.5") == "MCI"
    assert parser.map_cdr_to_diagnosis(1.0) == "AD"
    assert parser.map_cdr_to_diagnosis(2.0) == "AD"
    assert parser.map_cdr_to_diagnosis(3.0) == "AD"
    assert parser.map_cdr_to_diagnosis(None) is None


def test_oasis_normalization():
    """Verify OASIS-3 subject ID normalization."""
    parser = OASIS3MetadataParser()

    assert parser.normalize_subject_id("OAS30001") == "OASIS3_OAS30001"
    assert parser.normalize_subject_id("OASIS3_OAS30001") == "OASIS3_OAS30001"


def test_oasis_parse_dataframe():
    """Verify parsing simulated OASIS-3 clinical records."""
    raw_df = pd.DataFrame([
        {"OASISID": "OAS30001", "SCAN_ID": "MR1", "CDR": 0.0, "AGE": 65.4, "GENDER": "M", "VISIT": "d0000", "FILE_PATH": "/mock/p1.nii"},
        {"OASISID": "OAS30002", "SCAN_ID": "MR2", "CDR": 0.5, "AGE": 72.1, "GENDER": "F", "VISIT": "d0500", "FILE_PATH": "/mock/p2.nii"},
        {"OASISID": "OAS30003", "SCAN_ID": "MR3", "CDR": 1.0, "AGE": 84.5, "GENDER": "M", "VISIT": "d0000", "FILE_PATH": "/mock/p3.nii"},
    ])

    parser = OASIS3MetadataParser()
    norm_df = parser.parse(metadata_df=raw_df)

    assert len(norm_df) == 3
    assert norm_df["subject_id"].tolist() == ["OASIS3_OAS30001", "OASIS3_OAS30002", "OASIS3_OAS30003"]
    assert norm_df["diagnosis"].tolist() == ["CN", "MCI", "AD"]
    assert norm_df["dataset"].tolist() == ["OASIS3", "OASIS3", "OASIS3"]

    errors = validate_dataframe_schema(norm_df)
    assert len(errors) == 0
