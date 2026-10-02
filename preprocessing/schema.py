"""
Unified Dataset Schema and Scan Normalization Specifications.
Defines common dataclass and DataFrame schema for ADNI, OASIS-3, and external cohorts.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import pandas as pd


COMMON_SCHEMA_COLUMNS = [
    "subject_id",
    "scan_id",
    "file_path",
    "dataset",
    "diagnosis",
    "age",
    "sex",
    "visit",
    "modality",
    "acquisition_info",
    "is_valid_scan",
    "exclusion_reason",
]


@dataclass
class NormalizedScanMetadata:
    """
    Standardized schema representation of a single brain MRI volume.
    """
    subject_id: str
    scan_id: str
    file_path: str
    dataset: str
    diagnosis: Optional[str] = None  # "CN", "MCI", "AD"
    age: Optional[float] = None
    sex: Optional[str] = None  # "M", "F"
    visit: Optional[str] = None
    modality: str = "T1w"
    acquisition_info: Dict[str, Any] = field(default_factory=dict)
    is_valid_scan: bool = True
    exclusion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["acquisition_info"] = str(d["acquisition_info"])
        return d


def validate_dataframe_schema(df: pd.DataFrame) -> List[str]:
    """
    Validate that a parsed metadata DataFrame conforms to the common internal schema.
    Returns list of schema violation error messages.
    """
    errors = []
    required_cols = ["subject_id", "scan_id", "file_path", "dataset", "is_valid_scan"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required schema column: '{col}'")

    if "diagnosis" in df.columns:
        valid_dx = {"CN", "MCI", "AD", None}
        actual_dx = set(df["diagnosis"].dropna().unique())
        invalid_dx = actual_dx - {"CN", "MCI", "AD"}
        if invalid_dx:
            errors.append(f"Invalid diagnosis labels found: {invalid_dx}. Allowed: CN, MCI, AD.")

    if "age" in df.columns:
        valid_ages = df["age"].dropna()
        if (valid_ages < 0).any() or (valid_ages > 125).any():
            errors.append("Age values outside realistic human range [0, 125] detected.")

    return errors
