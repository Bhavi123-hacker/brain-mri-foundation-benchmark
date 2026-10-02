"""
OASIS-3 Metadata Parser and External Dataset Ingestion Engine.
Normalizes OASIS-3 clinical tables (CDR, demographics) into the common internal schema.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from preprocessing.schema import NormalizedScanMetadata, COMMON_SCHEMA_COLUMNS


class OASIS3MetadataParser:
    """
    Parser for OASIS-3 longitudinal clinical tables and BIDS/XNAT NIfTI directories.
    Maps Clinical Dementia Rating (CDR) scores to standardized CN, MCI, and AD categories.
    """
    def __init__(self, root_dir: Optional[str] = None, metadata_path: Optional[str] = None):
        self.root_dir = root_dir or os.getenv("OASIS3_DATA_DIR", "data/oasis3")
        self.metadata_path = metadata_path or os.getenv("OASIS3_METADATA_PATH", os.path.join(self.root_dir, "oasis3_metadata.csv"))

    def normalize_subject_id(self, raw_id: Any) -> str:
        """Standardize OASIS-3 subject ID to OASIS3_OASXXXXX."""
        clean_id = str(raw_id).strip().replace("-", "_")
        if not clean_id.startswith("OASIS3_"):
            clean_id = f"OASIS3_{clean_id}"
        return clean_id

    def map_cdr_to_diagnosis(self, raw_cdr: Any) -> Optional[str]:
        """
        Map Clinical Dementia Rating (CDR) to diagnostic class:
        - CDR 0.0 -> CN (Cognitively Normal)
        - CDR 0.5 -> MCI (Very Mild Dementia / MCI equivalent)
        - CDR >= 1.0 -> AD (Alzheimer's Disease Dementia)
        """
        if pd.isna(raw_cdr) or raw_cdr is None:
            return None
        try:
            cdr_val = float(raw_cdr)
            if np.isclose(cdr_val, 0.0):
                return "CN"
            elif np.isclose(cdr_val, 0.5):
                return "MCI"
            elif cdr_val >= 1.0:
                return "AD"
            return None
        except (ValueError, TypeError):
            return None

    def normalize_sex(self, raw_sex: Any) -> Optional[str]:
        if pd.isna(raw_sex) or raw_sex is None:
            return None
        sex_str = str(raw_sex).strip().upper()
        if sex_str in ["M", "MALE", "1"]:
            return "M"
        elif sex_str in ["F", "FEMALE", "2"]:
            return "F"
        return None

    def discover_nifti_files(self, scan_dir: str) -> Dict[str, str]:
        """Discover NIfTI files in OASIS-3 directory structure."""
        discovered = {}
        if not os.path.exists(scan_dir):
            return discovered

        for ext in ["*.nii", "*.nii.gz"]:
            pattern = os.path.join(scan_dir, "**", ext)
            for path in glob.glob(pattern, recursive=True):
                abs_path = os.path.abspath(path)
                filename = os.path.basename(path)
                discovered[filename] = abs_path
                stem = filename.replace(".nii.gz", "").replace(".nii", "")
                discovered[stem] = abs_path
        return discovered

    def parse(self, metadata_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Parse OASIS-3 metadata and return normalized DataFrame.
        """
        if metadata_df is None:
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(
                    f"OASIS-3 metadata file not found at: {self.metadata_path}. "
                    "Ensure OASIS3_DATA_DIR or OASIS3_METADATA_PATH environment variable is configured."
                )
            metadata_df = pd.read_csv(self.metadata_path)

        discovered_files = self.discover_nifti_files(self.root_dir)
        records = []

        cols = {c.upper(): c for c in metadata_df.columns}
        subj_col = cols.get("OASISID") or cols.get("OASIS_ID") or cols.get("SUBJECT_ID") or metadata_df.columns[0]
        cdr_col = cols.get("CDR") or cols.get("CDRSB") or cols.get("CLINICAL_DEM")
        age_col = cols.get("AGE") or cols.get("AGE_AT_VISIT") or cols.get("EXAM_AGE")
        sex_col = cols.get("GENDER") or cols.get("SEX")
        visit_col = cols.get("VISIT") or cols.get("DAYS_TO_VISIT") or cols.get("SESSION")
        path_col = cols.get("FILE_PATH") or cols.get("PATH") or cols.get("SCAN_ID") or cols.get("FILENAME")
        scan_id_col = cols.get("SCAN_ID") or cols.get("MR_ID") or cols.get("EXPERIMENT_ID")

        for idx, row in metadata_df.iterrows():
            raw_subj = row.get(subj_col, f"OAS_{idx:04d}")
            subject_id = self.normalize_subject_id(raw_subj)
            scan_id = str(row.get(scan_id_col, f"OAS_SCAN_{idx:04d}"))
            
            raw_cdr = row.get(cdr_col) if cdr_col else None
            diagnosis = self.map_cdr_to_diagnosis(raw_cdr)

            # Age
            raw_age = row.get(age_col) if age_col else None
            try:
                age = float(raw_age) if pd.notna(raw_age) else None
            except (ValueError, TypeError):
                age = None

            sex = self.normalize_sex(row.get(sex_col)) if sex_col else None
            visit = str(row.get(visit_col, "d0000")) if visit_col and pd.notna(row.get(visit_col)) else "d0000"

            # Resolve file path
            file_path = ""
            if path_col and pd.notna(row.get(path_col)):
                candidate = str(row.get(path_col))
                if os.path.isabs(candidate) and os.path.exists(candidate):
                    file_path = candidate
                elif os.path.exists(os.path.join(self.root_dir, candidate)):
                    file_path = os.path.abspath(os.path.join(self.root_dir, candidate))
                elif candidate in discovered_files:
                    file_path = discovered_files[candidate]

            if not file_path:
                for fname, fpath in discovered_files.items():
                    if str(raw_subj) in fname:
                        file_path = fpath
                        break

            # Eligibility & Exclusion
            is_valid = True
            exclusion_reasons = []

            if not file_path or not os.path.exists(file_path):
                is_valid = False
                exclusion_reasons.append("MRI file not found on disk")

            if diagnosis is None:
                is_valid = False
                exclusion_reasons.append(f"Unmapped or missing CDR ({raw_cdr})")

            if age is None or age < 40 or age > 115:
                is_valid = False
                exclusion_reasons.append(f"Invalid age ({age})")

            rec = NormalizedScanMetadata(
                subject_id=subject_id,
                scan_id=scan_id,
                file_path=file_path,
                dataset="OASIS3",
                diagnosis=diagnosis,
                age=age,
                sex=sex,
                visit=visit,
                modality="T1w",
                acquisition_info={"raw_cdr": str(raw_cdr), "raw_visit": str(visit)},
                is_valid_scan=is_valid,
                exclusion_reason="; ".join(exclusion_reasons) if exclusion_reasons else None
            )
            records.append(rec.to_dict())

        return pd.DataFrame(records)
