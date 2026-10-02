"""
ADNI (Alzheimer's Disease Neuroimaging Initiative) Metadata Parser and Ingestion Engine.
Normalizes ADNI tabular records and MRI file paths into the common internal schema.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from preprocessing.schema import NormalizedScanMetadata, COMMON_SCHEMA_COLUMNS


class ADNIMetadataParser:
    """
    Parser for ADNI clinical tables and NIfTI file directory structures.
    Supports ADNIMERGE.csv, DXSUM, or standard LONI image search exports.
    """
    DIAGNOSIS_MAP = {
        "CN": "CN",
        "NL": "CN",
        "CONTROL": "CN",
        "NORMAL": "CN",
        "COGNITIVELY NORMAL": "CN",
        "MCI": "MCI",
        "LMCI": "MCI",
        "EMCI": "MCI",
        "SMC": "MCI",
        "MILD COGNITIVE IMPAIRMENT": "MCI",
        "AD": "AD",
        "DEMENTIA": "AD",
        "ALZHEIMER": "AD",
        "ALZHEIMER'S DISEASE": "AD",
    }

    def __init__(self, root_dir: Optional[str] = None, metadata_path: Optional[str] = None):
        self.root_dir = root_dir or os.getenv("ADNI_DATA_DIR", "data/adni")
        self.metadata_path = metadata_path or os.getenv("ADNI_METADATA_PATH", os.path.join(self.root_dir, "adni_metadata.csv"))

    def normalize_subject_id(self, raw_id: Any) -> str:
        """Standardize ADNI subject ID to ADNI_XXX_S_XXXX."""
        clean_id = str(raw_id).strip().replace("-", "_")
        if not clean_id.startswith("ADNI_"):
            clean_id = f"ADNI_{clean_id}"
        return clean_id

    def normalize_diagnosis(self, raw_dx: Any) -> Optional[str]:
        """Standardize diagnosis to CN, MCI, or AD."""
        if pd.isna(raw_dx) or raw_dx is None:
            return None
        dx_str = str(raw_dx).strip().upper()
        return self.DIAGNOSIS_MAP.get(dx_str, None)

    def normalize_sex(self, raw_sex: Any) -> Optional[str]:
        """Standardize gender to M or F."""
        if pd.isna(raw_sex) or raw_sex is None:
            return None
        sex_str = str(raw_sex).strip().upper()
        if sex_str in ["M", "MALE", "1"]:
            return "M"
        elif sex_str in ["F", "FEMALE", "2"]:
            return "F"
        return None

    def discover_nifti_files(self, scan_dir: str) -> Dict[str, str]:
        """
        Recursively discover all .nii and .nii.gz files in scan_dir.
        Returns a mapping from filename/stem to absolute file path.
        """
        discovered = {}
        if not os.path.exists(scan_dir):
            return discovered

        for ext in ["*.nii", "*.nii.gz"]:
            pattern = os.path.join(scan_dir, "**", ext)
            for path in glob.glob(pattern, recursive=True):
                abs_path = os.path.abspath(path)
                filename = os.path.basename(path)
                discovered[filename] = abs_path
                # Also index stem without extensions
                stem = filename.replace(".nii.gz", "").replace(".nii", "")
                discovered[stem] = abs_path
        return discovered

    def parse(self, metadata_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Parse metadata and return normalized DataFrame conforming to COMMON_SCHEMA_COLUMNS.
        """
        if metadata_df is None:
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(
                    f"ADNI metadata file not found at: {self.metadata_path}. "
                    "Ensure ADNI_DATA_DIR or ADNI_METADATA_PATH environment variable is configured."
                )
            metadata_df = pd.read_csv(self.metadata_path)

        discovered_files = self.discover_nifti_files(self.root_dir)
        records = []

        # Find appropriate column names from common ADNI schemas
        cols = {c.upper(): c for c in metadata_df.columns}
        ptid_col = cols.get("PTID") or cols.get("SUBJECT_ID") or cols.get("RID") or metadata_df.columns[0]
        dx_col = cols.get("DX") or cols.get("DX_BL") or cols.get("DXCURREN") or cols.get("DIAGNOSIS")
        age_col = cols.get("AGE") or cols.get("AGE_BL") or cols.get("EXAMAGE")
        sex_col = cols.get("PTGENDER") or cols.get("GENDER") or cols.get("SEX")
        visit_col = cols.get("VISCODE") or cols.get("VISIT") or cols.get("VISCODE2")
        path_col = cols.get("FILE_PATH") or cols.get("PATH") or cols.get("IMAGEUID") or cols.get("FILENAME")
        img_id_col = cols.get("IMAGEUID") or cols.get("IMAGE_ID") or cols.get("SCAN_ID")

        for idx, row in metadata_df.iterrows():
            raw_ptid = row.get(ptid_col, f"SUBJ_{idx:04d}")
            subject_id = self.normalize_subject_id(raw_ptid)
            scan_id = str(row.get(img_id_col, f"SCAN_{idx:04d}"))
            raw_dx = row.get(dx_col) if dx_col else None
            diagnosis = self.normalize_diagnosis(raw_dx)
            
            # Age
            raw_age = row.get(age_col) if age_col else None
            try:
                age = float(raw_age) if pd.notna(raw_age) else None
            except (ValueError, TypeError):
                age = None

            # Sex & Visit
            sex = self.normalize_sex(row.get(sex_col)) if sex_col else None
            visit = str(row.get(visit_col, "m00")) if visit_col and pd.notna(row.get(visit_col)) else "m00"

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

            # If path not found from column, search in discovered files
            if not file_path:
                for fname, fpath in discovered_files.items():
                    if str(raw_ptid) in fname and str(visit) in fname:
                        file_path = fpath
                        break

            # Eligibility & Exclusion Checks
            is_valid = True
            exclusion_reasons = []

            if not file_path or not os.path.exists(file_path):
                is_valid = False
                exclusion_reasons.append("MRI file not found on disk")

            if diagnosis is None:
                is_valid = False
                exclusion_reasons.append(f"Unmapped or missing diagnosis ('{raw_dx}')")

            if age is None or age < 40 or age > 110:
                is_valid = False
                exclusion_reasons.append(f"Invalid age ({age})")

            rec = NormalizedScanMetadata(
                subject_id=subject_id,
                scan_id=scan_id,
                file_path=file_path,
                dataset="ADNI",
                diagnosis=diagnosis,
                age=age,
                sex=sex,
                visit=visit,
                modality="T1w",
                acquisition_info={"raw_diagnosis": str(raw_dx), "raw_visit": str(visit)},
                is_valid_scan=is_valid,
                exclusion_reason="; ".join(exclusion_reasons) if exclusion_reasons else None
            )
            records.append(rec.to_dict())

        normalized_df = pd.DataFrame(records)
        return normalized_df
