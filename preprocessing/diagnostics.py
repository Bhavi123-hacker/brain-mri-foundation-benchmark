"""
Dataset Diagnostics, NIfTI Header Inspection, and Data Quality Auditing Engine.
"""

import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, Any, List, Optional, Tuple


class DatasetDiagnostics:
    """
    Computes rigorous clinical data diagnostics, NIfTI image integrity checks,
    missing-file tracking, and task feasibility summaries.
    """
    def __init__(self, normalized_df: pd.DataFrame, dataset_name: str = "Dataset"):
        self.df = normalized_df.copy()
        self.dataset_name = dataset_name

    def run_image_header_diagnostics(self, max_samples: int = 50) -> Dict[str, Any]:
        """
        Inspect physical NIfTI files to extract spatial shape, voxel spacing, orientation,
        and detect corrupt or unreadable volumes.
        """
        valid_scans = self.df[self.df["file_path"].apply(lambda p: os.path.exists(str(p)))]
        sample_subset = valid_scans.head(max_samples)

        shapes = []
        spacings = []
        orientations = []
        corrupt_files = []

        for _, row in sample_subset.iterrows():
            fpath = row["file_path"]
            try:
                img = nib.load(fpath)
                header = img.header
                shapes.append(list(img.shape))
                spacings.append([round(float(s), 3) for s in header.get_zooms()[:3]])
                
                # Orientation
                axcodes = nib.aff2axcodes(img.affine)
                orientations.append("".join(axcodes))
            except Exception as e:
                corrupt_files.append({"file_path": fpath, "error": str(e)})

        # Aggregate summary
        unique_shapes = list({tuple(s) for s in shapes})
        unique_spacings = list({tuple(s) for s in spacings})
        unique_orientations = list(set(orientations))

        return {
            "inspected_files_count": len(sample_subset),
            "corrupt_files_count": len(corrupt_files),
            "corrupt_files": corrupt_files,
            "detected_shapes": [list(s) for s in unique_shapes],
            "detected_spacings_mm": [list(s) for s in unique_spacings],
            "detected_orientations": unique_orientations,
        }

    def generate_clinical_report(self) -> Dict[str, Any]:
        """
        Generate complete statistical breakdown of subjects, scans, labels, and exclusions.
        """
        total_records = len(self.df)
        total_subjects = self.df["subject_id"].nunique() if total_records > 0 else 0

        # Usable scans
        usable_mask = self.df["is_valid_scan"] == True
        usable_df = self.df[usable_mask]
        excluded_df = self.df[~usable_mask]

        usable_scans = len(usable_df)
        usable_subjects = usable_df["subject_id"].nunique() if usable_scans > 0 else 0

        # Diagnosis counts
        dx_counts = self.df["diagnosis"].value_counts(dropna=False).to_dict()
        usable_dx_counts = usable_df["diagnosis"].value_counts(dropna=False).to_dict()

        # Subject-level diagnosis (using earliest/majority diagnosis per subject)
        subj_dx = usable_df.groupby("subject_id")["diagnosis"].first().value_counts().to_dict() if usable_scans > 0 else {}

        # Age statistics
        valid_ages = usable_df["age"].dropna()
        age_stats = {
            "mean": round(float(valid_ages.mean()), 2) if len(valid_ages) > 0 else 0.0,
            "std": round(float(valid_ages.std()), 2) if len(valid_ages) > 0 else 0.0,
            "min": round(float(valid_ages.min()), 2) if len(valid_ages) > 0 else 0.0,
            "max": round(float(valid_ages.max()), 2) if len(valid_ages) > 0 else 0.0,
            "missing_count": int(self.df["age"].isna().sum()),
        }

        # Sex counts
        sex_counts = usable_df["sex"].value_counts(dropna=False).to_dict()

        # Duplicate scan / subject checks
        duplicate_scans = int(self.df.duplicated(subset=["scan_id"]).sum())
        duplicate_files = int(self.df.duplicated(subset=["file_path"]).sum())

        # Exclusions breakdown
        exclusions_breakdown = excluded_df["exclusion_reason"].value_counts().to_dict() if len(excluded_df) > 0 else {}

        # Task Feasibility
        task1_ad_cn_samples = len(usable_df[usable_df["diagnosis"].isin(["CN", "AD"])])
        task1_ad_cn_subjects = usable_df[usable_df["diagnosis"].isin(["CN", "AD"])]["subject_id"].nunique()

        task2_brain_age_samples = len(usable_df.dropna(subset=["age"]))
        task2_brain_age_subjects = usable_df.dropna(subset=["age"])["subject_id"].nunique()

        task3_mci_cn_samples = len(usable_df[usable_df["diagnosis"].isin(["CN", "MCI"])])
        task3_mci_cn_subjects = usable_df[usable_df["diagnosis"].isin(["CN", "MCI"])]["subject_id"].nunique()

        return {
            "dataset_name": self.dataset_name,
            "total_records": total_records,
            "total_subjects": total_subjects,
            "usable_scans": usable_scans,
            "usable_subjects": usable_subjects,
            "excluded_scans": len(excluded_df),
            "duplicate_scans_detected": duplicate_scans,
            "duplicate_file_paths": duplicate_files,
            "subject_level_diagnoses": {
                "CN": int(subj_dx.get("CN", 0)),
                "MCI": int(subj_dx.get("MCI", 0)),
                "AD": int(subj_dx.get("AD", 0)),
            },
            "scan_level_diagnoses": {
                "CN": int(usable_dx_counts.get("CN", 0)),
                "MCI": int(usable_dx_counts.get("MCI", 0)),
                "AD": int(usable_dx_counts.get("AD", 0)),
            },
            "age_statistics": age_stats,
            "sex_distribution": {str(k): int(v) for k, v in sex_counts.items()},
            "exclusions_breakdown": {str(k): int(v) for k, v in exclusions_breakdown.items()},
            "task_feasibility": {
                "Task_1_AD_vs_CN": {
                    "usable_scans": task1_ad_cn_samples,
                    "usable_subjects": task1_ad_cn_subjects,
                    "is_feasible": task1_ad_cn_subjects >= 6,
                },
                "Task_2_Brain_Age": {
                    "usable_scans": task2_brain_age_samples,
                    "usable_subjects": task2_brain_age_subjects,
                    "is_feasible": task2_brain_age_subjects >= 6,
                },
                "Task_3_MCI_vs_CN": {
                    "usable_scans": task3_mci_cn_samples,
                    "usable_subjects": task3_mci_cn_subjects,
                    "is_feasible": task3_mci_cn_subjects >= 6,
                },
            }
        }
