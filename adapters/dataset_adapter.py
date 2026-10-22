"""
adapters/dataset_adapter.py
Extensible Dataset Adapter Framework for SHIFT-MRI Benchmark.

Provides a unified contract for neuroimaging dataset ingestion, metadata schema normalization,
clinical-label alignment, quality control gating, and external evaluation eligibility.
"""

import abc
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


class BaseDatasetAdapter(abc.ABC):
    """Abstract Base Class for Neuroimaging Dataset Adapters."""

    def __init__(self, dataset_name: str, root_path: Path):
        self.dataset_name = dataset_name
        self.root_path = Path(root_path)

    @abc.abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Returns cohort-level metadata, scanner details, DUA terms, and sample counts."""
        pass

    @abc.abstractmethod
    def list_subjects(self) -> List[Dict[str, Any]]:
        """Indexes all eligible subjects/sessions with standardized schema."""
        pass

    @abc.abstractmethod
    def get_clinical_label_mapping(self) -> Dict[str, Any]:
        """Defines explicit clinical diagnostic mapping (e.g. CDR vs NINCDS-ADRDA)."""
        pass

    @abc.abstractmethod
    def check_data_availability(self) -> bool:
        """Verifies whether authorized local imaging files are present on disk."""
        pass


class OASIS1Adapter(BaseDatasetAdapter):
    """Adapter for the OASIS-1 Cross-Sectional MRI Cohort (N=198)."""

    def __init__(self, root_path: Path = Path("D:/brain_mri_data/OASIS1")):
        super().__init__("OASIS-1", root_path)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "OASIS-1",
            "full_title": "Open Access Series of Imaging Studies (Cross-Sectional)",
            "benchmark_role": "Primary In-Domain Training, Validation & Locked Test",
            "total_participants": 198,
            "splits": {"train": 138, "val": 30, "test": 30},
            "scanners": ["Siemens Vision 1.5T"],
            "resolution": "1.0 x 1.0 x 1.25 mm (resampled to 128^3 isotropic)",
            "data_use_terms": "Open access for scientific and educational use",
            "is_authorized": True
        }

    def get_clinical_label_mapping(self) -> Dict[str, Any]:
        return {
            "standard": "Clinical Dementia Rating (CDR)",
            "negative_class": "CDR 0.0 (Cognitively Normal)",
            "positive_class": "CDR >= 0.5 (Very Mild to Moderate Cognitive Impairment)",
            "prevalence": 0.50
        }

    def check_data_availability(self) -> bool:
        manifest = self.root_path / "manifests" / "oasis1_primary_cohort_manifest.csv"
        return manifest.exists()

    def list_subjects(self) -> List[Dict[str, Any]]:
        manifest = self.root_path / "manifests" / "oasis1_primary_cohort_manifest.csv"
        if not manifest.exists():
            return []
        df = pd.read_csv(manifest)
        return df.to_dict(orient="records")


class ADNIAdapter(BaseDatasetAdapter):
    """Adapter for the ADNI Multi-Site External Generalization Cohort (N=500)."""

    def __init__(self, root_path: Path = Path("D:/brain_mri_data/ADNI")):
        super().__init__("ADNI", root_path)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "ADNI",
            "full_title": "Alzheimer's Disease Neuroimaging Initiative",
            "benchmark_role": "Zero-Shot Multi-Site External Generalization Cohort",
            "total_participants": 500,
            "classes": {"CN": 303, "AD": 197},
            "prevalence": 0.394,
            "scanners": ["Siemens (N=370, 1.5T & 3.0T)", "Philips (N=130, 3.0T)"],
            "data_use_terms": "Restricted DUA (Research use only; redistribution prohibited)",
            "is_authorized": True
        }

    def get_clinical_label_mapping(self) -> Dict[str, Any]:
        return {
            "standard": "NINCDS-ADRDA Consensus Diagnostic Criteria",
            "negative_class": "Cognitively Normal (CN)",
            "positive_class": "Clinically Confirmed Alzheimer's Disease (AD)",
            "prevalence": 0.394
        }

    def check_data_availability(self) -> bool:
        manifest = self.root_path / "manifests" / "adni_full_preprocessed_manifest.csv"
        return manifest.exists()

    def list_subjects(self) -> List[Dict[str, Any]]:
        manifest = self.root_path / "manifests" / "adni_full_preprocessed_manifest.csv"
        if not manifest.exists():
            return []
        df = pd.read_csv(manifest)
        return df.to_dict(orient="records")


class OASIS2Adapter(BaseDatasetAdapter):
    """Adapter for the OASIS-2 Longitudinal Cohort (N=150, Exploratory)."""

    def __init__(self, root_path: Path = Path("D:/brain_mri_data/OASIS2")):
        super().__init__("OASIS-2", root_path)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": "OASIS-2",
            "full_title": "Longitudinal MRI Study of Nondemented and Demented Older Adults",
            "benchmark_role": "Secondary Exploratory Cohort (Unverified Overlap)",
            "total_participants": 150,
            "classes": {"CN": 85, "AD": 65},
            "prevalence": 0.433,
            "scanners": ["Siemens Vision 1.5T"],
            "data_use_terms": "Open access for scientific research",
            "is_authorized": True
        }

    def get_clinical_label_mapping(self) -> Dict[str, Any]:
        return {
            "standard": "Clinical Dementia Rating (CDR)",
            "negative_class": "Nondemented (CDR 0)",
            "positive_class": "Demented (CDR >= 0.5)",
            "prevalence": 0.433
        }

    def check_data_availability(self) -> bool:
        manifest = self.root_path / "manifests" / "oasis2_baseline_manifest.csv"
        return manifest.exists()

    def list_subjects(self) -> List[Dict[str, Any]]:
        manifest = self.root_path / "manifests" / "oasis2_baseline_manifest.csv"
        if not manifest.exists():
            return []
        df = pd.read_csv(manifest)
        return df.to_dict(orient="records")


class FutureDatasetAdapterTemplate(BaseDatasetAdapter):
    """
    Template adapter for prospective future cohorts (e.g., AIBL, MIRIAD, OASIS-3).
    Honoring research integrity: datasets not yet locally authorized return availability=False.
    """

    def __init__(self, dataset_name: str = "AIBL", root_path: Path = Path("D:/brain_mri_data/AIBL")):
        super().__init__(dataset_name, root_path)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "benchmark_role": "Prospective External Cohort (Pending Data Access Agreement)",
            "data_use_terms": "Formal DUA Application Required",
            "is_authorized": False,
            "status": "PROSPECTIVE_STUDY_PROPOSED"
        }

    def get_clinical_label_mapping(self) -> Dict[str, Any]:
        return {
            "standard": "Compatible Diagnostic Criteria Required",
            "status": "PENDING_ACQUISITION"
        }

    def check_data_availability(self) -> bool:
        return self.root_path.exists() and len(list(self.root_path.glob("*.csv"))) > 0

    def list_subjects(self) -> List[Dict[str, Any]]:
        return []
