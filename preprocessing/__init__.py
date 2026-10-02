"""Preprocessing, Data Ingestion, Diagnostics, and Splitting Package."""
from preprocessing.schema import NormalizedScanMetadata, COMMON_SCHEMA_COLUMNS, validate_dataframe_schema
from preprocessing.transforms import get_preprocessing_transforms
from preprocessing.pipeline import create_patient_level_splits, save_splits, BrainMRIDataset
from preprocessing.mock_data import generate_mock_dataset, create_3d_brain_phantom
from preprocessing.adni_parser import ADNIMetadataParser
from preprocessing.oasis_parser import OASIS3MetadataParser
from preprocessing.diagnostics import DatasetDiagnostics
from preprocessing.split_manager import PersistentSplitManager
from preprocessing.cache_manager import PreprocessedCacheManager
from preprocessing.dataloader import create_benchmark_dataloaders

__all__ = [
    "NormalizedScanMetadata",
    "COMMON_SCHEMA_COLUMNS",
    "validate_dataframe_schema",
    "get_preprocessing_transforms",
    "create_patient_level_splits",
    "save_splits",
    "BrainMRIDataset",
    "generate_mock_dataset",
    "create_3d_brain_phantom",
    "ADNIMetadataParser",
    "OASIS3MetadataParser",
    "DatasetDiagnostics",
    "PersistentSplitManager",
    "PreprocessedCacheManager",
    "create_benchmark_dataloaders",
]
