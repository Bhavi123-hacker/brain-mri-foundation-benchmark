"""
Data Pipeline and Dataset Diagnostics Execution CLI.
Ingests, normalizes, audits, and splits real/mock neuroimaging datasets.
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, Any, Optional

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_parser import load_config
from utils.logging_utils import get_logger
from preprocessing.adni_parser import ADNIMetadataParser
from preprocessing.oasis_parser import OASIS3MetadataParser
from preprocessing.diagnostics import DatasetDiagnostics
from preprocessing.split_manager import PersistentSplitManager
from preprocessing.mock_data import generate_mock_dataset
from preprocessing.schema import validate_dataframe_schema


def run_pipeline(config_dir: str = "configs", output_dir: str = "results") -> Dict[str, Any]:
    logger = get_logger("DataPipelineRunner")
    logger.info("Starting Data Pipeline and Quality Audit...")

    config = load_config(config_dir)
    os.makedirs(output_dir, exist_ok=True)
    split_manager = PersistentSplitManager(splits_dir="data/splits")

    reports = {}

    # Check ADNI
    adni_cfg = config.get("datasets", {}).get("adni", {})
    adni_root = adni_cfg.get("root_dir", "data/adni")
    adni_meta = adni_cfg.get("metadata_path", "data/adni/adni_metadata.csv")

    if os.path.exists(adni_meta):
        logger.info(f"ADNI metadata discovered at: {adni_meta}. Ingesting real ADNI dataset...")
        adni_parser = ADNIMetadataParser(root_dir=adni_root, metadata_path=adni_meta)
        adni_df = adni_parser.parse()
        schema_errs = validate_dataframe_schema(adni_df)
        if schema_errs:
            logger.warning(f"ADNI schema warnings: {schema_errs}")

        adni_diag = DatasetDiagnostics(adni_df, dataset_name="ADNI")
        adni_report = adni_diag.generate_clinical_report()
        adni_header_report = adni_diag.run_image_header_diagnostics()
        adni_report.update({"image_diagnostics": adni_header_report})
        reports["ADNI"] = adni_report
    else:
        logger.info("Real ADNI metadata not present on local machine. (Can be mounted via ADNI_DATA_DIR / ADNI_METADATA_PATH)")

    # Check OASIS-3
    oasis_cfg = config.get("datasets", {}).get("oasis3", {})
    oasis_root = oasis_cfg.get("root_dir", "data/oasis3")
    oasis_meta = oasis_cfg.get("metadata_path", "data/oasis3/oasis3_metadata.csv")

    if os.path.exists(oasis_meta):
        logger.info(f"OASIS-3 metadata discovered at: {oasis_meta}. Ingesting real OASIS-3 dataset...")
        oasis_parser = OASIS3MetadataParser(root_dir=oasis_root, metadata_path=oasis_meta)
        oasis_df = oasis_parser.parse()
        oasis_diag = DatasetDiagnostics(oasis_df, dataset_name="OASIS3")
        oasis_report = oasis_diag.generate_clinical_report()
        oasis_header_report = oasis_diag.run_image_header_diagnostics()
        oasis_report.update({"image_diagnostics": oasis_header_report})
        reports["OASIS3"] = oasis_report
    else:
        logger.info("Real OASIS-3 metadata not present on local machine. (Can be mounted via OASIS3_DATA_DIR / OASIS3_METADATA_PATH)")

    # Generate synthetic mock data for verification if real datasets are not yet mounted
    logger.info("Generating and auditing synthetic verification cohort for end-to-end testing...")
    mock_meta_path = generate_mock_dataset(
        output_dir="data/mock_data",
        num_patients=12,
        scans_per_patient=2,
        volume_shape=(32, 32, 32),
        seed=42
    )
    mock_raw_df = pd.read_csv(mock_meta_path)
    # Parse mock data through ADNI parser
    mock_parser = ADNIMetadataParser(root_dir="data/mock_data", metadata_path=mock_meta_path)
    mock_df = mock_parser.parse(metadata_df=mock_raw_df)
    mock_diag = DatasetDiagnostics(mock_df, dataset_name="SYNTHETIC_MOCK")
    mock_report = mock_diag.generate_clinical_report()
    mock_header_report = mock_diag.run_image_header_diagnostics()
    mock_report.update({"image_diagnostics": mock_header_report})
    reports["SYNTHETIC_MOCK"] = mock_report

    # Create persistent splits for all 3 tasks on the verification cohort
    tasks_to_split = [
        ("AD_vs_CN", mock_df[mock_df["diagnosis"].isin(["CN", "AD"])]),
        ("Brain_Age", mock_df.dropna(subset=["age"])),
        ("MCI_vs_CN", mock_df[mock_df["diagnosis"].isin(["CN", "MCI"])]),
    ]

    split_manifests = {}
    for task_name, task_df in tasks_to_split:
        if len(task_df["subject_id"].unique()) >= 3:
            splits = split_manager.create_and_persist_splits(
                df=task_df,
                dataset_name="SYNTHETIC_MOCK",
                task_name=task_name,
                subject_col="subject_id",
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                seed=42,
            )
            leakage_check = split_manager.check_leakage(splits, subject_col="subject_id")
            split_manifests[task_name] = {
                "train_scans": len(splits["train"]),
                "val_scans": len(splits["val"]),
                "test_scans": len(splits["test"]),
                "leakage_detected": leakage_check["has_leakage"],
            }
            logger.info(f"Persistent splits created for {task_name}: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}, Leakage={leakage_check['has_leakage']}")

    # Save summary report JSON
    final_output = {
        "datasets_audited": list(reports.keys()),
        "reports": reports,
        "split_manifests": split_manifests,
    }

    report_json_path = os.path.join(output_dir, "data_diagnostics_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"Dataset diagnostics report successfully saved to: {report_json_path}")
    return final_output


if __name__ == "__main__":
    run_pipeline()
