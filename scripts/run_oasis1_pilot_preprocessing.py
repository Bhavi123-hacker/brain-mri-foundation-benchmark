"""
Phase 5B Real Data Preprocessing Validation & Audit Script.
Processes 50 selected OASIS-1 pilot subjects from D:\\OASIS1_PILOT using the project's
unified MONAI/PyTorch preprocessing pipeline and PreprocessedCacheManager.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from preprocessing.transforms import get_preprocessing_transforms
from preprocessing.pipeline import BrainMRIDataset
from preprocessing.cache_manager import PreprocessedCacheManager
from preprocessing.dataloader import create_benchmark_dataloaders


def run_oasis1_pilot_preprocessing(data_dir: str = "D:\\OASIS1_PILOT", output_dir: str = "results"):
    logger = get_logger("OASIS1PilotPreprocessing")
    logger.info("=================================================================")
    logger.info("Starting Phase 5B Real-Data Preprocessing Validation (OASIS-1 Pilot)")
    logger.info(f"Source Directory: {data_dir}")
    logger.info("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, "pilot_manifest.csv")
    splits_path = os.path.join(data_dir, "pilot_splits.csv")

    assert os.path.exists(manifest_path), f"Manifest file missing: {manifest_path}"
    assert os.path.exists(splits_path), f"Splits file missing: {splits_path}"

    manifest_df = pd.read_csv(manifest_path)
    splits_df = pd.read_csv(splits_path)

    logger.info(f"Loaded pilot_manifest.csv: {len(manifest_df)} subjects")
    logger.info(f"Loaded pilot_splits.csv: {len(splits_df)} subjects")

    # 1. Resolve full file paths & check exclusion rules
    full_paths = []
    missing_files = []
    invalid_files = []
    raw_shapes = {}
    raw_dtypes = set()
    raw_stats_before = {"min": [], "max": [], "mean": [], "std": []}

    excluded_files = []

    for idx, row in splits_df.iterrows():
        subject_id = row["ID"]
        filename = row["mri_file"]
        file_path = os.path.join(data_dir, filename)

        # Exclusion check: OAS1_0061_MR2_t1.npy must NOT be used
        if "OAS1_0061_MR2" in filename or "OAS1_0061_MR2" in subject_id:
            excluded_files.append(filename)
            logger.warning(f"Excluding forbidden file: {filename}")
            continue

        if not os.path.exists(file_path):
            missing_files.append(file_path)
            logger.error(f"Missing file: {file_path}")
            continue

        # Inspect raw numpy array
        try:
            arr = np.load(file_path)
            shape_tuple = tuple(arr.shape)
            raw_shapes[str(shape_tuple)] = raw_shapes.get(str(shape_tuple), 0) + 1
            raw_dtypes.add(str(arr.dtype))

            # Integrity checks: NaN, Inf, empty/constant check
            if np.isnan(arr).any() or np.isinf(arr).any():
                invalid_files.append({"file": file_path, "reason": "Contains NaN/Inf"})
            elif arr.max() == arr.min() or np.std(arr) == 0:
                invalid_files.append({"file": file_path, "reason": "Empty/constant volume"})
            else:
                raw_stats_before["min"].append(float(arr.min()))
                raw_stats_before["max"].append(float(arr.max()))
                raw_stats_before["mean"].append(float(arr.mean()))
                raw_stats_before["std"].append(float(arr.std()))

        except Exception as e:
            invalid_files.append({"file": file_path, "reason": f"Failed to load numpy array: {str(e)}"})

        full_paths.append(file_path)

    splits_df["file_path"] = full_paths

    # 2. Patient-level split isolation check
    train_df = splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True)

    train_pts = set(train_df["ID"].unique())
    val_pts = set(val_df["ID"].unique())
    test_pts = set(test_df["ID"].unique())

    train_val_overlap = train_pts.intersection(val_pts)
    train_test_overlap = train_pts.intersection(test_pts)
    val_test_overlap = val_pts.intersection(test_pts)

    has_leakage = len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0
    assert not has_leakage, "LEAKAGE DETECTED across patient splits!"

    logger.info("Patient-level split validation: 0% Patient Leakage Verified")
    logger.info(f"Split breakdown -> Train: {len(train_df)} ({len(train_pts)} pts), Val: {len(val_df)} ({len(val_pts)} pts), Test: {len(test_df)} ({len(test_pts)} pts)")

    # 3. Preprocessing Execution & Disk Caching
    cache_dir = os.path.join("data", "cache", "oasis1_pilot")
    cache_manager = PreprocessedCacheManager(cache_dir=cache_dir)
    eval_transform = get_preprocessing_transforms(
        spatial_shape=(128, 128, 128),
        voxel_spacing=(1.0, 1.0, 1.0),
        intensity_norm="z_score",
        is_training=False,
    )

    logger.info("Executing spatial standardization (128x128x128) & non-zero Z-score normalization...")
    processed_count = 0
    post_stats_after = {"min": [], "max": [], "mean": [], "std": []}
    final_shape = None
    final_dtype = None

    for idx, row in splits_df.iterrows():
        fpath = row["file_path"]
        processed_tensor = eval_transform(fpath)

        if not isinstance(processed_tensor, torch.Tensor):
            processed_tensor = torch.as_tensor(processed_tensor, dtype=torch.float32)

        # Store in cache
        cache_manager.put(fpath, processed_tensor, transform_id="oasis1_zscore_128", dtype=torch.float16)
        processed_count += 1

        if final_shape is None:
            final_shape = list(processed_tensor.shape)
            final_dtype = str(processed_tensor.dtype)

        np_arr = processed_tensor.numpy()
        post_stats_after["min"].append(float(np_arr.min()))
        post_stats_after["max"].append(float(np_arr.max()))
        post_stats_after["mean"].append(float(np_arr.mean()))
        post_stats_after["std"].append(float(np_arr.std()))

    cache_size_mb = cache_manager.get_cache_size_mb()
    logger.info(f"Successfully preprocessed & cached {processed_count} tensors to: {cache_dir} (Total Size: {cache_size_mb} MB)")

    # 4. Verify PyTorch DataLoaders
    splits_dict = {"train": train_df, "val": val_df, "test": test_df}
    task_cfg = {
        "task_type": "classification",
        "target_column": "label",
        "positive_class": 1.0,
        "negative_class": 0.0,
    }

    train_loader, val_loader, test_loader = create_benchmark_dataloaders(
        splits=splits_dict,
        task_config=task_cfg,
        batch_size=4,
        num_workers=0,
        spatial_shape=(128, 128, 128),
        label_budget=1.0,
        seed=42,
        cache_manager=cache_manager,
    )

    # Test sample batch from Train DataLoader
    sample_inputs, sample_targets, sample_meta = next(iter(train_loader))
    logger.info(f"DataLoader verification -> Sample Batch Shape: {sample_inputs.shape}, Target Shape: {sample_targets.shape}")

    # 5. Compile Audit Report JSON
    cn_count = int((splits_df["label"] == 0.0).sum())
    dementia_count = int((splits_df["label"] == 1.0).sum())

    audit_report = {
        "manifest_subjects_count": len(splits_df),
        "successfully_loaded_count": processed_count,
        "split_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "class_distribution": {
            "CN_label_0": cn_count,
            "DEMENTIA_label_1": dementia_count,
        },
        "missing_files_count": len(missing_files),
        "missing_files": missing_files,
        "invalid_files_count": len(invalid_files),
        "invalid_files": invalid_files,
        "excluded_forbidden_files": excluded_files,
        "raw_spatial_shapes_before": raw_shapes,
        "raw_dtypes_before": list(raw_dtypes),
        "intensity_stats_before_norm": {
            "mean_of_mins": round(float(np.mean(raw_stats_before["min"])), 4),
            "mean_of_maxs": round(float(np.mean(raw_stats_before["max"])), 4),
            "global_mean": round(float(np.mean(raw_stats_before["mean"])), 4),
            "global_std": round(float(np.mean(raw_stats_before["std"])), 4),
        },
        "final_tensor_shape": final_shape,
        "final_dtype": final_dtype,
        "intensity_stats_after_norm": {
            "mean_of_mins": round(float(np.mean(post_stats_after["min"])), 4),
            "mean_of_maxs": round(float(np.mean(post_stats_after["max"])), 4),
            "global_mean": round(float(np.mean(post_stats_after["mean"])), 4),
            "global_std": round(float(np.mean(post_stats_after["std"])), 4),
        },
        "cache_location": cache_dir,
        "cache_size_mb": cache_size_mb,
        "preprocessing_failures_count": 0,
        "patient_leakage_detected": has_leakage,
        "patient_leakage_verification": "0% Patient Overlap (PASSED)",
    }

    report_file = os.path.join(output_dir, "oasis1_pilot_preprocessing_audit.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    logger.info(f"Preprocessing Audit Report saved to: {report_file}")
    logger.info("Phase 5B Real-Data Preprocessing Validation Completed Successfully!")
    return audit_report


if __name__ == "__main__":
    run_oasis1_pilot_preprocessing()
