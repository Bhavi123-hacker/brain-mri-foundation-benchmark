"""
Preprocesses and caches the 50 selected ADNI MPRAGE NIfTI volumes into standardized 3D tensors.
Spatial target: (1, 128, 128, 128), Float32.
Normalisation: Non-zero Z-score intensity normalization.
Caches:
  1. Standard .npy files at data/cache/adni_cn_ad/<subject_id>.npy
  2. PreprocessedCacheManager entries (.pt) + cache_index.json for seamless PyTorch DataLoader integration.
Saves audit report at results/adni_preprocessing_report.json and manifest at results/adni_preprocessing_manifest.csv
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from preprocessing.transforms import get_preprocessing_transforms
from preprocessing.cache_manager import PreprocessedCacheManager

def preprocess_and_cache_adni(
    manifest_path: str = "results/adni_cn_ad_benchmark_manifest.csv",
    cache_dir: str = "data/cache/adni_cn_ad",
    output_report: str = "results/adni_preprocessing_report.json",
    output_manifest: str = "results/adni_preprocessing_manifest.csv",
    spatial_shape=(128, 128, 128),
    voxel_spacing=(1.0, 1.0, 1.0),
):
    logger = get_logger("ADNIPreprocessing")
    logger.info("=================================================================")
    logger.info("Starting ADNI Preprocessing & Disk Caching (50 SELECTED Subjects)")
    logger.info("=================================================================")

    assert os.path.exists(manifest_path), f"Manifest missing: {manifest_path}"
    manifest_df = pd.read_csv(manifest_path)
    logger.info(f"Loaded {len(manifest_df)} subjects from {manifest_path}")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_report), exist_ok=True)

    cache_manager = PreprocessedCacheManager(cache_dir=cache_dir)
    transform = get_preprocessing_transforms(
        spatial_shape=spatial_shape,
        voxel_spacing=voxel_spacing,
        intensity_norm="z_score",
        is_training=False,
    )

    records = []
    stats_all = {"min": [], "max": [], "mean": [], "std": []}
    start_time = time.time()

    for idx, row in manifest_df.iterrows():
        sid = str(row["subject_id"]).strip()
        nifti_path = str(row["nifti_path"]).strip()
        label = float(row["label"])
        dx = str(row["baseline_diagnosis"]).strip()

        logger.info(f"[{idx+1}/{len(manifest_df)}] Processing {sid} ({dx})...")
        t0 = time.time()

        if not os.path.exists(nifti_path):
            raise FileNotFoundError(f"Missing NIfTI for {sid}: {nifti_path}")

        # Execute deterministic transform
        transformed = transform(nifti_path)
        if not isinstance(transformed, torch.Tensor):
            transformed = torch.as_tensor(transformed, dtype=torch.float32)

        if transformed.ndim == 3:
            transformed = transformed.unsqueeze(0)
        transformed = transformed.to(dtype=torch.float32)

        # Integrity checks
        has_nan = torch.isnan(transformed).any().item()
        has_inf = torch.isinf(transformed).any().item()
        assert not has_nan, f"NaN detected in preprocessed scan {sid}!"
        assert not has_inf, f"Inf detected in preprocessed scan {sid}!"
        assert tuple(transformed.shape) == (1, 128, 128, 128), f"Unexpected shape {transformed.shape} for {sid}"

        np_arr = transformed.cpu().numpy()
        min_v = float(np_arr.min())
        max_v = float(np_arr.max())
        mean_v = float(np_arr.mean())
        std_v = float(np_arr.std())

        assert std_v > 0, f"Zero variance (constant volume) for {sid}!"

        stats_all["min"].append(min_v)
        stats_all["max"].append(max_v)
        stats_all["mean"].append(mean_v)
        stats_all["std"].append(std_v)

        # Save .npy file
        npy_filename = f"{sid}.npy"
        npy_path = os.path.join(cache_dir, npy_filename)
        np.save(npy_path, np_arr)

        # Register in PreprocessedCacheManager (for DataLoader lookup)
        pt_path = cache_manager.put(nifti_path, transformed, transform_id="adni_zscore_128", dtype=torch.float16)

        elapsed = time.time() - t0
        logger.info(f"  Done {sid}: shape={transformed.shape}, min={min_v:.3f}, max={max_v:.3f}, mean={mean_v:.4f}, std={std_v:.4f} ({elapsed:.2f}s)")

        records.append({
            "subject_id": sid,
            "baseline_diagnosis": dx,
            "label": label,
            "nifti_path": nifti_path,
            "npy_cache_path": os.path.abspath(npy_path),
            "npy_file_size_bytes": os.path.getsize(npy_path),
            "pt_cache_path": pt_path,
            "tensor_shape": str(list(transformed.shape)),
            "dtype": str(transformed.dtype),
            "min": min_v,
            "max": max_v,
            "mean": mean_v,
            "std": std_v,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "status": "SUCCESS",
        })

    total_elapsed = time.time() - start_time
    cache_size_mb = cache_manager.get_cache_size_mb()

    # Save manifest CSV
    prep_manifest_df = pd.DataFrame(records)
    prep_manifest_df.to_csv(output_manifest, index=False)
    logger.info(f"Saved preprocessing manifest to {output_manifest}")

    # Save report JSON
    report = {
        "dataset_name": "ADNI MPRAGE CN vs AD Benchmark",
        "total_processed_subjects": len(records),
        "total_failures": 0,
        "preprocessed_tensor_shape": [1, 128, 128, 128],
        "tensor_dtype": "torch.float32 (cached as float16 in .pt, float32 in .npy)",
        "intensity_normalization": "non-zero z-score (channel-wise)",
        "spatial_resampling": "trilinear interpolation to (128, 128, 128)",
        "cache_directory": os.path.abspath(cache_dir),
        "total_cache_size_mb": cache_size_mb,
        "processing_time_seconds": round(total_elapsed, 2),
        "aggregate_intensity_statistics": {
            "mean_of_mins": round(float(np.mean(stats_all["min"])), 4),
            "mean_of_maxs": round(float(np.mean(stats_all["max"])), 4),
            "global_mean": round(float(np.mean(stats_all["mean"])), 4),
            "global_std": round(float(np.mean(stats_all["std"])), 4),
        },
        "all_validations_passed": True,
        "zero_nan_inf_verified": True,
        "class_distribution": {
            "CN": int((manifest_df["label"] == 0.0).sum()),
            "AD": int((manifest_df["label"] == 1.0).sum()),
        },
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved preprocessing report to {output_report}")
    logger.info(f"Preprocessing completed in {total_elapsed:.1f}s")
    return report

if __name__ == "__main__":
    preprocess_and_cache_adni()
