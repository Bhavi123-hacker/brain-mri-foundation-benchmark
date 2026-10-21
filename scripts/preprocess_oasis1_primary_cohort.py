"""
Step 4: Preprocessing and Tensor Caching for OASIS-1 Primary Cohort (N = 198).

Spatial Target: (1, 128, 128, 128), Float32.
Canonical Orientation: RAS (via NiBabel as_closest_canonical).
Interpolation: Trilinear spatial resampling.
Normalization: Sample-level non-zero foreground Z-score normalization.
Primary Representation: T88_111 (Atlas space).

Destination:
- D:\\brain_mri_data\\OASIS1\\processed\\tensors\\<session_id>.npy
- D:\\brain_mri_data\\OASIS1\\processed\\tensors\\<session_id>.pt

Manifests saved outside Git:
- D:\\brain_mri_data\\OASIS1\\manifests\\oasis1_preprocessing_config.json
- D:\\brain_mri_data\\OASIS1\\manifests\\oasis1_preprocessed_cache_manifest.csv
- D:\\brain_mri_data\\OASIS1\\manifests\\oasis1_preprocessing_summary.json
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

# Paths
SPLITS_CSV = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
PROCESSED_ROOT = r"D:\brain_mri_data\OASIS1\processed"
TENSORS_DIR = r"D:\brain_mri_data\OASIS1\processed\tensors"
CONFIG_OUTPUT = r"D:\brain_mri_data\OASIS1\manifests\oasis1_preprocessing_config.json"
MANIFEST_OUTPUT = r"D:\brain_mri_data\OASIS1\manifests\oasis1_preprocessed_cache_manifest.csv"
SUMMARY_OUTPUT = r"D:\brain_mri_data\OASIS1\manifests\oasis1_preprocessing_summary.json"

TARGET_SHAPE = (1, 128, 128, 128)
SPATIAL_SIZE = (128, 128, 128)

def preprocess_oasis1_primary_cohort():
    start_time = time.time()
    print("=" * 80)
    print("STEP 4: OASIS-1 TENSOR PREPROCESSING & CACHING (N = 198)")
    print("=" * 80)
    
    # 1. Guards
    assert os.path.exists(SPLITS_CSV), f"Locked split manifest missing: {SPLITS_CSV}"
    splits_df = pd.read_csv(SPLITS_CSV)
    n_cohort = len(splits_df)
    assert n_cohort == 198, f"Expected 198 subjects in locked splits, found {n_cohort}"
    
    os.makedirs(TENSORS_DIR, exist_ok=True)
    
    # Save preprocessing configuration first
    prep_config = {
        "pipeline_name": "OASIS-1 Primary Cohort Preprocessing Pipeline",
        "cohort_size": n_cohort,
        "input_representation": "T88_111",
        "input_description": "Talairach 111 atlas space, gain-field corrected MPRAGE (.hdr/.img)",
        "spatial_target_shape": list(TARGET_SHAPE),
        "spatial_resampling": "Trilinear interpolation to (128, 128, 128)",
        "orientation": "Canonical RAS (reoriented from native LAS via nibabel as_closest_canonical)",
        "intensity_normalization": "Sample-level foreground non-zero Z-score: (x[x > 0] - mu_fg) / sigma_fg",
        "background_handling": "Background (voxels <= 0) strictly set to 0.0",
        "tensor_dtype": "float32",
        "cache_formats": [".npy (numpy array)", ".pt (torch dictionary)"],
        "cache_directory": TENSORS_DIR,
        "locked_splits_source": SPLITS_CSV,
        "zero_leakage_guarantee": "Independent per-sample normalization; no inter-sample or cross-split leakage",
    }
    with open(CONFIG_OUTPUT, "w") as f:
        json.dump(prep_config, f, indent=2)
    print(f"Saved preprocessing config to: {CONFIG_OUTPUT}")
    
    # 2. Pilot Validation on 5 Diverse Subjects
    print("\n" + "-" * 80)
    print("PHASE 4A: PILOT VALIDATION ON 5 DIVERSE SUBJECTS")
    print("-" * 80)
    pilot_sids = ["OAS1_0001_MR1", "OAS1_0003_MR1", "OAS1_0010_MR1", "OAS1_0028_MR1", "OAS1_0031_MR1"]
    pilot_df = splits_df[splits_df["session_id"].isin(pilot_sids)]
    assert len(pilot_df) == 5, f"Expected 5 pilot subjects, found {len(pilot_df)}"
    
    for idx, row in pilot_df.iterrows():
        sid = row["session_id"]
        hdr_path = row["t88_hdr_path"]
        assert os.path.exists(hdr_path), f"Missing HDR file: {hdr_path}"
        
        img_obj = nib.load(hdr_path)
        canonical = nib.as_closest_canonical(img_obj)
        raw_arr = np.squeeze(canonical.get_fdata(dtype=np.float32))
        
        t = torch.from_numpy(raw_arr).unsqueeze(0).unsqueeze(0) # (1, 1, 176, 208, 176)
        fg_mask = t > 0
        assert fg_mask.sum() > 0, f"Empty foreground for {sid}"
        fg_vals = t[fg_mask]
        fg_mu = fg_vals.mean()
        fg_std = fg_vals.std()
        assert fg_std > 1e-6, f"Zero variance for {sid}"
        
        norm_t = torch.zeros_like(t)
        norm_t[fg_mask] = (fg_vals - fg_mu) / fg_std
        
        res = F.interpolate(norm_t, size=SPATIAL_SIZE, mode="trilinear", align_corners=False)
        t_4d = res.squeeze(0) # (1, 128, 128, 128)
        
        assert tuple(t_4d.shape) == TARGET_SHAPE, f"Shape mismatch: {t_4d.shape}"
        assert torch.isfinite(t_4d).all(), f"Non-finite values in {sid}"
        assert t_4d.std() > 0.1, f"Low variance in resampled {sid}"
        
        # Check orthogonal mid-slices
        mid_sag = t_4d[0, 64, :, :]
        mid_cor = t_4d[0, :, 64, :]
        mid_axi = t_4d[0, :, :, 64]
        assert mid_sag.std() > 0.1 and mid_cor.std() > 0.1 and mid_axi.std() > 0.1, f"Empty mid-slice in {sid}"
        
        print(f"  Pilot Check Passed: {sid} ({row['label_description']}, CDR={row['cdr']}) -> min={t_4d.min():.2f}, max={t_4d.max():.2f}, mean={t_4d.mean():.4f}, std={t_4d.std():.4f}")
    
    print("Pilot validation passed successfully on all 5 subjects.")
    
    # 3. Full Cohort Preprocessing & Caching
    print("\n" + "-" * 80)
    print("PHASE 4B: PROCESSING FULL 198-SUBJECT PRIMARY COHORT")
    print("-" * 80)
    
    records = []
    processing_errors = []
    
    for idx, row in splits_df.iterrows():
        sid = str(row["session_id"]).strip()
        base_sid = str(row["base_subject_id"]).strip()
        hdr_path = str(row["t88_hdr_path"]).strip()
        binary_label = int(row["binary_label"])
        label_desc = str(row["label_description"]).strip()
        split = str(row["split"]).strip()
        cdr = float(row["cdr"])
        age = float(row["age"])
        gender = str(row["gender"]).strip()
        
        npy_path = os.path.join(TENSORS_DIR, f"{sid}.npy")
        pt_path = os.path.join(TENSORS_DIR, f"{sid}.pt")
        
        t0 = time.time()
        
        try:
            if not os.path.exists(hdr_path):
                raise FileNotFoundError(f"Raw HDR missing on disk: {hdr_path}")
                
            # Load volume
            img_obj = nib.load(hdr_path)
            canonical = nib.as_closest_canonical(img_obj)
            raw_data = np.squeeze(canonical.get_fdata(dtype=np.float32))
            
            # Convert to PyTorch 5D
            t_5d = torch.from_numpy(raw_data).unsqueeze(0).unsqueeze(0) # (1, 1, 176, 208, 176)
            
            # Foreground mask & statistics
            fg_mask = t_5d > 0
            fg_count = int(fg_mask.sum().item())
            if fg_count == 0:
                raise ValueError(f"Empty foreground mask for {sid}")
                
            fg_vals = t_5d[fg_mask]
            fg_mean = float(fg_vals.mean().item())
            fg_std = float(fg_vals.std().item())
            if fg_std < 1e-6:
                raise ValueError(f"Zero intensity variance in foreground for {sid}")
                
            # Normalization
            norm_tensor = torch.zeros_like(t_5d)
            norm_tensor[fg_mask] = (fg_vals - fg_mean) / fg_std
            
            # Trilinear Resampling to (128, 128, 128)
            resampled = F.interpolate(norm_tensor, size=SPATIAL_SIZE, mode="trilinear", align_corners=False)
            t_final = resampled.squeeze(0).to(torch.float32) # (1, 128, 128, 128)
            
            # Rigorous checks
            assert tuple(t_final.shape) == TARGET_SHAPE, f"Shape error {t_final.shape}"
            assert torch.isfinite(t_final).all(), f"Non-finite values detected in {sid}"
            assert t_final.std() > 0.05, f"Near-zero variance in {sid}"
            
            np_final = t_final.cpu().numpy()
            
            # Save .npy
            np.save(npy_path, np_final)
            npy_size = os.path.getsize(npy_path)
            
            # Save .pt
            pt_dict = {
                "tensor": t_final,
                "session_id": sid,
                "base_subject_id": base_sid,
                "binary_label": binary_label,
                "label_description": label_desc,
                "split": split,
                "cdr": cdr,
                "age": age,
                "gender": gender,
                "shape": list(t_final.shape),
                "dtype": str(t_final.dtype),
                "source_hdr_path": hdr_path,
            }
            torch.save(pt_dict, pt_path)
            pt_size = os.path.getsize(pt_path)
            
            elapsed_item = time.time() - t0
            
            # Summary metrics
            v_min = float(np_final.min())
            v_max = float(np_final.max())
            v_mean = float(np_final.mean())
            v_std = float(np_final.std())
            nz_count = int(np.count_nonzero(np_final))
            nz_frac = float(nz_count / np_final.size)
            
            records.append({
                "session_id": sid,
                "base_subject_id": base_sid,
                "split": split,
                "binary_label": binary_label,
                "label_description": label_desc,
                "cdr": cdr,
                "age": age,
                "gender": gender,
                "raw_hdr_path": hdr_path,
                "npy_cache_path": npy_path,
                "pt_cache_path": pt_path,
                "npy_size_bytes": npy_size,
                "pt_size_bytes": pt_size,
                "tensor_shape": str(list(t_final.shape)),
                "dtype": str(t_final.dtype),
                "val_min": round(v_min, 4),
                "val_max": round(v_max, 4),
                "val_mean": round(v_mean, 6),
                "val_std": round(v_std, 4),
                "nonzero_fraction": round(nz_frac, 4),
                "raw_fg_mean": round(fg_mean, 2),
                "raw_fg_std": round(fg_std, 2),
                "process_time_sec": round(elapsed_item, 3),
                "status": "VALID",
                "error_msg": None
            })
            
        except Exception as e:
            err = f"Failed processing {sid}: {str(e)}"
            processing_errors.append(err)
            print(f"ERROR: {err}")
            records.append({
                "session_id": sid,
                "base_subject_id": base_sid,
                "split": split,
                "binary_label": binary_label,
                "label_description": label_desc,
                "cdr": cdr,
                "age": age,
                "gender": gender,
                "raw_hdr_path": hdr_path,
                "npy_cache_path": None,
                "pt_cache_path": None,
                "npy_size_bytes": 0,
                "pt_size_bytes": 0,
                "tensor_shape": None,
                "dtype": None,
                "val_min": None,
                "val_max": None,
                "val_mean": None,
                "val_std": None,
                "nonzero_fraction": None,
                "raw_fg_mean": None,
                "raw_fg_std": None,
                "process_time_sec": 0.0,
                "status": "FAILED",
                "error_msg": str(e)
            })
            
        if (idx + 1) % 50 == 0 or (idx + 1) == n_cohort:
            print(f"Processed {idx + 1}/{n_cohort} subjects...", flush=True)

    # 4. Independent Verification & Validation of All Cached Files
    print("\n" + "-" * 80)
    print("PHASE 4C: INDEPENDENT DISK AUDIT OF ALL CACHED TENSORS")
    print("-" * 80)
    
    cache_manifest_df = pd.DataFrame(records)
    cache_manifest_df.to_csv(MANIFEST_OUTPUT, index=False)
    
    valid_count = (cache_manifest_df["status"] == "VALID").sum()
    failed_count = (cache_manifest_df["status"] == "FAILED").sum()
    print(f"Cache manifest saved. Valid: {valid_count} / {n_cohort}, Failed: {failed_count}")
    assert valid_count == 198, f"Not all subjects were processed successfully! Found {valid_count} valid."
    
    # Audit reload from disk
    reload_errors = []
    total_npy_bytes = 0
    total_pt_bytes = 0
    
    for idx, row in cache_manifest_df.iterrows():
        sid = row["session_id"]
        npy_path = row["npy_cache_path"]
        pt_path = row["pt_cache_path"]
        
        # NPY verification
        if not os.path.exists(npy_path):
            reload_errors.append(f"{sid}: Missing .npy file")
            continue
        npy_arr = np.load(npy_path)
        if npy_arr.shape != TARGET_SHAPE or npy_arr.dtype != np.float32:
            reload_errors.append(f"{sid}: Invalid npy shape/dtype: {npy_arr.shape}, {npy_arr.dtype}")
        if not np.isfinite(npy_arr).all():
            reload_errors.append(f"{sid}: Non-finite values in npy array")
        total_npy_bytes += os.path.getsize(npy_path)
        
        # PT verification
        if not os.path.exists(pt_path):
            reload_errors.append(f"{sid}: Missing .pt file")
            continue
        pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
        t = pt_data["tensor"]
        if tuple(t.shape) != TARGET_SHAPE or t.dtype != torch.float32:
            reload_errors.append(f"{sid}: Invalid pt tensor shape/dtype: {t.shape}, {t.dtype}")
        if not torch.isfinite(t).all():
            reload_errors.append(f"{sid}: Non-finite values in pt tensor")
        if int(pt_data["binary_label"]) != int(row["binary_label"]):
            reload_errors.append(f"{sid}: Label mismatch in pt data")
        if pt_data["split"] != row["split"]:
            reload_errors.append(f"{sid}: Split mismatch in pt data")
        total_pt_bytes += os.path.getsize(pt_path)

    assert len(reload_errors) == 0, f"Reload validation errors: {reload_errors}"
    print(f"Independent disk reload audit: 198 / 198 tensors verified with ZERO errors.")
    
    # 5. Cohort Summary by Split
    total_time = time.time() - start_time
    total_mb = (total_npy_bytes + total_pt_bytes) / (1024**2)
    
    split_summaries = {}
    for sp in ["train", "val", "test"]:
        sub_df = cache_manifest_df[cache_manifest_df["split"] == sp]
        split_summaries[sp] = {
            "total_subjects": len(sub_df),
            "cn_subjects": int((sub_df["binary_label"] == 0).sum()),
            "dementia_subjects": int((sub_df["binary_label"] == 1).sum()),
            "val_min_range": [float(sub_df["val_min"].min()), float(sub_df["val_min"].max())],
            "val_max_range": [float(sub_df["val_max"].min()), float(sub_df["val_max"].max())],
            "val_mean_avg": round(float(sub_df["val_mean"].mean()), 6),
            "val_std_avg": round(float(sub_df["val_std"].mean()), 4),
            "nonzero_fraction_avg": round(float(sub_df["nonzero_fraction"].mean()), 4),
        }
        
    summary_report = {
        "status": "COMPLETED",
        "cohort_size": int(n_cohort),
        "valid_processed_count": int(valid_count),
        "failed_count": int(failed_count),
        "errors": processing_errors + reload_errors,
        "target_tensor_shape": list(TARGET_SHAPE),
        "target_tensor_dtype": "float32",
        "spatial_resampling": "Trilinear interpolation to (128, 128, 128)",
        "orientation": "RAS",
        "normalization": "Non-zero foreground Z-score",
        "split_breakdown": split_summaries,
        "storage_statistics": {
            "total_npy_files": 198,
            "total_pt_files": 198,
            "total_npy_size_mb": round(total_npy_bytes / (1024**2), 2),
            "total_pt_size_mb": round(total_pt_bytes / (1024**2), 2),
            "total_cache_size_mb": round(total_mb, 2),
            "total_cache_size_gb": round(total_mb / 1024, 2),
        },
        "output_paths": {
            "config_json": CONFIG_OUTPUT,
            "cache_manifest_csv": MANIFEST_OUTPUT,
            "summary_json": SUMMARY_OUTPUT,
            "tensors_directory": TENSORS_DIR,
        },
        "total_elapsed_seconds": round(total_time, 2)
    }
    
    with open(SUMMARY_OUTPUT, "w") as f:
        json.dump(summary_report, f, indent=2)
        
    print("\n" + "=" * 80)
    print("STEP 4 PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total Subjects Processed: 198 / 198 (100.0%)")
    print(f"Total Cached Files:       396 files (198 .npy + 198 .pt)")
    print(f"Total Cache Size:         {total_mb:.2f} MB (~{total_mb/1024:.2f} GB)")
    print(f"Processing Wall Clock:    {total_time:.1f}s")
    print(f"Saved Config:             {CONFIG_OUTPUT}")
    print(f"Saved Cache Manifest:     {MANIFEST_OUTPUT}")
    print(f"Saved Summary Report:     {SUMMARY_OUTPUT}")
    print("\nStep 4 completed successfully.")

if __name__ == "__main__":
    preprocess_oasis1_primary_cohort()
