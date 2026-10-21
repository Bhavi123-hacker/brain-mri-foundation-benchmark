"""
Step 2: OASIS-1 Scan Integrity and Dimension Audit Script.
Validates Analyze 7.5 (.hdr/.img) files for the 198 allowlisted cohort subjects
for both T88_111 and SUBJ_111 representations.

Performs:
1. Manifest and cohort validation (198 unique subjects, zero duplicates, path checks).
2. Non-destructive NiBabel loading of all 396 volume pairs.
3. 3D spatial dimension confirmation (len(shape) == 3).
4. Detailed shape, voxel spacing (zooms), affine, orientation codes, datatype, and file size tracking.
5. Voxel array NaN/Inf detection, intensity min, max, mean, std, and nonzero fraction.
6. Anomaly checks: empty arrays, zero variance, implausible spacing, unexpected shapes,
   corrupt files, and header/image byte size mismatches.
7. Label and ID cross-verification against primary cohort manifest.
8. Separate reporting for T88_111 and SUBJ_111.
9. Output to new audit CSV and summary JSON under D:\\brain_mri_data\\OASIS1\\manifests\\.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

COHORT_MANIFEST = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_manifest.csv"
SCANS_MANIFEST = r"D:\brain_mri_data\OASIS1\manifests\oasis1_extracted_scans_manifest.csv"
AUDIT_CSV_OUTPUT = r"D:\brain_mri_data\OASIS1\manifests\oasis1_scan_integrity_audit.csv"
SUMMARY_JSON_OUTPUT = r"D:\brain_mri_data\OASIS1\manifests\oasis1_scan_integrity_summary.json"

def run_integrity_audit():
    start_time = time.time()
    print("=" * 80)
    print("STEP 2: OASIS-1 SCAN INTEGRITY AND DIMENSION AUDIT (N = 198)")
    print("=" * 80)
    
    # 1. Manifest & Cohort Validation
    if not os.path.exists(COHORT_MANIFEST):
        raise FileNotFoundError(f"Cohort manifest missing: {COHORT_MANIFEST}")
    if not os.path.exists(SCANS_MANIFEST):
        raise FileNotFoundError(f"Scans manifest missing: {SCANS_MANIFEST}")
        
    cohort_df = pd.read_csv(COHORT_MANIFEST)
    scans_df = pd.read_csv(SCANS_MANIFEST)
    
    print(f"Loaded primary cohort manifest: {len(cohort_df)} records")
    print(f"Loaded extracted scans manifest: {len(scans_df)} records")
    
    assert len(cohort_df) == 198, f"Cohort size mismatch: {len(cohort_df)}"
    assert len(scans_df) == 198, f"Scans manifest size mismatch: {len(scans_df)}"
    
    unique_cohort_sids = set(cohort_df["session_id"].str.strip())
    unique_scans_sids = set(scans_df["session_id"].str.strip())
    
    sid_diff = unique_cohort_sids.symmetric_difference(unique_scans_sids)
    if sid_diff:
        raise ValueError(f"Subject ID mismatch between cohort and scans manifest: {sid_diff}")
    print(f"Cohort and scans manifest subject IDs match perfectly ({len(unique_scans_sids)} unique subjects).")
    
    records = []
    corruption_errors = []
    warnings_list = []
    
    shapes_by_rep = {"T88_111": {}, "SUBJ_111": {}}
    zooms_by_rep = {"T88_111": {}, "SUBJ_111": {}}
    orientations_by_rep = {"T88_111": {}, "SUBJ_111": {}}
    dtypes_by_rep = {"T88_111": {}, "SUBJ_111": {}}
    
    # Process each subject
    for idx, row in scans_df.iterrows():
        sid = str(row["session_id"]).strip()
        base_sid = str(row["base_subject_id"]).strip()
        cdr = float(row["cdr"])
        binary_label = int(row["binary_label"])
        
        # Verify cross-consistency with cohort_df
        cohort_row = cohort_df[cohort_df["session_id"] == sid].iloc[0]
        assert float(cohort_row["cdr"]) == cdr, f"CDR mismatch for {sid}"
        assert int(cohort_row["binary_label"]) == binary_label, f"Label mismatch for {sid}"
        
        for rep in ["T88_111", "SUBJ_111"]:
            hdr_col = "t88_hdr_path" if rep == "T88_111" else "sbj_hdr_path"
            img_col = "t88_img_path" if rep == "T88_111" else "sbj_img_path"
            
            hdr_path = str(row[hdr_col]) if pd.notna(row[hdr_col]) else None
            img_path = str(row[img_col]) if pd.notna(row[img_col]) else None
            
            hdr_exists = os.path.exists(hdr_path) if hdr_path else False
            img_exists = os.path.exists(img_path) if img_path else False
            
            hdr_size = os.path.getsize(hdr_path) if hdr_exists else 0
            img_size = os.path.getsize(img_path) if img_exists else 0
            
            rec = {
                "session_id": sid,
                "base_subject_id": base_sid,
                "representation": rep,
                "cdr": cdr,
                "binary_label": binary_label,
                "label_description": row["label_description"],
                "hdr_path": hdr_path,
                "img_path": img_path,
                "hdr_exists": hdr_exists,
                "img_exists": img_exists,
                "hdr_size_bytes": hdr_size,
                "img_size_bytes": img_size,
                "load_success": False,
                "is_3d": False,
                "raw_shape": None,
                "spatial_shape": None,
                "shape": None,
                "zooms": None,
                "orientation": None,
                "dtype": None,
                "val_min": None,
                "val_max": None,
                "val_mean": None,
                "val_std": None,
                "nonzero_fraction": None,
                "has_nan": None,
                "has_inf": None,
                "non_zero_variance": False,
                "expected_img_bytes": None,
                "header_size_match": False,
                "status": "UNPROCESSED",
                "error_msg": None
            }
            
            if not (hdr_exists and img_exists):
                err = f"{sid} ({rep}): Missing .hdr or .img on disk"
                rec["error_msg"] = err
                rec["status"] = "FILE_MISSING"
                corruption_errors.append(err)
                records.append(rec)
                continue
                
            try:
                # Load image via NiBabel without modifying source
                img_obj = nib.load(hdr_path)
                data = img_obj.get_fdata(dtype=np.float32)
                header = img_obj.header
                
                raw_shape = tuple(img_obj.shape)
                if len(raw_shape) == 4 and raw_shape[3] == 1:
                    spatial_shape = raw_shape[:3]
                    is_3d = True
                elif len(raw_shape) == 3:
                    spatial_shape = raw_shape
                    is_3d = True
                else:
                    spatial_shape = raw_shape
                    is_3d = False

                zooms = tuple(round(float(z), 4) for z in header.get_zooms()[:3])
                dtype_obj = header.get_data_dtype()
                dtype_str = str(dtype_obj)
                
                # Orientation
                try:
                    axcodes = "".join(nib.aff2axcodes(img_obj.affine))
                except Exception:
                    axcodes = "UNKNOWN"
                    
                # Header vs Image byte size check
                data_shape = header.get_data_shape()
                expected_data_voxels = int(np.prod(data_shape))
                bytes_per_voxel = dtype_obj.itemsize
                expected_img_bytes = expected_data_voxels * bytes_per_voxel
                header_size_match = (img_size == expected_img_bytes)
                
                # Intensity metrics
                has_nan = bool(np.isnan(data).any())
                has_inf = bool(np.isinf(data).any())
                v_min = float(np.min(data))
                v_max = float(np.max(data))
                v_mean = float(np.mean(data))
                v_std = float(np.std(data))
                nonzero_count = int(np.count_nonzero(data))
                nonzero_frac = float(nonzero_count / data.size)
                non_zero_var = bool(v_std > 1e-6)
                
                rec.update({
                    "load_success": True,
                    "is_3d": is_3d,
                    "raw_shape": str(raw_shape),
                    "spatial_shape": str(spatial_shape),
                    "shape": str(spatial_shape),
                    "zooms": str(zooms),
                    "orientation": axcodes,
                    "dtype": dtype_str,
                    "val_min": round(v_min, 4),
                    "val_max": round(v_max, 4),
                    "val_mean": round(v_mean, 4),
                    "val_std": round(v_std, 4),
                    "nonzero_fraction": round(nonzero_frac, 6),
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "non_zero_variance": non_zero_var,
                    "expected_img_bytes": expected_img_bytes,
                    "header_size_match": header_size_match,
                    "status": "VALID",
                })
                
                # Aggregate distributions
                shapes_by_rep[rep][str(spatial_shape)] = shapes_by_rep[rep].get(str(spatial_shape), 0) + 1
                zooms_by_rep[rep][str(zooms)] = zooms_by_rep[rep].get(str(zooms), 0) + 1
                orientations_by_rep[rep][axcodes] = orientations_by_rep[rep].get(axcodes, 0) + 1
                dtypes_by_rep[rep][dtype_str] = dtypes_by_rep[rep].get(dtype_str, 0) + 1
                
                # Check for anomalies
                anomalies = []
                if not is_3d:
                    anomalies.append(f"Not 3D: shape={raw_shape}")
                if has_nan:
                    anomalies.append("Contains NaN")
                if has_inf:
                    anomalies.append("Contains Inf")
                if not non_zero_var:
                    anomalies.append("Zero variance (flat image)")
                if nonzero_frac < 0.05:
                    anomalies.append(f"Low nonzero voxel fraction ({nonzero_frac:.4f})")
                if not header_size_match:
                    anomalies.append(f"Header/image byte mismatch: img={img_size} vs expected={expected_img_bytes}")
                if any(z <= 0 for z in zooms):
                    anomalies.append(f"Implausible spacing: {zooms}")
                    
                if anomalies:
                    err_str = "; ".join(anomalies)
                    rec["status"] = "ANOMALY"
                    rec["error_msg"] = err_str
                    corruption_errors.append(f"{sid} ({rep}): {err_str}")
                    
            except Exception as e:
                rec["status"] = "EXCEPTION"
                rec["error_msg"] = str(e)
                corruption_errors.append(f"{sid} ({rep}): Exception {e}")
                
            records.append(rec)
            
        if (idx + 1) % 50 == 0 or (idx + 1) == len(scans_df):
            print(f"Audited {idx + 1}/{len(scans_df)} subjects ({len(records)} volumes)...", flush=True)
            
    audit_df = pd.DataFrame(records)
    audit_df.to_csv(AUDIT_CSV_OUTPUT, index=False)
    
    elapsed = time.time() - start_time
    
    # Detailed representations summary
    rep_summaries = {}
    for rep in ["T88_111", "SUBJ_111"]:
        sub_df = audit_df[audit_df["representation"] == rep]
        valid_df = sub_df[sub_df["status"] == "VALID"]
        
        rep_summaries[rep] = {
            "total_scans": len(sub_df),
            "valid_scans": len(valid_df),
            "failed_scans": len(sub_df) - len(valid_df),
            "is_3d_count": int(sub_df["is_3d"].sum()),
            "nan_count": int((sub_df["has_nan"] == True).sum()),
            "inf_count": int((sub_df["has_inf"] == True).sum()),
            "zero_variance_count": int((sub_df["non_zero_variance"] == False).sum()),
            "header_size_match_count": int(sub_df["header_size_match"].sum()),
            "shapes_distribution": shapes_by_rep[rep],
            "zooms_distribution": zooms_by_rep[rep],
            "orientations_distribution": orientations_by_rep[rep],
            "dtypes_distribution": dtypes_by_rep[rep],
            "intensity_stats": {
                "val_min_range": [float(valid_df["val_min"].min()), float(valid_df["val_min"].max())],
                "val_max_range": [float(valid_df["val_max"].min()), float(valid_df["val_max"].max())],
                "val_mean_avg": round(float(valid_df["val_mean"].mean()), 4),
                "val_std_avg": round(float(valid_df["val_std"].mean()), 4),
                "nonzero_fraction_avg": round(float(valid_df["nonzero_fraction"].mean()), 4),
                "nonzero_fraction_min": round(float(valid_df["nonzero_fraction"].min()), 4),
                "nonzero_fraction_max": round(float(valid_df["nonzero_fraction"].max()), 4),
            }
        }
        
    summary = {
        "status": "COMPLETED",
        "cohort_size": 198,
        "total_volumes_audited": len(records),
        "total_errors": len(corruption_errors),
        "errors_list": corruption_errors,
        "representations": rep_summaries,
        "audit_csv_path": AUDIT_CSV_OUTPUT,
        "summary_json_path": SUMMARY_JSON_OUTPUT,
        "elapsed_seconds": round(elapsed, 2)
    }
    
    with open(SUMMARY_JSON_OUTPUT, "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 80)
    print("STEP 2 AUDIT COMPLETE")
    print("=" * 80)
    for rep, s in rep_summaries.items():
        print(f"\n--- Representation: {rep} ---")
        print(f"  Valid / Total:             {s['valid_scans']} / {s['total_scans']}")
        print(f"  Shape Distribution:        {s['shapes_distribution']}")
        print(f"  Zooms (Voxel Spacing):     {s['zooms_distribution']}")
        print(f"  Orientations:              {s['orientations_distribution']}")
        print(f"  Data Types:                {s['dtypes_distribution']}")
        print(f"  Header/Image Size Matches: {s['header_size_match_count']} / {s['total_scans']}")
        print(f"  NaN / Inf / Zero-Variance: {s['nan_count']} / {s['inf_count']} / {s['zero_variance_count']}")
        print(f"  Mean Non-zero Fraction:    {s['intensity_stats']['nonzero_fraction_avg']:.4f}")
        print(f"  Intensity Range [Min, Max]:{s['intensity_stats']['val_min_range']} to {s['intensity_stats']['val_max_range']}")
        
    print(f"\nTotal Errors:    {len(corruption_errors)}")
    print(f"Time Taken:      {elapsed:.1f}s")
    print(f"Saved CSV:       {AUDIT_CSV_OUTPUT}")
    print(f"Saved JSON:      {SUMMARY_JSON_OUTPUT}")

if __name__ == "__main__":
    run_integrity_audit()
