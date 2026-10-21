"""
Preprocess and Cache Full ADNI Verified Baseline Cohort (N = 500).

Authorized stage: Preprocessing-only. Strictly NO model inference or training.

Pipeline:
1. Load verified 500-subject baseline manifest
2. For each subject:
   - Volume reconstruction (GDCM slice sorting for multi-file; ImageFileReader for Enhanced Multi-Frame)
   - Canonical RAS reorientation via sitk.DICOMOrient
   - Physical-space isotropic resampling to 1.0 mm^3 grid
   - Otsu foreground masking to separate anatomical head tissue from background air
   - Sample-level foreground Z-score normalization with background clamped to 0.0
   - Standardized trilinear spatial resampling to (1, 128, 128, 128) Float32
3. Dual-cache outputs as .npy and .pt tensors in D:\\brain_mri_data\\ADNI\\processed\\tensors\\
4. Output full cache manifest and QC report in D:\\brain_mri_data\\ADNI\\manifests\\
"""

import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from skimage.filters import threshold_otsu

def process_subject(row_dict):
    sub_id = row_dict["subject_id"]
    img_id = row_dict["image_data_id"]
    folder = row_dict["series_folder"]
    is_single = row_dict["is_single_file_volume"]
    first_file = row_dict["first_file"]
    group = row_dict["csv_group"]
    mfg = row_dict["mfg_clean"]
    field = float(row_dict["field_clean"])
    visit = row_dict["csv_visit"]
    tensors_dir = row_dict["tensors_dir"]

    npy_path = os.path.join(tensors_dir, f"{sub_id}_{img_id}.npy")
    pt_path = os.path.join(tensors_dir, f"{sub_id}_{img_id}.pt")

    try:
        # 1. Volume reconstruction
        if is_single:
            r = sitk.ImageFileReader()
            r.SetFileName(first_file)
            img = r.Execute()
            file_count = 1
        else:
            r = sitk.ImageSeriesReader()
            files = r.GetGDCMSeriesFileNames(folder)
            if not files:
                return {"status": "FAIL", "subject_id": sub_id, "image_data_id": img_id, "error": "No DICOM files found"}
            r.SetFileNames(files)
            img = r.Execute()
            file_count = len(files)

        orig_size = list(img.GetSize())
        orig_spacing = [round(float(s), 4) for s in img.GetSpacing()]

        # 2. Canonical RAS reorientation
        ras = sitk.DICOMOrient(img, "RAS")
        ras_size = list(ras.GetSize())
        ras_spacing = [round(float(s), 4) for s in ras_spacing] if 'ras_spacing' in locals() else [round(float(s), 4) for s in ras.GetSpacing()]

        # 3. Physical-space isotropic resampling to 1.0 mm^3
        sz = ras.GetSize()
        sp = ras.GetSpacing()
        new_sz = [int(round(sz[i] * sp[i])) for i in range(3)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing([1.0, 1.0, 1.0])
        resampler.SetSize(new_sz)
        resampler.SetOutputOrigin(ras.GetOrigin())
        resampler.SetOutputDirection(ras.GetDirection())
        iso = resampler.Execute(ras)

        arr = sitk.GetArrayFromImage(iso).astype(np.float32)

        # 4. Otsu foreground head masking & sample-level Z-score normalization
        try:
            thresh = float(threshold_otsu(arr))
            fg = arr > thresh
        except Exception:
            thresh = 0.0
            fg = arr > 0.0

        if fg.sum() == 0:
            fg = arr > 0.0
        
        fg_vals = arr[fg]
        mu = float(fg_vals.mean())
        std = float(fg_vals.std())

        norm = np.zeros_like(arr)
        norm[fg] = (arr[fg] - mu) / max(std, 1e-5)

        # 5. Spatial resampling to target (1, 128, 128, 128)
        t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(t, size=(128, 128, 128), mode="trilinear", align_corners=False)
        final_tensor = resized.squeeze(0) # (1, 128, 128, 128)

        np_final = final_tensor.cpu().numpy()

        # 6. Integrity assertions
        assert not np.isnan(np_final).any(), "NaN detected"
        assert not np.isinf(np_final).any(), "Inf detected"
        assert np_final.shape == (1, 128, 128, 128), f"Bad shape: {np_final.shape}"

        min_val = float(np_final.min())
        max_val = float(np_final.max())
        mean_val = float(np_final.mean())
        std_val = float(np_final.std())
        nonzero_pct = float((np_final != 0).mean() * 100)

        # 7. Dual-cache
        np.save(npy_path, np_final)
        torch.save({
            "tensor": final_tensor,
            "subject_id": sub_id,
            "image_data_id": img_id,
            "group": group,
            "binary_label": 0 if group == "CN" else 1,
            "manufacturer": mfg,
            "field_strength": field,
            "visit": visit,
            "age": float(row_dict["csv_age"]),
            "sex": row_dict["csv_sex"],
            "stats": {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val, "nonzero_pct": nonzero_pct, "otsu_thresh": thresh}
        }, pt_path)

        return {
            "status": "PASS",
            "subject_id": sub_id,
            "image_data_id": img_id,
            "group": group,
            "binary_label": 0 if group == "CN" else 1,
            "manufacturer": mfg,
            "field_strength": field,
            "visit": visit,
            "acq_date": row_dict["csv_acq_date"],
            "file_count": file_count,
            "orig_size": orig_size,
            "orig_spacing": orig_spacing,
            "otsu_thresh": round(thresh, 2),
            "nonzero_pct": round(nonzero_pct, 2),
            "min_val": round(min_val, 4),
            "max_val": round(max_val, 4),
            "mean_val": round(mean_val, 4),
            "std_val": round(std_val, 4),
            "npy_path": npy_path,
            "pt_path": pt_path
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "subject_id": sub_id,
            "image_data_id": img_id,
            "error": str(e)
        }

def main():
    manifest_path = r"D:\brain_mri_data\ADNI\manifests\adni_data_readiness_manifest.csv"
    output_dir = r"D:\brain_mri_data\ADNI\manifests"
    tensors_dir = r"D:\brain_mri_data\ADNI\processed\tensors"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensors_dir, exist_ok=True)

    print("=================================================================")
    print("PREPROCESSING FULL ADNI VERIFIED BASELINE COHORT (N=500)")
    print("=================================================================")

    df = pd.read_csv(manifest_path)
    matched = df[df["matched_in_csv"]].copy()

    # Deterministic baseline selection (4-key hierarchical rule)
    matched["folder_timestamp"] = matched["series_folder"].apply(lambda p: os.path.basename(os.path.dirname(p)))
    matched["acq_dt"] = pd.to_datetime(matched["csv_acq_date"])
    matched["image_id_num"] = matched["image_data_id"].str.replace("I", "", regex=False).astype(int)

    df_sorted = matched.sort_values(["subject_id", "acq_dt", "folder_timestamp", "image_id_num"])
    baseline_cohort = df_sorted.groupby("subject_id").first().reset_index()

    baseline_cohort["mfg_clean"] = baseline_cohort["manufacturer_dcm"].str.strip().replace({
        "Philips Medical Systems": "Philips",
        "Philips Healthcare": "Philips",
        "SIEMENS": "Siemens"
    })
    baseline_cohort["field_clean"] = baseline_cohort["field_strength_dcm"].astype(float).round(1)
    baseline_cohort["tensors_dir"] = tensors_dir

    n_total = len(baseline_cohort)
    print(f"Total cohort to preprocess: {n_total} subjects (CN: {(baseline_cohort['csv_group']=='CN').sum()}, AD: {(baseline_cohort['csv_group']=='AD').sum()})")

    row_dicts = baseline_cohort.to_dict("records")

    start_time = time.time()
    results = []
    failed = []

    # Use 6 parallel workers for optimal throughput
    num_workers = min(6, os.cpu_count() or 4)
    print(f"Starting parallel execution with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, r): r["subject_id"] for r in row_dicts}
        done_count = 0
        for future in as_completed(futures):
            sub_id = futures[future]
            try:
                res = future.result()
                results.append(res)
                if res["status"] == "FAIL":
                    failed.append(res)
            except Exception as exc:
                failed.append({"status": "FAIL", "subject_id": sub_id, "error": str(exc)})
            
            done_count += 1
            if done_count % 50 == 0 or done_count == n_total:
                elapsed = time.time() - start_time
                rate = done_count / elapsed
                print(f"Progress: [{done_count}/{n_total}] ({done_count/n_total*100:.1f}%) | Rate: {rate:.1f} scans/s | Elapsed: {elapsed:.1f}s")

    elapsed_total = time.time() - start_time
    pass_count = len([r for r in results if r["status"] == "PASS"])
    fail_count = len(failed)

    print(f"\nPreprocessing finished in {elapsed_total:.1f}s.")
    print(f"Successfully preprocessed and cached: {pass_count} / {n_total}")
    print(f"Failed scans: {fail_count}")

    # Save full cache manifest and QC summary
    out_manifest_csv = os.path.join(output_dir, "adni_full_preprocessed_manifest.csv")
    out_qc_json = os.path.join(output_dir, "adni_full_preprocessing_qc.json")

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_manifest_csv, index=False)
    print(f"Saved full preprocessed manifest to: {out_manifest_csv}")

    qc_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_attempted": n_total,
        "successfully_preprocessed": pass_count,
        "failed_count": fail_count,
        "elapsed_seconds": round(elapsed_total, 2),
        "throughput_scans_per_sec": round(n_total / elapsed_total, 2),
        "target_shape": [1, 128, 128, 128],
        "preprocessing_pipeline": {
            "orientation": "Canonical RAS (SimpleITK DICOMOrient)",
            "resampling": "1.0 mm isotropic physical resampling (trilinear)",
            "air_masking": "Otsu threshold on resampled volume to separate air from head anatomy",
            "normalization": "Sample-level foreground non-zero Z-score: (x - mu_fg) / sigma_fg",
            "background_handling": "Background air voxels strictly clamped to 0.0"
        },
        "diagnostic_distribution": {
            "CN": int((df_res["group"] == "CN").sum()) if "group" in df_res.columns else 0,
            "AD": int((df_res["group"] == "AD").sum()) if "group" in df_res.columns else 0
        },
        "manufacturer_distribution": {
            str(k): int(v) for k, v in df_res["manufacturer"].value_counts().items()
        } if "manufacturer" in df_res.columns else {},
        "failures": failed
    }

    with open(out_qc_json, "w") as f:
        json.dump(qc_summary, f, indent=2)
    print(f"Saved full preprocessing QC JSON to: {out_qc_json}")

if __name__ == "__main__":
    main()
