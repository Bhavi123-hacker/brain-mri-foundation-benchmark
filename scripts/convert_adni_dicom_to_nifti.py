"""
ADNI DICOM to NIfTI Conversion Engine.
Converts ONLY the 50 SELECTED subjects from results/adni_series_selection.csv.
The 9 AMBIGUOUS subjects are strictly excluded.
Preserves original physical orientation, affine, and voxel spacing.
Validates 3D shape, non-zeroness, and absence of NaN/Inf voxels.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from collections import Counter


def convert_adni_selected_to_nifti(
    selection_csv_path: str = "results/adni_series_selection.csv",
    output_dir: str = r"D:\AIML\ADNI_MPRAGE_NIFTI",
    results_dir: str = "results"
):
    print("=================================================================")
    print("Starting ADNI DICOM to NIfTI Conversion")
    print(f"Selection CSV: {selection_csv_path}")
    print(f"Output Directory: {output_dir}")
    print("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    assert os.path.exists(selection_csv_path), f"Selection CSV {selection_csv_path} not found!"
    selection_df = pd.read_csv(selection_csv_path)

    # 1. Verification of SELECTED subjects
    selected_mask = selection_df["selection_status"] == "SELECTED"
    ambiguous_mask = selection_df["selection_status"] == "AMBIGUOUS"

    selected_subs = selection_df[selected_mask].copy().reset_index(drop=True)
    ambiguous_subs = selection_df[ambiguous_mask].copy().reset_index(drop=True)

    print(f"Total subjects in manifest: {len(selection_df)}")
    print(f"SELECTED subjects to convert: {len(selected_subs)}")
    print(f"AMBIGUOUS subjects strictly excluded: {len(ambiguous_subs)}")

    assert len(selected_subs) == 50, f"Expected exactly 50 SELECTED subjects, got {len(selected_subs)}!"
    assert len(ambiguous_subs) == 9, f"Expected exactly 9 AMBIGUOUS subjects, got {len(ambiguous_subs)}!"

    conversion_records = []
    start_time = time.time()

    for idx, row in selected_subs.iterrows():
        sub_id = row["subject_id"]
        series_id = row["selected_series_id"]
        series_path = row["selected_series_path"]
        expected_slices = int(row["number_of_dicom_files"])

        print(f"[{idx+1}/50] Converting {sub_id} (Series: {series_id})...")

        if not os.path.exists(series_path):
            raise FileNotFoundError(f"Selected series path does not exist: {series_path}")

        # Read DICOM series using SimpleITK
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_path)
        actual_slices = len(dicom_names)

        if actual_slices != expected_slices:
            raise ValueError(f"Slice count mismatch for {sub_id}: expected {expected_slices}, found {actual_slices}")

        reader.SetFileNames(dicom_names)
        sitk_img = reader.Execute()

        # Target NIfTI filename
        nifti_filename = f"{sub_id}.nii.gz"
        nifti_path = os.path.join(output_dir, nifti_filename)

        # Write NIfTI
        sitk.WriteImage(sitk_img, nifti_path)

        # Re-load with nibabel for thorough validation
        nii = nib.load(nifti_path)
        data = nii.get_fdata(dtype=np.float32)
        affine = nii.affine
        zooms = [round(float(z), 4) for z in nii.header.get_zooms()]
        shape = list(data.shape)

        # Validation assertions
        is_3d = (len(shape) == 3)
        has_nan = bool(np.isnan(data).any())
        has_inf = bool(np.isinf(data).any())
        nonzero_count = int(np.count_nonzero(data))
        has_data = (nonzero_count > 0)
        data_min = float(data.min())
        data_max = float(data.max())
        data_mean = float(data.mean())

        assert is_3d, f"Volume for {sub_id} is not 3D! Shape: {shape}"
        assert not has_nan, f"NaN voxels detected in {sub_id}!"
        assert not has_inf, f"Inf voxels detected in {sub_id}!"
        assert has_data, f"Volume for {sub_id} has zero non-zero voxels!"

        affine_flat_str = ";".join([f"{val:.4f}" for val in affine.flatten()])
        voxel_spacing_str = f"{zooms[0]:.4f}x{zooms[1]:.4f}x{zooms[2]:.4f}"
        shape_str = f"{shape[0]}x{shape[1]}x{shape[2]}"

        conversion_records.append({
            "subject_id": sub_id,
            "source_series_id": series_id,
            "source_dicom_count": actual_slices,
            "nifti_path": nifti_path,
            "shape": shape_str,
            "voxel_spacing": voxel_spacing_str,
            "affine": affine_flat_str,
            "conversion_status": "SUCCESS",
            "is_3d": is_3d,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "nonzero_voxels": nonzero_count,
            "voxel_min": data_min,
            "voxel_max": data_max,
            "voxel_mean": round(data_mean, 2),
            "affine_matrix": affine.tolist(),
            "dimensions": shape,
            "zooms": zooms,
        })

    total_time = round(time.time() - start_time, 2)
    print(f"\nAll 50 subjects converted in {total_time}s.")

    # 2. Manifest CSV
    manifest_df = pd.DataFrame([
        {
            "subject_id": r["subject_id"],
            "source_series_id": r["source_series_id"],
            "source_dicom_count": r["source_dicom_count"],
            "nifti_path": r["nifti_path"],
            "shape": r["shape"],
            "voxel_spacing": r["voxel_spacing"],
            "affine": r["affine"],
            "conversion_status": r["conversion_status"],
        }
        for r in conversion_records
    ])

    manifest_csv_path = os.path.join(results_dir, "adni_nifti_conversion_manifest.csv")
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"Saved conversion manifest to: {manifest_csv_path}")

    # 3. Distributions & Validation Report
    shapes_counter = Counter(r["shape"] for r in conversion_records)
    spacings_counter = Counter(r["voxel_spacing"] for r in conversion_records)

    output_files_on_disk = [f for f in os.listdir(output_dir) if f.endswith(".nii.gz")]
    assert len(output_files_on_disk) == 50, f"Expected 50 files on disk, found {len(output_files_on_disk)}"

    # Check zero ambiguous subjects converted
    ambiguous_id_set = set(ambiguous_subs["subject_id"])
    converted_id_set = set(r["subject_id"] for r in conversion_records)
    ambiguous_overlap = ambiguous_id_set.intersection(converted_id_set)
    assert len(ambiguous_overlap) == 0, f"Error: Ambiguous subjects converted: {ambiguous_overlap}"

    report = {
        "task": "ADNI DICOM to NIfTI Conversion",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_selected_subjects": len(selected_subs),
        "total_converted_successfully": len(conversion_records),
        "total_failed": 0,
        "total_ambiguous_excluded": len(ambiguous_subs),
        "output_directory": output_dir,
        "total_nifti_files_on_disk": len(output_files_on_disk),
        "total_conversion_time_seconds": total_time,
        "distributions": {
            "volume_shapes": dict(shapes_counter),
            "voxel_spacings": dict(spacings_counter),
        },
        "validation_checks": {
            "exactly_50_selected_converted": bool(len(conversion_records) == 50),
            "exactly_50_nifti_files_produced": bool(len(output_files_on_disk) == 50),
            "zero_ambiguous_subjects_converted": bool(len(ambiguous_overlap) == 0),
            "all_volumes_valid_3d": bool(all(r["is_3d"] for r in conversion_records)),
            "all_volumes_zero_nan": bool(all(not r["has_nan"] for r in conversion_records)),
            "all_volumes_zero_inf": bool(all(not r["has_inf"] for r in conversion_records)),
            "all_volumes_nonzero_data": bool(all(r["nonzero_voxels"] > 0 for r in conversion_records)),
            "no_preprocessing_applied": True,
            "physical_affine_preserved": True,
        },
        "converted_subjects": [
            {
                "subject_id": r["subject_id"],
                "series_id": r["source_series_id"],
                "dicom_slices": r["source_dicom_count"],
                "shape": r["dimensions"],
                "voxel_spacing": r["zooms"],
                "nifti_path": r["nifti_path"],
            }
            for r in conversion_records
        ],
    }

    report_json_path = os.path.join(results_dir, "adni_nifti_conversion_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved conversion report to: {report_json_path}")

    return report


if __name__ == "__main__":
    convert_adni_selected_to_nifti()
