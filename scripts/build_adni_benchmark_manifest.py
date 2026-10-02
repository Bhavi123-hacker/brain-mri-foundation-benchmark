"""
Builds the official benchmark manifest for the 50 SELECTED ADNI subjects.
Combines DICOM->NIfTI conversion manifest, series selection metadata, and diagnosis audit.
Outputs results/adni_cn_ad_benchmark_manifest.csv and results/adni_cn_ad_benchmark_manifest.json
"""

import os
import sys
import json
import pandas as pd

def build_benchmark_manifest(
    conv_manifest_path: str = "results/adni_nifti_conversion_manifest.csv",
    diagnosis_audit_path: str = "results/adni_diagnosis_audit.csv",
    series_selection_path: str = "results/adni_series_selection.csv",
    ida_search_path: str = r"C:\Users\Dell\Downloads\idaSearch_8_21_2026 (1).csv",
    output_csv: str = "results/adni_cn_ad_benchmark_manifest.csv",
    output_json: str = "results/adni_cn_ad_benchmark_manifest.json",
):
    print("Building ADNI CN vs AD Benchmark Manifest...")
    assert os.path.exists(conv_manifest_path), f"Missing: {conv_manifest_path}"
    assert os.path.exists(diagnosis_audit_path), f"Missing: {diagnosis_audit_path}"
    assert os.path.exists(series_selection_path), f"Missing: {series_selection_path}"

    conv_df = pd.read_csv(conv_manifest_path)
    diag_df = pd.read_csv(diagnosis_audit_path)
    sel_df = pd.read_csv(series_selection_path)

    ida_demographics = {}
    if os.path.exists(ida_search_path):
        ida_df = pd.read_csv(ida_search_path)
        for _, row in ida_df.iterrows():
            sid = str(row["Subject ID"]).strip()
            if sid not in ida_demographics:
                ida_demographics[sid] = {
                    "sex": str(row["Sex"]).strip(),
                    "age": float(row["Age"]) if pd.notnull(row["Age"]) else None,
                }

    diag_map = {row["subject_id"]: row for _, row in diag_df.iterrows()}
    sel_map = {row["subject_id"]: row for _, row in sel_df.iterrows()}

    records = []
    for _, row in conv_df.iterrows():
        sid = row["subject_id"]
        diag_info = diag_map.get(sid, {})
        sel_info = sel_map.get(sid, {})
        demo = ida_demographics.get(sid, {"sex": "UNKNOWN", "age": None})

        dx = diag_info.get("baseline_diagnosis", "UNKNOWN")
        assert dx in ["CN", "AD"], f"Unexpected diagnosis {dx} for {sid}"
        label = 0.0 if dx == "CN" else 1.0

        nifti_path = row["nifti_path"]
        assert os.path.exists(nifti_path), f"NIfTI path missing: {nifti_path}"

        rec = {
            "subject_id": sid,
            "nifti_path": nifti_path,
            "label": label,
            "diagnosis_source": diag_info.get("diagnosis_source", "idaSearch_8_21_2026 (1).csv"),
            "baseline_diagnosis": dx,
            "manufacturer": sel_info.get("manufacturer", "UNKNOWN"),
            "field_strength": float(sel_info.get("field_strength", 3.0)),
            "voxel_spacing": row["voxel_spacing"],
            "shape": row["shape"],
            "sex": demo["sex"],
            "age": demo["age"],
            "inclusion_status": "SELECTED",
        }
        records.append(rec)

    manifest_df = pd.DataFrame(records).sort_values(by="subject_id").reset_index(drop=True)
    manifest_df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} with {len(manifest_df)} rows")

    cn_count = int((manifest_df["label"] == 0.0).sum())
    ad_count = int((manifest_df["label"] == 1.0).sum())

    manifest_summary = {
        "benchmark_dataset_name": "ADNI MPRAGE CN vs AD Benchmark",
        "total_selected_subjects": len(manifest_df),
        "class_distribution": {"CN": cn_count, "AD": ad_count},
        "target_mapping": {"0.0": "CN", "1.0": "AD"},
        "manufacturers": manifest_df["manufacturer"].value_counts().to_dict(),
        "sex_distribution": manifest_df["sex"].value_counts().to_dict(),
        "mean_age": round(float(manifest_df["age"].mean()), 2) if manifest_df["age"].notnull().any() else None,
        "spatial_shapes": manifest_df["shape"].value_counts().to_dict(),
        "voxel_spacings": manifest_df["voxel_spacing"].value_counts().to_dict(),
        "subjects": manifest_df.to_dict(orient="records"),
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2)
    print(f"Wrote {output_json}")

if __name__ == "__main__":
    build_benchmark_manifest()
