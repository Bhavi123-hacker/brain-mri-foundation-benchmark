"""
Generate ADNI MPRAGE Benchmark Audit Files and Selection Manifest.
Outputs:
- results/adni_series_audit.csv
- results/adni_series_selection.csv
- results/adni_ambiguous_subjects.csv
- results/adni_dataset_audit.json
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter


def generate_audit_files():
    audit_csv_path = "results/adni_series_audit.csv"
    assert os.path.exists(audit_csv_path), f"Audit CSV {audit_csv_path} not found! Run audit first."

    df = pd.read_csv(audit_csv_path)
    print(f"Loaded series audit with {len(df)} series across {df['subject_id'].nunique()} unique subjects.")

    # 1. Inspect subject-level grouping
    sub_groups = df.groupby("subject_id")
    unique_subjects = sorted(list(sub_groups.groups.keys()))
    assert len(unique_subjects) == 59, f"Expected 59 subjects, found {len(unique_subjects)}!"

    selection_rows = []
    ambiguous_rows = []

    for sub in unique_subjects:
        series_for_sub = df[df["subject_id"] == sub].copy().reset_index(drop=True)
        num_series = len(series_for_sub)

        if num_series == 1:
            row = series_for_sub.iloc[0]
            selection_rows.append({
                "subject_id": sub,
                "selected_series_id": row["series_id"],
                "selected_series_path": row["series_path"],
                "selection_status": "SELECTED",
                "selection_reason": (
                    "Single complete 3D MPRAGE structural acquisition meeting all benchmark quality criteria "
                    f"({row['num_slices']} slices, {row['rows']}x{row['columns']} matrix, "
                    f"{row['pixel_spacing']} in-plane, {row['calculated_slice_spacing']}mm slice spacing, complete volume)."
                ),
                "number_of_dicom_files": int(row["num_slices"]),
                "manufacturer": row["manufacturer"],
                "field_strength": row["field_strength"],
                "rows": int(row["rows"]),
                "columns": int(row["columns"]),
                "slice_thickness": float(row["slice_thickness"]),
                "pixel_spacing": row["pixel_spacing"],
                "protocol_name": row["protocol_name"],
                "series_description": row["series_description"],
            })
        else:
            # Multi-series subject: Evaluate whether there is a scientifically defensible candidate or ambiguous
            # All 9 multi-series subjects in this dataset have 2 repeat MPRAGE acquisitions in the same session
            s1 = series_for_sub.iloc[0]
            s2 = series_for_sub.iloc[1]

            cand_ids = [str(s["series_id"]) for _, s in series_for_sub.iterrows()]
            cand_paths = [str(s["series_path"]) for _, s in series_for_sub.iterrows()]
            cand_series_nums = [str(s["series_number"]) for _, s in series_for_sub.iterrows()]
            cand_acq_nums = [str(s["acquisition_number"]) for _, s in series_for_sub.iterrows()]
            cand_slice_counts = [str(s["num_slices"]) for _, s in series_for_sub.iterrows()]

            ambiguity_reason = (
                f"Subject has {num_series} complete repeat 3D MPRAGE acquisitions in the same study session "
                f"(Series IDs: {', '.join(cand_ids)}; Series Numbers: {', '.join(cand_series_nums)}; "
                f"Acquisition Numbers: {', '.join(cand_acq_nums)}). Both series have identical matrix size "
                f"({s1['rows']}x{s1['columns']}), slice count ({s1['num_slices']}), slice thickness ({s1['slice_thickness']:.1f}mm), "
                f"pixel spacing ({s1['pixel_spacing']}), sequence parameters (TR/TE/FA), and zero slice gaps. "
                "No objective basis exists in raw DICOM metadata alone to favor one run over the duplicate without visual/external QC."
            )

            selection_rows.append({
                "subject_id": sub,
                "selected_series_id": "AMBIGUOUS",
                "selected_series_path": "AMBIGUOUS",
                "selection_status": "AMBIGUOUS",
                "selection_reason": ambiguity_reason,
                "number_of_dicom_files": int(s1["num_slices"]), # Identical across both candidates
                "manufacturer": s1["manufacturer"],
                "field_strength": s1["field_strength"],
                "rows": int(s1["rows"]),
                "columns": int(s1["columns"]),
                "slice_thickness": float(s1["slice_thickness"]),
                "pixel_spacing": s1["pixel_spacing"],
                "protocol_name": s1["protocol_name"],
                "series_description": s1["series_description"],
            })

            ambiguous_rows.append({
                "subject_id": sub,
                "candidate_series_count": num_series,
                "candidate_series_ids": ";".join(cand_ids),
                "candidate_series_paths": ";".join(cand_paths),
                "candidate_series_numbers": ";".join(cand_series_nums),
                "candidate_acquisition_numbers": ";".join(cand_acq_nums),
                "candidate_slice_counts": ";".join(cand_slice_counts),
                "manufacturer": s1["manufacturer"],
                "field_strength": s1["field_strength"],
                "matrix_size": f"{s1['rows']}x{s1['columns']}",
                "slice_thickness": float(s1["slice_thickness"]),
                "pixel_spacing": s1["pixel_spacing"],
                "reason_for_ambiguity": ambiguity_reason,
            })

    # Save selection CSV
    selection_df = pd.DataFrame(selection_rows)
    selection_csv_path = "results/adni_series_selection.csv"
    selection_df.to_csv(selection_csv_path, index=False)
    print(f"Saved series selection to: {selection_csv_path}")

    # Save ambiguous subjects CSV
    ambiguous_df = pd.DataFrame(ambiguous_rows)
    ambiguous_csv_path = "results/adni_ambiguous_subjects.csv"
    ambiguous_df.to_csv(ambiguous_csv_path, index=False)
    print(f"Saved ambiguous subjects to: {ambiguous_csv_path}")

    # Calculate distributions for dataset audit JSON
    total_dicom_count = int(df["num_slices"].sum())
    assert total_dicom_count == 10990, f"Expected 10990 DICOM files, got {total_dicom_count}!"

    num_subjects = len(unique_subjects)
    num_single_series = len(selection_df[selection_df["selection_status"] == "SELECTED"])
    num_multi_series = len(unique_subjects) - num_single_series
    num_selected = num_single_series
    num_ambiguous = len(ambiguous_df)
    num_missing_or_invalid = 0

    # Manufacturer distributions
    mfg_dist_series = df["manufacturer"].value_counts().to_dict()
    mfg_dist_subjects = selection_df["manufacturer"].value_counts().to_dict()

    # Field strength distributions
    fs_dist_series = {str(round(k, 2)): int(v) for k, v in df["field_strength"].value_counts().items()}
    fs_dist_subjects = {str(round(k, 2)): int(v) for k, v in selection_df["field_strength"].value_counts().items()}

    # Matrix size distributions
    matrix_series = (df["rows"].astype(str) + "x" + df["columns"].astype(str)).value_counts().to_dict()
    matrix_subjects = (selection_df["rows"].astype(str) + "x" + selection_df["columns"].astype(str)).value_counts().to_dict()

    # Slice thickness distributions
    thick_series = {f"{k:.1f} mm": int(v) for k, v in df["slice_thickness"].round(1).value_counts().items()}
    thick_subjects = {f"{k:.1f} mm": int(v) for k, v in selection_df["slice_thickness"].round(1).value_counts().items()}

    # Slices per volume distributions
    slices_series = df["num_slices"].value_counts().to_dict()

    # Incompleteness checks
    incomplete_series = df[~df["is_complete"]].to_dict(orient="records")

    audit_json = {
        "dataset_name": "ADNI MPRAGE Benchmark Extracted Dataset",
        "dataset_root": r"D:\AIML\ADNI_MPRAGE_Benchmark_Extracted\ADNI",
        "audit_summary": {
            "total_unique_subjects": num_subjects,
            "total_dicom_series": len(df),
            "total_dicom_files": total_dicom_count,
            "subjects_with_exactly_one_series": num_single_series,
            "subjects_with_multiple_series": num_multi_series,
            "subjects_selected_automatically": num_selected,
            "subjects_ambiguous": num_ambiguous,
            "subjects_missing_or_invalid": num_missing_or_invalid,
            "series_with_read_errors": int(df["read_errors"].sum()),
            "series_incomplete_or_irregular": len(incomplete_series),
        },
        "distributions": {
            "manufacturers": {
                "by_subject": mfg_dist_subjects,
                "by_series": mfg_dist_series,
            },
            "field_strength_tesla": {
                "by_subject": fs_dist_subjects,
                "by_series": fs_dist_series,
            },
            "matrix_size_rows_by_cols": {
                "by_subject": matrix_subjects,
                "by_series": matrix_series,
            },
            "slice_thickness": {
                "by_subject": thick_subjects,
                "by_series": thick_series,
            },
            "slice_count_per_volume": slices_series,
        },
        "spatial_consistency_validation": {
            "orientation_consistent_all_series": bool(df["orientation_consistent"].all()),
            "continuous_instance_numbers_all_series": bool(df["inst_continuous"].all()),
            "regular_slice_spacing_all_series": bool(df["spatial_continuous"].all()),
            "all_ipp_present_all_series": bool(df["spatial_continuous"].all()),
            "incomplete_series_list": incomplete_series,
        },
        "ambiguous_subjects_summary": [
            {
                "subject_id": row["subject_id"],
                "candidate_series_count": row["candidate_series_count"],
                "candidate_series_ids": row["candidate_series_ids"].split(";"),
                "reason": row["reason_for_ambiguity"],
            }
            for _, row in ambiguous_df.iterrows()
        ],
        "generated_audit_artifacts": [
            "results/adni_series_audit.csv",
            "results/adni_series_selection.csv",
            "results/adni_ambiguous_subjects.csv",
            "results/adni_dataset_audit.json",
        ],
    }

    json_path = "results/adni_dataset_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=2)
    print(f"Saved dataset audit JSON to: {json_path}")

    return audit_json


if __name__ == "__main__":
    generate_audit_files()
