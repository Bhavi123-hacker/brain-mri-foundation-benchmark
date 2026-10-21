"""
Reproducible Cohort Builder & Clinical Verification Script for OASIS-1 Benchmark.
Processes C:\\Users\\Lenovo\\Desktop\\oasis\\oasis_cross-sectional-5708aa0a98d82080.xlsx
Generates local, git-ignored cohort manifests in D:\\brain_mri_data\\OASIS1\\manifests\\
"""

import os
import sys
import json
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
SPREADSHEET_PATH = r"C:\Users\Lenovo\Desktop\oasis\oasis_cross-sectional-5708aa0a98d82080.xlsx"
RELIABILITY_PATH = r"C:\Users\Lenovo\Desktop\oasis\oasis_cross-sectional-reliability-063c8642b909ee76.xlsx"
ARCHIVE_DIR = r"C:\Users\Lenovo\Desktop\oasis"
OUTPUT_DIR = r"D:\brain_mri_data\OASIS1\manifests"

def build_oasis1_primary_cohort():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 1: RE-READING AND VERIFYING OASIS-1 CLINICAL SPREADSHEET")
    print("=" * 80)
    
    assert os.path.exists(SPREADSHEET_PATH), f"Spreadsheet missing: {SPREADSHEET_PATH}"
    excel_file = pd.ExcelFile(SPREADSHEET_PATH)
    sheet_name = excel_file.sheet_names[0]
    df = pd.read_excel(SPREADSHEET_PATH, sheet_name=sheet_name)
    
    total_raw_rows = len(df)
    print(f"Spreadsheet loaded: '{sheet_name}', total rows: {total_raw_rows}, columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    # Subject and session parsing
    df["session_id"] = df["ID"].astype(str).str.strip()
    df["base_subject_id"] = df["session_id"].apply(lambda s: "_".join(s.split("_")[:2]))
    df["session_type"] = df["session_id"].apply(lambda s: s.split("_")[2] if len(s.split("_")) > 2 else "UNKNOWN")
    
    # Verify uniqueness
    assert df["session_id"].nunique() == total_raw_rows, "Duplicate session_id detected in spreadsheet!"
    unique_base_subjects = df["base_subject_id"].nunique()
    print(f"Unique session IDs: {total_raw_rows}, Unique base subjects: {unique_base_subjects}")
    
    # Categorization and Inclusion/Exclusion Audit
    records = []
    
    # Counts
    count_total = total_raw_rows
    count_mr2_reliability = 0
    count_mr1 = 0
    count_missing_cdr = 0
    count_cdr_notna = 0
    count_under_60 = 0
    count_eligible_primary = 0
    count_cn = 0
    count_dementia = 0
    
    for idx, row in df.iterrows():
        sid = row["session_id"]
        base_id = row["base_subject_id"]
        stype = row["session_type"]
        age = float(row["Age"])
        raw_cdr = row["CDR"]
        mmse = float(row["MMSE"]) if pd.notna(row["MMSE"]) else None
        gender = str(row["M/F"])
        educ = float(row["Educ"]) if pd.notna(row["Educ"]) else None
        ses = float(row["SES"]) if pd.notna(row["SES"]) else None
        etiv = float(row["eTIV"])
        nwbv = float(row["nWBV"])
        asf = float(row["ASF"])
        delay = float(row["Delay"]) if pd.notna(row["Delay"]) else None
        
        # Determine status
        is_mr2 = (stype == "MR2")
        has_cdr = pd.notna(raw_cdr)
        cdr_val = float(raw_cdr) if has_cdr else None
        age_ge_60 = (age >= 60.0)
        
        inclusion_status = "EXCLUDED"
        exclusion_reasons = []
        clinical_stage = "UNKNOWN"
        binary_label = None # 0.0 = CN, 1.0 = Dementia
        
        if is_mr2:
            count_mr2_reliability += 1
            exclusion_reasons.append("Repeat reliability scan (MR2) excluded to prevent longitudinal leakage")
        else:
            count_mr1 += 1
            if not has_cdr:
                count_missing_cdr += 1
                exclusion_reasons.append(f"Missing CDR evaluation (Young/middle-aged cohort, age={int(age)})")
                clinical_stage = "Unrated_Young_Reference"
            else:
                count_cdr_notna += 1
                # Clinical staging descriptor
                if cdr_val == 0.0:
                    clinical_stage = "Cognitively_Normal_CDR_0.0"
                elif cdr_val == 0.5:
                    clinical_stage = "Very_Mild_Dementia_CDR_0.5"
                elif cdr_val == 1.0:
                    clinical_stage = "Mild_Dementia_CDR_1.0"
                elif cdr_val == 2.0:
                    clinical_stage = "Moderate_Dementia_CDR_2.0"
                else:
                    clinical_stage = f"Uncommon_CDR_{cdr_val}"
                
                if not age_ge_60:
                    count_under_60 += 1
                    exclusion_reasons.append(f"Age < 60 (age={int(age)}) excluded to avoid age-confounding with dementia cohort")
                else:
                    # Eligible for primary benchmark!
                    count_eligible_primary += 1
                    inclusion_status = "INCLUDED"
                    if cdr_val == 0.0:
                        binary_label = 0.0 # Cognitively Normal
                        count_cn += 1
                    elif cdr_val >= 0.5:
                        binary_label = 1.0 # Clinically rated dementia / impairment
                        count_dementia += 1
        
        rec = {
            "session_id": sid,
            "base_subject_id": base_id,
            "session_type": stype,
            "age": age,
            "gender": gender,
            "cdr": cdr_val,
            "clinical_stage": clinical_stage,
            "binary_label": binary_label,
            "label_description": "CN" if binary_label == 0.0 else ("Dementia" if binary_label == 1.0 else "Unlabeled"),
            "mmse": mmse,
            "educ": educ,
            "ses": ses,
            "etiv": etiv,
            "nwbv": nwbv,
            "asf": asf,
            "delay": delay,
            "inclusion_status": inclusion_status,
            "exclusion_reason": "; ".join(exclusion_reasons) if exclusion_reasons else "None (Meets all inclusion criteria)"
        }
        records.append(rec)
        
    audit_df = pd.DataFrame(records)
    
    print("\n" + "=" * 80)
    print("INCLUSION / EXCLUSION AUDIT SUMMARY")
    print("=" * 80)
    print(f"Total rows inspected:                           {count_total}")
    print(f"  - Repeat reliability sessions (MR2) excluded: {count_mr2_reliability}")
    print(f"  - Baseline sessions (MR1):                    {count_mr1}")
    print(f"      * Unrated CDR (ages 18-58) excluded:      {count_missing_cdr}")
    print(f"      * Rated CDR sessions:                     {count_cdr_notna}")
    print(f"          - Age < 60 (CDR=0.0) excluded:        {count_under_60}")
    print(f"          - Age >= 60 PRIMARY ELIGIBLE COHORT:  {count_eligible_primary}")
    print(f"              * Cognitively Normal (CDR 0.0):   {count_cn} ({count_cn/count_eligible_primary*100:.1f}%)")
    print(f"              * Dementia (CDR >= 0.5):          {count_dementia} ({count_dementia/count_eligible_primary*100:.1f}%)")

    # Detailed CDR breakdown in primary eligible cohort
    primary_cohort_df = audit_df[audit_df["inclusion_status"] == "INCLUDED"].copy().reset_index(drop=True)
    
    print("\n" + "=" * 80)
    print("PRIMARY ELIGIBLE COHORT CLINICAL STAGING BREAKDOWN (N = 198)")
    print("=" * 80)
    stage_counts = primary_cohort_df["clinical_stage"].value_counts()
    for st, cnt in stage_counts.items():
        print(f"  {st:35s}: {cnt:3d} subjects ({cnt/len(primary_cohort_df)*100:5.2f}%)")
        
    print("\nAge distributions in Primary Eligible Cohort:")
    cn_ages = primary_cohort_df[primary_cohort_df["binary_label"] == 0.0]["age"]
    dem_ages = primary_cohort_df[primary_cohort_df["binary_label"] == 1.0]["age"]
    print(f"  CN (N={len(cn_ages)}):       Mean={cn_ages.mean():.2f} ± {cn_ages.std():.2f}, Range=[{cn_ages.min():.0f}, {cn_ages.max():.0f}]")
    print(f"  Dementia (N={len(dem_ages)}): Mean={dem_ages.mean():.2f} ± {dem_ages.std():.2f}, Range=[{dem_ages.min():.0f}, {dem_ages.max():.0f}]")
    
    print("\nGender distributions in Primary Eligible Cohort:")
    print(pd.crosstab(primary_cohort_df["label_description"], primary_cohort_df["gender"], margins=True))

    print("\n" + "=" * 80)
    print("MAPPING PRIMARY SESSIONS TO 12 DISK ARCHIVES")
    print("=" * 80)
    
    # Find which disc contains each session
    archive_map = {}
    for i in range(1, 13):
        tar_name = f"oasis_cross-sectional_disc{i}.tar.gz"
        tar_path = os.path.join(ARCHIVE_DIR, tar_name)
        if not os.path.exists(tar_path):
            continue
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                # Members have format: discX/OAS1_XXXX_MR1/...
                parts = member.name.split("/")
                if len(parts) >= 2 and parts[1].startswith("OAS1_"):
                    sid = parts[1]
                    if sid not in archive_map:
                        archive_map[sid] = {
                            "archive_name": tar_name,
                            "archive_path": tar_path,
                            "disc_folder": parts[0]
                        }

    # Verify every primary cohort subject is found in the archives
    missing_in_archives = []
    primary_records_with_paths = []
    
    for idx, row in primary_cohort_df.iterrows():
        sid = row["session_id"]
        if sid in archive_map:
            arch_info = archive_map[sid]
            rec_dict = row.to_dict()
            rec_dict["archive_name"] = arch_info["archive_name"]
            rec_dict["archive_path"] = arch_info["archive_path"]
            rec_dict["archive_session_prefix"] = f"{arch_info['disc_folder']}/{sid}"
            # Expected raw/processed file paths
            rec_dict["expected_raw_hdr"] = f"{arch_info['disc_folder']}/{sid}/RAW/{sid}_mpr-1_anon.hdr"
            rec_dict["expected_raw_img"] = f"{arch_info['disc_folder']}/{sid}/RAW/{sid}_mpr-1_anon.img"
            rec_dict["expected_t88_gfc_hdr"] = f"{arch_info['disc_folder']}/{sid}/PROCESSED/MPRAGE/T88_111/{sid}_mpr_n4_anon_111_t88_gfc.hdr"
            rec_dict["expected_t88_gfc_img"] = f"{arch_info['disc_folder']}/{sid}/PROCESSED/MPRAGE/T88_111/{sid}_mpr_n4_anon_111_t88_gfc.img"
            rec_dict["expected_sbj_hdr"] = f"{arch_info['disc_folder']}/{sid}/PROCESSED/MPRAGE/SUBJ_111/{sid}_mpr_n4_anon_sbj_111.hdr"
            rec_dict["expected_sbj_img"] = f"{arch_info['disc_folder']}/{sid}/PROCESSED/MPRAGE/SUBJ_111/{sid}_mpr_n4_anon_sbj_111.img"
            primary_records_with_paths.append(rec_dict)
        else:
            missing_in_archives.append(sid)

    print(f"Primary cohort subjects mapped to archives: {len(primary_records_with_paths)} / {len(primary_cohort_df)}")
    print(f"Subjects missing in archives:               {len(missing_in_archives)}")
    assert len(missing_in_archives) == 0, f"Error: {len(missing_in_archives)} primary subjects not found in archives!"
    
    manifest_df = pd.DataFrame(primary_records_with_paths)
    
    # Save Outputs
    manifest_csv_path = os.path.join(OUTPUT_DIR, "oasis1_primary_cohort_manifest.csv")
    manifest_json_path = os.path.join(OUTPUT_DIR, "oasis1_primary_cohort_manifest.json")
    audit_csv_path = os.path.join(OUTPUT_DIR, "oasis1_full_audit_table.csv")
    summary_json_path = os.path.join(OUTPUT_DIR, "oasis1_cohort_summary_report.json")
    
    manifest_df.to_csv(manifest_csv_path, index=False)
    manifest_df.to_json(manifest_json_path, orient="records", indent=2)
    audit_df.to_csv(audit_csv_path, index=False)
    
    summary_report = {
        "dataset_name": "OASIS-1 Cross-Sectional",
        "spreadsheet_source": SPREADSHEET_PATH,
        "total_spreadsheet_rows": count_total,
        "unique_base_subjects": unique_base_subjects,
        "repeat_reliability_mr2_excluded": count_mr2_reliability,
        "unrated_young_cohort_excluded": count_missing_cdr,
        "under_60_cdr0_excluded": count_under_60,
        "primary_eligible_cohort_total": count_eligible_primary,
        "primary_class_counts": {
            "CN_label_0.0": count_cn,
            "Dementia_label_1.0": count_dementia
        },
        "primary_clinical_stages": stage_counts.to_dict(),
        "age_statistics": {
            "CN": {"mean": float(cn_ages.mean()), "std": float(cn_ages.std()), "min": float(cn_ages.min()), "max": float(cn_ages.max())},
            "Dementia": {"mean": float(dem_ages.mean()), "std": float(dem_ages.std()), "min": float(dem_ages.min()), "max": float(dem_ages.max())}
        },
        "all_primary_subjects_verified_in_archives": (len(missing_in_archives) == 0),
        "manifest_csv_path": manifest_csv_path,
        "manifest_json_path": manifest_json_path
    }
    
    with open(summary_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
        
    print(f"\nArtifacts successfully generated in {OUTPUT_DIR}:")
    print(f"  - {manifest_csv_path}")
    print(f"  - {manifest_json_path}")
    print(f"  - {summary_json_path}")
    print(f"  - {audit_csv_path}")
    print("\nPhase 1 verification complete.")

if __name__ == "__main__":
    build_oasis1_primary_cohort()
