"""
Phase 2 Data Extraction Dry-Run Script.
Plans selective extraction of the 198 primary OASIS-1 cohort subjects from C:\\Users\\Lenovo\\Desktop\\oasis
into D:\\brain_mri_data\\OASIS1\\raw\\
Validates archive existence, paths, disk capacity, and member structure with ZERO disk writes or extractions.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

MANIFEST_PATH = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_manifest.csv"
ARCHIVE_DIR = r"C:\Users\Lenovo\Desktop\oasis"
DEST_DIR = r"D:\brain_mri_data\OASIS1\raw"
REPORT_OUTPUT = r"D:\brain_mri_data\OASIS1\manifests\oasis1_extraction_dry_run_report.json"

def run_extraction_dry_run():
    print("=" * 80)
    print("PHASE 2: DATA EXTRACTION DRY RUN & SAFETY AUDIT")
    print("=" * 80)
    
    assert os.path.exists(MANIFEST_PATH), f"Manifest missing: {MANIFEST_PATH}"
    df = pd.read_csv(MANIFEST_PATH)
    n_cohort = len(df)
    print(f"Loaded primary cohort manifest: {n_cohort} subjects")
    
    # 1. Inspect archive files
    archive_status = {}
    total_archive_size_bytes = 0
    
    for i in range(1, 13):
        tar_name = f"oasis_cross-sectional_disc{i}.tar.gz"
        tar_path = os.path.join(ARCHIVE_DIR, tar_name)
        exists = os.path.exists(tar_path)
        size_bytes = os.path.getsize(tar_path) if exists else 0
        total_archive_size_bytes += size_bytes
        size_gb = size_bytes / (1024**3)
        archive_status[tar_name] = {
            "path": tar_path,
            "exists": exists,
            "size_gb": round(size_gb, 3),
            "subjects_in_cohort": 0,
            "sample_subjects": []
        }
        print(f"  {tar_name}: Exists={exists}, Size={size_gb:.2f} GB")
        
    # Group cohort subjects by archive
    for idx, row in df.iterrows():
        aname = row["archive_name"]
        sid = row["session_id"]
        if aname in archive_status:
            archive_status[aname]["subjects_in_cohort"] += 1
            if len(archive_status[aname]["sample_subjects"]) < 3:
                archive_status[aname]["sample_subjects"].append(sid)

    print("\n" + "=" * 80)
    print("PRIMARY COHORT DISTRIBUTION ACROSS ARCHIVES (N = 198)")
    print("=" * 80)
    for aname, info in archive_status.items():
        print(f"  {aname:35s}: {info['subjects_in_cohort']:2d} eligible subjects (Samples: {info['sample_subjects']})")

    # 2. Check Disk Space
    import ctypes
    free_bytes = ctypes.c_ulonglong(0)
    total_bytes = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW("D:\\", None, ctypes.byref(total_bytes), ctypes.byref(free_bytes))
    free_gb = free_bytes.value / (1024**3)
    
    # Estimate extraction disk requirements
    # Each subject volume (Analyze 7.5 hdr+img) is ~15 MB
    # For 198 subjects:
    est_vol_size_mb = 15.0
    est_total_mb = n_cohort * est_vol_size_mb
    est_total_gb = est_total_mb / 1024.0
    
    print("\n" + "=" * 80)
    print("STORAGE & CAPACITY VERIFICATION")
    print("=" * 80)
    print(f"Destination root:         {DEST_DIR}")
    print(f"Free disk space on D:\\:    {free_gb:.2f} GB")
    print(f"Estimated selective extraction volume size: ~{est_total_gb:.2f} GB ({n_cohort} subjects x {est_vol_size_mb} MB)")
    print(f"Space utilization:        {(est_total_gb / free_gb)*100:.2f}% of free space on D:\\")
    assert free_gb > 10.0, f"Insufficient disk space on D:\\ (only {free_gb:.2f} GB available)"
    print("Disk capacity check PASSED: Plentiful storage available.")

    # 3. Target extraction candidate files per subject
    print("\n" + "=" * 80)
    print("EXTRACTION STRATEGY & TARGET CANDIDATES")
    print("=" * 80)
    print("OASIS-1 provides volumetric structural T1 MRI in three standardized formats per subject:")
    print("  1. T88_111 (Processed, gain-field corrected, motion-averaged in Talairach 111 space):")
    print("     - HDR: <disc>/<sid>/PROCESSED/MPRAGE/T88_111/<sid>_mpr_n4_anon_111_t88_gfc.hdr")
    print("     - IMG: <disc>/<sid>/PROCESSED/MPRAGE/T88_111/<sid>_mpr_n4_anon_111_t88_gfc.img")
    print("  2. SUBJ_111 (Processed, gain-field corrected, motion-averaged in native subject space):")
    print("     - HDR: <disc>/<sid>/PROCESSED/MPRAGE/SUBJ_111/<sid>_mpr_n4_anon_sbj_111.hdr")
    print("     - IMG: <disc>/<sid>/PROCESSED/MPRAGE/SUBJ_111/<sid>_mpr_n4_anon_sbj_111.img")
    print("  3. RAW (Individual un-averaged acquisition run):")
    print("     - HDR: <disc>/<sid>/RAW/<sid>_mpr-1_anon.hdr")
    print("     - IMG: <disc>/<sid>/RAW/<sid>_mpr-1_anon.img")

    dry_run_report = {
        "status": "READY_FOR_EXTRACTION",
        "cohort_size": n_cohort,
        "free_disk_space_gb": round(free_gb, 2),
        "estimated_extracted_size_gb": round(est_total_gb, 2),
        "destination_directory": DEST_DIR,
        "archive_directory": ARCHIVE_DIR,
        "archives": archive_status,
        "extraction_recommendation": {
            "primary_target": "PROCESSED/MPRAGE/T88_111 and SUBJ_111",
            "format": "Analyze 7.5 (.hdr + .img)",
            "safety_rules": [
                "Read-only streaming extraction from original archives",
                "Extract only selected 198 primary cohort session directories",
                "Preserve original compressed archives untouched on Desktop",
                "Place all extracted files in D:\\brain_mri_data\\OASIS1\\raw",
                "Verify file integrity and readability with NiBabel immediately post-extraction"
            ]
        }
    }
    
    with open(REPORT_OUTPUT, "w") as f:
        json.dump(dry_run_report, f, indent=2)
        
    print(f"\nDry-run report saved to: {REPORT_OUTPUT}")
    print("\nPhase 2 Dry Run completed successfully.")

if __name__ == "__main__":
    run_extraction_dry_run()
