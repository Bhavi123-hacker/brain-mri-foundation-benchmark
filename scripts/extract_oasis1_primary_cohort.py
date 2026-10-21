"""
Step 1: Selective OASIS-1 Volumetric Scan Extractor.
Extracts both T88_111 and SUBJ_111 representations for the 198 allowlisted subjects.
Destination: D:\\brain_mri_data\\OASIS1\\raw\\
Zero modification to original archives. Zero ADNI access.
"""

import os
import sys
import time
import json
import tarfile
import pandas as pd
from pathlib import Path

# Paths
MANIFEST_PATH = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_manifest.csv"
ARCHIVE_DIR = r"C:\Users\Lenovo\Desktop\oasis"
DEST_RAW_DIR = r"D:\brain_mri_data\OASIS1\raw"
LOG_OUTPUT = r"D:\brain_mri_data\OASIS1\manifests\oasis1_extraction_log.json"
SCANS_MANIFEST_OUTPUT = r"D:\brain_mri_data\OASIS1\manifests\oasis1_extracted_scans_manifest.csv"

def extract_primary_cohort():
    start_time = time.time()
    print("=" * 80)
    print("STEP 1: SELECTIVE EXTRACTION OF T88_111 AND SUBJ_111 SCANS (N = 198)")
    print("=" * 80)
    
    assert os.path.exists(MANIFEST_PATH), f"Manifest missing: {MANIFEST_PATH}"
    cohort_df = pd.read_csv(MANIFEST_PATH)
    allowlisted_sessions = set(cohort_df["session_id"].str.strip())
    n_cohort = len(allowlisted_sessions)
    print(f"Allowlisted Cohort Size: {n_cohort} subjects")
    assert n_cohort == 198, f"Expected 198 allowlisted subjects, found {n_cohort}"
    
    t88_dest_dir = os.path.join(DEST_RAW_DIR, "T88_111")
    sbj_dest_dir = os.path.join(DEST_RAW_DIR, "SUBJ_111")
    os.makedirs(t88_dest_dir, exist_ok=True)
    os.makedirs(sbj_dest_dir, exist_ok=True)
    
    extracted_records = []
    extraction_errors = []
    
    # Process each of the 12 archives
    for disc_idx in range(1, 13):
        tar_name = f"oasis_cross-sectional_disc{disc_idx}.tar.gz"
        tar_path = os.path.join(ARCHIVE_DIR, tar_name)
        if not os.path.exists(tar_path):
            extraction_errors.append(f"Archive missing on disk: {tar_path}")
            print(f"ERROR: Archive missing: {tar_path}")
            continue
            
        disc_start = time.time()
        print(f"\nProcessing {tar_name}...", flush=True)
        disc_extracted_count = 0
        
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                # Security / Path Traversal Guard
                if ".." in member.name or member.name.startswith("/") or member.name.startswith("\\"):
                    print(f"WARNING: Skipping unsafe member path: {member.name}")
                    continue
                
                parts = member.name.split("/")
                # Pattern: discX/OAS1_XXXX_MR1/...
                if len(parts) < 2:
                    continue
                session_id = parts[1]
                if session_id not in allowlisted_sessions:
                    continue
                
                # Check target representations
                is_t88 = False
                is_sbj = False
                
                # Target T88_111: GFC volume (hdr and img), excluding masked
                if "PROCESSED/MPRAGE/T88_111" in member.name and member.name.endswith((".hdr", ".img")):
                    if "_t88_gfc" in member.name and "masked" not in member.name:
                        is_t88 = True
                        
                # Target SUBJ_111: Subject native 1.0mm volume (hdr and img)
                elif "PROCESSED/MPRAGE/SUBJ_111" in member.name and member.name.endswith((".hdr", ".img")):
                    if "_sbj_111" in member.name:
                        is_sbj = True
                        
                if not (is_t88 or is_sbj):
                    continue
                
                # Determine target destination directory
                rep_name = "T88_111" if is_t88 else "SUBJ_111"
                rep_root = t88_dest_dir if is_t88 else sbj_dest_dir
                target_subj_dir = os.path.join(rep_root, session_id)
                os.makedirs(target_subj_dir, exist_ok=True)
                
                filename = os.path.basename(member.name)
                dest_file_path = os.path.join(target_subj_dir, filename)
                
                # Guard against accidental overwrite
                if os.path.exists(dest_file_path):
                    print(f"Warning: Destination file already exists: {dest_file_path}")
                    
                # Extract file contents safely
                f_in = tar.extractfile(member)
                if f_in is not None:
                    with open(dest_file_path, "wb") as f_out:
                        f_out.write(f_in.read())
                    f_size = os.path.getsize(dest_file_path)
                    disc_extracted_count += 1
                    
                    extracted_records.append({
                        "session_id": session_id,
                        "representation": rep_name,
                        "filename": filename,
                        "file_ext": os.path.splitext(filename)[1],
                        "dest_path": dest_file_path,
                        "file_size_bytes": f_size,
                        "archive_source": tar_name,
                        "archive_member_path": member.name,
                    })
                else:
                    extraction_errors.append(f"Failed to read member stream: {member.name}")
                    
        elapsed_disc = time.time() - disc_start
        print(f"  Extracted {disc_extracted_count} volume files from {tar_name} in {elapsed_disc:.1f}s")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("EXTRACTION POST-CHECK & COMPLETENESS VERIFICATION")
    print("=" * 80)
    print(f"Total volume files extracted: {len(extracted_records)}")
    print(f"Extraction errors encountered: {len(extraction_errors)}")
    
    # Subject-level completeness check
    extracted_df = pd.DataFrame(extracted_records)
    
    coverage_report = {}
    for rep in ["T88_111", "SUBJ_111"]:
        rep_df = extracted_df[extracted_df["representation"] == rep]
        subjs_with_hdr = set(rep_df[rep_df["file_ext"] == ".hdr"]["session_id"])
        subjs_with_img = set(rep_df[rep_df["file_ext"] == ".img"]["session_id"])
        subjs_complete = subjs_with_hdr.intersection(subjs_with_img)
        missing_hdr = allowlisted_sessions - subjs_with_hdr
        missing_img = allowlisted_sessions - subjs_with_img
        
        coverage_report[rep] = {
            "total_files": len(rep_df),
            "subjects_complete_pairs": len(subjs_complete),
            "missing_hdr_count": len(missing_hdr),
            "missing_img_count": len(missing_img),
            "missing_subjects": sorted(list(allowlisted_sessions - subjs_complete))
        }
        
        print(f"\n--- Representation: {rep} ---")
        print(f"  Total files extracted:          {len(rep_df)}")
        print(f"  Subjects with full (.hdr+.img): {len(subjs_complete)} / {n_cohort} ({len(subjs_complete)/n_cohort*100:.1f}%)")
        print(f"  Missing HDR files:              {len(missing_hdr)}")
        print(f"  Missing IMG files:              {len(missing_img)}")

    # Build per-subject scan path catalog
    subject_scans = []
    for sid in sorted(list(allowlisted_sessions)):
        meta_row = cohort_df[cohort_df["session_id"] == sid].iloc[0]
        t88_dir = os.path.join(t88_dest_dir, sid)
        sbj_dir = os.path.join(sbj_dest_dir, sid)
        
        t88_hdrs = list(Path(t88_dir).glob("*.hdr")) if os.path.exists(t88_dir) else []
        t88_imgs = list(Path(t88_dir).glob("*.img")) if os.path.exists(t88_dir) else []
        sbj_hdrs = list(Path(sbj_dir).glob("*.hdr")) if os.path.exists(sbj_dir) else []
        sbj_imgs = list(Path(sbj_dir).glob("*.img")) if os.path.exists(sbj_dir) else []
        
        subject_scans.append({
            "session_id": sid,
            "base_subject_id": meta_row["base_subject_id"],
            "age": meta_row["age"],
            "gender": meta_row["gender"],
            "cdr": meta_row["cdr"],
            "binary_label": meta_row["binary_label"],
            "label_description": meta_row["label_description"],
            "t88_hdr_path": str(t88_hdrs[0]) if t88_hdrs else None,
            "t88_img_path": str(t88_imgs[0]) if t88_imgs else None,
            "sbj_hdr_path": str(sbj_hdrs[0]) if sbj_hdrs else None,
            "sbj_img_path": str(sbj_imgs[0]) if sbj_imgs else None,
            "is_t88_complete": len(t88_hdrs) == 1 and len(t88_imgs) == 1,
            "is_sbj_complete": len(sbj_hdrs) == 1 and len(sbj_imgs) == 1,
        })
        
    scans_manifest_df = pd.DataFrame(subject_scans)
    scans_manifest_df.to_csv(SCANS_MANIFEST_OUTPUT, index=False)
    
    total_bytes_extracted = sum(r["file_size_bytes"] for r in extracted_records)
    total_mb_extracted = total_bytes_extracted / (1024**2)
    
    final_summary = {
        "status": "COMPLETED",
        "cohort_size_requested": n_cohort,
        "total_files_extracted": len(extracted_records),
        "total_extracted_mb": round(total_mb_extracted, 2),
        "total_time_seconds": round(total_time, 2),
        "t88_coverage": coverage_report["T88_111"],
        "sbj_coverage": coverage_report["SUBJ_111"],
        "extraction_errors": extraction_errors,
        "scans_manifest_csv": SCANS_MANIFEST_OUTPUT
    }
    
    with open(LOG_OUTPUT, "w") as f:
        json.dump(final_summary, f, indent=2)
        
    print(f"\nTotal extraction size: {total_mb_extracted:.2f} MB (~{total_mb_extracted/1024:.2f} GB)")
    print(f"Total extraction time: {total_time:.1f}s")
    print(f"Log written to:         {LOG_OUTPUT}")
    print(f"Scans manifest saved:   {SCANS_MANIFEST_OUTPUT}")
    print("\nStep 1 extraction completed successfully.")

if __name__ == "__main__":
    extract_primary_cohort()
