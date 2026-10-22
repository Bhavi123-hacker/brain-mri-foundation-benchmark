import os
import torch
import pandas as pd

# 1. OASIS-1 check
o1_splits_csv = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
o1_tensors_dir = r"D:\brain_mri_data\OASIS1\processed\tensors"
df_o1 = pd.read_csv(o1_splits_csv)

print(f"Checking {len(df_o1)} OASIS-1 scans...")
o1_found = 0
for _, row in df_o1.iterrows():
    sid = str(row["session_id"]).strip()
    pt = os.path.join(o1_tensors_dir, f"{sid}.pt")
    if os.path.exists(pt):
        o1_found += 1
print(f"OASIS-1: Found {o1_found}/{len(df_o1)} tensors on disk.")

# 2. ADNI check
adni_manifest_csv = r"D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv"
adni_tensors_dir = r"D:\brain_mri_data\ADNI\processed\tensors"
df_adni = pd.read_csv(adni_manifest_csv)

print(f"\nChecking {len(df_adni)} ADNI scans...")
adni_found = 0
for _, row in df_adni.iterrows():
    sid = str(row["subject_id"]).strip()
    img_id = str(row["image_data_id"]).strip()
    pt_name = f"{sid}_{img_id}.pt"
    pt = os.path.join(adni_tensors_dir, pt_name)
    if os.path.exists(pt):
        adni_found += 1
    else:
        # Check alternative naming
        matches = [f for f in os.listdir(adni_tensors_dir) if f.startswith(sid) and f.endswith(".pt")]
        if matches:
            adni_found += 1

print(f"ADNI: Found {adni_found}/{len(df_adni)} tensors on disk.")
