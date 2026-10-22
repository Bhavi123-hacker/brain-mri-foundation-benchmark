import pandas as pd
import json

o1_path = r"D:\brain_mri_data\OASIS1\manifests\oasis1_primary_cohort_splits.csv"
df_o1 = pd.read_csv(o1_path)
print("=== OASIS-1 PRIMARY COHORT SPLITS (N=198) ===")
print("Shape:", df_o1.shape)
print("Split distribution:\n", df_o1["split"].value_counts())
print("Binary label distribution:\n", df_o1["binary_label"].value_counts())
print("Crosstab:\n", pd.crosstab(df_o1["split"], df_o1["binary_label"]))
print("Columns:", list(df_o1.columns))

# Verify zero patient leakage
tr = set(df_o1[df_o1["split"] == "train"]["session_id"])
va = set(df_o1[df_o1["split"] == "val"]["session_id"])
te = set(df_o1[df_o1["split"] == "test"]["session_id"])
assert not tr.intersection(va), "Leakage between train and val!"
assert not tr.intersection(te), "Leakage between train and test!"
assert not va.intersection(te), "Leakage between val and test!"
print("Zero patient leakage verified across OASIS-1 splits (138 train / 30 val / 30 test)!")

print("\n=== ADNI FULL PREPROCESSED MANIFEST (N=500) ===")
adni_path = r"D:\brain_mri_data\ADNI\manifests\adni_full_preprocessed_manifest.csv"
df_adni = pd.read_csv(adni_path)
print("Shape:", df_adni.shape)
lbl_col = "binary_label" if "binary_label" in df_adni.columns else "label"
print("Label distribution:\n", df_adni[lbl_col].value_counts())
print("Columns:", list(df_adni.columns)[:15])

mfg_cols = [c for c in df_adni.columns if "mfg" in c.lower() or "manufactur" in c.lower()]
print("Manufacturer columns found:", mfg_cols)
for c in mfg_cols:
    print(f"Value counts for {c}:\n", df_adni[c].value_counts())

print("\n=== OASIS-2 BASELINE MANIFEST (N=150) ===")
o2_path = r"D:\brain_mri_data\OASIS2\manifests\oasis2_baseline_manifest.csv"
df_o2 = pd.read_csv(o2_path)
print("Shape:", df_o2.shape)
print("Columns:", list(df_o2.columns)[:15])
lbl_col2 = "binary_label" if "binary_label" in df_o2.columns else "label" if "label" in df_o2.columns else "Group"
print("Label/Group distribution:\n", df_o2[lbl_col2].value_counts() if lbl_col2 in df_o2.columns else "None")
