# Forensic Split Integrity and Patient Isolation Audit

- **Audit Date**: 2026-09-14T14:22:51.234863
- **Verdict**: **AUDIT PASSED - ZERO LEAKAGE CONFIRMED**
- **Test Leakage Detected**: **False**

## 1. Verified Locked Test Sets

### OASIS-1 Pilot (N=8)
- Ground truth file: `D:/OASIS1_PILOT/pilot_splits.csv`
- Subjects: `OAS1_0122_MR1, OAS1_0033_MR1, OAS1_0070_MR1, OAS1_0053_MR1, OAS1_0072_MR1, OAS1_0373_MR1, OAS1_0074_MR1, OAS1_0223_MR1`
- Class distribution: 4 CN (`OAS1_0033_MR1`, `OAS1_0070_MR1`, `OAS1_0072_MR1`, `OAS1_0074_MR1`), 4 Dementia (`OAS1_0053_MR1`, `OAS1_0122_MR1`, `OAS1_0223_MR1`, `OAS1_0373_MR1`)
- Overlap with development/train/val: **0 subjects (Disjointness 100%)**

### ADNI Benchmark (N=8)
- Ground truth file: `results/adni_cn_ad_split.csv`
- Subjects: `023_S_0058, 023_S_1190, 027_S_0404, 031_S_1209, 033_S_0920, 051_S_1123, 067_S_0812, 116_S_0487`
- Class distribution: 4 CN (`023_S_0058`, `023_S_1190`, `033_S_0920`, `051_S_1123`), 4 AD (`027_S_0404`, `031_S_1209`, `067_S_0812`, `116_S_0487`)
- Overlap with development/train/val: **0 subjects (Disjointness 100%)**

## 2. Leakage Protection Verification
- All selection / tuning flags for test subjects: **FALSE**
- Test subjects in CV folds: **NONE**
- Test subjects in Grad-CAM selection: **NONE**
