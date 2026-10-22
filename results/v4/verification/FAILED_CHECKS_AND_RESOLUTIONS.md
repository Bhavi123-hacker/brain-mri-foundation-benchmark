# SHIFT-MRI V4 — Failed Checks and Resolutions Log

**Author:** Bhavy Garg (24BAI1088)  
**Status:** All identified checks resolved and double verified.

---

### Check 1: Checkpoint File Preservation
- **Check:** Verify that every trained model has a preserved PyTorch `.pt` file on disk with valid state dictionary.
- **Result:** PASS. 14/14 deep learning checkpoints verified and reloadable. 1 classical machine learning baseline verified via fitted pipeline.

### Check 2: Sign Convention Consistency
- **Check:** Ensure Generalization Gap is uniformly defined across all figures, tables, and manuscript.
- **Defect in Prior Historical V3 Reports:** Some pilot files used target minus source, while others used source minus target.
- **Resolution:** Uniformly locked to: $\Delta\text{AUC} = \text{AUC}_{\text{OASIS-1 Test}} - \text{AUC}_{\text{ADNI}}$. Positive values indicate generalization drop; negative values indicate out-of-distribution gain.

### Check 3: Scanner Cohort Integrity
- **Check:** Verify GE scanners in ADNI.
- **Result:** Confirmed 0 GE scanners present. ADNI cohort contains strictly Siemens ($N=370, 74.0\%$) and Philips ($N=130, 26.0\%$). Wording corrected across all documentation and frontend text.

### Check 4: Positive Class Calibration
- **Check:** Confirm that probability values represent probability of class 1 (AD / Dementia) consistently across all cohorts.
- **Result:** PASS. Verified against ground truth labels without heuristic flips.
