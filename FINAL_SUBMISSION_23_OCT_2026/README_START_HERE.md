# SHIFT-MRI V4: Final Project Submission

**Student:** Bhavy Garg  
**Registration Number:** 24BAI1088  
**Department:** Artificial Intelligence and Data Science  
**Institution:** Vellore Institute of Technology (VIT), Chennai  
**Date:** Friday, 23 October 2026  

Welcome to the official final submission package for **SHIFT-MRI: Cross-Cohort Generalization and Reliability of Brain-MRI Foundation Models and Supervised Networks**.

---

## Submission Directory Structure

```text
FINAL_SUBMISSION_23_OCT_2026/
├── 01_IEEE_RESEARCH_PAPER/
│   ├── SHIFT_MRI_V4_IEEE_Access_Paper.docx    <- Full 34-section IEEE Access Manuscript (Word)
│   ├── SHIFT_MRI_V4_IEEE_Access_Paper.pdf     <- Matching PDF
│   └── IEEE_Submission_Checklist.md
│
├── 02_TEACHER_PROJECT_REPORT/
│   ├── SHIFT_MRI_V4_Project_Report.docx       <- Complete Academic Project Report (Word)
│   └── SHIFT_MRI_V4_Project_Report.pdf        <- Matching PDF
│
├── 03_EXPERIMENTAL_RESULTS/
│   ├── V4_Master_Model_Comparison.csv         <- All 15 models with OASIS-1, ADNI, OASIS-2 metrics
│   ├── V4_Foundation_Comparison.csv           <- Foundation models vs 3D ResNet-18
│   ├── V4_Three_Cohort_Comparison.csv         <- 10-architecture 3-cohort summary table
│   ├── V4_Statistical_Analysis.csv            <- 9 Matched bootstrap tests with Holm-Bonferroni
│   └── V4_TrustGate_Results.csv               <- Failure detection gating metrics on ADNI
│
├── 04_PROOF_AND_VERIFICATION/
│   ├── Double_Verification_Report.pdf (.md)   <- Independent metric recalculation audit
│   ├── Model_Authenticity_Report.pdf (.md)    <- Official checkpoint SHA256 & provenance
│   ├── Dataset_Integrity_Report.pdf (.json)   <- 848 scans audited, 0% leakage
│   ├── Checkpoint_Hash_Manifest.json          <- Cryptographic hashes of all 14 .pt files
│   ├── Test_Results.md                        <- 123/123 pytest passing evidence
│   └── Reproducibility_Guide.md
│
├── 05_FIGURES/
│   ├── fig1_leaderboard_roc_curves.png
│   ├── fig2_cross_cohort_generalization_gap.png
│   ├── fig3_foundation_vs_supervised.png
│   ├── fig4_fine_tuning_impact.png
│   ├── fig5_trustgate_abstention_curves.png
│   ├── fig6_subgroup_scanner_transfer.png
│   ├── fig7_parameter_efficiency_tradeoff.png
│   ├── fig8_confusion_matrices.png
│   ├── fig9_calibration_curves.png
│   └── fig10_three_cohort_summary.png
│
├── 06_PRESENTATION/
│   └── SHIFT_MRI_Final_Presentation.pptx      <- 16:9 Final Project Defense Presentation
│
├── 07_DASHBOARD_SCREENSHOTS/
│   └── [Audited browser screenshots of all research sections]
│
├── 08_EXECUTIVE_SUMMARY/
│   ├── Final_Findings_One_Page.pdf            <- 1-page executive summary for professors
│   ├── Morning_Handover.md                    <- Answers to all 20 specific handover questions
│   └── Submission_Readiness_Checklist.md
│
└── SHIFT_MRI_V4_Submission_Package.zip        <- Clean academic handover ZIP archive
```

---

## Key Empirical Conclusions
1. **Swin UNETR SSL Superiority:** MONAI Swin UNETR SSL linear probe achieved superior out-of-distribution transfer on ADNI (**AUC = 0.5756**), significantly outperforming 3D ResNet-18 (**AUC = 0.4601**, $p = 0.036$).
2. **Fine-Tuning Failure:** Adapting foundation model backbones on small source data ($N=138$) consistently hurt external generalization ($\Delta\text{AUC} = -0.01$ to $-0.07$). Linear probing on frozen representations is strictly recommended.
3. **BrainIAC vs Control:** Pretrained BrainIAC ViT-B/16 did not demonstrate statistically significant advantage over a randomly initialized ViT-B/16 control ($p = 0.138$).
4. **TrustGate Collapse:** Source-calibrated confidence estimators collapsed under domain shift (error-detection AUROC = 0.4722), replicating the documented negative finding.
