# Campaign Execution Log: Master Autonomous Research Campaign

- **Campaign Title**: Fair Benchmarking of Brain-MRI Foundation Models Against Well-Tuned Supervised Deep Learning Baselines
- **Execution Timestamp**: 2026-09-14 14:13:03
- **Operating Environment**: Windows 11, Python 3.12.4, PyTorch 2.5.1+cpu, MONAI 1.4.0
- **Primary Milestones Accomplished**:
  1. Complete DICOM and NIfTI audit across 50 OASIS-1 pilot and 50 ADNI MPRAGE subjects.
  2. Construction of patient-isolated development cross-validation splits ($K=5$, Seeds 42, 43, 44) and locked holdout test sets ($N=8$ per cohort).
  3. Controlled benchmarking of 5 models: Supervised 3D ResNet-18, Supervised 3D ResNet-34, BrainIAC ViT-Base, FOMO25 Winner, and MONAI Swin UNETR SSL.
  4. Formulation and evaluation of 4 core research novelties:
     - Novelty 1: Seed-Stability Adjusted Performance (SSAP with $\lambda \in {0.25, 0.50, 1.00}$).
     - Novelty 2: Parameter-Efficiency Pareto Frontier (mapping AUC per million trainable parameters).
     - Novelty 3: Label-Efficiency Sample Scaling Curves (10%, 25%, 50%, 100% data budgets).
     - Novelty 4: Foundation + CNN Hybrid Feature Fusion (achieving 0.9800 CV AUC).
  5. Cross-scanner domain generalization audit (1.5T OASIS-1 to 3.0T ADNI transfer).
  6. Temperature-scaling probability calibration and threshold optimization on development partitions.
  7. Cross-Dataset CNN Interpretability Study (Experiment 40): 3D Grad-CAM generation, spatial consistency metrics ($r = +0.6219$, Dice $= 0.4967$), and frontend asset tree creation.
  8. Full Section 31 Visualization Suite (Figures `01_` through `27_` and `prediction_confidence_distribution.png`).
  9. Section 25 Dataset Characterization figures (`figures/dataset/`).
  10. Publication Deliverables: 38-section `FINAL_RESEARCH_REPORT.md`, 22-question `EXECUTIVE_SUMMARY.md`, template-conforming `FINAL_DECISION.md`, `RESULT_RECONCILIATION.md`, and `FINAL_TEST_LEAKAGE_AUDIT.md`.
  11. Automated Test Suite: 43/43 passing tests verified.
