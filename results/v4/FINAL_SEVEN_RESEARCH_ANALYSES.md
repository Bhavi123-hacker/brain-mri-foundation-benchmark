# SHIFT-MRI V4 — Seven Core Research Analyses (N01 – N07)

**Author:** Bhavy Garg (24BAI1088), Vellore Institute of Technology, Chennai  
**Dataset Roots:** OASIS-1 (N=198), ADNI (N=500), OASIS-2 (N=150)

---

## N01: Cross-Cohort Discrimination and Calibration Reliability
Under source in-distribution evaluation (OASIS-1 locked test), 3D ResNet-18 achieves AUC=0.6477 and Classical Radiomics achieves AUC=0.6467. Under zero-shot external transfer to ADNI (N=500), 3D ResNet-18 collapses to AUC=0.4601 (ΔAUC=+0.1876), while MONAI Swin UNETR SSL maintains AUC=0.5756 (ΔAUC=+0.0255).

## N02: Decision-Boundary Collapse and Saturation
Deep 3D CNNs exhibit extreme overconfidence on external data: 3D ResNet-18 pushes over 60% of ADNI predictions into the confident tails (<0.1 or >0.9) despite a 40.4% error rate. Foundation models with linear probes exhibit more moderate probability dispersion (std=0.18–0.24).

## N03: Demographic Baselines (Age + Sex)
A logistic regression model trained strictly on age and sex achieves an ADNI external AUC of 0.6120. This establishes that age and biological sex provide strong non-imaging priors in Alzheimer's disease cohorts, and imaging-only models must be evaluated against this demographic benchmark.

## N04: Selective Prediction and Risk-Coverage Analysis
Evaluating confidence-based abstention on ADNI: rejecting 20% of the cohort based on confidence margin does not improve accuracy (0.5875 vs 0.5960 baseline, -0.85% drop). Only ensemble disagreement provides a modest gain (+2.40% at 20% rejection), confirming that source-calibrated confidence fails under domain shift.

## N05: Clinical Severity Subgroups
Discrimination between very mild cognitive impairment (CDR=0.5) and cognitively normal (CDR=0.0) yields AUC=0.5833, whereas mild/moderate impairment (CDR=1.0) achieves AUC=0.6667, confirming that subtle early structural atrophy is the primary source of classification challenge.

## N06: Perturbation Robustness
Simulated Gaussian intensity perturbations demonstrate that foundation feature representations retain 92% of baseline AUC at 10% noise, whereas scratch CNN representations degrade faster.

## N07: Latent Geometry and Out-of-Distribution Failure
Feature centroid distance in anatomical morphometric space yields an error-detection AUROC of only 0.5113 on ADNI, confirming that OOD distance from source training centroids does not reliably identify deep classification failures under covariate shift.
