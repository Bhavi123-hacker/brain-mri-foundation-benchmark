# MIRIAD Official Access & Integration Guide
**Cohort**: Minimal Interval Resonance Imaging in Alzheimer's Disease (MIRIAD)  
**Host**: Dementia Research Centre (DRC), University College London (UCL)  
**Purpose**: External multi-vendor zero-shot validation of 3D/2D brain MRI models on General Electric (GE) 1.5T hardware.

---

## 1. Why User Action is Required

During the dataset expansion audit and Phase A verification, we resolved the licensing and distribution terms for MIRIAD:
1. **Governing Agreement**: **MIRIAD Data Use Agreement** (DRC UCL).
2. **Redistribution Prohibition**: The agreement explicitly forbids redistribution of raw MRI data or copies to third parties.
3. **Manual Account Approval**: Scans must be acquired directly by a registered user through the UCL XNAT database portal. Automated anonymous downloading without an approved account is not permitted.

To maintain strict research compliance and avoid scientific misconduct, you must complete the free user registration below.

---

## 2. Step-by-Step Registration Instructions

1. **Navigate to the Portal**:
   - Primary Portal: [http://miriad.drc.ion.ucl.ac.uk/](http://miriad.drc.ion.ucl.ac.uk/) (or UCL DRC MIRIAD Database page).
2. **Create a Free Account**:
   - Click **Register** on the XNAT interface.
   - Enter your academic/institutional email, name, and institution.
3. **Accept the Data Use Agreement**:
   - You must review and accept the MIRIAD Data Use Agreement terms:
     - Will not attempt to identify individual participants.
     - Will not redistribute the data to third parties.
     - Agree to include the mandatory citation:
       > *Malone IB, Cash D, Ridgway GR, Macmanus DG, Ourselin S, Fox NC, Schott JM. MIRIAD—Public release of a multiple time point Alzheimer's MR imaging dataset. Neuroimage. 2012 Dec 28;70C:33-36. doi:10.1016/j.neuroimage.2012.12.044.*
     - Agree to include the mandatory acknowledgment:
       > *"Data used in the preparation of this article were obtained from the MIRIAD database (http://miriad.drc.ion.ucl.ac.uk). The MIRIAD investigators did not participate in analysis or writing of this report."*
4. **Wait for Email Activation**:
   - UCL administrators manually approve accounts (typically approved within 24–48 hours).
5. **Download the Baseline Scans Archive**:
   - Once logged in, navigate to **Projects -> MIRIAD**.
   - Download the volumetric T1 NIfTI archive (or baseline `SCAN1` volumes for all 69 participants: 46 AD, 23 Controls).
   - Alternatively, download the demographic/clinical spreadsheet (`miriad_clinical_data.csv`).

---

## 3. Placement of Files in Benchmark Workspace

Once downloaded, place the uncompressed/compressed NIfTI scans into the designated local data directory:
```text
D:\brain_mri_data\MIRIAD\
├── raw\
│   ├── miriad_188_MR_1.nii.gz
│   ├── miriad_189_MR_1.nii.gz
│   └── ... (69 baseline scans)
└── clinical\
    └── miriad_clinical_data.csv
```

---

## 4. Automated Processing & Evaluation Ready

Once files are placed in `D:\brain_mri_data\MIRIAD\raw\`, run the preconfigured execution script:
```bash
& "D:\ml project\.venv\Scripts\python.exe" scripts/run_miriad_external_evaluation.py
```
This script will automatically:
1. Parse participant IDs and verify the 46 AD / 23 CN class balance.
2. Verify voxel dimensions ($0.9375 \times 0.9375 \times 1.5\text{ mm}^3$) and run affine registration + N4 bias correction.
3. Execute frozen inference across all 6 benchmark models (ResNet-18, ResNet-34, 2D Transfer ResNet, Radiomics, Swin UNETR SSL, and FOMO25 AMAES ResEnc-B).
4. Output verified zero-shot ROC-AUC with 95% bootstrap confidence intervals to `results/miriad_external_validation/`.
