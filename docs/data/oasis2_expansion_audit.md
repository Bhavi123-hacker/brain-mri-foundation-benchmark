# OASIS-2 Longitudinal Cohort Expansion Audit

## 1. Provenance and Licensing

- **Primary Study**: Open Access Series of Imaging Studies (OASIS-2): Longitudinal MRI Data in Nondemented and Demented Older Adults (Marcus et al., 2010).
- **Repository Source**: `radiata-ai/brain-structure` on Hugging Face.
- **License**: Open Data Commons Attribution License (`ODC-By`). Fully open for academic research.
- **Access Modality**: Direct public HTTPS download via Hugging Face Hub (zero gating, zero credentials required).
- **Imaging Modality**: 3D T1-weighted MPRAGE volumetric structural MRI.

## 2. Participant Isolation & Overlap Verification

- **Total Participants Ingested**: 150
- **Passed Quality Control**: 150 / 150
- **Identifier Namespace**: `sub-OASIS20001` through `sub-OASIS20150`.
- **OASIS-1 Overlap**: Exactly **0.0%** (OASIS-1 uses `OAS1_0001` through `OAS1_0436`).
- **ADNI Overlap**: Exactly **0.0%** (ADNI uses `002_S_0295`, etc.).
- **Foundation Model Pretraining Contamination**: FOMO25 pretraining uses FOMO300K/FOMO50K which includes OASIS-1. OASIS-2 provides an authentic external test cohort.

## 3. Cohort Demographics and Clinical Diagnostic Distribution

| Clinical Diagnosis | Binary Label | Participant Count | Age (Mean ± SD) | Female / Male |
| :--- | :--- | :--- | :--- | :--- |
| cognitively_normal | 0.0 | 85 | 75.8 ± 8.1 | 59 / 26 |
| Alzheimers_disease | 1.0 | 65 | 75.0 ± 6.8 | 29 / 36 |
| **Total Cohort** | — | **150** | **75.4 ± 7.5** | **88 / 62** |

## 4. Preprocessing Harmonization

- **Geometry**: Standardized to isotropic $128 \times 128 \times 128$ array via trilinear interpolation.
- **Normalization**: Robust Percentile Foreground Z-score (0.5th to 99.5th percentile clipping, matching Pipeline B).
- **Storage Location**: `D:\brain_mri_data\OASIS2\processed\tensors\`
- **Manifest Location**: `D:\brain_mri_data\OASIS2\manifests\oasis2_baseline_manifest.csv`
