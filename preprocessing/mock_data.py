"""
Synthetic / Mock 3D Brain MRI Data Generator.

IMPORTANT NOTICE:
This module generates strictly synthetic, geometric 3D MRI phantoms with simulated
longitudinal metadata FOR PIPELINE VERIFICATION AND UNIT TESTING ONLY.
It contains ZERO real patient medical data and must NEVER be used to report benchmark findings.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Tuple, List, Optional


def create_3d_brain_phantom(shape: Tuple[int, int, int] = (64, 64, 64), seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a synthetic 3D Brain Phantom volume with simulated anatomical structures:
    - Background (air)
    - Skull/scalp boundary
    - Parenchyma (Grey & White matter intensity gradient)
    - Ventricular cavities (CSF)
    - Random Gaussian noise
    """
    if seed is not None:
        np.random.seed(seed)

    D, H, W = shape
    z, y, x = np.ogrid[:D, :H, :W]
    cz, cy, cx = D / 2.0, H / 2.0, W / 2.0

    # Normalized ellipsoidal distances
    r_outer = ((z - cz) / (0.42 * D)) ** 2 + ((y - cy) / (0.45 * H)) ** 2 + ((x - cx) / (0.38 * W)) ** 2
    r_inner = ((z - cz) / (0.38 * D)) ** 2 + ((y - cy) / (0.40 * H)) ** 2 + ((x - cx) / (0.34 * W)) ** 2
    
    # Ventricle cavities (two small central ellipsoids)
    r_vent1 = ((z - cz) / (0.15 * D)) ** 2 + ((y - (cy - 0.08 * H)) / (0.12 * H)) ** 2 + ((x - cx) / (0.08 * W)) ** 2
    r_vent2 = ((z - cz) / (0.15 * D)) ** 2 + ((y - (cy + 0.08 * H)) / (0.12 * H)) ** 2 + ((x - cx) / (0.08 * W)) ** 2

    # Base zero volume
    volume = np.zeros(shape, dtype=np.float32)

    # Skull boundary
    volume = np.where(r_outer <= 1.0, 0.3, volume)

    # Brain parenchyma (WM/GM) with gradient
    parenchyma_pattern = (0.8 + 0.15 * np.sin(0.1 * x + 0.1 * y)).astype(np.float32)
    volume = np.where(r_inner <= 1.0, parenchyma_pattern, volume)

    # Ventricles (CSF, lower T1 intensity)
    ventricle_mask = (r_vent1 <= 1.0) | (r_vent2 <= 1.0)
    volume = np.where(ventricle_mask, 0.15, volume)

    # Add Gaussian noise
    noise = np.random.normal(0, 0.03, shape).astype(np.float32)
    volume = np.clip(volume + noise, 0.0, 1.0)

    # Scale to typical 16-bit integer MRI range
    volume_mri = (volume * 1000.0).astype(np.float32)
    return volume_mri


def generate_mock_dataset(
    output_dir: str = "data/mock_data",
    num_patients: int = 12,
    scans_per_patient: int = 2,
    volume_shape: Tuple[int, int, int] = (64, 64, 64),
    seed: int = 42
) -> str:
    """
    Generate a self-contained mock dataset containing synthetic NIfTI volumes
    and a longitudinal metadata CSV with patient IDs, diagnoses, and ages.

    Returns:
        Path to the generated metadata CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    scans_dir = os.path.join(output_dir, "scans")
    os.makedirs(scans_dir, exist_ok=True)

    np.random.seed(seed)
    records = []
    diagnoses_pool = ["CN", "MCI", "AD"]
    
    affine = np.eye(4)
    affine[0, 0] = 1.0
    affine[1, 1] = 1.0
    affine[2, 2] = 1.0

    scan_idx = 0
    for p_idx in range(num_patients):
        patient_id = f"MOCK_PT_{p_idx:03d}"
        base_age = float(np.random.uniform(58.0, 82.0))
        gender = np.random.choice(["M", "F"])
        patient_dx = np.random.choice(diagnoses_pool, p=[0.4, 0.35, 0.25])

        for v_idx in range(scans_per_patient):
            visit_code = f"m{v_idx * 12:02d}"
            scan_age = round(base_age + (v_idx * 1.0), 1)
            image_uid = f"MOCK_IMG_{scan_idx:04d}"
            nii_filename = f"{patient_id}_{visit_code}_{image_uid}.nii.gz"
            abs_file_path = os.path.abspath(os.path.join(scans_dir, nii_filename))

            # Generate and save synthetic 3D volume
            vol_data = create_3d_brain_phantom(shape=volume_shape, seed=seed + scan_idx)
            nii_img = nib.Nifti1Image(vol_data, affine)
            nib.save(nii_img, abs_file_path)

            records.append({
                "PTID": patient_id,
                "IMAGEUID": image_uid,
                "VISCODE": visit_code,
                "FILE_PATH": abs_file_path,
                "DX": patient_dx,
                "AGE": scan_age,
                "PTGENDER": gender,
                "MMSE": int(np.random.randint(18, 31)),
                "CDRSB": float(np.random.choice([0.0, 0.5, 1.0, 2.0, 4.5])),
                "IS_SYNTHETIC": True
            })
            scan_idx += 1

    df = pd.DataFrame(records)
    metadata_csv_path = os.path.join(output_dir, "mock_metadata.csv")
    df.to_csv(metadata_csv_path, index=False)
    
    return metadata_csv_path


if __name__ == "__main__":
    csv_path = generate_mock_dataset()
    print(f"Successfully generated synthetic mock dataset at: {csv_path}")
