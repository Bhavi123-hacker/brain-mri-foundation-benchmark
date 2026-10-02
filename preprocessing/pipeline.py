"""
Dataset Reading, Patient-Level Splitting, and Preprocessing Execution Engine.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import GroupShuffleSplit
from preprocessing.transforms import get_preprocessing_transforms


def create_patient_level_splits(
    metadata_df: pd.DataFrame,
    patient_col: str = "PTID",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_col: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Split dataset into Train, Validation, and Test partitions strictly by Patient ID.
    Guarantees 0% patient-level data leakage across partitions.

    Args:
        metadata_df: Metadata DataFrame with scan paths and patient IDs.
        patient_col: Column name identifying unique subjects/patients.
        train_ratio: Fraction for training split.
        val_ratio: Fraction for validation split.
        test_ratio: Fraction for test split.
        seed: Random seed for reproducibility.
        stratify_col: Optional column for group-aware stratification.

    Returns:
        Dictionary containing {'train': df, 'val': df, 'test': df}.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    unique_patients = metadata_df[patient_col].unique()
    num_patients = len(unique_patients)

    if num_patients < 3:
        raise ValueError(f"Need at least 3 unique patients to create 3 splits, got {num_patients}")

    np.random.seed(seed)
    shuffled_patients = np.random.permutation(unique_patients)

    n_train = int(np.floor(train_ratio * num_patients))
    n_val = int(np.floor(val_ratio * num_patients))
    # Ensure at least 1 patient in each split
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    
    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train:n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val:])

    # Edge case: if test is empty due to rounding, shift one from train
    if len(test_patients) == 0 and len(train_patients) > 1:
        moved_pt = train_patients.pop()
        test_patients.add(moved_pt)

    # Verify zero intersection
    assert len(train_patients.intersection(val_patients)) == 0, "Leakage detected: Train & Val overlap!"
    assert len(train_patients.intersection(test_patients)) == 0, "Leakage detected: Train & Test overlap!"
    assert len(val_patients.intersection(test_patients)) == 0, "Leakage detected: Val & Test overlap!"

    train_df = metadata_df[metadata_df[patient_col].isin(train_patients)].copy().reset_index(drop=True)
    val_df = metadata_df[metadata_df[patient_col].isin(val_patients)].copy().reset_index(drop=True)
    test_df = metadata_df[metadata_df[patient_col].isin(test_patients)].copy().reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }


def save_splits(splits_dict: Dict[str, pd.DataFrame], output_dir: str = "data/splits", split_name: str = "default_split") -> None:
    """Save split DataFrames to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    for split_key, df in splits_dict.items():
        csv_path = os.path.join(output_dir, f"{split_name}_{split_key}.csv")
        df.to_csv(csv_path, index=False)


class BrainMRIDataset(Dataset):
    """
    Standard PyTorch Dataset for 3D Brain MRI Volumes.
    Supports NIfTI files (.nii, .nii.gz) and NumPy files (.npy) with on-the-fly or cached preprocessing.
    """
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        target_column: str = "DX",
        task_type: str = "classification",
        positive_class: Any = "AD",
        negative_class: Any = "CN",
        transform: Optional[Any] = None,
        file_path_col: Optional[str] = None,
        cache_manager: Optional[Any] = None,
        transform_id: str = "default_128",
    ):
        self.metadata_df = metadata_df.copy().reset_index(drop=True)
        self.target_column = target_column
        self.task_type = task_type
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.transform = transform
        self.cache_manager = cache_manager
        self.transform_id = transform_id

        # Auto-detect file path column
        if file_path_col and file_path_col in self.metadata_df.columns:
            self.file_path_col = file_path_col
        else:
            for col in ["file_path", "FILE_PATH", "mri_file", "full_path"]:
                if col in self.metadata_df.columns:
                    self.file_path_col = col
                    break
            else:
                self.file_path_col = self.metadata_df.columns[0]

        # Filter out rows missing target or invalid for binary task
        if task_type == "classification":
            valid_classes = [positive_class, negative_class]
            # Handle float vs int vs string matching
            self.metadata_df = self.metadata_df[
                self.metadata_df[target_column].apply(lambda v: v in valid_classes or str(v) in [str(c) for c in valid_classes])
            ].reset_index(drop=True)
        elif task_type == "regression":
            self.metadata_df = self.metadata_df.dropna(subset=[target_column]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.metadata_df.iloc[idx]
        file_path = str(row[self.file_path_col])

        # Load & Transform 3D Tensor (with optional cache manager lookup)
        volume_tensor = None
        if self.cache_manager is not None and self.cache_manager.has(file_path, self.transform_id):
            volume_tensor = self.cache_manager.get(file_path, self.transform_id)

        if volume_tensor is None:
            if self.transform is not None:
                volume_tensor = self.transform(file_path)
            else:
                transform = get_preprocessing_transforms(is_training=False)
                volume_tensor = transform(file_path)

            if not isinstance(volume_tensor, torch.Tensor):
                volume_tensor = torch.as_tensor(volume_tensor, dtype=torch.float32)

            if self.cache_manager is not None:
                self.cache_manager.put(file_path, volume_tensor, self.transform_id)

        if volume_tensor.dtype != torch.float32:
            volume_tensor = volume_tensor.to(torch.float32)

        # Extract Target Label
        if self.task_type == "classification":
            raw_label = row[self.target_column]
            is_pos = (raw_label == self.positive_class) or (str(raw_label) == str(self.positive_class))
            label_val = 1.0 if is_pos else 0.0
            target = torch.tensor([label_val], dtype=torch.float32)
        elif self.task_type == "regression":
            label_val = float(row[self.target_column])
            target = torch.tensor([label_val], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        metadata = {
            "patient_id": row.get("ID", row.get("PTID", row.get("OASISID", "UNKNOWN"))),
            "visit": row.get("visit", row.get("VISCODE", row.get("VISIT", "MR1"))),
            "file_path": file_path,
        }

        return volume_tensor, target, metadata
