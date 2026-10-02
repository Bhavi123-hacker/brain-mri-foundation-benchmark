"""
Persistent Patient-Level Split Manager and Leakage Verification Engine.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class PersistentSplitManager:
    """
    Manages deterministic, persistent patient-level data partitions.
    Guarantees that all longitudinal scans of any patient are strictly isolated to one split.
    """
    def __init__(self, splits_dir: str = "data/splits"):
        self.splits_dir = splits_dir
        os.makedirs(splits_dir, exist_ok=True)

    def check_leakage(self, splits: Dict[str, pd.DataFrame], subject_col: str = "subject_id") -> Dict[str, Any]:
        """
        Verify that no subject IDs overlap across any splits.
        """
        train_pts = set(splits["train"][subject_col].unique()) if "train" in splits else set()
        val_pts = set(splits["val"][subject_col].unique()) if "val" in splits else set()
        test_pts = set(splits["test"][subject_col].unique()) if "test" in splits else set()

        train_val_overlap = train_pts.intersection(val_pts)
        train_test_overlap = train_pts.intersection(test_pts)
        val_test_overlap = val_pts.intersection(test_pts)

        has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

        return {
            "has_leakage": has_leakage,
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
            "train_val_overlap_subjects": list(train_val_overlap),
            "train_test_overlap_subjects": list(train_test_overlap),
            "val_test_overlap_subjects": list(val_test_overlap),
        }

    def create_and_persist_splits(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        task_name: str,
        subject_col: str = "subject_id",
        stratify_col: Optional[str] = "diagnosis",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """
        Create and persist fixed splits to disk with an integrity manifest.
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
        
        # Filter to usable rows
        usable_df = df[df["is_valid_scan"] == True].copy().reset_index(drop=True)
        unique_subjects = usable_df[subject_col].unique()
        num_subjects = len(unique_subjects)

        if num_subjects < 3:
            raise ValueError(f"Cannot create 3 splits from only {num_subjects} subjects in {dataset_name}-{task_name}")

        np.random.seed(seed)
        shuffled_subjects = np.random.permutation(unique_subjects)

        n_train = max(1, int(np.floor(train_ratio * num_subjects)))
        n_val = max(1, int(np.floor(val_ratio * num_subjects)))

        train_pts = set(shuffled_subjects[:n_train])
        val_pts = set(shuffled_subjects[n_train:n_train + n_val])
        test_pts = set(shuffled_subjects[n_train + n_val:])

        # Guard against empty test set due to rounding
        if len(test_pts) == 0 and len(train_pts) > 1:
            moved = train_pts.pop()
            test_pts.add(moved)

        splits = {
            "train": usable_df[usable_df[subject_col].isin(train_pts)].copy().reset_index(drop=True),
            "val": usable_df[usable_df[subject_col].isin(val_pts)].copy().reset_index(drop=True),
            "test": usable_df[usable_df[subject_col].isin(test_pts)].copy().reset_index(drop=True),
        }

        # Verify zero leakage
        leakage_info = self.check_leakage(splits, subject_col=subject_col)
        if leakage_info["has_leakage"]:
            raise RuntimeError(f"Data leakage detected during split creation: {leakage_info}")

        # Save CSV files and manifest
        prefix = f"{dataset_name.lower()}_{task_name.lower()}"
        file_paths = {}
        for split_name, split_df in splits.items():
            csv_filename = f"{prefix}_{split_name}.csv"
            csv_path = os.path.join(self.splits_dir, csv_filename)
            split_df.to_csv(csv_path, index=False)
            file_paths[split_name] = csv_path

        manifest = {
            "dataset": dataset_name,
            "task": task_name,
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "counts": {
                "train_subjects": len(train_pts),
                "train_scans": len(splits["train"]),
                "val_subjects": len(val_pts),
                "val_scans": len(splits["val"]),
                "test_subjects": len(test_pts),
                "test_scans": len(splits["test"]),
            },
            "leakage_verification": leakage_info,
            "files": file_paths,
        }

        manifest_path = os.path.join(self.splits_dir, f"{prefix}_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return splits

    def load_existing_splits(self, dataset_name: str, task_name: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load previously saved splits if available.
        """
        prefix = f"{dataset_name.lower()}_{task_name.lower()}"
        manifest_path = os.path.join(self.splits_dir, f"{prefix}_manifest.json")
        if not os.path.exists(manifest_path):
            return None

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        splits = {}
        for split_name, csv_path in manifest["files"].items():
            if os.path.exists(csv_path):
                splits[split_name] = pd.read_csv(csv_path)
            else:
                return None
        return splits
