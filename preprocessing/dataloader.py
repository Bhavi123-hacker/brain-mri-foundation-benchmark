"""
DataLoader Factory and Multiprocessing Integration for Brain MRI Benchmarks.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader
from preprocessing.pipeline import BrainMRIDataset
from preprocessing.transforms import get_preprocessing_transforms
from preprocessing.cache_manager import PreprocessedCacheManager
from training.budget_sampler import sample_training_budget


def create_benchmark_dataloaders(
    splits: Dict[str, pd.DataFrame],
    task_config: Dict[str, Any],
    batch_size: int = 4,
    num_workers: int = 0,
    spatial_shape: Tuple[int, int, int] = (128, 128, 128),
    label_budget: float = 1.0,
    seed: int = 42,
    cache_manager: Optional[PreprocessedCacheManager] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct Train, Validation, and Test DataLoaders with deterministic label budgets.
    """
    task_type = task_config.get("task_type", "classification")
    target_col = task_config.get("target_column", "diagnosis")
    pos_class = task_config.get("positive_class", "AD")
    neg_class = task_config.get("negative_class", "CN")

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Apply label budget subsampling to training set if budget < 1.0
    if label_budget < 1.0:
        subject_col = "subject_id" if "subject_id" in train_df.columns else "PTID"
        train_df = sample_training_budget(
            train_df=train_df,
            budget_fraction=label_budget,
            patient_col=subject_col,
            stratify_col=target_col if task_type == "classification" else None,
            seed=seed,
        )

    # Transforms
    train_transform = get_preprocessing_transforms(
        spatial_shape=spatial_shape,
        voxel_spacing=(1.0, 1.0, 1.0),
        intensity_norm="z_score",
        is_training=True,
    )
    eval_transform = get_preprocessing_transforms(
        spatial_shape=spatial_shape,
        voxel_spacing=(1.0, 1.0, 1.0),
        intensity_norm="z_score",
        is_training=False,
    )

    train_ds = BrainMRIDataset(
        metadata_df=train_df,
        target_column=target_col,
        task_type=task_type,
        positive_class=pos_class,
        negative_class=neg_class,
        transform=train_transform,
        file_path_col="file_path" if "file_path" in train_df.columns else "FILE_PATH",
    )
    val_ds = BrainMRIDataset(
        metadata_df=val_df,
        target_column=target_col,
        task_type=task_type,
        positive_class=pos_class,
        negative_class=neg_class,
        transform=eval_transform,
        file_path_col="file_path" if "file_path" in val_df.columns else "FILE_PATH",
    )
    test_ds = BrainMRIDataset(
        metadata_df=test_df,
        target_column=target_col,
        task_type=task_type,
        positive_class=pos_class,
        negative_class=neg_class,
        transform=eval_transform,
        file_path_col="file_path" if "file_path" in test_df.columns else "FILE_PATH",
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
