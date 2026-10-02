"""
Phase 5B Real-Data 3D ResNet-18 Pre-Training Smoke Test.
Validates single forward pass, BCEWithLogitsLoss computation, backward gradient flow,
optimizer weight update, and split isolation on real OASIS-1 preprocessed tensors.
"""

import os
import sys
import json
import torch
import pandas as pd

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from models.baseline.resnet3d import build_resnet18_3d
from preprocessing.cache_manager import PreprocessedCacheManager
from preprocessing.dataloader import create_benchmark_dataloaders
from tasks.classification import ClassificationTask


def run_pretraining_smoke_test(data_dir: str = "D:\\OASIS1_PILOT", output_dir: str = "results"):
    logger = get_logger("PretrainingSmokeTest")
    logger.info("=================================================================")
    logger.info("Starting Phase 5B Real-Data 3D ResNet-18 Pre-Training Smoke Test")
    logger.info("=================================================================")

    os.makedirs(output_dir, exist_ok=True)
    splits_path = os.path.join(data_dir, "pilot_splits.csv")
    splits_df = pd.read_csv(splits_path)

    # Exclude forbidden file if present
    splits_df = splits_df[~splits_df["mri_file"].str.contains("OAS1_0061_MR2")].reset_index(drop=True)
    splits_df["file_path"] = [os.path.join(data_dir, fname) for fname in splits_df["mri_file"]]

    train_df = splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True)
    val_df = splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True)
    test_df = splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True)

    splits_dict = {"train": train_df, "val": val_df, "test": test_df}

    # 1. Device check
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Host)"

    logger.info(f"Execution Device: {device} ({device_name}), CUDA Available: {cuda_avail}")

    # 2. Attach Cache Manager
    cache_dir = os.path.join("data", "cache", "oasis1_pilot")
    cache_manager = PreprocessedCacheManager(cache_dir=cache_dir)

    task_cfg = {
        "task_type": "classification",
        "target_column": "label",
        "positive_class": 1.0,
        "negative_class": 0.0,
    }

    train_loader, val_loader, test_loader = create_benchmark_dataloaders(
        splits=splits_dict,
        task_config=task_cfg,
        batch_size=4,
        num_workers=0,
        spatial_shape=(128, 128, 128),
        label_budget=1.0,
        seed=42,
        cache_manager=cache_manager,
    )

    # 3. Load exactly ONE batch from train_loader
    train_iter = iter(train_loader)
    inputs, targets, metadata = next(train_iter)

    inputs = inputs.to(device)
    targets = targets.to(device)

    # 4. Check split isolation of batch metadata
    batch_patient_ids = metadata["patient_id"]
    train_patient_set = set(train_df["ID"])
    val_patient_set = set(val_df["ID"])
    test_patient_set = set(test_df["ID"])

    is_train_only = all(pid in train_patient_set for pid in batch_patient_ids)
    has_val_contamination = any(pid in val_patient_set for pid in batch_patient_ids)
    has_test_contamination = any(pid in test_patient_set for pid in batch_patient_ids)

    split_check_passed = is_train_only and not has_val_contamination and not has_test_contamination

    # 5. Instantiate 3D ResNet-18
    model = build_resnet18_3d(output_dim=1, task_type="classification", dropout=0.2).to(device)
    model.train()

    # Capture initial weights for optimizer step verification
    param_to_track = list(model.parameters())[0]
    initial_weight = param_to_track.clone().detach()

    # 6. Single Forward Pass
    logits = model(inputs)
    output_shape = list(logits.shape)
    expected_output_shape = [inputs.shape[0], 1]
    shape_check_passed = (output_shape == expected_output_shape)

    # 7. Classification Loss Computation
    task = ClassificationTask(task_name="AD_vs_Dementia")
    loss = task.compute_loss(logits, targets)
    loss_val = float(loss.item())
    loss_finite = torch.isfinite(loss).item()

    # 8. Backward Pass & Gradient Flow Verification
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    optimizer.zero_grad()
    loss.backward()

    all_grads_finite = True
    has_nonzero_grad = False

    for name, p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                all_grads_finite = False
            if (p.grad != 0).any():
                has_nonzero_grad = True

    grad_check_passed = loss_finite and all_grads_finite and has_nonzero_grad

    # 9. Single Optimizer Step & Parameter Update Verification
    optimizer.step()
    updated_weight = param_to_track.clone().detach()
    weight_difference = float(torch.abs(updated_weight - initial_weight).sum().item())
    optimizer_step_passed = weight_difference > 0.0

    # 10. Overall Smoke Test Status
    all_passed = (
        shape_check_passed and
        loss_finite and
        grad_check_passed and
        optimizer_step_passed and
        split_check_passed
    )

    smoke_report = {
        "status": "PASS" if all_passed else "FAIL",
        "device": device,
        "device_name": device_name,
        "cuda_available": cuda_avail,
        "input_shape": list(inputs.shape),
        "input_dtype": str(inputs.dtype),
        "target_shape": list(targets.shape),
        "target_dtype": str(targets.dtype),
        "output_shape": output_shape,
        "expected_output_shape": expected_output_shape,
        "shape_check_passed": shape_check_passed,
        "loss_value": round(loss_val, 6),
        "loss_is_finite": loss_finite,
        "all_gradients_finite": all_grads_finite,
        "has_nonzero_gradients": has_nonzero_grad,
        "gradient_check_passed": grad_check_passed,
        "weight_difference_after_step": round(weight_difference, 6),
        "optimizer_step_passed": optimizer_step_passed,
        "batch_patient_ids": list(batch_patient_ids),
        "split_check_passed": split_check_passed,
        "val_contamination_detected": has_val_contamination,
        "test_contamination_detected": has_test_contamination,
    }

    report_path = os.path.join(output_dir, "oasis1_pretraining_smoke_test.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(smoke_report, f, indent=2)

    logger.info(f"Pre-Training Smoke Test Report saved to: {report_path}")
    logger.info(f"FINAL SMOKE TEST RESULT: {smoke_report['status']}")
    return smoke_report


if __name__ == "__main__":
    run_pretraining_smoke_test()
