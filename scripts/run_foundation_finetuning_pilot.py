"""
Priority 6 / Experiment 3: Foundation Model Fine-Tuning Pilot.
Compares:
1. Frozen Linear Probe (Backbone frozen, head trainable)
2. Partial Fine-Tuning (Top layers / projection + head trainable)
3. Full Fine-Tuning (All layers trainable)

For:
- BrainIAC (weights/brainiac_vit_base.pt)
- FOMO25 Winner (weights/fomo25_winner_encoder.pt)
- MONAI Swin UNETR SSL (weights/model_swinvit.pt)

Dataset: OASIS-1 Pilot (35 Train, 7 Val, 8 Test).
Outputs:
- results/extensions/fine_tuning/fine_tuning_comparison.csv
- results/extensions/fine_tuning/fine_tuning_comparison.json
- results/extensions/figures/figure_FT1_adaptation_comparison.png
"""

import os
import sys
import json
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_utils import get_logger
from utils.seed import set_seed
from models.foundation_brainiac.adapter import BrainIACAdapter
from models.foundation_fomo25.adapter import FOMO25Adapter
from models.foundation_swin.adapter import SwinUNETRSSLAdapter
from preprocessing.cache_manager import PreprocessedCacheManager
from preprocessing.dataloader import create_benchmark_dataloaders
from tasks.classification import ClassificationTask
from training.optimizer import build_optimizer_and_scheduler
from training.trainer import BenchmarkTrainer
from evaluation.metrics import compute_classification_metrics


def setup_partial_finetuning(model, model_key: str):
    """
    Unfreezes top layers (projection/final layer norm) while keeping early layers frozen.
    """
    if model_key == "brainiac":
        # Freeze all first
        for p in model.patch_embed.parameters():
            p.requires_grad = False
        # Unfreeze final linear projection and LayerNorm (patch_embed[9] and [10])
        for p in model.patch_embed[9].parameters():
            p.requires_grad = True
        for p in model.patch_embed[10].parameters():
            p.requires_grad = True
    elif model_key == "fomo25":
        for p in model.encoder.parameters():
            p.requires_grad = False
        # Unfreeze final linear projection and LayerNorm (encoder[11] and [12])
        for p in model.encoder[11].parameters():
            p.requires_grad = True
        for p in model.encoder[12].parameters():
            p.requires_grad = True
    elif model_key == "swin_unetr":
        for p in model.swin_encoder.parameters():
            p.requires_grad = False
        # Unfreeze final linear projection and LayerNorm (swin_encoder[10] and [11])
        for p in model.swin_encoder[10].parameters():
            p.requires_grad = True
        for p in model.swin_encoder[11].parameters():
            p.requires_grad = True

    # Always ensure classification head is trainable
    for p in model.head.parameters():
        p.requires_grad = True


def run_experiment(
    model_name: str,
    model_key: str,
    adaptation_strategy: str,  # 'frozen_linear_probe', 'partial_finetune', 'full_finetune'
    adapter_class,
    adapter_kwargs: dict,
    weights_path: str,
    splits_dict: dict,
    cache_manager: PreprocessedCacheManager,
    device: str,
    output_dir: str,
    logger,
    lr: float = 1e-4,
    max_epochs: int = 15,
    seed: int = 42
) -> dict:
    set_seed(seed)
    logger.info(f"\n--- Running {model_name} | Strategy: {adaptation_strategy} ---")

    # Instantiate adapter
    model = adapter_class(
        output_dim=1,
        task_type="classification",
        weights_path=weights_path,
        adaptation_mode="fine_tune" if adaptation_strategy == "full_finetune" else "linear_probe",
        **adapter_kwargs
    )

    if adaptation_strategy == "frozen_linear_probe":
        model.freeze_backbone()
    elif adaptation_strategy == "partial_finetune":
        setup_partial_finetuning(model, model_key)
    elif adaptation_strategy == "full_finetune":
        model.unfreeze_backbone()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"  Params: Total={total_params:,} | Trainable={trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Dataloaders
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
        seed=seed,
        cache_manager=cache_manager,
    )

    task = ClassificationTask(task_name="ad_vs_cn_binary")
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=1e-3,
        optimizer_type="adamw",
        scheduler_type="cosine",
        total_epochs=max_epochs,
    )

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    exp_name = f"{model_key}_{adaptation_strategy}_s{seed}"

    trainer = BenchmarkTrainer(
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=False,
        early_stopping_patience=5,
        checkpoint_dir=checkpoint_dir,
        experiment_name=exp_name,
        logger=logger,
    )

    start_t = time.time()
    fit_res = trainer.fit(train_loader=train_loader, val_loader=val_loader, max_epochs=max_epochs)
    train_time = round(time.time() - start_t, 2)

    # Validation eval
    val_loss, val_logits, val_targets = trainer.evaluate(val_loader)
    val_probs = torch.sigmoid(val_logits).numpy().flatten()
    val_targets_np = val_targets.numpy().flatten()
    val_metrics = compute_classification_metrics(val_targets_np, val_probs)

    # Test eval (final reporting only)
    test_loss, test_logits, test_targets = trainer.evaluate(test_loader)
    test_probs = torch.sigmoid(test_logits).numpy().flatten()
    test_targets_np = test_targets.numpy().flatten()
    test_metrics = compute_classification_metrics(test_targets_np, test_probs)

    logger.info(f"  Val AUC: {val_metrics['roc_auc']:.4f} | Test AUC: {test_metrics['roc_auc']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | Train Time: {train_time}s")

    return {
        "model": model_name,
        "model_key": model_key,
        "adaptation_strategy": adaptation_strategy,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_pct": round(trainable_params / total_params * 100, 3),
        "learning_rate": lr,
        "epochs_trained": len(fit_res["history"]),
        "best_epoch": next((item["epoch"] for item in fit_res["history"] if item.get("is_best")), len(fit_res["history"])),
        "best_val_loss": round(fit_res["best_val_loss"], 4),
        "val_roc_auc": round(val_metrics["roc_auc"], 4),
        "val_accuracy": round(val_metrics["accuracy"], 4),
        "val_f1": round(val_metrics["f1"], 4),
        "test_roc_auc": round(test_metrics["roc_auc"], 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_balanced_accuracy": round(test_metrics["balanced_accuracy"], 4),
        "test_f1": round(test_metrics["f1"], 4),
        "test_brier": round(test_metrics["brier_score"], 4),
        "training_time_sec": train_time,
        "checkpoint_path": trainer.best_checkpoint_path,
    }


def main():
    logger = get_logger("FoundationFineTuningPilot")
    logger.info("=================================================================")
    logger.info("Priority 6: Foundation Model Adaptation Pilot (Head vs Partial vs Full)")
    logger.info("=================================================================")

    data_dir = "D:\\OASIS1_PILOT"
    output_dir = "results/extensions/fine_tuning"
    figures_dir = "results/extensions/figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load splits
    splits_path = os.path.join(data_dir, "pilot_splits.csv")
    splits_df = pd.read_csv(splits_path)
    splits_df = splits_df[~splits_df["mri_file"].str.contains("OAS1_0061_MR2")].reset_index(drop=True)
    splits_df["file_path"] = [os.path.join(data_dir, fname) for fname in splits_df["mri_file"]]

    splits_dict = {
        "train": splits_df[splits_df["split"] == "train"].copy().reset_index(drop=True),
        "val": splits_df[splits_df["split"] == "val"].copy().reset_index(drop=True),
        "test": splits_df[splits_df["split"] == "test"].copy().reset_index(drop=True),
    }

    cache_dir = os.path.join("data", "cache", "oasis1_pilot")
    cache_manager = PreprocessedCacheManager(cache_dir=cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_config = [
        {
            "model_name": "BrainIAC",
            "model_key": "brainiac",
            "weights_path": os.path.abspath(os.path.join("weights", "brainiac_vit_base.pt")),
            "adapter_class": BrainIACAdapter,
            "adapter_kwargs": {"embed_dim": 768},
        },
        {
            "model_name": "FOMO25 Winner",
            "model_key": "fomo25",
            "weights_path": os.path.abspath(os.path.join("weights", "fomo25_winner_encoder.pt")),
            "adapter_class": FOMO25Adapter,
            "adapter_kwargs": {"embed_dim": 512},
        },
        {
            "model_name": "MONAI Swin UNETR SSL",
            "model_key": "swin_unetr",
            "weights_path": os.path.abspath(os.path.join("weights", "model_swinvit.pt")),
            "adapter_class": SwinUNETRSSLAdapter,
            "adapter_kwargs": {"embed_dim": 768, "feature_size": 48},
        },
    ]

    strategies = [
        ("frozen_linear_probe", 1e-3, 15),
        ("partial_finetune", 1e-4, 15),
        ("full_finetune", 5e-5, 15),
    ]

    all_results = []

    for cfg in models_config:
        for strat, lr, epochs in strategies:
            res = run_experiment(
                model_name=cfg["model_name"],
                model_key=cfg["model_key"],
                adaptation_strategy=strat,
                adapter_class=cfg["adapter_class"],
                adapter_kwargs=cfg["adapter_kwargs"],
                weights_path=cfg["weights_path"],
                splits_dict=splits_dict,
                cache_manager=cache_manager,
                device=device,
                output_dir=output_dir,
                logger=logger,
                lr=lr,
                max_epochs=epochs,
                seed=42,
            )
            all_results.append(res)

    # Save results
    res_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "fine_tuning_comparison.csv")
    res_df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved fine-tuning comparison CSV to {csv_path}")

    json_path = os.path.join(output_dir, "fine_tuning_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved fine-tuning comparison JSON to {json_path}")

    # Generate Figure FT1
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    models = ["BrainIAC", "FOMO25 Winner", "MONAI Swin UNETR SSL"]
    x = np.arange(len(models))
    width = 0.25

    strats = ["frozen_linear_probe", "partial_finetune", "full_finetune"]
    labels = ["Frozen Linear Probe", "Partial Fine-Tuning", "Full Fine-Tuning"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (s, label, color) in enumerate(zip(strats, labels, colors)):
        vals = [res_df[(res_df["model"] == m) & (res_df["adaptation_strategy"] == s)]["test_roc_auc"].values[0] for m in models]
        rects = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color, alpha=0.85, edgecolor="black")
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_title("Adaptation Strategy Comparison on OASIS-1 Pilot", fontsize=13, fontweight="bold")
    ax.set_xlabel("Foundation Model", fontsize=11)
    ax.set_ylabel("Test ROC-AUC", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right")

    fig_path = os.path.join(figures_dir, "figure_FT1_adaptation_comparison.png")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved Figure FT1 to {fig_path}")

    return res_df


if __name__ == "__main__":
    main()
