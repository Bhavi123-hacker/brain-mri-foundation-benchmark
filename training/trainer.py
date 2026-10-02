"""
Modular Benchmark Trainer with Early Stopping, Mixed Precision, and Compute Profiling.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, Callable
from utils.logging_utils import get_logger


class BenchmarkTrainer:
    def __init__(
        self,
        model: nn.Module,
        task: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        use_amp: bool = False,
        grad_accum_steps: int = 1,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "results/checkpoints",
        experiment_name: str = "exp_01",
        logger: Optional[Any] = None,
    ):
        self.model = model.to(device)
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp and (device == "cuda")
        self.grad_accum_steps = grad_accum_steps
        self.patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.logger = logger or get_logger(experiment_name)
        
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pt")

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, (inputs, targets, _) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.task.compute_loss(outputs, targets) / self.grad_accum_steps
                self.scaler.scale(loss).backward()
                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs)
                loss = self.task.compute_loss(outputs, targets) / self.grad_accum_steps
                loss.backward()
                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps

        return total_loss / max(1, len(train_loader))

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, torch.Tensor, torch.Tensor]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.task.compute_loss(outputs, targets)
                total_loss += loss.item()

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / max(1, len(val_loader))
        preds_tensor = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0)
        targets_tensor = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0)

        return avg_loss, preds_tensor, targets_tensor

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 50,
        eval_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute full training loop with early stopping and profiling.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        training_history = []
        start_time = time.time()

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_preds, val_targets = self.evaluate(val_loader)
            epoch_time = time.time() - epoch_start

            if self.scheduler is not None:
                self.scheduler.step()

            # Check for improvement
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
            else:
                patience_counter += 1

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_time_seconds": epoch_time,
                "is_best": is_best,
            }
            training_history.append(record)

            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        total_training_time = time.time() - start_time

        # Load best model for evaluation
        if os.path.exists(self.best_checkpoint_path):
            self.model.load_state_dict(torch.load(self.best_checkpoint_path, map_location=self.device))

        return {
            "best_val_loss": best_val_loss,
            "total_epochs": len(training_history),
            "total_training_time_seconds": total_training_time,
            "best_checkpoint_path": self.best_checkpoint_path,
            "history": training_history,
        }
