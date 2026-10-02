"""
Compute Tracking, GPU Profiler, and Resource Accounting Engine.
"""

import time
import os
import psutil
import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class ComputeTracker:
    """
    Measures and logs computational resources:
    - GPU Hours
    - Peak VRAM Allocation (MB)
    - Host RAM Usage (MB)
    - Inference Latency (ms/sample)
    - Checkpoint Size (MB)
    - Active Parameter Counts
    """
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self):
        """Start recording computation."""
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop recording and return elapsed time in seconds."""
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()
        self.end_time = time.time()
        return self.end_time - (self.start_time or self.end_time)

    def get_summary(self, model: Optional[nn.Module] = None, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile resource summary dictionary.
        """
        elapsed_sec = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0.0
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_hours = (elapsed_sec / 3600.0) * max(1, num_gpus) if "cuda" in self.device else 0.0

        peak_vram_mb = 0.0
        if torch.cuda.is_available() and "cuda" in self.device:
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024.0 * 1024.0)

        summary = {
            "elapsed_seconds": round(elapsed_sec, 2),
            "gpu_hours": round(gpu_hours, 5),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "host_ram_mb": round(ram_mb, 2),
            "device": self.device,
        }

        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            summary["total_parameters"] = total_params
            summary["trainable_parameters"] = trainable_params

        if checkpoint_path and os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024.0 * 1024.0)
            summary["checkpoint_size_mb"] = round(size_mb, 2)

        return summary


def measure_inference_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    num_warmup: int = 5,
    num_repeats: int = 20,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Measure inference latency per 3D sample in milliseconds.
    """
    model.eval()
    model.to(device)
    sample_input = sample_input.to(device)

    with torch.no_grad():
        # Warmup
        for _ in range(num_warmup):
            _ = model(sample_input)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()

        # Timing
        start = time.time()
        for _ in range(num_repeats):
            _ = model(sample_input)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        total_time = time.time() - start

    batch_size = sample_input.shape[0]
    latency_ms_per_sample = (total_time / (num_repeats * batch_size)) * 1000.0

    return {
        "latency_ms_per_sample": round(latency_ms_per_sample, 2),
        "throughput_samples_per_sec": round((num_repeats * batch_size) / max(1e-6, total_time), 2),
    }
