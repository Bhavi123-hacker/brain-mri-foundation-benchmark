"""Reproducibility and deterministic random seed management."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cuda: bool = True) -> int:
    """
    Set seeds across all random number generators for strict reproducibility.

    Args:
        seed: Integer seed value.
        deterministic_cuda: Whether to enforce deterministic CUDA operations.

    Returns:
        The seed integer that was set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    return seed
