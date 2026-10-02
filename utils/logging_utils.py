"""Structured logging framework for experiments, diagnostics, and metrics."""
import os
import sys
import logging
from typing import Optional


def get_logger(name: str = "brain_benchmark", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a standardized structured logger.

    Args:
        name: Name of the logger component.
        log_file: Optional filepath to persist logs to disk.
        level: Logging verbosity level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (if requested)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_experiment_logging(experiment_name: str, log_dir: str = "results/logs") -> logging.Logger:
    """
    Initialize a dedicated log file for an individual experimental run.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{experiment_name}.log")
    return get_logger(name=experiment_name, log_file=log_path)
