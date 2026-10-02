"""Models Package and Factory Dispatcher."""
from typing import Dict, Any
from models.base import BaseBenchModel
from models.baseline.resnet3d import (
    build_resnet18_3d,
    build_resnet34_3d,
    build_resnet50_3d,
    ResNet3D,
)
from models.foundation_brainiac.adapter import BrainIACAdapter
from models.foundation_fomo25.adapter import FOMO25Adapter
from models.foundation_swin.adapter import SwinUNETRSSLAdapter


def build_model(model_key: str, task_type: str, output_dim: int = 1, config: Dict[str, Any] = None) -> BaseBenchModel:
    """
    Factory function to instantiate models from configuration keys.
    """
    if config is None:
        config = {}

    model_cfg = config.get("models", {}).get(model_key, {})
    adaptation_mode = model_cfg.get("adaptation_mode", "linear_probe")
    weights_path = model_cfg.get("weights_path", None)
    dropout = model_cfg.get("dropout", 0.2)

    if model_key == "baseline_resnet18":
        return build_resnet18_3d(output_dim=output_dim, task_type=task_type, dropout=dropout)
    elif model_key == "baseline_resnet34":
        return build_resnet34_3d(output_dim=output_dim, task_type=task_type, dropout=dropout)
    elif model_key == "baseline_resnet50":
        return build_resnet50_3d(output_dim=output_dim, task_type=task_type, dropout=dropout)
    elif model_key == "foundation_brainiac":
        return BrainIACAdapter(
            output_dim=output_dim,
            task_type=task_type,
            weights_path=weights_path,
            adaptation_mode=adaptation_mode
        )
    elif model_key == "foundation_fomo25":
        return FOMO25Adapter(
            output_dim=output_dim,
            task_type=task_type,
            weights_path=weights_path,
            adaptation_mode=adaptation_mode
        )
    elif model_key == "foundation_swin_unetr":
        return SwinUNETRSSLAdapter(
            output_dim=output_dim,
            task_type=task_type,
            weights_path=weights_path,
            adaptation_mode=adaptation_mode
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")


__all__ = [
    "BaseBenchModel",
    "ResNet3D",
    "build_resnet18_3d",
    "build_resnet34_3d",
    "build_resnet50_3d",
    "BrainIACAdapter",
    "FOMO25Adapter",
    "SwinUNETRSSLAdapter",
    "build_model",
]
