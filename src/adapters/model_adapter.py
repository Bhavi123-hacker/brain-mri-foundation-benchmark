"""
adapters/model_adapter.py
Extensible Model Adapter Framework for SHIFT-MRI Benchmark.

Provides a unified contract for:
1. Architecture definition and parameter tracking.
2. Checkpoint loading and hash verification.
3. Live forward pass inference and feature extraction.
4. Model-specific validation-locked decision thresholds.
5. Explainability (Grad-CAM) support verification.
"""

import abc
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np


class BaseModelAdapter(abc.ABC):
    """Abstract Base Class for SHIFT-MRI Model Adapters."""

    def __init__(
        self,
        model_name: str,
        paradigm: str,
        checkpoint_path: Optional[Path] = None,
        val_threshold: float = 0.5000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.paradigm = paradigm
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.val_threshold = val_threshold
        self.device = device
        self.model_instance: Optional[torch.nn.Module] = None

    @abc.abstractmethod
    def build_architecture(self) -> torch.nn.Module:
        """Constructs and returns the PyTorch model module."""
        pass

    @abc.abstractmethod
    def forward_inference(self, volume: np.ndarray) -> float:
        """Executes authentic forward pass on preprocessed (128^3) volume and returns probability."""
        pass

    def get_checkpoint_sha256(self) -> Optional[str]:
        """Computes SHA-256 hash of the checkpoint file for reproducibility."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return None
        h = hashlib.sha256()
        with open(self.checkpoint_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def load_weights(self) -> bool:
        """Loads state_dict from checkpoint onto target device."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return False
        if self.model_instance is None:
            self.model_instance = self.build_architecture()
        state = torch.load(self.checkpoint_path, map_location=self.device)
        self.model_instance.load_state_dict(state)
        self.model_instance.to(self.device)
        self.model_instance.eval()
        return True

    def get_model_specs(self) -> Dict[str, Any]:
        """Returns metadata specifications and parameter counts."""
        num_params = 0
        if self.model_instance is not None:
            num_params = sum(p.numel() for p in self.model_instance.parameters())
        return {
            "model_name": self.model_name,
            "paradigm": self.paradigm,
            "val_threshold": self.val_threshold,
            "parameters": num_params,
            "checkpoint_exists": self.checkpoint_path.exists() if self.checkpoint_path else False,
            "checkpoint_hash": self.get_checkpoint_sha256()
        }


class ResNet10Adapter(BaseModelAdapter):
    """Adapter for 3D ResNet-10 (Pipeline B)."""

    def __init__(self, ckpt_path: Path = Path("D:/ml project/results/v2/checkpoints/3d_resnet_10_seed42_best.pt")):
        super().__init__(
            model_name="3D ResNet-10 (Pipeline B)",
            paradigm="Supervised 3D CNN (Compact Residual)",
            checkpoint_path=ckpt_path,
            val_threshold=0.6860
        )

    def build_architecture(self) -> torch.nn.Module:
        from models.v2_models import ResNet10_3D
        return ResNet10_3D(output_dim=1, dropout=0.2)

    def forward_inference(self, volume: np.ndarray) -> float:
        if self.model_instance is None:
            self.load_weights()
        x = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model_instance(x)
            prob = float(torch.sigmoid(logit).item())
        return round(prob, 4)


class DenseNet121Adapter(BaseModelAdapter):
    """Adapter for 3D DenseNet-121 (Pipeline B)."""

    def __init__(self, ckpt_path: Path = Path("D:/ml project/results/v2/checkpoints/3d_densenet_121_seed42_best.pt")):
        super().__init__(
            model_name="3D DenseNet-121 (Pipeline B)",
            paradigm="Supervised 3D CNN (Dense Feature Reuse)",
            checkpoint_path=ckpt_path,
            val_threshold=0.4651
        )

    def build_architecture(self) -> torch.nn.Module:
        from models.v2_models import DenseNet121_3D
        return DenseNet121_3D(output_dim=1, dropout=0.2)

    def forward_inference(self, volume: np.ndarray) -> float:
        if self.model_instance is None:
            self.load_weights()
        x = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model_instance(x)
            prob = float(torch.sigmoid(logit).item())
        return round(prob, 4)


class MultiView25DAdapter(BaseModelAdapter):
    """Adapter for 2.5D Multi-View ResNet-18 (Pipeline B)."""

    def __init__(self, ckpt_path: Path = Path("D:/ml project/results/v2/checkpoints/2.5d_multi_view_resnet_18_seed42_best.pt")):
        super().__init__(
            model_name="2.5D Multi-View ResNet-18 (Pipeline B)",
            paradigm="2.5D Triplanar Transfer (ImageNet Pretrained)",
            checkpoint_path=ckpt_path,
            val_threshold=0.9741
        )

    def build_architecture(self) -> torch.nn.Module:
        from models.v2_models import MultiView25D_ResNet18
        return MultiView25D_ResNet18(pretrained=False, output_dim=1, dropout=0.3)

    def forward_inference(self, volume: np.ndarray) -> float:
        if self.model_instance is None:
            self.load_weights()
        # Passes full 5D volume tensor (1, 1, 128, 128, 128)
        x = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model_instance(x)
            prob = float(torch.sigmoid(logit).item())
        return round(prob, 4)


class DualBranchHybridAdapter(BaseModelAdapter):
    """Adapter for Dual-Branch Hybrid (ResNet-10 + 6D Radiomics)."""

    def __init__(self, ckpt_path: Path = Path("D:/ml project/results/v2/checkpoints/dual_branch_hybrid_seed42_best.pt")):
        super().__init__(
            model_name="Dual-Branch Hybrid (ResNet-10 + Radiomics)",
            paradigm="Hybrid Deep Spatial + Anatomical Radiomics",
            checkpoint_path=ckpt_path,
            val_threshold=0.4763
        )

    def build_architecture(self) -> torch.nn.Module:
        from models.v2_models import DualBranchHybridModel
        return DualBranchHybridModel(spatial_backbone="resnet10", num_radiomics=6, output_dim=1)

    def forward_inference(self, volume: np.ndarray) -> float:
        if self.model_instance is None:
            self.load_weights()
        from preprocessing.enhanced_pipeline import AnatomicalROIExtractor
        extractor = AnatomicalROIExtractor()
        b = extractor.compute_morphometric_biomarkers(volume)
        rad_vec = np.array([
            b["total_brain_voxels"] / 1e5,
            b["ventricular_csf_voxels"] / 1e4,
            b["hippocampal_tissue_voxels"] / 1e4,
            b["ventricle_to_brain_ratio"] * 10.0,
            b["hippocampus_ratio"] * 10.0,
            b["intensity_skewness"]
        ], dtype=np.float32)

        x = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(self.device)
        rad = torch.from_numpy(rad_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model_instance(x, rad)
            prob = float(torch.sigmoid(logit).item())
        return round(prob, 4)


class NewModelTemplateAdapter(BaseModelAdapter):
    """
    Template adapter for integrating prospective new architectures (e.g. Vision Mamba, Med3D).
    """

    def __init__(self, model_name: str = "Vision Mamba 3D", ckpt_path: Optional[Path] = None):
        super().__init__(
            model_name=model_name,
            paradigm="Prospective Architecture",
            checkpoint_path=ckpt_path,
            val_threshold=0.5000
        )

    def build_architecture(self) -> torch.nn.Module:
        return torch.nn.Identity()

    def forward_inference(self, volume: np.ndarray) -> float:
        raise NotImplementedError("Checkpoints for prospective new models not yet trained.")
