"""
Version 2 Benchmark Model Architectures:
1. ResNet10_3D: Compact 3D Residual CNN (~14.4M params)
2. DenseNet121_3D: 3D Densely Connected CNN (~11.2M params)
3. MultiView25D_ResNet18: Multi-Planar Orthogonal Slice Transfer Model (~11.2M params)
4. DualBranchHybridModel: Joint Spatial Deep Embedding + Morphometric Radiomics Fusion

All models inherit from BaseBenchModel and support forward pass, feature extraction,
parameter profiling, and freeze/unfreeze controls.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseBenchModel

# MONAI 3D models
try:
    from monai.networks.nets import resnet10, DenseNet121
    HAS_MONAI_NETS = True
except ImportError:
    HAS_MONAI_NETS = False

# Torchvision 2D transfer
try:
    from torchvision.models import resnet18, ResNet18_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class ResNet10_3D(BaseBenchModel):
    """
    Compact 3D ResNet-10 for low-data volumetric neuroimaging.
    Tests Hypothesis H2: lower parameter capacity reduces overfitting on small cohorts.
    """
    def __init__(self, task_type: str = "classification", output_dim: int = 1, dropout: float = 0.2):
        super().__init__(model_name="3D ResNet-10", task_type=task_type, output_dim=output_dim)
        assert HAS_MONAI_NETS, "MONAI is required for ResNet10_3D"
        self.backbone = resnet10(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=512,  # feature projection
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(512, output_dim)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        feat = self.dropout(feat)
        out = self.head(feat)
        return out


class DenseNet121_3D(BaseBenchModel):
    """
    3D DenseNet-121. Leverages dense connectivity and continuous feature reuse
    with a compact ~11.2M parameter footprint.
    """
    def __init__(self, task_type: str = "classification", output_dim: int = 1, dropout: float = 0.2):
        super().__init__(model_name="3D DenseNet-121", task_type=task_type, output_dim=output_dim)
        assert HAS_MONAI_NETS, "MONAI is required for DenseNet121_3D"
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=512,
            dropout_prob=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(512, output_dim)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        feat = self.dropout(feat)
        out = self.head(feat)
        return out


class MultiView25D_ResNet18(BaseBenchModel):
    """
    Multi-Planar 2.5D Orthogonal Slice Transfer Model.
    Extracts central Orthogonal views (Axial, Coronal, Sagittal) across 3 key offsets each,
    projects through a shared ImageNet-pretrained ResNet-18 2D backbone,
    and fuses representations via late pooling.
    Tests Hypothesis H3: 2.5D multi-view transfer balances 3D context with 2D pre-trained priors.
    """
    def __init__(
        self,
        task_type: str = "classification",
        output_dim: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super().__init__(model_name="2.5D Multi-View ResNet-18", task_type=task_type, output_dim=output_dim)
        assert HAS_TORCHVISION, "Torchvision is required for MultiView25D_ResNet18"

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # Remove default fc
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # Output: (B, 512, 1, 1)

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(p=dropout)
        # 3 views fused: axial, coronal, sagittal (each 512-dim -> 512 via view pooling or 1536 concatenation)
        self.fusion = nn.Linear(512 * 3, 256)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Linear(256, output_dim)

    def _extract_triplanar_slices(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts 3-channel slice groups for Axial, Coronal, Sagittal planes.
        x shape: (B, 1, D, H, W).
        Returns 3 tensors each of shape (B, 3, 224, 224).
        """
        B, C, D, H, W = x.shape
        # Central slices with +/- 6 offset
        # Axial (along D)
        ax = x[:, 0, [D // 2 - 6, D // 2, D // 2 + 6], :, :]  # (B, 3, H, W)
        # Coronal (along H)
        cor = x[:, 0, :, [H // 2 - 6, H // 2, H // 2 + 6], :]  # (B, D, 3, W) -> permute to (B, 3, D, W)
        cor = cor.permute(0, 2, 1, 3)
        # Sagittal (along W)
        sag = x[:, 0, :, :, [W // 2 - 6, W // 2, W // 2 + 6]]  # (B, D, H, 3) -> permute to (B, 3, D, H)
        sag = sag.permute(0, 3, 1, 2)

        # Bilinear resize to standard 224x224 for ResNet-18
        ax_224 = F.interpolate(ax, size=(224, 224), mode="bilinear", align_corners=False)
        cor_224 = F.interpolate(cor, size=(224, 224), mode="bilinear", align_corners=False)
        sag_224 = F.interpolate(sag, size=(224, 224), mode="bilinear", align_corners=False)

        return ax_224, cor_224, sag_224

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        ax, cor, sag = self._extract_triplanar_slices(x)
        # Encode each view
        f_ax = torch.flatten(self.encoder(ax), 1)    # (B, 512)
        f_cor = torch.flatten(self.encoder(cor), 1)  # (B, 512)
        f_sag = torch.flatten(self.encoder(sag), 1)  # (B, 512)

        # Multi-planar feature concatenation
        concat_feats = torch.cat([f_ax, f_cor, f_sag], dim=1)  # (B, 1536)
        fused = self.relu(self.fusion(concat_feats))           # (B, 256)
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        feats = self.dropout(feats)
        out = self.head(feats)
        return out


class DualBranchHybridModel(BaseBenchModel):
    """
    Dual-Branch Deep + Anatomical Radiomics Hybrid Model.
    Combines 3D convolutional spatial embeddings with 6 explicit morphometric
    volumetric biomarkers (hippocampal ratio, ventricle-to-brain ratio, tissue skewness).
    Tests Hypothesis H5: Synergy between macro-structural anatomy and deep distributed features.
    """
    def __init__(
        self,
        spatial_backbone: str = "resnet10",
        task_type: str = "classification",
        output_dim: int = 1,
        num_radiomics: int = 6,
        dropout: float = 0.25
    ):
        super().__init__(model_name=f"Dual-Branch Hybrid ({spatial_backbone} + Radiomics)", task_type=task_type, output_dim=output_dim)
        self.num_radiomics = num_radiomics

        if spatial_backbone == "resnet10":
            self.cnn = resnet10(spatial_dims=3, n_input_channels=1, num_classes=256)
            spatial_dim = 256
        elif spatial_backbone == "densenet121":
            self.cnn = DenseNet121(spatial_dims=3, in_channels=1, out_channels=256, dropout_prob=dropout)
            spatial_dim = 256
        else:
            raise ValueError(f"Unknown spatial backbone: {spatial_backbone}")

        # Radiomics projection head
        self.rad_mlp = nn.Sequential(
            nn.Linear(num_radiomics, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Combined classification head
        self.dropout = nn.Dropout(p=dropout)
        self.fusion_head = nn.Sequential(
            nn.Linear(spatial_dim + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_dim)
        )

    def extract_features(self, x: torch.Tensor, radiomics: Optional[torch.Tensor] = None) -> torch.Tensor:
        spatial_feat = self.cnn(x)
        if radiomics is None:
            # Fallback zero radiomics
            radiomics = torch.zeros((x.shape[0], self.num_radiomics), device=x.device, dtype=torch.float32)
        rad_feat = self.rad_mlp(radiomics)
        fused = torch.cat([spatial_feat, rad_feat], dim=1)
        return fused

    def forward(self, x: torch.Tensor, radiomics: Optional[torch.Tensor] = None) -> torch.Tensor:
        fused = self.extract_features(x, radiomics)
        out = self.fusion_head(fused)
        return out
