import os
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.v2_models import ResNet10_3D, DenseNet121_3D, MultiView25D_ResNet18, DualBranchHybridModel
from models.baseline.resnet3d import build_resnet18_3d, build_resnet34_3d
from models.foundation_brainiac.vit_adapter import BrainIACViTAdapter
from models.foundation_swin.adapter import SwinUNETRSSLAdapter
from models.foundation_fomo25.amaes_adapter import AMAESResEncBAdapter
from preprocessing.enhanced_pipeline import AnatomicalROIExtractor

print("Testing model instantiations...")

m1 = ResNet10_3D(output_dim=1)
print("ResNet10_3D instantiated.")

m2 = build_resnet18_3d(output_dim=1)
print("ResNet18_3D instantiated.")

m3 = build_resnet34_3d(output_dim=1)
print("ResNet34_3D instantiated.")

m4 = DenseNet121_3D(output_dim=1)
print("DenseNet121_3D instantiated.")

m5 = MultiView25D_ResNet18(pretrained=True, output_dim=1)
print("MultiView25D_ResNet18 instantiated.")

m6 = DualBranchHybridModel(spatial_backbone="resnet10", num_radiomics=6, output_dim=1)
print("DualBranchHybridModel instantiated.")

m7 = BrainIACViTAdapter(adaptation_mode="linear_probe")
print("BrainIACViTAdapter instantiated.")

m8 = SwinUNETRSSLAdapter(weights_path=r"weights\model_swinvit.pt", adaptation_mode="linear_probe")
print("SwinUNETRSSLAdapter instantiated.")

m9 = AMAESResEncBAdapter()
print("AMAESResEncBAdapter instantiated.")

roi = AnatomicalROIExtractor()
print("AnatomicalROIExtractor instantiated.")

print("\nALL MODELS INSTANTIATED SUCCESSFULLY!")
