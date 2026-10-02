"""
Supervised 3D ResNet Baseline Architecture for Volumetric Medical Imaging.
Implements 3D ResNet-18, ResNet-34, and ResNet-50 for classification and regression tasks.
"""

import torch
import torch.nn as nn
from typing import Type, List, Optional, Union
from models.base import BaseBenchModel


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BasicBlock3D(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck3D(nn.Module):
    expansion: int = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet3D(BaseBenchModel):
    def __init__(
        self,
        block: Union[Type[BasicBlock3D], Type[Bottleneck3D]] = BasicBlock3D,
        layers: List[int] = [2, 2, 2, 2], # ResNet-18 default
        in_channels: int = 1,
        output_dim: int = 1,
        dropout: float = 0.2,
        task_type: str = "classification",
        model_name: str = "3D-ResNet-18"
    ):
        super().__init__(model_name=model_name, task_type=task_type, output_dim=output_dim)
        self.in_planes = 64
        self.block_expansion = block.expansion
        
        # Initial 3D Convolutional Stem
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # 4 Residual Stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global Pooling and Output Head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(512 * self.block_expansion, output_dim)

        self._init_weights()

    def _make_layer(
        self,
        block: Union[Type[BasicBlock3D], Type[Bottleneck3D]],
        planes: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        feats = self.dropout(feats)
        return self.fc(feats)


def build_resnet18_3d(output_dim: int = 1, task_type: str = "classification", in_channels: int = 1, dropout: float = 0.2) -> ResNet3D:
    """Factory for standard 3D ResNet-18 (~33M parameters)."""
    return ResNet3D(
        block=BasicBlock3D,
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
        output_dim=output_dim,
        dropout=dropout,
        task_type=task_type,
        model_name="3D-ResNet-18"
    )


def build_resnet34_3d(output_dim: int = 1, task_type: str = "classification", in_channels: int = 1, dropout: float = 0.2) -> ResNet3D:
    """Factory for 3D ResNet-34 (~63M parameters)."""
    return ResNet3D(
        block=BasicBlock3D,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        output_dim=output_dim,
        dropout=dropout,
        task_type=task_type,
        model_name="3D-ResNet-34"
    )


def build_resnet50_3d(output_dim: int = 1, task_type: str = "classification", in_channels: int = 1, dropout: float = 0.2) -> ResNet3D:
    """Factory for 3D ResNet-50 (~46M parameters with Bottlenecks)."""
    return ResNet3D(
        block=Bottleneck3D,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        output_dim=output_dim,
        dropout=dropout,
        task_type=task_type,
        model_name="3D-ResNet-50"
    )
