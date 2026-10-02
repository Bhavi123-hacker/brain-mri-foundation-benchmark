"""
MONAI & PyTorch 3D Medical Preprocessing and Augmentation Transforms.
Supports NIfTI (.nii, .nii.gz) and NumPy (.npy) volumetric medical scans seamlessly.
"""

from typing import Tuple, List, Optional, Union, Any
import numpy as np
import torch
from monai.transforms import (
    Transform,
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    NormalizeIntensity,
    ScaleIntensity,
    Resize,
    RandFlip,
    RandGaussianNoise,
    ToTensor,
)


class LoadVolumetricImage(Transform):
    """
    Unified image loader supporting NIfTI (.nii, .nii.gz) via MONAI LoadImage
    and NumPy array volumes (.npy) via numpy.load.
    Always produces a 4D PyTorch FloatTensor of shape (1, D, H, W).
    """
    def __init__(self, reader: Optional[Any] = None):
        super().__init__()
        self.monai_loader = LoadImage(image_only=True, ensure_channel_first=True, reader=reader)
        self.orientation = Orientation(axcodes="RAS")

    def __call__(self, img_input: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(img_input, str):
            if img_input.endswith(".npy"):
                data = np.load(img_input)
                tensor_data = torch.from_numpy(data).to(torch.float32)
                if tensor_data.ndim == 3:
                    tensor_data = tensor_data.unsqueeze(0)  # Shape: (1, D, H, W)
                return tensor_data
            else:
                loaded = self.monai_loader(img_input)
                try:
                    loaded = self.orientation(loaded)
                except Exception:
                    pass
                return torch.as_tensor(loaded, dtype=torch.float32)
        elif isinstance(img_input, np.ndarray):
            tensor_data = torch.from_numpy(img_input).to(torch.float32)
            if tensor_data.ndim == 3:
                tensor_data = tensor_data.unsqueeze(0)
            return tensor_data
        elif isinstance(img_input, torch.Tensor):
            tensor_data = img_input.to(torch.float32)
            if tensor_data.ndim == 3:
                tensor_data = tensor_data.unsqueeze(0)
            return tensor_data
        else:
            raise TypeError(f"Unsupported image input type: {type(img_input)}")


class SafeSpacing(Transform):
    """
    Applies MONAI Spacing only when spatial affine metadata is attached to MetaTensor.
    Gracefully passes raw 4D volumetric tensors through if affine is absent.
    """
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        self.spacing_transform = Spacing(pixdim=voxel_spacing, mode="bilinear", align_corners=True)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(img_tensor, "meta") and "affine" in img_tensor.meta:
            try:
                return self.spacing_transform(img_tensor)
            except Exception:
                return img_tensor
        return img_tensor


def get_preprocessing_transforms(
    spatial_shape: Tuple[int, int, int] = (128, 128, 128),
    voxel_spacing: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    intensity_norm: str = "z_score",
    is_training: bool = False,
) -> Compose:
    """
    Construct a deterministic or augmented 3D MONAI preprocessing pipeline.

    Args:
        spatial_shape: Target 3D volume shape (D, H, W).
        voxel_spacing: Target voxel spacing in mm (e.g., 1.0, 1.0, 1.0).
        intensity_norm: Type of intensity normalization ('z_score' or 'min_max').
        is_training: If True, appends 3D data augmentations.

    Returns:
        MONAI Compose transform object.
    """
    transforms_list = [
        LoadVolumetricImage(),
    ]

    if voxel_spacing is not None:
        transforms_list.append(SafeSpacing(voxel_spacing=voxel_spacing))

    # Intensity Normalization
    if intensity_norm == "z_score":
        transforms_list.append(
            NormalizeIntensity(nonzero=True, channel_wise=True)
        )
    elif intensity_norm == "min_max":
        transforms_list.append(
            ScaleIntensity(minv=0.0, maxv=1.0)
        )

    # Spatial Resizing / Cropping
    transforms_list.append(Resize(spatial_size=spatial_shape, mode="trilinear"))

    # Training Augmentations
    if is_training:
        transforms_list.extend([
            RandFlip(prob=0.5, spatial_axis=0),
            RandFlip(prob=0.5, spatial_axis=1),
            RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
        ])

    transforms_list.append(ToTensor(dtype=torch.float32))

    return Compose(transforms_list)
