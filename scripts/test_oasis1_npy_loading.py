"""
Quick diagnostic loader test for OASIS-1 .npy files using MONAI / PyTorch.
"""
import os
import numpy as np
import torch
from monai.transforms import LoadImage, EnsureChannelFirst, Resize, ScaleIntensity, NormalizeIntensity, Compose

def test_load():
    filepath = "D:/OASIS1_PILOT/OAS1_0044_MR1_t1.npy"
    print("Testing file:", filepath)
    
    # Direct numpy load
    raw_arr = np.load(filepath)
    print("Direct numpy shape:", raw_arr.shape, "dtype:", raw_arr.dtype)
    
    # Test MONAI LoadImage or custom loader
    # For numpy array, add channel dimension: (1, D, H, W)
    tensor_img = torch.from_numpy(raw_arr).unsqueeze(0)
    print("Tensor with channel dim:", tensor_img.shape)

if __name__ == "__main__":
    test_load()
