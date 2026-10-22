import numpy as np

arr = np.load(r"D:\brain_mri_data\OASIS1\processed\tensors\OAS1_0001_MR1.npy")
print("OASIS1 npy shape:", arr.shape, "dtype:", arr.dtype, "min:", arr.min(), "max:", arr.max())

arr2 = np.load(r"D:\brain_mri_data\ADNI\processed\tensors\002_S_0295_I238627.npy")
print("ADNI npy shape:", arr2.shape, "dtype:", arr2.dtype, "min:", arr2.min(), "max:", arr2.max())
