import torch

p1 = r"D:\brain_mri_data\OASIS1\processed\tensors\OAS1_0001_MR1.pt"
d1 = torch.load(p1, map_location="cpu", weights_only=False)
print("OASIS1 .pt keys:", list(d1.keys()))
t1 = d1["tensor"]
print(f"OASIS1 shape: {t1.shape}, dtype: {t1.dtype}, min: {t1.min():.4f}, max: {t1.max():.4f}, mean: {t1.mean():.4f}")
print("OASIS1 label:", d1.get("binary_label"), "cdr:", d1.get("cdr"))

p2 = r"D:\brain_mri_data\ADNI\processed\tensors\002_S_0295_I238627.pt"
d2 = torch.load(p2, map_location="cpu", weights_only=False)
print("\nADNI .pt keys:", list(d2.keys()))
t2 = d2["tensor"]
print(f"ADNI shape: {t2.shape}, dtype: {t2.dtype}, min: {t2.min():.4f}, max: {t2.max():.4f}, mean: {t2.mean():.4f}")
print("ADNI label:", d2.get("binary_label"), "subject_id:", d2.get("subject_id"), "mfg:", d2.get("manufacturer"))
