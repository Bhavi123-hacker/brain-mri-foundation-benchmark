import os
import pandas as pd
import numpy as np

m_path = r"D:\brain_mri_data\OASIS2\manifests\oasis2_baseline_manifest.csv"
if os.path.exists(m_path):
    df = pd.read_csv(m_path)
    print("OASIS2 manifest length:", len(df))
    print(df.head(2))
    t_dir = r"D:\brain_mri_data\OASIS2\processed\tensors"
    if os.path.exists(t_dir):
        files = os.listdir(t_dir)
        print("OASIS2 tensors count:", len(files), files[:3])
        sample = np.load(os.path.join(t_dir, files[0]))
        print("OASIS2 sample shape:", sample.shape, sample.dtype)
else:
    print("OASIS2 manifest not found at", m_path)
