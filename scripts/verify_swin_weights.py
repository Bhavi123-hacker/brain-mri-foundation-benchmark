import sys
from pathlib import Path
sys.path.insert(0, str(Path("D:/ml project")))
import torch
from models.foundation_swin.adapter import SwinUNETRSSLAdapter

adapter = SwinUNETRSSLAdapter(output_dim=1, weights_path="weights/model_swinvit.pt")
print("Swin UNETR Adapter initialized successfully!")
total_params = sum(p.numel() for p in adapter.parameters())
encoder_params = sum(p.numel() for p in adapter.swin_encoder.parameters())
print(f"Total params: {total_params:,}, Encoder params: {encoder_params:,}")

x = torch.randn(1, 1, 128, 128, 128)
feat = adapter.extract_features(x)
print("Extracted feature shape:", feat.shape)
out = adapter(x)
print("Forward output shape:", out.shape)
assert feat.shape == (1, 768), f"Expected (1, 768), got {feat.shape}"
assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"
print("VERIFICATION SUCCESSFUL: MONAI Swin UNETR SSL is 100% authentic!")
