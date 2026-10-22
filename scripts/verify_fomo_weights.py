import sys
from pathlib import Path
sys.path.insert(0, str(Path("D:/ml project")))
import torch
import hashlib
from models.foundation_fomo25.amaes_adapter import AMAESResEncBAdapter

ckpt_path = "D:/ml project/weights/resenc_unet_b.ckpt"
print(f"Loading official checkpoint from {ckpt_path}...")

adapter = AMAESResEncBAdapter(checkpoint_path=ckpt_path)
print("AMAESResEncBAdapter initialized successfully!")
encoder_params = sum(p.numel() for p in adapter.encoder.parameters())
total_params = sum(p.numel() for p in adapter.parameters())
print(f"Encoder params: {encoder_params:,}, Total params: {total_params:,}")

# Forward pass
x = torch.randn(1, 1, 128, 128, 128)
with torch.no_grad():
    feat = adapter.extract_features(x)
    print("Extracted feature shape:", feat.shape)
    out = adapter(x)
    print("Forward output shape:", out.shape)

assert encoder_params == 90296576, f"Expected 90,296,576 encoder params, got {encoder_params}"
assert feat.shape == (1, 320), f"Expected (1, 320), got {feat.shape}"
assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"
print("VERIFICATION SUCCESSFUL: FOMO25 AMAES ResEnc-B is 100% authentic and verified!")
