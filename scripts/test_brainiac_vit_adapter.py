import sys
from pathlib import Path
sys.path.insert(0, str(Path("D:/ml project")))
import torch
from models.foundation_brainiac.vit_adapter import BrainIACViTAdapter

print("Initializing BrainIACViTAdapter...")
model = BrainIACViTAdapter(adaptation_mode="linear_probe")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}, Trainable (linear probe): {trainable_params:,}")

# Test on 128^3 input (interpolated to 96^3 internally)
x = torch.randn(2, 1, 128, 128, 128)
with torch.no_grad():
    feat = model.extract_features(x)
    out = model(x)
print(f"Extracted feature shape: {feat.shape}, Logits shape: {out.shape}")
assert feat.shape == (2, 768)
assert out.shape == (2, 1)

# Test partial fine-tuning mode
model.set_adaptation_mode("partial_ft")
trainable_partial = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable in partial_ft (top 2 blocks + head): {trainable_partial:,}")

print("BrainIACViTAdapter verified successfully!")
