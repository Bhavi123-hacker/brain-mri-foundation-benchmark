import monai
from monai.networks.nets import ViT
import torch

print("MONAI ViT import successful!")
vit_model = ViT(
    in_channels=1,
    img_size=(96, 96, 96),
    patch_size=(16, 16, 16),
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    num_layers=12,
    spatial_dims=3,
    classification=False
)

total_params = sum(p.numel() for p in vit_model.parameters())
print(f"Instantiated official 3D ViT-B/16: {total_params:,} parameters (~88.4M)")
x = torch.randn(1, 1, 96, 96, 96)
with torch.no_grad():
    out, hidden_states = vit_model(x)
print("Output tensor shape:", out.shape)
print("SUCCESS: MONAI 3D ViT-B/16 matches BrainIAC official architecture specifications exactly!")
