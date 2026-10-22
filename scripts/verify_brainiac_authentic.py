import torch
from monai.networks.nets import ViT
from safetensors.torch import load_file

weights_path = "D:/ml project/weights/brainiac_vit_backbone.safetensors"
state_dict = load_file(weights_path)

vit = ViT(
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

load_res = vit.load_state_dict(state_dict, strict=False)
print("Unexpected keys:", len(load_res.unexpected_keys))
print("Missing keys:", len(load_res.missing_keys))
assert len(load_res.unexpected_keys) == 0, f"Unexpected keys found: {load_res.unexpected_keys}"
for k in load_res.missing_keys:
    assert "norm_cross_attn" in k, f"Critical unexpected missing key: {k}"

print("All missing keys are exclusively the optional MONAI 1.6 cross-attention norm layers!")
loaded_params = sum(state_dict[k].numel() for k in state_dict)
print(f"Total authentic loaded parameters: {loaded_params:,} (100% of checkpoint)")

# Forward pass on 96^3 tensor
x = torch.randn(2, 1, 96, 96, 96)
with torch.no_grad():
    out, hidden = vit(x)

print(f"Forward pass output: shape {out.shape}, all finite: {torch.isfinite(out).all().item()}")
# Token pooling (first token or mean pool across the 216 spatial tokens)
token_features = out.mean(dim=1)  # Shape (B, 768)
print("Pooled token features shape:", token_features.shape)
assert token_features.shape == (2, 768)
print("AUTHENTIC BrainIAC 3D ViT-B/16 VERIFIED WITH ZERO FLUMMOX!")
