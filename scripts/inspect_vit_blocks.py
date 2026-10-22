import inspect
from monai.networks.nets import ViT
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

print("ViT signature:", inspect.signature(ViT.__init__))
print("TransformerBlock signature:", inspect.signature(TransformerBlock.__init__))
