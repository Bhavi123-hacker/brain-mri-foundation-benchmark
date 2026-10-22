"""
adapters package
Extensible dataset and model adapters for SHIFT-MRI.
"""

from .dataset_adapter import (
    BaseDatasetAdapter,
    OASIS1Adapter,
    ADNIAdapter,
    OASIS2Adapter,
    FutureDatasetAdapterTemplate
)

from .model_adapter import (
    BaseModelAdapter,
    ResNet10Adapter,
    DenseNet121Adapter,
    MultiView25DAdapter,
    DualBranchHybridAdapter,
    NewModelTemplateAdapter
)
