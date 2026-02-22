"""
Protocol definitions for the universal model registry.

This module contains zero heavy imports (no torch/timm/transformers) and defines
the core abstractions: ModelConfig, EmbeddingModel, and ModelProvider.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, runtime_checkable

import numpy as np

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


@dataclass
class ModelConfig:
    """Configuration describing a model in the registry.

    Attributes:
        model_id: Provider-specific identifier (HF repo, timm architecture, ONNX path).
        provider: Provider name (``"huggingface"``, ``"timm"``, ``"onnx"``, ``"torch"``).
        task: ``"image"`` or ``"text"`` -- determines wrapper class in HF provider.
        embedding_dim: Output embedding dimension. 0 means auto-detect.
        input_size: Expected input spatial size (pixels).
        normalize_mean: Channel-wise mean for input normalization.
        normalize_std: Channel-wise std for input normalization.
        pooling: Pooling strategy (``"cls"``, ``"mean"``, ``"cls_mean"``).
        provider_kwargs: Escape hatch for provider-specific settings.
        description: Human-readable description of the model.
    """

    model_id: str
    provider: str
    task: str = "image"
    embedding_dim: int = 0
    input_size: int = 224
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    pooling: str = "cls"
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@runtime_checkable
class EmbeddingModel(Protocol):
    """Universal interface for embedding models."""

    @property
    def embedding_dim(self) -> int: ...

    @property
    def device(self) -> str: ...

    def generate_embeddings(self, inputs: Any, batch_size: int = 32) -> np.ndarray: ...


class ModelProvider(Protocol):
    """Knows how to create :class:`EmbeddingModel` instances from a specific ecosystem."""

    def load(
        self,
        config: ModelConfig,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> EmbeddingModel: ...
