"""
Universal provider-based model registry for HoneyBee.

Public API::

    from honeybee.models.registry import (
        EmbeddingModel,
        ModelConfig,
        load_model,
        register_model,
        register_provider,
        list_models,
    )
"""

from .protocol import EmbeddingModel, ModelConfig, ModelProvider
from .registry import (
    _PRESET_REGISTRY,
    list_models,
    load_model,
    register_model,
    register_provider,
)

__all__ = [
    "EmbeddingModel",
    "ModelConfig",
    "ModelProvider",
    "load_model",
    "register_model",
    "register_provider",
    "list_models",
    "_PRESET_REGISTRY",
]
