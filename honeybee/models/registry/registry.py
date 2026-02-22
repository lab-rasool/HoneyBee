"""
Model registry with preset aliases and provider dispatching.

Provides :func:`load_model` (the main entry point), preset model configs,
and extensibility via :func:`register_model` / :func:`register_provider`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .protocol import EmbeddingModel, ModelConfig, ModelProvider

# ---------------------------------------------------------------------------
# Preset model configurations
# ---------------------------------------------------------------------------

_PRESET_REGISTRY: Dict[str, ModelConfig] = {
    "uni": ModelConfig(
        model_id="MahmoodLab/UNI",
        provider="timm",
        embedding_dim=1024,
        input_size=224,
        pooling="cls",
        description="UNI ViT-L/16 pathology foundation model (MahmoodLab)",
        provider_kwargs={
            "img_size": 224,
            "patch_size": 16,
            "init_values": 1e-5,
            "dynamic_img_size": True,
        },
    ),
    "uni2": ModelConfig(
        model_id="MahmoodLab/UNI2-h",
        provider="timm",
        embedding_dim=1536,
        input_size=224,
        pooling="cls",
        description="UNI2-h ViT-H/14 pathology foundation model (MahmoodLab)",
        provider_kwargs={
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "mlp_layer": "timm.layers.SwiGLUPacked",
            "act_layer": "torch.nn.SiLU",
            "no_embed_class": True,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
    ),
    "virchow2": ModelConfig(
        model_id="paige-ai/Virchow2",
        provider="timm",
        embedding_dim=2560,
        input_size=224,
        pooling="cls_mean",
        description="Virchow2 ViT-H/14 pathology model (Paige AI) - cls+mean pooling",
        provider_kwargs={
            "mlp_layer": "timm.layers.SwiGLUPacked",
            "act_layer": "torch.nn.SiLU",
        },
    ),
    "remedis": ModelConfig(
        model_id="remedis",
        provider="onnx",
        embedding_dim=2048,
        input_size=224,
        description="REMEDIS CXR model (Google) - requires ONNX model_path",
    ),
    "h-optimus": ModelConfig(
        model_id="bioptimus/H-optimus-0",
        provider="timm",
        embedding_dim=1536,
        input_size=224,
        pooling="cls",
        normalize_mean=(0.707223, 0.578729, 0.703617),
        normalize_std=(0.211883, 0.230117, 0.177517),
        description="H-optimus-0 pathology foundation model (Bioptimus)",
    ),
    "gigapath": ModelConfig(
        model_id="prov-gigapath/prov-gigapath",
        provider="timm",
        embedding_dim=1536,
        input_size=224,
        pooling="cls",
        description="Prov-GigaPath DINOv2-based pathology model",
    ),
    "phikon-v2": ModelConfig(
        model_id="owkin/phikon-v2",
        provider="huggingface",
        embedding_dim=1024,
        input_size=224,
        pooling="cls",
        description="Phikon-v2 pathology foundation model (Owkin)",
    ),
    "medsiglip": ModelConfig(
        model_id="google/medsiglip-448",
        provider="huggingface",
        embedding_dim=1152,
        input_size=448,
        pooling="mean",
        description="MedSigLIP medical image-text model (Google) - 448x448",
    ),
    "biomedclip": ModelConfig(
        model_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        provider="open_clip",
        embedding_dim=512,
        input_size=224,
        pooling="cls",
        description="BiomedCLIP vision-language model (Microsoft) - image encoder",
    ),
    "rad-dino": ModelConfig(
        model_id="microsoft/rad-dino",
        provider="huggingface",
        embedding_dim=768,
        input_size=518,
        pooling="cls",
        description="RAD-DINO radiology foundation model (Microsoft)",
    ),
    "torchxrayvision-densenet": ModelConfig(
        model_id="densenet121-res224-all",
        provider="torchxrayvision",
        embedding_dim=1024,
        input_size=224,
        pooling="mean",
        description="TorchXRayVision DenseNet121 (all datasets, 224px)",
    ),
    "torchxrayvision-resnet": ModelConfig(
        model_id="resnet50-res512-all",
        provider="torchxrayvision",
        embedding_dim=2048,
        input_size=512,
        pooling="mean",
        description="TorchXRayVision ResNet50 (all datasets, 512px)",
    ),
    "radimagenet-resnet50": ModelConfig(
        model_id="ResNet50",
        provider="torch",
        embedding_dim=2048,
        input_size=224,
        pooling="mean",
        description="RadImageNet ResNet50 pretrained model",
        provider_kwargs={"arch": "resnet50"},
    ),
    "radimagenet-densenet121": ModelConfig(
        model_id="DenseNet121",
        provider="torch",
        embedding_dim=1024,
        input_size=224,
        pooling="mean",
        description="RadImageNet DenseNet121 pretrained model",
        provider_kwargs={"arch": "densenet121"},
    ),
}


# ---------------------------------------------------------------------------
# Provider registry (lazily populated)
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: Dict[str, ModelProvider] = {}
_PROVIDERS_INITIALIZED = False


def _ensure_providers() -> None:
    global _PROVIDERS_INITIALIZED
    if _PROVIDERS_INITIALIZED:
        return

    from .providers import HuggingFaceProvider, OnnxProvider, TimmProvider, TorchProvider

    _PROVIDER_REGISTRY.setdefault("timm", TimmProvider())
    _PROVIDER_REGISTRY.setdefault("huggingface", HuggingFaceProvider())
    _PROVIDER_REGISTRY.setdefault("onnx", OnnxProvider())
    _PROVIDER_REGISTRY.setdefault("torch", TorchProvider())

    # Optional providers - gracefully handle missing dependencies
    try:
        from .providers import OpenCLIPProvider
        _PROVIDER_REGISTRY.setdefault("open_clip", OpenCLIPProvider())
    except ImportError:
        pass

    try:
        from .providers import TorchXRayVisionProvider
        _PROVIDER_REGISTRY.setdefault("torchxrayvision", TorchXRayVisionProvider())
    except ImportError:
        pass

    _PROVIDERS_INITIALIZED = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(
    model: str,
    provider: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> EmbeddingModel:
    """Resolve a model alias or explicit provider and return an :class:`EmbeddingModel`.

    Parameters
    ----------
    model:
        A preset alias (e.g. ``"uni"``, ``"h-optimus"``) or a provider-specific
        model identifier (e.g. ``"bioptimus/H-optimus-0"``).
    provider:
        Override the provider. Required when *model* is not a preset alias.
    model_path:
        Path to local model weights. Required for ONNX models.
    device:
        Device string (``"cuda"``, ``"cpu"``). Auto-detected if ``None``.
    **kwargs:
        Extra keys merged into :attr:`ModelConfig.provider_kwargs`.

    Returns
    -------
    EmbeddingModel
        Ready-to-use embedding model.

    Raises
    ------
    ValueError
        If *model* is not a known preset and *provider* is not given, or if
        the specified provider is not registered.
    """
    _ensure_providers()

    # Lookup preset
    config = _PRESET_REGISTRY.get(model)

    if config is not None:
        # Override provider / kwargs if caller supplied them
        effective_provider = provider or config.provider
        merged_kwargs = {**config.provider_kwargs, **kwargs}
        config = ModelConfig(
            model_id=config.model_id,
            provider=effective_provider,
            task=config.task,
            embedding_dim=config.embedding_dim,
            input_size=config.input_size,
            normalize_mean=config.normalize_mean,
            normalize_std=config.normalize_std,
            pooling=config.pooling,
            provider_kwargs=merged_kwargs,
            description=config.description,
        )
    elif provider is not None:
        # Ad-hoc config from explicit provider
        config = ModelConfig(
            model_id=model,
            provider=provider,
            provider_kwargs=dict(kwargs),
        )
    else:
        preset_list = ", ".join(sorted(_PRESET_REGISTRY))
        raise ValueError(
            f"Unknown model: {model!r}. "
            f"Known presets: {preset_list}. "
            f"For a custom model, pass provider= explicitly, e.g.:\n"
            f"  PathologyProcessor(model={model!r}, provider='timm')\n"
            f"Or register it first:\n"
            f"  from honeybee.models.registry import register_model, ModelConfig\n"
            f"  register_model({model!r}, ModelConfig(model_id=..., provider=...))"
        )

    # Resolve provider instance
    prov = _PROVIDER_REGISTRY.get(config.provider)
    if prov is None:
        registered = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(
            f"Unknown provider: {config.provider!r}. Registered: {registered}"
        )

    return prov.load(config, model_path=model_path, device=device)


def register_model(alias: str, config: ModelConfig) -> None:
    """Register a custom model alias in the preset registry.

    Example::

        register_model("my-vit", ModelConfig(
            model_id="my-org/my-pathology-vit",
            provider="timm",
            embedding_dim=768,
        ))
    """
    _PRESET_REGISTRY[alias] = config


def register_provider(name: str, provider: ModelProvider) -> None:
    """Register a custom model provider."""
    _ensure_providers()
    _PROVIDER_REGISTRY[name] = provider


def list_models(task: Optional[str] = None) -> List[Dict[str, Any]]:
    """List registered model presets, optionally filtered by task.

    Returns a list of dicts with keys: ``alias``, ``model_id``, ``provider``,
    ``embedding_dim``, ``task``, ``description``.
    """
    results = []
    for alias, cfg in sorted(_PRESET_REGISTRY.items()):
        if task is not None and cfg.task != task:
            continue
        results.append(
            {
                "alias": alias,
                "model_id": cfg.model_id,
                "provider": cfg.provider,
                "embedding_dim": cfg.embedding_dim,
                "task": cfg.task,
                "description": cfg.description,
            }
        )
    return results
