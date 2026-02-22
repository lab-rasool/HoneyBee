"""
Built-in model providers for the universal registry.

Each provider knows how to instantiate an :class:`EmbeddingModel` from a
specific ecosystem (timm, HuggingFace transformers, ONNX Runtime, PyTorch).

Heavy imports (torch, timm, transformers, onnxruntime) are deferred to the
provider's ``load`` method so that importing this module is cheap.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from .protocol import EmbeddingModel, ModelConfig

# ---------------------------------------------------------------------------
# Timm provider
# ---------------------------------------------------------------------------


class _TimmImageModel:
    """Wraps a timm model as an :class:`EmbeddingModel`."""

    def __init__(self, model, transform, embedding_dim: int, device: str, pooling: str):
        self._model = model
        self._transform = transform
        self._embedding_dim = embedding_dim
        self._device = device
        self._pooling = pooling

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return self._device

    def generate_embeddings(self, inputs: Any, batch_size: int = 32) -> np.ndarray:
        import torch
        from PIL import Image

        patches = self._to_list(inputs)
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            tensors = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if hasattr(img, "mode") and img.mode != "RGB":
                    img = img.convert("RGB")
                tensors.append(self._transform(img))

            batch_tensor = torch.stack(tensors).to(self._device)
            with torch.inference_mode():
                out = self._model(batch_tensor)

            out = self._pool(out)
            all_embeddings.append(out.cpu().numpy())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    # -- helpers --

    @staticmethod
    def _to_list(inputs) -> list:
        if isinstance(inputs, np.ndarray) and inputs.ndim == 4:
            return [inputs[i] for i in range(inputs.shape[0])]
        if isinstance(inputs, np.ndarray) and inputs.ndim == 3:
            return [inputs]
        if isinstance(inputs, list):
            return inputs
        return [inputs]

    def _pool(self, out):
        import torch

        if out.ndim == 2:
            return out
        # (B, seq, D)
        if self._pooling == "cls":
            return out[:, 0]
        elif self._pooling == "mean":
            return out.mean(dim=1)
        elif self._pooling == "cls_mean":
            cls_token = out[:, 0]
            mean_token = out[:, 1:].mean(dim=1)
            return torch.cat([cls_token, mean_token], dim=-1)
        return out[:, 0]


class TimmProvider:
    """Creates :class:`EmbeddingModel` from the ``timm`` library."""

    def load(
        self,
        config: ModelConfig,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> EmbeddingModel:
        import timm
        import torch
        from torchvision import transforms

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build timm kwargs from provider_kwargs
        timm_kwargs = dict(config.provider_kwargs)

        # Resolve string class references (e.g. "timm.layers.SwiGLUPacked" → class)
        for key in ("mlp_layer", "act_layer"):
            val = timm_kwargs.get(key)
            if isinstance(val, str):
                import importlib

                module_path, class_name = val.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                timm_kwargs[key] = getattr(mod, class_name)

        if model_path:
            # Load architecture then state dict
            model = timm.create_model(
                config.model_id,
                pretrained=False,
                num_classes=0,
                **timm_kwargs,
            )
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
        else:
            # Try HF hub first, then plain timm
            try:
                model = timm.create_model(
                    f"hf-hub:{config.model_id}",
                    pretrained=True,
                    num_classes=0,
                    **timm_kwargs,
                )
            except Exception:
                model = timm.create_model(
                    config.model_id,
                    pretrained=True,
                    num_classes=0,
                    **timm_kwargs,
                )

        model = model.to(device)
        model.eval()

        # Auto-detect embedding dim
        embedding_dim = config.embedding_dim
        if embedding_dim == 0:
            with torch.inference_mode():
                dummy = torch.randn(1, 3, config.input_size, config.input_size, device=device)
                dummy_out = model(dummy)
                if dummy_out.ndim == 2:
                    embedding_dim = dummy_out.shape[-1]
                elif dummy_out.ndim == 3:
                    if config.pooling == "cls_mean":
                        embedding_dim = dummy_out.shape[-1] * 2
                    else:
                        embedding_dim = dummy_out.shape[-1]

        transform = transforms.Compose(
            [
                transforms.Resize(config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=list(config.normalize_mean),
                    std=list(config.normalize_std),
                ),
            ]
        )

        return _TimmImageModel(model, transform, embedding_dim, device, config.pooling)


# ---------------------------------------------------------------------------
# HuggingFace provider
# ---------------------------------------------------------------------------


class _HFVisionModel:
    """Wraps a HuggingFace vision model as an :class:`EmbeddingModel`."""

    def __init__(self, model, processor, embedding_dim: int, device: str, pooling: str):
        self._model = model
        self._processor = processor
        self._embedding_dim = embedding_dim
        self._device = device
        self._pooling = pooling

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return self._device

    def generate_embeddings(self, inputs: Any, batch_size: int = 32) -> np.ndarray:
        import torch
        from PIL import Image

        patches = _TimmImageModel._to_list(inputs)
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            pil_images = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if hasattr(img, "mode") and img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img)

            proc_inputs = self._processor(images=pil_images, return_tensors="pt")
            proc_inputs = {k: v.to(self._device) for k, v in proc_inputs.items()}

            with torch.inference_mode():
                if hasattr(self._model, "get_image_features"):
                    # CLIP/SigLIP vision-language models: extract image
                    # features directly (full forward requires text input_ids)
                    emb = self._model.get_image_features(**proc_inputs)
                else:
                    outputs = self._model(**proc_inputs)
                    if hasattr(outputs, "last_hidden_state"):
                        hidden = outputs.last_hidden_state
                        emb = self._pool(hidden)
                    else:
                        raise ValueError(f"Unsupported model output: {type(outputs)}")
            all_embeddings.append(emb.cpu().numpy())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    def _pool(self, hidden):
        if self._pooling == "cls":
            return hidden[:, 0]
        elif self._pooling == "mean":
            return hidden.mean(dim=1)
        elif self._pooling == "cls_mean":
            import torch

            return torch.cat([hidden[:, 0], hidden[:, 1:].mean(dim=1)], dim=-1)
        return hidden[:, 0]


class _HFTextModel:
    """Wraps a HuggingFace text model as an :class:`EmbeddingModel`."""

    def __init__(self, model, tokenizer, embedding_dim: int, device: str, pooling: str):
        self._model = model
        self._tokenizer = tokenizer
        self._embedding_dim = embedding_dim
        self._device = device
        self._pooling = pooling

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return self._device

    def generate_embeddings(
        self,
        inputs: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        import torch

        if isinstance(inputs, str):
            inputs = [inputs]

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            with torch.inference_mode():
                outputs = self._model(**encoded)
                hidden = outputs.last_hidden_state

            emb = self._pool(hidden)
            all_embeddings.append(emb.cpu().numpy())

        return np.vstack(all_embeddings)

    def _pool(self, hidden):
        if self._pooling == "mean":
            return hidden.mean(dim=1)
        return hidden[:, 0]  # CLS


class HuggingFaceProvider:
    """Creates :class:`EmbeddingModel` from HuggingFace ``transformers``."""

    def load(
        self,
        config: ModelConfig,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> EmbeddingModel:
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        trust_remote_code = config.provider_kwargs.get("trust_remote_code", True)
        model_id = model_path or config.model_id

        if config.task == "text":
            return self._load_text(model_id, config, device, trust_remote_code)
        return self._load_vision(model_id, config, device, trust_remote_code)

    def _load_vision(self, model_id, config, device, trust_remote_code):
        from transformers import AutoImageProcessor, AutoModel

        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        ).to(device)
        model.eval()

        try:
            processor = AutoImageProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
        except Exception:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )

        embedding_dim = config.embedding_dim
        if embedding_dim == 0:
            embedding_dim = getattr(model.config, "hidden_size", 0)

        return _HFVisionModel(model, processor, embedding_dim, device, config.pooling)

    def _load_text(self, model_id, config, device, trust_remote_code):
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        ).to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )

        embedding_dim = config.embedding_dim
        if embedding_dim == 0:
            embedding_dim = getattr(model.config, "hidden_size", 0)

        pooling = config.provider_kwargs.get("text_pooling", config.pooling)
        return _HFTextModel(model, tokenizer, embedding_dim, device, pooling)


# ---------------------------------------------------------------------------
# ONNX provider
# ---------------------------------------------------------------------------


class _OnnxImageModel:
    """Wraps an ONNX Runtime session as an :class:`EmbeddingModel`."""

    def __init__(self, session, embedding_dim: int, input_size: int, mean, std):
        self._session = session
        self._embedding_dim = embedding_dim
        self._input_size = input_size
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return "cpu"

    def generate_embeddings(self, inputs: Any, batch_size: int = 32) -> np.ndarray:
        import cv2

        patches = _TimmImageModel._to_list(inputs)
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            preprocessed = np.empty(
                (len(batch), self._input_size, self._input_size, 3), dtype=np.float32
            )
            for j, patch in enumerate(batch):
                if hasattr(patch, "mode"):
                    patch = np.asarray(patch)
                resized = cv2.resize(
                    patch, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR
                )
                preprocessed[j] = (resized.astype(np.float32) / 255.0 - self._mean) / self._std

            pred = self._session.run([self._output_name], {self._input_name: preprocessed})[0]
            all_embeddings.append(pred)

        return np.vstack(all_embeddings)


class OnnxProvider:
    """Creates :class:`EmbeddingModel` from an ONNX model file."""

    def load(
        self,
        config: ModelConfig,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> EmbeddingModel:
        if model_path is None:
            raise ValueError(
                f"model_path is required for ONNX provider (model={config.model_id!r}). "
                "Pass model_path to PathologyProcessor or load_model()."
            )

        import onnxruntime as ort

        providers = [
            (
                "CUDAExecutionProvider",
                {"device_id": 0, "gpu_mem_limit": 24 * 1024 * 1024 * 1024},
            ),
            "CPUExecutionProvider",
        ]
        session = ort.InferenceSession(model_path, providers=providers)

        embedding_dim = config.embedding_dim
        if embedding_dim == 0:
            dummy = np.random.randn(
                1, config.input_size, config.input_size, 3
            ).astype(np.float32)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            out = session.run([output_name], {input_name: dummy})[0]
            embedding_dim = out.shape[-1]

        return _OnnxImageModel(
            session,
            embedding_dim,
            config.input_size,
            config.normalize_mean,
            config.normalize_std,
        )


# ---------------------------------------------------------------------------
# Torch provider
# ---------------------------------------------------------------------------


class _TorchImageModel:
    """Wraps a raw PyTorch model as an :class:`EmbeddingModel`."""

    def __init__(self, model, transform, embedding_dim: int, device: str):
        self._model = model
        self._transform = transform
        self._embedding_dim = embedding_dim
        self._device = device

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return self._device

    def generate_embeddings(self, inputs: Any, batch_size: int = 32) -> np.ndarray:
        import torch
        from PIL import Image

        patches = _TimmImageModel._to_list(inputs)
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            tensors = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if hasattr(img, "mode") and img.mode != "RGB":
                    img = img.convert("RGB")
                tensors.append(self._transform(img))

            batch_tensor = torch.stack(tensors).to(self._device)
            with torch.inference_mode():
                out = self._model(batch_tensor)

            if out.ndim == 3:
                out = out[:, 0]
            all_embeddings.append(out.cpu().numpy())

        return np.vstack(all_embeddings)


class TorchProvider:
    """Creates :class:`EmbeddingModel` from PyTorch state dicts or ``torch.hub``."""

    def load(
        self,
        config: ModelConfig,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> EmbeddingModel:
        import torch
        from torchvision import transforms

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        arch = config.provider_kwargs.get("arch", config.model_id)

        if model_path:
            # Load torchvision/custom architecture + state dict
            import torchvision.models as tv_models

            model_fn = getattr(tv_models, arch, None)
            if model_fn is None:
                raise ValueError(f"Unknown torchvision architecture: {arch}")
            model = model_fn(weights=None)

            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)

            # Remove classifier head if present
            if hasattr(model, "fc"):
                embedding_dim = model.fc.in_features
                model.fc = torch.nn.Identity()
            elif hasattr(model, "classifier"):
                if hasattr(model.classifier, "in_features"):
                    embedding_dim = model.classifier.in_features
                else:
                    embedding_dim = config.embedding_dim
                model.classifier = torch.nn.Identity()
            else:
                embedding_dim = config.embedding_dim
        else:
            # Try torch.hub
            hub_repo = config.provider_kwargs.get("hub_repo")
            if hub_repo:
                model = torch.hub.load(hub_repo, arch, pretrained=True)
            else:
                import torchvision.models as tv_models

                model_fn = getattr(tv_models, arch, None)
                if model_fn is None:
                    raise ValueError(f"Unknown torchvision architecture: {arch}")
                model = model_fn(weights="DEFAULT")

            if hasattr(model, "fc"):
                embedding_dim = model.fc.in_features
                model.fc = torch.nn.Identity()
            elif hasattr(model, "classifier"):
                if hasattr(model.classifier, "in_features"):
                    embedding_dim = model.classifier.in_features
                else:
                    embedding_dim = config.embedding_dim
                model.classifier = torch.nn.Identity()
            else:
                embedding_dim = config.embedding_dim

        if config.embedding_dim > 0:
            embedding_dim = config.embedding_dim

        model = model.to(device)
        model.eval()

        transform = transforms.Compose(
            [
                transforms.Resize(config.input_size),
                transforms.CenterCrop(config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=list(config.normalize_mean),
                    std=list(config.normalize_std),
                ),
            ]
        )

        return _TorchImageModel(model, transform, embedding_dim, device)


# ---------------------------------------------------------------------------
# OpenCLIP provider
# ---------------------------------------------------------------------------


class _OpenCLIPImageModel:
    """Wraps an OpenCLIP vision model as an :class:`EmbeddingModel`."""

    def __init__(self, model, preprocess, embedding_dim: int, device: str):
        self._model = model
        self._preprocess = preprocess
        self._embedding_dim = embedding_dim
        self._device = device

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return self._device

    def generate_embeddings(self, inputs: Any, batch_size: int = 32) -> np.ndarray:
        import torch
        from PIL import Image

        patches = _TimmImageModel._to_list(inputs)
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            tensors = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if hasattr(img, "mode") and img.mode != "RGB":
                    img = img.convert("RGB")
                tensors.append(self._preprocess(img))

            batch_tensor = torch.stack(tensors).to(self._device)
            with torch.inference_mode():
                features = self._model.encode_image(batch_tensor)
                # Normalize features
                features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(all_embeddings)


class OpenCLIPProvider:
    """Creates :class:`EmbeddingModel` from the ``open_clip`` library."""

    def load(
        self,
        config: ModelConfig,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> EmbeddingModel:
        import open_clip
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            config.model_id, pretrained=model_path or ""
        )
        model = model.to(device)
        model.train(False)

        embedding_dim = config.embedding_dim
        if embedding_dim == 0:
            with torch.inference_mode():
                dummy = torch.randn(1, 3, config.input_size, config.input_size, device=device)
                out = model.encode_image(dummy)
                embedding_dim = out.shape[-1]

        return _OpenCLIPImageModel(model, preprocess_val, embedding_dim, device)


# ---------------------------------------------------------------------------
# TorchXRayVision provider
# ---------------------------------------------------------------------------


class _TorchXRayVisionModel:
    """Wraps a torchxrayvision model as an :class:`EmbeddingModel`."""

    def __init__(self, model, embedding_dim: int, input_size: int, device: str):
        self._model = model
        self._embedding_dim = embedding_dim
        self._input_size = input_size
        self._device = device

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> str:
        return self._device

    def generate_embeddings(self, inputs: Any, batch_size: int = 32) -> np.ndarray:
        import torch
        import torchxrayvision as xrv

        patches = _TimmImageModel._to_list(inputs)
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            tensors = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    processed = self._prepare_xrv_input(img)
                else:
                    processed = self._prepare_xrv_input(np.asarray(img))
                tensors.append(processed)

            batch_tensor = torch.stack(tensors).to(self._device)
            with torch.inference_mode():
                features = self._model.features(batch_tensor)
                # Global average pooling
                if features.ndim == 4:
                    features = features.mean(dim=[2, 3])
            all_embeddings.append(features.cpu().numpy())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    def _prepare_xrv_input(self, img: np.ndarray) -> "torch.Tensor":
        """Prepare image for torchxrayvision (single-channel, [-1024, 1024] range)."""
        import cv2
        import torch

        # Convert to grayscale if RGB
        if img.ndim == 3 and img.shape[-1] == 3:
            gray = np.mean(img, axis=-1)
        elif img.ndim == 3 and img.shape[0] == 3:
            gray = np.mean(img, axis=0)
        else:
            gray = img.copy()

        # Resize
        gray = cv2.resize(gray.astype(np.float32), (self._input_size, self._input_size))

        # Scale to [-1024, 1024] (torchxrayvision convention)
        gray = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8)) * 2048 - 1024

        # Add channel dimension: (1, H, W)
        tensor = torch.from_numpy(gray).unsqueeze(0).float()
        return tensor


class TorchXRayVisionProvider:
    """Creates :class:`EmbeddingModel` from the ``torchxrayvision`` library."""

    def load(
        self,
        config: ModelConfig,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> EmbeddingModel:
        import torch
        import torchxrayvision as xrv

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_id = config.model_id

        # Determine model type from model_id
        if "densenet" in model_id.lower():
            model = xrv.models.DenseNet(weights=model_id)
        elif "resnet" in model_id.lower():
            model = xrv.models.ResNet(weights=model_id)
        else:
            raise ValueError(
                f"Unknown torchxrayvision model: {model_id}. "
                f"Expected format: 'densenet121-res224-all' or 'resnet50-res512-all'"
            )

        model = model.to(device)
        model.train(False)

        embedding_dim = config.embedding_dim
        if embedding_dim == 0:
            with torch.inference_mode():
                dummy = torch.randn(1, 1, config.input_size, config.input_size, device=device)
                features = model.features(dummy)
                if features.ndim == 4:
                    features = features.mean(dim=[2, 3])
                embedding_dim = features.shape[-1]

        return _TorchXRayVisionModel(model, embedding_dim, config.input_size, device)
