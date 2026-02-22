"""
Unit tests for the universal model registry.

All tests are heavily mocked -- no real model downloads or GPU required.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from honeybee.models.registry import (
    EmbeddingModel,
    ModelConfig,
    _PRESET_REGISTRY,
    list_models,
    load_model,
    register_model,
    register_provider,
)
from honeybee.models.registry.protocol import ModelProvider
from honeybee.models.registry.registry import _PROVIDER_REGISTRY, _ensure_providers


# ============================================================================
# ModelConfig
# ============================================================================


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_creation_minimal(self):
        cfg = ModelConfig(model_id="my/model", provider="timm")
        assert cfg.model_id == "my/model"
        assert cfg.provider == "timm"

    def test_defaults(self):
        cfg = ModelConfig(model_id="x", provider="huggingface")
        assert cfg.task == "image"
        assert cfg.embedding_dim == 0
        assert cfg.input_size == 224
        assert cfg.normalize_mean == (0.485, 0.456, 0.406)
        assert cfg.normalize_std == (0.229, 0.224, 0.225)
        assert cfg.pooling == "cls"
        assert cfg.provider_kwargs == {}
        assert cfg.description == ""

    def test_provider_kwargs(self):
        cfg = ModelConfig(
            model_id="x",
            provider="timm",
            provider_kwargs={"init_values": 1e-5, "dynamic_img_size": True},
        )
        assert cfg.provider_kwargs["init_values"] == 1e-5
        assert cfg.provider_kwargs["dynamic_img_size"] is True


# ============================================================================
# Preset Registry
# ============================================================================


class TestPresetRegistry:
    """Test preset registry contents."""

    EXPECTED_PRESETS = [
        "uni", "uni2", "virchow2", "remedis",
        "h-optimus", "gigapath", "phikon-v2", "medsiglip",
        "biomedclip", "rad-dino",
        "torchxrayvision-densenet", "torchxrayvision-resnet",
        "radimagenet-resnet50", "radimagenet-densenet121",
    ]

    @pytest.mark.parametrize("alias", EXPECTED_PRESETS)
    def test_preset_exists(self, alias):
        assert alias in _PRESET_REGISTRY

    @pytest.mark.parametrize("alias", EXPECTED_PRESETS)
    def test_preset_has_required_fields(self, alias):
        cfg = _PRESET_REGISTRY[alias]
        assert cfg.model_id != ""
        assert cfg.provider in (
            "timm", "huggingface", "onnx", "torch", "open_clip", "torchxrayvision"
        )
        assert cfg.embedding_dim > 0

    def test_list_models_returns_all(self):
        models = list_models()
        aliases = {m["alias"] for m in models}
        for preset in self.EXPECTED_PRESETS:
            assert preset in aliases

    def test_list_models_filter_by_task(self):
        models = list_models(task="image")
        assert len(models) > 0
        for m in models:
            assert m["task"] == "image"

    def test_list_models_returns_dicts(self):
        models = list_models()
        for m in models:
            assert "alias" in m
            assert "model_id" in m
            assert "provider" in m
            assert "embedding_dim" in m

    def test_uni_config(self):
        cfg = _PRESET_REGISTRY["uni"]
        assert cfg.embedding_dim == 1024
        assert cfg.provider == "timm"
        assert cfg.input_size == 224

    def test_uni2_config(self):
        cfg = _PRESET_REGISTRY["uni2"]
        assert cfg.embedding_dim == 1536
        assert cfg.provider == "timm"

    def test_virchow2_config(self):
        cfg = _PRESET_REGISTRY["virchow2"]
        assert cfg.embedding_dim == 2560
        assert cfg.pooling == "cls_mean"

    def test_remedis_config(self):
        cfg = _PRESET_REGISTRY["remedis"]
        assert cfg.embedding_dim == 2048
        assert cfg.provider == "onnx"


# ============================================================================
# load_model
# ============================================================================


class TestLoadModel:
    """Test load_model dispatching."""

    @patch("honeybee.models.registry.providers.TimmProvider.load")
    def test_preset_alias_resolution(self, mock_load):
        mock_model = MagicMock(spec=["embedding_dim", "device", "generate_embeddings"])
        mock_load.return_value = mock_model

        result = load_model("uni", model_path="/fake/weights.pt")

        assert result is mock_model
        mock_load.assert_called_once()
        config_arg = mock_load.call_args[0][0]
        assert config_arg.model_id == "MahmoodLab/UNI"
        assert config_arg.provider == "timm"

    @patch("honeybee.models.registry.providers.TimmProvider.load")
    def test_explicit_provider(self, mock_load):
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        result = load_model("some-org/some-model", provider="timm")

        assert result is mock_model
        config_arg = mock_load.call_args[0][0]
        assert config_arg.model_id == "some-org/some-model"

    def test_unknown_model_no_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            load_model("nonexistent-model-xyz")

    def test_unknown_model_error_lists_presets(self):
        with pytest.raises(ValueError, match="uni"):
            load_model("nonexistent-model-xyz")

    @patch("honeybee.models.registry.providers.TimmProvider.load")
    def test_model_path_forwarded(self, mock_load):
        mock_load.return_value = MagicMock()
        load_model("uni", model_path="/my/weights.pt")
        assert mock_load.call_args[1]["model_path"] == "/my/weights.pt"

    @patch("honeybee.models.registry.providers.TimmProvider.load")
    def test_kwargs_merged_into_provider_kwargs(self, mock_load):
        mock_load.return_value = MagicMock()
        load_model("uni", model_path="/w.pt", custom_flag=True)
        config_arg = mock_load.call_args[0][0]
        assert config_arg.provider_kwargs.get("custom_flag") is True

    @patch("honeybee.models.registry.providers.HuggingFaceProvider.load")
    def test_huggingface_preset(self, mock_load):
        mock_load.return_value = MagicMock()
        load_model("phikon-v2")
        config_arg = mock_load.call_args[0][0]
        assert config_arg.model_id == "owkin/phikon-v2"
        assert config_arg.provider == "huggingface"

    @patch("honeybee.models.registry.providers.OnnxProvider.load")
    def test_onnx_preset(self, mock_load):
        mock_load.return_value = MagicMock()
        load_model("remedis", model_path="/fake.onnx")
        config_arg = mock_load.call_args[0][0]
        assert config_arg.provider == "onnx"

    @patch("honeybee.models.registry.providers.TimmProvider.load")
    def test_provider_override_on_preset(self, mock_load):
        """Caller can override the default provider for a preset."""
        mock_load.return_value = MagicMock()
        load_model("phikon-v2", provider="timm")
        config_arg = mock_load.call_args[0][0]
        assert config_arg.provider == "timm"

    def test_unknown_provider_raises(self):
        _ensure_providers()
        with pytest.raises(ValueError, match="Unknown provider"):
            load_model("my-model", provider="nonexistent_provider")


# ============================================================================
# TimmProvider (mocked)
# ============================================================================


class TestTimmProvider:
    """Test TimmProvider with mocked timm."""

    def test_load_with_model_path(self):
        from honeybee.models.registry.providers import TimmProvider

        with patch("timm.create_model") as mock_create, \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.load", return_value={}):
            mock_model = MagicMock()
            mock_create.return_value = mock_model

            config = ModelConfig(
                model_id="vit_large_patch16_224",
                provider="timm",
                embedding_dim=1024,
            )

            provider = TimmProvider()
            result = provider.load(config, model_path="/fake/weights.pt")

            assert result is not None
            mock_create.assert_called_once()

    def test_load_pretrained_hf_hub(self):
        from honeybee.models.registry.providers import TimmProvider

        with patch("timm.create_model") as mock_create, \
             patch("torch.cuda.is_available", return_value=False):
            mock_model = MagicMock()
            mock_create.return_value = mock_model

            config = ModelConfig(
                model_id="MahmoodLab/UNI",
                provider="timm",
                embedding_dim=1024,
            )

            provider = TimmProvider()
            result = provider.load(config)

            assert result is not None
            # Should try hf-hub: prefix first
            call_args = mock_create.call_args_list[0]
            assert "hf-hub:" in call_args[0][0]


# ============================================================================
# HuggingFaceProvider (mocked)
# ============================================================================


class TestHuggingFaceProvider:
    """Test HuggingFaceProvider with mocked transformers."""

    def test_load_vision(self):
        from honeybee.models.registry.providers import HuggingFaceProvider

        with patch("torch.cuda.is_available", return_value=False), \
             patch("transformers.AutoModel") as mock_am, \
             patch("transformers.AutoImageProcessor") as mock_proc:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 1024
            mock_am.from_pretrained.return_value.to.return_value = mock_model
            mock_proc.from_pretrained.return_value = MagicMock()

            config = ModelConfig(
                model_id="owkin/phikon-v2",
                provider="huggingface",
                embedding_dim=1024,
                task="image",
            )

            provider = HuggingFaceProvider()
            result = provider.load(config)
            assert result is not None
            assert result.embedding_dim == 1024

    def test_load_text(self):
        from honeybee.models.registry.providers import HuggingFaceProvider

        with patch("torch.cuda.is_available", return_value=False), \
             patch("transformers.AutoModel") as mock_am, \
             patch("transformers.AutoTokenizer") as mock_tok:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_am.from_pretrained.return_value.to.return_value = mock_model
            mock_tok.from_pretrained.return_value = MagicMock()

            config = ModelConfig(
                model_id="microsoft/BiomedNLP-PubMedBERT",
                provider="huggingface",
                embedding_dim=768,
                task="text",
            )

            provider = HuggingFaceProvider()
            result = provider.load(config)
            assert result is not None
            assert result.embedding_dim == 768


# ============================================================================
# OnnxProvider (mocked)
# ============================================================================


class TestOnnxProvider:
    """Test OnnxProvider with mocked onnxruntime."""

    def test_requires_model_path(self):
        from honeybee.models.registry.providers import OnnxProvider

        config = ModelConfig(model_id="remedis", provider="onnx", embedding_dim=2048)
        provider = OnnxProvider()

        with pytest.raises(ValueError, match="model_path is required"):
            provider.load(config, model_path=None)

    def test_load_with_path(self):
        from honeybee.models.registry.providers import OnnxProvider

        with patch("onnxruntime.InferenceSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_input = MagicMock()
            mock_input.name = "input"
            mock_output = MagicMock()
            mock_output.name = "output"
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]
            mock_session_cls.return_value = mock_session

            config = ModelConfig(model_id="remedis", provider="onnx", embedding_dim=2048)
            provider = OnnxProvider()
            result = provider.load(config, model_path="/fake/model.onnx")

            assert result is not None
            assert result.embedding_dim == 2048


# ============================================================================
# TorchProvider (mocked)
# ============================================================================


class TestTorchProvider:
    """Test TorchProvider with mocked torch."""

    def test_load_with_model_path(self):
        from honeybee.models.registry.providers import TorchProvider

        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.load", return_value={}), \
             patch("torch.nn.Identity", return_value=MagicMock()), \
             patch("torchvision.models") as mock_tv:
            mock_model = MagicMock()
            mock_model.fc = MagicMock()
            mock_model.fc.in_features = 2048
            mock_tv.resnet50 = MagicMock(return_value=mock_model)

            config = ModelConfig(
                model_id="resnet50",
                provider="torch",
                provider_kwargs={"arch": "resnet50"},
            )

            provider = TorchProvider()
            result = provider.load(config, model_path="/fake/weights.pt")
            assert result is not None


# ============================================================================
# register_model / register_provider
# ============================================================================


class TestRegisterModel:
    """Test custom model registration."""

    def test_register_and_lookup(self):
        alias = "_test_custom_model"
        config = ModelConfig(
            model_id="my-org/my-vit",
            provider="timm",
            embedding_dim=768,
        )
        register_model(alias, config)

        assert alias in _PRESET_REGISTRY
        assert _PRESET_REGISTRY[alias].embedding_dim == 768

        # Cleanup
        del _PRESET_REGISTRY[alias]

    def test_register_shows_in_list_models(self):
        alias = "_test_list_model"
        config = ModelConfig(model_id="test/model", provider="timm", embedding_dim=512)
        register_model(alias, config)

        aliases = {m["alias"] for m in list_models()}
        assert alias in aliases

        # Cleanup
        del _PRESET_REGISTRY[alias]


class TestRegisterProvider:
    """Test custom provider registration."""

    def test_register_custom_provider(self):
        mock_provider = MagicMock(spec=["load"])
        register_provider("_test_provider", mock_provider)

        assert "_test_provider" in _PROVIDER_REGISTRY

        # Cleanup
        del _PROVIDER_REGISTRY["_test_provider"]


# ============================================================================
# EmbeddingModel Protocol
# ============================================================================


class TestEmbeddingModelProtocol:
    """Test EmbeddingModel protocol runtime checking."""

    def test_conforming_class(self):
        class GoodModel:
            @property
            def embedding_dim(self) -> int:
                return 768

            @property
            def device(self) -> str:
                return "cpu"

            def generate_embeddings(self, inputs, batch_size=32):
                return np.zeros((1, 768))

        assert isinstance(GoodModel(), EmbeddingModel)

    def test_non_conforming_class(self):
        class BadModel:
            pass

        assert not isinstance(BadModel(), EmbeddingModel)

    def test_mock_model_with_spec(self):
        mock = MagicMock()
        mock.embedding_dim = 1024
        mock.device = "cpu"
        mock.generate_embeddings = MagicMock(return_value=np.zeros((1, 1024)))
        # MagicMock with these attributes satisfies the protocol
        assert hasattr(mock, "embedding_dim")
        assert hasattr(mock, "generate_embeddings")


class TestHFVisionModelSigLIPOutput:
    """Test _HFVisionModel handles SigLIP/CLIP output format."""

    def test_siglip_image_embeds_output(self):
        """Test that CLIP/SigLIP models use get_image_features for embedding extraction."""
        from honeybee.models.registry.providers import _HFVisionModel

        import torch

        # Create mock model with get_image_features (like SigLIP/CLIP)
        mock_model = MagicMock()
        mock_model.get_image_features = MagicMock(return_value=torch.randn(1, 768))

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        hf_model = _HFVisionModel(
            model=mock_model,
            processor=mock_processor,
            embedding_dim=768,
            device="cpu",
            pooling="cls",
        )

        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        embeddings = hf_model.generate_embeddings(test_image)

        assert embeddings is not None
        assert embeddings.shape == (1, 768)
        mock_model.get_image_features.assert_called_once()

    def test_standard_last_hidden_state_output(self):
        """Test that standard models with last_hidden_state still work."""
        from honeybee.models.registry.providers import _HFVisionModel

        import torch

        mock_model = MagicMock(spec=[])  # empty spec so get_image_features is absent
        mock_outputs = MagicMock()
        mock_outputs.image_embeds = None
        mock_outputs.last_hidden_state = torch.randn(1, 197, 768)
        mock_model.return_value = mock_outputs

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        hf_model = _HFVisionModel(
            model=mock_model,
            processor=mock_processor,
            embedding_dim=768,
            device="cpu",
            pooling="cls",
        )

        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        embeddings = hf_model.generate_embeddings(test_image)

        assert embeddings is not None
        assert embeddings.shape == (1, 768)
