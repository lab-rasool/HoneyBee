"""
Unit tests for PathologyProcessor

Tests all functionality of the pathology/WSI processing module including
tissue detection, stain normalization, patch extraction, and embedding generation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from honeybee.processors import PathologyProcessor


class TestPathologyProcessorInitialization:
    """Test PathologyProcessor initialization"""

    def test_init_default_model(self):
        """Test initialization with default UNI model"""
        processor = PathologyProcessor(model="uni")
        assert processor is not None
        assert processor.model_name == "uni"
        assert processor.embedding_model is None  # Lazy loading

    def test_init_with_model_path(self):
        """Test initialization with model path"""
        model_path = "/path/to/model.pt"
        processor = PathologyProcessor(model="uni", model_path=model_path)
        assert processor.model_path == model_path

    def test_init_different_models(self):
        """Test initialization with different models"""
        for model in ["uni", "uni2", "virchow2", "remedis"]:
            processor = PathologyProcessor(model=model)
            assert processor.model_name == model.lower()

    def test_init_invalid_model(self):
        """Test that invalid model name raises error"""
        with pytest.raises(ValueError, match="Unknown model"):
            PathologyProcessor(model="invalid_model")


@pytest.mark.requires_sample_data
class TestWSILoading:
    """Test WSI loading functionality"""

    def test_load_wsi_basic(self, sample_wsi_path):
        """Test basic WSI loading"""
        if sample_wsi_path is None:
            pytest.skip("Sample WSI not available")

        processor = PathologyProcessor()
        wsi = processor.load_wsi(sample_wsi_path)

        assert wsi is not None
        assert hasattr(wsi, "slide")
        assert hasattr(wsi, "slide_image_path")

    def test_load_wsi_with_params(self, sample_wsi_path):
        """Test WSI loading with custom parameters"""
        if sample_wsi_path is None:
            pytest.skip("Sample WSI not available")

        processor = PathologyProcessor()
        wsi = processor.load_wsi(sample_wsi_path, tile_size=256, max_patches=50, verbose=False)

        assert wsi is not None
        assert wsi.tileSize == 256


class TestTissueDetection:
    """Test tissue detection methods"""

    def test_detect_tissue_otsu(self, sample_wsi_patch):
        """Test tissue detection with Otsu method"""
        processor = PathologyProcessor()
        mask = processor.detect_tissue(sample_wsi_patch, method="otsu")

        assert mask is not None
        assert mask.dtype == bool or mask.dtype == np.uint8
        assert mask.shape == (sample_wsi_patch.shape[0], sample_wsi_patch.shape[1])

    def test_detect_tissue_hsv(self, sample_wsi_patch):
        """Test tissue detection with HSV method"""
        processor = PathologyProcessor()
        mask = processor.detect_tissue(sample_wsi_patch, method="hsv")

        assert mask is not None
        assert mask.shape == (sample_wsi_patch.shape[0], sample_wsi_patch.shape[1])

    def test_detect_tissue_otsu_hsv(self, sample_wsi_patch):
        """Test tissue detection with combined Otsu+HSV method"""
        processor = PathologyProcessor()
        mask = processor.detect_tissue(sample_wsi_patch, method="otsu_hsv")

        assert mask is not None
        assert mask.shape == (sample_wsi_patch.shape[0], sample_wsi_patch.shape[1])

    @pytest.mark.parametrize("method", ["otsu", "hsv", "otsu_hsv"])
    def test_detect_tissue_all_methods(self, sample_wsi_patch, method):
        """Test all tissue detection methods"""
        processor = PathologyProcessor()
        mask = processor.detect_tissue(sample_wsi_patch, method=method)

        assert mask is not None
        assert mask.shape[0] == sample_wsi_patch.shape[0]
        assert mask.shape[1] == sample_wsi_patch.shape[1]

    def test_detect_tissue_invalid_method(self, sample_wsi_patch):
        """Test that invalid method raises error"""
        processor = PathologyProcessor()
        with pytest.raises(Exception):
            processor.detect_tissue(sample_wsi_patch, method="invalid")


class TestStainNormalization:
    """Test stain normalization methods"""

    def test_normalize_stain_reinhard(self, sample_wsi_patch):
        """Test Reinhard stain normalization"""
        processor = PathologyProcessor()
        normalized = processor.normalize_stain(
            sample_wsi_patch, method="reinhard", use_target_params=True
        )

        assert normalized is not None
        assert normalized.shape == sample_wsi_patch.shape
        assert normalized.dtype == np.uint8

    def test_normalize_stain_macenko(self, sample_wsi_patch):
        """Test Macenko stain normalization"""
        processor = PathologyProcessor()
        normalized = processor.normalize_stain(
            sample_wsi_patch, method="macenko", use_target_params=True
        )

        assert normalized is not None
        assert normalized.shape == sample_wsi_patch.shape

    def test_normalize_stain_vahadane(self, sample_wsi_patch):
        """Test Vahadane stain normalization"""
        processor = PathologyProcessor()
        normalized = processor.normalize_stain(
            sample_wsi_patch, method="vahadane", use_target_params=True
        )

        assert normalized is not None
        assert normalized.shape == sample_wsi_patch.shape

    @pytest.mark.parametrize("method", ["reinhard", "macenko", "vahadane"])
    def test_normalize_all_methods(self, sample_wsi_patch, method):
        """Test all normalization methods"""
        processor = PathologyProcessor()
        normalized = processor.normalize_stain(
            sample_wsi_patch, method=method, use_target_params=True
        )

        assert normalized is not None
        assert normalized.shape == sample_wsi_patch.shape

    def test_normalize_with_target_image(self, sample_wsi_patch):
        """Test normalization with custom target image"""
        processor = PathologyProcessor()
        target = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        normalized = processor.normalize_stain(
            sample_wsi_patch, method="reinhard", target=target, use_target_params=False
        )

        assert normalized is not None

    def test_normalize_invalid_method(self, sample_wsi_patch):
        """Test that invalid method raises error"""
        processor = PathologyProcessor()
        with pytest.raises(ValueError, match="Unknown normalization method"):
            processor.normalize_stain(sample_wsi_patch, method="invalid")


class TestStainSeparation:
    """Test stain separation (H&E deconvolution)"""

    def test_separate_stains_basic(self, sample_wsi_patch):
        """Test basic stain separation"""
        processor = PathologyProcessor()
        stains = processor.separate_stains(sample_wsi_patch, method="hed")

        assert stains is not None
        assert isinstance(stains, dict)

        # Check for required channels
        assert "hematoxylin" in stains
        assert "eosin" in stains
        assert "dab" in stains

    def test_separate_stains_channels(self, sample_wsi_patch):
        """Test that separated channels have correct shapes"""
        processor = PathologyProcessor()
        stains = processor.separate_stains(sample_wsi_patch)

        h = stains["hematoxylin"]
        e = stains["eosin"]

        assert h.shape[0] == sample_wsi_patch.shape[0]
        assert h.shape[1] == sample_wsi_patch.shape[1]
        assert e.shape[0] == sample_wsi_patch.shape[0]
        assert e.shape[1] == sample_wsi_patch.shape[1]

    def test_separate_stains_rgb_outputs(self, sample_wsi_patch):
        """Test that RGB visualizations are provided"""
        processor = PathologyProcessor()
        stains = processor.separate_stains(sample_wsi_patch)

        # Check for RGB outputs
        assert "rgb_h" in stains
        assert "rgb_e" in stains

        if stains["rgb_h"] is not None:
            assert stains["rgb_h"].shape == sample_wsi_patch.shape


class TestPatchExtraction:
    """Test patch extraction from WSI"""

    @pytest.mark.requires_sample_data
    def test_extract_patches_basic(self, sample_wsi_path):
        """Test basic patch extraction"""
        if sample_wsi_path is None:
            pytest.skip("Sample WSI not available")

        processor = PathologyProcessor()
        wsi = processor.load_wsi(sample_wsi_path, tile_size=256, max_patches=10)

        patches = processor.extract_patches(
            wsi, patch_size=256, min_tissue_percentage=0.3, target_patch_size=224
        )

        assert patches is not None
        assert isinstance(patches, np.ndarray)
        if len(patches) > 0:
            assert patches.shape[1:] == (224, 224, 3)

    def test_extract_patches_with_mask(self, sample_wsi_path):
        """Test patch extraction with tissue mask"""
        if sample_wsi_path is None:
            pytest.skip("Sample WSI not available")

        processor = PathologyProcessor()
        wsi = processor.load_wsi(sample_wsi_path, max_patches=10)

        # Create dummy tissue mask
        tissue_mask = np.random.rand(10, 10) > 0.5

        patches = processor.extract_patches(
            wsi, tissue_mask=tissue_mask, patch_size=256, min_tissue_percentage=0.5
        )

        assert patches is not None


@pytest.mark.requires_models
class TestEmbeddingGeneration:
    """Test embedding generation (requires model weights)"""

    @patch("honeybee.models.UNI.uni.UNI")
    def test_generate_embeddings_mock(self, mock_uni, sample_wsi_patches):
        """Test embedding generation with mocked model"""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(10, 1024)
        mock_model_instance.load_model_and_predict.return_value = mock_tensor
        mock_uni.return_value = mock_model_instance

        processor = PathologyProcessor(model="uni", model_path="/fake/path.pt")
        processor.embedding_model = mock_model_instance

        embeddings = processor.generate_embeddings(sample_wsi_patches, batch_size=4)

        assert embeddings is not None
        assert embeddings.shape == (10, 1024)

    @patch("honeybee.models.UNI.uni.UNI")
    def test_generate_embeddings_batch_processing(self, mock_uni, sample_wsi_patches):
        """Test batch processing of patches"""
        mock_model_instance = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(4, 1024)
        mock_model_instance.load_model_and_predict.return_value = mock_tensor
        mock_uni.return_value = mock_model_instance

        processor = PathologyProcessor(model="uni", model_path="/fake/path.pt")
        processor.embedding_model = mock_model_instance

        embeddings = processor.generate_embeddings(sample_wsi_patches, batch_size=4)

        assert embeddings is not None
        # Should have called model multiple times for batches
        assert mock_model_instance.load_model_and_predict.call_count >= 1


class TestEmbeddingAggregation:
    """Test embedding aggregation methods"""

    def test_aggregate_mean(self, sample_embeddings):
        """Test mean aggregation"""
        processor = PathologyProcessor()
        aggregated = processor.aggregate_embeddings(sample_embeddings, method="mean")

        assert aggregated is not None
        assert aggregated.shape == (sample_embeddings.shape[1],)
        assert np.allclose(aggregated, np.mean(sample_embeddings, axis=0))

    def test_aggregate_max(self, sample_embeddings):
        """Test max aggregation"""
        processor = PathologyProcessor()
        aggregated = processor.aggregate_embeddings(sample_embeddings, method="max")

        assert aggregated is not None
        assert aggregated.shape == (sample_embeddings.shape[1],)
        assert np.allclose(aggregated, np.max(sample_embeddings, axis=0))

    def test_aggregate_median(self, sample_embeddings):
        """Test median aggregation"""
        processor = PathologyProcessor()
        aggregated = processor.aggregate_embeddings(sample_embeddings, method="median")

        assert aggregated is not None
        assert aggregated.shape == (sample_embeddings.shape[1],)

    def test_aggregate_std(self, sample_embeddings):
        """Test std aggregation"""
        processor = PathologyProcessor()
        aggregated = processor.aggregate_embeddings(sample_embeddings, method="std")

        assert aggregated is not None
        assert aggregated.shape == (sample_embeddings.shape[1],)

    def test_aggregate_concat(self, sample_embeddings):
        """Test concatenation aggregation (mean + std)"""
        processor = PathologyProcessor()
        aggregated = processor.aggregate_embeddings(sample_embeddings, method="concat")

        assert aggregated is not None
        assert aggregated.shape == (sample_embeddings.shape[1] * 2,)

    @pytest.mark.parametrize("method", ["mean", "max", "median", "std", "concat"])
    def test_aggregate_all_methods(self, sample_embeddings, method):
        """Test all aggregation methods"""
        processor = PathologyProcessor()
        aggregated = processor.aggregate_embeddings(sample_embeddings, method=method)

        assert aggregated is not None
        assert len(aggregated.shape) == 1

    def test_aggregate_empty_embeddings(self):
        """Test that empty embeddings raise error"""
        processor = PathologyProcessor()
        with pytest.raises(ValueError, match="No embeddings to aggregate"):
            processor.aggregate_embeddings(np.array([]))

    def test_aggregate_invalid_method(self, sample_embeddings):
        """Test that invalid method raises error"""
        processor = PathologyProcessor()
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            processor.aggregate_embeddings(sample_embeddings, method="invalid")


@pytest.mark.slow
@pytest.mark.requires_sample_data
class TestProcessSlide:
    """Test complete slide processing pipeline"""

    @patch("honeybee.models.UNI.uni.UNI")
    def test_process_slide_complete(self, mock_uni, sample_wsi_path):
        """Test complete slide processing pipeline"""
        if sample_wsi_path is None:
            pytest.skip("Sample WSI not available")

        # Setup mock model
        mock_model_instance = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(5, 1024)
        mock_model_instance.load_model_and_predict.return_value = mock_tensor
        mock_uni.return_value = mock_model_instance

        processor = PathologyProcessor(model="uni", model_path="/fake/path.pt")
        processor.embedding_model = mock_model_instance

        result = processor.process_slide(
            sample_wsi_path,
            normalize_stain=True,
            normalization_method="macenko",
            patch_size=256,
            min_tissue_percentage=0.3,
            max_patches=10,
        )

        assert result is not None
        assert "slide" in result
        assert "patches" in result
        assert "num_patches" in result

    def test_process_slide_without_normalization(self, sample_wsi_path):
        """Test slide processing without stain normalization"""
        if sample_wsi_path is None:
            pytest.skip("Sample WSI not available")

        processor = PathologyProcessor()
        result = processor.process_slide(sample_wsi_path, normalize_stain=False, max_patches=5)

        assert result is not None
        assert "slide" in result


class TestModelSupport:
    """Test support for different embedding models"""

    def test_uni_model_support(self):
        """Test UNI model initialization"""
        processor = PathologyProcessor(model="uni", model_path="/fake/path.pt")
        assert processor.model_name == "uni"

    def test_uni2_model_support(self):
        """Test UNI2 model initialization"""
        processor = PathologyProcessor(model="uni2", model_path="/fake/path.pt")
        assert processor.model_name == "uni2"

    def test_virchow2_model_support(self):
        """Test Virchow2 model initialization"""
        processor = PathologyProcessor(model="virchow2", model_path="/fake/path.pt")
        assert processor.model_name == "virchow2"

    def test_remedis_model_support(self):
        """Test REMEDIS model initialization"""
        processor = PathologyProcessor(model="remedis")
        assert processor.model_name == "remedis"


class TestErrorHandling:
    """Test error handling in various scenarios"""

    def test_load_nonexistent_wsi(self):
        """Test loading non-existent WSI file"""
        processor = PathologyProcessor()
        with pytest.raises(Exception):
            processor.load_wsi("/nonexistent/file.svs")

    def test_generate_embeddings_without_model(self, sample_wsi_patches):
        """Test that generating embeddings without model raises error"""
        processor = PathologyProcessor(model="uni")  # No model_path
        with pytest.raises(ValueError, match="Model path required"):
            processor.generate_embeddings(sample_wsi_patches)

    def test_normalize_invalid_shape(self):
        """Test normalization with invalid image shape"""
        processor = PathologyProcessor()
        invalid_image = np.random.rand(10, 10)  # Missing color channel

        with pytest.raises(Exception):
            processor.normalize_stain(invalid_image, method="reinhard")
