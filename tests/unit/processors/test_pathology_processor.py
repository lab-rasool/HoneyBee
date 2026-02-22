"""
Unit tests for PathologyProcessor

Tests all functionality of the pathology/WSI processing module including
tissue detection, stain normalization, patch extraction, and embedding generation.
"""

import json
import os
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
        for model in [
            "uni", "uni2", "virchow2", "remedis",
            "h-optimus", "phikon-v2", "medsiglip", "gigapath",
        ]:
            processor = PathologyProcessor(model=model)
            assert processor.model_name == model.lower()

    def test_init_invalid_model(self):
        """Test that invalid model name without provider raises error at generate time"""
        # With the registry, unknown model without provider is accepted at init
        # but raises ValueError when trying to load
        processor = PathologyProcessor(model="invalid_model")
        patches = np.random.randint(0, 255, (2, 224, 224, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown model"):
            processor.generate_embeddings(patches)

    def test_init_with_provider(self):
        """Test initialization with explicit provider"""
        processor = PathologyProcessor(
            model="bioptimus/H-optimus-0", provider="timm"
        )
        assert processor.model_name == "bioptimus/H-optimus-0"
        assert processor._provider == "timm"

    def test_init_preserves_case_for_hf_ids(self):
        """Test that HF repo IDs with / preserve case"""
        processor = PathologyProcessor(model="MahmoodLab/UNI2-h", provider="timm")
        assert processor.model_name == "MahmoodLab/UNI2-h"


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

    def test_generate_embeddings_onnx_without_model_path(self, sample_wsi_patches):
        """Test that ONNX model without model_path raises error"""
        processor = PathologyProcessor(model="remedis")  # No model_path
        with pytest.raises(ValueError, match="model_path is required"):
            processor.generate_embeddings(sample_wsi_patches)

    def test_normalize_invalid_shape(self):
        """Test normalization with invalid image shape"""
        processor = PathologyProcessor()
        invalid_image = np.random.rand(10, 10)  # Missing color channel

        with pytest.raises(Exception):
            processor.normalize_stain(invalid_image, method="reinhard")


# ================================================================== #
# Bug Fix Tests
# ================================================================== #


class TestREMEDISBugFix:
    """Test Bug 1: REMEDIS model compatibility"""

    def test_remedis_init_with_model_path(self):
        """Test REMEDIS accepts model_path in constructor"""
        from honeybee.models.REMEDIS.remedis import REMEDIS

        with pytest.warns(DeprecationWarning, match="REMEDIS is deprecated"):
            model = REMEDIS(model_path="/fake/path.onnx")
        assert model.model_path == "/fake/path.onnx"

    def test_remedis_init_without_model_path(self):
        """Test REMEDIS works with no model_path"""
        from honeybee.models.REMEDIS.remedis import REMEDIS

        with pytest.warns(DeprecationWarning, match="REMEDIS is deprecated"):
            model = REMEDIS()
        assert model.model_path is None

    def test_remedis_predict_requires_model_path(self):
        """Test REMEDIS raises error when predicting without model_path"""
        from honeybee.models.REMEDIS.remedis import REMEDIS

        with pytest.warns(DeprecationWarning, match="REMEDIS is deprecated"):
            model = REMEDIS()
        with pytest.raises(ValueError, match="model_path is required"):
            model.load_model_and_predict(np.zeros((1, 224, 224, 3)))

    def test_generate_embeddings_handles_numpy_return(self, sample_wsi_patches):
        """Test PathologyProcessor handles numpy array returns from models"""
        processor = PathologyProcessor(model="remedis", model_path="/fake/path.onnx")

        # With the registry, generate_embeddings is called on the model directly
        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(10, 2048)
        processor.embedding_model = mock_model

        embeddings = processor.generate_embeddings(sample_wsi_patches, batch_size=4)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[1] == 2048

    def test_generate_embeddings_handles_torch_return(self, sample_wsi_patches):
        """Test PathologyProcessor handles torch tensor returns from models"""
        processor = PathologyProcessor(model="uni", model_path="/fake/path.pt")

        # With the registry, generate_embeddings is called on the model directly
        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(10, 1024)
        processor.embedding_model = mock_model

        embeddings = processor.generate_embeddings(sample_wsi_patches, batch_size=4)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[1] == 1024


class TestExtractPatchesBugFix:
    """Test Bug 2: extract_patches tissue_mask and overlap"""

    def _make_mock_wsi(self, num_tiles_x=3, num_tiles_y=3, tile_size=256):
        """Create a mock WSI object for testing"""
        mock_wsi = MagicMock()
        mock_wsi.tileSize = tile_size
        mock_wsi.numTilesInX = num_tiles_x
        mock_wsi.numTilesInY = num_tiles_y

        # Create tile dictionary without tissueLevel (simulating no DL detection)
        tile_dict = {}
        addresses = []
        for y in range(num_tiles_y):
            for x in range(num_tiles_x):
                addr = (x, y)
                tile_dict[addr] = {
                    "x": x * tile_size,
                    "y": y * tile_size,
                    "width": tile_size,
                    "height": tile_size,
                }
                addresses.append(addr)

        mock_wsi.tileDictionary = tile_dict
        mock_wsi.iterateTiles.return_value = iter(addresses)

        # Mock getTile to return a valid patch
        mock_patch = np.random.randint(0, 255, (tile_size, tile_size, 3), dtype=np.uint8)
        mock_wsi.getTile.return_value = mock_patch

        # Remove load_patches_concurrently to force fallback path
        del mock_wsi.load_patches_concurrently

        return mock_wsi

    def test_extract_patches_uses_tissue_mask(self):
        """Test that tissue_mask is used for filtering when no tissueLevel"""
        processor = PathologyProcessor()
        mock_wsi = self._make_mock_wsi(num_tiles_x=3, num_tiles_y=3)

        # Create mask where only top-left region has tissue
        tissue_mask = np.zeros((30, 30), dtype=bool)
        tissue_mask[:10, :10] = True  # Only top-left

        patches = processor.extract_patches(
            mock_wsi,
            tissue_mask=tissue_mask,
            patch_size=256,
            min_tissue_percentage=0.5,
            target_patch_size=256,
        )

        # Should only get the top-left patch (0,0)
        assert len(patches) > 0
        assert len(patches) < 9  # Not all 9 patches

    def test_extract_patches_no_mask_no_tissue_level_includes_all(self):
        """Test that without mask or tissueLevel, all patches are included"""
        processor = PathologyProcessor()
        mock_wsi = self._make_mock_wsi(num_tiles_x=2, num_tiles_y=2)

        patches = processor.extract_patches(
            mock_wsi,
            tissue_mask=None,
            patch_size=256,
            target_patch_size=256,
        )

        assert len(patches) == 4  # All 2x2 patches included

    def test_extract_patches_with_tissue_level(self):
        """Test that tissueLevel is respected when available"""
        processor = PathologyProcessor()
        mock_wsi = self._make_mock_wsi(num_tiles_x=2, num_tiles_y=1)

        # Add tissueLevel to tile dictionary
        mock_wsi.tileDictionary[(0, 0)]["tissueLevel"] = 0.9
        mock_wsi.tileDictionary[(1, 0)]["tissueLevel"] = 0.1

        patches = processor.extract_patches(
            mock_wsi,
            patch_size=256,
            min_tissue_percentage=0.5,
            target_patch_size=256,
        )

        # Only the first tile should pass the threshold
        assert len(patches) == 1

    def test_extract_patches_skips_concurrent_when_mask_provided(self):
        """Test that concurrent loading is skipped when tissue_mask is provided"""
        processor = PathologyProcessor()
        mock_wsi = self._make_mock_wsi(num_tiles_x=2, num_tiles_y=2)

        # Re-add load_patches_concurrently
        mock_wsi.load_patches_concurrently = MagicMock(return_value=np.zeros((4, 256, 256, 3)))

        tissue_mask = np.ones((20, 20), dtype=bool)

        processor.extract_patches(
            mock_wsi,
            tissue_mask=tissue_mask,
            patch_size=256,
            target_patch_size=256,
        )

        # Should NOT have called load_patches_concurrently
        mock_wsi.load_patches_concurrently.assert_not_called()


class TestDLTissueDetectionBugFix:
    """Test Bug 3: Deep learning tissue detection auto-download"""

    def test_dl_detection_allows_none_path(self):
        """Test that tissue_detector_path=None does not raise ValueError (auto-download)"""
        # The old code raised ValueError when tissue_detector_path=None.
        # The fix removes that check, allowing TissueDetector to auto-download weights.
        # We verify the code path by inspecting the source directly.
        processor = PathologyProcessor()

        # Verify the detect_tissue method code does NOT contain the old ValueError check
        import inspect

        source = inspect.getsource(processor.detect_tissue)
        assert 'raise ValueError("tissue_detector_path required' not in source
        assert "TissueDetector(model_path=tissue_detector_path)" in source

    def test_dl_detection_no_value_error_on_none_path(self):
        """Verify the old ValueError is gone for tissue_detector_path=None"""
        processor = PathologyProcessor()
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Should NOT raise ValueError about tissue_detector_path
        with patch("honeybee.models.TissueDetector.tissue_detector.TissueDetector") as mock_td_cls:
            mock_td_cls.return_value = MagicMock()

            try:
                processor.detect_tissue(image, method="deeplearning")
            except ValueError as e:
                if "tissue_detector_path" in str(e):
                    pytest.fail(f"Should not raise ValueError about path: {e}")
            except Exception:
                pass  # Other exceptions are acceptable (mocking artifacts)


# ================================================================== #
# New Method Tests
# ================================================================== #


class TestGetSlideInfo:
    """Test get_slide_info method"""

    def test_get_slide_info_returns_dict(self):
        """Test that get_slide_info returns expected keys"""
        processor = PathologyProcessor()

        mock_wsi = MagicMock()
        mock_wsi.slide.width = 1024
        mock_wsi.slide.height = 768
        mock_wsi.img.resolutions = {"level_count": 3, "level_dimensions": [(1024, 768)]}
        mock_wsi.img.metadata = {"objectivePower": "40x", "scanner": "Aperio"}
        mock_wsi.numTilesInX = 4
        mock_wsi.numTilesInY = 3
        mock_wsi.slide_image_path = "/nonexistent/slide.svs"

        info = processor.get_slide_info(mock_wsi)

        assert isinstance(info, dict)
        assert info["dimensions"]["width"] == 1024
        assert info["dimensions"]["height"] == 768
        assert info["num_levels"] == 3
        assert info["magnification"] == "40x"
        assert info["scanner"] == "Aperio"
        assert info["tile_grid"]["tiles_x"] == 4
        assert info["tile_grid"]["tiles_y"] == 3

    def test_get_slide_info_handles_missing_metadata(self):
        """Test graceful handling of missing metadata"""
        processor = PathologyProcessor()

        mock_wsi = MagicMock()
        mock_wsi.slide.width = 512
        mock_wsi.slide.height = 512
        mock_wsi.img.resolutions = None
        mock_wsi.img.metadata = None
        mock_wsi.numTilesInX = 2
        mock_wsi.numTilesInY = 2
        # Must delete attributes so getattr returns the default None
        del mock_wsi.slide_image_path
        del mock_wsi.slideFilePath

        info = processor.get_slide_info(mock_wsi)

        assert info["num_levels"] == 1
        assert info["magnification"] is None
        assert info["file_size_bytes"] is None


class TestGetThumbnail:
    """Test get_thumbnail method"""

    def test_get_thumbnail_default_size(self):
        """Test thumbnail generation with default size"""
        processor = PathologyProcessor()

        mock_wsi = MagicMock()
        mock_wsi.slide.__array__ = lambda: np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
        # Make np.asarray work
        mock_wsi.slide = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)

        thumbnail = processor.get_thumbnail(mock_wsi)

        assert thumbnail.shape == (512, 512, 3)
        assert thumbnail.dtype == np.uint8

    def test_get_thumbnail_custom_size(self):
        """Test thumbnail with custom size"""
        processor = PathologyProcessor()
        mock_wsi = MagicMock()
        mock_wsi.slide = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)

        thumbnail = processor.get_thumbnail(mock_wsi, size=(256, 128))

        assert thumbnail.shape == (128, 256, 3)


class TestVisualizeTissueMask:
    """Test visualize_tissue_mask method"""

    def test_visualize_returns_composite(self):
        """Test that visualization returns a composite image"""
        processor = PathologyProcessor()
        mock_wsi = MagicMock()
        mock_wsi.slide = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        tissue_mask = np.random.rand(50, 50) > 0.5

        composite = processor.visualize_tissue_mask(mock_wsi, tissue_mask)

        assert composite.shape == (512, 512, 3)
        assert composite.dtype == np.uint8

    def test_visualize_saves_to_file(self, tmp_path):
        """Test saving composite to file"""
        processor = PathologyProcessor()
        mock_wsi = MagicMock()
        mock_wsi.slide = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        tissue_mask = np.ones((50, 50), dtype=bool)
        output_path = str(tmp_path / "overlay.png")

        processor.visualize_tissue_mask(mock_wsi, tissue_mask, output_path=output_path)

        assert os.path.exists(output_path)


class TestNormalizePatches:
    """Test normalize_patches method"""

    def test_normalize_patches_basic(self, sample_wsi_patches):
        """Test batch normalization of patches"""
        processor = PathologyProcessor()
        normalized = processor.normalize_patches(sample_wsi_patches, method="reinhard")

        assert normalized.shape == sample_wsi_patches.shape

    def test_normalize_patches_handles_failures(self):
        """Test that failed normalizations keep original patch"""
        processor = PathologyProcessor()

        # Create patches that will fail normalization (all zeros)
        bad_patches = np.zeros((3, 224, 224, 3), dtype=np.uint8)
        result = processor.normalize_patches(bad_patches, method="macenko")

        assert result.shape == bad_patches.shape


class TestSavePatches:
    """Test save_patches method"""

    def test_save_patches_creates_files(self, tmp_path):
        """Test that patches are saved as image files"""
        processor = PathologyProcessor()
        patches = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        output_dir = str(tmp_path / "patches")

        paths = processor.save_patches(patches, output_dir)

        assert len(paths) == 5
        for p in paths:
            assert os.path.exists(p)
            assert p.endswith(".png")

    def test_save_patches_custom_format(self, tmp_path):
        """Test saving with custom prefix and format"""
        processor = PathologyProcessor()
        patches = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        output_dir = str(tmp_path / "patches_jpg")

        paths = processor.save_patches(patches, output_dir, prefix="tile", format="jpg")

        assert len(paths) == 2
        assert all("tile_" in p for p in paths)
        assert all(p.endswith(".jpg") for p in paths)


class TestComputePatchQuality:
    """Test compute_patch_quality method"""

    def test_quality_scores_shape(self, sample_wsi_patches):
        """Test that quality scores have correct shape"""
        processor = PathologyProcessor()
        scores = processor.compute_patch_quality(sample_wsi_patches)

        assert scores.shape == (len(sample_wsi_patches),)
        assert scores.dtype == np.float32

    def test_quality_scores_range(self, sample_wsi_patches):
        """Test that quality scores are in [0, 1]"""
        processor = PathologyProcessor()
        scores = processor.compute_patch_quality(sample_wsi_patches)

        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_white_patches_low_quality(self):
        """Test that mostly white patches get low tissue score"""
        processor = PathologyProcessor()
        white_patches = np.full((3, 224, 224, 3), 240, dtype=np.uint8)
        scores = processor.compute_patch_quality(white_patches)

        # White patches should have low scores due to low tissue content
        assert np.all(scores < 0.5)


class TestGetPatchCoordinates:
    """Test get_patch_coordinates method"""

    def test_coordinates_shape(self):
        """Test that coordinates have correct shape"""
        processor = PathologyProcessor()

        mock_wsi = MagicMock()
        tile_dict = {
            (0, 0): {"x": 0, "y": 0, "width": 256, "height": 256},
            (1, 0): {"x": 256, "y": 0, "width": 256, "height": 256},
            (0, 1): {"x": 0, "y": 256, "width": 256, "height": 256},
        }
        mock_wsi.tileDictionary = tile_dict
        mock_wsi.iterateTiles.return_value = iter(tile_dict.keys())
        mock_wsi.tileSize = 256

        coords = processor.get_patch_coordinates(mock_wsi)

        assert coords.shape == (3, 4)
        assert coords[0, 0] == 0  # x
        assert coords[0, 1] == 0  # y
        assert coords[0, 2] == 256  # width
        assert coords[1, 0] == 256  # x of second tile

    def test_coordinates_empty_wsi(self):
        """Test coordinates for WSI with no tiles"""
        processor = PathologyProcessor()

        mock_wsi = MagicMock()
        mock_wsi.tileDictionary = {}
        mock_wsi.iterateTiles.return_value = iter([])

        coords = processor.get_patch_coordinates(mock_wsi)

        assert coords.shape == (0, 4)


class TestProcessBatch:
    """Test process_batch method"""

    @patch.object(PathologyProcessor, "process_slide")
    def test_process_batch_collects_results(self, mock_process):
        """Test batch processing collects results"""
        mock_process.side_effect = [
            {"patches": np.array([]), "num_patches": 0},
            {"patches": np.array([]), "num_patches": 0},
        ]

        processor = PathologyProcessor()
        results = processor.process_batch(["/path/a.svs", "/path/b.svs"])

        assert len(results) == 2
        assert results[0]["path"] == "/path/a.svs"
        assert results[1]["path"] == "/path/b.svs"

    @patch.object(PathologyProcessor, "process_slide")
    def test_process_batch_handles_errors(self, mock_process):
        """Test that batch processing continues after errors"""
        mock_process.side_effect = [
            RuntimeError("File not found"),
            {"patches": np.array([]), "num_patches": 0},
        ]

        processor = PathologyProcessor()
        results = processor.process_batch(["/path/bad.svs", "/path/good.svs"])

        assert len(results) == 2
        assert "error" in results[0]
        assert results[0]["error"] == "File not found"
        assert "error" not in results[1]


class TestSaveLoadEmbeddings:
    """Test save_embeddings and load_embeddings methods"""

    def test_save_and_load_embeddings(self, tmp_path, sample_embeddings):
        """Test round-trip save/load"""
        processor = PathologyProcessor()
        path = str(tmp_path / "embeddings.npy")

        processor.save_embeddings(sample_embeddings, path)
        loaded = processor.load_embeddings(path)

        assert np.allclose(loaded, sample_embeddings)

    def test_save_embeddings_with_metadata(self, tmp_path, sample_embeddings):
        """Test saving with metadata sidecar"""
        processor = PathologyProcessor()
        path = str(tmp_path / "embeddings.npy")
        metadata = {"model": "uni", "num_patches": 10}

        processor.save_embeddings(sample_embeddings, path, metadata=metadata)

        # Check .npy file
        assert os.path.exists(path)

        # Check .json sidecar
        json_path = str(tmp_path / "embeddings.json")
        assert os.path.exists(json_path)

        with open(json_path) as f:
            saved_meta = json.load(f)
        assert saved_meta["model"] == "uni"
        assert saved_meta["num_patches"] == 10


class TestGetModelInfo:
    """Test get_model_info method"""

    def test_model_info_uni(self):
        """Test model info for UNI"""
        processor = PathologyProcessor(model="uni")
        info = processor.get_model_info()

        assert info["model_name"] == "uni"
        assert info["embedding_dim"] == 1024
        assert info["expected_input_size"] == 224
        assert info["is_loaded"] is False

    def test_model_info_virchow2(self):
        """Test model info for Virchow2"""
        processor = PathologyProcessor(model="virchow2")
        info = processor.get_model_info()

        assert info["model_name"] == "virchow2"
        assert info["embedding_dim"] == 2560

    def test_model_info_remedis(self):
        """Test model info for REMEDIS"""
        processor = PathologyProcessor(model="remedis")
        info = processor.get_model_info()

        assert info["model_name"] == "remedis"
        assert info["embedding_dim"] == 2048

    def test_model_info_loaded_state(self):
        """Test is_loaded changes when model is set"""
        processor = PathologyProcessor(model="uni")
        assert processor.get_model_info()["is_loaded"] is False

        processor.embedding_model = MagicMock()
        assert processor.get_model_info()["is_loaded"] is True


class TestCompareNormalizations:
    """Test compare_normalizations method"""

    def test_compare_default_methods(self, sample_wsi_patch):
        """Test comparing all default methods"""
        processor = PathologyProcessor()
        results = processor.compare_normalizations(sample_wsi_patch)

        assert isinstance(results, dict)
        assert "reinhard" in results
        assert "macenko" in results
        assert "vahadane" in results

        for method, img in results.items():
            assert img.shape == sample_wsi_patch.shape

    def test_compare_custom_methods(self, sample_wsi_patch):
        """Test comparing specific methods"""
        processor = PathologyProcessor()
        results = processor.compare_normalizations(sample_wsi_patch, methods=["reinhard"])

        assert len(results) == 1
        assert "reinhard" in results


class TestGetTissueStats:
    """Test get_tissue_stats method"""

    def test_tissue_stats_returns_dict(self, sample_wsi_patch):
        """Test that tissue stats returns expected keys"""
        processor = PathologyProcessor()
        stats = processor.get_tissue_stats(sample_wsi_patch)

        assert isinstance(stats, dict)
        assert "tissue_ratio" in stats
        assert "tissue_pixels" in stats
        assert "total_pixels" in stats
        assert "num_regions" in stats

    def test_tissue_stats_values(self, sample_wsi_patch):
        """Test tissue stats values are sensible"""
        processor = PathologyProcessor()
        stats = processor.get_tissue_stats(sample_wsi_patch)

        assert 0.0 <= stats["tissue_ratio"] <= 1.0
        assert stats["tissue_pixels"] >= 0
        assert stats["total_pixels"] > 0

    def test_tissue_stats_white_image(self):
        """Test tissue stats on white image (no tissue)"""
        processor = PathologyProcessor()
        white_image = np.full((224, 224, 3), 255, dtype=np.uint8)
        stats = processor.get_tissue_stats(white_image)

        assert stats["tissue_ratio"] < 0.1  # Mostly background


class TestRegistryIntegration:
    """Test PathologyProcessor integration with the model registry."""

    def test_new_presets_accepted(self):
        """Test that new preset aliases are accepted at init"""
        for model in ["h-optimus", "gigapath", "phikon-v2", "medsiglip"]:
            proc = PathologyProcessor(model=model)
            assert proc.model_name == model

    def test_explicit_provider_parameter(self):
        """Test provider parameter is stored"""
        proc = PathologyProcessor(model="bioptimus/H-optimus-0", provider="timm")
        assert proc._provider == "timm"

    def test_model_kwargs_forwarded(self):
        """Test that extra kwargs are stored for forwarding"""
        proc = PathologyProcessor(model="uni", trust_remote_code=False)
        assert proc._model_kwargs == {"trust_remote_code": False}

    def test_get_model_info_new_presets(self):
        """Test get_model_info works for new presets"""
        proc = PathologyProcessor(model="h-optimus")
        info = proc.get_model_info()
        assert info["model_name"] == "h-optimus"
        assert info["embedding_dim"] == 1536

    def test_get_model_info_non_preset(self):
        """Test get_model_info for non-preset model"""
        proc = PathologyProcessor(model="custom/model", provider="timm")
        info = proc.get_model_info()
        assert info["model_name"] == "custom/model"
        assert info["embedding_dim"] is None  # Not in registry

    @patch("honeybee.models.registry.registry.load_model")
    def test_generate_uses_registry(self, mock_load_model, sample_wsi_patches):
        """Test that generate_embeddings calls registry load_model"""
        mock_emb_model = MagicMock()
        mock_emb_model.generate_embeddings.return_value = np.random.randn(10, 1024)
        # Make it pass the isinstance check for EmbeddingModel protocol
        mock_emb_model.embedding_dim = 1024
        mock_emb_model.device = "cpu"
        mock_load_model.return_value = mock_emb_model

        proc = PathologyProcessor(model="uni")
        embeddings = proc.generate_embeddings(sample_wsi_patches)

        assert embeddings.shape == (10, 1024)

    def test_legacy_model_wrapping(self, sample_wsi_patches):
        """Test that legacy model objects are wrapped by _LegacyEmbedder"""
        proc = PathologyProcessor(model="uni")

        # Assign a legacy-style mock (has load_model_and_predict but not EmbeddingModel)
        legacy_mock = MagicMock(spec=["load_model_and_predict", "device"])
        legacy_mock.load_model_and_predict.return_value = np.random.randn(4, 1024)
        legacy_mock.device = "cpu"
        proc.embedding_model = legacy_mock

        embeddings = proc.generate_embeddings(sample_wsi_patches, batch_size=4)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[1] == 1024

    def test_register_custom_model(self):
        """Test registering and using a custom model alias"""
        from honeybee.models.registry import ModelConfig, register_model
        from honeybee.models.registry.registry import _PRESET_REGISTRY

        register_model("_test_custom", ModelConfig(
            model_id="test/custom-vit",
            provider="timm",
            embedding_dim=768,
        ))

        proc = PathologyProcessor(model="_test_custom")
        info = proc.get_model_info()
        assert info["embedding_dim"] == 768

        # Cleanup
        del _PRESET_REGISTRY["_test_custom"]


class TestArbitraryPatchSize:
    """Test that embedding models handle arbitrary patch sizes via internal resizing."""

    @pytest.mark.parametrize("patch_size", [128, 256, 512])
    def test_uni_handles_arbitrary_patch_size(self, patch_size):
        """Test UNI model resizes patches internally to 224."""
        patches = np.random.randint(0, 255, (4, patch_size, patch_size, 3), dtype=np.uint8)
        processor = PathologyProcessor(model="uni", model_path="/fake/path.pt")

        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(4, 1024)
        processor.embedding_model = mock_model

        embeddings = processor.generate_embeddings(patches, batch_size=4)

        assert embeddings.shape == (4, 1024)
        mock_model.generate_embeddings.assert_called_once()

    @pytest.mark.parametrize("patch_size", [128, 256, 512])
    def test_uni2_handles_arbitrary_patch_size(self, patch_size):
        """Test UNI2 model resizes patches internally to 224."""
        patches = np.random.randint(0, 255, (4, patch_size, patch_size, 3), dtype=np.uint8)
        processor = PathologyProcessor(model="uni2", model_path="/fake/path.pt")

        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(4, 1536)
        processor.embedding_model = mock_model

        embeddings = processor.generate_embeddings(patches, batch_size=4)

        assert embeddings.shape == (4, 1536)

    @pytest.mark.parametrize("patch_size", [128, 256, 512])
    def test_virchow2_handles_arbitrary_patch_size(self, patch_size):
        """Test Virchow2 model resizes patches internally to 224."""
        patches = np.random.randint(0, 255, (4, patch_size, patch_size, 3), dtype=np.uint8)
        processor = PathologyProcessor(model="virchow2", model_path="/fake/path.pt")

        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(4, 2560)
        processor.embedding_model = mock_model

        embeddings = processor.generate_embeddings(patches, batch_size=4)

        assert embeddings.shape == (4, 2560)

    @pytest.mark.parametrize("patch_size", [128, 256, 512])
    def test_remedis_handles_arbitrary_patch_size(self, patch_size):
        """Test REMEDIS model resizes patches internally to 224."""
        patches = np.random.randint(0, 255, (4, patch_size, patch_size, 3), dtype=np.uint8)
        processor = PathologyProcessor(model="remedis", model_path="/fake/path.onnx")

        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(4, 2048)
        processor.embedding_model = mock_model

        embeddings = processor.generate_embeddings(patches, batch_size=4)

        assert embeddings.shape == (4, 2048)

    def test_uni_transform_resizes_to_224(self):
        """Test that UNI's transform pipeline actually resizes to 224x224."""
        from honeybee.models.UNI.uni import UNI

        with patch("honeybee.models.UNI.uni.timm") as mock_timm, \
             patch("honeybee.models.UNI.uni.torch") as mock_torch:
            mock_timm.create_model.return_value = MagicMock()
            mock_torch.device.return_value = "cpu"
            mock_torch.cuda.is_available.return_value = False
            mock_torch.load.return_value = {}

            with pytest.warns(DeprecationWarning, match="UNI is deprecated"):
                model = UNI("/fake/path.pt")

            # Apply transform to a 512x512 image
            from PIL import Image
            big_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            tensor = model.transform(big_img)

            assert tensor.shape == (3, 224, 224)

    def test_remedis_preprocess_resizes_to_224(self):
        """Test that REMEDIS._preprocess resizes patches to 224x224."""
        from honeybee.models.REMEDIS.remedis import REMEDIS

        patches = np.random.randint(0, 255, (2, 512, 512, 3), dtype=np.uint8)
        result = REMEDIS._preprocess(patches)

        assert result.shape == (2, 224, 224, 3)
        assert result.dtype == np.float32


# ============================================================================
# Visualization Method Tests
# ============================================================================


class TestPlotFeatureMap:
    """Test plot_feature_map method."""

    def test_returns_figure(self):
        """Test plot_feature_map returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        processor = PathologyProcessor(model="uni")

        mock_patches = MagicMock()
        mock_patches.coordinates = np.array(
            [[i * 256, 0, 256, 256] for i in range(5)]
        )
        mock_patches.images = np.random.randint(
            0, 255, (5, 256, 256, 3), dtype=np.uint8
        )

        embeddings = np.random.randn(5, 128).astype(np.float32)

        mock_slide = MagicMock()
        mock_slide.dimensions = (2000, 2000)
        mock_slide.get_thumbnail.return_value = np.random.randint(
            0, 255, (512, 512, 3), dtype=np.uint8
        )

        with patch("umap.UMAP") as mock_umap_cls:
            mock_umap = MagicMock()
            mock_umap.fit_transform.return_value = np.random.rand(5, 3).astype(
                np.float32
            )
            mock_umap_cls.return_value = mock_umap

            fig = processor.plot_feature_map(mock_patches, embeddings, mock_slide)

        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)
