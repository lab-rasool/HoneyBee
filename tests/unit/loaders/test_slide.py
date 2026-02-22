"""
Unit tests for the revamped Slide class and WSI backend abstraction.

Tests cover:
- Backend factory (get_backend) with auto-detection and explicit selection
- Slide new API properties and methods
- Backward compatibility with legacy tile-based constructor
"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pre-populate sys.modules so that importing honeybee.loaders.Slide does NOT
# trigger honeybee.loaders.__init__.py (which pulls in langchain via Reader).
# We only need the Slide sub-package, not the full loaders namespace.
# ---------------------------------------------------------------------------
_LOADERS_KEY = "honeybee.loaders"
if _LOADERS_KEY not in sys.modules:
    import honeybee

    _stub = ModuleType(_LOADERS_KEY)
    _stub.__path__ = [str(Path(honeybee.__path__[0]) / "loaders")]  # type: ignore[attr-defined]
    _stub.__package__ = _LOADERS_KEY
    sys.modules[_LOADERS_KEY] = _stub

from honeybee.loaders.Slide._backend import (  # noqa: E402
    CuCIMBackend,
    OpenSlideBackend,
    WSIBackend,
    get_backend,
)
from honeybee.loaders.Slide.slide import Slide  # noqa: E402

# ============================================================================
# Backend Tests
# ============================================================================


class TestWSIBackend:
    """Test backend abstraction and factory."""

    def test_get_backend_cucim(self):
        """Test explicit cucim backend selection."""
        backend = get_backend("cucim")
        assert isinstance(backend, CuCIMBackend)
        assert backend.name == "cucim"

    def test_get_backend_openslide(self):
        """Test explicit openslide backend selection."""
        backend = get_backend("openslide")
        assert isinstance(backend, OpenSlideBackend)
        assert backend.name == "openslide"

    def test_get_backend_auto_detect_cucim_first(self):
        """Test auto-detection prefers cucim when available."""
        # Mock both cucim and openslide as importable
        cucim_mock = MagicMock()
        with patch.dict(sys.modules, {"cucim": cucim_mock}):
            backend = get_backend(None)
            assert isinstance(backend, CuCIMBackend)

    def test_get_backend_auto_detect_fallback_openslide(self):
        """Test fallback to openslide when cucim is unavailable."""
        openslide_mock = MagicMock()
        # Remove cucim from modules if present, add openslide
        with patch.dict(sys.modules, {"cucim": None, "openslide": openslide_mock}):
            # cucim import will raise ImportError when module value is None
            # We need a more explicit approach
            pass

        # Use importlib side-effect approach instead
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None

        def mock_import(name, *args, **kwargs):
            if name == "cucim":
                raise ImportError("No cucim")
            if name == "openslide":
                return MagicMock()
            if original_import:
                return original_import(name, *args, **kwargs)
            import builtins

            return builtins.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            backend = get_backend(None)
            assert isinstance(backend, OpenSlideBackend)

    def test_get_backend_neither_available(self):
        """Test error when no backend available."""

        def mock_import(name, *args, **kwargs):
            if name in ("cucim", "openslide"):
                raise ImportError(f"No {name}")
            import builtins

            return builtins.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="No WSI backend found"):
                get_backend(None)

    def test_get_backend_invalid_name(self):
        """Test error for invalid backend name."""
        with pytest.raises(ValueError, match="Unknown WSI backend"):
            get_backend("invalid_backend")

    def test_get_backend_case_insensitive(self):
        """Test that backend name is case-insensitive."""
        backend = get_backend("CuCIM")
        assert isinstance(backend, CuCIMBackend)

        backend = get_backend("OPENSLIDE")
        assert isinstance(backend, OpenSlideBackend)

    def test_get_backend_strips_whitespace(self):
        """Test that backend name is stripped of whitespace."""
        backend = get_backend("  cucim  ")
        assert isinstance(backend, CuCIMBackend)


# ============================================================================
# Helper: build a fully-mocked Slide
# ============================================================================


def _make_mock_backend():
    """Create a mock WSIBackend with realistic return values."""
    backend = MagicMock(spec=WSIBackend)
    backend.name = "mock"
    backend.open.return_value = MagicMock(name="mock_handle")
    backend.get_level_count.return_value = 3
    backend.get_level_dimensions.return_value = [
        (10000, 10000),
        (2500, 2500),
        (625, 625),
    ]
    backend.get_level_downsamples.return_value = [1.0, 4.0, 16.0]
    backend.get_properties.return_value = {
        "magnification": 40.0,
        "mpp_x": 0.25,
        "mpp_y": 0.25,
        "vendor": "Aperio",
        "raw_metadata": {},
    }
    backend.get_thumbnail.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    backend.read_region.return_value = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return backend


# ============================================================================
# Slide New API Tests
# ============================================================================


class TestSlideNewAPI:
    """Test new Slide class API."""

    @pytest.fixture
    def mock_slide(self):
        """Create a Slide with mocked backend."""
        mock_backend = _make_mock_backend()
        with patch("honeybee.loaders.Slide._backend.get_backend", return_value=mock_backend):
            slide = Slide("/fake/slide.svs")
        # Expose backend for assertions
        slide._mock_backend = mock_backend
        return slide

    def test_properties(self, mock_slide):
        """Test level_count, dimensions, magnification, mpp."""
        assert mock_slide.level_count == 3
        assert mock_slide.dimensions == (10000, 10000)
        assert mock_slide.magnification == 40.0
        assert mock_slide.mpp == 0.25  # average of 0.25 and 0.25

    def test_level_dimensions(self, mock_slide):
        """Test level_dimensions property."""
        dims = mock_slide.level_dimensions
        assert len(dims) == 3
        assert dims[0] == (10000, 10000)
        assert dims[1] == (2500, 2500)
        assert dims[2] == (625, 625)

    def test_level_downsamples(self, mock_slide):
        """Test level_downsamples property."""
        ds = mock_slide.level_downsamples
        assert len(ds) == 3
        assert ds[0] == 1.0
        assert ds[1] == 4.0
        assert ds[2] == 16.0

    def test_read_region_default_level(self, mock_slide):
        """Test read_region with default level=0."""
        region = mock_slide.read_region((0, 0), (512, 512))
        assert region.shape == (256, 256, 3)  # from mocked return
        assert region.dtype == np.uint8
        mock_slide._mock_backend.read_region.assert_called_with(
            mock_slide._handle, location=(0, 0), level=0, size=(512, 512)
        )

    def test_read_region_explicit_level(self, mock_slide):
        """Test read_region with explicit level."""
        mock_slide.read_region((100, 200), (256, 256), level=1)
        mock_slide._mock_backend.read_region.assert_called_with(
            mock_slide._handle, location=(100, 200), level=1, size=(256, 256)
        )

    def test_read_region_with_magnification(self, mock_slide):
        """Test read_region with magnification parameter."""
        # magnification=20.0 -> downsample=40/20=2.0 -> best level for ds=2.0 is level 0
        # (since ds[0]=1.0 <= 2.0, ds[1]=4.0 > 2.0, so best_level=0)
        mock_slide.read_region((0, 0), (256, 256), magnification=20.0)
        mock_slide._mock_backend.read_region.assert_called_with(
            mock_slide._handle, location=(0, 0), level=0, size=(256, 256)
        )

    def test_read_region_with_magnification_10(self, mock_slide):
        """Test read_region with magnification 10 picks level 1."""
        # magnification=10.0 -> downsample=40/10=4.0 -> best level for ds=4.0 is level 1
        mock_slide.read_region((0, 0), (256, 256), magnification=10.0)
        mock_slide._mock_backend.read_region.assert_called_with(
            mock_slide._handle, location=(0, 0), level=1, size=(256, 256)
        )

    def test_get_thumbnail_cached(self, mock_slide):
        """Test thumbnail is cached after first call."""
        thumb1 = mock_slide.get_thumbnail((512, 512))
        thumb2 = mock_slide.get_thumbnail((512, 512))
        np.testing.assert_array_equal(thumb1, thumb2)
        # Backend should be called only once for same size
        assert mock_slide._mock_backend.get_thumbnail.call_count == 1

    def test_get_thumbnail_different_sizes_not_shared(self, mock_slide):
        """Test different thumbnail sizes are cached independently."""
        mock_slide._mock_backend.get_thumbnail.side_effect = [
            np.zeros((512, 512, 3), dtype=np.uint8),
            np.ones((256, 256, 3), dtype=np.uint8),
        ]
        _ = mock_slide.get_thumbnail((512, 512))
        _ = mock_slide.get_thumbnail((256, 256))
        assert mock_slide._mock_backend.get_thumbnail.call_count == 2

    def test_get_best_level_for_magnification(self, mock_slide):
        """Test magnification to level mapping."""
        # 40x -> ds=1.0 -> level 0
        assert mock_slide.get_best_level_for_magnification(40.0) == 0
        # 10x -> ds=4.0 -> level 1
        assert mock_slide.get_best_level_for_magnification(10.0) == 1
        # 2.5x -> ds=16.0 -> level 2
        assert mock_slide.get_best_level_for_magnification(2.5) == 2

    def test_get_best_level_for_magnification_no_native(self, mock_slide):
        """Test that unknown magnification returns level 0."""
        mock_slide._mock_backend.get_properties.return_value = {
            "magnification": None,
            "mpp_x": None,
            "mpp_y": None,
            "vendor": None,
        }
        assert mock_slide.get_best_level_for_magnification(20.0) == 0

    def test_get_best_level_for_downsample(self, mock_slide):
        """Test downsample to level mapping."""
        assert mock_slide.get_best_level_for_downsample(1.0) == 0
        assert mock_slide.get_best_level_for_downsample(3.0) == 0
        assert mock_slide.get_best_level_for_downsample(4.0) == 1
        assert mock_slide.get_best_level_for_downsample(5.0) == 1
        assert mock_slide.get_best_level_for_downsample(16.0) == 2
        assert mock_slide.get_best_level_for_downsample(100.0) == 2

    def test_detect_tissue(self, mock_slide):
        """Test tissue detection stores mask."""
        fake_mask = np.ones((512, 512), dtype=bool)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = fake_mask

        with patch(
            "honeybee.processors.wsi.tissue_detection.ClassicalTissueDetector",
            return_value=mock_detector,
        ):
            result = mock_slide.detect_tissue(method="otsu")

        mock_detector.detect.assert_called_once()
        np.testing.assert_array_equal(result, fake_mask)
        np.testing.assert_array_equal(mock_slide.tissue_mask, fake_mask)

    def test_info_property(self, mock_slide):
        """Test info dict contains expected keys."""
        info = mock_slide.info
        expected_keys = {
            "path",
            "backend",
            "dimensions",
            "level_count",
            "level_dimensions",
            "level_downsamples",
            "magnification",
            "mpp",
            "vendor",
        }
        assert set(info.keys()) == expected_keys
        assert info["backend"] == "mock"
        assert info["dimensions"] == (10000, 10000)
        assert info["level_count"] == 3
        assert info["magnification"] == 40.0
        assert info["vendor"] == "Aperio"

    def test_tissue_mask_none_before_detection(self, mock_slide):
        """Test tissue_mask is None before detect_tissue()."""
        assert mock_slide.tissue_mask is None

    def test_mpp_single_axis(self, mock_slide):
        """Test mpp when only one axis is available."""
        mock_slide._mock_backend.get_properties.return_value = {
            "magnification": 40.0,
            "mpp_x": 0.5,
            "mpp_y": None,
            "vendor": "Aperio",
        }
        assert mock_slide.mpp == 0.5

    def test_mpp_none(self, mock_slide):
        """Test mpp returns None when no MPP data available."""
        mock_slide._mock_backend.get_properties.return_value = {
            "magnification": 40.0,
            "mpp_x": None,
            "mpp_y": None,
            "vendor": "Aperio",
        }
        assert mock_slide.mpp is None

    def test_repr(self, mock_slide):
        """Test string representation."""
        r = repr(mock_slide)
        assert "Slide" in r
        assert "slide.svs" in r
        assert "mock" in r

    def test_path_attribute(self, mock_slide):
        """Test path is set correctly."""
        assert mock_slide.path == Path("/fake/slide.svs")


# ============================================================================
# Backward Compatibility Tests
# ============================================================================


class TestSlideBackwardCompat:
    """Test backward compatibility with old Slide constructor."""

    @pytest.fixture
    def mock_backend_for_legacy(self):
        """Create a mock backend for legacy mode tests."""
        backend = _make_mock_backend()
        # For legacy mode, read_region is called to cache the full level
        backend.read_region.return_value = np.random.randint(0, 255, (625, 625, 3), dtype=np.uint8)
        return backend

    def test_old_constructor_signature(self, mock_backend_for_legacy):
        """Test that slide_image_path= kwarg works."""
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend_for_legacy,
        ):
            slide = Slide(slide_image_path="/fake/old_slide.svs", tile_size=256)
        assert slide.path == Path("/fake/old_slide.svs")
        assert slide.slide_image_path == "/fake/old_slide.svs"

    def test_legacy_attributes(self, mock_backend_for_legacy):
        """Test slide_image_path, slideFileName, tileSize etc."""
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend_for_legacy,
        ):
            slide = Slide(
                slide_image_path="/fake/sample_slide.svs",
                tile_size=512,
                max_patches=500,
            )
        assert slide.slideFileName == "sample_slide"
        assert slide.slideFilePath == Path("/fake/sample_slide.svs")
        assert slide.tileSize == 512

    def test_legacy_tile_dictionary(self, mock_backend_for_legacy):
        """Test tileDictionary is populated."""
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend_for_legacy,
        ):
            slide = Slide(
                slide_image_path="/fake/slide.svs",
                tile_size=256,
                max_patches=500,
            )
        # tileDictionary should be populated (level is selected based on max_patches)
        assert isinstance(slide.tileDictionary, dict)
        # With level 2 (625x625), tile_size=256 -> 2 tiles per axis = 4 total
        assert slide.numTilesInX > 0
        assert slide.numTilesInY > 0
        assert len(slide.tileDictionary) == slide.numTilesInX * slide.numTilesInY

    def test_legacy_get_tile(self, mock_backend_for_legacy):
        """Test getTile method works."""
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend_for_legacy,
        ):
            slide = Slide(
                slide_image_path="/fake/slide.svs",
                tile_size=256,
                max_patches=500,
            )
        # Get the first tile address
        addresses = list(slide.tileDictionary.keys())
        if len(addresses) > 0:
            tile = slide.getTile(addresses[0], writeToNumpy=True)
            assert tile is not None
            assert tile.ndim == 3
            assert tile.shape[2] == 3

    def test_legacy_get_tile_invalid_address(self, mock_backend_for_legacy):
        """Test getTile with invalid address returns None."""
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend_for_legacy,
        ):
            slide = Slide(
                slide_image_path="/fake/slide.svs",
                tile_size=256,
                max_patches=500,
            )
        result = slide.getTile((-999, -999), writeToNumpy=True)
        assert result is None

    def test_legacy_iterate_tiles(self, mock_backend_for_legacy):
        """Test iterateTiles method works."""
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend_for_legacy,
        ):
            slide = Slide(
                slide_image_path="/fake/slide.svs",
                tile_size=256,
                max_patches=500,
            )
        tiles = list(slide.iterateTiles())
        assert len(tiles) == len(slide.tileDictionary)
        # Each yielded item is a tuple of (x, y)
        for addr in tiles:
            assert isinstance(addr, tuple)
            assert len(addr) == 2

    def test_legacy_suitable_tile_addresses(self, mock_backend_for_legacy):
        """Test suitableTileAddresses method."""
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend_for_legacy,
        ):
            slide = Slide(
                slide_image_path="/fake/slide.svs",
                tile_size=256,
                max_patches=500,
            )
        addresses = slide.suitableTileAddresses()
        assert isinstance(addresses, list)
        assert len(addresses) == len(slide.tileDictionary)

    def test_legacy_img_proxy(self, mock_backend_for_legacy):
        """Test the _LegacyImgProxy provides expected attributes."""
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend_for_legacy,
        ):
            slide = Slide(
                slide_image_path="/fake/slide.svs",
                tile_size=256,
                max_patches=500,
            )
        # img.resolutions should work
        res = slide.img.resolutions
        assert "level_count" in res
        assert "level_dimensions" in res
        assert "level_downsamples" in res
        assert res["level_count"] == 3

        # img.metadata should work
        meta = slide.img.metadata
        assert "cucim" in meta
        assert "magnification" in meta

    def test_no_path_raises(self):
        """Test that omitting path raises ValueError."""
        mock_backend = _make_mock_backend()
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend,
        ):
            with pytest.raises(ValueError, match="slide path is required"):
                Slide()

    def test_new_api_defaults_for_legacy_attrs(self):
        """Test that new-style Slide still has legacy attrs with defaults."""
        mock_backend = _make_mock_backend()
        with patch(
            "honeybee.loaders.Slide._backend.get_backend",
            return_value=mock_backend,
        ):
            slide = Slide("/fake/slide.svs")
        # These should be set even in new-mode
        assert hasattr(slide, "tileSize")
        assert hasattr(slide, "tileDictionary")
        assert hasattr(slide, "numTilesInX")
        assert hasattr(slide, "numTilesInY")
        assert slide.tileDictionary == {}
        assert slide.numTilesInX == 0
        assert slide.numTilesInY == 0


# ============================================================================
# DL Tissue Detection Integration Tests
# ============================================================================


class TestSlideDLTissueDetection:
    """Test detect_tissue() with DL detector integration."""

    @pytest.fixture
    def mock_slide(self):
        """Create a Slide with mocked backend."""
        mock_backend = _make_mock_backend()
        with patch("honeybee.loaders.Slide._backend.get_backend", return_value=mock_backend):
            slide = Slide("/fake/slide.svs")
        slide._mock_backend = mock_backend
        return slide

    def test_detect_tissue_method_dl(self, mock_slide):
        """Test detect_tissue(method='dl') uses TissueDetector."""
        fake_mask = np.ones((512, 512), dtype=bool)
        fake_pred_map = np.random.rand(8, 10, 3).astype(np.float32)

        mock_detector_instance = MagicMock()
        mock_detector_instance.detect.return_value = (fake_mask, fake_pred_map)

        with patch(
            "honeybee.models.TissueDetector.tissue_detector.TissueDetector",
            return_value=mock_detector_instance,
        ):
            result = mock_slide.detect_tissue(method="dl", device="cpu")

        np.testing.assert_array_equal(result, fake_mask)
        np.testing.assert_array_equal(mock_slide.tissue_mask, fake_mask)
        np.testing.assert_array_equal(mock_slide.prediction_map, fake_pred_map)

    def test_detect_tissue_with_detector_instance(self, mock_slide):
        """Test detect_tissue(detector=instance) uses provided detector."""
        fake_mask = np.zeros((512, 512), dtype=bool)
        fake_pred_map = np.zeros((4, 5, 3), dtype=np.float32)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = (fake_mask, fake_pred_map)

        result = mock_slide.detect_tissue(detector=mock_detector)

        mock_detector.detect.assert_called_once()
        np.testing.assert_array_equal(result, fake_mask)
        np.testing.assert_array_equal(mock_slide.tissue_mask, fake_mask)
        np.testing.assert_array_equal(mock_slide.prediction_map, fake_pred_map)

    def test_prediction_map_none_before_dl_detection(self, mock_slide):
        """Test prediction_map is None before DL detection."""
        assert mock_slide.prediction_map is None

    def test_prediction_map_none_after_classical_detection(self, mock_slide):
        """Test prediction_map stays None after classical detection."""
        fake_mask = np.ones((512, 512), dtype=bool)
        mock_classical = MagicMock()
        mock_classical.detect.return_value = fake_mask

        with patch(
            "honeybee.processors.wsi.tissue_detection.ClassicalTissueDetector",
            return_value=mock_classical,
        ):
            mock_slide.detect_tissue(method="otsu")

        assert mock_slide.prediction_map is None

    def test_detector_kwarg_overrides_method(self, mock_slide):
        """Test that detector= takes precedence over method."""
        fake_mask = np.ones((512, 512), dtype=bool)
        fake_pred_map = np.ones((4, 5, 3), dtype=np.float32)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = (fake_mask, fake_pred_map)

        # Even though method="otsu", the detector kwarg should take priority
        result = mock_slide.detect_tissue(method="otsu", detector=mock_detector)
        mock_detector.detect.assert_called_once()
        np.testing.assert_array_equal(result, fake_mask)

    def test_dl_kwargs_forwarded(self, mock_slide):
        """Test that level/threshold/num_workers are forwarded to detect()."""
        fake_mask = np.ones((512, 512), dtype=bool)
        fake_pred_map = np.zeros((4, 5, 3), dtype=np.float32)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = (fake_mask, fake_pred_map)

        mock_slide.detect_tissue(
            detector=mock_detector,
            level=1,
            threshold=0.7,
            num_workers=8,
        )

        call_kwargs = mock_detector.detect.call_args.kwargs
        assert call_kwargs["level"] == 1
        assert call_kwargs["threshold"] == 0.7
        assert call_kwargs["num_workers"] == 8


# ============================================================================
# Visualization Method Tests
# ============================================================================


class TestSlideVisualization:
    """Test new visualization methods on Slide."""

    @pytest.fixture
    def mock_slide(self):
        """Create a Slide with mocked backend."""
        mock_backend = _make_mock_backend()
        with patch("honeybee.loaders.Slide._backend.get_backend", return_value=mock_backend):
            slide = Slide("/fake/slide.svs")
        slide._mock_backend = mock_backend
        return slide

    def test_plot_tissue_detection_no_mask_raises(self, mock_slide):
        """Test ValueError when no mask available."""
        with pytest.raises(ValueError, match="No tissue mask"):
            mock_slide.plot_tissue_detection()

    def test_plot_tissue_detection_classical_2panel(self, mock_slide):
        """Test 2-panel layout with classical mask."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mock_slide._tissue_mask = np.ones((512, 512), dtype=bool)
        fig = mock_slide.plot_tissue_detection()
        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_tissue_detection_dl_4panel(self, mock_slide):
        """Test 4-panel layout with DL prediction map."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mock_slide._tissue_mask = np.ones((512, 512), dtype=bool)
        mock_slide._prediction_map = np.random.rand(7, 7, 3).astype(np.float32)

        with patch(
            "honeybee.models.TissueDetector.tissue_detector.TissueDetector"
        ) as mock_td_cls:
            mock_td_cls.prediction_map_to_rgb.return_value = np.random.randint(
                0, 255, (7, 7, 3), dtype=np.uint8
            )
            fig = mock_slide.plot_tissue_detection()

        assert fig is not None
        # 4 panels + colorbar = 5 axes
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_compare_tissue_methods_returns_figure(self, mock_slide):
        """Test compare_tissue_methods returns a figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fake_mask = np.ones((512, 512), dtype=bool)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = fake_mask

        with patch(
            "honeybee.processors.wsi.tissue_detection.ClassicalTissueDetector",
            return_value=mock_detector,
        ):
            fig = mock_slide.compare_tissue_methods(methods=["otsu", "hsv"])

        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_compare_tissue_methods_restores_state(self, mock_slide):
        """Test that original mask is restored after comparison."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        original_mask = np.ones((512, 512), dtype=bool)
        mock_slide._tissue_mask = original_mask.copy()
        mock_slide._prediction_map = np.random.rand(4, 4, 3).astype(np.float32)

        different_mask = np.zeros((512, 512), dtype=bool)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = different_mask

        with patch(
            "honeybee.processors.wsi.tissue_detection.ClassicalTissueDetector",
            return_value=mock_detector,
        ):
            fig = mock_slide.compare_tissue_methods(methods=["otsu"])

        # Original mask should be restored
        np.testing.assert_array_equal(mock_slide._tissue_mask, original_mask)
        assert mock_slide._prediction_map is not None
        plt.close(fig)
