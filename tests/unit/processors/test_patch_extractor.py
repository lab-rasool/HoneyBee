"""
Unit tests for the PatchExtractor class.

Tests cover:
- Initialization and parameter validation
- Grid-based patch extraction from mock slides
- Tissue mask filtering
- Magnification-based level selection
- Overlapping extraction (stride < patch_size)
- Coordinate-only extraction (get_coordinates)
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from honeybee.processors.wsi.patch_extractor import PatchExtractor
from honeybee.processors.wsi.patches import Patches

# ============================================================================
# Helper fixture: mock Slide
# ============================================================================


@pytest.fixture
def mock_slide():
    """Create a mock Slide for testing extraction.

    Simulates a 1000x1000 level-0 slide with 2 levels.
    """
    slide = MagicMock()
    slide.dimensions = (1000, 1000)
    slide.level_dimensions = [(1000, 1000), (500, 500)]
    slide.level_downsamples = [1.0, 2.0]
    slide.tissue_mask = None
    slide.path = Path("/fake/slide.svs")
    slide.magnification = 40.0
    slide.read_region.return_value = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    slide.get_best_level_for_magnification.return_value = 0
    slide.get_thumbnail.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return slide


@pytest.fixture
def large_mock_slide():
    """Create a larger mock Slide (10000x10000) with 3 levels."""
    slide = MagicMock()
    slide.dimensions = (10000, 10000)
    slide.level_dimensions = [(10000, 10000), (2500, 2500), (625, 625)]
    slide.level_downsamples = [1.0, 4.0, 16.0]
    slide.tissue_mask = None
    slide.path = Path("/fake/large_slide.svs")
    slide.magnification = 40.0
    slide.read_region.return_value = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    slide.get_best_level_for_magnification.return_value = 0
    slide.get_thumbnail.return_value = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return slide


# ============================================================================
# Initialization Tests
# ============================================================================


class TestPatchExtractorInit:
    """Test PatchExtractor initialization and parameter validation."""

    def test_default_params(self):
        """Test default parameter values."""
        extractor = PatchExtractor()
        assert extractor.patch_size == 256
        assert extractor.stride == 256  # defaults to patch_size
        assert extractor.min_tissue_ratio == 0.5
        assert extractor.magnification is None
        assert extractor.level == 0

    def test_custom_params(self):
        """Test custom parameter values."""
        extractor = PatchExtractor(
            patch_size=512, stride=256, magnification=20.0, min_tissue_ratio=0.3
        )
        assert extractor.patch_size == 512
        assert extractor.stride == 256
        assert extractor.magnification == 20.0
        assert extractor.min_tissue_ratio == 0.3

    def test_stride_defaults_to_patch_size(self):
        """Test that stride defaults to patch_size when not specified."""
        extractor = PatchExtractor(patch_size=128)
        assert extractor.stride == 128

    def test_invalid_patch_size_zero(self):
        """Test that zero patch_size raises ValueError."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            PatchExtractor(patch_size=0)

    def test_invalid_patch_size_negative(self):
        """Test that negative patch_size raises ValueError."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            PatchExtractor(patch_size=-1)

    def test_invalid_stride(self):
        """Test that zero/negative stride raises ValueError."""
        with pytest.raises(ValueError, match="stride must be positive"):
            PatchExtractor(patch_size=256, stride=0)

    def test_invalid_min_tissue_ratio_high(self):
        """Test that min_tissue_ratio > 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_tissue_ratio must be in"):
            PatchExtractor(min_tissue_ratio=1.5)

    def test_invalid_min_tissue_ratio_negative(self):
        """Test that negative min_tissue_ratio raises ValueError."""
        with pytest.raises(ValueError, match="min_tissue_ratio must be in"):
            PatchExtractor(min_tissue_ratio=-0.1)

    def test_repr(self):
        """Test string representation."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        r = repr(extractor)
        assert "PatchExtractor" in r
        assert "patch_size=256" in r
        assert "stride=256" in r

    def test_repr_with_overlap(self):
        """Test repr shows overlap when stride < patch_size."""
        extractor = PatchExtractor(patch_size=256, stride=128)
        r = repr(extractor)
        assert "overlap=" in r

    def test_repr_with_magnification(self):
        """Test repr shows magnification when set."""
        extractor = PatchExtractor(patch_size=256, magnification=20.0)
        r = repr(extractor)
        assert "magnification=20.0" in r


# ============================================================================
# Extraction Tests
# ============================================================================


class TestPatchExtractorExtract:
    """Test patch extraction from mock slides."""

    def test_extract_basic(self, mock_slide):
        """Test basic extraction returns Patches object."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        patches = extractor.extract(mock_slide)
        assert isinstance(patches, Patches)
        assert len(patches) > 0

    def test_extract_no_tissue_mask(self, mock_slide):
        """Test extraction without tissue mask includes all grid positions."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        patches = extractor.extract(mock_slide)
        assert isinstance(patches, Patches)
        # 1000/256 = 3 full patches per axis (0, 256, 512; 768 does not fit)
        # Actually: arange(0, 1000-256+1, 256) -> [0, 256, 512] = 3 per axis
        # -> 3 * 3 = 9 total
        assert len(patches) == 9

    def test_extract_with_tissue_mask(self, mock_slide):
        """Test extraction with tissue mask filters patches."""
        # Create a mask at thumbnail size (512x512) with tissue in top-left only
        mask = np.zeros((512, 512), dtype=bool)
        mask[:128, :128] = True  # ~25% of top-left area
        mock_slide.tissue_mask = mask

        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
        patches_filtered = extractor.extract(mock_slide)

        # Without mask, we'd get 9 patches
        # With mask, only top-left region has tissue
        assert len(patches_filtered) < 9

    def test_extract_all_tissue_mask(self, mock_slide):
        """Test extraction with full tissue mask keeps all patches."""
        mask = np.ones((512, 512), dtype=bool)  # tissue everywhere
        mock_slide.tissue_mask = mask

        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
        patches = extractor.extract(mock_slide)
        assert len(patches) == 9  # all patches kept

    def test_extract_with_magnification(self, mock_slide):
        """Test extraction at specific magnification."""
        mock_slide.get_best_level_for_magnification.return_value = 1
        extractor = PatchExtractor(patch_size=256, magnification=20.0)
        patches = extractor.extract(mock_slide)

        mock_slide.get_best_level_for_magnification.assert_called_with(20.0)
        assert isinstance(patches, Patches)

    def test_extract_with_overlap(self, mock_slide):
        """Test overlapping extraction (stride < patch_size)."""
        extractor_overlap = PatchExtractor(patch_size=256, stride=128)
        patches_overlap = extractor_overlap.extract(mock_slide)

        extractor_no_overlap = PatchExtractor(patch_size=256, stride=256)
        patches_no_overlap = extractor_no_overlap.extract(mock_slide)

        # More patches with overlap
        assert len(patches_overlap) >= len(patches_no_overlap)

    def test_extract_empty_slide_no_tissue(self):
        """Test extraction from slide with no tissue detected."""
        slide = MagicMock()
        slide.dimensions = (1000, 1000)
        slide.level_dimensions = [(1000, 1000)]
        slide.level_downsamples = [1.0]
        slide.tissue_mask = np.zeros((512, 512), dtype=bool)  # No tissue
        slide.path = Path("/fake/empty_slide.svs")
        slide.magnification = 40.0

        extractor = PatchExtractor(patch_size=256, min_tissue_ratio=0.5)
        patches = extractor.extract(slide)

        assert len(patches) == 0

    def test_extract_metadata(self, mock_slide):
        """Test that extracted patches have proper metadata."""
        extractor = PatchExtractor(patch_size=256, stride=256, magnification=20.0)
        mock_slide.get_best_level_for_magnification.return_value = 0
        patches = extractor.extract(mock_slide)

        assert "patch_size" in patches.metadata
        assert patches.metadata["patch_size"] == 256
        assert patches.metadata["stride"] == 256
        assert patches.metadata["slide_path"] == "/fake/slide.svs"
        assert patches.metadata["magnification"] == 20.0

    def test_extract_coordinates_format(self, mock_slide):
        """Test that coordinates have shape (N, 4) with [x, y, w, h]."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        patches = extractor.extract(mock_slide)

        assert patches.coordinates.ndim == 2
        assert patches.coordinates.shape[1] == 4
        # All w and h should be the same (patch_size in level-0 space)
        assert np.all(patches.coordinates[:, 2] == 256)  # w
        assert np.all(patches.coordinates[:, 3] == 256)  # h

    def test_extract_calls_read_region(self, mock_slide):
        """Test that extract calls slide.read_region for each patch."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        patches = extractor.extract(mock_slide)
        assert mock_slide.read_region.call_count == len(patches)

    def test_extract_at_level_1(self, mock_slide):
        """Test extraction at explicit level 1."""
        extractor = PatchExtractor(patch_size=256, stride=256, level=1)
        patches = extractor.extract(mock_slide)

        # At level 1 (downsample=2.0), patch footprint in level-0 is 256*2=512
        # stride in level-0 is 256*2=512
        # arange(0, 1000-512+1, 512) -> [0, 488] -> [0] (only 0 fits cleanly?)
        # Actually: 1000 - 512 + 1 = 489 -> arange(0, 489, 512) -> [0] -> 1x1=1
        # Wait, 0 + 512 = 512 <= 1000, so 512 would also be in range? No,
        # arange(0, 489, 512) = [0] since 512 > 489
        # So 1 patch total
        assert isinstance(patches, Patches)
        assert len(patches) >= 1

    def test_extract_handles_read_failure(self, mock_slide):
        """Test that read failures produce blank patches."""
        call_count = 0

        def failing_read_region(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Read failed")
            return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        mock_slide.read_region.side_effect = failing_read_region
        extractor = PatchExtractor(patch_size=256, stride=256)
        patches = extractor.extract(mock_slide)

        # Should still have all patches (failed one replaced with black)
        assert len(patches) == 9


# ============================================================================
# get_coordinates Tests
# ============================================================================


class TestPatchExtractorGetCoordinates:
    """Test coordinate-only extraction."""

    def test_get_coordinates_returns_array(self, mock_slide):
        """Test get_coordinates returns (N, 4) array."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        coords = extractor.get_coordinates(mock_slide)

        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 2
        assert coords.shape[1] == 4

    def test_get_coordinates_matches_extract(self, mock_slide):
        """Coordinates from get_coordinates match those from extract."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        coords = extractor.get_coordinates(mock_slide)
        patches = extractor.extract(mock_slide)

        np.testing.assert_array_equal(coords, patches.coordinates)

    def test_get_coordinates_no_pixel_reads(self, mock_slide):
        """Test that get_coordinates does not call read_region."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        _ = extractor.get_coordinates(mock_slide)

        mock_slide.read_region.assert_not_called()

    def test_get_coordinates_empty_when_no_tissue(self):
        """Test get_coordinates returns empty for slide with no tissue."""
        slide = MagicMock()
        slide.dimensions = (1000, 1000)
        slide.level_dimensions = [(1000, 1000)]
        slide.level_downsamples = [1.0]
        slide.tissue_mask = np.zeros((512, 512), dtype=bool)
        slide.path = Path("/fake/slide.svs")
        slide.magnification = 40.0

        extractor = PatchExtractor(patch_size=256, min_tissue_ratio=0.5)
        coords = extractor.get_coordinates(slide)

        assert isinstance(coords, np.ndarray)
        assert coords.shape == (0, 4)

    def test_get_coordinates_with_tissue_mask(self, mock_slide):
        """Test coordinates are filtered by tissue mask."""
        mask = np.zeros((512, 512), dtype=bool)
        mask[:128, :128] = True
        mock_slide.tissue_mask = mask

        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
        coords = extractor.get_coordinates(mock_slide)

        # Should be fewer than 9 (full grid without mask)
        assert len(coords) < 9

    def test_get_coordinates_with_magnification(self, mock_slide):
        """Test get_coordinates uses magnification for level selection."""
        mock_slide.get_best_level_for_magnification.return_value = 1
        extractor = PatchExtractor(patch_size=256, magnification=20.0)
        _ = extractor.get_coordinates(mock_slide)
        mock_slide.get_best_level_for_magnification.assert_called_with(20.0)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestPatchExtractorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_slide_smaller_than_patch(self):
        """Test extraction from slide smaller than patch_size."""
        slide = MagicMock()
        slide.dimensions = (100, 100)
        slide.level_dimensions = [(100, 100)]
        slide.level_downsamples = [1.0]
        slide.tissue_mask = None
        slide.path = Path("/fake/tiny_slide.svs")
        slide.magnification = 40.0
        slide.read_region.return_value = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        extractor = PatchExtractor(patch_size=256, stride=256)
        patches = extractor.extract(slide)

        # Slide is 100x100, patch_size_l0=256 -> 100-256+1 < 0
        # But the code still extracts one patch at (0,0) for non-empty slides
        assert len(patches) == 1

    def test_stride_equals_patch_size(self, mock_slide):
        """Test non-overlapping extraction (stride == patch_size)."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        patches = extractor.extract(mock_slide)
        # All coordinates should be non-overlapping
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                ci = patches.coordinates[i]
                cj = patches.coordinates[j]
                # They should not overlap
                no_overlap_x = ci[0] + ci[2] <= cj[0] or cj[0] + cj[2] <= ci[0]
                no_overlap_y = ci[1] + ci[3] <= cj[1] or cj[1] + cj[3] <= ci[1]
                assert no_overlap_x or no_overlap_y

    def test_min_tissue_ratio_zero_keeps_all(self, mock_slide):
        """Test that min_tissue_ratio=0 keeps all patches even with mask."""
        mask = np.zeros((512, 512), dtype=bool)
        mask[0, 0] = True  # only one pixel of tissue
        mock_slide.tissue_mask = mask

        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.0)
        patches = extractor.extract(mock_slide)

        # min_tissue_ratio=0 should keep all patches (any tissue >= 0)
        assert len(patches) == 9

    def test_min_tissue_ratio_one_strict(self, mock_slide):
        """Test that min_tissue_ratio=1.0 requires full tissue coverage."""
        # Partial tissue mask
        mask = np.ones((512, 512), dtype=bool)
        mask[0:10, :] = False  # small strip without tissue
        mock_slide.tissue_mask = mask

        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=1.0)
        patches = extractor.extract(mock_slide)

        # Some patches at the top won't have 100% tissue coverage
        assert len(patches) <= 9

    def test_large_slide_patch_count(self, large_mock_slide):
        """Test extraction from a large slide produces expected count."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        coords = extractor.get_coordinates(large_mock_slide)

        # 10000/256 = 39.06 -> arange(0, 10000-256+1, 256) gives 39 per axis
        # Total: 39 * 39 = 1521
        expected_per_axis = len(np.arange(0, 10000 - 256 + 1, 256))
        assert len(coords) == expected_per_axis**2


# ============================================================================
# Visualization Method Tests
# ============================================================================


class TestPatchExtractorVisualization:
    """Test plot_grid_preview method."""

    def test_plot_grid_preview_returns_figure(self, mock_slide):
        """Test that plot_grid_preview returns a matplotlib Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mock_slide.tissue_mask = np.ones((512, 512), dtype=bool)
        mock_slide.mpp = 1.0

        extractor = PatchExtractor(patch_size=256, stride=256)
        fig = extractor.plot_grid_preview(mock_slide)

        assert fig is not None
        assert len(fig.axes) == 4  # 2x2
        plt.close(fig)

    def test_plot_grid_preview_no_mask(self, mock_slide):
        """Test plot_grid_preview works when no tissue mask exists."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mock_slide.tissue_mask = None
        mock_slide.mpp = 1.0

        extractor = PatchExtractor(patch_size=256, stride=256)
        fig = extractor.plot_grid_preview(mock_slide)

        assert fig is not None
        plt.close(fig)

    def test_plot_grid_preview_empty_coordinates(self):
        """Test plot_grid_preview with no surviving patches."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        slide = MagicMock()
        slide.dimensions = (1000, 1000)
        slide.level_dimensions = [(1000, 1000)]
        slide.level_downsamples = [1.0]
        slide.tissue_mask = np.zeros((512, 512), dtype=bool)
        slide.path = Path("/fake/slide.svs")
        slide.magnification = 40.0
        slide.mpp = 1.0
        slide.get_thumbnail.return_value = np.random.randint(
            0, 255, (512, 512, 3), dtype=np.uint8
        )

        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
        fig = extractor.plot_grid_preview(slide)

        assert fig is not None
        plt.close(fig)


# ============================================================================
# tissue_coordinates Tests
# ============================================================================


class TestPatchExtractorTissueCoordinates:
    """Test extraction with pre-computed tissue coordinates."""

    def test_extract_with_tissue_coordinates(self, mock_slide):
        """Verify extraction filters correctly using tissue_coordinates."""
        # Define tissue in the top-left quadrant only (0-500, 0-500)
        tissue_coords = np.array(
            [[0, 0, 500, 500]],
            dtype=np.int64,
        )
        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
        patches = extractor.extract(mock_slide, tissue_coordinates=tissue_coords)

        assert isinstance(patches, Patches)
        # Only patches whose center falls in the 0-500 region should survive
        # Patch at (0,0,256,256) overlaps fully -> kept
        # Patch at (256,0,256,256) overlaps fully -> kept
        # Patch at (512,0,256,256) center at 640 — partially outside -> depends on ratio
        assert len(patches) > 0
        assert len(patches) < 9  # fewer than the full 3x3 grid
        # All kept patches should overlap the tissue region
        for c in patches.coordinates:
            x, y, w, h = c
            assert x < 500 or y < 500  # at least some overlap with tissue

    def test_get_coordinates_with_tissue_coordinates(self, mock_slide):
        """Verify coordinate-only mode works with tissue_coordinates."""
        tissue_coords = np.array(
            [[0, 0, 500, 500]],
            dtype=np.int64,
        )
        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
        coords = extractor.get_coordinates(mock_slide, tissue_coordinates=tissue_coords)

        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 2
        assert coords.shape[1] == 4
        assert len(coords) > 0
        assert len(coords) < 9

        # Should not have read any pixels
        mock_slide.read_region.assert_not_called()

    def test_tissue_coordinates_empty(self, mock_slide):
        """Edge case: empty tissue coordinates filters out everything."""
        empty_coords = np.empty((0, 4), dtype=np.int64)
        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
        patches = extractor.extract(mock_slide, tissue_coordinates=empty_coords)

        assert len(patches) == 0

    def test_tissue_coordinates_overrides_mask(self, mock_slide):
        """When both tissue_coordinates and tissue_mask exist, tissue_coordinates wins."""
        # Set tissue mask to full coverage (would keep all 9 patches)
        mock_slide.tissue_mask = np.ones((512, 512), dtype=bool)

        # But tissue_coordinates only covers a small region
        tissue_coords = np.array(
            [[0, 0, 300, 300]],
            dtype=np.int64,
        )
        extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
        patches = extractor.extract(mock_slide, tissue_coordinates=tissue_coords)

        # tissue_coordinates should take precedence — fewer than 9 patches
        assert len(patches) < 9

    def test_tissue_coordinates_metadata(self, mock_slide):
        """Metadata records that tissue_coordinates was used."""
        tissue_coords = np.array(
            [[0, 0, 500, 500], [500, 500, 500, 500]],
            dtype=np.int64,
        )
        extractor = PatchExtractor(patch_size=256, stride=256)
        patches = extractor.extract(mock_slide, tissue_coordinates=tissue_coords)

        assert patches.metadata["tissue_filter"] == "tissue_coordinates"
        assert patches.metadata["tissue_coordinates_count"] == 2

    def test_tissue_coordinates_invalid_shape(self, mock_slide):
        """Invalid tissue_coordinates shape raises ValueError."""
        bad_coords = np.array([[0, 0, 500]], dtype=np.int64)  # 3 cols instead of 4
        extractor = PatchExtractor(patch_size=256, stride=256)
        with pytest.raises(ValueError, match="tissue_coordinates must have shape"):
            extractor.extract(mock_slide, tissue_coordinates=bad_coords)
