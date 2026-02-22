"""Tests for WSI visualization utility functions."""

import numpy as np
import pytest

from honeybee.processors.wsi._vis_utils import _composite_overlay, _rasterize_patches


class TestRasterizePatches:
    """Test _rasterize_patches helper."""

    def test_basic_shape(self):
        """Test output shape matches thumbnail_size."""
        coords = np.array([[0, 0, 100, 100], [200, 200, 100, 100]], dtype=np.int64)
        values = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        result = _rasterize_patches(coords, values, (1000, 1000), (100, 100))
        assert result.shape == (100, 100, 4)
        assert result.dtype == np.float32

    def test_empty_coordinates(self):
        """Test with no coordinates returns zeros."""
        coords = np.empty((0, 4), dtype=np.int64)
        values = np.empty((0, 3), dtype=np.float32)
        result = _rasterize_patches(coords, values, (1000, 1000), (100, 100))
        assert result.shape == (100, 100, 4)
        assert np.all(result == 0)

    def test_alpha_value(self):
        """Test that alpha is set correctly in filled regions."""
        coords = np.array([[0, 0, 500, 500]], dtype=np.int64)
        values = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        result = _rasterize_patches(coords, values, (1000, 1000), (100, 100), alpha=0.5)
        # Top-left quadrant should have alpha=0.5
        assert result[0, 0, 3] == pytest.approx(0.5)

    def test_values_set_correctly(self):
        """Test that RGB values are placed correctly."""
        coords = np.array([[0, 0, 1000, 1000]], dtype=np.int64)
        values = np.array([[0.5, 0.3, 0.7]], dtype=np.float32)
        result = _rasterize_patches(coords, values, (1000, 1000), (10, 10))
        assert result[5, 5, 0] == pytest.approx(0.5)
        assert result[5, 5, 1] == pytest.approx(0.3)
        assert result[5, 5, 2] == pytest.approx(0.7)

    def test_zero_slide_dimensions(self):
        """Test with zero slide dimensions returns zeros."""
        coords = np.array([[0, 0, 100, 100]], dtype=np.int64)
        values = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        result = _rasterize_patches(coords, values, (0, 0), (100, 100))
        assert np.all(result == 0)


class TestCompositeOverlay:
    """Test _composite_overlay helper."""

    def test_output_shape(self):
        """Test output shape matches input."""
        thumb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        overlay = np.zeros((100, 100, 4), dtype=np.float32)
        result = _composite_overlay(thumb, overlay)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float32

    def test_no_overlay_returns_thumbnail(self):
        """Test zero overlay returns normalized thumbnail."""
        thumb = np.full((10, 10, 3), 128, dtype=np.uint8)
        overlay = np.zeros((10, 10, 4), dtype=np.float32)
        result = _composite_overlay(thumb, overlay)
        expected = 128.0 / 255.0
        assert result[5, 5, 0] == pytest.approx(expected, abs=0.01)

    def test_full_overlay_uses_tissue_blend(self):
        """Test full alpha overlay respects tissue_blend parameter."""
        thumb = np.full((10, 10, 3), 255, dtype=np.uint8)
        overlay = np.zeros((10, 10, 4), dtype=np.float32)
        overlay[:, :, 0] = 1.0  # Red
        overlay[:, :, 3] = 1.0  # Full alpha
        result = _composite_overlay(thumb, overlay, tissue_blend=0.0)
        # Should be pure overlay color
        assert result[5, 5, 0] == pytest.approx(1.0, abs=0.01)

    def test_output_clipped_to_01(self):
        """Test output is clipped to [0, 1]."""
        thumb = np.full((10, 10, 3), 255, dtype=np.uint8)
        overlay = np.ones((10, 10, 4), dtype=np.float32) * 2.0
        result = _composite_overlay(thumb, overlay)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_float_thumbnail_input(self):
        """Test that float32 thumbnail input works."""
        thumb = np.full((10, 10, 3), 0.5, dtype=np.float32)
        overlay = np.zeros((10, 10, 4), dtype=np.float32)
        result = _composite_overlay(thumb, overlay)
        assert result[5, 5, 0] == pytest.approx(0.5, abs=0.01)
