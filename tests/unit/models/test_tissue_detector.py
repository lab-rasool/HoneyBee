"""
Unit tests for the TissueDetector DL tissue detection model.

Tests cover:
- Initialisation with custom patch_size / batch_size
- predict_batch() output shape and probability properties
- detect() end-to-end with a mocked Slide
- Edge cases: tiny slide, various thresholds
- prediction_map_to_rgb() helper
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tissue_detector(patch_size=224, batch_size=32, device="cpu"):
    """Build a TissueDetector whose model is a simple mock (no real weights)."""
    with patch(
        "honeybee.models.TissueDetector.tissue_detector.TissueDetector._download_weights",
        return_value="/fake/weights.pt",
    ), patch(
        "honeybee.models.TissueDetector.tissue_detector.TissueDetector._load_model",
    ) as mock_load:
        # Create a tiny model stub that returns (N, 3) logits
        fake_model = MagicMock()

        def _forward(x):
            n = x.shape[0]
            # Return logits that, after softmax, give ~equal probs
            return torch.zeros(n, 3)

        fake_model.side_effect = _forward
        fake_model.return_value = torch.zeros(1, 3)
        mock_load.return_value = fake_model

        from honeybee.models.TissueDetector.tissue_detector import TissueDetector

        detector = TissueDetector(
            model_path="/fake/weights.pt",
            device=device,
            patch_size=patch_size,
            batch_size=batch_size,
        )
        # Replace model with a callable that returns proper logits
        detector.model = fake_model
    return detector


def _make_mock_slide(
    level_dims=None, level_downsamples=None, level_count=3, thumb_shape=(512, 512, 3)
):
    """Create a mock Slide with realistic attributes."""
    if level_dims is None:
        level_dims = [(10000, 10000), (2500, 2500), (625, 625)]
    if level_downsamples is None:
        level_downsamples = [1.0, 4.0, 16.0]

    slide = MagicMock()
    slide.level_count = level_count
    slide.level_dimensions = level_dims
    slide.level_downsamples = level_downsamples
    slide.dimensions = level_dims[0]
    slide.magnification = 40.0

    def _read_region(location, size, level=0):
        w, h = size
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    slide.read_region.side_effect = _read_region
    slide.get_thumbnail.return_value = np.random.randint(
        0, 255, thumb_shape, dtype=np.uint8
    )
    slide.get_best_level_for_magnification.return_value = 1
    return slide


# ============================================================================
# Tests
# ============================================================================


class TestTissueDetectorInit:
    """Test constructor parameters."""

    def test_default_params(self):
        detector = _make_mock_tissue_detector()
        assert detector.patch_size == 224
        assert detector.batch_size == 32
        assert detector.device == torch.device("cpu")

    def test_custom_patch_size(self):
        detector = _make_mock_tissue_detector(patch_size=128)
        assert detector.patch_size == 128

    def test_custom_batch_size(self):
        detector = _make_mock_tissue_detector(batch_size=64)
        assert detector.batch_size == 64

    def test_class_names(self):
        from honeybee.models.TissueDetector.tissue_detector import TissueDetector

        assert TissueDetector.CLASS_NAMES == ("artifact", "background", "tissue")


class TestPredictBatch:
    """Test predict_batch() output shape and properties."""

    def test_output_shape(self):
        detector = _make_mock_tissue_detector()
        # Mock model to return proper logits
        detector.model = MagicMock(
            side_effect=lambda x: torch.randn(x.shape[0], 3)
        )
        images = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
        probs = detector.predict_batch(images)
        assert probs.shape == (4, 3)
        assert probs.dtype == np.float32

    def test_probabilities_sum_to_one(self):
        detector = _make_mock_tissue_detector()
        detector.model = MagicMock(
            side_effect=lambda x: torch.randn(x.shape[0], 3)
        )
        images = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        probs = detector.predict_batch(images)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_probabilities_non_negative(self):
        detector = _make_mock_tissue_detector()
        detector.model = MagicMock(
            side_effect=lambda x: torch.randn(x.shape[0], 3)
        )
        images = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
        probs = detector.predict_batch(images)
        assert (probs >= 0).all()

    def test_single_image(self):
        detector = _make_mock_tissue_detector()
        detector.model = MagicMock(
            side_effect=lambda x: torch.randn(x.shape[0], 3)
        )
        images = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
        probs = detector.predict_batch(images)
        assert probs.shape == (1, 3)


class TestDetect:
    """Test detect() end-to-end with mock Slide."""

    def _make_detector_with_tissue_output(self):
        """Create a detector whose model always predicts tissue class."""
        detector = _make_mock_tissue_detector()
        # Model returns high tissue logit: [low, low, high]
        detector.model = MagicMock(
            side_effect=lambda x: torch.tensor([[0.0, 0.0, 5.0]] * x.shape[0])
        )
        return detector

    def _make_detector_with_background_output(self):
        """Create a detector whose model always predicts background."""
        detector = _make_mock_tissue_detector()
        detector.model = MagicMock(
            side_effect=lambda x: torch.tensor([[0.0, 5.0, 0.0]] * x.shape[0])
        )
        return detector

    def test_returns_mask_and_pred_map(self):
        detector = self._make_detector_with_tissue_output()
        slide = _make_mock_slide()
        mask, pred_map = detector.detect(slide)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert isinstance(pred_map, np.ndarray)
        assert pred_map.ndim == 3
        assert pred_map.shape[2] == 3

    def test_mask_shape_matches_thumbnail(self):
        detector = self._make_detector_with_tissue_output()
        slide = _make_mock_slide(thumb_shape=(512, 512, 3))
        mask, _ = detector.detect(slide, thumbnail_size=(512, 512))
        assert mask.shape == (512, 512)

    def test_all_tissue_gives_full_mask(self):
        detector = self._make_detector_with_tissue_output()
        slide = _make_mock_slide()
        mask, pred_map = detector.detect(slide, threshold=0.5)
        # All predictions are tissue -> mask should be mostly True
        # (may not be 100% due to grid not covering every pixel)
        assert mask.mean() > 0.5

    def test_all_background_gives_empty_mask(self):
        detector = self._make_detector_with_background_output()
        slide = _make_mock_slide()
        mask, _ = detector.detect(slide, threshold=0.5)
        assert mask.mean() == 0.0

    def test_default_level_is_lowest_res(self):
        detector = self._make_detector_with_tissue_output()
        slide = _make_mock_slide()
        detector.detect(slide)
        # Should read at level 2 (lowest res) by default
        # Check that read_region was called with level=2
        for call in slide.read_region.call_args_list:
            assert call.kwargs.get("level", call.args[2] if len(call.args) > 2 else None) == 2

    def test_explicit_level(self):
        detector = self._make_detector_with_tissue_output()
        slide = _make_mock_slide()
        detector.detect(slide, level=1)
        for call in slide.read_region.call_args_list:
            assert call.kwargs.get("level") == 1

    def test_magnification_param(self):
        detector = self._make_detector_with_tissue_output()
        slide = _make_mock_slide()
        slide.get_best_level_for_magnification.return_value = 1
        detector.detect(slide, magnification=10.0)
        slide.get_best_level_for_magnification.assert_called_with(10.0)

    def test_custom_threshold(self):
        detector = _make_mock_tissue_detector()
        # Return borderline tissue probability (~0.6)
        detector.model = MagicMock(
            side_effect=lambda x: torch.tensor([[0.0, 0.0, 0.5]] * x.shape[0])
        )
        slide = _make_mock_slide()
        mask_low, _ = detector.detect(slide, threshold=0.3)
        mask_high, _ = detector.detect(slide, threshold=0.9)
        # Lower threshold should detect more tissue
        assert mask_low.mean() >= mask_high.mean()

    def test_tiny_slide(self):
        """Slide too small to form even one tile."""
        detector = _make_mock_tissue_detector(patch_size=224)
        slide = _make_mock_slide(
            level_dims=[(100, 100)],
            level_downsamples=[1.0],
            level_count=1,
            thumb_shape=(100, 100, 3),
        )
        mask, pred_map = detector.detect(slide)
        assert mask.shape == (100, 100)
        assert mask.sum() == 0  # empty mask
        assert pred_map.shape[0] == 0  # no predictions

    def test_pred_map_probabilities_valid(self):
        detector = self._make_detector_with_tissue_output()
        slide = _make_mock_slide()
        _, pred_map = detector.detect(slide)
        assert pred_map.min() >= 0.0
        assert pred_map.max() <= 1.0
        # Each pixel's probs should sum to ~1
        if pred_map.size > 0:
            sums = pred_map.sum(axis=2)
            np.testing.assert_allclose(sums, 1.0, atol=1e-5)


class TestPredictionMapToRGB:
    """Test the static visualisation helper."""

    def test_output_shape_and_dtype(self):
        from honeybee.models.TissueDetector.tissue_detector import TissueDetector

        pred_map = np.random.rand(8, 10, 3).astype(np.float32)
        rgb = TissueDetector.prediction_map_to_rgb(pred_map)
        assert rgb.shape == (8, 10, 3)
        assert rgb.dtype == np.uint8

    def test_all_tissue_is_blue(self):
        from honeybee.models.TissueDetector.tissue_detector import TissueDetector

        pred_map = np.zeros((2, 2, 3), dtype=np.float32)
        pred_map[:, :, 2] = 1.0  # tissue = 1.0
        rgb = TissueDetector.prediction_map_to_rgb(pred_map)
        assert rgb[0, 0, 2] == 255  # blue channel
        assert rgb[0, 0, 0] == 0  # red channel
        assert rgb[0, 0, 1] == 0  # green channel

    def test_values_clipped(self):
        from honeybee.models.TissueDetector.tissue_detector import TissueDetector

        pred_map = np.full((2, 2, 3), 1.5, dtype=np.float32)
        rgb = TissueDetector.prediction_map_to_rgb(pred_map)
        assert rgb.max() == 255
