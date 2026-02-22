"""
Unit tests for the Patches container and compute_patch_quality function.

Tests cover:
- Patches creation and validation
- Container protocol (__len__, __getitem__, __iter__)
- Filtering by mask and quality
- Quality score computation (lazy evaluation)
- Stain normalization and separation
- I/O operations (save, to_numpy)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from honeybee.processors.wsi.patches import Patches, compute_patch_quality

# ============================================================================
# Helper fixtures
# ============================================================================


@pytest.fixture
def sample_patches():
    """Create a sample Patches object with 10 patches."""
    images = np.random.randint(0, 255, (10, 256, 256, 3), dtype=np.uint8)
    coordinates = np.array([[i * 256, 0, 256, 256] for i in range(10)])
    return Patches(images, coordinates)


@pytest.fixture
def small_patches():
    """Create a small Patches object with 3 patches for focused tests."""
    images = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
    coordinates = np.array([[0, 0, 64, 64], [64, 0, 64, 64], [128, 0, 64, 64]])
    return Patches(images, coordinates)


# ============================================================================
# Creation Tests
# ============================================================================


class TestPatchesCreation:
    """Test Patches object creation and validation."""

    def test_creation_basic(self, sample_patches):
        """Test basic creation."""
        assert len(sample_patches) == 10
        assert sample_patches.images.shape == (10, 256, 256, 3)
        assert sample_patches.coordinates.shape == (10, 4)

    def test_creation_with_metadata(self):
        """Test creation with metadata."""
        images = np.random.randint(0, 255, (5, 128, 128, 3), dtype=np.uint8)
        coords = np.zeros((5, 4), dtype=np.int64)
        meta = {"slide_path": "/fake/slide.svs", "level": 0}
        p = Patches(images, coords, metadata=meta)
        assert p.metadata["slide_path"] == "/fake/slide.svs"
        assert p.metadata["level"] == 0

    def test_creation_default_metadata(self, sample_patches):
        """Test that default metadata is an empty dict."""
        assert sample_patches.metadata == {}

    def test_creation_validates_image_shape_not_4d(self):
        """Test that non-4D images raise ValueError."""
        images = np.random.randint(0, 255, (10, 256, 3), dtype=np.uint8)
        coords = np.zeros((10, 4), dtype=np.int64)
        with pytest.raises(ValueError, match="images must have shape"):
            Patches(images, coords)

    def test_creation_validates_image_channels(self):
        """Test that non-3-channel images raise ValueError."""
        images = np.random.randint(0, 255, (10, 256, 256, 4), dtype=np.uint8)
        coords = np.zeros((10, 4), dtype=np.int64)
        with pytest.raises(ValueError, match="images must have shape"):
            Patches(images, coords)

    def test_creation_validates_coordinates_shape(self):
        """Test that wrong coordinate shape raises ValueError."""
        images = np.random.randint(0, 255, (10, 256, 256, 3), dtype=np.uint8)
        coords = np.zeros((10, 3), dtype=np.int64)  # wrong: 3 instead of 4
        with pytest.raises(ValueError, match="coordinates must have shape"):
            Patches(images, coords)

    def test_creation_validates_length_mismatch(self):
        """Test that mismatched image/coordinate counts raise ValueError."""
        images = np.random.randint(0, 255, (10, 256, 256, 3), dtype=np.uint8)
        coords = np.zeros((5, 4), dtype=np.int64)
        with pytest.raises(ValueError, match="must have the same length"):
            Patches(images, coords)

    def test_creation_empty(self):
        """Test creation with zero patches."""
        images = np.empty((0, 256, 256, 3), dtype=np.uint8)
        coords = np.empty((0, 4), dtype=np.int64)
        p = Patches(images, coords)
        assert len(p) == 0


# ============================================================================
# Container Protocol Tests
# ============================================================================


class TestPatchesContainer:
    """Test container protocol methods."""

    def test_len(self, sample_patches):
        """Test __len__."""
        assert len(sample_patches) == 10

    def test_getitem_int(self, sample_patches):
        """Test indexing with single int."""
        subset = sample_patches[0]
        assert isinstance(subset, Patches)
        assert len(subset) == 1
        np.testing.assert_array_equal(subset.images[0], sample_patches.images[0])

    def test_getitem_slice(self, sample_patches):
        """Test indexing with slice."""
        subset = sample_patches[2:5]
        assert isinstance(subset, Patches)
        assert len(subset) == 3
        np.testing.assert_array_equal(subset.images, sample_patches.images[2:5])
        np.testing.assert_array_equal(subset.coordinates, sample_patches.coordinates[2:5])

    def test_getitem_array(self, sample_patches):
        """Test indexing with integer array."""
        indices = np.array([0, 3, 7])
        subset = sample_patches[indices]
        assert isinstance(subset, Patches)
        assert len(subset) == 3
        np.testing.assert_array_equal(subset.images[0], sample_patches.images[0])
        np.testing.assert_array_equal(subset.images[1], sample_patches.images[3])
        np.testing.assert_array_equal(subset.images[2], sample_patches.images[7])

    def test_getitem_bool_mask(self, sample_patches):
        """Test indexing with boolean mask."""
        mask = np.array([True, False] * 5)
        subset = sample_patches[mask]
        assert isinstance(subset, Patches)
        assert len(subset) == 5

    def test_iter(self, sample_patches):
        """Test iteration yields individual patch images."""
        patches_list = list(sample_patches)
        assert len(patches_list) == 10
        for patch_img in patches_list:
            assert patch_img.shape == (256, 256, 3)
            assert patch_img.dtype == np.uint8

    def test_repr(self, sample_patches):
        """Test string representation."""
        r = repr(sample_patches)
        assert "Patches" in r
        assert "n=10" in r
        assert "256x256" in r

    def test_repr_empty(self):
        """Test repr for empty patches."""
        images = np.empty((0, 64, 64, 3), dtype=np.uint8)
        coords = np.empty((0, 4), dtype=np.int64)
        p = Patches(images, coords)
        assert "n=0" in repr(p)


# ============================================================================
# Filtering Tests
# ============================================================================


class TestPatchesFiltering:
    """Test filtering operations."""

    def test_filter_by_mask(self, sample_patches):
        """Test filter with boolean mask."""
        mask = np.array([True, True, False, False, True, False, True, False, True, False])
        filtered = sample_patches.filter(mask=mask)
        assert isinstance(filtered, Patches)
        assert len(filtered) == 5

    def test_filter_by_quality(self, sample_patches):
        """Test filter by minimum quality score."""
        # Mock compute_patch_quality to return known scores
        fake_scores = np.linspace(0.0, 1.0, 10).astype(np.float32)
        with patch(
            "honeybee.processors.wsi.patches.compute_patch_quality",
            return_value=fake_scores,
        ):
            filtered = sample_patches.filter(min_quality=0.5)
        # Scores >= 0.5: indices 5,6,7,8,9 (5 values out of 10 linspaced 0..1)
        assert len(filtered) >= 4  # at least 0.556, 0.667, 0.778, 0.889, 1.0

    def test_filter_returns_new_object(self, sample_patches):
        """Test that filter returns a new Patches object (non-destructive)."""
        mask = np.ones(10, dtype=bool)
        mask[0] = False
        filtered = sample_patches.filter(mask=mask)
        assert filtered is not sample_patches
        assert len(sample_patches) == 10  # original unchanged
        assert len(filtered) == 9

    def test_filter_invalid_mask_length(self, sample_patches):
        """Test that wrong mask length raises ValueError."""
        mask = np.array([True, False, True])  # length 3, not 10
        with pytest.raises(ValueError, match="mask length"):
            sample_patches.filter(mask=mask)

    def test_filter_combined_mask_and_quality(self, sample_patches):
        """Test filter with both mask and min_quality."""
        mask = np.ones(10, dtype=bool)
        mask[0:3] = False  # remove first 3
        fake_scores = np.array(
            [0.9, 0.9, 0.9, 0.1, 0.2, 0.8, 0.9, 0.1, 0.8, 0.95],
            dtype=np.float32,
        )
        with patch(
            "honeybee.processors.wsi.patches.compute_patch_quality",
            return_value=fake_scores,
        ):
            filtered = sample_patches.filter(mask=mask, min_quality=0.5)
        # mask removes [0,1,2]; quality >= 0.5 from remaining: [5]=0.8, [6]=0.9, [8]=0.8, [9]=0.95
        assert len(filtered) == 4


# ============================================================================
# Quality Score Tests
# ============================================================================


class TestPatchesQuality:
    """Test quality score computation and propagation."""

    def test_quality_scores_lazy(self, sample_patches):
        """Quality scores are computed on first access, not at creation."""
        # Internal cache should be None initially
        assert sample_patches._quality_scores is None

        # Patch out compute_patch_quality so we don't need cv2
        fake_scores = np.random.rand(10).astype(np.float32)
        with patch(
            "honeybee.processors.wsi.patches.compute_patch_quality",
            return_value=fake_scores,
        ) as mock_cpq:
            scores = sample_patches.quality_scores
            mock_cpq.assert_called_once()
            np.testing.assert_array_equal(scores, fake_scores)

    def test_quality_scores_cached(self, sample_patches):
        """Quality scores are only computed once."""
        fake_scores = np.random.rand(10).astype(np.float32)
        with patch(
            "honeybee.processors.wsi.patches.compute_patch_quality",
            return_value=fake_scores,
        ) as mock_cpq:
            _ = sample_patches.quality_scores
            _ = sample_patches.quality_scores  # second access
            mock_cpq.assert_called_once()

    def test_quality_scores_empty_patches(self):
        """Quality scores for empty patches returns empty array."""
        images = np.empty((0, 64, 64, 3), dtype=np.uint8)
        coords = np.empty((0, 4), dtype=np.int64)
        p = Patches(images, coords)
        scores = p.quality_scores
        assert len(scores) == 0
        assert scores.dtype == np.float32

    def test_quality_propagated_through_indexing(self, sample_patches):
        """Quality scores transfer when slicing."""
        fake_scores = np.arange(10, dtype=np.float32) / 10.0
        with patch(
            "honeybee.processors.wsi.patches.compute_patch_quality",
            return_value=fake_scores,
        ):
            _ = sample_patches.quality_scores  # trigger computation

        subset = sample_patches[3:6]
        np.testing.assert_array_equal(subset._quality_scores, fake_scores[3:6])

    def test_quality_not_propagated_if_not_computed(self, sample_patches):
        """Quality scores are not propagated if never computed."""
        subset = sample_patches[0:3]
        assert subset._quality_scores is None


# ============================================================================
# Stain Operation Tests
# ============================================================================


class TestPatchesStainOps:
    """Test stain normalization and separation operations."""

    def test_normalize_returns_new_patches(self, small_patches):
        """normalize() returns new Patches, original unchanged."""
        mock_normalizer = MagicMock()
        mock_normalizer.transform.side_effect = lambda x: x  # identity

        with patch(
            "honeybee.processors.wsi.stain_normalization.ReinhardNormalizer",
            return_value=mock_normalizer,
        ), patch(
            "honeybee.processors.wsi.stain_normalization.STAIN_NORM_TARGETS",
            {"tcga_avg": {"mean_lab": np.zeros(3), "std_lab": np.ones(3)}},
        ):
            result = small_patches.normalize(method="reinhard")

        assert isinstance(result, Patches)
        assert result is not small_patches
        assert len(result) == len(small_patches)

    def test_normalize_methods(self, small_patches):
        """Test reinhard, macenko, vahadane methods create correct normalizers."""
        for method, cls_name in [
            ("reinhard", "ReinhardNormalizer"),
            ("macenko", "MacenkoNormalizer"),
            ("vahadane", "VahadaneNormalizer"),
        ]:
            mock_normalizer = MagicMock()
            mock_normalizer.transform.side_effect = lambda x: x

            with patch(
                f"honeybee.processors.wsi.stain_normalization.{cls_name}",
                return_value=mock_normalizer,
            ), patch(
                "honeybee.processors.wsi.stain_normalization.STAIN_NORM_TARGETS",
                {"tcga_avg": {"mean_lab": np.zeros(3), "std_lab": np.ones(3)}},
            ):
                result = small_patches.normalize(method=method)
                assert len(result) == 3
                assert result.metadata.get("stain_normalization") == method

    def test_normalize_invalid_method(self, small_patches):
        """Test invalid normalization method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            small_patches.normalize(method="invalid")

    def test_normalize_preserves_on_failure(self, small_patches):
        """Test that failed normalization keeps original patch."""
        mock_normalizer = MagicMock()
        mock_normalizer.transform.side_effect = RuntimeError("Normalization failed")

        with patch(
            "honeybee.processors.wsi.stain_normalization.MacenkoNormalizer",
            return_value=mock_normalizer,
        ), patch(
            "honeybee.processors.wsi.stain_normalization.STAIN_NORM_TARGETS",
            {"tcga_avg": {"mean_lab": np.zeros(3), "std_lab": np.ones(3)}},
        ):
            result = small_patches.normalize(method="macenko")

        # Should keep original patches on failure
        np.testing.assert_array_equal(result.images, small_patches.images)

    def test_separate_stains(self, small_patches):
        """Test stain separation returns dict with expected keys."""
        mock_separator = MagicMock()
        mock_separator.separate.return_value = {
            "hematoxylin": np.random.rand(64, 64),
            "eosin": np.random.rand(64, 64),
            "background": np.random.rand(64, 64),
        }

        with patch(
            "honeybee.processors.wsi.stain_separation.StainSeparator",
            return_value=mock_separator,
        ):
            result = small_patches.separate_stains(method="hed")

        assert "hematoxylin" in result
        assert "eosin" in result
        assert "background" in result
        assert result["hematoxylin"].shape[0] == 3  # 3 patches
        assert result["eosin"].shape[0] == 3
        mock_separator.separate.assert_called()

    def test_separate_stains_empty(self):
        """Test stain separation on empty patches."""
        images = np.empty((0, 64, 64, 3), dtype=np.uint8)
        coords = np.empty((0, 4), dtype=np.int64)
        p = Patches(images, coords)

        # The method should return empty arrays without calling separator
        result = p.separate_stains()
        assert result["hematoxylin"].shape[0] == 0
        assert result["eosin"].shape[0] == 0
        assert result["background"].shape[0] == 0


# ============================================================================
# I/O Tests
# ============================================================================


class TestPatchesIO:
    """Test I/O operations."""

    def test_save(self, small_patches, tmp_path):
        """Test saving patches to files."""
        output_dir = str(tmp_path / "patches_out")
        saved = small_patches.save(output_dir, prefix="test", format="png")
        assert len(saved) == 3
        for fpath in saved:
            assert Path(fpath).exists()
            assert fpath.endswith(".png")

    def test_save_creates_directory(self, small_patches, tmp_path):
        """Test that save creates the output directory."""
        output_dir = str(tmp_path / "new_dir" / "patches")
        small_patches.save(output_dir)
        assert Path(output_dir).exists()

    def test_to_numpy(self, sample_patches):
        """Test to_numpy returns images array."""
        arr = sample_patches.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 256, 256, 3)
        np.testing.assert_array_equal(arr, sample_patches.images)


# ============================================================================
# compute_patch_quality Tests
# ============================================================================


class TestComputePatchQuality:
    """Test the standalone compute_patch_quality function."""

    def test_quality_shape(self):
        """Test output shape matches input count."""
        patches = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        scores = compute_patch_quality(patches)
        assert scores.shape == (5,)

    def test_quality_dtype(self):
        """Test output dtype is float32."""
        patches = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
        scores = compute_patch_quality(patches)
        assert scores.dtype == np.float32

    def test_quality_range(self):
        """Test scores are in [0, 1]."""
        patches = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        scores = compute_patch_quality(patches)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_white_patches_low_quality(self):
        """White patches should get low quality scores (low tissue %)."""
        white_patches = np.full((3, 64, 64, 3), 240, dtype=np.uint8)
        scores = compute_patch_quality(white_patches)
        # All white -> tissue_score ~ 0, blur_score varies
        # But tissue contribution should pull scores down
        assert np.all(scores < 0.5)

    def test_invalid_input_shape(self):
        """Test that wrong input shape raises ValueError."""
        patches_3d = np.random.randint(0, 255, (5, 64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected patches with shape"):
            compute_patch_quality(patches_3d)

    def test_invalid_channels(self):
        """Test that wrong channel count raises ValueError."""
        patches_4ch = np.random.randint(0, 255, (5, 64, 64, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected patches with shape"):
            compute_patch_quality(patches_4ch)


# ============================================================================
# Visualization Method Tests
# ============================================================================


class TestPatchesVisualization:
    """Test new visualization methods on Patches."""

    @pytest.fixture
    def vis_patches(self):
        """Patches with pre-set quality scores for visualization tests."""
        images = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        coords = np.array([[i * 64, 0, 64, 64] for i in range(10)])
        p = Patches(images, coords)
        # Pre-set quality scores to avoid cv2 dependency
        p._quality_scores = np.linspace(0.1, 1.0, 10).astype(np.float32)
        return p

    def test_plot_quality_distribution_returns_figure(self, vis_patches):
        """Test plot_quality_distribution returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = vis_patches.plot_quality_distribution()
        assert fig is not None
        plt.close(fig)

    def test_plot_quality_distribution_with_threshold(self, vis_patches):
        """Test threshold line is drawn."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = vis_patches.plot_quality_distribution(threshold=0.5)
        assert fig is not None
        plt.close(fig)

    def test_plot_quality_distribution_accepts_ax(self, vis_patches):
        """Test drawing on provided axes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_ext, ax_ext = plt.subplots()
        fig = vis_patches.plot_quality_distribution(ax=ax_ext)
        assert fig is fig_ext
        plt.close(fig_ext)

    def test_plot_normalization_comparison_returns_figure(self, vis_patches):
        """Test plot_normalization_comparison returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mock_normalizer = MagicMock()
        mock_normalizer.transform.side_effect = lambda x: x

        with patch(
            "honeybee.processors.wsi.stain_normalization.ReinhardNormalizer",
            return_value=mock_normalizer,
        ), patch(
            "honeybee.processors.wsi.stain_normalization.MacenkoNormalizer",
            return_value=mock_normalizer,
        ), patch(
            "honeybee.processors.wsi.stain_normalization.VahadaneNormalizer",
            return_value=mock_normalizer,
        ), patch(
            "honeybee.processors.wsi.stain_normalization.STAIN_NORM_TARGETS",
            {"tcga_avg": {"mean_lab": np.zeros(3), "std_lab": np.ones(3)}},
        ):
            fig = vis_patches.plot_normalization_comparison()

        assert fig is not None
        assert len(fig.axes) == 4  # original + 3 methods
        plt.close(fig)

    def test_plot_normalization_before_after_returns_figure(self, vis_patches):
        """Test plot_normalization_before_after returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use same patches as "normalized" for simplicity
        fig = vis_patches.plot_normalization_before_after(vis_patches)
        assert fig is not None
        plt.close(fig)

    def test_plot_normalization_before_after_empty(self):
        """Test with empty patches."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        empty = Patches(
            np.empty((0, 64, 64, 3), dtype=np.uint8),
            np.empty((0, 4), dtype=np.int64),
        )
        fig = empty.plot_normalization_before_after(empty)
        assert fig is not None
        plt.close(fig)

    def test_plot_stain_separation_returns_figure(self, vis_patches):
        """Test plot_stain_separation returns a Figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mock_separator = MagicMock()
        mock_separator.separate.return_value = {
            "hematoxylin": np.random.rand(64, 64),
            "eosin": np.random.rand(64, 64),
            "background": np.random.rand(64, 64),
        }

        with patch(
            "honeybee.processors.wsi.stain_separation.StainSeparator",
            return_value=mock_separator,
        ):
            fig = vis_patches.plot_stain_separation(patch_index=0)

        assert fig is not None
        assert len(fig.axes) == 4
        plt.close(fig)


# ============================================================================
# to_records Tests
# ============================================================================


class TestPatchesToRecords:
    """Test the to_records() export method."""

    def test_to_records_basic(self, small_patches):
        """Returns list of dicts with correct keys."""
        records = small_patches.to_records()
        assert isinstance(records, list)
        assert len(records) == 3
        for i, rec in enumerate(records):
            assert rec["index"] == i
            assert "x" in rec
            assert "y" in rec
            assert "width" in rec
            assert "height" in rec
            # No quality_score since not computed
            assert "quality_score" not in rec
            # No embedding since not provided
            assert "embedding" not in rec
            # No image since include_images=False
            assert "image" not in rec

    def test_to_records_with_embeddings(self, small_patches):
        """Embeddings included in records when provided."""
        embeddings = np.random.rand(3, 128).astype(np.float32)
        records = small_patches.to_records(embeddings=embeddings)
        assert len(records) == 3
        for i, rec in enumerate(records):
            assert "embedding" in rec
            np.testing.assert_array_equal(rec["embedding"], embeddings[i])

    def test_to_records_with_images(self, small_patches):
        """Images included when requested."""
        records = small_patches.to_records(include_images=True)
        assert len(records) == 3
        for i, rec in enumerate(records):
            assert "image" in rec
            np.testing.assert_array_equal(rec["image"], small_patches.images[i])

    def test_to_records_with_quality(self, small_patches):
        """Quality scores included when pre-computed."""
        small_patches._quality_scores = np.array([0.5, 0.7, 0.9], dtype=np.float32)
        records = small_patches.to_records()
        for i, rec in enumerate(records):
            assert "quality_score" in rec
            assert rec["quality_score"] == pytest.approx(small_patches._quality_scores[i])

    def test_to_records_includes_metadata(self):
        """Collection-level metadata is merged into each record."""
        images = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        coords = np.array([[0, 0, 64, 64], [64, 0, 64, 64]])
        meta = {"slide_path": "/fake/slide.svs", "patch_size": 64}
        p = Patches(images, coords, metadata=meta)
        records = p.to_records()
        for rec in records:
            assert rec["slide_path"] == "/fake/slide.svs"
            assert rec["patch_size"] == 64

    def test_to_records_empty_patches(self):
        """Edge case: empty patches returns empty list."""
        images = np.empty((0, 64, 64, 3), dtype=np.uint8)
        coords = np.empty((0, 4), dtype=np.int64)
        p = Patches(images, coords)
        records = p.to_records()
        assert records == []

    def test_to_records_embedding_length_mismatch(self, small_patches):
        """Wrong embedding count raises ValueError."""
        embeddings = np.random.rand(5, 128).astype(np.float32)  # 5 != 3
        with pytest.raises(ValueError, match="embeddings length"):
            small_patches.to_records(embeddings=embeddings)
