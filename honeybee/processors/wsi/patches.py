"""
Patches Container for Whole Slide Image Processing

Rich container that holds extracted WSI patches together with their coordinates
and metadata. Supports filtering, stain operations, visualization, and I/O.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

logger = logging.getLogger(__name__)

__all__ = [
    "Patches",
    "compute_patch_quality",
]


# ---------------------------------------------------------------------------
# Module-level quality scoring
# ---------------------------------------------------------------------------


def compute_patch_quality(patches: np.ndarray) -> np.ndarray:
    """Score each patch on a 0-1 scale for quality.

    The algorithm combines three signals:

    1. **Blur detection** -- Laplacian variance (higher is sharper), normalised
       to [0, 1] with a 500.0 scale factor.
    2. **Tissue percentage** -- ratio of pixels where *all* RGB channels are
       below 220 (i.e. non-white).
    3. **Pen-mark detection** -- high-saturation pixels whose hue falls into
       typical pen-ink ranges (red, green, blue markers).

    The final score is ``0.5 * blur_score + 0.5 * tissue_score``, multiplied
    by a penalty if pen marks exceed 1 % of the patch area.

    Args:
        patches: Array of shape ``(N, H, W, 3)`` with uint8 RGB images.

    Returns:
        Array of shape ``(N,)`` with float32 quality scores in [0, 1].
    """
    import cv2

    if patches.ndim != 4 or patches.shape[-1] != 3:
        raise ValueError(f"Expected patches with shape (N, H, W, 3), got {patches.shape}")

    scores = np.zeros(len(patches), dtype=np.float32)

    for i, patch in enumerate(patches):
        patch_uint8 = patch.astype(np.uint8) if patch.dtype != np.uint8 else patch

        # 1. Blur detection via Laplacian variance (higher = sharper)
        gray = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500.0)

        # 2. Tissue percentage (non-white pixels)
        white_thresh = 220
        non_white = np.mean(np.all(patch_uint8 < white_thresh, axis=-1))
        tissue_score = float(non_white)

        # 3. Pen mark detection (saturated non-tissue colours)
        hsv = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2HSV)
        high_sat = hsv[:, :, 1] > 100
        hue = hsv[:, :, 0]
        # Pen hue ranges in OpenCV (0-180):
        #   red: <10 or >170, green/blue: 35-130
        pen_hue = (hue < 10) | (hue > 170) | ((hue > 35) & (hue < 130))
        pen_ratio = float(np.mean(high_sat & pen_hue))

        # Combined score with pen penalty
        score = 0.5 * blur_score + 0.5 * tissue_score
        if pen_ratio > 0.01:
            score *= max(0.0, 1.0 - pen_ratio * 5)  # 20 % pen -> score x 0

        scores[i] = score

    return scores


# ---------------------------------------------------------------------------
# Patches container
# ---------------------------------------------------------------------------


class Patches:
    """Rich container for extracted WSI patches.

    Holds patch images together with their level-0 coordinates and optional
    metadata.  All mutating operations (filtering, normalisation) return
    **new** ``Patches`` objects, keeping the original intact.

    Parameters
    ----------
    images : np.ndarray
        Patch images with shape ``(N, H, W, 3)`` and dtype ``uint8``.
    coordinates : np.ndarray
        Level-0 coordinates with shape ``(N, 4)`` -- columns are
        ``[x, y, width, height]``.
    metadata : dict, optional
        Arbitrary key/value metadata attached to this collection.

    Examples
    --------
    >>> import numpy as np
    >>> imgs = np.random.randint(0, 255, (10, 256, 256, 3), dtype=np.uint8)
    >>> coords = np.tile([0, 0, 256, 256], (10, 1))
    >>> p = Patches(imgs, coords)
    >>> len(p)
    10
    >>> subset = p[0:5]
    >>> len(subset)
    5
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        images: np.ndarray,
        coordinates: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        images = np.asarray(images)
        coordinates = np.asarray(coordinates)

        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"images must have shape (N, H, W, 3), got {images.shape}")
        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise ValueError(f"coordinates must have shape (N, 4), got {coordinates.shape}")
        if len(images) != len(coordinates):
            raise ValueError(
                f"images ({len(images)}) and coordinates ({len(coordinates)}) "
                "must have the same length"
            )

        self._images: np.ndarray = images
        self._coordinates: np.ndarray = coordinates
        self._quality_scores: Optional[np.ndarray] = None
        self._metadata: dict = metadata if metadata is not None else {}

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def images(self) -> np.ndarray:
        """Patch images -- shape ``(N, H, W, 3)``, dtype ``uint8``."""
        return self._images

    @property
    def coordinates(self) -> np.ndarray:
        """Level-0 coordinates -- shape ``(N, 4)`` ``[x, y, w, h]``."""
        return self._coordinates

    @property
    def quality_scores(self) -> np.ndarray:
        """Per-patch quality scores (computed lazily on first access)."""
        if self._quality_scores is None:
            if len(self._images) == 0:
                self._quality_scores = np.array([], dtype=np.float32)
            else:
                self._quality_scores = compute_patch_quality(self._images)
        return self._quality_scores

    @property
    def metadata(self) -> dict:
        """Arbitrary metadata dictionary."""
        return self._metadata

    # ------------------------------------------------------------------ #
    # Container protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx) -> "Patches":
        """Index into the collection.

        Supports ``int``, ``slice``, integer arrays, and boolean masks.
        Always returns a new :class:`Patches` object.
        """
        if isinstance(idx, (int, np.integer)):
            # Single int -> still return a Patches (with 1 element)
            idx = [int(idx)]

        images = self._images[idx]
        coordinates = self._coordinates[idx]

        # Propagate pre-computed quality scores if available
        quality = None
        if self._quality_scores is not None:
            quality = self._quality_scores[idx]

        child = Patches(images, coordinates, metadata=self._metadata.copy())
        if quality is not None:
            child._quality_scores = quality
        return child

    def __iter__(self):
        """Iterate over individual patch images (yields ``np.ndarray``)."""
        for i in range(len(self._images)):
            yield self._images[i]

    def __repr__(self) -> str:
        if len(self) == 0:
            return "Patches(n=0)"
        h, w = self._images.shape[1], self._images.shape[2]
        return f"Patches(n={len(self)}, patch_size={h}x{w})"

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #

    def filter(
        self,
        mask: Optional[np.ndarray] = None,
        min_quality: Optional[float] = None,
    ) -> "Patches":
        """Return a filtered subset of patches.

        Parameters
        ----------
        mask : np.ndarray, optional
            Boolean array of length ``N``.  ``True`` keeps the patch.
        min_quality : float, optional
            Minimum quality score (triggers lazy quality computation).

        Returns
        -------
        Patches
            New ``Patches`` object containing only the selected patches.

        Notes
        -----
        When both *mask* and *min_quality* are given they are combined with
        logical AND.
        """
        combined_mask = np.ones(len(self), dtype=bool)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if len(mask) != len(self):
                raise ValueError(f"mask length ({len(mask)}) != number of patches ({len(self)})")
            combined_mask &= mask

        if min_quality is not None:
            combined_mask &= self.quality_scores >= min_quality

        return self[combined_mask]

    # ------------------------------------------------------------------ #
    # Stain operations
    # ------------------------------------------------------------------ #

    def normalize(self, method: str = "macenko", **kwargs) -> "Patches":
        """Return a new ``Patches`` with stain-normalised images.

        Uses the normaliser classes from
        :mod:`honeybee.processors.wsi.stain_normalization`.  Each patch is
        normalised independently; failures are handled gracefully by keeping
        the original patch.

        Parameters
        ----------
        method : str
            One of ``"reinhard"``, ``"macenko"``, ``"vahadane"``.
        **kwargs
            Forwarded to the normaliser constructor.

        Returns
        -------
        Patches
            New ``Patches`` object with normalised images.
        """
        from .stain_normalization import (
            STAIN_NORM_TARGETS,
            MacenkoNormalizer,
            ReinhardNormalizer,
            VahadaneNormalizer,
        )

        method_lower = method.lower()
        if method_lower == "reinhard":
            normalizer = ReinhardNormalizer()
        elif method_lower == "macenko":
            normalizer = MacenkoNormalizer(**kwargs)
        elif method_lower == "vahadane":
            normalizer = VahadaneNormalizer(**kwargs)
        else:
            raise ValueError(
                f"Unknown normalization method: {method}. " "Supported: reinhard, macenko, vahadane"
            )

        # Fit to default TCGA target parameters
        normalizer.set_target_params(STAIN_NORM_TARGETS["tcga_avg"])

        normalized_images = np.empty_like(self._images)
        for i, patch in enumerate(self._images):
            try:
                normalized_images[i] = normalizer.transform(patch)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                logger.debug("Normalization failed for patch %d: %s", i, exc)
                normalized_images[i] = patch

        child = Patches(
            normalized_images,
            self._coordinates.copy(),
            metadata={**self._metadata, "stain_normalization": method_lower},
        )
        return child

    def separate_stains(self, method: str = "hed") -> Dict[str, np.ndarray]:
        """Separate stains for every patch.

        Parameters
        ----------
        method : str
            Separation method forwarded to
            :class:`~honeybee.processors.wsi.stain_separation.StainSeparator`.

        Returns
        -------
        dict
            Keys are stain names (``"hematoxylin"``, ``"eosin"``,
            ``"background"``).  Values are arrays of shape ``(N, H, W)``.
        """
        from .stain_separation import StainSeparator

        separator = StainSeparator(method=method)

        n = len(self._images)
        if n == 0:
            return {
                "hematoxylin": np.empty((0,), dtype=np.float64),
                "eosin": np.empty((0,), dtype=np.float64),
                "background": np.empty((0,), dtype=np.float64),
            }

        h_channels: list = []
        e_channels: list = []
        bg_channels: list = []

        for patch in self._images:
            result = separator.separate(patch)
            h_channels.append(result["hematoxylin"])
            e_channels.append(result["eosin"])
            bg_channels.append(result.get("background", np.zeros_like(result["hematoxylin"])))

        return {
            "hematoxylin": np.stack(h_channels),
            "eosin": np.stack(e_channels),
            "background": np.stack(bg_channels),
        }

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def plot_gallery(
        self,
        cols: int = 8,
        max_patches: int = 64,
        ax: Optional["matplotlib.axes.Axes"] = None,
        interactive: bool = False,
    ) -> "matplotlib.figure.Figure":
        """Display a grid of patch thumbnails.

        Parameters
        ----------
        cols : int
            Number of columns in the grid.
        max_patches : int
            Maximum number of patches to display.
        ax : matplotlib.axes.Axes, optional
            If provided, draw onto this axes (``cols`` is ignored and a
            single-axes layout is assumed).  Otherwise a new figure is
            created.
        interactive : bool
            If ``True``, call ``plt.show()`` after drawing.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure containing the gallery.
        """
        import matplotlib.pyplot as plt

        created_fig = ax is None
        n = min(len(self), max_patches)
        if n == 0:
            fig, single_ax = plt.subplots(1, 1, figsize=(4, 4))
            single_ax.text(0.5, 0.5, "No patches", ha="center", va="center")
            single_ax.set_axis_off()
            if interactive:
                plt.show()
            if created_fig:
                plt.close(fig)
            return fig

        rows = int(np.ceil(n / cols))

        if ax is not None:
            # Draw a composite image onto the provided axes
            fig = ax.get_figure()
            patch_h, patch_w = self._images.shape[1], self._images.shape[2]
            canvas = np.ones((rows * patch_h, cols * patch_w, 3), dtype=np.uint8) * 255
            for i in range(n):
                r, c = divmod(i, cols)
                canvas[
                    r * patch_h : (r + 1) * patch_h,
                    c * patch_w : (c + 1) * patch_w,
                ] = self._images[i]
            ax.imshow(canvas)
            ax.set_axis_off()
            ax.set_title(f"Patches ({n}/{len(self)})")
        else:
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5), squeeze=False)
            for i in range(rows * cols):
                r, c = divmod(i, cols)
                if i < n:
                    axes[r][c].imshow(self._images[i])
                axes[r][c].set_axis_off()
            fig.suptitle(f"Patches ({n}/{len(self)})", fontsize=12)
            fig.tight_layout()

        if interactive:
            plt.show()

        if created_fig:
            plt.close(fig)
        return fig

    def plot_on_slide(
        self,
        slide,
        alpha: float = 0.3,
        color: str = "cyan",
        interactive: bool = False,
        ax: Optional["matplotlib.axes.Axes"] = None,
        thumbnail_size: Tuple[int, int] = (2048, 2048),
    ) -> "matplotlib.figure.Figure":
        """Overlay patch locations on a slide thumbnail.

        Parameters
        ----------
        slide : Slide
            A loaded Slide object.  The method calls
            ``slide.get_thumbnail()`` if available, otherwise falls back to
            reading ``slide.slide`` and down-sampling.
        alpha : float
            Rectangle transparency.
        color : str
            Rectangle edge colour (any matplotlib colour string).
        interactive : bool
            If ``True``, call ``plt.show()`` after drawing.
        ax : matplotlib.axes.Axes, optional
            Draw onto an existing axes instead of creating a new figure.
        thumbnail_size : tuple of int
            Target ``(width, height)`` for the slide thumbnail.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure with the overlay.
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        # Obtain thumbnail
        thumbnail = self._get_slide_thumbnail(slide, size=thumbnail_size)
        thumb_h, thumb_w = thumbnail.shape[:2]

        # Determine full-resolution slide dimensions for coordinate scaling
        slide_w, slide_h = self._get_slide_dimensions(slide)

        scale_x = thumb_w / slide_w if slide_w > 0 else 1.0
        scale_y = thumb_h / slide_h if slide_h > 0 else 1.0

        created_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        else:
            fig = ax.get_figure()

        ax.imshow(thumbnail)

        for coord in self._coordinates:
            x, y, w, h = coord
            rect = mpatches.Rectangle(
                (x * scale_x, y * scale_y),
                w * scale_x,
                h * scale_y,
                linewidth=1,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)

        ax.set_axis_off()
        ax.set_title(f"Patch locations ({len(self)} patches)")

        if interactive:
            plt.show()

        if created_fig:
            plt.close(fig)
        return fig

    def plot_quality_distribution(
        self,
        bins: int = 30,
        threshold: Optional[float] = None,
        figsize: Tuple[int, int] = (8, 4),
        ax: Optional["matplotlib.axes.Axes"] = None,
    ) -> "matplotlib.figure.Figure":
        """Histogram of per-patch quality scores.

        Parameters
        ----------
        bins : int
            Number of histogram bins.
        threshold : float, optional
            If set, draws a vertical line and annotates keep/drop counts.
        figsize : tuple
            Figure size when *ax* is ``None``.
        ax : matplotlib Axes, optional
            Draw onto existing axes.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        scores = self.quality_scores

        created_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.hist(scores, bins=bins, edgecolor="black")
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Count")
        ax.set_title("Patch Quality Distribution")

        if threshold is not None:
            ax.axvline(threshold, color="red", linestyle="--", linewidth=2)
            n_keep = int(np.sum(scores >= threshold))
            n_drop = len(scores) - n_keep
            ax.text(
                threshold + 0.02,
                ax.get_ylim()[1] * 0.9,
                f"keep={n_keep}\ndrop={n_drop}",
                fontsize=10,
                color="red",
            )

        if created_fig:
            plt.close(fig)
        return fig

    def plot_normalization_comparison(
        self,
        patch_index: Optional[int] = None,
        methods: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
    ) -> "matplotlib.figure.Figure":
        """Show one patch under original + each normalization method.

        Parameters
        ----------
        patch_index : int, optional
            Index of the patch to display.  Defaults to the best-quality patch.
        methods : list of str, optional
            Normalization methods.  Defaults to ``["reinhard", "macenko", "vahadane"]``.
        figsize : tuple, optional
            Figure size.
        ax : matplotlib Axes, optional
            Not commonly used (creates multi-axes internally).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if methods is None:
            methods = ["reinhard", "macenko", "vahadane"]

        if patch_index is None:
            patch_index = int(self.quality_scores.argmax())

        n_panels = 1 + len(methods)

        if figsize is None:
            figsize = (4 * n_panels, 4)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)

        # Original
        axes[0].imshow(self._images[patch_index])
        axes[0].set_title("Original")
        axes[0].axis("off")

        # Each normalization method
        single = self[patch_index]
        for i, method in enumerate(methods):
            try:
                normed = single.normalize(method=method)
                axes[i + 1].imshow(normed.images[0])
            except (ValueError, np.linalg.LinAlgError):
                axes[i + 1].imshow(self._images[patch_index])
            axes[i + 1].set_title(method.capitalize())
            axes[i + 1].axis("off")

        fig.suptitle("Stain Normalization Comparison")
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_normalization_before_after(
        self,
        normalized: "Patches",
        n_show: int = 8,
        figsize: Optional[Tuple[int, int]] = None,
        ax: Optional["matplotlib.axes.Axes"] = None,
    ) -> "matplotlib.figure.Figure":
        """2-row grid: top = original, bottom = normalized.

        Parameters
        ----------
        normalized : Patches
            The normalized version of this ``Patches`` object.
        n_show : int
            Number of patches to display.
        figsize : tuple, optional
            Figure size.
        ax : matplotlib Axes, optional
            Not commonly used.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        n_show = min(n_show, len(self), len(normalized))
        if n_show == 0:
            fig, single_ax = plt.subplots(1, 1, figsize=(4, 4))
            single_ax.text(0.5, 0.5, "No patches", ha="center", va="center")
            single_ax.set_axis_off()
            plt.close(fig)
            return fig

        if figsize is None:
            figsize = (2 * n_show, 4)
        fig, axes = plt.subplots(2, n_show, figsize=figsize, squeeze=False)

        for i in range(n_show):
            axes[0, i].imshow(self._images[i])
            axes[0, i].axis("off")
            axes[1, i].imshow(normalized.images[i])
            axes[1, i].axis("off")

        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel("Normalized")
        fig.suptitle("Batch Stain Normalization")
        fig.tight_layout()
        plt.close(fig)
        return fig

    def plot_stain_separation(
        self,
        patch_index: Optional[int] = None,
        method: str = "hed",
        figsize: Tuple[int, int] = (16, 4),
        ax: Optional["matplotlib.axes.Axes"] = None,
    ) -> "matplotlib.figure.Figure":
        """4-panel: original + 3 stain channels (hematoxylin, eosin, background).

        Parameters
        ----------
        patch_index : int, optional
            Index of the patch.  Defaults to the best-quality patch.
        method : str
            Stain separation method.
        figsize : tuple
            Figure size.
        ax : matplotlib Axes, optional
            Not commonly used.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if patch_index is None:
            patch_index = int(self.quality_scores.argmax())

        single = self[patch_index]
        stains = single.separate_stains(method=method)

        fig, axes = plt.subplots(1, 4, figsize=figsize)

        axes[0].imshow(self._images[patch_index])
        axes[0].set_title("Original")
        axes[0].axis("off")

        for ax_i, (key, title) in zip(
            axes[1:],
            [("hematoxylin", "Hematoxylin"), ("eosin", "Eosin"), ("background", "Background")],
        ):
            ax_i.imshow(stains[key][0], cmap="gray")
            ax_i.set_title(title)
            ax_i.axis("off")

        fig.suptitle("Stain Separation")
        fig.tight_layout()
        plt.close(fig)
        return fig

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def to_records(
        self,
        embeddings: Optional[np.ndarray] = None,
        include_images: bool = False,
    ) -> List[Dict]:
        """Return per-patch data as a list of dicts.

        Each record contains:

        - ``'index'``: int -- position in this collection
        - ``'x'``, ``'y'``, ``'width'``, ``'height'``: level-0 coordinates
        - ``'quality_score'``: float (only if quality scores have been computed)
        - ``'embedding'``: np.ndarray (only if *embeddings* is provided)
        - ``'image'``: np.ndarray (only if *include_images* is ``True``)

        Plus all keys from :attr:`metadata`.

        Parameters
        ----------
        embeddings : np.ndarray, optional
            Embedding matrix of shape ``(N, D)``.  If provided, each record
            gets an ``'embedding'`` key with the corresponding row.
        include_images : bool
            If ``True``, include the patch image array in each record.

        Returns
        -------
        list of dict
        """
        if embeddings is not None:
            embeddings = np.asarray(embeddings)
            if len(embeddings) != len(self):
                raise ValueError(
                    f"embeddings length ({len(embeddings)}) != "
                    f"number of patches ({len(self)})"
                )

        has_quality = self._quality_scores is not None

        records: List[Dict] = []
        for i in range(len(self)):
            rec: Dict = {
                "index": i,
                "x": int(self._coordinates[i, 0]),
                "y": int(self._coordinates[i, 1]),
                "width": int(self._coordinates[i, 2]),
                "height": int(self._coordinates[i, 3]),
            }
            if has_quality:
                rec["quality_score"] = float(self._quality_scores[i])
            if embeddings is not None:
                rec["embedding"] = embeddings[i]
            if include_images:
                rec["image"] = self._images[i]
            # Merge collection-level metadata
            rec.update(self._metadata)
            records.append(rec)

        return records

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #

    def save(
        self,
        output_dir: str,
        prefix: str = "patch",
        format: str = "png",
    ) -> List[str]:
        """Save patches as individual image files.

        Parameters
        ----------
        output_dir : str
            Directory to write files into (created if absent).
        prefix : str
            Filename prefix.
        format : str
            Image file extension / format (e.g. ``"png"``, ``"jpg"``).

        Returns
        -------
        list of str
            Absolute paths of saved files.
        """
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        saved_paths: List[str] = []
        for i, patch in enumerate(self._images):
            fname = f"{prefix}_{i:05d}.{format}"
            fpath = os.path.join(output_dir, fname)
            Image.fromarray(patch.astype(np.uint8)).save(fpath)
            saved_paths.append(os.path.abspath(fpath))
        return saved_paths

    def to_numpy(self) -> np.ndarray:
        """Return the patch images as a numpy array.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, H, W, 3)`` with dtype ``uint8``.
        """
        return self._images

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_slide_thumbnail(slide, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Best-effort thumbnail extraction from a Slide-like object."""
        # Try the standard get_thumbnail method first
        if callable(getattr(slide, "get_thumbnail", None)):
            thumb = slide.get_thumbnail(size=size) if size is not None else slide.get_thumbnail()
            return np.asarray(thumb)[:, :, :3]

        # Fall back to reading the slide property at low resolution
        if hasattr(slide, "img") and hasattr(slide.img, "resolutions"):
            try:
                levels = slide.img.resolutions["level_dimensions"]
                # Pick the lowest-resolution level
                best_level = len(levels) - 1
                region = slide.img.read_region(location=[0, 0], level=best_level)
                image = np.asarray(region)
                return image[:, :, :3] if image.ndim == 3 and image.shape[2] > 3 else image
            except (IndexError, ValueError, OSError, AttributeError):
                pass

        # Last resort: use slide.slide
        if hasattr(slide, "slide"):
            arr = np.asarray(slide.slide)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            return arr

        raise ValueError("Cannot extract a thumbnail from the provided slide object")

    @staticmethod
    def _get_slide_dimensions(slide) -> tuple:
        """Return (width, height) of the slide at level 0."""
        # Fast-path: use slide.dimensions directly (new-style Slide API)
        dims = getattr(slide, "dimensions", None)
        if dims is not None and len(dims) >= 2:
            try:
                return (int(dims[0]), int(dims[1]))
            except (TypeError, ValueError):
                pass

        # CuImage-backed Slide stores dimensions on slide.slide
        if hasattr(slide, "slide"):
            s = slide.slide
            width = getattr(s, "width", None)
            height = getattr(s, "height", None)
            if width is not None and height is not None:
                return (int(width), int(height))

        # Try resolutions dict (CuImage)
        if hasattr(slide, "img") and hasattr(slide.img, "resolutions"):
            try:
                dims = slide.img.resolutions["level_dimensions"][0]
                return (int(dims[0]), int(dims[1]))
            except (KeyError, IndexError):
                pass

        # Fallback: use the thumbnail shape itself (coordinates won't scale)
        warnings.warn(
            "Could not determine slide dimensions; coordinate scaling will be approximate.",
            stacklevel=2,
        )
        return (1, 1)
