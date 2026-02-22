"""
PatchExtractor -- grid-based patch extraction from whole-slide images.

Sits between :class:`~honeybee.loaders.Slide.slide.Slide` (loading / tissue
detection) and downstream embedding or analysis code.  Reads patches at a
chosen pyramid level (or magnification) and returns a rich
:class:`~honeybee.processors.wsi.patches.Patches` container.

Typical usage::

    from honeybee.loaders.Slide.slide import Slide
    from honeybee.processors.wsi.patch_extractor import PatchExtractor

    slide = Slide("tumor.svs")
    slide.detect_tissue()

    extractor = PatchExtractor(patch_size=256, stride=256, min_tissue_ratio=0.5)
    patches = extractor.extract(slide)       # Returns Patches object
    coords  = extractor.get_coordinates(slide)  # Fast preview (no pixel reads)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from honeybee.loaders.Slide.slide import Slide

    from .patches import Patches

logger = logging.getLogger(__name__)

__all__ = ["PatchExtractor"]


class PatchExtractor:
    """Grid-based patch extraction from whole-slide images.

    Parameters
    ----------
    patch_size : int
        Side length of each extracted patch **at the target level**, in pixels.
    stride : int or None
        Step size between consecutive patches at the target level.
        ``None`` (default) means ``stride = patch_size`` (no overlap).
        A value smaller than *patch_size* produces overlapping patches.
    magnification : float or None
        If set, the extractor picks the pyramid level closest to this
        objective magnification (e.g. ``20.0``).  Mutually exclusive with
        *level* -- when both are given, *magnification* takes precedence.
    level : int or None
        Pyramid level to extract from (0 = highest resolution).  Ignored
        when *magnification* is set.  Defaults to ``0``.
    min_tissue_ratio : float
        Minimum fraction of a candidate patch region that must overlap
        tissue (according to the slide's tissue mask) for the patch to be
        kept.  Only effective when the slide carries a tissue mask.
    compute_quality : bool
        Reserved for future use (quality scoring is available lazily on
        the returned :class:`Patches` object).
    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: Optional[int] = None,
        magnification: Optional[float] = None,
        level: Optional[int] = None,
        min_tissue_ratio: float = 0.5,
        compute_quality: bool = True,
    ) -> None:
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if stride is not None and stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if not 0.0 <= min_tissue_ratio <= 1.0:
            raise ValueError(f"min_tissue_ratio must be in [0, 1], got {min_tissue_ratio}")

        self.patch_size: int = patch_size
        self.stride: int = stride if stride is not None else patch_size
        self.magnification: Optional[float] = magnification
        self.level: int = level if level is not None else 0
        self.min_tissue_ratio: float = min_tissue_ratio
        self.compute_quality: bool = compute_quality

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        slide: "Slide",
        tissue_coordinates: Optional[np.ndarray] = None,
    ) -> "Patches":
        """Extract patches from a :class:`Slide` object.

        Steps
        -----
        1. Determine the target pyramid level.
        2. Compute a grid of candidate patch coordinates in level-0 space.
        3. Filter candidates by *tissue_coordinates* (if given) **or** the
           slide's tissue mask (if present).
        4. Read surviving patches via ``slide.read_region``.
        5. Return a :class:`Patches` container.

        Parameters
        ----------
        slide : Slide
            An opened slide object (optionally with a tissue mask set via
            ``slide.detect_tissue()``).
        tissue_coordinates : np.ndarray, optional
            Pre-computed tissue coordinates from a different extraction pass,
            shape ``(M, 4)`` with columns ``[x, y, w, h]`` in level-0 space.
            When provided, these take precedence over the slide's tissue mask.

        Returns
        -------
        Patches
            Container with images, level-0 coordinates, and metadata.
        """
        from .patches import Patches

        target_level = self._resolve_level(slide)
        coordinates = self._build_grid(slide, target_level)

        if len(coordinates) == 0:
            logger.info("No candidate patch coordinates generated for %s", slide.path)
            return Patches(
                images=np.empty((0, self.patch_size, self.patch_size, 3), dtype=np.uint8),
                coordinates=np.empty((0, 4), dtype=np.int64),
                metadata=self._build_metadata(slide, target_level, tissue_coordinates),
            )

        # Filter by tissue_coordinates (priority) or tissue mask
        if tissue_coordinates is not None:
            coordinates = self._filter_by_coordinates(
                coordinates, tissue_coordinates, slide.dimensions
            )
        elif slide.tissue_mask is not None:
            coordinates = self._filter_by_tissue(slide, coordinates)

        if len(coordinates) == 0:
            logger.info("All patches filtered out for %s", slide.path)
            return Patches(
                images=np.empty((0, self.patch_size, self.patch_size, 3), dtype=np.uint8),
                coordinates=np.empty((0, 4), dtype=np.int64),
                metadata=self._build_metadata(slide, target_level, tissue_coordinates),
            )

        # Read patches
        patch_images = self._read_patches(slide, coordinates, target_level)

        return Patches(
            images=np.array(patch_images, dtype=np.uint8),
            coordinates=coordinates,
            metadata=self._build_metadata(slide, target_level, tissue_coordinates),
        )

    def get_coordinates(
        self,
        slide: "Slide",
        tissue_coordinates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return patch coordinates without reading pixel data.

        This is useful for a fast preview of what will be extracted, or for
        visualising patch locations on a slide thumbnail before committing
        to the (potentially slow) pixel-read step.

        Parameters
        ----------
        slide : Slide
            An opened slide object.
        tissue_coordinates : np.ndarray, optional
            Pre-computed tissue coordinates from a different extraction pass,
            shape ``(M, 4)`` with columns ``[x, y, w, h]`` in level-0 space.
            When provided, these take precedence over the slide's tissue mask.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 4)`` with columns ``[x, y, w, h]`` in
            level-0 coordinates.  Empty ``(0, 4)`` if no coordinates survive
            filtering.
        """
        target_level = self._resolve_level(slide)
        coordinates = self._build_grid(slide, target_level)
        if len(coordinates) == 0:
            return np.empty((0, 4), dtype=np.int64)

        if tissue_coordinates is not None:
            coordinates = self._filter_by_coordinates(
                coordinates, tissue_coordinates, slide.dimensions
            )
        else:
            coordinates = self._filter_by_tissue(slide, coordinates)

        return coordinates

    def plot_grid_preview(
        self,
        slide: "Slide",
        thumbnail_size: tuple = (2048, 2048),
        zoom_size: int = 1500,
        zoom_display: int = 600,
        figsize: tuple = (14, 12),
        ax=None,
    ):
        """2x2 preview: tissue mask, coverage overlay, zoomed boundary, statistics.

        Parameters
        ----------
        slide : Slide
            An opened slide with tissue mask.
        thumbnail_size : tuple
            Thumbnail resolution for overview panels.
        zoom_size : int
            Side length of the zoom crop in slide coordinates.
        zoom_display : int
            Pixel size of the rendered zoom panel.
        figsize : tuple
            Figure size.
        ax : array of Axes, optional
            Pre-created 2x2 axes.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle as MplRect

        coords = self.get_coordinates(slide)
        thumb = slide.get_thumbnail(size=thumbnail_size)
        thumb_h, thumb_w = thumb.shape[:2]
        slide_w, slide_h = slide.dimensions
        scale_x = thumb_w / slide_w
        scale_y = thumb_h / slide_h

        tissue_mask = slide.tissue_mask
        if tissue_mask is None:
            tissue_mask = np.ones((thumb_h, thumb_w), dtype=bool)
        elif tissue_mask.shape[:2] != (thumb_h, thumb_w):
            tissue_mask = cv2.resize(
                tissue_mask.astype(np.uint8), (thumb_w, thumb_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # --- Rasterize boolean coverage at thumbnail resolution ---
        coverage = np.zeros((thumb_h, thumb_w), dtype=bool)
        if len(coords) > 0:
            x1 = np.clip((coords[:, 0] * scale_x).astype(int), 0, thumb_w - 1)
            y1 = np.clip((coords[:, 1] * scale_y).astype(int), 0, thumb_h - 1)
            x2 = np.clip(((coords[:, 0] + coords[:, 2]) * scale_x).astype(int), 0, thumb_w)
            y2 = np.clip(((coords[:, 1] + coords[:, 3]) * scale_y).astype(int), 0, thumb_h)
            for i in range(len(coords)):
                coverage[y1[i]:y2[i], x1[i]:x2[i]] = True

        # --- Coverage statistics ---
        tissue_pixels = tissue_mask.sum()
        covered_tissue = (coverage & tissue_mask).sum()
        missed_tissue = (tissue_mask & ~coverage).sum()
        tissue_coverage_pct = 100.0 * covered_tissue / max(tissue_pixels, 1)
        missed_pct = 100.0 * missed_tissue / max(tissue_pixels, 1)

        # --- RGBA overlay ---
        overlay = np.zeros((thumb_h, thumb_w, 4), dtype=np.uint8)
        overlay[coverage & tissue_mask] = [0, 255, 255, 90]
        overlay[tissue_mask & ~coverage] = [255, 80, 80, 120]

        thumb_f = thumb.astype(np.float32) / 255.0
        overlay_f = overlay.astype(np.float32) / 255.0
        alpha = overlay_f[:, :, 3:4]
        composite = (1.0 - alpha) * thumb_f + alpha * overlay_f[:, :, :3]
        composite = np.clip(composite, 0, 1)

        # --- Find tissue boundary center for zoom ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tissue_u8 = tissue_mask.astype(np.uint8)
        dilated = cv2.dilate(tissue_u8, kernel)
        eroded = cv2.erode(tissue_u8, kernel)
        boundary_band = (dilated - eroded).astype(bool)

        boundary_ys, boundary_xs = np.where(boundary_band & tissue_mask)
        if len(boundary_ys) > 0:
            cy_thumb = int(np.median(boundary_ys))
            cx_thumb = int(np.median(boundary_xs))
        else:
            cy_thumb, cx_thumb = thumb_h // 2, thumb_w // 2

        cx_slide = int(cx_thumb / scale_x)
        cy_slide = int(cy_thumb / scale_y)

        # --- Define crop in slide coordinates ---
        crop_half = zoom_size // 2
        crop_x1 = max(0, cx_slide - crop_half)
        crop_y1 = max(0, cy_slide - crop_half)
        crop_x2 = min(slide_w, cx_slide + crop_half)
        crop_y2 = min(slide_h, cy_slide + crop_half)
        crop_x1 = max(0, crop_x2 - zoom_size) if crop_x2 - crop_x1 < zoom_size else crop_x1
        crop_y1 = max(0, crop_y2 - zoom_size) if crop_y2 - crop_y1 < zoom_size else crop_y1
        crop_w_slide = crop_x2 - crop_x1
        crop_h_slide = crop_y2 - crop_y1

        zoom_scale_x = zoom_display / max(crop_w_slide, 1)
        zoom_scale_y = zoom_display / max(crop_h_slide, 1)

        # --- Get thumbnail crop for zoom ---
        tz_x1 = max(0, int(crop_x1 * scale_x))
        tz_y1 = max(0, int(crop_y1 * scale_y))
        tz_x2 = min(thumb_w, int(crop_x2 * scale_x))
        tz_y2 = min(thumb_h, int(crop_y2 * scale_y))

        thumb_crop = thumb[tz_y1:tz_y2, tz_x1:tz_x2]
        if thumb_crop.size > 0:
            zoom_bg = cv2.resize(thumb_crop, (zoom_display, zoom_display))
        else:
            zoom_bg = np.zeros((zoom_display, zoom_display, 3), dtype=np.uint8)

        mask_crop = tissue_mask[tz_y1:tz_y2, tz_x1:tz_x2]
        if mask_crop.size > 0:
            mask_resized = cv2.resize(
                mask_crop.astype(np.uint8), (zoom_display, zoom_display),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask_resized = np.zeros((zoom_display, zoom_display), dtype=bool)

        # --- Filter patches in crop box ---
        if len(coords) > 0:
            px1, py1 = coords[:, 0], coords[:, 1]
            px2, py2 = coords[:, 0] + coords[:, 2], coords[:, 1] + coords[:, 3]
            in_crop = (px2 > crop_x1) & (px1 < crop_x2) & (py2 > crop_y1) & (py1 < crop_y2)
            crop_coords = coords[in_crop]
        else:
            crop_coords = np.empty((0, 4), dtype=np.int64)

        # --- Build patch_coverage mask at zoom resolution ---
        patch_coverage = np.zeros((zoom_display, zoom_display), dtype=bool)
        for c in crop_coords:
            rx1 = max(0, int((c[0] - crop_x1) * zoom_scale_x))
            ry1 = max(0, int((c[1] - crop_y1) * zoom_scale_y))
            rx2 = min(zoom_display, int((c[0] + c[2] - crop_x1) * zoom_scale_x))
            ry2 = min(zoom_display, int((c[1] + c[3] - crop_y1) * zoom_scale_y))
            if rx2 > rx1 and ry2 > ry1:
                patch_coverage[ry1:ry2, rx1:rx2] = True

        # --- Three-zone rendering ---
        zoom_canvas = zoom_bg.copy().astype(np.float32)
        zoom_canvas[~mask_resized] = (
            zoom_canvas[~mask_resized] * 0.3
            + np.array([80, 0, 0], dtype=np.float32) * 0.7
        )
        missed = mask_resized & ~patch_coverage
        zoom_canvas[missed] = (
            zoom_canvas[missed] * 0.25
            + np.array([200, 50, 50], dtype=np.float32) * 0.75
        )
        zoom_canvas = np.clip(zoom_canvas, 0, 255).astype(np.uint8)

        coverage_tint = np.zeros_like(zoom_canvas)
        coverage_tint[patch_coverage] = [0, 60, 60]
        zoom_canvas = cv2.add(zoom_canvas, coverage_tint)

        # Contours
        tissue_contours, _ = cv2.findContours(
            mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(zoom_canvas, tissue_contours, -1, (255, 50, 50), 1)
        coverage_contours, _ = cv2.findContours(
            patch_coverage.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(zoom_canvas, coverage_contours, -1, (0, 255, 100), 3)

        # Patch rectangles
        patch_display_px = self.patch_size * zoom_scale_x
        line_thick = 1 if patch_display_px < 15 else 2
        for c in crop_coords:
            rx1 = max(0, int((c[0] - crop_x1) * zoom_scale_x))
            ry1 = max(0, int((c[1] - crop_y1) * zoom_scale_y))
            rx2 = min(zoom_display, int((c[0] + c[2] - crop_x1) * zoom_scale_x))
            ry2 = min(zoom_display, int((c[1] + c[3] - crop_y1) * zoom_scale_y))
            if rx2 > rx1 and ry2 > ry1:
                cv2.rectangle(zoom_canvas, (rx1, ry1), (rx2, ry2), (0, 255, 255), line_thick)

        # --- Statistics ---
        mpp = slide.mpp if slide.mpp else 1.0
        patch_mm = self.patch_size * mpp / 1000.0
        slide_area_mm2 = (slide_w * mpp / 1000.0) * (slide_h * mpp / 1000.0)
        tissue_area_mm2 = slide_area_mm2 * tissue_mask.mean()
        density = len(coords) / tissue_area_mm2 if tissue_area_mm2 > 0 else 0

        zoom_tissue_px = int(mask_resized.sum())
        zoom_coverage_px = int((patch_coverage & mask_resized).sum())
        zoom_cov_pct = 100.0 * zoom_coverage_px / max(zoom_tissue_px, 1)

        # --- 2x2 figure ---
        created_fig = ax is None
        if ax is not None:
            axes = np.asarray(ax).reshape(2, 2)
            fig = axes[0, 0].figure
        else:
            fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Top-left: tissue mask
        axes[0, 0].imshow(tissue_mask, cmap="gray")
        axes[0, 0].set_title(f"Tissue Mask ({tissue_mask.mean():.1%} tissue)")
        axes[0, 0].axis("off")

        # Top-right: coverage overlay + zoom rectangle
        axes[0, 1].imshow(composite)
        axes[0, 1].set_title(f"Coverage: {tissue_coverage_pct:.1f}% tissue covered")
        rect_x1 = crop_x1 * scale_x
        rect_y1 = crop_y1 * scale_y
        rect_x2 = crop_x2 * scale_x
        rect_y2 = crop_y2 * scale_y
        rect = MplRect(
            (rect_x1, rect_y1), rect_x2 - rect_x1, rect_y2 - rect_y1,
            linewidth=2, edgecolor="yellow", facecolor="none", linestyle="--",
        )
        axes[0, 1].add_patch(rect)
        axes[0, 1].axis("off")

        # Bottom-left: zoom panel
        axes[1, 0].imshow(zoom_canvas)
        axes[1, 0].set_title(
            f"Zoom: patch={self.patch_size}px | "
            f"bright=covered, red=missed, dark=background"
        )
        axes[1, 0].text(
            0.02, 0.98, f"Coverage: {zoom_cov_pct:.1f}%",
            transform=axes[1, 0].transAxes, fontsize=12,
            color="white", fontweight="bold", va="top",
            bbox=dict(facecolor="black", alpha=0.6, pad=3),
        )
        axes[1, 0].axis("off")

        # Bottom-right: statistics
        axes[1, 1].axis("off")
        stats_text = (
            f"Configuration\n"
            f"{'─' * 25}\n"
            f"  PATCH_SIZE:       {self.patch_size}x{self.patch_size} px\n"
            f"  STRIDE:           {self.stride} px\n"
            f"  MIN_TISSUE_RATIO: {self.min_tissue_ratio}\n"
            f"\n"
            f"Results\n"
            f"{'─' * 25}\n"
            f"  Total patches:       {len(coords):,}\n"
            f"  Tissue coverage:     {tissue_coverage_pct:.1f}%\n"
            f"  Missed tissue:       {missed_pct:.1f}%\n"
            f"\n"
            f"Grid Density\n"
            f"{'─' * 25}\n"
            f"  Patch size:          {patch_mm:.3f} mm\n"
            f"  Patches/mm\u00b2:         {density:.1f}\n"
            f"  Tissue area:         {tissue_area_mm2:.1f} mm\u00b2\n"
            f"\n"
            f"Zoom Panel\n"
            f"{'─' * 25}\n"
            f"  Patches in view: {len(crop_coords):,}\n"
            f"  Zoom coverage: {zoom_cov_pct:.1f}%"
        )
        axes[1, 1].text(
            0.05, 0.95, stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9),
        )
        axes[1, 1].set_title("Grid Statistics")

        fig.suptitle(
            f"Patch Grid Preview \u2014 patch_size={self.patch_size}  |  "
            f"{len(coords):,} patches  |  {density:.0f} patches/mm\u00b2",
            fontsize=14,
        )
        fig.tight_layout()
        if created_fig:
            plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_level(self, slide: "Slide") -> int:
        """Determine the target pyramid level."""
        if self.magnification is not None:
            return slide.get_best_level_for_magnification(self.magnification)
        return self.level

    def _build_grid(self, slide: "Slide", target_level: int) -> np.ndarray:
        """Compute candidate patch coordinates in level-0 space.

        Each row is ``[x, y, patch_size_l0, patch_size_l0]`` where
        ``patch_size_l0`` is the patch footprint mapped back to level 0.
        """
        slide_w, slide_h = slide.dimensions  # level-0 (width, height)

        downsamples = slide.level_downsamples
        if target_level >= len(downsamples):
            logger.warning(
                "Requested level %d but slide only has %d levels; " "falling back to last level.",
                target_level,
                len(downsamples),
            )
            target_level = len(downsamples) - 1

        downsample = downsamples[target_level]

        # Patch footprint and stride in level-0 pixel space
        patch_size_l0 = int(self.patch_size * downsample)
        stride_l0 = int(self.stride * downsample)

        if patch_size_l0 <= 0 or stride_l0 <= 0:
            return np.empty((0, 4), dtype=np.int64)

        # Generate grid -- only emit patches whose origin allows a full
        # patch_size_l0 footprint to fit within the slide.
        xs = np.arange(0, slide_w - patch_size_l0 + 1, stride_l0, dtype=np.int64)
        ys = np.arange(0, slide_h - patch_size_l0 + 1, stride_l0, dtype=np.int64)

        if len(xs) == 0 or len(ys) == 0:
            # Slide is smaller than a single patch -- still extract one
            # patch starting at (0, 0) if the slide has any area at all.
            if slide_w > 0 and slide_h > 0:
                return np.array([[0, 0, patch_size_l0, patch_size_l0]], dtype=np.int64)
            return np.empty((0, 4), dtype=np.int64)

        # Meshgrid -> (N, 4) coordinate array
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        grid_x = grid_x.ravel()
        grid_y = grid_y.ravel()

        coords = np.column_stack(
            [
                grid_x,
                grid_y,
                np.full_like(grid_x, patch_size_l0),
                np.full_like(grid_y, patch_size_l0),
            ]
        )
        return coords

    def _filter_by_coordinates(
        self,
        coordinates: np.ndarray,
        tissue_coordinates: np.ndarray,
        slide_dimensions: tuple,
    ) -> np.ndarray:
        """Filter candidates by overlap with pre-computed tissue coordinates.

        Rasterises *tissue_coordinates* into a binary mask at an internal
        resolution (up to 4096x4096), then checks each candidate patch for
        sufficient overlap — identical in spirit to :meth:`_filter_by_tissue`
        but without boundary erosion (the tissue regions are already
        precisely defined by the coordinates).

        Parameters
        ----------
        coordinates : np.ndarray
            Candidate patches, shape ``(N, 4)`` in level-0 space.
        tissue_coordinates : np.ndarray
            Tissue patch coordinates, shape ``(M, 4)`` in level-0 space.
        slide_dimensions : tuple
            ``(width, height)`` of the slide at level 0.
        """
        tissue_coordinates = np.asarray(tissue_coordinates)
        if tissue_coordinates.ndim != 2 or tissue_coordinates.shape[1] != 4:
            raise ValueError(
                f"tissue_coordinates must have shape (M, 4), got {tissue_coordinates.shape}"
            )

        if len(tissue_coordinates) == 0:
            logger.debug("Empty tissue_coordinates — all patches filtered out")
            return np.empty((0, 4), dtype=np.int64)

        slide_w, slide_h = slide_dimensions

        # Internal rasterisation resolution (cap at 4096 to limit memory)
        max_res = 4096
        if slide_w >= slide_h:
            mask_w = min(max_res, slide_w)
            mask_h = max(1, int(mask_w * slide_h / slide_w))
        else:
            mask_h = min(max_res, slide_h)
            mask_w = max(1, int(mask_h * slide_w / slide_h))

        scale_x = mask_w / slide_w
        scale_y = mask_h / slide_h

        # Rasterise tissue_coordinates into a boolean mask
        mask = np.zeros((mask_h, mask_w), dtype=bool)
        for tc in tissue_coordinates:
            x, y, w, h = tc
            mx0 = max(0, int(x * scale_x))
            my0 = max(0, int(y * scale_y))
            mx1 = min(mask_w, int((x + w) * scale_x) + 1)
            my1 = min(mask_h, int((y + h) * scale_y) + 1)
            if mx1 > mx0 and my1 > my0:
                mask[my0:my1, mx0:mx1] = True

        # Check each candidate against the rasterised mask
        keep: List[bool] = []
        for coord in coordinates:
            x, y, w, h = coord
            mx0 = max(0, int(x * scale_x))
            my0 = max(0, int(y * scale_y))
            mx1 = min(mask_w, int((x + w) * scale_x) + 1)
            my1 = min(mask_h, int((y + h) * scale_y) + 1)

            if mx1 <= mx0 or my1 <= my0:
                keep.append(False)
                continue

            region = mask[my0:my1, mx0:mx1]
            tissue_ratio = float(np.mean(region.astype(np.float32)))
            keep.append(tissue_ratio >= self.min_tissue_ratio)

        keep_arr = np.array(keep, dtype=bool)
        filtered = coordinates[keep_arr]

        logger.debug(
            "Tissue-coordinate filtering: %d / %d patches kept (min_tissue_ratio=%.2f)",
            len(filtered),
            len(coordinates),
            self.min_tissue_ratio,
        )
        return filtered

    def _filter_by_tissue(self, slide: "Slide", coordinates: np.ndarray) -> np.ndarray:
        """Remove patches that have insufficient tissue overlap.

        The tissue mask is stored at thumbnail resolution.  We scale each
        candidate patch's level-0 bounding box into thumbnail space, then
        compute what fraction of the corresponding mask region is ``True``.

        **Boundary-aware erosion** -- Before checking tissue ratios the mask
        is eroded by half the patch footprint (in mask pixels).  This
        accounts for the fact that a patch of size *S* placed at the tissue
        boundary extends ~S/2 into non-tissue.  Without erosion, coarse
        binary masks (e.g. from DL tissue detectors whose small prediction
        grid is upscaled with nearest-neighbour interpolation) give every
        boundary patch a tissue ratio of exactly 0 or 1, hiding the effect
        of patch size on coverage.  The erosion is capped at 2 % of the
        mask dimensions to avoid over-eroding very small masks.
        """
        tissue_mask = slide.tissue_mask
        if tissue_mask is None:
            return coordinates

        if tissue_mask.size == 0:
            return coordinates

        mask_h, mask_w = tissue_mask.shape[:2]
        slide_w, slide_h = slide.dimensions

        if slide_w == 0 or slide_h == 0:
            return coordinates

        # Scale factors from level-0 to tissue-mask space
        scale_x = mask_w / slide_w
        scale_y = mask_h / slide_h

        # --- Boundary-aware erosion --------------------------------
        # Erode by half the patch footprint in mask pixels so that
        # larger patches produce a larger exclusion zone at tissue
        # boundaries.  Cap to 2% of mask size for safety.
        patch_w_mask = coordinates[0, 2] * scale_x
        patch_h_mask = coordinates[0, 3] * scale_y
        erode_x = int(patch_w_mask / 2)
        erode_y = int(patch_h_mask / 2)
        max_erode = max(1, int(min(mask_h, mask_w) * 0.02))
        erode_x = min(erode_x, max_erode)
        erode_y = min(erode_y, max_erode)

        if erode_x >= 1 and erode_y >= 1:
            try:
                import cv2

                kernel = np.ones((2 * erode_y + 1, 2 * erode_x + 1), np.uint8)
                effective_mask = cv2.erode(
                    tissue_mask.astype(np.uint8), kernel, iterations=1
                ).astype(bool)
            except ImportError:
                effective_mask = tissue_mask
        else:
            effective_mask = tissue_mask

        keep: List[bool] = []
        for coord in coordinates:
            x, y, w, h = coord

            # Map to mask coordinates (clamp to valid range)
            mx0 = max(0, int(x * scale_x))
            my0 = max(0, int(y * scale_y))
            mx1 = min(mask_w, int((x + w) * scale_x) + 1)
            my1 = min(mask_h, int((y + h) * scale_y) + 1)

            # Guard against degenerate regions
            if mx1 <= mx0 or my1 <= my0:
                keep.append(False)
                continue

            region = effective_mask[my0:my1, mx0:mx1]
            tissue_ratio = float(np.mean(region.astype(np.float32)))
            keep.append(tissue_ratio >= self.min_tissue_ratio)

        keep_arr = np.array(keep, dtype=bool)
        filtered = coordinates[keep_arr]

        logger.debug(
            "Tissue filtering: %d / %d patches kept (min_tissue_ratio=%.2f)",
            len(filtered),
            len(coordinates),
            self.min_tissue_ratio,
        )
        return filtered

    def _read_patches(
        self,
        slide: "Slide",
        coordinates: np.ndarray,
        target_level: int,
    ) -> List[np.ndarray]:
        """Read pixel data for each coordinate from the slide.

        Each coordinate is ``[x, y, w_l0, h_l0]`` in level-0 space.
        ``slide.read_region`` takes the *location* in level-0 pixels and
        the *size* in target-level pixels.
        """
        patch_images: List[np.ndarray] = []
        target_size = (self.patch_size, self.patch_size)

        for coord in coordinates:
            x, y = int(coord[0]), int(coord[1])
            try:
                region = slide.read_region(
                    location=(x, y),
                    size=target_size,
                    level=target_level,
                )
                # Ensure RGB uint8 (strip alpha channel if present)
                if region.ndim == 3 and region.shape[2] > 3:
                    region = region[:, :, :3]
                if region.dtype != np.uint8:
                    region = region.astype(np.uint8)

                # Handle edge cases where the returned region is smaller
                # than expected (e.g. near slide borders).
                rh, rw = region.shape[:2]
                if rh != self.patch_size or rw != self.patch_size:
                    padded = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
                    padded[:rh, :rw] = region[:, :, :3] if region.ndim == 3 else 0
                    region = padded

                patch_images.append(region)
            except (IndexError, ValueError, RuntimeError, OSError) as exc:
                logger.warning(
                    "Failed to read patch at (%d, %d) level %d: %s",
                    x,
                    y,
                    target_level,
                    exc,
                )
                # Insert a blank patch so coordinate alignment is preserved
                patch_images.append(np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8))

        return patch_images

    def _build_metadata(
        self,
        slide: "Slide",
        target_level: int,
        tissue_coordinates: Optional[np.ndarray] = None,
    ) -> dict:
        """Assemble metadata dict for the Patches container."""
        meta = {
            "patch_size": self.patch_size,
            "stride": self.stride,
            "level": target_level,
            "magnification": self.magnification,
            "slide_path": str(slide.path),
            "slide_dimensions": slide.dimensions,
            "min_tissue_ratio": self.min_tissue_ratio,
        }
        if tissue_coordinates is not None:
            meta["tissue_filter"] = "tissue_coordinates"
            meta["tissue_coordinates_count"] = len(tissue_coordinates)
        else:
            meta["tissue_filter"] = "tissue_mask" if slide.tissue_mask is not None else "none"
        return meta

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        overlap = 1.0 - self.stride / self.patch_size if self.patch_size > 0 else 0.0
        parts = [
            f"patch_size={self.patch_size}",
            f"stride={self.stride}",
        ]
        if overlap > 0:
            parts.append(f"overlap={overlap:.1%}")
        if self.magnification is not None:
            parts.append(f"magnification={self.magnification}")
        else:
            parts.append(f"level={self.level}")
        parts.append(f"min_tissue_ratio={self.min_tissue_ratio}")
        return f"PatchExtractor({', '.join(parts)})"
