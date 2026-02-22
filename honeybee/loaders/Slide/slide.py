"""
Slide -- high-level wrapper for whole-slide images.

Provides a clean, backend-agnostic API for reading WSI files while
maintaining full backward compatibility with the legacy tile-based
constructor used by :class:`PathologyProcessor` and
:class:`WholeSlideImageDataset`.
"""

import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset


class WholeSlideImageDataset(Dataset):
    def __init__(self, slideClass, transform=None):
        self.slideClass = slideClass
        self.transform = transform
        self.suitableTileAddresses = self.slideClass.suitableTileAddresses()

    def __len__(self):
        return len(self.suitableTileAddresses)

    def __getitem__(self, idx):
        tileAddress = self.suitableTileAddresses[idx]
        img = self.slideClass.getTile(tileAddress, writeToNumpy=True)[..., :3]
        img = self.transform(Image.fromarray(img).convert("RGB"))
        return {"image": img, "tileAddress": tileAddress}


# ---------------------------------------------------------------------------
# Lazy import helper -- keeps module import fast and avoids hard dependencies
# on matplotlib / plotly / tissue detection until actually needed.
# ---------------------------------------------------------------------------

# PIL.Image is needed by WholeSlideImageDataset at runtime, but imported
# lazily here so the module can still be *imported* without Pillow installed.
try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment,misc]


class Slide:
    """Unified wrapper around a whole-slide image file.

    **New-style usage** (clean API)::

        slide = Slide("path/to/slide.svs")
        region = slide.read_region((0, 0), (512, 512))
        thumb  = slide.get_thumbnail()

    **Legacy usage** (backward-compatible with the old tile-based constructor)::

        slide = Slide(
            slide_image_path="path/to/slide.svs",
            tile_size=512,
            max_patches=500,
        )
        for addr in slide.iterateTiles():
            tile = slide.getTile(addr, writeToNumpy=True)

    Parameters
    ----------
    path : str or Path, optional
        Path to the WSI file.  Mutually exclusive with *slide_image_path*.
    backend : str, optional
        ``"cucim"`` or ``"openslide"``.  ``None`` = auto-detect.
    slide_image_path : str, optional
        Legacy positional path (old constructor signature).
    tile_size : int
        Tile side length in pixels (legacy API).
    tileOverlap : float
        Overlap ratio between tiles (0 -- 1).  Legacy API.
    max_patches : int
        Maximum number of tiles to allow; controls automatic level selection.
    visualize : bool
        If ``True``, run the legacy :meth:`visualize` method after init.
    tissue_detector : object, optional
        A deep-learning tissue detector instance (legacy API).
    path_to_store_visualization : str
        Directory for saving legacy visualisation PNGs.
    verbose : bool
        Print level/dimension information during init.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        backend: Optional[str] = None,
        *,
        # Legacy keyword arguments --------------------------------
        slide_image_path: Optional[str] = None,
        tile_size: int = 512,
        tileOverlap: float = 0,
        max_patches: int = 500,
        visualize: bool = False,
        tissue_detector: Any = None,
        path_to_store_visualization: str = "./visualizations",
        verbose: bool = False,
    ):
        # Resolve *path* -- prefer explicit ``path``, fall back to legacy
        # ``slide_image_path`` keyword.
        if path is None and slide_image_path is not None:
            path = slide_image_path
        if path is None:
            raise ValueError(
                "A slide path is required.  Pass it as the first positional "
                "argument or via the slide_image_path keyword."
            )

        self.path: Path = Path(path)
        self.verbose = verbose

        # ----- Backend & handle -----------------------------------
        from ._backend import get_backend

        self._backend = get_backend(backend)
        self._handle = self._backend.open(str(self.path))

        # ----- New-API state --------------------------------------
        self._tissue_mask: Optional[np.ndarray] = None
        self._prediction_map: Optional[np.ndarray] = None
        self._thumbnail_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # ----- Legacy attributes (always set) ---------------------
        self.slide_image_path: str = str(self.path)
        self.slideFileName: str = self.path.stem
        self.slideFilePath: Path = self.path

        # ``img`` -- callers (PathologyProcessor._get_overview_image,
        # get_slide_info) access ``wsi.img.resolutions``,
        # ``wsi.img.metadata``, ``wsi.img.read_region``.
        # We expose a lightweight shim so those attribute chains keep working.
        self.img = self._LegacyImgProxy(self)

        # Determine whether this is a legacy-style invocation.  When the
        # caller supplies *slide_image_path* or any tile/patch keyword that
        # differs from the default, we set up the full tile infrastructure.
        _legacy_mode = slide_image_path is not None

        if _legacy_mode:
            self._init_legacy_tiles(
                tile_size=tile_size,
                tileOverlap=tileOverlap,
                max_patches=max_patches,
                tissue_detector=tissue_detector,
                path_to_store_visualization=path_to_store_visualization,
                visualize_flag=visualize,
            )
        else:
            # Minimal defaults so attribute access never raises AttributeError
            self.tileSize: int = tile_size
            self.tileOverlap: int = round(tileOverlap * tile_size)
            self.tileDictionary: Dict[Tuple[int, int], dict] = {}
            self.numTilesInX: int = 0
            self.numTilesInY: int = 0
            self.tissue_detector = tissue_detector
            self.path_to_store_visualization: str = path_to_store_visualization
            self.selected_level: int = 0
            # ``slide`` -- in legacy mode this held the region read at the
            # selected level.  In new mode we leave it as ``None``; callers
            # should use :meth:`read_region` / :meth:`get_thumbnail` instead.
            self.slide: Any = None

    # ------------------------------------------------------------------
    # Legacy initialisation helpers
    # ------------------------------------------------------------------

    def _init_legacy_tiles(
        self,
        tile_size: int,
        tileOverlap: float,
        max_patches: int,
        tissue_detector: Any,
        path_to_store_visualization: str,
        visualize_flag: bool,
    ) -> None:
        """Set up the full tile-based infrastructure identical to the old
        ``Slide.__init__`` so that downstream code sees the same attributes.
        """
        self.tileSize = tile_size
        self.tileOverlap = round(tileOverlap * tile_size)
        self.tileDictionary: Dict[Tuple[int, int], dict] = {}
        self.tissue_detector = tissue_detector
        self.path_to_store_visualization = path_to_store_visualization

        # Select level --------------------------------------------------
        self.selected_level = self._select_level(max_patches)

        # Read the whole slide at the selected level into ``self.slide``.
        # This mirrors the old CuImage-based flow where ``self.slide`` was
        # the region object returned by ``CuImage.read_region``.
        level_dims = self._backend.get_level_dimensions(self._handle)
        sel_w, sel_h = level_dims[self.selected_level]
        slide_arr = self._backend.read_region(
            self._handle,
            location=(0, 0),
            level=self.selected_level,
            size=(sel_w, sel_h),
        )
        # Wrap in a small object that exposes ``.width``, ``.height``,
        # ``.metadata``, ``np.asarray()`` and ``read_region`` so that
        # existing callers keep working.
        self.slide = self._LegacySlideRegion(slide_arr, self)

        if self.verbose:
            print(f"Selected level {self.selected_level} with dimensions: " f"{sel_h}x{sel_w}")

        # Generate tile grid -------------------------------------------
        self.numTilesInX = (
            sel_w // (self.tileSize - self.tileOverlap)
            if (self.tileSize - self.tileOverlap) > 0
            else 0
        )
        self.numTilesInY = (
            sel_h // (self.tileSize - self.tileOverlap)
            if (self.tileSize - self.tileOverlap) > 0
            else 0
        )
        self.tileDictionary = self._generate_tile_dictionary()

        if self.tissue_detector is not None:
            self.detectTissue()

        if visualize_flag:
            self._legacy_visualize()

    def _select_level(self, max_patches: int) -> int:
        """Select the highest-resolution pyramid level whose total tile count
        does not exceed *max_patches*.  Falls back to the lowest-resolution
        level.
        """
        level_dims = self._backend.get_level_dimensions(self._handle)
        level_count = self._backend.get_level_count(self._handle)

        if self.verbose:
            print(f"Resolutions: level_count={level_count}, dims={level_dims}")

        selected_level = level_count - 1  # safest fallback
        for level in range(level_count):
            width, height = level_dims[level]
            numTilesInX = width // self.tileSize if self.tileSize > 0 else 0
            numTilesInY = height // self.tileSize if self.tileSize > 0 else 0
            if self.verbose:
                print(
                    f"Level {level}: {numTilesInX}x{numTilesInY} "
                    f"({numTilesInX * numTilesInY}) \t "
                    f"Resolution: {width}x{height}"
                )
            if numTilesInX * numTilesInY <= max_patches:
                selected_level = level
                break

        return selected_level

    def _generate_tile_dictionary(self) -> Dict[Tuple[int, int], dict]:
        tile_dict: Dict[Tuple[int, int], dict] = {}
        stride = self.tileSize - self.tileOverlap
        if stride <= 0:
            return tile_dict
        for y in range(self.numTilesInY):
            for x in range(self.numTilesInX):
                tile_dict[(x, y)] = {
                    "x": x * stride,
                    "y": y * stride,
                    "width": self.tileSize,
                    "height": self.tileSize,
                }
        return tile_dict

    # ------------------------------------------------------------------
    # Properties (new API)
    # ------------------------------------------------------------------

    @property
    def level_count(self) -> int:
        """Number of pyramid levels in the slide."""
        return self._backend.get_level_count(self._handle)

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        """``(width, height)`` for every pyramid level, highest-res first."""
        return self._backend.get_level_dimensions(self._handle)

    @property
    def level_downsamples(self) -> List[float]:
        """Downsample factor for every pyramid level (level 0 is ``1.0``)."""
        return self._backend.get_level_downsamples(self._handle)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Level-0 ``(width, height)``."""
        return self.level_dimensions[0]

    @property
    def magnification(self) -> Optional[float]:
        """Objective magnification (e.g. 40.0), or ``None`` if unknown."""
        props = self._backend.get_properties(self._handle)
        return props.get("magnification")

    @property
    def mpp(self) -> Optional[float]:
        """Microns per pixel at level 0, or ``None`` if unknown.

        Returns the average of mpp_x and mpp_y when both are available.
        """
        props = self._backend.get_properties(self._handle)
        mpp_x = props.get("mpp_x")
        mpp_y = props.get("mpp_y")
        if mpp_x is not None and mpp_y is not None:
            return (mpp_x + mpp_y) / 2.0
        return mpp_x or mpp_y

    @property
    def tissue_mask(self) -> Optional[np.ndarray]:
        """Binary tissue mask set by :meth:`detect_tissue`, or ``None``."""
        return self._tissue_mask

    @property
    def prediction_map(self) -> Optional[np.ndarray]:
        """3-class prediction map from DL tissue detection, or ``None``.

        Shape ``(rows, cols, 3)`` with probabilities for
        ``[artifact, background, tissue]``.
        """
        return self._prediction_map

    @property
    def info(self) -> dict:
        """Summary dictionary of slide metadata."""
        props = self._backend.get_properties(self._handle)
        return {
            "path": str(self.path),
            "backend": self._backend.name,
            "dimensions": self.dimensions,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "level_downsamples": self.level_downsamples,
            "magnification": self.magnification,
            "mpp": self.mpp,
            "vendor": props.get("vendor"),
        }

    # ------------------------------------------------------------------
    # Core reading methods (new API)
    # ------------------------------------------------------------------

    def read_region(
        self,
        location: Tuple[int, int],
        size: Tuple[int, int],
        level: Optional[int] = None,
        magnification: Optional[float] = None,
    ) -> np.ndarray:
        """Read a rectangular region from the slide.

        Parameters
        ----------
        location : (x, y)
            Top-left corner in **level-0** pixel coordinates.
        size : (width, height)
            Region size in pixels *at the target level/magnification*.
        level : int, optional
            Pyramid level (0 = highest resolution).  Default ``0``.
        magnification : float, optional
            Target objective magnification (e.g. ``20.0``).  If given,
            the closest pyramid level is selected automatically and
            *level* is ignored.

        Returns
        -------
        np.ndarray
            RGB image with shape ``(height, width, 3)`` and dtype ``uint8``.
        """
        if magnification is not None:
            level = self.get_best_level_for_magnification(magnification)
        if level is None:
            level = 0

        return self._backend.read_region(self._handle, location=location, level=level, size=size)

    def get_thumbnail(self, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Return a thumbnail that fits within *size*.  Cached after the
        first call for each distinct *size*.

        Parameters
        ----------
        size : (width, height)
            Maximum bounding box for the thumbnail.

        Returns
        -------
        np.ndarray
            RGB ``uint8`` array.
        """
        key = tuple(size)
        if key not in self._thumbnail_cache:
            self._thumbnail_cache[key] = self._backend.get_thumbnail(self._handle, size)
        return self._thumbnail_cache[key]

    def get_best_level_for_magnification(self, target_mag: float) -> int:
        """Find the pyramid level closest to *target_mag*.

        If the slide's native magnification is unknown, returns level 0.
        """
        native_mag = self.magnification
        if native_mag is None or native_mag <= 0:
            return 0
        target_downsample = native_mag / target_mag
        return self.get_best_level_for_downsample(target_downsample)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Return the pyramid level whose downsample factor is closest to
        (but not exceeding) *downsample*.
        """
        downsamples = self.level_downsamples
        best_level = 0
        for i, ds in enumerate(downsamples):
            if ds <= downsample:
                best_level = i
            else:
                break
        return best_level

    # ------------------------------------------------------------------
    # Tissue detection (new API)
    # ------------------------------------------------------------------

    def detect_tissue(
        self,
        method: str = "otsu",
        min_tissue_size: int = 1000,
        thumbnail_size: Tuple[int, int] = (2048, 2048),
        *,
        detector: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Detect tissue on a low-resolution thumbnail and store the result.

        Parameters
        ----------
        method : str
            Detection method: ``"otsu"``, ``"hsv"``, ``"otsu_hsv"``,
            ``"gradient"``, ``"multi_otsu"``, or ``"dl"`` for the
            DenseNet121-based deep-learning detector.
        min_tissue_size : int
            Minimum tissue region size in pixels (at thumbnail resolution).
            Only used for classical methods.
        thumbnail_size : (width, height)
            Size of the thumbnail used for detection / mask upscaling.
        detector : TissueDetector, optional
            Pre-initialised :class:`TissueDetector` instance.  When
            provided, ``method`` is ignored and this detector is used
            directly.
        **kwargs
            For classical methods: forwarded to
            :class:`ClassicalTissueDetector`.
            For ``method="dl"``: ``device``, ``patch_size``,
            ``batch_size``, ``model_path``, ``level``, ``magnification``,
            ``threshold``, ``num_workers`` are extracted and forwarded
            to :class:`TissueDetector`.

        Returns
        -------
        np.ndarray
            Binary boolean mask at thumbnail resolution.
        """
        if detector is not None or method == "dl":
            from honeybee.models.TissueDetector.tissue_detector import TissueDetector

            if detector is None:
                init_kwargs = {}
                for key in ("device", "patch_size", "batch_size", "model_path"):
                    if key in kwargs:
                        init_kwargs[key] = kwargs.pop(key)
                detector = TissueDetector(**init_kwargs)

            binary_mask, pred_map = detector.detect(
                self, thumbnail_size=thumbnail_size, **kwargs
            )
            self._tissue_mask = binary_mask
            self._prediction_map = pred_map
            return binary_mask

        from honeybee.processors.wsi.tissue_detection import ClassicalTissueDetector

        thumb = self.get_thumbnail(thumbnail_size)
        classical = ClassicalTissueDetector(
            method=method, min_tissue_size=min_tissue_size, **kwargs
        )
        mask = classical.detect(thumb)
        self._tissue_mask = mask
        return mask

    # ------------------------------------------------------------------
    # Visualisation (new API)
    # ------------------------------------------------------------------

    def visualize(
        self,
        show_tissue: bool = True,
        show_patches: bool = False,
        patch_coords: Optional[np.ndarray] = None,
        interactive: bool = False,
        ax: Any = None,
    ) -> Any:
        """All-in-one slide visualisation.

        Parameters
        ----------
        show_tissue : bool
            Overlay tissue mask (green = tissue, red = background).
        show_patches : bool
            Draw tile-grid rectangles.
        patch_coords : array-like, optional
            ``(N, 4)`` array of ``[x, y, w, h]`` rectangles to highlight.
        interactive : bool
            Use Plotly instead of Matplotlib when ``True``.
        ax : matplotlib Axes, optional
            Existing axes to draw on (for subplot layouts).

        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure or None
        """
        thumb = self.get_thumbnail()

        if interactive:
            return self._visualize_plotly(
                thumb,
                show_tissue=show_tissue,
                show_patches=show_patches,
                patch_coords=patch_coords,
            )

        return self._visualize_matplotlib(
            thumb,
            show_tissue=show_tissue,
            show_patches=show_patches,
            patch_coords=patch_coords,
            ax=ax,
        )

    def plot_tissue_mask(
        self,
        alpha: float = 0.4,
        interactive: bool = False,
        ax: Any = None,
    ) -> Any:
        """Overlay the tissue mask on a thumbnail.

        Calls :meth:`detect_tissue` automatically if the mask has not yet
        been computed.
        """
        if self._tissue_mask is None:
            self.detect_tissue()

        return self.visualize(
            show_tissue=True,
            show_patches=False,
            interactive=interactive,
            ax=ax,
        )

    def plot_region(
        self,
        location: Tuple[int, int],
        size: Tuple[int, int],
        level: int = 0,
        ax: Any = None,
    ) -> Any:
        """Read and display a specific region.

        Returns the matplotlib ``Figure`` (or ``None`` if *ax* is provided).
        """
        import matplotlib.pyplot as plt

        region = self.read_region(location, size, level=level)
        created_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        else:
            fig = ax.figure
        ax.imshow(region)
        ax.set_title(
            f"Region ({location[0]}, {location[1]}) " f"size {size[0]}x{size[1]} level={level}"
        )
        ax.axis("off")
        if created_fig:
            plt.close(fig)
        return fig

    def plot_tissue_detection(
        self,
        thumbnail_size: Tuple[int, int] = (2048, 2048),
        figsize: Optional[Tuple[float, float]] = None,
        ax: Any = None,
    ) -> Any:
        """Visualize tissue detection results.

        When a DL prediction map is available (from ``detect_tissue(method="dl")``),
        draws a 4-panel figure: thumbnail, 3-class prediction RGB, tissue probability
        heatmap, and binary mask overlay.  Falls back to a 2-panel layout (thumbnail +
        mask) when only a classical mask exists.

        Parameters
        ----------
        thumbnail_size : (int, int)
            Size passed to :meth:`get_thumbnail`.
        figsize : (float, float), optional
            Figure size.  Defaults to ``(20, 5)`` for 4-panel, ``(12, 5)`` for 2-panel.
        ax : matplotlib Axes or array of Axes, optional
            Pre-created axes to draw on.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if self._tissue_mask is None:
            raise ValueError(
                "No tissue mask available. Call detect_tissue() first."
            )

        created_fig = ax is None
        thumbnail = self.get_thumbnail(thumbnail_size)
        has_dl = self._prediction_map is not None

        if has_dl:
            import cv2
            from honeybee.models.TissueDetector.tissue_detector import TissueDetector

            pred_map = self._prediction_map
            tissue_prob = pred_map[:, :, 2]

            if ax is not None:
                axes = np.atleast_1d(ax)
                fig = axes[0].figure
            else:
                if figsize is None:
                    figsize = (20, 5)
                fig, axes = plt.subplots(1, 4, figsize=figsize)

            # 1 - Thumbnail
            axes[0].imshow(thumbnail)
            axes[0].set_title("Thumbnail")
            axes[0].axis("off")

            # 2 - 3-class prediction RGB
            pred_rgb = TissueDetector.prediction_map_to_rgb(pred_map)
            axes[1].imshow(pred_rgb, interpolation="nearest")
            axes[1].set_title("3-Class Prediction (R=artifact, G=bg, B=tissue)")
            axes[1].axis("off")

            # 3 - Tissue probability heatmap
            im = axes[2].imshow(
                tissue_prob, cmap="hot", vmin=0, vmax=1, interpolation="nearest"
            )
            axes[2].set_title("Tissue Probability Heatmap")
            axes[2].axis("off")
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

            # 4 - Binary mask overlay
            dl_mask_small = (tissue_prob > 0.5).astype(np.uint8)
            thumb_small = cv2.resize(
                thumbnail, (pred_map.shape[1], pred_map.shape[0])
            )
            overlay = thumb_small.copy()
            mask_rgb = np.zeros_like(overlay)
            mask_rgb[dl_mask_small.astype(bool)] = [0, 255, 0]
            overlay = cv2.addWeighted(overlay, 0.7, mask_rgb, 0.3, 0)
            axes[3].imshow(overlay)
            axes[3].set_title(
                f"DL Mask Overlay ({self._tissue_mask.mean():.1%} tissue)"
            )
            axes[3].axis("off")

            fig.suptitle("Deep Learning Tissue Detection (DenseNet121)", fontsize=14)
        else:
            # Classical mask only — 2-panel
            if ax is not None:
                axes = np.atleast_1d(ax)
                fig = axes[0].figure
            else:
                if figsize is None:
                    figsize = (12, 5)
                fig, axes = plt.subplots(1, 2, figsize=figsize)

            axes[0].imshow(thumbnail)
            axes[0].set_title("Thumbnail")
            axes[0].axis("off")

            axes[1].imshow(self._tissue_mask, cmap="gray")
            axes[1].set_title(
                f"Tissue Mask ({self._tissue_mask.mean():.1%} tissue)"
            )
            axes[1].axis("off")

        fig.tight_layout()
        if created_fig:
            plt.close(fig)
        return fig

    def compare_tissue_methods(
        self,
        methods: Optional[List[str]] = None,
        thumbnail_size: Tuple[int, int] = (2048, 2048),
        figsize: Optional[Tuple[float, float]] = None,
        ax: Any = None,
    ) -> Any:
        """Compare multiple tissue detection methods side-by-side.

        Runs each method via :meth:`detect_tissue`, displays masks with tissue %
        in titles, then restores the original mask.

        Parameters
        ----------
        methods : list of str, optional
            Detection methods to compare.  Defaults include ``"dl"`` only if a
            DL prediction map already exists (to avoid expensive re-computation).
        thumbnail_size : (int, int)
            Thumbnail size for classical detection.
        figsize : (float, float), optional
            Figure size.
        ax : array of matplotlib Axes, optional
            Pre-created axes.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if methods is None:
            methods = ["otsu", "hsv", "otsu_hsv"]
            if self._prediction_map is not None:
                methods = ["dl"] + methods

        # Save current state
        saved_mask = self._tissue_mask
        saved_pred = self._prediction_map

        created_fig = ax is None
        n = len(methods)
        if ax is not None:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure
        else:
            if figsize is None:
                figsize = (5 * n, 5)
            fig, axes = plt.subplots(1, n, figsize=figsize)
            if n == 1:
                axes = [axes]

        try:
            for i, method in enumerate(methods):
                if method == "dl" and saved_pred is not None:
                    # Reuse existing DL mask instead of re-running
                    mask = saved_mask
                else:
                    self.detect_tissue(
                        method=method, thumbnail_size=thumbnail_size
                    )
                    mask = self._tissue_mask

                axes[i].imshow(mask, cmap="gray")
                axes[i].set_title(f"{method} ({mask.mean():.1%} tissue)")
                axes[i].axis("off")
        finally:
            # Restore original state
            self._tissue_mask = saved_mask
            self._prediction_map = saved_pred

        fig.suptitle("Tissue Detection: Method Comparison", fontsize=14)
        fig.tight_layout()
        if created_fig:
            plt.close(fig)
        return fig

    # ---- internal visualisation helpers ----

    def _visualize_matplotlib(
        self,
        thumb: np.ndarray,
        show_tissue: bool,
        show_patches: bool,
        patch_coords: Optional[np.ndarray],
        ax: Any = None,
    ) -> Any:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            created_fig = True
        else:
            fig = ax.figure

        ax.imshow(thumb)
        ax.set_title(self.path.name)
        ax.axis("off")

        # Tissue overlay ------------------------------------------------
        if show_tissue and self._tissue_mask is not None:
            mask = self._tissue_mask
            # Resize mask to thumbnail size
            from PIL import Image as PILImage

            mask_resized = np.array(
                PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                    (thumb.shape[1], thumb.shape[0]),
                    PILImage.NEAREST,
                )
            )
            overlay = np.zeros((*thumb.shape[:2], 4), dtype=np.uint8)
            tissue = mask_resized > 127
            overlay[tissue] = [0, 200, 0, 80]  # green
            overlay[~tissue] = [200, 0, 0, 60]  # red
            ax.imshow(overlay)

        # Patch grid / highlights ----------------------------------------
        if show_patches and self.tileDictionary:
            scale_x = thumb.shape[1] / self.dimensions[0]
            scale_y = thumb.shape[0] / self.dimensions[1]
            for info in self.tileDictionary.values():
                rect = Rectangle(
                    (info["x"] * scale_x, info["y"] * scale_y),
                    info["width"] * scale_x,
                    info["height"] * scale_y,
                    linewidth=0.5,
                    edgecolor="cyan",
                    facecolor="none",
                )
                ax.add_patch(rect)

        if patch_coords is not None:
            scale_x = thumb.shape[1] / self.dimensions[0]
            scale_y = thumb.shape[0] / self.dimensions[1]
            coords = np.asarray(patch_coords)
            for row in coords:
                x, y, w, h = row[:4]
                rect = Rectangle(
                    (x * scale_x, y * scale_y),
                    w * scale_x,
                    h * scale_y,
                    linewidth=1,
                    edgecolor="yellow",
                    facecolor="none",
                )
                ax.add_patch(rect)

        if created_fig:
            plt.close(fig)
        return fig if created_fig else None

    def _visualize_plotly(
        self,
        thumb: np.ndarray,
        show_tissue: bool,
        show_patches: bool,
        patch_coords: Optional[np.ndarray],
    ) -> Any:
        import plotly.graph_objects as go
        from PIL import Image as PILImage

        fig = go.Figure()
        fig.add_trace(go.Image(z=thumb))

        if show_tissue and self._tissue_mask is not None:
            mask = self._tissue_mask
            mask_resized = np.array(
                PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                    (thumb.shape[1], thumb.shape[0]),
                    PILImage.NEAREST,
                )
            )
            overlay = np.zeros_like(thumb)
            tissue = mask_resized > 127
            overlay[tissue] = [0, 200, 0]
            overlay[~tissue] = [200, 0, 0]
            fig.add_trace(go.Image(z=overlay, opacity=0.3))

        if patch_coords is not None:
            scale_x = thumb.shape[1] / self.dimensions[0]
            scale_y = thumb.shape[0] / self.dimensions[1]
            coords = np.asarray(patch_coords)
            for row in coords:
                x, y, w, h = row[:4]
                sx, sy = x * scale_x, y * scale_y
                sw, sh = w * scale_x, h * scale_y
                fig.add_shape(
                    type="rect",
                    x0=sx,
                    y0=sy,
                    x1=sx + sw,
                    y1=sy + sh,
                    line=dict(color="yellow", width=2),
                )

        fig.update_layout(
            title=self.path.name,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x"),
        )
        return fig

    # ------------------------------------------------------------------
    # Legacy tile-based API
    # ------------------------------------------------------------------

    def suitableTileAddresses(self) -> List[Tuple[int, int]]:
        """Return a list of all tile addresses."""
        return list(self.iterateTiles())

    def getTile(
        self,
        tileAddress: Tuple[int, int],
        writeToNumpy: bool = False,
    ) -> Optional[np.ndarray]:
        """Read a tile by its ``(x, y)`` grid address.

        In the new backend world every read returns a NumPy array, so
        *writeToNumpy* is accepted for compatibility but has no effect.
        """
        if (
            len(tileAddress) == 2
            and isinstance(tileAddress, tuple)
            and tileAddress in self.tileDictionary
        ):
            tile_info = self.tileDictionary[tileAddress]
            if self.slide is not None:
                # Legacy mode: read from the cached slide region
                x, y = tile_info["x"], tile_info["y"]
                w, h = tile_info["width"], tile_info["height"]
                arr = np.asarray(self.slide)
                # Bounds check
                if y + h <= arr.shape[0] and x + w <= arr.shape[1]:
                    return arr[y : y + h, x : x + w].copy()
                # Partial region at edge
                out = np.zeros((h, w, 3), dtype=np.uint8)
                valid_h = min(h, arr.shape[0] - y) if y < arr.shape[0] else 0
                valid_w = min(w, arr.shape[1] - x) if x < arr.shape[1] else 0
                if valid_h > 0 and valid_w > 0:
                    out[:valid_h, :valid_w] = arr[y : y + valid_h, x : x + valid_w]
                return out
            else:
                # New mode fallback: use the backend directly at the
                # selected level.  Tile coordinates are in selected-level
                # space; we need to map back to level-0 coordinates.
                ds = self.level_downsamples[self.selected_level]
                loc_x = int(tile_info["x"] * ds)
                loc_y = int(tile_info["y"] * ds)
                w, h = tile_info["width"], tile_info["height"]
                return self._backend.read_region(
                    self._handle,
                    location=(loc_x, loc_y),
                    level=self.selected_level,
                    size=(w, h),
                )
        return None

    def iterateTiles(
        self,
        tileDictionary: Optional[dict] = False,
        includeImage: bool = False,
        writeToNumpy: bool = False,
    ):
        """Iterate over tile addresses (and optionally images)."""
        td = self.tileDictionary if not tileDictionary else tileDictionary
        for key in td:
            if includeImage:
                yield key, self.getTile(key, writeToNumpy=writeToNumpy)
            else:
                yield key

    def detectTissue(
        self,
        tissueDetectionUpsampleFactor: int = 4,
        batchSize: int = 20,
        numWorkers: int = 1,
    ) -> None:
        """Run deep-learning tissue detection (legacy API).

        This mirrors the old ``Slide.detectTissue`` method exactly.
        """
        import torch
        from skimage.transform import resize

        detector = self.tissue_detector
        predictionKey = "tissue_detector"
        device = detector.device
        model = detector.model
        data_transforms = detector.transforms
        pathSlideDataset = WholeSlideImageDataset(self, transform=data_transforms)
        pathSlideDataloader = torch.utils.data.DataLoader(
            pathSlideDataset,
            batch_size=batchSize,
            shuffle=False,
            num_workers=numWorkers,
        )

        for batch_index, inputs in enumerate(pathSlideDataloader):
            inputTile = inputs["image"].to(device)
            output = model(inputTile)
            batch_prediction = torch.nn.functional.softmax(output, dim=1).cpu().data.numpy()
            for index in range(len(inputTile)):
                tileAddress = (
                    inputs["tileAddress"][0][index].item(),
                    inputs["tileAddress"][1][index].item(),
                )
                self.tileDictionary[tileAddress][predictionKey] = batch_prediction[index, ...]

        upsampleFactor = tissueDetectionUpsampleFactor
        for orphanTileAddress in self.iterateTiles():
            self.tileDictionary[orphanTileAddress].update(
                {
                    "x": self.tileDictionary[orphanTileAddress]["x"] * upsampleFactor,
                    "y": self.tileDictionary[orphanTileAddress]["y"] * upsampleFactor,
                    "width": self.tileDictionary[orphanTileAddress]["width"] * upsampleFactor,
                    "height": self.tileDictionary[orphanTileAddress]["height"] * upsampleFactor,
                }
            )

        self.predictionMap = np.zeros([self.numTilesInY, self.numTilesInX, 3])
        for address in self.iterateTiles():
            if "tissue_detector" in self.tileDictionary[address]:
                self.predictionMap[address[1], address[0], :] = self.tileDictionary[address][
                    "tissue_detector"
                ]

        predictionMap2 = np.zeros([self.numTilesInY, self.numTilesInX])
        predictionMap1res = resize(
            self.predictionMap, predictionMap2.shape, order=0, anti_aliasing=False
        )

        for address in self.iterateTiles():
            self.tileDictionary[address].update(
                {"artifactLevel": predictionMap1res[address[1], address[0]][0]}
            )
            self.tileDictionary[address].update(
                {"backgroundLevel": predictionMap1res[address[1], address[0]][1]}
            )
            self.tileDictionary[address].update(
                {"tissueLevel": predictionMap1res[address[1], address[0]][2]}
            )

    def load_tile_thread(
        self,
        start_loc: Tuple[int, int],
        patch_size: int,
        target_size: int,
    ) -> np.ndarray:
        """Load and resize a single tile (thread-safe helper)."""
        try:
            from albumentations import Compose, Resize

            tile = self._backend.read_region(
                self._handle,
                location=start_loc,
                level=0,
                size=(patch_size, patch_size),
            )
            if tile.ndim == 3 and tile.shape[2] == 3:
                transform = Compose([Resize(height=target_size, width=target_size)])
                tile = transform(image=tile)["image"]
                return tile
            else:
                return np.zeros((target_size, target_size, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error reading tile at {start_loc}: {e}")
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    def load_patches_concurrently(self, target_patch_size: int) -> np.ndarray:
        """Load tissue patches in parallel (legacy API)."""
        threshold = 0.8
        tissue_coordinates = []
        for address in self.iterateTiles():
            if self.tileDictionary[address].get("tissueLevel", 0) > threshold:
                tissue_coordinates.append(
                    (
                        self.tileDictionary[address]["x"],
                        self.tileDictionary[address]["y"],
                    )
                )
        num_patches = len(tissue_coordinates)
        patches = np.zeros((num_patches, target_patch_size, target_patch_size, 3), dtype=np.uint8)

        def load_and_store_patch(index: int) -> None:
            start_loc = tissue_coordinates[index]
            patches[index] = self.load_tile_thread(start_loc, self.tileSize, target_patch_size)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            executor.map(load_and_store_patch, range(num_patches))

        return patches.astype(np.float32)

    def get_patch_coords(self) -> np.ndarray:
        """Return coordinates of all suitable tile addresses."""
        coords = []
        for address in self.suitableTileAddresses():
            coords.append((self.tileDictionary[address]["x"], self.tileDictionary[address]["y"]))
        return np.array(coords) if coords else np.zeros((0, 2))

    # ------------------------------------------------------------------
    # Legacy visualise (old-style, writes to disk)
    # ------------------------------------------------------------------

    def _legacy_visualize(self) -> None:
        """Reproduce the old ``Slide.visualize()`` behaviour (save PNG)."""
        import matplotlib.pyplot as plt

        os.makedirs(self.path_to_store_visualization, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(np.asarray(self.slide))
        ax[0].set_title("original")
        pred_map = getattr(self, "predictionMap", None)
        if pred_map is not None:
            ax[1].imshow(pred_map)
            ax[1].set_title("deep tissue detection")
        else:
            ax[1].set_title("(no tissue detection)")
        plt.savefig(
            f"{self.path_to_store_visualization}/{self.path.stem}.png",
            dpi=300,
        )
        plt.close(fig)

    # ------------------------------------------------------------------
    # Internal proxy classes for backward compatibility
    # ------------------------------------------------------------------

    class _LegacySlideRegion:
        """Thin proxy that wraps a NumPy array to mimic the object returned
        by ``CuImage.read_region`` -- which had ``.width``, ``.height``,
        ``.metadata``, and was convertible via ``np.asarray()``.  It also
        supports ``read_region()`` calls (used by the old ``getTile``).
        """

        def __init__(self, array: np.ndarray, slide: "Slide"):
            self._array = array
            self._slide = slide
            self.height: int = array.shape[0]
            self.width: int = array.shape[1]
            # Build a metadata dict that matches what CuImage exposes so that
            # ``wsi.slide.metadata["cucim"]["shape"]`` keeps working.
            self.metadata: dict = {
                "cucim": {
                    "shape": list(array.shape),
                }
            }

        def read_region(
            self,
            location: Tuple[int, int],
            size: Tuple[int, int],
            level: int = 0,
        ) -> np.ndarray:
            """Read a sub-region from the cached array.

            This method is called by the legacy ``getTile`` on
            ``self.slide.read_region(...)``.  *location* and *size* are in
            the coordinate system of the cached region (i.e. the selected
            pyramid level).
            """
            x, y = int(location[0]), int(location[1])
            w, h = int(size[0]), int(size[1])
            arr = self._array
            # Clip to valid range
            y_end = min(y + h, arr.shape[0])
            x_end = min(x + w, arr.shape[1])
            region = arr[y:y_end, x:x_end]
            return region

        # Support ``np.asarray(slide.slide)``
        def __array__(self, dtype=None):
            if dtype is not None:
                return self._array.astype(dtype)
            return self._array

        # Support iteration / len expected by some callers
        def __repr__(self) -> str:
            return f"<_LegacySlideRegion shape={self._array.shape} " f"dtype={self._array.dtype}>"

    class _LegacyImgProxy:
        """Proxy for ``wsi.img`` attribute access.

        Exposes ``resolutions``, ``metadata``, ``raw_metadata``, and
        ``read_region()`` so that :class:`PathologyProcessor` code like
        ``wsi.img.resolutions["level_dimensions"]`` keeps working.
        """

        def __init__(self, slide: "Slide"):
            self._slide = slide

        @property
        def resolutions(self) -> dict:
            s = self._slide
            return {
                "level_count": s._backend.get_level_count(s._handle),
                "level_dimensions": s._backend.get_level_dimensions(s._handle),
                "level_downsamples": s._backend.get_level_downsamples(s._handle),
            }

        @property
        def metadata(self) -> dict:
            props = self._slide._backend.get_properties(self._slide._handle)
            # Mimic the CuImage metadata shape expected by callers.
            return {
                "cucim": {
                    "shape": list(self._slide.dimensions) + [3],
                    "objectivePower": props.get("magnification"),
                    "spacing": [props.get("mpp_x"), props.get("mpp_y")],
                    "vendor": props.get("vendor"),
                },
                "objectivePower": props.get("magnification"),
                "magnification": props.get("magnification"),
                "scanner": props.get("vendor"),
                "vendor": props.get("vendor"),
            }

        @property
        def raw_metadata(self) -> Any:
            props = self._slide._backend.get_properties(self._slide._handle)
            return props.get("raw_metadata", "")

        def read_region(
            self,
            location: Union[list, tuple] = (0, 0),
            level: int = 0,
            size: Optional[Union[list, tuple]] = None,
        ) -> np.ndarray:
            """Backend-agnostic ``read_region`` matching the old
            ``CuImage.read_region`` call signature.

            When *size* is ``None`` the entire level is read.
            """
            s = self._slide
            if size is None:
                dims = s._backend.get_level_dimensions(s._handle)
                w, h = dims[level]
                size = (w, h)
            loc = (int(location[0]), int(location[1]))
            sz = (int(size[0]), int(size[1]))
            return s._backend.read_region(s._handle, location=loc, level=level, size=sz)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        w, h = self.dimensions
        return (
            f"<Slide path='{self.path.name}' "
            f"dimensions=({w}, {h}) "
            f"levels={self.level_count} "
            f"backend='{self._backend.name}'>"
        )
