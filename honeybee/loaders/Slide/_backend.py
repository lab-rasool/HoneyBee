"""
WSI Backend Abstraction Layer

Normalizes the API differences between CuImage (cucim) and OpenSlide behind a
common interface. All backends guarantee that ``read_region`` returns an RGB
``np.ndarray`` with dtype ``uint8`` and shape ``(H, W, 3)``.

Usage:
    >>> from honeybee.loaders.Slide._backend import get_backend
    >>> backend = get_backend()            # auto-detect best available backend
    >>> handle = backend.open("slide.svs")
    >>> region = backend.read_region(handle, location=(0, 0), level=0, size=(512, 512))
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class WSIBackend(ABC):
    """Abstract interface for whole-slide-image reading backends.

    Concrete subclasses wrap a specific WSI library (e.g. cucim, openslide)
    and expose a uniform API so that downstream code never needs to know
    which library is actually being used.
    """

    # Human-readable name for logging / repr
    name: str = "base"

    @abstractmethod
    def open(self, path: str) -> Any:
        """Open a whole-slide image and return a library-specific handle.

        Args:
            path: Filesystem path to the WSI file (.svs, .tiff, .ndpi, etc.).

        Returns:
            A backend-specific slide handle that must be passed to every other
            method on this class.
        """

    @abstractmethod
    def read_region(
        self,
        handle: Any,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """Read a rectangular region from the slide.

        Args:
            handle: Slide handle returned by :meth:`open`.
            location: ``(x, y)`` pixel coordinates of the top-left corner
                at **level 0** (full resolution).
            level: Pyramid level to read from.
            size: ``(width, height)`` of the region **at the requested level**.

        Returns:
            RGB image as ``np.ndarray`` with dtype ``uint8`` and shape
            ``(height, width, 3)``.
        """

    @abstractmethod
    def get_level_count(self, handle: Any) -> int:
        """Return the number of pyramid levels in the slide.

        Args:
            handle: Slide handle returned by :meth:`open`.

        Returns:
            Number of levels (>= 1).
        """

    @abstractmethod
    def get_level_dimensions(self, handle: Any) -> List[Tuple[int, int]]:
        """Return ``(width, height)`` for every pyramid level.

        Args:
            handle: Slide handle returned by :meth:`open`.

        Returns:
            List of ``(width, height)`` tuples, one per level, ordered from
            highest to lowest resolution.
        """

    @abstractmethod
    def get_level_downsamples(self, handle: Any) -> List[float]:
        """Return the downsample factor for every pyramid level.

        Args:
            handle: Slide handle returned by :meth:`open`.

        Returns:
            List of floats where ``downsamples[0]`` is always ``1.0``.
        """

    @abstractmethod
    def get_properties(self, handle: Any) -> Dict[str, Any]:
        """Return slide metadata as a normalised dictionary.

        The dictionary always contains the following keys (values may be
        ``None`` when the information is unavailable in the file):

        - ``"vendor"``        -- scanner manufacturer string
        - ``"magnification"`` -- objective power (e.g. ``40.0``)
        - ``"mpp_x"``        -- microns per pixel along X
        - ``"mpp_y"``        -- microns per pixel along Y

        Args:
            handle: Slide handle returned by :meth:`open`.

        Returns:
            Dictionary of slide properties.
        """

    @abstractmethod
    def get_thumbnail(self, handle: Any, size: Tuple[int, int]) -> np.ndarray:
        """Return a thumbnail image of the whole slide.

        The thumbnail fits inside the requested ``(width, height)`` bounding
        box while preserving the aspect ratio.

        Args:
            handle: Slide handle returned by :meth:`open`.
            size: Maximum ``(width, height)`` of the thumbnail.

        Returns:
            RGB image as ``np.ndarray`` with dtype ``uint8`` and shape
            ``(H, W, 3)`` where ``H <= size[1]`` and ``W <= size[0]``.
        """


# ---------------------------------------------------------------------------
# CuCIM backend
# ---------------------------------------------------------------------------


class CuCIMBackend(WSIBackend):
    """WSI backend that wraps :class:`cucim.CuImage`.

    CuImage uses a ``resolutions`` dictionary for level information and
    returns RGB data from ``read_region`` by default.
    """

    name = "cucim"

    def open(self, path: str) -> Any:
        from cucim import CuImage

        return CuImage(path)

    def read_region(
        self,
        handle: Any,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> np.ndarray:
        # CuImage.read_region expects location as a list/tuple and size as a
        # list/tuple.  It returns an object that can be converted to ndarray.
        # The returned image is already RGB.
        region = handle.read_region(
            location=list(location),
            level=level,
            size=list(size),
        )
        arr = np.asarray(region, dtype=np.uint8)

        # Defensive: strip alpha channel if unexpectedly present
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        return arr

    def get_level_count(self, handle: Any) -> int:
        return int(handle.resolutions["level_count"])

    def get_level_dimensions(self, handle: Any) -> List[Tuple[int, int]]:
        dims = handle.resolutions["level_dimensions"]
        return [(int(w), int(h)) for w, h in dims]

    def get_level_downsamples(self, handle: Any) -> List[float]:
        return [float(d) for d in handle.resolutions["level_downsamples"]]

    def get_properties(self, handle: Any) -> Dict[str, Any]:
        metadata = handle.metadata if hasattr(handle, "metadata") else {}
        raw = handle.raw_metadata if hasattr(handle, "raw_metadata") else ""

        # CuImage stores magnification and MPP in its metadata dict.
        magnification = _safe_float(metadata.get("cucim", {}).get("objectivePower"))
        mpp_x = _safe_float(metadata.get("cucim", {}).get("spacing", [None])[0])
        mpp_y = _safe_float(metadata.get("cucim", {}).get("spacing", [None, None])[1])

        return {
            "vendor": metadata.get("cucim", {}).get("vendor"),
            "magnification": magnification,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "raw_metadata": raw,
        }

    def get_thumbnail(self, handle: Any, size: Tuple[int, int]) -> np.ndarray:
        # CuImage has no native thumbnail helper.  We read the lowest-resolution
        # level and resize to the requested bounding box.
        level_count = self.get_level_count(handle)
        lowest_level = level_count - 1
        dims = self.get_level_dimensions(handle)
        w, h = dims[lowest_level]

        region = self.read_region(handle, location=(0, 0), level=lowest_level, size=(w, h))

        # Resize while preserving aspect ratio to fit within *size*.
        return _fit_thumbnail(region, size)


# ---------------------------------------------------------------------------
# OpenSlide backend
# ---------------------------------------------------------------------------


class OpenSlideBackend(WSIBackend):
    """WSI backend that wraps :class:`openslide.OpenSlide`.

    OpenSlide returns RGBA :class:`PIL.Image` objects from ``read_region``.
    This backend strips the alpha channel to guarantee RGB output.
    """

    name = "openslide"

    def open(self, path: str) -> Any:
        import openslide

        return openslide.OpenSlide(path)

    def read_region(
        self,
        handle: Any,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> np.ndarray:
        # OpenSlide.read_region returns a PIL RGBA Image.
        pil_img = handle.read_region(location, level, size)
        arr = np.asarray(pil_img, dtype=np.uint8)

        # Strip alpha channel (RGBA -> RGB)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        return arr

    def get_level_count(self, handle: Any) -> int:
        return int(handle.level_count)

    def get_level_dimensions(self, handle: Any) -> List[Tuple[int, int]]:
        return [(int(w), int(h)) for w, h in handle.level_dimensions]

    def get_level_downsamples(self, handle: Any) -> List[float]:
        return [float(d) for d in handle.level_downsamples]

    def get_properties(self, handle: Any) -> Dict[str, Any]:
        props = dict(handle.properties)

        magnification = _safe_float(props.get("openslide.objective-power"))
        mpp_x = _safe_float(props.get("openslide.mpp-x"))
        mpp_y = _safe_float(props.get("openslide.mpp-y"))
        vendor = props.get("openslide.vendor")

        return {
            "vendor": vendor,
            "magnification": magnification,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "raw_metadata": props,
        }

    def get_thumbnail(self, handle: Any, size: Tuple[int, int]) -> np.ndarray:
        # OpenSlide has a native get_thumbnail that returns an RGB PIL Image.
        pil_thumb = handle.get_thumbnail(size)
        arr = np.asarray(pil_thumb.convert("RGB"), dtype=np.uint8)
        return arr


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(backend: Optional[str] = None) -> WSIBackend:
    """Return a :class:`WSIBackend` instance.

    Args:
        backend: Explicit backend name (``"cucim"`` or ``"openslide"``).
            If ``None``, auto-detection is used: cucim is preferred when
            available, with openslide as fallback.

    Returns:
        A ready-to-use :class:`WSIBackend` instance.

    Raises:
        ValueError: If *backend* is not a recognised name.
        ImportError: If no suitable backend library can be imported.
    """
    if backend is not None:
        backend = backend.lower().strip()
        if backend == "cucim":
            return CuCIMBackend()
        if backend == "openslide":
            return OpenSlideBackend()
        raise ValueError(f"Unknown WSI backend '{backend}'. Choose 'cucim' or 'openslide'.")

    # Auto-detect: prefer cucim (GPU-accelerated), fall back to openslide.
    try:
        import cucim  # noqa: F401

        return CuCIMBackend()
    except ImportError:
        pass

    try:
        import openslide  # noqa: F401

        return OpenSlideBackend()
    except ImportError:
        pass

    raise ImportError(
        "No WSI backend found. Install one of the following:\n"
        "  pip install cucim          # GPU-accelerated (Linux only)\n"
        "  pip install openslide-python  # Cross-platform"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    """Convert *value* to float, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fit_thumbnail(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize *image* so it fits inside *size* ``(width, height)`` while
    preserving aspect ratio.

    Args:
        image: RGB ``np.ndarray`` of shape ``(H, W, 3)``.
        size: Target bounding box ``(max_width, max_height)``.

    Returns:
        Resized RGB ``np.ndarray`` with dtype ``uint8``.
    """
    from PIL import Image as PILImage

    max_w, max_h = size
    h, w = image.shape[:2]

    if h == 0 or w == 0:
        return np.zeros((max_h, max_w, 3), dtype=np.uint8)

    scale = min(max_w / w, max_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    pil_img = PILImage.fromarray(image)
    pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
    return np.asarray(pil_img, dtype=np.uint8)
