"""
Shared rasterization helpers for WSI visualization.

Used by Slide, PatchExtractor, and PathologyProcessor plot methods to map
patch-level data onto slide thumbnails.
"""

from __future__ import annotations

import numpy as np


def _rasterize_patches(
    coordinates: np.ndarray,
    values: np.ndarray,
    slide_dimensions: tuple,
    thumbnail_size: tuple,
    alpha: float = 0.85,
) -> np.ndarray:
    """Map (N,4) coords + (N,3) RGB values to an RGBA overlay at thumbnail resolution.

    Parameters
    ----------
    coordinates : np.ndarray
        Shape ``(N, 4)`` with columns ``[x, y, w, h]`` in level-0 space.
    values : np.ndarray
        Shape ``(N, 3)`` with RGB float values in [0, 1].
    slide_dimensions : (int, int)
        ``(width, height)`` of the slide at level 0.
    thumbnail_size : (int, int)
        ``(height, width)`` of the target thumbnail/overlay.
    alpha : float
        Alpha value for filled patch regions (0-1).

    Returns
    -------
    np.ndarray
        RGBA overlay with shape ``(H, W, 4)``, dtype ``float32``, values in [0, 1].
    """
    thumb_h, thumb_w = thumbnail_size
    slide_w, slide_h = slide_dimensions

    overlay = np.zeros((thumb_h, thumb_w, 4), dtype=np.float32)

    if len(coordinates) == 0 or slide_w == 0 or slide_h == 0:
        return overlay

    scale_x = thumb_w / slide_w
    scale_y = thumb_h / slide_h

    for i in range(len(coordinates)):
        x, y, w, h = coordinates[i]
        x1 = int(round(x * scale_x))
        y1 = int(round(y * scale_y))
        x2 = int(round((x + w) * scale_x))
        y2 = int(round((y + h) * scale_y))
        x1, x2 = max(0, x1), min(thumb_w, x2)
        y1, y2 = max(0, y1), min(thumb_h, y2)
        if x2 > x1 and y2 > y1:
            overlay[y1:y2, x1:x2, :3] = values[i]
            overlay[y1:y2, x1:x2, 3] = alpha

    return overlay


def _composite_overlay(
    thumbnail: np.ndarray,
    overlay: np.ndarray,
    tissue_blend: float = 0.15,
) -> np.ndarray:
    """Alpha-composite an RGBA overlay onto an RGB thumbnail.

    Parameters
    ----------
    thumbnail : np.ndarray
        RGB image, uint8 shape ``(H, W, 3)`` or float32 in [0, 1].
    overlay : np.ndarray
        RGBA overlay, float32 shape ``(H, W, 4)`` with values in [0, 1].
    tissue_blend : float
        How much of the underlying tissue to blend through where the overlay
        has data.  ``0.0`` = pure overlay color, ``1.0`` = pure tissue.

    Returns
    -------
    np.ndarray
        Composited RGB image, float32 in [0, 1], shape ``(H, W, 3)``.
    """
    if thumbnail.dtype == np.uint8:
        thumb_f = thumbnail.astype(np.float32) / 255.0
    else:
        thumb_f = thumbnail.astype(np.float32)

    alpha = overlay[:, :, 3:4]
    composite = (1.0 - alpha) * thumb_f + alpha * (
        tissue_blend * thumb_f + (1.0 - tissue_blend) * overlay[:, :, :3]
    )
    return np.clip(composite, 0, 1)
