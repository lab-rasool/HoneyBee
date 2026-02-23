"""
Backward-compatible wrapper for RadiologyProcessor.

This module re-exports RadiologyProcessor from its new location.
It will be removed in a future version.
"""

import warnings

warnings.warn(
    "Importing from honeybee.processors.radiology_processor is deprecated. "
    "Use honeybee.processors.radiology instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .radiology import RadiologyProcessor  # noqa: E402, F401

__all__ = ["RadiologyProcessor"]
