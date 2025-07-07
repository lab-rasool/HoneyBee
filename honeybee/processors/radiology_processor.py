"""
Radiology Processor - Legacy Compatibility Layer

This module provides backward compatibility for existing code using RadiologyProcessor.
It imports from the new modular structure while maintaining the original API.
"""

import warnings
from .radiology import RadiologyProcessor

# Show deprecation warning
warnings.warn(
    "Importing from 'radiology_processor' is deprecated. "
    "Please import from 'honeybee.processors.radiology' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Export for backward compatibility
__all__ = ['RadiologyProcessor']