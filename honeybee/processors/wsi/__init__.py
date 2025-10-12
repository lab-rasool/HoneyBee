"""
HoneyBee WSI Processing Module

Provides comprehensive whole slide image (WSI) processing capabilities including
tissue detection, stain normalization, and embedding generation.
"""

# Re-export preprocessing components for backward compatibility
from .stain_normalization import (
    ReinhardNormalizer,
    MacenkoNormalizer,
    VahadaneNormalizer,
    ColorAugmenter,
    normalize_reinhard,
    normalize_macenko,
    normalize_vahadane,
    normalize_stain_tissue_aware,
    STAIN_NORM_TARGETS
)

from .stain_separation import (
    StainSeparator,
    separate_stains,
    get_stain_concentrations,
    visualize_stains
)

from .tissue_detection import (
    ClassicalTissueDetector,
    detect_tissue,
    get_tissue_bounding_boxes,
    tissue_mask_to_contours
)

__all__ = [
    # Stain normalization
    'ReinhardNormalizer',
    'MacenkoNormalizer',
    'VahadaneNormalizer',
    'ColorAugmenter',
    'normalize_reinhard',
    'normalize_macenko',
    'normalize_vahadane',
    'normalize_stain_tissue_aware',
    'STAIN_NORM_TARGETS',
    
    # Stain separation
    'StainSeparator',
    'separate_stains',
    'get_stain_concentrations',
    'visualize_stains',
    
    # Tissue detection
    'ClassicalTissueDetector',
    'detect_tissue',
    'get_tissue_bounding_boxes',
    'tissue_mask_to_contours'
]