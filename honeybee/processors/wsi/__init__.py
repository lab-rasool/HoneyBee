"""
HoneyBee WSI Processing Module

Provides comprehensive whole slide image (WSI) processing capabilities including
tissue detection, stain normalization, and embedding generation.
"""

# Re-export preprocessing components for backward compatibility
from .patch_extractor import PatchExtractor
from .patches import Patches, compute_patch_quality
from .stain_normalization import (
    STAIN_NORM_TARGETS,
    ColorAugmenter,
    MacenkoNormalizer,
    ReinhardNormalizer,
    VahadaneNormalizer,
    normalize_macenko,
    normalize_reinhard,
    normalize_stain_tissue_aware,
    normalize_vahadane,
)
from .stain_separation import (
    StainSeparator,
    get_stain_concentrations,
    separate_stains,
    visualize_stains,
)
from .tissue_detection import (
    ClassicalTissueDetector,
    detect_tissue,
    get_tissue_bounding_boxes,
    tissue_mask_to_contours,
)

__all__ = [
    # Stain normalization
    "ReinhardNormalizer",
    "MacenkoNormalizer",
    "VahadaneNormalizer",
    "ColorAugmenter",
    "normalize_reinhard",
    "normalize_macenko",
    "normalize_vahadane",
    "normalize_stain_tissue_aware",
    "STAIN_NORM_TARGETS",
    # Stain separation
    "StainSeparator",
    "separate_stains",
    "get_stain_concentrations",
    "visualize_stains",
    # Tissue detection
    "ClassicalTissueDetector",
    "detect_tissue",
    "get_tissue_bounding_boxes",
    "tissue_mask_to_contours",
    # Patch extraction
    "PatchExtractor",
    "Patches",
    "compute_patch_quality",
]
