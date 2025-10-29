"""
HoneyBee Processors Module

Provides unified interfaces for different data modality processors.
"""

from .clinical_processor import (
    ClinicalProcessor,
)
from .pathology_processor import PathologyProcessor

# Import from new modular structure
from .radiology import RadiologyProcessor

# Legacy imports no longer available from radiology_processor
RadiologyImage = None
create_radimagenet_processor = None


__all__ = [
    "ClinicalProcessor",
    "RadiologyProcessor",
    "PathologyProcessor",
    "RadiologyImage",
    "create_radimagenet_processor",
]
