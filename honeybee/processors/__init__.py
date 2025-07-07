"""
HoneyBee Processors Module

Provides unified interfaces for different data modality processors.
"""

from .clinical_processor import (
    ClinicalProcessor,
)
# Import from new modular structure
from .radiology import RadiologyProcessor

# Legacy imports no longer available from radiology_processor
RadiologyImage = None
create_radimagenet_processor = None
from .pathology_processor import (
    PathologyProcessor,
    process_pathology,
)

__all__ = [
    "ClinicalProcessor",
    "RadiologyProcessor",
    "RadiologyImage",
    "create_radimagenet_processor",
    "PathologyProcessor",
    "process_pathology",
]
