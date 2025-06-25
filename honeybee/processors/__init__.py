"""
HoneyBee Processors Module

Provides unified interfaces for different data modality processors.
"""

from .clinical_processor import (
    ClinicalProcessor,
)
from .radiology_processor import (
    RadiologyProcessor,
    RadiologyImage,
)

__all__ = [
    "ClinicalProcessor",
    "RadiologyProcessor",
    "RadiologyImage",
]
