"""
Backward-compatibility shim for honeybee.processors.clinical_processor.

All functionality has been moved to honeybee.processors.clinical.processor.
This module re-exports everything so existing imports continue to work.
"""

from .clinical.processor import *  # noqa: F401,F403
from .clinical.processor import ClinicalProcessor  # noqa: F401
from .clinical.ontologies import ONTOLOGY_MAPPINGS  # noqa: F401
from .clinical.processor import (  # noqa: F401
    BIOMEDICAL_MODELS,
    CANCER_PATTERNS,
    CLINICAL_DOCUMENT_TYPES,
    MEDICAL_ABBREVIATIONS,
    SUPPORTED_EHR_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
)
