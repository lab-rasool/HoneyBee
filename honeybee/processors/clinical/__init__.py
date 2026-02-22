"""
HoneyBee Clinical Processor subpackage.

Provides ClinicalProcessor for clinical text processing, entity extraction,
ontology normalization, and interoperability with FHIR/HL7 standards.
"""

# Re-export ontology mappings for backward compatibility
from .ontologies import ONTOLOGY_MAPPINGS
from .processor import (
    BIOMEDICAL_MODELS,
    CANCER_PATTERNS,
    CLINICAL_DOCUMENT_TYPES,
    MEDICAL_ABBREVIATIONS,
    SUPPORTED_EHR_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
    ClinicalProcessor,
)

__all__ = [
    "ClinicalProcessor",
    "BIOMEDICAL_MODELS",
    "CANCER_PATTERNS",
    "CLINICAL_DOCUMENT_TYPES",
    "MEDICAL_ABBREVIATIONS",
    "ONTOLOGY_MAPPINGS",
    "SUPPORTED_EHR_FORMATS",
    "SUPPORTED_IMAGE_FORMATS",
]
