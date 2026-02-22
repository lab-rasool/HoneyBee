"""
Clinical data interoperability module.

Provides FHIR R4 and HL7 v2 support for clinical data exchange.
"""

from .fhir_converter import _FHIR_AVAILABLE, FHIRConverter
from .hl7_parser import _HL7_AVAILABLE, HL7Parser

__all__ = [
    "FHIRConverter",
    "HL7Parser",
    "_FHIR_AVAILABLE",
    "_HL7_AVAILABLE",
]
