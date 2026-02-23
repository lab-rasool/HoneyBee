"""
HL7 v2 message parsing for clinical data interoperability.

Requires optional dependency: hl7apy>=1.3.4
Install with: pip install hl7apy
"""

import logging
from typing import Dict

from ..types import ClinicalDocument

logger = logging.getLogger(__name__)

try:
    from hl7apy.parser import parse_message  # noqa: F401

    _HL7_AVAILABLE = True
except ImportError:
    _HL7_AVAILABLE = False


def _require_hl7():
    """Raise ImportError if hl7apy is not available."""
    if not _HL7_AVAILABLE:
        raise ImportError(
            "hl7apy is required for HL7 v2 support. "
            "Install with: pip install 'hl7apy>=1.3.4'"
        )


# Segment extraction helpers

def _safe_field(segment, field_idx: int, default: str = "") -> str:
    """Safely extract a field value from a segment string."""
    parts = segment.split("|")
    if field_idx < len(parts):
        return parts[field_idx].strip()
    return default


def _parse_pid(segment: str) -> Dict:
    """Parse PID (Patient Identification) segment."""
    return {
        "patient_id": _safe_field(segment, 3),
        "patient_name": _safe_field(segment, 5),
        "date_of_birth": _safe_field(segment, 7),
        "sex": _safe_field(segment, 8),
    }


def _parse_obx(segment: str) -> Dict:
    """Parse OBX (Observation/Result) segment."""
    return {
        "observation_id": _safe_field(segment, 3),
        "value_type": _safe_field(segment, 2),
        "value": _safe_field(segment, 5),
        "units": _safe_field(segment, 6),
        "reference_range": _safe_field(segment, 7),
        "abnormal_flags": _safe_field(segment, 8),
        "status": _safe_field(segment, 11),
    }


def _parse_dg1(segment: str) -> Dict:
    """Parse DG1 (Diagnosis) segment."""
    return {
        "diagnosis_code": _safe_field(segment, 3),
        "description": _safe_field(segment, 4),
        "diagnosis_type": _safe_field(segment, 6),
    }


def _parse_rxa(segment: str) -> Dict:
    """Parse RXA (Pharmacy/Treatment Administration) segment."""
    return {
        "administration_datetime": _safe_field(segment, 3),
        "administered_code": _safe_field(segment, 5),
        "administered_amount": _safe_field(segment, 6),
        "administered_units": _safe_field(segment, 7),
    }


# Segment type to parser mapping
_SEGMENT_PARSERS = {
    "PID": _parse_pid,
    "OBX": _parse_obx,
    "DG1": _parse_dg1,
    "RXA": _parse_rxa,
}


class HL7Parser:
    """Parse HL7 v2 messages and extract clinical data."""

    def __init__(self):
        _require_hl7()

    def parse(self, message: str) -> Dict:
        """Parse an HL7 v2 message.

        Returns:
            Dict with 'message_type', 'segments', 'text', and 'document'.
            The 'document' key contains a ClinicalDocument.
        """
        _require_hl7()

        # Normalize line endings
        message = message.strip().replace("\n", "\r")
        segments_raw = [
            s for s in message.split("\r") if s.strip()
        ]

        result = {
            "message_type": "",
            "patient": {},
            "observations": [],
            "diagnoses": [],
            "medications": [],
            "segments": {},
            "text": "",
        }

        for seg_str in segments_raw:
            seg_type = seg_str[:3]

            if seg_type == "MSH":
                result["message_type"] = _safe_field(seg_str, 8)

            parser = _SEGMENT_PARSERS.get(seg_type)
            if parser:
                parsed = parser(seg_str)

                if seg_type == "PID":
                    result["patient"] = parsed
                elif seg_type == "OBX":
                    result["observations"].append(parsed)
                elif seg_type == "DG1":
                    result["diagnoses"].append(parsed)
                elif seg_type == "RXA":
                    result["medications"].append(parsed)

                if seg_type not in result["segments"]:
                    result["segments"][seg_type] = []
                result["segments"][seg_type].append(parsed)

        # Generate text for NLP processing
        result["text"] = self.to_text(result)

        # Also provide a ClinicalDocument
        result["document"] = ClinicalDocument(
            text=result["text"],
            metadata={
                "source_type": "hl7",
                "message_type": result["message_type"],
            },
        )

        return result

    def to_text(self, parsed: Dict) -> str:
        """Convert parsed HL7 data to clinical text for NLP."""
        parts = []

        patient = parsed.get("patient", {})
        if patient.get("patient_name"):
            parts.append(f"Patient: {patient['patient_name']}")
        if patient.get("date_of_birth"):
            parts.append(f"DOB: {patient['date_of_birth']}")
        if patient.get("sex"):
            parts.append(f"Sex: {patient['sex']}")

        for dx in parsed.get("diagnoses", []):
            desc = dx.get("description", "")
            code = dx.get("diagnosis_code", "")
            if desc:
                parts.append(f"Diagnosis: {desc} ({code})")

        for obs in parsed.get("observations", []):
            obs_id = obs.get("observation_id", "")
            value = obs.get("value", "")
            units = obs.get("units", "")
            if obs_id and value:
                parts.append(f"{obs_id}: {value} {units}".strip())

        for med in parsed.get("medications", []):
            code = med.get("administered_code", "")
            amount = med.get("administered_amount", "")
            units = med.get("administered_units", "")
            if code:
                parts.append(
                    f"Medication: {code} {amount} {units}".strip()
                )

        return ". ".join(parts)
