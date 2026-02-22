"""
FHIR R4 resource construction and parsing for clinical data interoperability.

Builds FHIR-compliant JSON dicts. When fhir.resources is installed,
resources can be validated via validate().

Install optional validation with: pip install honeybee-ml[clinical-interop]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

try:
    from fhir.resources.bundle import Bundle

    _FHIR_AVAILABLE = True
except ImportError:
    _FHIR_AVAILABLE = False

# Ontology system URIs
ONTOLOGY_SYSTEM_URIS = {
    "snomed_ct": "http://snomed.info/sct",
    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "loinc": "http://loinc.org",
}

# Entity type to FHIR resource type mapping
ENTITY_FHIR_MAP = {
    "condition": "Condition",
    "tumor": "Condition",
    "medication": "MedicationStatement",
    "measurement": "Observation",
    "biomarker": "Observation",
    "staging": "Observation",
    "response": "Observation",
    "temporal": None,
    "dosage": None,
    "anatomy": None,
    "procedure": None,
}


class FHIRConverter:
    """Convert between HoneyBee processor output and FHIR R4 resources.

    Produces plain JSON-serializable dicts that conform to FHIR R4.
    Does NOT require fhir.resources at runtime. When fhir.resources is
    installed, use validate() to validate output against the spec.
    """

    def to_fhir_bundle(
        self,
        result: Dict,
        patient_id: Optional[str] = None,
    ) -> Dict:
        """Convert ClinicalProcessor output to a FHIR R4 Bundle.

        Args:
            result: Output from ClinicalProcessor.process_text() or process().
            patient_id: Optional patient reference ID.

        Returns:
            FHIR Bundle as a JSON-serializable dict.
        """
        entries: List[Dict] = []
        patient_ref = (
            {"reference": f"Patient/{patient_id}"} if patient_id else None
        )

        # Convert entities to FHIR resources
        for entity in result.get("entities", []):
            resource = self._entity_to_resource(entity, patient_ref)
            if resource is not None:
                entries.append({
                    "fullUrl": f"urn:uuid:{uuid4()}",
                    "resource": resource,
                    "request": {
                        "method": "POST",
                        "url": resource["resourceType"],
                    },
                })

        # Add DiagnosticReport if text is available
        if result.get("text"):
            report = self._create_diagnostic_report(result, patient_ref)
            entries.append({
                "fullUrl": f"urn:uuid:{uuid4()}",
                "resource": report,
                "request": {
                    "method": "POST",
                    "url": "DiagnosticReport",
                },
            })

        bundle = {
            "resourceType": "Bundle",
            "type": "transaction",
        }
        if entries:
            bundle["entry"] = entries

        return bundle

    def _entity_to_resource(
        self,
        entity: Dict,
        patient_ref: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Convert a single entity to the appropriate FHIR resource dict."""
        entity_type = entity.get("type", "")
        fhir_type = ENTITY_FHIR_MAP.get(entity_type)

        if fhir_type is None:
            return None

        code = self._build_codeable_concept(entity)

        if fhir_type == "Condition":
            return self._create_condition(entity, code, patient_ref)
        elif fhir_type == "MedicationStatement":
            return self._create_medication_statement(entity, code, patient_ref)
        elif fhir_type == "Observation":
            return self._create_observation(entity, code, patient_ref)

        return None

    def _build_codeable_concept(self, entity: Dict) -> Dict:
        """Build a CodeableConcept dict from entity text and ontology links."""
        codings = []
        ontology_links = entity.get("properties", {}).get(
            "ontology_links", []
        )

        for link in ontology_links:
            system_uri = ONTOLOGY_SYSTEM_URIS.get(link["ontology"], "")
            codings.append({
                "system": system_uri,
                "code": link["concept_id"],
                "display": link["concept_name"],
            })

        concept: Dict[str, Any] = {"text": entity.get("text", "")}
        if codings:
            concept["coding"] = codings
        return concept

    def _create_condition(
        self,
        entity: Dict,
        code: Dict,
        patient_ref: Optional[Dict],
    ) -> Dict:
        """Create a FHIR Condition resource dict."""
        resource: Dict[str, Any] = {
            "resourceType": "Condition",
            "code": code,
            "clinicalStatus": {
                "coding": [{
                    "system": (
                        "http://terminology.hl7.org/CodeSystem"
                        "/condition-clinical"
                    ),
                    "code": "active",
                }]
            },
        }
        if patient_ref:
            resource["subject"] = patient_ref
        return resource

    def _create_medication_statement(
        self,
        entity: Dict,
        code: Dict,
        patient_ref: Optional[Dict],
    ) -> Dict:
        """Create a FHIR MedicationStatement resource dict."""
        resource: Dict[str, Any] = {
            "resourceType": "MedicationStatement",
            "status": "active",
            "medicationCodeableConcept": code,
        }
        if patient_ref:
            resource["subject"] = patient_ref
        return resource

    def _create_observation(
        self,
        entity: Dict,
        code: Dict,
        patient_ref: Optional[Dict],
    ) -> Dict:
        """Create a FHIR Observation resource dict."""
        resource: Dict[str, Any] = {
            "resourceType": "Observation",
            "status": "final",
            "code": code,
        }
        if patient_ref:
            resource["subject"] = patient_ref

        props = entity.get("properties", {})
        if props.get("value"):
            resource["valueString"] = (
                f"{props['value']} {props.get('unit', '')}"
            ).strip()

        return resource

    def _create_diagnostic_report(
        self,
        result: Dict,
        patient_ref: Optional[Dict],
    ) -> Dict:
        """Create a DiagnosticReport dict from the full result."""
        text_content = result.get("text", "")[:5000]
        resource: Dict[str, Any] = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {"text": "Clinical NLP Analysis"},
            "conclusion": text_content,
        }
        if patient_ref:
            resource["subject"] = patient_ref
        return resource

    # --- Inbound (FHIR -> HoneyBee) ---

    def from_fhir_bundle(self, bundle_json: Dict) -> Dict:
        """Extract narrative text from a FHIR Bundle for NLP processing.

        Args:
            bundle_json: FHIR Bundle as dict.

        Returns:
            Dict with 'text' (combined narrative), 'resources' (parsed),
            and 'resource_count'.
        """
        texts = []
        resources = []

        for entry in bundle_json.get("entry", []):
            resource = entry.get("resource", {})
            parsed = self.from_fhir_resource(resource)
            if parsed.get("text"):
                texts.append(parsed["text"])
            resources.append(parsed)

        return {
            "text": " ".join(texts),
            "resources": resources,
            "resource_count": len(resources),
        }

    def from_fhir_resource(self, resource_json: Dict) -> Dict:
        """Parse a single FHIR resource to extract clinical text.

        Args:
            resource_json: Single FHIR resource as dict.

        Returns:
            Dict with 'resource_type', 'text', and extracted fields.
        """
        resource_type = resource_json.get("resourceType", "Unknown")
        result: Dict[str, Any] = {"resource_type": resource_type, "text": ""}

        if resource_type == "Condition":
            code = resource_json.get("code", {})
            result["text"] = code.get("text", "")
            if not result["text"] and code.get("coding"):
                result["text"] = code["coding"][0].get("display", "")
            result["clinical_status"] = (
                resource_json.get("clinicalStatus", {})
                .get("coding", [{}])[0]
                .get("code", "")
            )

        elif resource_type == "MedicationStatement":
            med = resource_json.get("medicationCodeableConcept", {})
            result["text"] = med.get("text", "")
            if not result["text"] and med.get("coding"):
                result["text"] = med["coding"][0].get("display", "")
            result["status"] = resource_json.get("status", "")

        elif resource_type == "Observation":
            code = resource_json.get("code", {})
            result["text"] = code.get("text", "")
            if not result["text"] and code.get("coding"):
                result["text"] = code["coding"][0].get("display", "")
            if resource_json.get("valueString"):
                result["text"] += f": {resource_json['valueString']}"
            result["status"] = resource_json.get("status", "")

        elif resource_type == "DiagnosticReport":
            result["text"] = resource_json.get("conclusion", "")
            result["status"] = resource_json.get("status", "")

        else:
            # Try to extract text from narrative
            narrative = resource_json.get("text", {})
            result["text"] = narrative.get("div", "")

        return result

    def validate(self, bundle_dict: Dict) -> bool:
        """Validate a bundle dict against FHIR R4 spec using fhir.resources.

        Requires fhir.resources to be installed.

        Returns:
            True if valid.

        Raises:
            ImportError: If fhir.resources is not installed.
            ValidationError: If the bundle is invalid.
        """
        if not _FHIR_AVAILABLE:
            raise ImportError(
                "fhir.resources is required for FHIR validation. "
                "Install with: pip install 'fhir.resources>=7.0.0' "
                "or: pip install 'honeybee-ml[clinical-interop]'"
            )
        Bundle.model_validate(bundle_dict)
        return True
