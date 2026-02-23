"""
FHIR R4 resource construction and parsing for clinical data interoperability.

Builds FHIR-compliant JSON dicts. When fhir.resources is installed,
resources can be validated via validate().

Install optional validation with: pip install fhir.resources>=7.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from ..types import ClinicalEntity, ClinicalResult

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
    "icd10cm": "http://hl7.org/fhir/sid/icd-10-cm",
    "umls": "http://www.nlm.nih.gov/research/umls",
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
    """Convert between HoneyBee ClinicalResult and FHIR R4 resources."""

    def to_fhir_bundle(
        self,
        result: Union[ClinicalResult, Dict],
        patient_id: Optional[str] = None,
    ) -> Dict:
        """Convert ClinicalResult (or legacy dict) to a FHIR R4 Bundle."""
        entries: List[Dict] = []
        patient_ref = (
            {"reference": f"Patient/{patient_id}"} if patient_id else None
        )

        # Support both ClinicalResult and legacy dict
        if isinstance(result, ClinicalResult):
            entities = result.entities
            text_content = result.text
        else:
            entities = [
                self._dict_to_entity(e) for e in result.get("entities", [])
            ]
            text_content = result.get("text", "")

        for entity in entities:
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
        if text_content:
            report = self._create_diagnostic_report(text_content, patient_ref)
            entries.append({
                "fullUrl": f"urn:uuid:{uuid4()}",
                "resource": report,
                "request": {
                    "method": "POST",
                    "url": "DiagnosticReport",
                },
            })

        bundle: Dict[str, Any] = {
            "resourceType": "Bundle",
            "type": "transaction",
        }
        if entries:
            bundle["entry"] = entries

        return bundle

    def _entity_to_resource(
        self,
        entity: ClinicalEntity,
        patient_ref: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Convert a ClinicalEntity to the appropriate FHIR resource dict."""
        fhir_type = ENTITY_FHIR_MAP.get(entity.type)
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

    def _build_codeable_concept(self, entity: ClinicalEntity) -> Dict:
        """Build a CodeableConcept from entity text and ontology codes."""
        codings = []

        # Use OntologyCode objects from the new pipeline
        for oc in entity.ontology_codes:
            system_uri = ONTOLOGY_SYSTEM_URIS.get(oc.system, "")
            codings.append({
                "system": system_uri,
                "code": oc.code,
                "display": oc.display,
            })

        # Also check legacy ontology_links in properties
        for link in entity.properties.get("ontology_links", []):
            system_uri = ONTOLOGY_SYSTEM_URIS.get(link.get("ontology", ""), "")
            codings.append({
                "system": system_uri,
                "code": link.get("concept_id", ""),
                "display": link.get("concept_name", ""),
            })

        concept: Dict[str, Any] = {"text": entity.text}
        if codings:
            concept["coding"] = codings
        return concept

    def _create_condition(
        self,
        entity: ClinicalEntity,
        code: Dict,
        patient_ref: Optional[Dict],
    ) -> Dict:
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
        entity: ClinicalEntity,
        code: Dict,
        patient_ref: Optional[Dict],
    ) -> Dict:
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
        entity: ClinicalEntity,
        code: Dict,
        patient_ref: Optional[Dict],
    ) -> Dict:
        resource: Dict[str, Any] = {
            "resourceType": "Observation",
            "status": "final",
            "code": code,
        }
        if patient_ref:
            resource["subject"] = patient_ref

        props = entity.properties
        if props.get("value"):
            resource["valueString"] = (
                f"{props['value']} {props.get('unit', '')}"
            ).strip()

        return resource

    def _create_diagnostic_report(
        self,
        text_content: str,
        patient_ref: Optional[Dict],
    ) -> Dict:
        resource: Dict[str, Any] = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "code": {"text": "Clinical NLP Analysis"},
            "conclusion": text_content[:5000],
        }
        if patient_ref:
            resource["subject"] = patient_ref
        return resource

    # --- Inbound (FHIR -> HoneyBee) ---

    def from_fhir_bundle(self, bundle_json: Dict) -> Dict:
        """Extract narrative text from a FHIR Bundle for NLP processing."""
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
        """Parse a single FHIR resource to extract clinical text."""
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
            narrative = resource_json.get("text", {})
            result["text"] = narrative.get("div", "")

        return result

    def validate(self, bundle_dict: Dict) -> bool:
        """Validate a bundle dict against FHIR R4 spec using fhir.resources."""
        if not _FHIR_AVAILABLE:
            raise ImportError(
                "fhir.resources is required for FHIR validation. "
                "Install with: pip install 'fhir.resources>=7.0.0'"
            )
        Bundle.model_validate(bundle_dict)
        return True

    # --- Helpers ---

    @staticmethod
    def _dict_to_entity(d: Dict) -> ClinicalEntity:
        """Convert a legacy entity dict to ClinicalEntity."""
        return ClinicalEntity(
            text=d.get("text", ""),
            type=d.get("type", ""),
            start=d.get("start", 0),
            end=d.get("end", 0),
            confidence=d.get("confidence", 1.0),
            properties=d.get("properties", {}),
        )
