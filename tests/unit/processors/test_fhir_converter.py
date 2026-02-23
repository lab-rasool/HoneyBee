"""
Unit tests for FHIR R4 converter.

FHIRConverter produces plain dicts — no fhir.resources dependency at runtime.
"""

import pytest

from honeybee.processors.clinical.interop.fhir_converter import (
    FHIRConverter,
    _FHIR_AVAILABLE,
    ENTITY_FHIR_MAP,
    ONTOLOGY_SYSTEM_URIS,
)


@pytest.fixture
def converter():
    return FHIRConverter()


@pytest.fixture
def sample_processor_output():
    """Sample ClinicalProcessor output for conversion."""
    return {
        "text": "Patient diagnosed with breast cancer. ER positive.",
        "entities": [
            {
                "text": "breast cancer",
                "type": "condition",
                "start": 25,
                "end": 38,
                "properties": {
                    "ontology_links": [
                        {
                            "ontology": "snomed_ct",
                            "concept_id": "254838004",
                            "concept_name": "Carcinoma of breast",
                        }
                    ]
                },
            },
            {
                "text": "ER positive",
                "type": "biomarker",
                "start": 40,
                "end": 51,
                "properties": {},
            },
            {
                "text": "tamoxifen 20 mg daily",
                "type": "medication",
                "start": 53,
                "end": 74,
                "properties": {
                    "ontology_links": [
                        {
                            "ontology": "rxnorm",
                            "concept_id": "10324",
                            "concept_name": "Tamoxifen",
                        }
                    ]
                },
            },
        ],
        "entity_relationships": [],
    }


class TestFHIRConverterConstants:
    """Test FHIR converter constants and mappings."""

    def test_entity_fhir_map_has_condition(self):
        assert ENTITY_FHIR_MAP["condition"] == "Condition"

    def test_entity_fhir_map_has_medication(self):
        assert ENTITY_FHIR_MAP["medication"] == "MedicationStatement"

    def test_entity_fhir_map_has_observation_types(self):
        assert ENTITY_FHIR_MAP["measurement"] == "Observation"
        assert ENTITY_FHIR_MAP["biomarker"] == "Observation"

    def test_ontology_system_uris(self):
        assert "snomed_ct" in ONTOLOGY_SYSTEM_URIS
        assert "rxnorm" in ONTOLOGY_SYSTEM_URIS
        assert "loinc" in ONTOLOGY_SYSTEM_URIS


class TestFHIRConverterOutbound:
    """Test converting HoneyBee output to FHIR."""

    def test_to_fhir_bundle_structure(self, converter, sample_processor_output):
        """Bundle should have type=transaction and entries."""
        bundle = converter.to_fhir_bundle(sample_processor_output, patient_id="P001")
        assert bundle["type"] == "transaction"
        assert "entry" in bundle
        assert len(bundle["entry"]) > 0

    def test_condition_resource_created(self, converter, sample_processor_output):
        """Condition entities should produce Condition resources."""
        bundle = converter.to_fhir_bundle(sample_processor_output)
        resource_types = [
            e["resource"]["resourceType"]
            for e in bundle.get("entry", [])
            if "resource" in e
        ]
        assert "Condition" in resource_types

    def test_observation_resource_created(self, converter, sample_processor_output):
        """Biomarker entities should produce Observation resources."""
        bundle = converter.to_fhir_bundle(sample_processor_output)
        resource_types = [
            e["resource"]["resourceType"]
            for e in bundle.get("entry", [])
            if "resource" in e
        ]
        assert "Observation" in resource_types

    def test_medication_resource_created(self, converter, sample_processor_output):
        """Medication entities should produce MedicationStatement resources."""
        bundle = converter.to_fhir_bundle(sample_processor_output)
        resource_types = [
            e["resource"]["resourceType"]
            for e in bundle.get("entry", [])
            if "resource" in e
        ]
        assert "MedicationStatement" in resource_types

    def test_diagnostic_report_included(self, converter, sample_processor_output):
        """DiagnosticReport should be included when text is present."""
        bundle = converter.to_fhir_bundle(sample_processor_output)
        resource_types = [
            e["resource"]["resourceType"]
            for e in bundle.get("entry", [])
            if "resource" in e
        ]
        assert "DiagnosticReport" in resource_types

    def test_ontology_coding_uris(self, converter, sample_processor_output):
        """SNOMED coding should use correct system URI."""
        bundle = converter.to_fhir_bundle(sample_processor_output)
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            if res.get("resourceType") == "Condition":
                codings = res.get("code", {}).get("coding", [])
                if codings:
                    assert codings[0]["system"] == "http://snomed.info/sct"
                    assert codings[0]["code"] == "254838004"
                    return
        pytest.fail("No Condition with SNOMED coding found")

    def test_patient_reference(self, converter, sample_processor_output):
        """Patient reference should be set when patient_id provided."""
        bundle = converter.to_fhir_bundle(
            sample_processor_output, patient_id="P001"
        )
        for entry in bundle.get("entry", []):
            res = entry.get("resource", {})
            if res.get("subject"):
                assert res["subject"]["reference"] == "Patient/P001"
                return
        pytest.fail("No resource with patient reference found")

    def test_empty_entities(self, converter):
        """Empty entities should produce bundle with just DiagnosticReport."""
        result = {"text": "Some text", "entities": []}
        bundle = converter.to_fhir_bundle(result)
        assert bundle["type"] == "transaction"
        assert len(bundle.get("entry", [])) == 1
        assert bundle["entry"][0]["resource"]["resourceType"] == "DiagnosticReport"

    def test_no_entries_no_text(self, converter):
        """No entities and no text should produce bundle with no entries."""
        result = {"text": "", "entities": []}
        bundle = converter.to_fhir_bundle(result)
        assert bundle["type"] == "transaction"
        assert "entry" not in bundle

    def test_unmapped_entity_type_skipped(self, converter):
        """Entity types not in ENTITY_FHIR_MAP should be skipped."""
        result = {
            "text": "",
            "entities": [
                {"text": "left lung", "type": "anatomy", "start": 0, "end": 9, "properties": {}},
            ],
        }
        bundle = converter.to_fhir_bundle(result)
        # anatomy maps to None, so no entries
        assert "entry" not in bundle

    def test_entry_has_request(self, converter, sample_processor_output):
        """Each entry should have a request with method and url."""
        bundle = converter.to_fhir_bundle(sample_processor_output)
        for entry in bundle.get("entry", []):
            assert "request" in entry
            assert entry["request"]["method"] == "POST"
            assert entry["request"]["url"] in [
                "Condition", "MedicationStatement", "Observation", "DiagnosticReport"
            ]

    def test_codeable_concept_text(self, converter):
        """CodeableConcept should include entity text."""
        from honeybee.processors.clinical.types import ClinicalEntity

        entity = ClinicalEntity(text="lung cancer", type="condition", start=0, end=11)
        concept = converter._build_codeable_concept(entity)
        assert concept["text"] == "lung cancer"


class TestFHIRConverterInbound:
    """Test parsing FHIR bundles back to HoneyBee format."""

    def test_from_fhir_bundle(self, converter):
        """from_fhir_bundle should extract text from resources."""
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {"text": "breast cancer"},
                        "clinicalStatus": {
                            "coding": [{"code": "active"}]
                        },
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"text": "hemoglobin"},
                        "status": "final",
                        "valueString": "12.5 g/dL",
                    }
                },
            ],
        }
        result = converter.from_fhir_bundle(bundle)
        assert "text" in result
        assert "breast cancer" in result["text"]
        assert result["resource_count"] == 2

    def test_from_fhir_resource_condition(self, converter):
        """Parsing a Condition resource should extract text."""
        resource = {
            "resourceType": "Condition",
            "code": {"text": "lung cancer"},
            "clinicalStatus": {"coding": [{"code": "active"}]},
        }
        result = converter.from_fhir_resource(resource)
        assert result["resource_type"] == "Condition"
        assert "lung cancer" in result["text"]

    def test_from_fhir_resource_observation(self, converter):
        """Parsing an Observation resource should extract value."""
        resource = {
            "resourceType": "Observation",
            "code": {"text": "hemoglobin"},
            "status": "final",
            "valueString": "12.5 g/dL",
        }
        result = converter.from_fhir_resource(resource)
        assert "hemoglobin" in result["text"]
        assert "12.5" in result["text"]

    def test_from_fhir_resource_medication(self, converter):
        """Parsing a MedicationStatement resource."""
        resource = {
            "resourceType": "MedicationStatement",
            "medicationCodeableConcept": {"text": "tamoxifen"},
            "status": "active",
        }
        result = converter.from_fhir_resource(resource)
        assert "tamoxifen" in result["text"]

    def test_from_fhir_resource_diagnostic_report(self, converter):
        """Parsing a DiagnosticReport resource."""
        resource = {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "conclusion": "No abnormalities detected.",
        }
        result = converter.from_fhir_resource(resource)
        assert "No abnormalities detected" in result["text"]

    def test_from_fhir_resource_unknown(self, converter):
        """Unknown resource types should try narrative text."""
        resource = {
            "resourceType": "AllergyIntolerance",
            "text": {"div": "<div>Penicillin allergy</div>"},
        }
        result = converter.from_fhir_resource(resource)
        assert result["resource_type"] == "AllergyIntolerance"
        assert "Penicillin" in result["text"]

    def test_condition_coding_fallback(self, converter):
        """Condition without text should fall back to coding display."""
        resource = {
            "resourceType": "Condition",
            "code": {"coding": [{"display": "Diabetes mellitus"}]},
            "clinicalStatus": {"coding": [{"code": "active"}]},
        }
        result = converter.from_fhir_resource(resource)
        assert "Diabetes mellitus" in result["text"]


class TestFHIRValidation:
    """Test optional FHIR validation."""

    def test_validate_requires_fhir_resources(self, converter):
        """validate() should raise ImportError without fhir.resources."""
        if _FHIR_AVAILABLE:
            pytest.skip("fhir.resources is installed")
        with pytest.raises(ImportError, match="fhir.resources"):
            converter.validate({"resourceType": "Bundle", "type": "transaction"})
