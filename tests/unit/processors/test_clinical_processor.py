"""
Unit tests for the new modular ClinicalProcessor pipeline.

Tests types, ingestion, NER, ontology, temporal, embeddings, and the orchestrator.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from honeybee.processors.clinical import (
    ClinicalProcessor,
    ClinicalDocument,
    ClinicalEntity,
    ClinicalResult,
    OntologyCode,
    TimelineEvent,
    DocumentIngester,
    NEREngine,
    OntologyResolver,
    TimelineExtractor,
    EmbeddingEngine,
)


# ===========================================================================
# Types
# ===========================================================================


class TestTypes:
    """Test shared dataclasses."""

    def test_clinical_entity_to_dict(self):
        entity = ClinicalEntity(
            text="breast cancer",
            type="condition",
            start=0,
            end=13,
            confidence=0.9,
            ontology_codes=[
                OntologyCode(system="snomed_ct", code="254837009", display="Breast cancer")
            ],
        )
        d = entity.to_dict()
        assert d["text"] == "breast cancer"
        assert d["type"] == "condition"
        assert len(d["ontology_codes"]) == 1
        assert d["ontology_codes"][0]["system"] == "snomed_ct"

    def test_clinical_result_to_dict(self):
        doc = ClinicalDocument(text="test text", sections={"diagnosis": "cancer"})
        result = ClinicalResult(
            document=doc,
            entities=[
                ClinicalEntity(text="cancer", type="condition", start=0, end=6)
            ],
            timeline=[
                TimelineEvent(date_text="Jan 1", sentence="Diagnosed Jan 1")
            ],
        )
        d = result.to_dict()
        assert d["text"] == "test text"
        assert len(d["entities"]) == 1
        assert len(d["timeline"]) == 1

    def test_clinical_result_text_shortcut(self):
        doc = ClinicalDocument(text="hello")
        result = ClinicalResult(document=doc)
        assert result.text == "hello"

    def test_ontology_code_defaults(self):
        oc = OntologyCode(system="snomed_ct", code="123", display="Test")
        assert oc.source_api == "local"


# ===========================================================================
# Document Ingestion
# ===========================================================================


class TestDocumentIngester:
    """Test DocumentIngester."""

    def test_ingest_text(self):
        ingester = DocumentIngester()
        doc = ingester.ingest_text("Patient with cancer.")
        assert isinstance(doc, ClinicalDocument)
        assert doc.text == "Patient with cancer."
        assert doc.metadata["source_type"] == "text"

    def test_ingest_text_sections(self):
        text = """
        DIAGNOSIS: Breast cancer.

        MEDICATIONS: Tamoxifen 20mg daily.

        ALLERGIES: None.
        """
        ingester = DocumentIngester()
        doc = ingester.ingest_text(text)
        assert doc.sections == {}

    def test_ingest_raw_string_as_text(self):
        ingester = DocumentIngester()
        doc = ingester.ingest("Patient has stage III lung cancer.")
        assert isinstance(doc, ClinicalDocument)
        assert "lung cancer" in doc.text

    def test_ingest_json_file(self, temp_dir):
        json_path = temp_dir / "record.json"
        json_path.write_text(json.dumps({"diagnosis": "lung cancer", "treatment": "chemo"}))
        ingester = DocumentIngester()
        doc = ingester.ingest(json_path)
        assert "lung cancer" in doc.text
        assert doc.source_path == json_path

    def test_ingest_xml_file(self, temp_dir):
        xml_path = temp_dir / "record.xml"
        xml_path.write_text(
            "<record><diagnosis>breast cancer</diagnosis></record>"
        )
        ingester = DocumentIngester()
        doc = ingester.ingest(xml_path)
        assert "breast cancer" in doc.text

    def test_ingest_csv_file(self, temp_dir):
        csv_path = temp_dir / "record.csv"
        csv_path.write_text("name,diagnosis\nJohn,colon cancer\n")
        ingester = DocumentIngester()
        doc = ingester.ingest(csv_path)
        assert "colon cancer" in doc.text

    def test_ingest_text_file(self, temp_dir):
        txt_path = temp_dir / "report.txt"
        txt_path.write_text("Patient with metastatic disease.")
        ingester = DocumentIngester()
        doc = ingester.ingest(txt_path)
        assert "metastatic disease" in doc.text

    def test_unsupported_ehr_format(self, temp_dir):
        yaml_path = temp_dir / "record.yaml"
        yaml_path.write_text("diagnosis: cancer")
        ingester = DocumentIngester()
        # .yaml is not in EHR extensions, so it falls through to plain text
        doc = ingester.ingest(yaml_path)
        assert "cancer" in doc.text

    def test_section_detection_empty_text(self):
        ingester = DocumentIngester()
        doc = ingester.ingest_text("")
        assert doc.sections == {}


# ===========================================================================
# NER Engine
# ===========================================================================


class TestNEREngine:
    """Test NER engine mechanics."""

    def test_default_backend_is_empty(self):
        engine = NEREngine()
        assert engine.backend_names == []

    def test_merge_overlapping_same_type(self):
        entities = [
            ClinicalEntity(text="ductal carcinoma", type="tumor", start=5, end=22),
            ClinicalEntity(text="carcinoma", type="tumor", start=12, end=22),
        ]
        merged = NEREngine._merge_entities(entities)
        assert len(merged) == 1

    def test_merge_overlapping_different_types_preserved(self):
        entities = [
            ClinicalEntity(text="lung cancer", type="tumor", start=10, end=21),
            ClinicalEntity(text="lung", type="condition", start=10, end=14),
        ]
        merged = NEREngine._merge_entities(entities)
        types = {e.type for e in merged}
        assert "tumor" in types
        assert "condition" in types

    def test_merge_empty(self):
        assert NEREngine._merge_entities([]) == []

    def test_unknown_backend_warning(self):
        engine = NEREngine(backends=["nonexistent_backend"])
        assert engine.backend_names == []

    def test_extract_no_backends_returns_empty(self):
        engine = NEREngine()
        entities = engine.extract("Patient diagnosed with breast cancer.")
        assert entities == []


# ===========================================================================
# Ontology Resolution
# ===========================================================================


class TestOntologyResolver:
    """Test OntologyResolver."""

    def test_empty_backends(self):
        resolver = OntologyResolver(backends=[])
        entities = [ClinicalEntity(text="cancer", type="condition", start=0, end=6)]
        result = resolver.resolve(entities)
        # No backends → entities unchanged
        assert result is entities

    def test_unknown_backend_warning(self):
        resolver = OntologyResolver(backends=["nonexistent"])
        assert len(resolver._clients) == 0

    def test_umls_missing_key(self):
        resolver = OntologyResolver(backends=["umls"], config={})
        assert len(resolver._clients) == 0

    @patch("honeybee.processors.clinical.ontology.snowstorm_client.SnowstormClient.search")
    def test_snowstorm_resolve(self, mock_search):
        mock_search.return_value = [
            OntologyCode(system="snomed_ct", code="254837009", display="Breast cancer", source_api="snowstorm")
        ]
        resolver = OntologyResolver(backends=["snowstorm"])
        entities = [ClinicalEntity(text="breast cancer", type="condition", start=0, end=13)]
        resolver.resolve(entities)
        assert len(entities[0].ontology_codes) > 0
        assert entities[0].ontology_codes[0].system == "snomed_ct"


# ===========================================================================
# Timeline Extraction
# ===========================================================================


class TestTimelineExtractor:
    """Test TimelineExtractor."""

    def test_extract_dates(self):
        extractor = TimelineExtractor()
        doc = ClinicalDocument(text="Diagnosed January 15, 2024. Treatment started February 1, 2024.")
        entities = []
        timeline = extractor.extract(doc, entities)
        assert len(timeline) > 0

    def test_timeline_ordering(self):
        extractor = TimelineExtractor()
        doc = ClinicalDocument(
            text="March 1, 2024: Follow-up. January 15, 2024: Diagnosis. February 1, 2024: Treatment."
        )
        timeline = extractor.extract(doc, [])
        dates = [e.date for e in timeline if e.date is not None]
        assert dates == sorted(dates)

    def test_empty_text(self):
        extractor = TimelineExtractor()
        doc = ClinicalDocument(text="")
        assert extractor.extract(doc, []) == []


# ===========================================================================
# Embedding Engine
# ===========================================================================


class TestEmbeddingEngine:
    """Test EmbeddingEngine."""

    @patch("honeybee.processors.clinical.embeddings.local_backend.LocalEmbeddingBackend.embed")
    @patch("honeybee.processors.clinical.embeddings.local_backend.LocalEmbeddingBackend._load")
    def test_local_embed(self, mock_load, mock_embed):
        mock_embed.return_value = np.random.randn(1, 768)
        engine = EmbeddingEngine(mode="local", config={"model": "bioclinicalbert"})
        result = engine.embed("Patient with cancer")
        assert result.shape == (1, 768)

    def test_unknown_mode_raises(self):
        engine = EmbeddingEngine(mode="invalid")
        with pytest.raises(ValueError, match="Unknown embedding mode"):
            engine.embed("test")


# ===========================================================================
# ClinicalProcessor (Orchestrator)
# ===========================================================================


class TestClinicalProcessorInitialization:
    """Test ClinicalProcessor initialization."""

    def test_init_without_config(self):
        processor = ClinicalProcessor()
        assert processor is not None
        assert processor.config is not None
        assert "ner" in processor.config
        assert "embeddings" in processor.config

    def test_init_with_config(self, clinical_config):
        processor = ClinicalProcessor(config=clinical_config)
        assert processor is not None
        assert processor.config["ner"]["backends"] == []

    def test_default_config_merge(self):
        custom = {"ner": {"backends": ["scispacy"]}}
        processor = ClinicalProcessor(config=custom)
        assert processor.config["ner"]["backends"] == ["scispacy"]
        # Default temporal should still exist
        assert "temporal" in processor.config


class TestTextProcessing:
    """Test text processing via the orchestrator."""

    def test_process_text_basic(self, sample_clinical_text):
        processor = ClinicalProcessor()
        result = processor.process_text(sample_clinical_text)

        assert isinstance(result, ClinicalResult)
        assert result.text == sample_clinical_text

    def test_process_text_sections(self, sample_clinical_text):
        processor = ClinicalProcessor()
        result = processor.process_text(sample_clinical_text)
        assert result.document.sections == {}

    def test_process_text_empty(self):
        processor = ClinicalProcessor()
        result = processor.process_text("")
        assert isinstance(result, ClinicalResult)
        assert result.text == ""

    def test_process_text_very_long(self):
        long_text = "Patient with cancer. " * 1000
        processor = ClinicalProcessor()
        result = processor.process_text(long_text)
        assert isinstance(result, ClinicalResult)

    def test_process_text_special_chars(self):
        text = "Patient with ER+/PR+/HER2- breast cancer (stage II-III)"
        processor = ClinicalProcessor()
        result = processor.process_text(text)
        assert isinstance(result, ClinicalResult)


class TestTemporalTimeline:
    """Test temporal timeline extraction via the orchestrator."""

    def test_timeline_generation(self):
        text = """
        January 15, 2024: Initial diagnosis of breast cancer.
        February 1, 2024: Started chemotherapy.
        March 10, 2024: Mid-treatment scan shows partial response.
        """
        processor = ClinicalProcessor()
        result = processor.process_text(text)
        assert len(result.timeline) > 0

    def test_timeline_ordering(self):
        text = """
        March 1, 2024: Follow-up visit.
        January 15, 2024: Initial diagnosis.
        February 1, 2024: Treatment started.
        """
        processor = ClinicalProcessor()
        result = processor.process_text(text)
        dates = [e.date for e in result.timeline if e.date is not None]
        assert dates == sorted(dates)


class TestBatchProcessing:
    """Test batch document processing."""

    def test_batch_process(self, temp_dir):
        for i in range(3):
            (temp_dir / f"test_{i}.txt").write_text(f"Patient {i} with breast cancer")

        processor = ClinicalProcessor()
        results = processor.process_batch(temp_dir, file_pattern="*.txt")
        assert len(results) == 3
        for r in results:
            assert isinstance(r, ClinicalResult)

    def test_batch_empty_dir(self, temp_dir):
        processor = ClinicalProcessor()
        results = processor.process_batch(temp_dir, file_pattern="*.pdf")
        assert results == []


class TestEmbeddingGeneration:
    """Test embedding generation via the orchestrator."""

    @patch("honeybee.processors.clinical.embeddings.local_backend.LocalEmbeddingBackend.embed")
    @patch("honeybee.processors.clinical.embeddings.local_backend.LocalEmbeddingBackend._load")
    def test_generate_single(self, mock_load, mock_embed):
        mock_embed.return_value = np.random.randn(1, 768)
        processor = ClinicalProcessor()
        result = processor.generate_embeddings("Patient with cancer")
        assert result.shape == (1, 768)

    @patch("honeybee.processors.clinical.embeddings.local_backend.LocalEmbeddingBackend.embed")
    @patch("honeybee.processors.clinical.embeddings.local_backend.LocalEmbeddingBackend._load")
    def test_generate_batch(self, mock_load, mock_embed):
        mock_embed.return_value = np.random.randn(3, 768)
        processor = ClinicalProcessor()
        texts = ["Text 1", "Text 2", "Text 3"]
        result = processor.generate_embeddings(texts)
        assert result.shape == (3, 768)

    @patch("honeybee.processors.clinical.embeddings.local_backend.LocalEmbeddingBackend.embed")
    @patch("honeybee.processors.clinical.embeddings.local_backend.LocalEmbeddingBackend._load")
    def test_generate_with_model_name(self, mock_load, mock_embed):
        mock_embed.return_value = np.random.randn(1, 768)
        processor = ClinicalProcessor()
        result = processor.generate_embeddings("test", model_name="pubmedbert")
        assert result is not None


class TestSummaryStatistics:
    """Test summary statistics."""

    def test_get_summary_statistics(self, sample_clinical_text):
        processor = ClinicalProcessor()
        result = processor.process_text(sample_clinical_text)
        stats = processor.get_summary_statistics(result)
        assert "text_length" in stats
        assert "num_entities" in stats
        assert "entity_types" in stats
        assert stats["text_length"] > 0

    def test_entity_type_counts(self):
        processor = ClinicalProcessor()
        result = processor.process_text(
            "ER positive, HER2 negative. Stage pT2 N0 M0. Grade 2 invasive ductal carcinoma."
        )
        stats = processor.get_summary_statistics(result)
        assert isinstance(stats["entity_types"], dict)


class TestInterop:
    """Test FHIR and HL7 interop via processor."""

    def test_to_fhir(self, sample_clinical_text):
        processor = ClinicalProcessor()
        result = processor.process_text(sample_clinical_text)
        bundle = processor.to_fhir(result, patient_id="test-123")
        assert bundle["resourceType"] == "Bundle"
        assert "entry" in bundle

    def test_process_fhir(self):
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {"text": "breast cancer"},
                    }
                }
            ],
        }
        processor = ClinicalProcessor()
        result = processor.process_fhir(bundle)
        assert isinstance(result, ClinicalResult)


class TestEHRProcessing:
    """Test EHR document processing."""

    def test_json_ehr(self, temp_dir):
        json_path = temp_dir / "record.json"
        json_path.write_text(json.dumps({
            "patient": {"name": "John Doe", "diagnosis": "lung cancer"},
            "treatment": "chemotherapy",
        }))
        processor = ClinicalProcessor()
        result = processor.process(json_path)
        assert isinstance(result, ClinicalResult)
        assert "lung cancer" in result.text

    def test_xml_ehr(self, temp_dir):
        xml_path = temp_dir / "record.xml"
        xml_path.write_text(
            "<record><patient><diagnosis>breast cancer</diagnosis></patient></record>"
        )
        processor = ClinicalProcessor()
        result = processor.process(xml_path)
        assert isinstance(result, ClinicalResult)
        assert "breast cancer" in result.text

    def test_csv_ehr(self, temp_dir):
        csv_path = temp_dir / "record.csv"
        csv_path.write_text("name,diagnosis\nJohn,colon cancer\n")
        processor = ClinicalProcessor()
        result = processor.process(csv_path)
        assert isinstance(result, ClinicalResult)
        assert "colon cancer" in result.text
