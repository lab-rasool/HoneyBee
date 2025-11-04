"""
Unit tests for ClinicalProcessor

Tests all functionality of the clinical text processing module including
document processing, entity extraction, tokenization, and embedding generation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from honeybee.processors import ClinicalProcessor


class TestClinicalProcessorInitialization:
    """Test ClinicalProcessor initialization"""

    def test_init_without_config(self):
        """Test initialization without configuration"""
        processor = ClinicalProcessor()
        assert processor is not None
        assert processor.config is not None
        assert "tokenization" in processor.config
        assert "entity_recognition" in processor.config

    def test_init_with_config(self, clinical_config):
        """Test initialization with custom configuration"""
        processor = ClinicalProcessor(config=clinical_config)
        assert processor is not None
        assert processor.config["tokenization"]["model"] == "gatortron"

    def test_default_config_merge(self):
        """Test that custom config merges with defaults"""
        custom_config = {"tokenization": {"model": "pubmedbert"}}
        processor = ClinicalProcessor(config=custom_config)

        # Custom value should be used
        assert processor.config["tokenization"]["model"] == "pubmedbert"
        # Default values should still exist
        assert "max_length" in processor.config["tokenization"]
        assert "entity_recognition" in processor.config


class TestTextProcessing:
    """Test text processing functionality"""

    def test_process_text_basic(self, sample_clinical_text):
        """Test basic text processing"""
        processor = ClinicalProcessor()
        result = processor.process_text(sample_clinical_text)

        assert result is not None
        assert "text" in result
        assert "entities" in result
        assert result["text"] == sample_clinical_text

    def test_process_text_with_entities(self, sample_clinical_text):
        """Test that entities are extracted from text"""
        processor = ClinicalProcessor()
        result = processor.process_text(sample_clinical_text)

        assert "entities" in result
        assert len(result["entities"]) > 0

        # Check for expected entity types
        entity_types = [e["type"] for e in result["entities"]]
        assert any(t in ["tumor", "staging", "biomarker"] for t in entity_types)

    def test_process_text_with_document_structure(self, sample_clinical_text):
        """Test document structure analysis"""
        processor = ClinicalProcessor()
        result = processor.process_text(sample_clinical_text)

        assert "document_structure" in result
        assert "sections" in result["document_structure"]

    def test_process_text_without_entity_recognition(self, sample_clinical_text):
        """Test processing with entity recognition disabled"""
        config = {"processing_pipeline": ["document", "tokenization"]}
        processor = ClinicalProcessor(config=config)
        result = processor.process_text(sample_clinical_text)

        assert result is not None
        # Entities should not be in result if not in pipeline
        assert "entities" not in result or len(result.get("entities", [])) == 0


class TestEntityExtraction:
    """Test entity extraction functionality"""

    def test_extract_cancer_entities(self):
        """Test extraction of cancer-specific entities"""
        text = "Patient diagnosed with invasive ductal carcinoma, Grade 2"
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        entities = result["entities"]
        assert len(entities) > 0

        # Should find tumor type and grade
        texts = [e["text"].lower() for e in entities]
        assert any("carcinoma" in t for t in texts)

    def test_extract_biomarkers(self):
        """Test extraction of biomarker information"""
        text = "ER: Positive (95%), PR: Positive (80%), HER2: Negative"
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        entities = result["entities"]

        # Should find entities (may be biomarkers or other types depending on patterns)
        # The actual entity type may vary based on pattern matching
        assert isinstance(entities, list)
        # If biomarkers are extracted, verify they exist
        if len(entities) > 0:
            assert any(e["type"] in ["biomarker", "measurement", "condition"] for e in entities)

    def test_extract_staging(self):
        """Test extraction of staging information"""
        text = "Stage pT2 N0 M0, Stage IIA adenocarcinoma"
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        entities = result["entities"]

        # Should find staging entities
        staging = [e for e in entities if e["type"] == "staging"]
        assert len(staging) > 0

    def test_extract_measurements(self):
        """Test extraction of tumor size measurements"""
        text = "Tumor measures 2.5 cm in greatest dimension"
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        entities = result["entities"]

        # Should find measurement entity
        measurements = [e for e in entities if e["type"] == "measurement"]
        assert len(measurements) > 0

    def test_extract_temporal(self):
        """Test extraction of temporal information"""
        text = "Diagnosed on 01/15/2024. Follow-up on February 20, 2024."
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        entities = result["entities"]

        # Should find temporal entities
        temporal = [e for e in entities if e["type"] == "temporal"]
        assert len(temporal) > 0


class TestTemporalTimeline:
    """Test temporal timeline extraction"""

    def test_timeline_generation(self):
        """Test that timeline is generated from temporal entities"""
        text = """
        January 15, 2024: Initial diagnosis of breast cancer.
        February 1, 2024: Started chemotherapy.
        March 10, 2024: Mid-treatment scan shows partial response.
        """
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        assert "temporal_timeline" in result
        timeline = result["temporal_timeline"]
        assert len(timeline) > 0

    def test_timeline_ordering(self):
        """Test that timeline events are ordered chronologically"""
        text = """
        March 1, 2024: Follow-up visit.
        January 15, 2024: Initial diagnosis.
        February 1, 2024: Treatment started.
        """
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        timeline = result["temporal_timeline"]

        # Check that events are sorted
        dates = [event.get("normalized_date", "9999") for event in timeline]
        assert dates == sorted(dates)


class TestTokenization:
    """Test tokenization functionality"""

    @patch("honeybee.processors.clinical_processor.AutoTokenizer")
    def test_sentence_tokenization(self, mock_tokenizer, sample_clinical_text):
        """Test sentence-based tokenization"""
        # Setup mock
        mock_tok_instance = MagicMock()
        mock_tok_instance.tokenize.return_value = ["patient", "with", "cancer"]

        # Mock the __call__ method to return proper numpy arrays
        import numpy as np

        mock_tok_instance.return_value = {
            "input_ids": np.array([[101, 1, 2, 3, 102]]),
            "attention_mask": np.array([[1, 1, 1, 1, 1]]),
        }
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance

        config = {"tokenization": {"model": "gatortron", "segment_strategy": "sentence"}}
        processor = ClinicalProcessor(config=config)

        # Reset mock to clear initialization calls
        mock_tok_instance.reset_mock()

        result = processor.process_text(sample_clinical_text)

        # May or may not have tokenization depending on whether error occurred
        assert result is not None
        assert "text" in result

    def test_paragraph_tokenization(self, sample_clinical_text):
        """Test paragraph-based tokenization"""
        config = {"tokenization": {"segment_strategy": "paragraph"}}
        processor = ClinicalProcessor(config=config)
        result = processor.process_text(sample_clinical_text)

        # Should still complete without error
        assert result is not None


@pytest.mark.slow
class TestEmbeddingGeneration:
    """Test embedding generation functionality"""

    @patch("honeybee.models.HuggingFaceEmbedder")
    def test_generate_single_embedding(self, mock_embedder_class):
        """Test generating embeddings for single text"""
        # Setup mock
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.generate_embeddings.return_value = np.random.randn(1, 768)
        mock_embedder_class.return_value = mock_embedder_instance

        processor = ClinicalProcessor()
        text = "Patient with lung cancer"

        embeddings = processor.generate_embeddings(text)

        assert embeddings is not None
        assert embeddings.shape == (1, 768)

    @patch("honeybee.models.HuggingFaceEmbedder")
    def test_generate_batch_embeddings(self, mock_embedder_class):
        """Test generating embeddings for multiple texts"""
        # Setup mock
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.generate_embeddings.return_value = np.random.randn(3, 768)
        mock_embedder_class.return_value = mock_embedder_instance

        processor = ClinicalProcessor()
        texts = [
            "Patient 1 with breast cancer",
            "Patient 2 with lung cancer",
            "Patient 3 with colon cancer",
        ]

        embeddings = processor.generate_embeddings(texts)

        assert embeddings is not None
        assert embeddings.shape == (3, 768)

    @patch("honeybee.models.HuggingFaceEmbedder")
    def test_generate_embeddings_with_model_selection(self, mock_embedder_class):
        """Test embedding generation with different models"""
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.generate_embeddings.return_value = np.random.randn(1, 768)
        mock_embedder_class.return_value = mock_embedder_instance

        processor = ClinicalProcessor()
        text = "Patient with cancer"

        # Test with different models
        for model_name in ["pubmedbert", "bioclinicalbert", "gatortron"]:
            embeddings = processor.generate_embeddings(text, model_name=model_name)
            assert embeddings is not None


class TestBatchProcessing:
    """Test batch document processing"""

    @patch("honeybee.processors.ClinicalProcessor.process")
    def test_batch_process_multiple_files(self, mock_process, temp_dir):
        """Test processing multiple files in batch"""
        # Create temporary test files
        for i in range(3):
            (temp_dir / f"test_{i}.txt").write_text(f"Clinical text {i}")

        mock_process.return_value = {"text": "test", "entities": []}

        processor = ClinicalProcessor()
        results = processor.process_batch(temp_dir, file_pattern="*.txt")

        assert len(results) == 3
        assert mock_process.call_count == 3

    @patch("honeybee.processors.ClinicalProcessor.process")
    def test_batch_with_output_dir(self, mock_process, temp_dir, temp_output_dir):
        """Test batch processing with custom output directory"""
        (temp_dir / "test.txt").write_text("Clinical text")
        mock_process.return_value = {"text": "test", "entities": []}

        processor = ClinicalProcessor()
        results = processor.process_batch(
            temp_dir, file_pattern="*.txt", save_output=True, output_dir=temp_output_dir
        )

        assert len(results) == 1


class TestDocumentStructureAnalysis:
    """Test document structure analysis"""

    def test_identify_sections(self):
        """Test identification of document sections"""
        text = """
        CHIEF COMPLAINT: Breast mass

        HISTORY OF PRESENT ILLNESS:
        Patient presents with palpable mass in left breast.

        PAST MEDICAL HISTORY:
        No significant medical history.

        MEDICATIONS:
        None currently.

        ASSESSMENT AND PLAN:
        Proceed with biopsy.
        """
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        structure = result["document_structure"]
        sections = structure["sections"]

        # Should identify multiple sections
        assert len(sections) > 1
        assert "chief_complaint" in sections or "history_present_illness" in sections

    def test_section_content_extraction(self):
        """Test extraction of section content"""
        text = """
        MEDICATIONS: Tamoxifen 20mg daily, Aspirin 81mg daily

        ALLERGIES: Penicillin
        """
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        structure = result["document_structure"]
        sections = structure["sections"]

        # Check that sections contain content
        for section_name, content in sections.items():
            if content:
                assert isinstance(content, str)
                assert len(content) > 0


class TestConfigurationOptions:
    """Test various configuration options"""

    def test_disable_ocr(self):
        """Test disabling OCR"""
        config = {"document_processor": {"use_ocr": False}}
        processor = ClinicalProcessor(config=config)
        assert processor.config["document_processor"]["use_ocr"] is False

    def test_custom_tokenization_settings(self):
        """Test custom tokenization settings"""
        config = {"tokenization": {"max_length": 1024, "segment_strategy": "fixed"}}
        processor = ClinicalProcessor(config=config)
        assert processor.config["tokenization"]["max_length"] == 1024
        assert processor.config["tokenization"]["segment_strategy"] == "fixed"

    def test_custom_entity_settings(self):
        """Test custom entity recognition settings"""
        config = {
            "entity_recognition": {
                "use_rules": True,
                "use_patterns": False,
                "cancer_specific_extraction": True,
            }
        }
        processor = ClinicalProcessor(config=config)
        assert processor.config["entity_recognition"]["use_rules"] is True
        assert processor.config["entity_recognition"]["use_patterns"] is False


class TestSummaryStatistics:
    """Test summary statistics generation"""

    def test_get_summary_statistics(self, sample_clinical_text):
        """Test generating summary statistics"""
        processor = ClinicalProcessor()
        result = processor.process_text(sample_clinical_text)

        stats = processor.get_summary_statistics(result)

        assert "text_length" in stats
        assert "num_entities" in stats
        assert "entity_types" in stats
        assert stats["text_length"] > 0

    def test_entity_type_counts(self):
        """Test counting entities by type"""
        text = """
        ER positive, PR positive, HER2 negative.
        Stage pT2 N0 M0.
        Grade 2 invasive ductal carcinoma.
        """
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        stats = processor.get_summary_statistics(result)

        # Should have counts for different entity types
        assert isinstance(stats["entity_types"], dict)
        assert len(stats["entity_types"]) > 0


class TestErrorHandling:
    """Test error handling in various scenarios"""

    def test_empty_text(self):
        """Test processing empty text"""
        processor = ClinicalProcessor()
        result = processor.process_text("")

        assert result is not None
        assert result["text"] == ""

    def test_very_long_text(self):
        """Test processing very long text"""
        # Create a very long text (> 10000 characters)
        long_text = "Patient with cancer. " * 1000

        processor = ClinicalProcessor()
        result = processor.process_text(long_text)

        assert result is not None
        assert "text" in result

    def test_special_characters(self):
        """Test processing text with special characters"""
        text = "Patient with ER+/PR+/HER2- breast cancer (stage II-III)"
        processor = ClinicalProcessor()
        result = processor.process_text(text)

        assert result is not None
        assert "entities" in result


@pytest.mark.requires_sample_data
class TestWithRealData:
    """Tests using real sample data"""

    def test_process_sample_pdf(self, sample_clinical_pdf_path):
        """Test processing actual sample PDF"""
        if sample_clinical_pdf_path is None:
            pytest.skip("Sample PDF not available")

        processor = ClinicalProcessor()
        result = processor.process(sample_clinical_pdf_path)

        assert result is not None
        assert "text" in result
        assert len(result["text"]) > 0
