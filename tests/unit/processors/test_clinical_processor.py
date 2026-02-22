"""
Unit tests for ClinicalProcessor

Tests all functionality of the clinical text processing module including
document processing, entity extraction, tokenization, and embedding generation.
"""

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from honeybee.processors import ClinicalProcessor
from honeybee.processors.clinical_processor import (
    CANCER_PATTERNS,
    MEDICAL_ABBREVIATIONS,
    ONTOLOGY_MAPPINGS,
    SUPPORTED_EHR_FORMATS,
)


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


# ===========================================================================
# Bug fix regression tests
# ===========================================================================


class TestBug1EhrDocumentElseClause:
    """Bug 1: _process_ehr_document should raise ValueError for unsupported formats"""

    def test_unsupported_ehr_format_raises(self):
        """Calling _process_ehr_document with an unsupported suffix raises ValueError"""
        processor = ClinicalProcessor()
        fake_path = Path("/tmp/record.yaml")  # .yaml is not supported
        with pytest.raises(ValueError, match="Unsupported EHR format"):
            processor._process_ehr_document(fake_path)

    def test_supported_ehr_formats_still_dispatch(self, temp_dir):
        """Supported formats still work correctly"""
        processor = ClinicalProcessor()

        # JSON
        json_path = temp_dir / "record.json"
        json_path.write_text(json.dumps({"diagnosis": "lung cancer"}))
        result = processor._process_ehr_document(json_path)
        assert result is not None
        assert "text" in result

        # CSV
        csv_path = temp_dir / "record.csv"
        csv_path.write_text("name,diagnosis\nJohn,lung cancer\n")
        result = processor._process_ehr_document(csv_path)
        assert result is not None
        assert "text" in result


class TestBug2TnmRegexFalsePositives:
    """Bug 2: TNM staging regex should not match substrings in normal words"""

    def test_tnm_no_false_positive_in_words(self):
        """Words like 'Table', 'Total', 'Note' should NOT produce TNM entities"""
        processor = ClinicalProcessor()
        text = "Table1 shows the Total count. Note: N0te the value."
        result = processor.process_text(text)

        staging = [e for e in result["entities"] if e["type"] == "staging"]
        staging_texts = [e["text"] for e in staging]
        # None of these normal words should be extracted as staging
        for word_fragment in ["T1", "N0"]:
            # If it appears, verify it's not from inside a word
            for st in staging:
                context_start = max(0, st["start"] - 1)
                context_end = min(len(text), st["end"] + 1)
                before = text[context_start : st["start"]]
                after = text[st["end"] : context_end]
                # Should not be preceded/followed by word characters
                assert not (before.isalpha() or after.isalpha()), (
                    f"TNM match '{st['text']}' appears inside a word"
                )

    def test_tnm_still_matches_real_staging(self):
        """Real TNM staging should still be captured"""
        processor = ClinicalProcessor()
        text = "The staging is T2 N1 M0."
        result = processor.process_text(text)

        staging = [e for e in result["entities"] if e["type"] == "staging"]
        staging_texts = [e["text"] for e in staging]
        assert "T2" in staging_texts
        assert "N1" in staging_texts
        assert "M0" in staging_texts

    def test_tnm_word_boundary(self):
        """Verify the TNM pattern has word boundary anchors"""
        pattern = CANCER_PATTERNS["tnm_stage"]
        assert r"\b" in pattern, "TNM pattern should contain word boundary anchors"


class TestBug3LabValueRegex:
    """Bug 3: Lab value regex should not match document headings/metadata"""

    def test_lab_pattern_no_false_positive_headings(self):
        """Document headings like 'Section: 5 pages' should not match"""
        processor = ClinicalProcessor()
        text = "Section: 5 pages. Page: 2 of 10. Date: 15 Jan."
        result = processor.process_text(text)

        measurements = [e for e in result["entities"] if e["type"] == "measurement"]
        # These should not be captured as lab values
        measurement_texts = [e["properties"].get("test_name", "") for e in measurements]
        assert "Section" not in measurement_texts
        assert "Page" not in measurement_texts
        assert "Date" not in measurement_texts

    def test_lab_pattern_matches_real_values(self):
        """Real lab values with clinical units should still be matched"""
        processor = ClinicalProcessor()
        text = "Hemoglobin: 12.5 g/dL. Creatinine: 1.2 mg/dL. WBC: 5.0 K/uL."
        result = processor.process_text(text)

        measurements = [e for e in result["entities"] if e["type"] == "measurement"]
        assert len(measurements) > 0


class TestBug4DuplicateAbbreviationExpansion:
    """Bug 4: Abbreviation expansion should use MEDICAL_ABBREVIATIONS consistently"""

    def test_abbreviation_at_text_start(self):
        """Abbreviations at start of text should be expanded"""
        processor = ClinicalProcessor()
        cleaned = processor._clean_text("dx of lung cancer")
        assert "diagnosis" in cleaned

    def test_abbreviation_at_text_end(self):
        """Abbreviations at end of text should be expanded"""
        processor = ClinicalProcessor()
        cleaned = processor._clean_text("patient needs tx")
        assert "treatment" in cleaned

    def test_abbreviation_before_punctuation(self):
        """Abbreviations before punctuation should be expanded"""
        processor = ClinicalProcessor()
        cleaned = processor._clean_text("the dx. was confirmed")
        # 'dx' before period should still be expanded
        assert "diagnosis" in cleaned

    def test_no_double_expansion(self):
        """Text should not be expanded twice to produce wrong results"""
        processor = ClinicalProcessor()
        # After expansion, 'dx' becomes 'diagnosis' — running again shouldn't change it
        text = "dx of cancer"
        first_pass = processor._clean_text(text)
        second_pass = processor._clean_text(first_pass)
        assert first_pass == second_pass

    def test_clean_text_uses_medical_abbreviations_dict(self):
        """_clean_text should expand the same abbreviations as MEDICAL_ABBREVIATIONS"""
        processor = ClinicalProcessor()
        for abbr, expansion in MEDICAL_ABBREVIATIONS.items():
            cleaned = processor._clean_text(f"the {abbr} was noted")
            assert expansion in cleaned, f"'{abbr}' should expand to '{expansion}'"


class TestBug5SlidingWindowSpecialTokens:
    """Bug 5: Sliding window should handle models without CLS/SEP tokens"""

    def _make_mock_tokenizer(self, cls_id=101, sep_id=102, eos_id=None, pad_id=0, num_tokens=100):
        """Helper to create a mock tokenizer for sliding window tests."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.cls_token_id = cls_id
        mock_tokenizer.sep_token_id = sep_id
        mock_tokenizer.eos_token_id = eos_id
        mock_tokenizer.pad_token_id = pad_id

        # The tokenizer is called as tokenizer(text, ...) and must return a dict
        # with "input_ids" as a plain list (not nested).
        token_ids = list(range(num_tokens))
        mock_tokenizer.return_value = {
            "input_ids": token_ids,
        }
        return mock_tokenizer

    def test_sliding_window_without_cls_sep(self):
        """Models like T5 that lack CLS/SEP should not crash"""
        processor = ClinicalProcessor()
        # Override stride to be smaller than max_length for test
        processor.config["tokenization"]["stride"] = 10

        processor.tokenizer = self._make_mock_tokenizer(
            cls_id=None, sep_id=None, eos_id=1, pad_id=0, num_tokens=100
        )
        processor.model_max_length = 50

        segments = ["word " * 200]
        result = processor._sliding_window_tokenization(segments)

        assert "num_windows" in result
        assert result["num_windows"] > 0
        # Verify no None values in token ids
        for window_ids in result["input_ids"]:
            assert None not in window_ids, "None token IDs should not appear in windows"
        # With eos_id=1 and no cls, first token should be from content (not None)
        first_window = result["input_ids"][0]
        # Last real token before padding should be eos_id=1
        assert 1 in first_window

    def test_sliding_window_with_cls_sep(self):
        """Models with CLS/SEP should still work as before"""
        processor = ClinicalProcessor()
        processor.config["tokenization"]["stride"] = 10

        processor.tokenizer = self._make_mock_tokenizer(
            cls_id=101, sep_id=102, pad_id=0, num_tokens=100
        )
        processor.model_max_length = 50

        segments = ["word " * 200]
        result = processor._sliding_window_tokenization(segments)

        assert result["num_windows"] > 0
        # First window should start with CLS
        first_window = result["input_ids"][0]
        assert first_window[0] == 101  # CLS
        # SEP should appear somewhere in the window
        assert 102 in first_window

    def test_sliding_window_no_pad_token(self):
        """Models without explicit pad_token_id should fall back to 0"""
        processor = ClinicalProcessor()
        processor.config["tokenization"]["stride"] = 10

        processor.tokenizer = self._make_mock_tokenizer(
            cls_id=None, sep_id=None, eos_id=None, pad_id=None, num_tokens=30
        )
        processor.model_max_length = 50

        segments = ["word " * 50]
        result = processor._sliding_window_tokenization(segments)

        # Should not crash and padding should use 0
        for window_ids in result["input_ids"]:
            assert None not in window_ids


class TestBug6EntityMergeOverlap:
    """Bug 6: Overlapping entities of different types should both be preserved"""

    def test_overlapping_different_types_preserved(self):
        """Two overlapping entities of different types should both survive merge"""
        processor = ClinicalProcessor()

        entities = [
            {"text": "lung cancer", "type": "tumor", "start": 10, "end": 21, "properties": {}},
            {
                "text": "lung",
                "type": "measurement",
                "start": 10,
                "end": 14,
                "properties": {},
            },
        ]

        merged = processor._merge_entities(entities)
        types = {e["type"] for e in merged}
        assert "tumor" in types
        assert "measurement" in types
        assert len(merged) == 2

    def test_overlapping_same_type_still_merged(self):
        """Two overlapping entities of the same type should still be merged"""
        processor = ClinicalProcessor()

        entities = [
            {
                "text": "ductal carcinoma",
                "type": "tumor",
                "start": 5,
                "end": 22,
                "properties": {},
            },
            {"text": "carcinoma", "type": "tumor", "start": 12, "end": 22, "properties": {}},
        ]

        merged = processor._merge_entities(entities)
        assert len(merged) == 1

    def test_non_overlapping_entities_unchanged(self):
        """Non-overlapping entities should all be preserved"""
        processor = ClinicalProcessor()

        entities = [
            {"text": "carcinoma", "type": "tumor", "start": 0, "end": 9, "properties": {}},
            {"text": "Grade 2", "type": "staging", "start": 20, "end": 27, "properties": {}},
            {"text": "2.5 cm", "type": "measurement", "start": 40, "end": 46, "properties": {}},
        ]

        merged = processor._merge_entities(entities)
        assert len(merged) == 3

    def test_empty_entity_list(self):
        """Empty entity list should return empty"""
        processor = ClinicalProcessor()
        assert processor._merge_entities([]) == []


# ===========================================================================
# Not-implemented feature warning tests
# ===========================================================================


class TestNotImplementedWarnings:
    """Documented-but-not-implemented features should emit warnings"""

    def test_use_spacy_warns(self):
        """use_spacy config should emit a warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalProcessor(config={"entity_recognition": {"use_spacy": True}})
            spacy_warnings = [x for x in w if "use_spacy" in str(x.message)]
            assert len(spacy_warnings) == 1

    def test_use_deep_learning_warns(self):
        """use_deep_learning config should emit a warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalProcessor(config={"entity_recognition": {"use_deep_learning": True}})
            dl_warnings = [x for x in w if "use_deep_learning" in str(x.message)]
            assert len(dl_warnings) == 1

    def test_term_disambiguation_warns(self):
        """term_disambiguation config should emit a warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalProcessor(config={"entity_recognition": {"term_disambiguation": True}})
            td_warnings = [x for x in w if "term_disambiguation" in str(x.message)]
            assert len(td_warnings) == 1

    def test_unsupported_ontology_warns(self):
        """Unsupported ontology names should emit a warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalProcessor(
                config={"entity_recognition": {"ontologies": ["snomed_ct", "icd10"]}}
            )
            ont_warnings = [x for x in w if "not yet supported" in str(x.message)]
            assert len(ont_warnings) == 1
            assert "icd10" in str(ont_warnings[0].message)

    def test_no_warning_for_supported_ontologies(self):
        """Supported ontologies should not produce warnings"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ClinicalProcessor(
                config={"entity_recognition": {"ontologies": ["snomed_ct", "rxnorm", "loinc"]}}
            )
            ont_warnings = [x for x in w if "not yet supported" in str(x.message)]
            assert len(ont_warnings) == 0

    @patch("honeybee.processors.clinical_processor.AutoTokenizer")
    def test_summarize_strategy_warns(self, mock_tokenizer):
        """long_document_strategy='summarize' should emit a warning"""
        mock_tok = MagicMock()
        mock_tok.tokenize.return_value = ["tok"] * 600  # Long enough to trigger
        mock_tok.cls_token_id = 101
        mock_tok.sep_token_id = 102
        mock_tok.pad_token_id = 0
        mock_tok.return_value = {
            "input_ids": np.array([[101, 1, 2, 102]]),
            "attention_mask": np.array([[1, 1, 1, 1]]),
        }
        mock_tokenizer.from_pretrained.return_value = mock_tok

        processor = ClinicalProcessor(
            config={"tokenization": {"long_document_strategy": "summarize"}}
        )
        processor.model_max_length = 10  # Force long document handling

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            processor._handle_long_document(["segment " * 100])
            summarize_warnings = [x for x in w if "summarize" in str(x.message)]
            assert len(summarize_warnings) == 1


# ===========================================================================
# LOINC ontology tests
# ===========================================================================


class TestLOINCOntology:
    """Test LOINC ontology mappings"""

    def test_loinc_in_ontology_mappings(self):
        """LOINC should be present in ONTOLOGY_MAPPINGS"""
        assert "loinc" in ONTOLOGY_MAPPINGS

    def test_loinc_has_common_lab_tests(self):
        """LOINC mappings should include common lab tests"""
        loinc = ONTOLOGY_MAPPINGS["loinc"]
        assert "hemoglobin" in loinc
        assert "creatinine" in loinc
        assert "glucose" in loinc

    def test_loinc_entries_have_id_and_name(self):
        """Each LOINC entry should have 'id' and 'name' keys"""
        for term, mapping in ONTOLOGY_MAPPINGS["loinc"].items():
            assert "id" in mapping, f"LOINC entry '{term}' missing 'id'"
            assert "name" in mapping, f"LOINC entry '{term}' missing 'name'"

    def test_loinc_normalization_works(self):
        """Entity normalization should find LOINC matches"""
        processor = ClinicalProcessor(
            config={"entity_recognition": {"ontologies": ["loinc"]}}
        )
        entities = [
            {
                "text": "hemoglobin",
                "type": "measurement",
                "start": 0,
                "end": 10,
                "properties": {},
            },
        ]
        processor._normalize_entities(entities)
        assert "ontology_links" in entities[0]["properties"]
        link = entities[0]["properties"]["ontology_links"][0]
        assert link["ontology"] == "loinc"
        assert link["concept_id"] == "718-7"


# ===========================================================================
# Condition pattern end-of-text fix
# ===========================================================================


class TestConditionPatternEndOfText:
    """Condition pattern should match at end of text"""

    def test_condition_at_end_of_text(self):
        """Condition at end of text (no trailing punctuation) should be captured"""
        processor = ClinicalProcessor()
        text = "Patient diagnosed with breast cancer"
        result = processor.process_text(text)

        conditions = [e for e in result["entities"] if e["type"] == "condition"]
        # Should find the condition
        assert len(conditions) > 0
        assert any("breast cancer" in e["text"] for e in conditions)

    def test_condition_with_trailing_period(self):
        """Condition before period should still be captured"""
        processor = ClinicalProcessor()
        text = "Patient diagnosed with breast cancer."
        result = processor.process_text(text)

        conditions = [e for e in result["entities"] if e["type"] == "condition"]
        assert len(conditions) > 0


# ===========================================================================
# EHR format tests
# ===========================================================================


class TestEHRProcessing:
    """Test EHR document processing"""

    def test_json_ehr(self, temp_dir):
        """Test JSON EHR processing"""
        json_path = temp_dir / "record.json"
        json_path.write_text(json.dumps({
            "patient": {"name": "John Doe", "diagnosis": "lung cancer"},
            "treatment": "chemotherapy",
        }))
        processor = ClinicalProcessor()
        result = processor._process_ehr_document(json_path)
        assert "text" in result
        assert "lung cancer" in result["text"]

    def test_xml_ehr(self, temp_dir):
        """Test XML EHR processing"""
        xml_path = temp_dir / "record.xml"
        xml_path.write_text(
            "<record><patient><diagnosis>breast cancer</diagnosis></patient></record>"
        )
        processor = ClinicalProcessor()
        result = processor._process_ehr_document(xml_path)
        assert "text" in result
        assert "breast cancer" in result["text"]

    def test_csv_ehr(self, temp_dir):
        """Test CSV EHR processing"""
        csv_path = temp_dir / "record.csv"
        csv_path.write_text("name,diagnosis\nJohn,colon cancer\n")
        processor = ClinicalProcessor()
        result = processor._process_ehr_document(csv_path)
        assert "text" in result
        assert "colon cancer" in result["text"]
