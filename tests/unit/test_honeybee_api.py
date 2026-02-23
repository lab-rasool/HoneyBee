"""
Unit tests for the main HoneyBee API

Tests the core HoneyBee class and its methods for multimodal integration.
"""

from unittest.mock import patch

import numpy as np
import pytest

from honeybee import HoneyBee


class TestHoneyBeeInitialization:
    """Test HoneyBee initialization"""

    def test_init_without_config(self):
        """Test initialization without configuration"""
        honeybee = HoneyBee()
        assert honeybee is not None
        assert honeybee.clinical_processor is not None
        assert honeybee.config == {}

    def test_init_with_config(self, honeybee_config):
        """Test initialization with configuration"""
        honeybee = HoneyBee(config=honeybee_config)
        assert honeybee is not None
        assert honeybee.config == honeybee_config
        assert honeybee.clinical_processor is not None

    def test_clinical_processor_initialized(self):
        """Test that clinical processor is properly initialized"""
        honeybee = HoneyBee()
        from honeybee.processors import ClinicalProcessor

        assert isinstance(honeybee.clinical_processor, ClinicalProcessor)


class TestGenerateEmbeddings:
    """Test generate_embeddings method"""

    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_clinical_embeddings_single_text(self, mock_generate):
        """Test generating embeddings for single clinical text"""
        mock_generate.return_value = np.random.randn(1, 768)

        honeybee = HoneyBee()
        text = "Patient diagnosed with lung cancer"
        embeddings = honeybee.generate_embeddings(text, modality="clinical")

        assert embeddings is not None
        assert embeddings.shape == (1, 768)
        mock_generate.assert_called_once_with(text)

    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_clinical_embeddings_multiple_texts(self, mock_generate):
        """Test generating embeddings for multiple clinical texts"""
        mock_generate.return_value = np.random.randn(3, 768)

        honeybee = HoneyBee()
        texts = [
            "Patient with breast cancer",
            "Stage III adenocarcinoma",
            "ER positive HER2 negative",
        ]
        embeddings = honeybee.generate_embeddings(texts, modality="clinical")

        assert embeddings is not None
        assert embeddings.shape == (3, 768)
        mock_generate.assert_called_once_with(texts)

    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_clinical_embeddings_with_model_name(self, mock_generate):
        """Test generating embeddings with specific model"""
        mock_generate.return_value = np.random.randn(1, 768)

        honeybee = HoneyBee()
        text = "Patient with cancer"
        embeddings = honeybee.generate_embeddings(
            text, modality="clinical", model_name="pubmedbert"
        )

        assert embeddings is not None
        mock_generate.assert_called_once_with(text, model_name="pubmedbert")

    def test_pathology_modality_placeholder(self):
        """Test that pathology modality returns placeholder"""
        honeybee = HoneyBee()
        embeddings = honeybee.generate_embeddings(
            np.random.randn(224, 224, 3), modality="pathology"
        )

        # Should return placeholder embedding
        assert embeddings is not None
        assert embeddings.shape == (1, 768)

    @patch("honeybee.RadiologyProcessor")
    def test_radiology_modality(self, mock_rad_cls):
        """Test generating embeddings for radiology modality"""
        mock_processor = mock_rad_cls.return_value
        mock_processor.generate_embeddings.return_value = np.random.randn(2048)

        honeybee = HoneyBee()
        image = np.random.randn(64, 128, 128)
        embeddings = honeybee.generate_embeddings(image, modality="radiology")

        assert embeddings is not None
        mock_processor.generate_embeddings.assert_called_once()

    def test_radiology_modality_invalid_input(self):
        """Test that non-array input for radiology raises error"""
        honeybee = HoneyBee()
        with pytest.raises(ValueError, match="Radiology modality requires numpy array"):
            honeybee.generate_embeddings("text input", modality="radiology")

    def test_radiology_processor_lazy_init(self):
        """Test that radiology processor is lazily initialized"""
        honeybee = HoneyBee()
        assert honeybee._radiology_processor is None

    def test_invalid_clinical_input_type(self):
        """Test that non-text input for clinical raises error"""
        honeybee = HoneyBee()
        with pytest.raises(ValueError, match="Clinical modality requires text input"):
            honeybee.generate_embeddings(np.random.randn(100, 100), modality="clinical")


class TestIntegrateEmbeddings:
    """Test integrate_embeddings method"""

    def test_integrate_single_modality(self):
        """Test integrating embeddings from single modality"""
        honeybee = HoneyBee()
        emb1 = np.random.randn(1, 768)

        integrated = honeybee.integrate_embeddings([emb1])

        assert integrated is not None
        assert integrated.shape == (1, 768)

    def test_integrate_two_modalities(self):
        """Test integrating embeddings from two modalities"""
        honeybee = HoneyBee()
        emb1 = np.random.randn(1, 768)
        emb2 = np.random.randn(1, 512)

        integrated = honeybee.integrate_embeddings([emb1, emb2])

        assert integrated is not None
        assert integrated.shape == (1, 768 + 512)

    def test_integrate_three_modalities(self):
        """Test integrating embeddings from three modalities"""
        honeybee = HoneyBee()
        emb1 = np.random.randn(1, 768)  # Clinical
        emb2 = np.random.randn(1, 1024)  # Pathology
        emb3 = np.random.randn(1, 2048)  # Radiology

        integrated = honeybee.integrate_embeddings([emb1, emb2, emb3])

        assert integrated is not None
        assert integrated.shape == (1, 768 + 1024 + 2048)

    def test_integrate_empty_list(self):
        """Test that empty list raises error"""
        honeybee = HoneyBee()
        with pytest.raises(ValueError, match="No embeddings provided"):
            honeybee.integrate_embeddings([])

    def test_integrate_batch_embeddings(self):
        """Test integrating batch embeddings"""
        honeybee = HoneyBee()
        emb1 = np.random.randn(5, 768)
        emb2 = np.random.randn(5, 512)

        integrated = honeybee.integrate_embeddings([emb1, emb2])

        assert integrated is not None
        assert integrated.shape == (5, 768 + 512)


class TestProcessClinical:
    """Test process_clinical method"""

    @patch("honeybee.processors.ClinicalProcessor.process")
    def test_process_document(self, mock_process, temp_pdf_file):
        """Test processing clinical document"""
        mock_process.return_value = {"text": "Patient with cancer", "entities": []}

        honeybee = HoneyBee()
        result = honeybee.process_clinical(document_path=temp_pdf_file)

        assert result is not None
        assert "text" in result
        mock_process.assert_called_once()

    @patch("honeybee.processors.ClinicalProcessor.process_text")
    def test_process_text(self, mock_process_text, sample_clinical_text):
        """Test processing clinical text directly"""
        mock_process_text.return_value = {"text": sample_clinical_text, "entities": []}

        honeybee = HoneyBee()
        result = honeybee.process_clinical(text=sample_clinical_text)

        assert result is not None
        assert "text" in result
        mock_process_text.assert_called_once_with(sample_clinical_text)

    def test_process_without_input(self):
        """Test that calling without input raises error"""
        honeybee = HoneyBee()
        with pytest.raises(ValueError, match="Either document_path or text must be provided"):
            honeybee.process_clinical()

    @patch("honeybee.processors.ClinicalProcessor.process")
    def test_process_with_save_output(self, mock_process, temp_pdf_file):
        """Test processing with save_output=True"""
        mock_process.return_value = {"text": "Patient with cancer", "entities": []}

        honeybee = HoneyBee()
        result = honeybee.process_clinical(document_path=temp_pdf_file, save_output=True)

        assert result is not None
        mock_process.assert_called_once_with(temp_pdf_file, save_output=True)


class TestProcessClinicalBatch:
    """Test process_clinical_batch method"""

    @patch("honeybee.processors.ClinicalProcessor.process_batch")
    def test_batch_processing(self, mock_process_batch, temp_dir):
        """Test batch processing of clinical documents"""
        mock_process_batch.return_value = [
            {"text": "Doc 1", "entities": []},
            {"text": "Doc 2", "entities": []},
        ]

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(input_dir=temp_dir)

        assert results is not None
        assert len(results) == 2
        mock_process_batch.assert_called_once()

    @patch("honeybee.processors.ClinicalProcessor.process_batch")
    def test_batch_with_pattern(self, mock_process_batch, temp_dir):
        """Test batch processing with file pattern"""
        mock_process_batch.return_value = []

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(input_dir=temp_dir, file_pattern="*.pdf")

        assert results is not None
        mock_process_batch.assert_called_once_with(
            input_dir=temp_dir, file_pattern="*.pdf", save_output=True, output_dir=None
        )

    @patch("honeybee.processors.ClinicalProcessor.process_batch")
    def test_batch_with_output_dir(self, mock_process_batch, temp_dir, temp_output_dir):
        """Test batch processing with custom output directory"""
        mock_process_batch.return_value = []

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(input_dir=temp_dir, output_dir=temp_output_dir)

        assert results is not None
        mock_process_batch.assert_called_once()


class TestProcessRadiology:
    """Test process_radiology method"""

    @patch("honeybee.RadiologyProcessor")
    def test_process_radiology_with_image(self, mock_rad_cls):
        """Test processing radiology with pre-loaded image"""
        mock_processor = mock_rad_cls.return_value

        honeybee = HoneyBee()
        image = np.random.randn(64, 128, 128)
        result = honeybee.process_radiology(image=image, preprocess=False)

        assert result is not None
        assert "image" in result
        assert np.array_equal(result["image"], image)
        assert result["metadata"] is None

    @patch("honeybee.RadiologyProcessor")
    def test_process_radiology_with_dicom(self, mock_rad_cls):
        """Test processing radiology with DICOM path"""
        from honeybee.processors.radiology.metadata import ImageMetadata

        mock_processor = mock_rad_cls.return_value
        mock_metadata = ImageMetadata(
            modality="CT",
            patient_id="TEST",
            study_date="20240101",
            series_description="CT",
            pixel_spacing=(1.0, 1.0, 1.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=[1, 0, 0, 0, 1, 0],
        )
        mock_processor.load_dicom.return_value = (
            np.random.randn(64, 128, 128),
            mock_metadata,
        )
        mock_processor.preprocess.return_value = np.random.randn(64, 128, 128)

        honeybee = HoneyBee()
        result = honeybee.process_radiology(dicom_path="/fake/dicom")

        assert result is not None
        assert "image" in result
        assert "metadata" in result
        mock_processor.load_dicom.assert_called_once_with("/fake/dicom")

    @patch("honeybee.RadiologyProcessor")
    def test_process_radiology_with_embeddings(self, mock_rad_cls):
        """Test process_radiology with embedding generation"""
        mock_processor = mock_rad_cls.return_value
        mock_processor.generate_embeddings.return_value = np.random.randn(2048)

        honeybee = HoneyBee()
        image = np.random.randn(64, 128, 128)
        result = honeybee.process_radiology(
            image=image, preprocess=False, generate_embeddings=True
        )

        assert "embeddings" in result
        mock_processor.generate_embeddings.assert_called_once()

    def test_process_radiology_no_input(self):
        """Test that calling without input raises error"""
        honeybee = HoneyBee()
        with pytest.raises(ValueError, match="One of dicom_path, nifti_path, or image"):
            honeybee.process_radiology()


class TestPredictSurvival:
    """Test predict_survival method"""

    def test_survival_prediction_placeholder(self):
        """Test survival prediction returns placeholder"""
        honeybee = HoneyBee()
        embeddings = np.random.randn(1, 2048)

        result = honeybee.predict_survival(embeddings)

        assert result is not None
        assert "survival_probability" in result
        assert "risk_score" in result
        assert "confidence" in result
        assert "time_to_event" in result

        # Check value ranges
        assert 0 <= result["survival_probability"] <= 1
        assert 0 <= result["risk_score"] <= 1
        assert 0 <= result["confidence"] <= 1
        assert result["time_to_event"] >= 0


class TestEndToEnd:
    """End-to-end tests for HoneyBee API"""

    @patch("honeybee.processors.ClinicalProcessor.process_text")
    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_complete_clinical_workflow(self, mock_generate, mock_process, sample_clinical_text):
        """Test complete workflow: text → processing → embeddings"""
        mock_process.return_value = {
            "text": sample_clinical_text,
            "entities": [{"type": "tumor", "text": "invasive ductal carcinoma"}],
        }
        mock_generate.return_value = np.random.randn(1, 768)

        honeybee = HoneyBee()

        # Step 1: Process text
        result = honeybee.process_clinical(text=sample_clinical_text)
        assert result is not None
        assert "entities" in result

        # Step 2: Generate embeddings
        embeddings = honeybee.generate_embeddings(sample_clinical_text, modality="clinical")
        assert embeddings is not None
        assert embeddings.shape == (1, 768)

    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_multimodal_integration_workflow(self, mock_generate):
        """Test multimodal integration workflow"""
        mock_generate.return_value = np.random.randn(1, 768)

        honeybee = HoneyBee()

        # Generate embeddings for each modality
        clinical_emb = honeybee.generate_embeddings("Patient with cancer", modality="clinical")
        pathology_emb = honeybee.generate_embeddings(np.ones((224, 224, 3)), modality="pathology")

        # Integrate
        integrated = honeybee.integrate_embeddings([clinical_emb, pathology_emb])
        assert integrated is not None
        assert integrated.shape == (1, 768 * 2)

        # Predict survival
        survival = honeybee.predict_survival(integrated)
        assert survival is not None
        assert "survival_probability" in survival
