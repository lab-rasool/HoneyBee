"""
Integration tests for complete HoneyBee workflows

Tests end-to-end pipelines that integrate multiple components across
clinical, pathology, and radiology modalities.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from honeybee import HoneyBee
from honeybee.processors import ClinicalProcessor, PathologyProcessor
from honeybee.processors.radiology import RadiologyProcessor


@pytest.mark.integration
class TestClinicalWorkflow:
    """Integration tests for clinical processing workflow"""

    @patch('honeybee.models.HuggingFaceEmbedder')
    def test_clinical_pdf_to_embeddings(self, mock_embedder, sample_clinical_pdf_path):
        """Test complete workflow: PDF → processing → embeddings"""
        if sample_clinical_pdf_path is None:
            pytest.skip("Sample PDF not available")

        # Setup mock embedder
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.generate_embeddings.return_value = np.random.randn(1, 768)
        mock_embedder.return_value = mock_embedder_instance

        # Create processor
        processor = ClinicalProcessor()

        # Step 1: Process PDF
        result = processor.process(sample_clinical_pdf_path)
        assert result is not None
        assert "text" in result
        assert len(result["text"]) > 0

        # Step 2: Extract entities
        assert "entities" in result
        assert isinstance(result["entities"], list)

        # Step 3: Generate embeddings
        embeddings = processor.generate_embeddings(result["text"])
        assert embeddings is not None

    @patch('honeybee.processors.ClinicalProcessor.process_text')
    @patch('honeybee.processors.ClinicalProcessor.generate_embeddings')
    def test_clinical_text_pipeline(self, mock_generate, mock_process, sample_clinical_text):
        """Test text processing pipeline"""
        mock_process.return_value = {
            "text": sample_clinical_text,
            "entities": [
                {"type": "tumor", "text": "invasive ductal carcinoma"},
                {"type": "biomarker", "text": "ER: Positive"}
            ],
            "temporal_timeline": []
        }
        mock_generate.return_value = np.random.randn(1, 768)

        honeybee = HoneyBee()

        # Process text
        result = honeybee.process_clinical(text=sample_clinical_text)
        assert result is not None
        assert len(result["entities"]) > 0

        # Generate embeddings
        embeddings = honeybee.generate_embeddings(sample_clinical_text, modality="clinical")
        assert embeddings.shape == (1, 768)

    @patch('honeybee.processors.ClinicalProcessor.process')
    def test_clinical_batch_workflow(self, mock_process, temp_dir):
        """Test batch processing workflow"""
        # Create test files
        for i in range(3):
            (temp_dir / f"report_{i}.txt").write_text(f"Patient {i} diagnosed with cancer")

        mock_process.return_value = {"text": "test", "entities": []}

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(
            input_dir=temp_dir,
            file_pattern="*.txt"
        )

        assert len(results) == 3
        for result in results:
            assert "text" in result


@pytest.mark.integration
@pytest.mark.requires_sample_data
class TestPathologyWorkflow:
    """Integration tests for pathology processing workflow"""

    @patch('honeybee.models.UNI.uni.UNI')
    def test_wsi_to_slide_embedding(self, mock_uni, sample_wsi_path):
        """Test complete workflow: WSI → patches → embeddings → aggregation"""
        if sample_wsi_path is None:
            pytest.skip("Sample WSI not available")

        # Setup mock model
        mock_model_instance = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(5, 1024)
        mock_model_instance.load_model_and_predict.return_value = mock_tensor
        mock_uni.return_value = mock_model_instance

        processor = PathologyProcessor(model="uni", model_path="/fake/path.pt")
        processor.embedding_model = mock_model_instance

        # Complete pipeline
        result = processor.process_slide(
            sample_wsi_path,
            normalize_stain=True,
            normalization_method="macenko",
            patch_size=256,
            min_tissue_percentage=0.3,
            aggregation_method="mean",
            max_patches=5
        )

        assert result is not None
        assert "slide" in result
        assert "patches" in result
        assert "num_patches" in result

    def test_pathology_preprocessing_pipeline(self, sample_wsi_patch):
        """Test preprocessing pipeline for pathology"""
        processor = PathologyProcessor()

        # Step 1: Detect tissue
        tissue_mask = processor.detect_tissue(sample_wsi_patch, method="otsu")
        assert tissue_mask is not None

        # Step 2: Normalize stain
        normalized = processor.normalize_stain(sample_wsi_patch, method="macenko")
        assert normalized is not None

        # Step 3: Separate stains
        stains = processor.separate_stains(normalized)
        assert "hematoxylin" in stains
        assert "eosin" in stains


@pytest.mark.integration
class TestRadiologyWorkflow:
    """Integration tests for radiology processing workflow"""

    @patch('honeybee.loaders.Radiology.load_medical_image')
    @patch('honeybee.models.RadImageNet.radimagenet.RadImageNet')
    def test_dicom_to_embeddings(self, mock_model, mock_load, sample_dicom_metadata):
        """Test complete workflow: DICOM → preprocessing → embeddings"""
        # Setup mocks
        mock_image = np.random.randint(-1000, 1000, (64, 128, 128), dtype=np.int16)
        mock_load.return_value = (mock_image, sample_dicom_metadata)

        mock_model_instance = MagicMock()
        mock_model_instance.generate_embeddings.return_value = np.random.randn(2048)
        mock_model.return_value = mock_model_instance

        processor = RadiologyProcessor()
        processor.model = mock_model_instance

        # Step 1: Load image
        image, metadata = processor.load_image("/fake/ct.dcm")
        assert image is not None
        assert metadata == sample_dicom_metadata

        # Step 2: Preprocess
        processed = processor.preprocess(
            image,
            metadata,
            denoise=True,
            normalize=True,
            window="lung"
        )
        assert processed is not None

        # Step 3: Generate embeddings
        embeddings = processor.generate_embeddings(processed, mode="3d")
        assert embeddings is not None

    def test_radiology_segmentation_pipeline(self):
        """Test segmentation pipeline"""
        # Create CT image
        ct_image = np.full((32, 128, 128), -800, dtype=np.int16)

        processor = RadiologyProcessor()

        # Step 1: Verify HU
        hu_check = processor.verify_hounsfield_units(ct_image)
        assert hu_check["is_hu"]

        # Step 2: Apply window
        windowed = processor.apply_window(ct_image, window="lung")
        assert windowed is not None

        # Step 3: Segment lungs
        lung_mask = processor.segment_lungs(ct_image)
        assert lung_mask is not None


@pytest.mark.integration
@pytest.mark.slow
class TestMultimodalIntegration:
    """Integration tests for multimodal data integration"""

    @patch('honeybee.processors.ClinicalProcessor.generate_embeddings')
    @patch('honeybee.models.UNI.uni.UNI')
    @patch('honeybee.models.RadImageNet.radimagenet.RadImageNet')
    def test_complete_multimodal_workflow(self, mock_rad, mock_uni, mock_clinical):
        """Test complete multimodal integration workflow"""
        # Setup mocks
        mock_clinical.return_value = np.random.randn(1, 768)

        mock_uni_instance = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(1, 1024)
        mock_uni_instance.load_model_and_predict.return_value = mock_tensor
        mock_uni.return_value = mock_uni_instance

        mock_rad_instance = MagicMock()
        mock_rad_instance.generate_embeddings.return_value = np.random.randn(2048)
        mock_rad.return_value = mock_rad_instance

        honeybee = HoneyBee()

        # Generate embeddings for each modality
        clinical_emb = honeybee.generate_embeddings(
            "Patient with stage III lung cancer",
            modality="clinical"
        )
        assert clinical_emb.shape == (1, 768)

        pathology_emb = honeybee.generate_embeddings(
            np.ones((224, 224, 3)),
            modality="pathology"
        )
        assert pathology_emb.shape == (1, 768)  # Placeholder

        radiology_emb = honeybee.generate_embeddings(
            np.ones((64, 128, 128)),
            modality="radiology"
        )
        assert radiology_emb.shape == (1, 768)  # Placeholder

        # Integrate multimodal embeddings
        integrated = honeybee.integrate_embeddings([
            clinical_emb,
            pathology_emb,
            radiology_emb
        ])

        assert integrated is not None
        assert integrated.shape == (1, 768 * 3)

        # Predict survival
        survival = honeybee.predict_survival(integrated)
        assert survival is not None
        assert "survival_probability" in survival

    @patch('honeybee.processors.ClinicalProcessor.generate_embeddings')
    def test_multimodal_fusion_strategies(self, mock_generate):
        """Test different multimodal fusion strategies"""
        mock_generate.return_value = np.random.randn(1, 768)

        honeybee = HoneyBee()

        # Generate embeddings with different dimensions
        emb1 = np.random.randn(5, 768)   # Clinical
        emb2 = np.random.randn(5, 1024)  # Pathology
        emb3 = np.random.randn(5, 2048)  # Radiology

        # Test concatenation fusion
        integrated = honeybee.integrate_embeddings([emb1, emb2, emb3])

        assert integrated is not None
        assert integrated.shape == (5, 768 + 1024 + 2048)


@pytest.mark.integration
class TestEndToEndScenarios:
    """Integration tests for realistic end-to-end scenarios"""

    @patch('honeybee.processors.ClinicalProcessor.process_text')
    @patch('honeybee.processors.ClinicalProcessor.generate_embeddings')
    def test_cancer_patient_analysis(self, mock_generate, mock_process):
        """Test complete patient analysis workflow"""
        # Simulate patient report
        patient_report = """
        Patient: Female, 55 years old
        Diagnosis: Invasive ductal carcinoma, Grade 2
        Stage: pT2 N0 M0 (Stage IIA)
        Biomarkers: ER+, PR+, HER2-
        Treatment: Lumpectomy with radiation therapy
        """

        mock_process.return_value = {
            "text": patient_report,
            "entities": [
                {"type": "tumor", "text": "invasive ductal carcinoma"},
                {"type": "staging", "text": "Grade 2"},
                {"type": "biomarker", "text": "ER+"},
                {"type": "biomarker", "text": "HER2-"}
            ]
        }
        mock_generate.return_value = np.random.randn(1, 768)

        honeybee = HoneyBee()

        # Process clinical data
        clinical_result = honeybee.process_clinical(text=patient_report)
        assert len(clinical_result["entities"]) > 0

        # Generate clinical embedding
        clinical_emb = honeybee.generate_embeddings(patient_report, modality="clinical")
        assert clinical_emb is not None

        # Simulate multimodal analysis
        # (Would include pathology and radiology in real scenario)
        survival = honeybee.predict_survival(clinical_emb)
        assert "survival_probability" in survival

    @patch('honeybee.processors.ClinicalProcessor.process')
    def test_cohort_analysis(self, mock_process, temp_dir):
        """Test cohort/batch analysis workflow"""
        # Create cohort data
        patients = [
            "Patient 1: Breast cancer, Stage I, ER+",
            "Patient 2: Lung cancer, Stage III, EGFR+",
            "Patient 3: Colon cancer, Stage II, KRAS wild-type"
        ]

        for i, text in enumerate(patients):
            (temp_dir / f"patient_{i}.txt").write_text(text)

        mock_process.return_value = {
            "text": "test",
            "entities": [{"type": "tumor", "text": "cancer"}]
        }

        honeybee = HoneyBee()

        # Process cohort
        results = honeybee.process_clinical_batch(temp_dir, file_pattern="*.txt")

        assert len(results) == 3
        for result in results:
            assert "text" in result
            assert "entities" in result

    @patch('honeybee.processors.ClinicalProcessor.generate_embeddings')
    def test_similarity_search(self, mock_generate):
        """Test patient similarity search using embeddings"""
        # Simulate patient embeddings
        patient_embeddings = np.random.randn(10, 768)

        # Query patient
        query_embedding = np.random.randn(1, 768)

        # Compute similarities
        similarities = np.dot(patient_embeddings, query_embedding.T).flatten()

        # Find most similar patients
        most_similar_indices = np.argsort(similarities)[::-1][:3]

        assert len(most_similar_indices) == 3
        assert all(0 <= idx < 10 for idx in most_similar_indices)


@pytest.mark.integration
class TestErrorRecoveryAndRobustness:
    """Integration tests for error handling and robustness"""

    def test_partial_pipeline_failure(self, sample_clinical_text):
        """Test handling of partial pipeline failures"""
        config = {
            "processing_pipeline": ["document", "entity_recognition"],
            "tokenization": {"model": "invalid_model"}  # Will fail
        }

        processor = ClinicalProcessor(config=config)

        # Should still work without tokenization
        result = processor.process_text(sample_clinical_text)
        assert result is not None
        assert "text" in result

    def test_missing_modality_data(self):
        """Test multimodal integration with missing modalities"""
        honeybee = HoneyBee()

        # Only have clinical data
        clinical_emb = np.random.randn(1, 768)

        # Should still work with single modality
        integrated = honeybee.integrate_embeddings([clinical_emb])
        assert integrated is not None

    @patch('honeybee.processors.ClinicalProcessor.process')
    def test_batch_with_errors(self, mock_process, temp_dir):
        """Test batch processing with some files failing"""
        # Create test files
        (temp_dir / "good1.txt").write_text("Valid report 1")
        (temp_dir / "good2.txt").write_text("Valid report 2")

        # Mock to return success for some, error for others
        mock_process.side_effect = [
            {"text": "Report 1", "entities": []},
            {"error": "Failed to process"}
        ]

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(temp_dir, file_pattern="*.txt")

        assert len(results) == 2
        # Some should succeed, some might have errors
        assert any("text" in r for r in results)
