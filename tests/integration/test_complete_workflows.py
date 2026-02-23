"""
Integration tests for complete HoneyBee workflows

Tests end-to-end pipelines that integrate multiple components across
clinical and pathology modalities.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from honeybee import HoneyBee
from honeybee.processors import ClinicalProcessor, PathologyProcessor, RadiologyProcessor


@pytest.mark.integration
class TestClinicalWorkflow:
    """Integration tests for clinical processing workflow"""

    @patch("honeybee.models.HuggingFaceEmbedder")
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

    @patch("honeybee.processors.ClinicalProcessor.process_text")
    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_clinical_text_pipeline(self, mock_generate, mock_process, sample_clinical_text):
        """Test text processing pipeline"""
        mock_process.return_value = {
            "text": sample_clinical_text,
            "entities": [
                {"type": "tumor", "text": "invasive ductal carcinoma"},
                {"type": "biomarker", "text": "ER: Positive"},
            ],
            "temporal_timeline": [],
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

    @patch("honeybee.processors.ClinicalProcessor.process")
    def test_clinical_batch_workflow(self, mock_process, temp_dir):
        """Test batch processing workflow"""
        # Create test files
        for i in range(3):
            (temp_dir / f"report_{i}.txt").write_text(f"Patient {i} diagnosed with cancer")

        mock_process.return_value = {"text": "test", "entities": []}

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(input_dir=temp_dir, file_pattern="*.txt")

        assert len(results) == 3
        for result in results:
            assert "text" in result


@pytest.mark.integration
@pytest.mark.requires_sample_data
class TestPathologyWorkflow:
    """Integration tests for pathology processing workflow"""

    @patch("honeybee.models.UNI.uni.UNI")
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
            max_patches=5,
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
@pytest.mark.slow
class TestMultimodalIntegration:
    """Integration tests for multimodal data integration"""

    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    @patch("honeybee.models.UNI.uni.UNI")
    def test_complete_multimodal_workflow(self, mock_uni, mock_clinical):
        """Test complete multimodal integration workflow"""
        # Setup mocks
        mock_clinical.return_value = np.random.randn(1, 768)

        mock_uni_instance = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(1, 1024)
        mock_uni_instance.load_model_and_predict.return_value = mock_tensor
        mock_uni.return_value = mock_uni_instance

        honeybee = HoneyBee()

        # Generate embeddings for each modality
        clinical_emb = honeybee.generate_embeddings(
            "Patient with stage III lung cancer", modality="clinical"
        )
        assert clinical_emb.shape == (1, 768)

        pathology_emb = honeybee.generate_embeddings(np.ones((224, 224, 3)), modality="pathology")
        assert pathology_emb.shape == (1, 768)  # Placeholder

        # Integrate multimodal embeddings
        integrated = honeybee.integrate_embeddings([clinical_emb, pathology_emb])

        assert integrated is not None
        assert integrated.shape == (1, 768 * 2)

        # Predict survival
        survival = honeybee.predict_survival(integrated)
        assert survival is not None
        assert "survival_probability" in survival

    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_multimodal_fusion_strategies(self, mock_generate):
        """Test different multimodal fusion strategies"""
        mock_generate.return_value = np.random.randn(1, 768)

        honeybee = HoneyBee()

        # Generate embeddings with different dimensions
        emb1 = np.random.randn(5, 768)  # Clinical
        emb2 = np.random.randn(5, 1024)  # Pathology

        # Test concatenation fusion
        integrated = honeybee.integrate_embeddings([emb1, emb2])

        assert integrated is not None
        assert integrated.shape == (5, 768 + 1024)


@pytest.mark.integration
class TestRadiologyWorkflow:
    """Integration tests for radiology processing workflow"""

    @patch("honeybee.processors.radiology.processor.preprocess_ct")
    def test_ct_preprocess_to_embeddings(self, mock_preprocess_ct):
        """Test CT preprocessing -> embedding generation pipeline"""
        mock_preprocess_ct.return_value = np.random.rand(64, 128, 128).astype(np.float32)

        processor = RadiologyProcessor(model="remedis", device="cpu")

        # Mock embedding model
        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(1, 2048).astype(np.float32)
        processor.embedding_model = mock_model

        # Step 1: Preprocess
        from honeybee.processors.radiology.metadata import ImageMetadata

        metadata = ImageMetadata(
            modality="CT",
            patient_id="INT001",
            study_date="20240115",
            series_description="CHEST CT",
            pixel_spacing=(1.0, 1.0, 2.5),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=[1, 0, 0, 0, 1, 0],
        )
        image = np.random.randint(-1000, 1000, (64, 128, 128), dtype=np.int16)
        preprocessed = processor.preprocess(image, metadata)
        assert preprocessed is not None

        # Step 2: Generate embeddings
        embeddings = processor.generate_embeddings(preprocessed)
        assert embeddings is not None
        mock_model.generate_embeddings.assert_called_once()

    def test_radiology_segment_and_crop(self):
        """Test segmentation -> crop pipeline"""
        processor = RadiologyProcessor(device="cpu")

        image = np.random.randint(-1000, 1000, (64, 128, 128), dtype=np.int16)
        mask = np.zeros((64, 128, 128), dtype=np.uint8)
        mask[20:40, 40:90, 40:90] = 1

        # Crop to ROI
        cropped = processor.crop_to_roi(image, mask)
        assert cropped.shape == (20, 50, 50)

        # Apply mask
        masked = processor.apply_mask(image, mask)
        assert np.all(masked[mask == 0] == 0)

    @patch("honeybee.processors.radiology.processor.RadiologyProcessor.generate_embeddings")
    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_honeybee_radiology_modality(self, mock_clinical_gen, mock_rad_gen):
        """Test HoneyBee API with radiology modality"""
        mock_clinical_gen.return_value = np.random.randn(1, 768)
        mock_rad_gen.return_value = np.random.randn(2048)

        honeybee = HoneyBee()

        # Clinical embeddings
        clinical_emb = honeybee.generate_embeddings("Patient with lung cancer", modality="clinical")
        assert clinical_emb.shape == (1, 768)

        # Radiology embeddings
        rad_image = np.random.randn(64, 128, 128)
        rad_emb = honeybee.generate_embeddings(rad_image, modality="radiology")
        assert rad_emb is not None

        # Integrate
        integrated = honeybee.integrate_embeddings(
            [clinical_emb, rad_emb.reshape(1, -1)]
        )
        assert integrated is not None


@pytest.mark.integration
class TestEndToEndScenarios:
    """Integration tests for realistic end-to-end scenarios"""

    @patch("honeybee.processors.ClinicalProcessor.process_text")
    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
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
                {"type": "biomarker", "text": "HER2-"},
            ],
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
        # (Would include pathology in real scenario)
        survival = honeybee.predict_survival(clinical_emb)
        assert "survival_probability" in survival

    @patch("honeybee.processors.ClinicalProcessor.process")
    def test_cohort_analysis(self, mock_process, temp_dir):
        """Test cohort/batch analysis workflow"""
        # Create cohort data
        patients = [
            "Patient 1: Breast cancer, Stage I, ER+",
            "Patient 2: Lung cancer, Stage III, EGFR+",
            "Patient 3: Colon cancer, Stage II, KRAS wild-type",
        ]

        for i, text in enumerate(patients):
            (temp_dir / f"patient_{i}.txt").write_text(text)

        mock_process.return_value = {
            "text": "test",
            "entities": [{"type": "tumor", "text": "cancer"}],
        }

        honeybee = HoneyBee()

        # Process cohort
        results = honeybee.process_clinical_batch(temp_dir, file_pattern="*.txt")

        assert len(results) == 3
        for result in results:
            assert "text" in result
            assert "entities" in result

    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
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
            "tokenization": {"model": "invalid_model"},  # Will fail
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

    @patch("honeybee.processors.ClinicalProcessor.process")
    def test_batch_with_errors(self, mock_process, temp_dir):
        """Test batch processing with some files failing"""
        # Create test files
        (temp_dir / "good1.txt").write_text("Valid report 1")
        (temp_dir / "good2.txt").write_text("Valid report 2")

        # Mock to return success for some, error for others
        mock_process.side_effect = [
            {"text": "Report 1", "entities": []},
            {"error": "Failed to process"},
        ]

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(temp_dir, file_pattern="*.txt")

        assert len(results) == 2
        # Some should succeed, some might have errors
        assert any("text" in r for r in results)
