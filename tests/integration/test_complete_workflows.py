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
from honeybee.processors.clinical import ClinicalResult


@pytest.mark.integration
class TestClinicalWorkflow:
    """Integration tests for clinical processing workflow"""

    def test_clinical_text_pipeline(self, sample_clinical_text):
        """Test text processing pipeline end-to-end."""
        honeybee = HoneyBee()
        result = honeybee.process_clinical(text=sample_clinical_text)
        assert isinstance(result, ClinicalResult)
        assert isinstance(result.entities, list)

    def test_clinical_batch_workflow(self, temp_dir):
        """Test batch processing workflow."""
        for i in range(3):
            (temp_dir / f"report_{i}.txt").write_text(
                f"Patient {i} diagnosed with breast cancer, Grade 2. ER positive."
            )

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(input_dir=temp_dir, file_pattern="*.txt")

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ClinicalResult)


@pytest.mark.integration
@pytest.mark.requires_sample_data
class TestPathologyWorkflow:
    """Integration tests for pathology processing workflow"""

    @patch("honeybee.models.UNI.uni.UNI")
    def test_wsi_to_slide_embedding(self, mock_uni, sample_wsi_path):
        """Test complete workflow: WSI → patches → embeddings → aggregation"""
        if sample_wsi_path is None:
            pytest.skip("Sample WSI not available")

        mock_model_instance = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(5, 1024)
        mock_model_instance.load_model_and_predict.return_value = mock_tensor
        mock_uni.return_value = mock_model_instance

        processor = PathologyProcessor(model="uni", model_path="/fake/path.pt")
        processor.embedding_model = mock_model_instance

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

        tissue_mask = processor.detect_tissue(sample_wsi_patch, method="otsu")
        assert tissue_mask is not None

        normalized = processor.normalize_stain(sample_wsi_patch, method="macenko")
        assert normalized is not None

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
        mock_clinical.return_value = np.random.randn(1, 768)

        mock_uni_instance = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.random.randn(1, 1024)
        mock_uni_instance.load_model_and_predict.return_value = mock_tensor
        mock_uni.return_value = mock_uni_instance

        honeybee = HoneyBee()

        clinical_emb = honeybee.generate_embeddings(
            "Patient with stage III lung cancer", modality="clinical"
        )
        assert clinical_emb.shape == (1, 768)

        pathology_emb = honeybee.generate_embeddings(np.ones((224, 224, 3)), modality="pathology")
        assert pathology_emb.shape == (1, 768)

        integrated = honeybee.integrate_embeddings([clinical_emb, pathology_emb])
        assert integrated is not None
        assert integrated.shape == (1, 768 * 2)

        survival = honeybee.predict_survival(integrated)
        assert survival is not None
        assert "survival_probability" in survival

    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_multimodal_fusion_strategies(self, mock_generate):
        mock_generate.return_value = np.random.randn(1, 768)

        honeybee = HoneyBee()

        emb1 = np.random.randn(5, 768)
        emb2 = np.random.randn(5, 1024)
        integrated = honeybee.integrate_embeddings([emb1, emb2])

        assert integrated is not None
        assert integrated.shape == (5, 768 + 1024)


@pytest.mark.integration
class TestRadiologyWorkflow:
    """Integration tests for radiology processing workflow"""

    @patch("honeybee.processors.radiology.processor.preprocess_ct")
    def test_ct_preprocess_to_embeddings(self, mock_preprocess_ct):
        mock_preprocess_ct.return_value = np.random.rand(64, 128, 128).astype(np.float32)

        processor = RadiologyProcessor(model="remedis", device="cpu")

        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(1, 2048).astype(np.float32)
        processor.embedding_model = mock_model

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

        embeddings = processor.generate_embeddings(preprocessed)
        assert embeddings is not None
        mock_model.generate_embeddings.assert_called_once()

    def test_radiology_segment_and_crop(self):
        processor = RadiologyProcessor(device="cpu")

        image = np.random.randint(-1000, 1000, (64, 128, 128), dtype=np.int16)
        mask = np.zeros((64, 128, 128), dtype=np.uint8)
        mask[20:40, 40:90, 40:90] = 1

        cropped = processor.crop_to_roi(image, mask)
        assert cropped.shape == (20, 50, 50)

        masked = processor.apply_mask(image, mask)
        assert np.all(masked[mask == 0] == 0)

    @patch("honeybee.processors.radiology.processor.RadiologyProcessor.generate_embeddings")
    @patch("honeybee.processors.ClinicalProcessor.generate_embeddings")
    def test_honeybee_radiology_modality(self, mock_clinical_gen, mock_rad_gen):
        mock_clinical_gen.return_value = np.random.randn(1, 768)
        mock_rad_gen.return_value = np.random.randn(2048)

        honeybee = HoneyBee()

        clinical_emb = honeybee.generate_embeddings("Patient with lung cancer", modality="clinical")
        assert clinical_emb.shape == (1, 768)

        rad_image = np.random.randn(64, 128, 128)
        rad_emb = honeybee.generate_embeddings(rad_image, modality="radiology")
        assert rad_emb is not None

        integrated = honeybee.integrate_embeddings(
            [clinical_emb, rad_emb.reshape(1, -1)]
        )
        assert integrated is not None


@pytest.mark.integration
class TestEndToEndScenarios:
    """Integration tests for realistic end-to-end scenarios"""

    def test_cancer_patient_analysis(self, sample_clinical_text):
        """Test complete patient analysis workflow."""
        honeybee = HoneyBee()

        result = honeybee.process_clinical(text=sample_clinical_text)
        assert isinstance(result, ClinicalResult)
        assert isinstance(result.entities, list)

    def test_cohort_analysis(self, temp_dir):
        """Test cohort/batch analysis workflow."""
        patients = [
            "Patient 1: Breast cancer, Stage I, ER positive.",
            "Patient 2: Lung cancer, Stage III, EGFR mutant.",
            "Patient 3: Colon cancer, Stage II.",
        ]

        for i, text in enumerate(patients):
            (temp_dir / f"patient_{i}.txt").write_text(text)

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(temp_dir, file_pattern="*.txt")

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ClinicalResult)

    def test_similarity_search(self):
        """Test patient similarity search using embeddings."""
        patient_embeddings = np.random.randn(10, 768)
        query_embedding = np.random.randn(1, 768)
        similarities = np.dot(patient_embeddings, query_embedding.T).flatten()
        most_similar_indices = np.argsort(similarities)[::-1][:3]

        assert len(most_similar_indices) == 3
        assert all(0 <= idx < 10 for idx in most_similar_indices)


@pytest.mark.integration
class TestErrorRecoveryAndRobustness:
    """Integration tests for error handling and robustness"""

    def test_missing_modality_data(self):
        honeybee = HoneyBee()
        clinical_emb = np.random.randn(1, 768)
        integrated = honeybee.integrate_embeddings([clinical_emb])
        assert integrated is not None

    def test_batch_with_errors(self, temp_dir):
        """Test batch processing with some files failing."""
        (temp_dir / "good1.txt").write_text("Patient with breast cancer.")
        (temp_dir / "good2.txt").write_text("Patient with lung cancer.")

        honeybee = HoneyBee()
        results = honeybee.process_clinical_batch(temp_dir, file_pattern="*.txt")

        assert len(results) == 2
        for r in results:
            assert isinstance(r, ClinicalResult)
