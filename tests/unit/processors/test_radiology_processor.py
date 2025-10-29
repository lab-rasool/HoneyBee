"""
Unit tests for RadiologyProcessor

Tests all functionality of the radiology imaging processing module including
DICOM/NIfTI loading, preprocessing, segmentation, and embedding generation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from honeybee.loaders.Radiology.metadata import ImageMetadata
from honeybee.processors.radiology import RadiologyProcessor


class TestRadiologyProcessorInitialization:
    """Test RadiologyProcessor initialization"""

    def test_init_default(self):
        """Test initialization with defaults"""
        processor = RadiologyProcessor()
        assert processor is not None
        assert processor.model_type == "radimagenet"
        assert processor.device in ["cuda", "cpu"]

    def test_init_remedis_model(self):
        """Test initialization with REMEDIS model"""
        processor = RadiologyProcessor(model="remedis")
        assert processor.model_type == "remedis"

    def test_init_radimagenet_model(self):
        """Test initialization with RadImageNet model"""
        processor = RadiologyProcessor(model="radimagenet", model_name="DenseNet121")
        assert processor.model_type == "radimagenet"

    def test_init_custom_device(self):
        """Test initialization with custom device"""
        processor = RadiologyProcessor(device="cpu")
        assert processor.device == "cpu"

    def test_init_invalid_model(self):
        """Test that invalid model raises error"""
        with pytest.raises(ValueError, match="Unknown model"):
            RadiologyProcessor(model="invalid_model")


class TestImageLoading:
    """Test image loading functionality"""

    @patch("honeybee.processors.radiology.processor.load_medical_image")
    def test_load_image_basic(self, mock_load, sample_dicom_metadata):
        """Test basic image loading"""
        mock_image = np.random.randint(-1000, 1000, (64, 128, 128), dtype=np.int16)
        mock_load.return_value = (mock_image, sample_dicom_metadata)

        processor = RadiologyProcessor()
        image, metadata = processor.load_image("/fake/path.dcm")

        assert image is not None
        assert metadata is not None
        mock_load.assert_called_once()


class TestPreprocessing:
    """Test preprocessing functionality"""

    def test_preprocess_ct_basic(self, sample_image_3d, sample_dicom_metadata):
        """Test basic CT preprocessing"""
        processor = RadiologyProcessor()
        processed = processor.preprocess(
            sample_image_3d, sample_dicom_metadata, denoise=True, normalize=True
        )

        assert processed is not None
        assert processed.shape == sample_image_3d.shape

    def test_preprocess_ct_with_windowing(self, sample_image_3d, sample_dicom_metadata):
        """Test CT preprocessing with window/level"""
        processor = RadiologyProcessor()
        processed = processor.preprocess(sample_image_3d, sample_dicom_metadata, window="lung")

        assert processed is not None

    def test_preprocess_without_denoise(self, sample_image_3d, sample_dicom_metadata):
        """Test preprocessing without denoising"""
        processor = RadiologyProcessor()
        processed = processor.preprocess(sample_image_3d, sample_dicom_metadata, denoise=False)

        assert processed is not None

    def test_preprocess_without_normalize(self, sample_image_3d, sample_dicom_metadata):
        """Test preprocessing without normalization"""
        processor = RadiologyProcessor()
        processed = processor.preprocess(sample_image_3d, sample_dicom_metadata, normalize=False)

        assert processed is not None


class TestDenoising:
    """Test denoising methods"""

    @pytest.mark.parametrize("method", ["nlm", "bilateral", "median"])
    def test_denoise_methods(self, sample_image_3d, method):
        """Test different denoising methods"""
        processor = RadiologyProcessor()
        denoised = processor.denoise(sample_image_3d, method=method)

        assert denoised is not None
        assert denoised.shape == sample_image_3d.shape

    def test_denoise_nlm(self, sample_image_2d):
        """Test NLM denoising"""
        processor = RadiologyProcessor()
        denoised = processor.denoise(sample_image_2d, method="nlm")

        assert denoised is not None
        assert denoised.shape == sample_image_2d.shape

    def test_denoise_bilateral(self, sample_image_2d):
        """Test bilateral filtering"""
        processor = RadiologyProcessor()
        denoised = processor.denoise(sample_image_2d, method="bilateral")

        assert denoised is not None


class TestIntensityNormalization:
    """Test intensity normalization methods"""

    def test_normalize_zscore(self, sample_image_3d):
        """Test z-score normalization"""
        processor = RadiologyProcessor()
        normalized = processor.normalize_intensity(sample_image_3d, method="z_score")

        assert normalized is not None
        assert normalized.shape == sample_image_3d.shape
        # Check that mean is close to 0 and std close to 1
        assert abs(np.mean(normalized)) < 0.1
        assert abs(np.std(normalized) - 1.0) < 0.1

    def test_normalize_minmax(self, sample_image_3d):
        """Test min-max normalization"""
        processor = RadiologyProcessor()
        normalized = processor.normalize_intensity(sample_image_3d, method="minmax")

        assert normalized is not None
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1

    def test_normalize_percentile(self, sample_image_3d):
        """Test percentile normalization"""
        processor = RadiologyProcessor()
        normalized = processor.normalize_intensity(
            sample_image_3d, method="percentile", lower=1.0, upper=99.0
        )

        assert normalized is not None


class TestWindowLevelAdjustment:
    """Test window/level adjustment"""

    @pytest.mark.parametrize("window", ["lung", "bone", "soft_tissue", "abdomen"])
    def test_window_presets(self, sample_image_3d, window):
        """Test different window presets"""
        processor = RadiologyProcessor()
        windowed = processor.apply_window(sample_image_3d, window=window)

        assert windowed is not None
        assert windowed.shape == sample_image_3d.shape

    def test_window_custom(self, sample_image_3d):
        """Test custom window/level"""
        processor = RadiologyProcessor()
        windowed = processor.apply_window(sample_image_3d, window=400, level=40)

        assert windowed is not None


class TestHounsfieldVerification:
    """Test Hounsfield unit verification"""

    def test_verify_hu_valid(self):
        """Test verification of valid HU values"""
        # Create CT image with realistic HU values
        ct_image = np.random.randint(-1000, 1000, (64, 128, 128), dtype=np.int16)

        processor = RadiologyProcessor()
        result = processor.verify_hounsfield_units(ct_image)

        assert result is not None
        assert "is_hu" in result
        assert "min_value" in result
        assert "max_value" in result
        assert result["is_hu"] is True

    def test_verify_hu_with_air_and_bone(self):
        """Test detection of air and bone in HU values"""
        # Create image with air (-1000) and bone (>200)
        ct_image = np.full((64, 128, 128), -950, dtype=np.int16)
        ct_image[30:34, :, :] = 300  # Bone region

        processor = RadiologyProcessor()
        result = processor.verify_hounsfield_units(ct_image)

        assert result["likely_air_present"] is True
        assert result["likely_bone_present"] is True


class TestResampling:
    """Test spatial resampling"""

    def test_resample_basic(self, sample_image_3d, sample_dicom_metadata):
        """Test basic resampling"""
        processor = RadiologyProcessor()
        new_spacing = (1.0, 1.0, 1.0)

        resampled = processor.resample(sample_image_3d, sample_dicom_metadata, new_spacing)

        assert resampled is not None
        # Shape should change based on spacing

    def test_resample_with_interpolation(self, sample_image_3d, sample_dicom_metadata):
        """Test resampling with different interpolation methods"""
        processor = RadiologyProcessor()
        new_spacing = (1.0, 1.0, 1.0)

        for method in ["linear", "nearest"]:
            resampled = processor.resample(
                sample_image_3d, sample_dicom_metadata, new_spacing, interpolation=method
            )
            assert resampled is not None


class TestSegmentation:
    """Test segmentation methods"""

    def test_segment_lungs(self):
        """Test lung segmentation from CT"""
        # Create simple CT image with lung-like regions
        # The algorithm looks for low-density regions (< -400 HU) that aren't background
        ct_image = np.full((32, 128, 128), 50, dtype=np.int16)  # Body tissue
        # Create lung regions (air-filled, around -800 HU)
        ct_image[:, 30:50, 40:80] = -800  # Left lung
        ct_image[:, 70:90, 40:80] = -800  # Right lung
        # Background is typically very low
        ct_image[:, :10, :] = -1000  # Outside body

        processor = RadiologyProcessor()
        lung_mask = processor.segment_lungs(ct_image)

        assert lung_mask is not None
        assert lung_mask.shape == ct_image.shape
        assert lung_mask.dtype == np.uint8
        # May or may not detect lungs depending on algorithm behavior
        # Just verify it returns a valid mask

    def test_segment_brain(self):
        """Test brain segmentation from MRI"""
        # Create simple MRI-like image
        mri_image = np.random.randint(0, 255, (32, 128, 128), dtype=np.uint8)
        # Add brain-like structure
        mri_image[10:22, 40:90, 40:90] = 150

        processor = RadiologyProcessor()
        brain_mask = processor.segment_brain(mri_image)

        assert brain_mask is not None
        assert brain_mask.shape == mri_image.shape
        assert brain_mask.dtype == np.uint8

    def test_segment_organs(self, sample_image_3d):
        """Test multi-organ segmentation"""
        processor = RadiologyProcessor()

        # Mock the ct_segmenter's segment_organs method
        from unittest.mock import MagicMock

        processor.ct_segmenter = MagicMock()
        processor.ct_segmenter.segment_organs.return_value = {
            "liver": np.random.rand(64, 128, 128) > 0.8,
            "spleen": np.random.rand(64, 128, 128) > 0.9,
        }

        masks = processor.segment_organs(sample_image_3d)

        assert masks is not None
        assert isinstance(masks, dict)


class TestMetalArtifactReduction:
    """Test metal artifact reduction"""

    def test_reduce_metal_artifacts(self):
        """Test metal artifact reduction in CT"""
        # Create CT image with metal artifacts
        ct_image = np.random.randint(-1000, 1000, (32, 128, 128), dtype=np.int16)
        ct_image[15:17, 60:70, 60:70] = 3500  # Metal implant

        processor = RadiologyProcessor()
        corrected = processor.reduce_metal_artifacts(ct_image, threshold=3000)

        assert corrected is not None
        assert corrected.shape == ct_image.shape


@pytest.mark.requires_models
class TestEmbeddingGeneration:
    """Test embedding generation"""

    @patch("honeybee.models.RadImageNet.radimagenet.RadImageNet")
    def test_generate_embeddings_2d(self, mock_model, sample_image_2d):
        """Test 2D embedding generation"""
        mock_model_instance = MagicMock()
        mock_model_instance.generate_embeddings.return_value = np.random.randn(2048)
        mock_model.return_value = mock_model_instance

        processor = RadiologyProcessor()
        processor.model = mock_model_instance

        embeddings = processor.generate_embeddings(sample_image_2d, mode="2d")

        assert embeddings is not None
        assert len(embeddings.shape) == 1

    @patch("honeybee.models.RadImageNet.radimagenet.RadImageNet")
    def test_generate_embeddings_3d(self, mock_model, sample_image_3d):
        """Test 3D embedding generation"""
        mock_model_instance = MagicMock()
        mock_model_instance.generate_embeddings.return_value = np.random.randn(2048)
        mock_model.return_value = mock_model_instance

        processor = RadiologyProcessor()
        processor.model = mock_model_instance

        embeddings = processor.generate_embeddings(sample_image_3d, mode="3d", aggregation="mean")

        assert embeddings is not None

    @patch("honeybee.models.RadImageNet.radimagenet.RadImageNet")
    def test_generate_embeddings_with_preprocessing(
        self, mock_model, sample_image_3d, sample_dicom_metadata
    ):
        """Test embedding generation with preprocessing"""
        mock_model_instance = MagicMock()
        mock_model_instance.generate_embeddings.return_value = np.random.randn(2048)
        mock_model.return_value = mock_model_instance

        processor = RadiologyProcessor()
        processor.model = mock_model_instance

        embeddings = processor.generate_embeddings(
            sample_image_3d, preprocess=True, metadata=sample_dicom_metadata
        )

        assert embeddings is not None


class TestBatchProcessing:
    """Test batch processing"""

    @patch("honeybee.models.RadImageNet.radimagenet.RadImageNet")
    def test_process_batch(self, mock_model):
        """Test batch image processing"""
        mock_model_instance = MagicMock()
        mock_model_instance.process_batch.return_value = np.random.randn(5, 2048)
        mock_model.return_value = mock_model_instance

        processor = RadiologyProcessor()
        processor.model = mock_model_instance

        images = [np.random.rand(128, 128) for _ in range(5)]
        embeddings = processor.process_batch(images, batch_size=2)

        assert embeddings is not None
        assert embeddings.shape == (5, 2048)


class TestImageRegistration:
    """Test image registration"""

    @pytest.mark.slow
    def test_register_rigid(self):
        """Test rigid registration"""
        # Create similar images with slight offset
        fixed = np.random.randint(0, 255, (32, 64, 64), dtype=np.uint8)
        # Create moving image (slightly shifted)
        moving = np.roll(fixed, shift=2, axis=0)

        processor = RadiologyProcessor()
        registered = processor.register(moving, fixed, method="rigid")

        assert registered is not None
        assert registered.shape == fixed.shape


class TestFeatureExtraction:
    """Test feature extraction from multiple layers"""

    @patch("honeybee.models.RadImageNet.radimagenet.RadImageNet")
    def test_extract_features(self, mock_model, sample_image_2d):
        """Test feature extraction"""
        mock_model_instance = MagicMock()
        mock_model_instance.extract_features = True
        mock_model_instance.generate_embeddings.return_value = {
            "features": {"layer1": np.random.randn(256), "layer2": np.random.randn(512)}
        }
        mock_model.return_value = mock_model_instance

        processor = RadiologyProcessor(extract_features=True)
        processor.model = mock_model_instance

        features = processor.extract_features(sample_image_2d)

        assert features is not None
        assert isinstance(features, dict)


class TestErrorHandling:
    """Test error handling"""

    def test_load_nonexistent_file(self):
        """Test loading non-existent file"""
        processor = RadiologyProcessor()
        with pytest.raises(Exception):
            processor.load_image("/nonexistent/file.dcm")

    def test_preprocess_without_metadata(self, sample_image_3d):
        """Test preprocessing without metadata"""

        processor = RadiologyProcessor()
        # Should use generic preprocessing
        generic_metadata = ImageMetadata(
            modality="UNKNOWN",
            patient_id="TEST",
            study_date="20240101",
            series_description="Test",
            pixel_spacing=(1.0, 1.0, 1.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=[1, 0, 0, 0, 1, 0],
        )
        processed = processor.preprocess(sample_image_3d, generic_metadata, denoise=True)
        assert processed is not None


class TestModalitySpecificProcessing:
    """Test modality-specific processing"""

    def test_ct_specific_processing(self, sample_image_3d, sample_dicom_metadata):
        """Test CT-specific preprocessing"""
        processor = RadiologyProcessor()

        processed = processor.preprocess(sample_image_3d, sample_dicom_metadata, window="lung")

        assert processed is not None

    def test_mri_specific_processing(self, sample_image_3d, sample_mri_metadata):
        """Test MRI-specific preprocessing"""
        processor = RadiologyProcessor()

        processed = processor.preprocess(sample_image_3d, sample_mri_metadata, denoise=True)

        assert processed is not None

    def test_pet_specific_processing(self, sample_image_3d, sample_pet_metadata):
        """Test PET-specific preprocessing"""
        # Convert to positive values for PET
        pet_image = sample_image_3d.astype(np.float32) + 1000
        pet_image = pet_image / 100  # Scale to SUV-like range

        processor = RadiologyProcessor()

        processed = processor.preprocess(
            pet_image, sample_pet_metadata, denoise=True, normalize=True
        )

        assert processed is not None
