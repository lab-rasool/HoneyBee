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
    """Test segmentation methods (via NNUNetSegmenter)"""

    def test_segment_lungs(self):
        """Test lung segmentation delegates to NNUNetSegmenter"""
        processor = RadiologyProcessor()
        processor.segmenter = MagicMock()
        expected = np.ones((32, 128, 128), dtype=np.uint8)
        processor.segmenter.segment_lungs.return_value = expected

        ct_image = np.random.randint(-1000, 1000, (32, 128, 128), dtype=np.int16)
        result = processor.segment_lungs(ct_image)

        processor.segmenter.segment_lungs.assert_called_once()
        assert result.dtype == np.uint8

    def test_segment_brain(self):
        """Test brain segmentation delegates to NNUNetSegmenter"""
        processor = RadiologyProcessor()
        processor.segmenter = MagicMock()
        expected = np.ones((32, 128, 128), dtype=np.uint8)
        processor.segmenter.extract_brain.return_value = expected

        mri_image = np.random.randint(0, 255, (32, 128, 128), dtype=np.uint8)
        result = processor.segment_brain(mri_image)

        processor.segmenter.extract_brain.assert_called_once()
        assert result.dtype == np.uint8

    def test_segment_organs(self, sample_image_3d):
        """Test multi-organ segmentation delegates to NNUNetSegmenter"""
        processor = RadiologyProcessor()
        processor.segmenter = MagicMock()
        processor.segmenter.segment_organs.return_value = {
            "liver": np.random.rand(64, 128, 128) > 0.8,
            "spleen": np.random.rand(64, 128, 128) > 0.9,
        }

        masks = processor.segment_organs(sample_image_3d)

        assert masks is not None
        assert isinstance(masks, dict)
        processor.segmenter.segment_organs.assert_called_once()


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


class TestCropToROI:
    """Test crop_to_roi method"""

    def test_crop_to_roi_3d(self, sample_image_3d):
        """Test ROI cropping with 3D image"""
        processor = RadiologyProcessor()
        mask = np.zeros_like(sample_image_3d, dtype=np.uint8)
        mask[10:30, 40:80, 40:80] = 1

        cropped = processor.crop_to_roi(sample_image_3d, mask)

        assert cropped is not None
        assert cropped.shape == (20, 40, 40)

    def test_crop_to_roi_empty_mask(self, sample_image_3d):
        """Test ROI cropping with empty mask returns original"""
        processor = RadiologyProcessor()
        mask = np.zeros_like(sample_image_3d, dtype=np.uint8)

        cropped = processor.crop_to_roi(sample_image_3d, mask)
        assert cropped.shape == sample_image_3d.shape


class TestApplyMask:
    """Test apply_mask method"""

    def test_apply_mask(self, sample_image_3d):
        """Test masking zeros outside region"""
        processor = RadiologyProcessor()
        mask = np.zeros_like(sample_image_3d, dtype=np.uint8)
        mask[10:30, 40:80, 40:80] = 1

        masked = processor.apply_mask(sample_image_3d, mask)

        assert masked.shape == sample_image_3d.shape
        assert np.all(masked[0:10] == 0)
        assert np.any(masked[10:30, 40:80, 40:80] != 0)


class TestCorrectBiasField:
    """Test correct_bias_field method"""

    @patch("honeybee.processors.radiology.processor.sitk")
    def test_correct_bias_field(self, mock_sitk, sample_image_3d):
        """Test N4 bias field correction delegates to SimpleITK"""
        processor = RadiologyProcessor()

        mock_sitk_image = MagicMock()
        mock_sitk.GetImageFromArray.return_value = mock_sitk_image

        corrected_sitk = MagicMock()
        mock_corrector = MagicMock()
        mock_sitk.N4BiasFieldCorrectionImageFilter.return_value = mock_corrector
        mock_corrector.Execute.return_value = corrected_sitk

        mock_sitk.GetArrayFromImage.return_value = sample_image_3d.astype(np.float32)

        result = processor.correct_bias_field(sample_image_3d)
        assert result is not None
        assert result.shape == sample_image_3d.shape
        mock_corrector.Execute.assert_called_once()


class TestCalculateSUV:
    """Test calculate_suv method"""

    def test_calculate_suv_basic(self):
        """Test SUV calculation formula"""
        processor = RadiologyProcessor()
        image = np.ones((10, 64, 64), dtype=np.float32) * 1000

        suv = processor.calculate_suv(image, patient_weight=70, injected_dose=10)

        assert suv is not None
        assert suv.shape == image.shape
        # SUV = pixel * weight_g / dose_bq
        # = 1000 * 70000 / (10 * 3.7e7) = 1000 * 70000 / 3.7e8
        expected = 1000 * 70000 / (10 * 3.7e7)
        np.testing.assert_allclose(suv[0, 0, 0], expected, rtol=1e-5)


class TestAggregateEmbeddings:
    """Test aggregate_embeddings method"""

    def test_aggregate_mean(self):
        """Test mean aggregation"""
        processor = RadiologyProcessor()
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        result = processor.aggregate_embeddings(embeddings, method="mean")

        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [2.5, 3.5, 4.5])

    def test_aggregate_max(self):
        """Test max aggregation"""
        processor = RadiologyProcessor()
        embeddings = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])

        result = processor.aggregate_embeddings(embeddings, method="max")

        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [4.0, 5.0, 6.0])

    def test_aggregate_concat(self):
        """Test concat aggregation"""
        processor = RadiologyProcessor()
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = processor.aggregate_embeddings(embeddings, method="concat")

        assert result.shape == (4,)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_aggregate_invalid_method(self):
        """Test invalid aggregation method raises error"""
        processor = RadiologyProcessor()
        embeddings = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="Unknown aggregation"):
            processor.aggregate_embeddings(embeddings, method="invalid")


class TestLoadAtlas:
    """Test load_atlas method"""

    def test_load_atlas_delegates(self):
        """Test load_atlas delegates to nifti_loader"""
        processor = RadiologyProcessor()
        processor.nifti_loader = MagicMock()
        mock_result = (np.zeros((10, 10, 10)), MagicMock())
        processor.nifti_loader.load_file.return_value = mock_result

        result = processor.load_atlas("/fake/atlas.nii.gz")

        assert result == mock_result
        processor.nifti_loader.load_file.assert_called_once_with("/fake/atlas.nii.gz")


class TestDenoiseRician:
    """Test Rician denoising"""

    def test_denoise_rician(self, sample_image_2d):
        """Test Rician denoising runs without error"""
        processor = RadiologyProcessor()
        denoised = processor.denoise(sample_image_2d.astype(np.float32), method="rician")

        assert denoised is not None
        assert denoised.shape == sample_image_2d.shape

    def test_denoise_rician_3d(self, sample_image_3d):
        """Test Rician denoising on 3D volume"""
        processor = RadiologyProcessor()
        denoised = processor.denoise(sample_image_3d.astype(np.float32), method="rician")

        assert denoised is not None
        assert denoised.shape == sample_image_3d.shape


class TestHUVerificationExtended:
    """Test extended HU verification fixes"""

    def test_verify_hu_ge_padding_value(self):
        """Test that GE padding value -2048 is recognized as valid HU"""
        ct_image = np.random.randint(-1000, 1000, (32, 64, 64), dtype=np.int16)
        ct_image[0, :, :] = -2048  # GE padding

        processor = RadiologyProcessor()
        result = processor.verify_hounsfield_units(ct_image)

        assert result["is_hu"] is True
        assert result["min_value"] == -2048.0

    def test_verify_hu_extended_bone_range(self):
        """Test that max value up to 4096 is valid HU"""
        ct_image = np.random.randint(-500, 500, (32, 64, 64), dtype=np.int16)
        ct_image[10, 30, 30] = 3500  # Dense bone/metal

        processor = RadiologyProcessor()
        result = processor.verify_hounsfield_units(ct_image)

        assert result["is_hu"] is True
        assert result["likely_bone_present"] is True

    def test_verify_hu_no_false_rescale_warning(self, sample_dicom_metadata):
        """Test that rescale warning doesn't fire for already-rescaled data"""
        ct_image = np.random.randint(-1000, 1000, (32, 64, 64), dtype=np.int16)

        processor = RadiologyProcessor()
        result = processor.verify_hounsfield_units(ct_image, metadata=sample_dicom_metadata)

        # Metadata has rescale_intercept=-1024, slope=1.0 but data is already in HU
        # So no "raw pixel data" warning should appear
        for w in result["warnings"]:
            assert "raw pixel data" not in w

    def test_verify_hu_air_detection_with_padding(self):
        """Test air detection works even when min is -2048 (padding)"""
        ct_image = np.full((32, 64, 64), 0, dtype=np.int16)
        ct_image[0, :, :] = -2048  # Padding (not air)
        # Add actual air voxels
        ct_image[5:10, 10:30, 10:30] = -1000

        processor = RadiologyProcessor()
        result = processor.verify_hounsfield_units(ct_image)

        assert result["likely_air_present"] is True


class TestSegmentationDelegation:
    """Test that segmentation methods delegate to NNUNetSegmenter"""

    def test_segment_lungs_delegates_to_segmenter(self):
        """Test segment_lungs delegates to NNUNetSegmenter"""
        processor = RadiologyProcessor()
        processor.segmenter = MagicMock()
        expected_mask = np.ones((32, 64, 64), dtype=np.uint8)
        processor.segmenter.segment_lungs.return_value = expected_mask

        ct_image = np.random.randint(-1000, 1000, (32, 64, 64), dtype=np.int16)
        result = processor.segment_lungs(ct_image)

        processor.segmenter.segment_lungs.assert_called_once()
        assert result.dtype == np.uint8

    def test_segment_brain_delegates_to_segmenter(self):
        """Test segment_brain delegates to NNUNetSegmenter"""
        processor = RadiologyProcessor()
        processor.segmenter = MagicMock()
        expected_mask = np.ones((32, 64, 64), dtype=np.uint8)
        processor.segmenter.extract_brain.return_value = expected_mask

        mri_image = np.random.randint(0, 255, (32, 64, 64), dtype=np.uint8)
        result = processor.segment_brain(mri_image)

        processor.segmenter.extract_brain.assert_called_once()
        assert result.dtype == np.uint8


class TestPrepareForModel:
    """Test prepare_for_model method"""

    def test_prepare_2d_image(self):
        """Test preparing a 2D image returns RGB uint8"""
        processor = RadiologyProcessor()
        image = np.random.randint(-1000, 1000, (128, 128), dtype=np.int16)

        result = processor.prepare_for_model(image)

        assert len(result) == 1
        assert result[0].dtype == np.uint8
        assert result[0].shape == (128, 128, 3)

    def test_prepare_3d_middle_slice(self):
        """Test preparing a 3D image returns middle slice by default"""
        processor = RadiologyProcessor()
        image = np.random.randint(-1000, 1000, (64, 128, 128), dtype=np.int16)

        result = processor.prepare_for_model(image)

        assert len(result) == 1
        assert result[0].shape == (128, 128, 3)

    def test_prepare_3d_n_slices(self):
        """Test preparing a 3D image with n_slices"""
        processor = RadiologyProcessor()
        image = np.random.randint(-1000, 1000, (64, 128, 128), dtype=np.int16)

        result = processor.prepare_for_model(image, n_slices=5)

        assert len(result) == 5
        for img in result:
            assert img.shape == (128, 128, 3)
            assert img.dtype == np.uint8

    def test_prepare_auto_window_detection(self, sample_dicom_metadata):
        """Test auto-window detection from metadata"""
        processor = RadiologyProcessor()

        # Chest CT should get lung window
        preset = processor._detect_window_preset(sample_dicom_metadata)
        assert preset == "lung"  # series_description = "Chest CT with contrast"

    def test_prepare_abdomen_window(self):
        """Test abdomen window detection"""
        from honeybee.loaders.Radiology.metadata import ImageMetadata

        metadata = ImageMetadata(
            modality="CT",
            patient_id="TEST",
            study_date="20240101",
            series_description="ABD/PELVIS CT",
            pixel_spacing=(1.0, 1.0, 1.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=[1, 0, 0, 0, 1, 0],
        )
        preset = RadiologyProcessor._detect_window_preset(metadata)
        assert preset == "abdomen"

    def test_prepare_with_explicit_window(self):
        """Test that explicit window overrides auto-detection"""
        processor = RadiologyProcessor()
        image = np.random.randint(-1000, 1000, (128, 128), dtype=np.int16)

        result = processor.prepare_for_model(image, window="bone")

        assert len(result) == 1
        assert result[0].dtype == np.uint8


class TestRegistryModelInit:
    """Test registry-based model initialization"""

    @patch("honeybee.models.registry.load_model")
    def test_init_registry_model(self, mock_load):
        """Test initializing a model from the registry"""
        mock_model = MagicMock()
        mock_model.embedding_dim = 512
        mock_load.return_value = mock_model

        processor = RadiologyProcessor(model="biomedclip")

        assert processor._registry_model is True
        mock_load.assert_called_once()

    def test_init_unknown_model_raises(self):
        """Test that truly unknown model raises ValueError"""
        with pytest.raises(ValueError, match="Unknown model"):
            RadiologyProcessor(model="completely_nonexistent_model_xyz")


class TestCorrectBiasFieldBackend:
    """Test correct_bias_field with backend parameter"""

    @patch("honeybee.processors.radiology.processor.sitk")
    def test_correct_bias_field_sitk_backend(self, mock_sitk, sample_image_3d):
        """Test N4 bias field correction with SimpleITK backend"""
        processor = RadiologyProcessor()

        mock_sitk_image = MagicMock()
        mock_sitk.GetImageFromArray.return_value = mock_sitk_image
        corrected_sitk = MagicMock()
        mock_corrector = MagicMock()
        mock_sitk.N4BiasFieldCorrectionImageFilter.return_value = mock_corrector
        mock_corrector.Execute.return_value = corrected_sitk
        mock_sitk.GetArrayFromImage.return_value = sample_image_3d.astype(np.float32)

        result = processor.correct_bias_field(sample_image_3d, backend="sitk")
        assert result is not None
        mock_corrector.Execute.assert_called_once()


class TestNNUNetSegmenter:
    """Test NNUNetSegmenter class"""

    def test_init_default(self):
        """Test NNUNetSegmenter with no model paths has empty tasks"""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        seg = NNUNetSegmenter()
        assert seg.available_tasks == []

    def test_init_with_model_paths(self, tmp_path):
        """Test NNUNetSegmenter with model paths registers tasks"""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        model_dir = tmp_path / "lung_model"
        model_dir.mkdir()
        seg = NNUNetSegmenter(model_paths={"lung": str(model_dir)})
        assert "lung" in seg.available_tasks

    def test_missing_task_raises(self):
        """Test that predict_raw raises ValueError for unconfigured task"""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        seg = NNUNetSegmenter()
        image = np.zeros((10, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="No model path configured"):
            seg.predict_raw(image, (1.0, 1.0, 1.0), task="nonexistent")

    def test_missing_model_dir_raises(self, tmp_path):
        """Test that predict_raw raises FileNotFoundError for bad path"""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        seg = NNUNetSegmenter(model_paths={"lung": str(tmp_path / "does_not_exist")})
        image = np.zeros((10, 64, 64), dtype=np.float32)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            seg.predict_raw(image, (1.0, 1.0, 1.0), task="lung")

    def test_set_model_path(self, tmp_path):
        """Test dynamic model path configuration"""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        seg = NNUNetSegmenter()
        assert "lung" not in seg.available_tasks

        model_dir = tmp_path / "lung_model"
        model_dir.mkdir()
        seg.set_model_path("lung", str(model_dir))
        assert "lung" in seg.available_tasks

    def test_set_label_map(self):
        """Test dynamic label map configuration"""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        seg = NNUNetSegmenter()
        custom_map = {1: "my_organ", 2: "other_organ"}
        seg.set_label_map("custom_task", custom_map)
        assert seg.get_label_map("custom_task") == custom_map

    @patch("honeybee.processors.radiology.segmentation.nnUNetPredictor", create=True)
    def test_segment_lungs_calls_predictor(self, mock_pred_cls, tmp_path):
        """Test that segment_lungs invokes the nnU-Net predictor"""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        model_dir = tmp_path / "lung_model"
        model_dir.mkdir()

        mock_predictor = MagicMock()
        mock_predictor.predict_single_npy_array.return_value = np.ones(
            (10, 64, 64), dtype=np.int32
        )

        seg = NNUNetSegmenter(model_paths={"lung": str(model_dir)})
        # Inject mock predictor directly
        seg._predictors["lung"] = mock_predictor

        image = np.zeros((10, 64, 64), dtype=np.float32)
        result = seg.segment_lungs(image, spacing=(1.0, 1.0, 1.0))

        mock_predictor.predict_single_npy_array.assert_called_once()
        assert result.dtype == np.uint8
        assert result.shape == (10, 64, 64)
        assert np.all(result == 1)

    def test_segment_organs_splits_labels(self, tmp_path):
        """Test that segment_organs splits integer map into per-organ masks"""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        model_dir = tmp_path / "organ_model"
        model_dir.mkdir()

        seg = NNUNetSegmenter(model_paths={"total_organs": str(model_dir)})

        # Create a mock predictor returning a map with labels 1 and 5
        mock_predictor = MagicMock()
        seg_map = np.zeros((10, 64, 64), dtype=np.int32)
        seg_map[0:3, 10:20, 10:20] = 1  # spleen
        seg_map[5:8, 30:40, 30:40] = 5  # liver
        mock_predictor.predict_single_npy_array.return_value = seg_map
        seg._predictors["total_organs"] = mock_predictor

        image = np.zeros((10, 64, 64), dtype=np.float32)
        result = seg.segment_organs(image, spacing=(1.0, 1.0, 1.0))

        assert isinstance(result, dict)
        assert "spleen" in result
        assert "liver" in result
        assert result["spleen"].sum() > 0
        assert result["liver"].sum() > 0

    def test_predictor_caching(self, tmp_path):
        """Test that same predictor instance is reused on second call"""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        model_dir = tmp_path / "lung_model"
        model_dir.mkdir()

        seg = NNUNetSegmenter(model_paths={"lung": str(model_dir)})

        mock_predictor = MagicMock()
        mock_predictor.predict_single_npy_array.return_value = np.zeros(
            (10, 64, 64), dtype=np.int32
        )
        seg._predictors["lung"] = mock_predictor

        image = np.zeros((10, 64, 64), dtype=np.float32)
        seg.segment_lungs(image)
        seg.segment_lungs(image)

        # Should use the same cached predictor (2 calls to predict)
        assert mock_predictor.predict_single_npy_array.call_count == 2


class TestDetectNodules:
    """Test standalone detect_nodules function"""

    def test_requires_lung_mask(self):
        """Test that None lung_mask raises ValueError"""
        from honeybee.processors.radiology.segmentation import detect_nodules

        image = np.zeros((10, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="lung_mask is required"):
            detect_nodules(image, lung_mask=None)

    def test_returns_list(self):
        """Test basic execution with synthetic data"""
        from honeybee.processors.radiology.segmentation import detect_nodules

        image = np.random.randn(64, 64).astype(np.float32) * 100
        mask = np.ones((64, 64), dtype=bool)

        result = detect_nodules(image, lung_mask=mask)
        assert isinstance(result, list)
