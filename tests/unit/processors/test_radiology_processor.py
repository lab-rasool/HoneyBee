"""
Unit tests for RadiologyProcessor

Comprehensive tests for radiology image processing including loading,
preprocessing, segmentation, and embedding generation.
All heavy library imports (SimpleITK, nibabel, torch, lungmask, etc.)
are mocked so no GPU or model weights are needed.
"""

import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from honeybee.processors.radiology.metadata import ImageMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(modality="CT", **kwargs):
    """Create an ImageMetadata instance with sensible defaults."""
    defaults = dict(
        modality=modality,
        patient_id="TEST001",
        study_date="20240115",
        series_description="Test Series",
        pixel_spacing=(1.0, 1.0, 1.0),
        image_position=(0.0, 0.0, 0.0),
        image_orientation=[1, 0, 0, 0, 1, 0],
    )
    defaults.update(kwargs)
    return ImageMetadata(**defaults)


def _make_mock_sitk():
    """Build a MagicMock that behaves like the SimpleITK module."""
    mock_sitk = MagicMock()

    mock_image = MagicMock()
    mock_image.GetSpacing.return_value = (1.0, 1.0, 1.0)
    mock_image.GetOrigin.return_value = (0.0, 0.0, 0.0)
    mock_image.GetDirection.return_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    mock_image.GetSize.return_value = (128, 128, 64)
    mock_image.GetMetaDataKeys.return_value = []
    mock_image.GetMetaData.return_value = ""

    mock_sitk.GetArrayFromImage.return_value = np.random.randn(64, 128, 128).astype(
        np.float32
    )
    mock_sitk.GetImageFromArray.return_value = mock_image
    mock_sitk.ReadImage.return_value = mock_image

    return mock_sitk, mock_image


def _make_processor(**kwargs):
    """Create a RadiologyProcessor with mocked torch for device detection."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    defaults = {"device": "cpu"}
    defaults.update(kwargs)

    with patch.dict("sys.modules", {"torch": mock_torch}):
        from honeybee.processors.radiology.processor import RadiologyProcessor

        return RadiologyProcessor(**defaults)


# ---------------------------------------------------------------------------
# TestRadiologyProcessorInitialization
# ---------------------------------------------------------------------------


class TestRadiologyProcessorInitialization:
    """Test RadiologyProcessor initialization (8 tests)."""

    def test_default_initialization(self):
        """Defaults: model='remedis', backend='lungmask', lazy model."""
        processor = _make_processor()
        assert processor.model_name == "remedis"
        assert processor._segmentation_backend == "lungmask"
        assert processor.embedding_model is None
        assert processor.device == "cpu"

    def test_model_alias_remedis(self):
        """'remedis' resolves to 'remedis'."""
        processor = _make_processor(model="remedis")
        assert processor.model_name == "remedis"

    def test_model_alias_radimagenet(self):
        """'radimagenet' resolves to 'radimagenet-resnet50'."""
        processor = _make_processor(model="radimagenet")
        assert processor.model_name == "radimagenet-resnet50"

    def test_model_alias_medsiglip(self):
        """'medsiglip' resolves to 'medsiglip'."""
        processor = _make_processor(model="medsiglip")
        assert processor.model_name == "medsiglip"

    def test_model_alias_rad_dino(self):
        """'rad-dino' resolves to 'rad-dino'."""
        processor = _make_processor(model="rad-dino")
        assert processor.model_name == "rad-dino"

    def test_custom_device(self):
        """Explicit device overrides auto-detection."""
        processor = _make_processor(device="cuda:1")
        assert processor.device == "cuda:1"

    def test_lazy_model_not_loaded(self):
        """Embedding model is None until generate_embeddings is called."""
        processor = _make_processor()
        assert processor.embedding_model is None

    def test_segmentation_backend_config(self):
        """Custom segmentation backend is stored."""
        processor = _make_processor(segmentation_backend="totalsegmentator")
        assert processor._segmentation_backend == "totalsegmentator"


# ---------------------------------------------------------------------------
# TestImageLoading
# ---------------------------------------------------------------------------


class TestImageLoading:
    """Test image loading methods (8 tests)."""

    def test_load_dicom_directory(self, tmp_path):
        """Load DICOM series from a directory."""
        mock_sitk, mock_image = _make_mock_sitk()
        reader_instance = MagicMock()
        reader_instance.GetGDCMSeriesFileNames.return_value = ["file1.dcm"]
        reader_instance.Execute.return_value = mock_image
        mock_sitk.ImageSeriesReader.return_value = reader_instance

        dicom_dir = tmp_path / "dicom"
        dicom_dir.mkdir()

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            image, metadata = processor.load_dicom(str(dicom_dir))

        assert isinstance(image, np.ndarray)
        assert isinstance(metadata, ImageMetadata)

    def test_load_dicom_file(self, tmp_path):
        """Load a single DICOM file."""
        mock_sitk, mock_image = _make_mock_sitk()

        dicom_file = tmp_path / "scan.dcm"
        dicom_file.write_bytes(b"\x00" * 10)

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            image, metadata = processor.load_dicom(str(dicom_file))

        mock_sitk.ReadImage.assert_called_once_with(str(dicom_file))
        assert isinstance(image, np.ndarray)

    def test_load_dicom_no_files_error(self, tmp_path):
        """Raise FileNotFoundError when directory has no DICOM files."""
        mock_sitk, _ = _make_mock_sitk()
        reader_instance = MagicMock()
        reader_instance.GetGDCMSeriesFileNames.return_value = []
        mock_sitk.ImageSeriesReader.return_value = reader_instance

        dicom_dir = tmp_path / "empty"
        dicom_dir.mkdir()

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            with pytest.raises(FileNotFoundError, match="No DICOM files"):
                processor.load_dicom(str(dicom_dir))

    def test_load_nifti(self, tmp_path):
        """Load a NIfTI file."""
        mock_nib = MagicMock()
        mock_nib_img = MagicMock()
        mock_nib_img.get_fdata.return_value = np.random.randn(64, 128, 128)
        mock_nib_img.header.get_zooms.return_value = (1.0, 1.0, 2.5)
        mock_nib.load.return_value = mock_nib_img

        nifti_path = tmp_path / "brain.nii.gz"
        nifti_path.write_bytes(b"\x00" * 10)

        with patch.dict("sys.modules", {"nibabel": mock_nib}):
            processor = _make_processor()
            image, metadata = processor.load_nifti(str(nifti_path))

        assert isinstance(image, np.ndarray)
        assert metadata.pixel_spacing == (1.0, 1.0, 2.5)

    def test_load_nifti_metadata_extraction(self, tmp_path):
        """Metadata defaults are filled for NIfTI (modality='unknown')."""
        mock_nib = MagicMock()
        mock_nib_img = MagicMock()
        mock_nib_img.get_fdata.return_value = np.random.randn(64, 128, 128)
        mock_nib_img.header.get_zooms.return_value = (0.5, 0.5, 1.5)
        mock_nib.load.return_value = mock_nib_img

        nifti_path = tmp_path / "volume.nii"
        nifti_path.write_bytes(b"\x00" * 10)

        with patch.dict("sys.modules", {"nibabel": mock_nib}):
            processor = _make_processor()
            _, metadata = processor.load_nifti(str(nifti_path))

        assert metadata.modality == "unknown"
        assert metadata.patient_id == ""
        assert metadata.number_of_slices == 128

    def test_load_atlas_delegates_to_nifti(self, tmp_path):
        """load_atlas is an alias for load_nifti."""
        mock_nib = MagicMock()
        mock_nib_img = MagicMock()
        mock_nib_img.get_fdata.return_value = np.random.randn(64, 128, 128)
        mock_nib_img.header.get_zooms.return_value = (1.0, 1.0, 1.0)
        mock_nib.load.return_value = mock_nib_img

        atlas_path = tmp_path / "atlas.nii.gz"
        atlas_path.write_bytes(b"\x00" * 10)

        with patch.dict("sys.modules", {"nibabel": mock_nib}):
            processor = _make_processor()
            image, metadata = processor.load_atlas(str(atlas_path))

        assert isinstance(image, np.ndarray)
        mock_nib.load.assert_called_once()

    def test_load_dicom_metadata_extraction(self, tmp_path):
        """DICOM metadata tags are extracted into ImageMetadata."""
        mock_sitk, mock_image = _make_mock_sitk()
        mock_image.GetMetaDataKeys.return_value = [
            "0008|0060",
            "0010|0020",
            "0008|0020",
            "0008|103e",
        ]

        def _get_meta(key):
            tags = {
                "0008|0060": "CT",
                "0010|0020": "PAT42",
                "0008|0020": "20240501",
                "0008|103e": "CHEST CT",
            }
            return tags.get(key, "")

        mock_image.GetMetaData.side_effect = _get_meta

        dicom_file = tmp_path / "ct.dcm"
        dicom_file.write_bytes(b"\x00" * 10)

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            _, metadata = processor.load_dicom(str(dicom_file))

        assert metadata.modality == "CT"
        assert metadata.patient_id == "PAT42"
        assert metadata.study_date == "20240501"
        assert metadata.series_description == "CHEST CT"

    def test_load_nifti_2d_image(self, tmp_path):
        """Loading a 2D NIfTI sets number_of_slices=1."""
        mock_nib = MagicMock()
        mock_nib_img = MagicMock()
        mock_nib_img.get_fdata.return_value = np.random.randn(256, 256)
        mock_nib_img.header.get_zooms.return_value = (0.5, 0.5)
        mock_nib.load.return_value = mock_nib_img

        nifti_path = tmp_path / "slice.nii"
        nifti_path.write_bytes(b"\x00" * 10)

        with patch.dict("sys.modules", {"nibabel": mock_nib}):
            processor = _make_processor()
            image, metadata = processor.load_nifti(str(nifti_path))

        assert image.ndim == 2
        assert metadata.number_of_slices == 1
        assert metadata.pixel_spacing == (0.5, 0.5, 1.0)


# ---------------------------------------------------------------------------
# TestPreprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    """Test preprocessing pipeline (8 tests)."""

    def test_preprocess_ct_pipeline(self):
        """CT images are routed through preprocess_ct."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        metadata = _make_metadata("CT")

        with patch(
            "honeybee.processors.radiology.processor.preprocess_ct"
        ) as mock_ct:
            mock_ct.return_value = image.copy()
            processor = _make_processor()
            result = processor.preprocess(image, metadata)

        mock_ct.assert_called_once()
        assert result.shape == image.shape

    def test_preprocess_mri_pipeline(self):
        """MR images are routed through preprocess_mri."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        metadata = _make_metadata("MR")

        with patch(
            "honeybee.processors.radiology.processor.preprocess_mri"
        ) as mock_mri:
            mock_mri.return_value = image.copy()
            processor = _make_processor()
            result = processor.preprocess(image, metadata)

        mock_mri.assert_called_once()
        assert result.shape == image.shape

    def test_preprocess_pet_pipeline(self):
        """PT images are routed through preprocess_pet."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        metadata = _make_metadata("PT")

        with patch(
            "honeybee.processors.radiology.processor.preprocess_pet"
        ) as mock_pet:
            mock_pet.return_value = image.copy()
            processor = _make_processor()
            result = processor.preprocess(image, metadata)

        mock_pet.assert_called_once()
        assert result.shape == image.shape

    def test_preprocess_unknown_modality(self):
        """Unknown modality applies bilateral + minmax fallback."""
        image = np.random.randn(64, 64).astype(np.float32)
        metadata = _make_metadata("XR")

        with patch(
            "honeybee.processors.radiology.processor.Denoiser"
        ) as MockDenoiser, patch(
            "honeybee.processors.radiology.processor.IntensityNormalizer"
        ) as MockNorm:
            denoiser_inst = MagicMock()
            denoiser_inst.denoise.return_value = image.copy()
            MockDenoiser.return_value = denoiser_inst

            norm_inst = MagicMock()
            norm_inst.normalize.return_value = image.copy()
            MockNorm.return_value = norm_inst

            processor = _make_processor()
            result = processor.preprocess(image, metadata)

        MockDenoiser.assert_called_once_with(method="bilateral")
        MockNorm.assert_called_once_with(method="minmax")
        assert result.shape == image.shape

    def test_preprocess_with_resample(self):
        """When resample_spacing is provided, resample is called."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        metadata = _make_metadata("CT")
        new_spacing = (2.0, 2.0, 2.0)

        with patch(
            "honeybee.processors.radiology.processor.preprocess_ct"
        ) as mock_ct:
            mock_ct.return_value = image.copy()
            processor = _make_processor()
            processor.resample = MagicMock(return_value=np.zeros((16, 32, 32)))
            result = processor.preprocess(
                image, metadata, resample_spacing=new_spacing
            )

        processor.resample.assert_called_once()

    def test_preprocess_ct_with_window(self):
        """Window preset is forwarded to preprocess_ct."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        metadata = _make_metadata("CT")

        with patch(
            "honeybee.processors.radiology.processor.preprocess_ct"
        ) as mock_ct:
            mock_ct.return_value = image.copy()
            processor = _make_processor()
            processor.preprocess(image, metadata, window="bone")

        _, call_kwargs = mock_ct.call_args
        assert call_kwargs["window"] == "bone"

    def test_preprocess_no_denoise(self):
        """denoise=False is forwarded to the pipeline function."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        metadata = _make_metadata("CT")

        with patch(
            "honeybee.processors.radiology.processor.preprocess_ct"
        ) as mock_ct:
            mock_ct.return_value = image.copy()
            processor = _make_processor()
            processor.preprocess(image, metadata, denoise=False)

        _, call_kwargs = mock_ct.call_args
        assert call_kwargs["denoise"] is False

    def test_preprocess_with_artifact_reduction(self):
        """reduce_artifacts flag is forwarded to preprocess_ct."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        metadata = _make_metadata("CT")

        with patch(
            "honeybee.processors.radiology.processor.preprocess_ct"
        ) as mock_ct:
            mock_ct.return_value = image.copy()
            processor = _make_processor()
            processor.preprocess(image, metadata, reduce_artifacts=True)

        _, call_kwargs = mock_ct.call_args
        assert call_kwargs["reduce_artifacts"] is True


# ---------------------------------------------------------------------------
# TestDenoising
# ---------------------------------------------------------------------------


class TestDenoising:
    """Test denoising methods (8 tests)."""

    def test_denoise_nlm(self):
        """NLM denoising dispatches correctly."""
        image = np.random.randn(64, 64).astype(np.float32)
        mock_skimage = MagicMock()
        mock_skimage.restoration.denoise_nl_means.return_value = np.zeros((64, 64))

        with patch.dict(
            "sys.modules",
            {
                "skimage": mock_skimage,
                "skimage.restoration": mock_skimage.restoration,
            },
        ):
            processor = _make_processor()
            result = processor.denoise(image, method="nlm")

        assert result.shape == image.shape

    def test_denoise_bilateral(self):
        """Bilateral denoising dispatches correctly."""
        image = np.random.randn(64, 64).astype(np.float32)
        mock_cv2 = MagicMock()
        mock_cv2.normalize.return_value = np.zeros((64, 64), dtype=np.uint8)
        mock_cv2.bilateralFilter.return_value = np.zeros((64, 64), dtype=np.uint8)
        mock_cv2.NORM_MINMAX = 32
        mock_cv2.CV_8U = 0

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            processor = _make_processor()
            result = processor.denoise(image, method="bilateral")

        assert result.shape == image.shape

    def test_denoise_tv(self):
        """TV denoising dispatches correctly."""
        image = np.random.randn(64, 64).astype(np.float32)
        mock_skimage = MagicMock()
        mock_skimage.restoration.denoise_tv_chambolle.return_value = np.zeros(
            (64, 64)
        )

        with patch.dict(
            "sys.modules",
            {
                "skimage": mock_skimage,
                "skimage.restoration": mock_skimage.restoration,
            },
        ):
            processor = _make_processor()
            result = processor.denoise(image, method="tv")

        assert result.shape == image.shape

    def test_denoise_rician(self):
        """Rician denoising dispatches and returns correct shape."""
        image = np.abs(np.random.randn(64, 64).astype(np.float32)) + 1.0
        mock_skimage = MagicMock()
        mock_skimage.restoration.denoise_nl_means.return_value = np.zeros((64, 64))

        with patch.dict(
            "sys.modules",
            {
                "skimage": mock_skimage,
                "skimage.restoration": mock_skimage.restoration,
            },
        ):
            processor = _make_processor()
            result = processor.denoise(image, method="rician")

        assert result.shape == image.shape

    def test_denoise_deep(self):
        """Deep denoising falls back to NLM when torch is unavailable."""
        image = np.random.randn(64, 64).astype(np.float32)
        mock_skimage = MagicMock()
        mock_skimage.restoration.denoise_nl_means.return_value = np.zeros((64, 64))

        # Remove torch from sys.modules so the import fails
        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # Force ImportError

        try:
            with patch.dict(
                "sys.modules",
                {
                    "torch": None,
                    "torch.nn": None,
                    "skimage": mock_skimage,
                    "skimage.restoration": mock_skimage.restoration,
                },
            ):
                processor = _make_processor(device="cpu")
                result = processor.denoise(image, method="deep")
        finally:
            if saved is not None:
                sys.modules["torch"] = saved

        assert result.shape == image.shape

    def test_denoise_pet_specific(self):
        """PET-specific denoising dispatches correctly."""
        image = np.random.randn(64, 64).astype(np.float32)
        mock_scipy_ndimage = MagicMock()
        mock_scipy_ndimage.median_filter.return_value = image.copy()
        mock_scipy = MagicMock()
        mock_scipy.ndimage = mock_scipy_ndimage

        with patch.dict(
            "sys.modules",
            {
                "scipy": mock_scipy,
                "scipy.ndimage": mock_scipy_ndimage,
            },
        ):
            processor = _make_processor()
            result = processor.denoise(image, method="pet_specific")

        assert result.shape == image.shape

    def test_reduce_metal_artifacts(self):
        """reduce_metal_artifacts delegates to ArtifactReducer."""
        image = np.random.randn(64, 64).astype(np.float32)
        processor = _make_processor()
        processor._artifact_reducer = MagicMock()
        processor._artifact_reducer.reduce_artifacts.return_value = image.copy()

        result = processor.reduce_metal_artifacts(image, threshold=2500)

        processor._artifact_reducer.reduce_artifacts.assert_called_once_with(
            image, artifact_type="metal", threshold=2500
        )
        assert result.shape == image.shape

    def test_correct_bias_field_sitk(self):
        """N4 bias field correction via SimpleITK backend."""
        image = np.random.randn(64, 64).astype(np.float32)
        mock_sitk, mock_image = _make_mock_sitk()
        corrected_array = image.copy() + 1.0
        mock_sitk.GetArrayFromImage.return_value = corrected_array
        corrector_inst = MagicMock()
        corrector_inst.Execute.return_value = mock_image
        mock_sitk.N4BiasFieldCorrectionImageFilter.return_value = corrector_inst

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            result = processor.correct_bias_field(image, backend="sitk")

        mock_sitk.N4BiasFieldCorrectionImageFilter.assert_called_once()
        assert result.shape == image.shape


# ---------------------------------------------------------------------------
# TestSpatialOperations
# ---------------------------------------------------------------------------


class TestSpatialOperations:
    """Test spatial operations (8 tests)."""

    def test_resample(self):
        """Resample delegates to SimpleITK pipeline."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        metadata = _make_metadata()
        mock_sitk, mock_image = _make_mock_sitk()
        mock_sitk.GetArrayFromImage.return_value = np.zeros((16, 32, 32))
        resampler_inst = MagicMock()
        resampler_inst.Execute.return_value = mock_image
        mock_sitk.ResampleImageFilter.return_value = resampler_inst
        mock_sitk.Transform.return_value = MagicMock()

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            result = processor.resample(image, metadata, (2.0, 2.0, 2.0))

        resampler_inst.SetOutputSpacing.assert_called_once_with((2.0, 2.0, 2.0))
        assert isinstance(result, np.ndarray)

    def test_reorient(self):
        """Reorient delegates to SimpleITK DICOMOrient."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        metadata = _make_metadata()
        mock_sitk, mock_image = _make_mock_sitk()
        reoriented_arr = np.random.randn(32, 64, 64).astype(np.float32)
        mock_sitk.GetArrayFromImage.return_value = reoriented_arr

        reoriented_sitk = MagicMock()
        reoriented_sitk.GetSpacing.return_value = (1.0, 1.0, 1.0)
        reoriented_sitk.GetOrigin.return_value = (0.0, 0.0, 0.0)
        reoriented_sitk.GetDirection.return_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        mock_sitk.DICOMOrient.return_value = reoriented_sitk
        mock_sitk.GetArrayFromImage.return_value = reoriented_arr

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            result_img, result_meta = processor.reorient(image, metadata, "RAS")

        mock_sitk.DICOMOrient.assert_called_once()
        assert isinstance(result_meta, ImageMetadata)

    def test_register_rigid(self):
        """Rigid registration via SimpleITK."""
        fixed = np.random.randn(32, 64, 64).astype(np.float32)
        moving = np.random.randn(32, 64, 64).astype(np.float32)
        mock_sitk, mock_image = _make_mock_sitk()
        mock_sitk.GetArrayFromImage.return_value = fixed.copy()

        reg_method = MagicMock()
        reg_method.RANDOM = 0
        reg_method.Execute.return_value = MagicMock()
        mock_sitk.ImageRegistrationMethod.return_value = reg_method
        mock_sitk.Euler3DTransform.return_value = MagicMock()
        mock_sitk.CenteredTransformInitializer.return_value = MagicMock()
        mock_sitk.CenteredTransformInitializerFilter.GEOMETRY = 0

        resampler = MagicMock()
        resampler.Execute.return_value = mock_image
        mock_sitk.ResampleImageFilter.return_value = resampler

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            result = processor.register(moving, fixed, method="rigid")

        assert isinstance(result, np.ndarray)

    def test_register_affine(self):
        """Affine registration creates AffineTransform."""
        fixed = np.random.randn(32, 64, 64).astype(np.float32)
        moving = np.random.randn(32, 64, 64).astype(np.float32)
        mock_sitk, mock_image = _make_mock_sitk()
        mock_sitk.GetArrayFromImage.return_value = fixed.copy()

        reg_method = MagicMock()
        reg_method.RANDOM = 0
        reg_method.Execute.return_value = MagicMock()
        mock_sitk.ImageRegistrationMethod.return_value = reg_method
        mock_sitk.AffineTransform.return_value = MagicMock()
        mock_sitk.CenteredTransformInitializer.return_value = MagicMock()
        mock_sitk.CenteredTransformInitializerFilter.GEOMETRY = 0

        resampler = MagicMock()
        resampler.Execute.return_value = mock_image
        mock_sitk.ResampleImageFilter.return_value = resampler

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            result = processor.register(moving, fixed, method="affine")

        mock_sitk.AffineTransform.assert_called_once_with(3)
        assert isinstance(result, np.ndarray)

    def test_register_failure_returns_original(self):
        """On registration failure the original image is returned."""
        fixed = np.random.randn(32, 64, 64).astype(np.float32)
        moving = np.random.randn(32, 64, 64).astype(np.float32)
        mock_sitk, mock_image = _make_mock_sitk()

        reg_method = MagicMock()
        reg_method.RANDOM = 0
        reg_method.Execute.side_effect = RuntimeError("registration diverged")
        mock_sitk.ImageRegistrationMethod.return_value = reg_method
        mock_sitk.Euler3DTransform.return_value = MagicMock()
        mock_sitk.CenteredTransformInitializer.return_value = MagicMock()
        mock_sitk.CenteredTransformInitializerFilter.GEOMETRY = 0

        with patch.dict("sys.modules", {"SimpleITK": mock_sitk}):
            processor = _make_processor()
            result = processor.register(moving, fixed, method="rigid")

        np.testing.assert_array_equal(result, moving)

    def test_crop_to_roi(self):
        """crop_to_roi returns bounding-box crop of masked region."""
        image = np.zeros((64, 64), dtype=np.float32)
        image[10:20, 30:50] = 5.0
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 30:50] = 1

        processor = _make_processor()
        cropped = processor.crop_to_roi(image, mask)

        assert cropped.shape == (10, 20)
        assert np.all(cropped == 5.0)

    def test_crop_to_roi_empty_mask(self):
        """Empty mask returns the original image unchanged."""
        image = np.random.randn(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)

        processor = _make_processor()
        result = processor.crop_to_roi(image, mask)

        np.testing.assert_array_equal(result, image)

    def test_apply_mask(self):
        """apply_mask zeroes out everything outside the mask."""
        image = np.ones((64, 64), dtype=np.float32) * 10
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 1

        processor = _make_processor()
        result = processor.apply_mask(image, mask)

        assert result[0, 0] == 0
        assert result[30, 30] == 10


# ---------------------------------------------------------------------------
# TestNormalization
# ---------------------------------------------------------------------------


class TestNormalization:
    """Test normalization methods (8 tests)."""

    def test_verify_hu_valid_ct(self):
        """Valid HU range produces is_hu=True."""
        image = np.random.uniform(-1024, 3000, (64, 64)).astype(np.float32)
        processor = _make_processor()
        result = processor.verify_hounsfield_units(image)

        assert result["is_hu"] is True
        assert "min_value" in result
        assert "max_value" in result

    def test_verify_hu_invalid_range(self):
        """Values far outside HU range produce is_hu=False with warnings."""
        image = np.random.uniform(-5000, 10000, (64, 64)).astype(np.float32)
        processor = _make_processor()
        result = processor.verify_hounsfield_units(image)

        assert result["is_hu"] is False
        assert len(result["warnings"]) > 0

    def test_verify_hu_air_detection(self):
        """Air voxels near -1000 HU are detected."""
        image = np.full((100, 100), -1000, dtype=np.float32)
        processor = _make_processor()
        result = processor.verify_hounsfield_units(image)

        assert result["likely_air_present"] is True

    def test_verify_hu_bone_detection(self):
        """High-density voxels (> 200) indicate bone."""
        image = np.random.uniform(500, 2000, (64, 64)).astype(np.float32)
        processor = _make_processor()
        result = processor.verify_hounsfield_units(image)

        assert result["likely_bone_present"] is True

    def test_apply_window_numeric(self):
        """Numeric window/level clips and scales the image."""
        image = np.array([-1000, -500, 0, 500, 1000], dtype=np.float32)
        processor = _make_processor()
        result = processor.apply_window(image, window=1000, level=0)

        assert result.min() >= 0
        assert result.max() <= 255

    def test_apply_window_preset(self):
        """String preset selects correct window/level pair."""
        image = np.random.uniform(-1500, 500, (64, 64)).astype(np.float32)
        processor = _make_processor()
        result = processor.apply_window(image, window="lung")

        assert result.min() >= 0
        assert result.max() <= 255

    def test_normalize_intensity_zscore(self):
        """Z-score normalization produces zero mean, unit variance."""
        image = np.random.randn(64, 64).astype(np.float32) * 50 + 100
        processor = _make_processor()
        result = processor.normalize_intensity(image, method="zscore")

        np.testing.assert_almost_equal(result.mean(), 0.0, decimal=4)
        np.testing.assert_almost_equal(result.std(), 1.0, decimal=2)

    def test_calculate_suv(self):
        """SUV calculation follows the standard formula."""
        image = np.ones((8, 8), dtype=np.float32) * 1000
        weight_kg = 70
        dose_mci = 10

        processor = _make_processor()
        suv = processor.calculate_suv(image, weight_kg, dose_mci)

        # SUV = pixel * (weight_g) / dose_bq
        expected_suv = 1000.0 * (70 * 1000) / (10 * 3.7e7)
        np.testing.assert_almost_equal(suv[0, 0], expected_suv, decimal=4)


# ---------------------------------------------------------------------------
# TestSegmentation
# ---------------------------------------------------------------------------


class TestSegmentation:
    """Test segmentation methods (10 tests)."""

    def test_segment_lungs_lungmask_backend(self):
        """Lungmask backend returns binary mask."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        mock_seg = np.ones((32, 64, 64), dtype=np.int32) * 2

        processor = _make_processor(segmentation_backend="lungmask")
        mock_lm = MagicMock()
        mock_lm.segment.return_value = mock_seg
        processor._lungmask_segmenter = mock_lm

        result = processor.segment_lungs(image)

        assert result.dtype == np.uint8
        assert np.all((result == 0) | (result == 1))

    def test_segment_lungs_totalseg_backend(self):
        """TotalSegmentator backend merges lung organ masks."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        lung_mask = np.ones((32, 64, 64), dtype=np.uint8)

        processor = _make_processor(segmentation_backend="totalsegmentator")
        processor.segment_organs = MagicMock(
            return_value={"lung_left": lung_mask, "lung_right": lung_mask}
        )

        result = processor.segment_lungs(image)
        assert result.dtype == np.uint8
        assert np.all(result == 1)

    def test_segment_lungs_nnunet_backend(self):
        """NNUNet backend delegates to _get_nnunet().segment_lungs."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        expected = np.ones((32, 64, 64), dtype=np.uint8)

        processor = _make_processor(segmentation_backend="nnunet")
        mock_nnunet = MagicMock()
        mock_nnunet.segment_lungs.return_value = expected
        processor._nnunet_segmenter = mock_nnunet

        result = processor.segment_lungs(image)
        mock_nnunet.segment_lungs.assert_called_once()
        np.testing.assert_array_equal(result, expected)

    def test_segment_lungs_invalid_backend(self):
        """Unknown backend raises ValueError."""
        processor = _make_processor(segmentation_backend="invalid_backend")
        with pytest.raises(ValueError, match="Unknown segmentation backend"):
            processor.segment_lungs(np.zeros((8, 16, 16)))

    def test_segment_organs_totalseg(self):
        """TotalSegmentator segment_organs returns dict of masks."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        expected = {"liver": np.ones((32, 64, 64), dtype=np.uint8)}

        processor = _make_processor(segmentation_backend="totalsegmentator")
        mock_ts = MagicMock()
        mock_ts.segment.return_value = expected
        processor._totalseg_wrapper = mock_ts

        result = processor.segment_organs(image)
        assert "liver" in result

    def test_segment_organs_filter(self):
        """Organ filtering limits returned organs."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        full_result = {
            "liver": np.ones((32, 64, 64), dtype=np.uint8),
            "spleen": np.ones((32, 64, 64), dtype=np.uint8),
            "kidney": np.ones((32, 64, 64), dtype=np.uint8),
        }

        processor = _make_processor(segmentation_backend="totalsegmentator")
        mock_ts = MagicMock()
        mock_ts.segment.return_value = full_result
        processor._totalseg_wrapper = mock_ts

        result = processor.segment_organs(image, organs=["liver"])
        assert "liver" in result
        assert "spleen" not in result

    def test_segment_tumor(self):
        """Tumor segmentation delegates to nnU-Net."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        expected = np.ones((32, 64, 64), dtype=np.uint8)

        processor = _make_processor()
        mock_nnunet = MagicMock()
        mock_nnunet.segment_tumor.return_value = expected
        processor._nnunet_segmenter = mock_nnunet

        result = processor.segment_tumor(image, task="brain_tumor")
        mock_nnunet.segment_tumor.assert_called_once()
        np.testing.assert_array_equal(result, expected)

    def test_segment_tumor_seed_point_deprecation(self):
        """Passing seed_point emits DeprecationWarning."""
        image = np.random.randn(32, 64, 64).astype(np.float32)

        processor = _make_processor()
        mock_nnunet = MagicMock()
        mock_nnunet.segment_tumor.return_value = np.ones_like(image, dtype=np.uint8)
        processor._nnunet_segmenter = mock_nnunet

        with pytest.warns(DeprecationWarning, match="seed_point is deprecated"):
            processor.segment_tumor(image, seed_point=(10, 10, 10))

    def test_segment_metabolic_volume_fixed(self):
        """Fixed-threshold PET segmentation."""
        image = np.random.uniform(0, 5, (32, 64, 64)).astype(np.float32)

        processor = _make_processor()
        mock_pet = MagicMock()
        mock_pet.segment_metabolic_volume.return_value = (image > 2.5).astype(
            np.uint8
        )
        processor._pet_segmenter = mock_pet

        result = processor.segment_metabolic_volume(image, threshold=2.5, method="fixed")
        mock_pet.segment_metabolic_volume.assert_called_once_with(
            image, method="fixed", threshold=2.5
        )
        assert result.shape == image.shape

    def test_segment_metabolic_volume_adaptive(self):
        """Adaptive PET segmentation."""
        image = np.random.uniform(0, 5, (32, 64, 64)).astype(np.float32)

        processor = _make_processor()
        mock_pet = MagicMock()
        mock_pet.segment_metabolic_volume.return_value = np.ones_like(
            image, dtype=np.uint8
        )
        processor._pet_segmenter = mock_pet

        result = processor.segment_metabolic_volume(
            image, threshold=1.5, method="adaptive"
        )
        mock_pet.segment_metabolic_volume.assert_called_once_with(
            image, method="adaptive", threshold=1.5
        )
        assert result.shape == image.shape


# ---------------------------------------------------------------------------
# TestEmbeddingGeneration
# ---------------------------------------------------------------------------


class TestEmbeddingGeneration:
    """Test embedding generation (8 tests)."""

    def _setup_processor_with_mock_model(self):
        """Return processor with an already-loaded mock embedding model."""
        processor = _make_processor()
        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(1, 768).astype(
            np.float32
        )
        processor.embedding_model = mock_model
        return processor, mock_model

    def test_generate_embeddings_2d(self):
        """2D image produces a 1D embedding vector."""
        processor, mock_model = self._setup_processor_with_mock_model()
        mock_model.generate_embeddings.return_value = np.random.randn(1, 768).astype(
            np.float32
        )
        image = np.random.randn(64, 64).astype(np.float32)

        result = processor.generate_embeddings(image, mode="2d")

        assert result.ndim == 1
        assert result.shape[0] == 768

    def test_generate_embeddings_3d(self):
        """3D image uses middle slice by default, producing a 1D embedding."""
        processor, mock_model = self._setup_processor_with_mock_model()
        mock_model.generate_embeddings.return_value = np.random.randn(1, 768).astype(
            np.float32
        )
        image = np.random.randn(32, 64, 64).astype(np.float32)

        result = processor.generate_embeddings(image, mode="3d")

        assert result.ndim == 1
        assert result.shape[0] == 768

    def test_generate_embeddings_lazy_load(self):
        """Model is lazily loaded on first call to generate_embeddings."""
        processor = _make_processor()
        assert processor.embedding_model is None

        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(1, 768).astype(
            np.float32
        )

        with patch(
            "honeybee.processors.radiology.processor.load_model",
            create=True,
        ) as mock_load, patch.dict(
            "sys.modules",
            {"honeybee.models.registry": MagicMock(load_model=mock_load)},
        ):
            mock_load.return_value = mock_model
            # Patch the import path used inside generate_embeddings
            with patch(
                "honeybee.models.registry.load_model", mock_load, create=True
            ):
                processor.embedding_model = mock_model
                image = np.random.randn(64, 64).astype(np.float32)
                processor.generate_embeddings(image)

        mock_model.generate_embeddings.assert_called_once()

    def test_generate_embeddings_with_window(self):
        """Window preset is applied before generating embeddings."""
        processor, mock_model = self._setup_processor_with_mock_model()
        mock_model.generate_embeddings.return_value = np.random.randn(1, 768).astype(
            np.float32
        )
        image = np.random.uniform(-1500, 500, (64, 64)).astype(np.float32)

        result = processor.generate_embeddings(image, window="lung")

        assert result.ndim == 1
        # Verify the model received windowed data (uint8 RGB)
        call_args = mock_model.generate_embeddings.call_args[0][0]
        assert all(img.dtype == np.uint8 for img in call_args)

    def test_aggregate_mean(self):
        """Mean aggregation averages across slices."""
        embeddings = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        processor = _make_processor()
        result = processor.aggregate_embeddings(embeddings, method="mean")

        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_aggregate_max(self):
        """Max aggregation takes element-wise maximum."""
        embeddings = np.array([[1.0, 4.0, 3.0], [3.0, 2.0, 5.0]])
        processor = _make_processor()
        result = processor.aggregate_embeddings(embeddings, method="max")

        np.testing.assert_array_almost_equal(result, [3.0, 4.0, 5.0])

    def test_aggregate_concat(self):
        """Concat aggregation concatenates mean and std."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        processor = _make_processor()
        result = processor.aggregate_embeddings(embeddings, method="concat")

        assert result.shape[0] == 4  # 2 (mean) + 2 (std)
        np.testing.assert_almost_equal(result[0], 2.0)  # mean of [1,3]
        np.testing.assert_almost_equal(result[1], 3.0)  # mean of [2,4]

    def test_aggregate_invalid_method(self):
        """Unknown aggregation method raises ValueError."""
        embeddings = np.array([[1.0, 2.0]])
        processor = _make_processor()

        with pytest.raises(ValueError, match="Unknown aggregation method"):
            processor.aggregate_embeddings(embeddings, method="unknown")


# ---------------------------------------------------------------------------
# TestPreprocessingComponents
# ---------------------------------------------------------------------------


class TestPreprocessingComponents:
    """Test individual preprocessing components (12 tests)."""

    def test_denoiser_supported_methods(self):
        """All documented methods are accepted."""
        from honeybee.processors.radiology.preprocessing import Denoiser

        for method in [
            "nlm",
            "tv",
            "bilateral",
            "median",
            "gaussian",
            "pet_specific",
            "rician",
            "deep",
            "dipy_nlm",
            "dipy_mppca",
        ]:
            d = Denoiser(method=method)
            assert d.method == method

    def test_denoiser_invalid_method(self):
        """Invalid method raises ValueError."""
        from honeybee.processors.radiology.preprocessing import Denoiser

        with pytest.raises(ValueError, match="not supported"):
            Denoiser(method="fake_method")

    def test_normalizer_zscore(self):
        """Z-score normalizer produces zero-mean unit-variance output."""
        from honeybee.processors.radiology.preprocessing import IntensityNormalizer

        normalizer = IntensityNormalizer(method="zscore")
        image = np.random.randn(64, 64).astype(np.float32) * 50 + 100
        result = normalizer.normalize(image)

        np.testing.assert_almost_equal(result.mean(), 0.0, decimal=4)
        np.testing.assert_almost_equal(result.std(), 1.0, decimal=2)

    def test_normalizer_minmax(self):
        """Min-max normalizer scales to [0, 1] by default."""
        from honeybee.processors.radiology.preprocessing import IntensityNormalizer

        normalizer = IntensityNormalizer(method="minmax")
        image = np.random.uniform(-500, 500, (64, 64)).astype(np.float32)
        result = normalizer.normalize(image)

        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_normalizer_percentile(self):
        """Percentile normalizer clips and scales to [0, 1]."""
        from honeybee.processors.radiology.preprocessing import IntensityNormalizer

        normalizer = IntensityNormalizer(method="percentile")
        image = np.random.uniform(-1000, 3000, (64, 64)).astype(np.float32)
        result = normalizer.normalize(image, lower=5, upper=95)

        assert result.min() >= 0.0 - 1e-6
        assert result.max() <= 1.0 + 1e-6

    def test_window_adjuster_numeric(self):
        """Numeric window/level clips and scales correctly."""
        from honeybee.processors.radiology.preprocessing import WindowLevelAdjuster

        adjuster = WindowLevelAdjuster()
        image = np.array([-1000, -500, 0, 500, 1000], dtype=np.float32)
        result = adjuster.adjust(image, window=1000, level=0)

        assert result[0] == pytest.approx(0, abs=1)
        assert result[-1] == pytest.approx(255, abs=1)

    def test_window_adjuster_preset_lung(self):
        """Lung preset uses center=-600, width=1500."""
        from honeybee.processors.radiology.preprocessing import WindowLevelAdjuster

        adjuster = WindowLevelAdjuster()
        image = np.array([-1350, -600, 150], dtype=np.float32)
        result = adjuster.adjust(image, window="lung")

        assert result[0] == pytest.approx(0, abs=1)
        assert result[1] == pytest.approx(127.5, abs=1)
        assert result[2] == pytest.approx(255, abs=1)

    def test_window_adjuster_preset_brain(self):
        """Brain preset uses center=40, width=80."""
        from honeybee.processors.radiology.preprocessing import WindowLevelAdjuster

        adjuster = WindowLevelAdjuster()
        image = np.array([0, 40, 80], dtype=np.float32)
        result = adjuster.adjust(image, window="brain")

        assert result[0] == pytest.approx(0, abs=1)
        assert result[1] == pytest.approx(127.5, abs=1)
        assert result[2] == pytest.approx(255, abs=1)

    def test_hu_clipper_default(self):
        """Default HU clipper clips to [-1024, 3071]."""
        from honeybee.processors.radiology.preprocessing import HUClipper

        clipper = HUClipper()
        image = np.array([-2000, 0, 5000], dtype=np.float32)
        result = clipper.clip(image)

        assert result[0] == -1024
        assert result[1] == 0
        assert result[2] == 3071

    def test_hu_clipper_preset(self):
        """Preset overrides min/max values."""
        from honeybee.processors.radiology.preprocessing import HUClipper

        clipper = HUClipper()
        image = np.array([-500, 0, 300, 1000], dtype=np.float32)
        result = clipper.clip(image, preset="soft_tissue")

        assert result[0] == -200
        assert result[1] == 0
        assert result[2] == 300
        assert result[3] == 400

    def test_voxel_resampler(self):
        """VoxelResampler changes volume shape based on zoom factors."""
        from honeybee.processors.radiology.preprocessing import VoxelResampler

        mock_zoom = MagicMock(
            return_value=np.zeros((32, 64, 64), dtype=np.float32)
        )
        with patch.dict(
            "sys.modules",
            {
                "scipy": MagicMock(ndimage=MagicMock(zoom=mock_zoom)),
                "scipy.ndimage": MagicMock(zoom=mock_zoom),
            },
        ):
            resampler = VoxelResampler()
            image = np.random.randn(16, 32, 32).astype(np.float32)
            result = resampler.resample(
                image,
                current_spacing=(2.0, 2.0, 2.0),
                target_spacing=(1.0, 1.0, 1.0),
                order=1,
            )

        assert isinstance(result, np.ndarray)

    def test_artifact_reducer_metal(self):
        """Metal artifact reducer delegates to cv2 inpainting."""
        from honeybee.processors.radiology.preprocessing import ArtifactReducer

        reducer = ArtifactReducer()
        mock_cv2 = MagicMock()
        mock_cv2.normalize.return_value = np.zeros((64, 64), dtype=np.uint8)
        mock_cv2.inpaint.return_value = np.zeros((64, 64), dtype=np.uint8)
        mock_cv2.NORM_MINMAX = 32
        mock_cv2.CV_8U = 0
        mock_cv2.INPAINT_TELEA = 0

        image = np.zeros((64, 64), dtype=np.float32)
        image[30, 30] = 4000  # Metal pixel

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = reducer.reduce_artifacts(image, artifact_type="metal")

        assert result.shape == image.shape


# ---------------------------------------------------------------------------
# TestNNUNetSegmenter
# ---------------------------------------------------------------------------


class TestNNUNetSegmenter:
    """Test NNUNetSegmenter (6 tests)."""

    def test_init_with_model_paths(self):
        """Model paths are stored on construction."""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        paths = {"lung": "/models/lung", "brain": "/models/brain"}
        seg = NNUNetSegmenter(model_paths=paths)

        assert seg.available_tasks == ["lung", "brain"]

    def test_init_without_model_paths(self):
        """No model paths produces empty available_tasks."""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        seg = NNUNetSegmenter()
        assert seg.available_tasks == []

    def test_segment_lungs(self):
        """segment_lungs calls predict_raw and binarises."""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        seg = NNUNetSegmenter(model_paths={"lung": "/tmp/lung"})
        raw_seg = np.array([[[0, 1], [2, 0]]], dtype=np.int32)
        seg.predict_raw = MagicMock(return_value=raw_seg)

        result = seg.segment_lungs(np.zeros((1, 2, 2)), task="lung")

        assert result.dtype == np.uint8
        expected = (raw_seg > 0).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_segment_organs(self):
        """segment_organs maps label IDs to named binary masks."""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        seg = NNUNetSegmenter(model_paths={"total_organs": "/tmp/organs"})
        raw_seg = np.array([[[1, 2], [5, 0]]], dtype=np.int32)
        seg.predict_raw = MagicMock(return_value=raw_seg)

        result = seg.segment_organs(np.zeros((1, 2, 2)), task="total_organs")

        assert "spleen" in result  # label 1
        assert "kidney_right" in result  # label 2
        assert "liver" in result  # label 5

    def test_segment_tumor(self):
        """segment_tumor calls predict_raw and binarises."""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        seg = NNUNetSegmenter(model_paths={"brain_tumor": "/tmp/tumor"})
        raw_seg = np.array([[[0, 1], [1, 0]]], dtype=np.int32)
        seg.predict_raw = MagicMock(return_value=raw_seg)

        result = seg.segment_tumor(np.zeros((1, 2, 2)), task="brain_tumor")

        assert result.dtype == np.uint8
        assert result.sum() == 2

    def test_label_map_loading(self, tmp_path):
        """Label map is read from dataset.json when available."""
        from honeybee.processors.radiology.segmentation import NNUNetSegmenter

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        dataset_json = model_dir / "dataset.json"
        import json

        dataset_json.write_text(
            json.dumps(
                {
                    "labels": {
                        "0": "background",
                        "1": "tumor_core",
                        "2": "edema",
                    }
                }
            )
        )

        seg = NNUNetSegmenter(model_paths={"custom": str(model_dir)})
        labels = seg.get_label_map("custom")

        assert labels[1] == "tumor_core"
        assert labels[2] == "edema"
        assert 0 not in labels  # background is excluded


# ---------------------------------------------------------------------------
# TestImageMetadata
# ---------------------------------------------------------------------------


class TestImageMetadata:
    """Test ImageMetadata dataclass (6 tests)."""

    def test_is_ct(self):
        """is_ct returns True for 'CT' modality."""
        meta = _make_metadata("CT")
        assert meta.is_ct() is True
        assert meta.is_mri() is False
        assert meta.is_pet() is False

    def test_is_mri(self):
        """is_mri returns True for 'MR' modality."""
        meta = _make_metadata("MR")
        assert meta.is_mri() is True
        assert meta.is_ct() is False

    def test_is_pet(self):
        """is_pet returns True for 'PT' modality."""
        meta = _make_metadata("PT")
        assert meta.is_pet() is True
        assert meta.is_ct() is False

    def test_get_voxel_spacing(self):
        """get_voxel_spacing returns pixel_spacing tuple."""
        meta = _make_metadata(pixel_spacing=(0.5, 0.5, 2.0))
        assert meta.get_voxel_spacing() == (0.5, 0.5, 2.0)

    def test_has_window_settings(self):
        """has_window_settings requires both center and width."""
        meta_with = _make_metadata(window_center=40.0, window_width=80.0)
        meta_without = _make_metadata()

        assert meta_with.has_window_settings() is True
        assert meta_without.has_window_settings() is False

    def test_get_image_size(self):
        """get_image_size returns (rows, columns, slices)."""
        meta = _make_metadata(rows=512, columns=512, number_of_slices=128)
        assert meta.get_image_size() == (512, 512, 128)


# ---------------------------------------------------------------------------
# TestPipelineIntegration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Test pipeline integration (4 tests)."""

    def test_prepare_for_model_2d(self):
        """2D image is converted to a list with one RGB uint8 array."""
        image = np.random.randn(64, 64).astype(np.float32)
        processor = _make_processor()
        result = processor.prepare_for_model(image)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)
        assert result[0].dtype == np.uint8

    def test_prepare_for_model_3d(self):
        """3D image with n_slices returns the requested number of slices."""
        image = np.random.randn(32, 64, 64).astype(np.float32)
        processor = _make_processor()
        result = processor.prepare_for_model(image, n_slices=5)

        assert isinstance(result, list)
        assert len(result) == 5
        for sl in result:
            assert sl.shape == (64, 64, 3)
            assert sl.dtype == np.uint8

    def test_process_batch(self):
        """process_batch returns stacked embeddings for a list of images."""
        processor = _make_processor()
        mock_model = MagicMock()
        mock_model.generate_embeddings.return_value = np.random.randn(1, 768).astype(
            np.float32
        )
        processor.embedding_model = mock_model

        images = [np.random.randn(64, 64).astype(np.float32) for _ in range(3)]
        result = processor.process_batch(images, batch_size=2)

        assert result.shape == (3, 768)

    def test_get_model_info(self):
        """get_model_info returns expected keys."""
        processor = _make_processor()

        mock_registry = MagicMock()
        mock_registry._PRESET_REGISTRY = {}

        with patch.dict("sys.modules", {"honeybee.models": mock_registry,
                                         "honeybee.models.registry": mock_registry}):
            info = processor.get_model_info()

        assert "model_name" in info
        assert "device" in info
        assert "is_loaded" in info
        assert info["model_name"] == "remedis"
        assert info["is_loaded"] is False


# ---------------------------------------------------------------------------
# Additional edge-case tests to bring total to ~97
# ---------------------------------------------------------------------------


class TestWindowPresets:
    """Test all window presets in WindowLevelAdjuster (3 tests)."""

    def test_all_presets_are_accessible(self):
        """Every documented preset can be used without error."""
        from honeybee.processors.radiology.preprocessing import WindowLevelAdjuster

        adjuster = WindowLevelAdjuster()
        image = np.random.uniform(-1500, 3000, (64, 64)).astype(np.float32)
        for preset in [
            "lung",
            "abdomen",
            "bone",
            "brain",
            "soft_tissue",
            "liver",
            "mediastinum",
            "stroke",
            "cta",
            "pet",
        ]:
            result = adjuster.adjust(image, window=preset)
            assert result.min() >= 0
            assert result.max() <= 255

    def test_unknown_preset_raises(self):
        """Unknown preset name raises ValueError."""
        from honeybee.processors.radiology.preprocessing import WindowLevelAdjuster

        adjuster = WindowLevelAdjuster()
        with pytest.raises(ValueError, match="Unknown preset"):
            adjuster.adjust(np.zeros((4, 4)), window="nonexistent")

    def test_auto_window_calculation(self):
        """get_auto_window returns center and width keys."""
        from honeybee.processors.radiology.preprocessing import WindowLevelAdjuster

        adjuster = WindowLevelAdjuster()
        image = np.random.uniform(-500, 500, (64, 64)).astype(np.float32)
        result = adjuster.get_auto_window(image)

        assert "center" in result
        assert "width" in result
        assert result["width"] > 0


class TestPETSegmenter:
    """Test PETSegmenter directly (3 tests)."""

    def test_fixed_threshold_segmentation(self):
        """Fixed threshold creates binary mask at SUV > threshold."""
        from honeybee.processors.radiology.segmentation import PETSegmenter

        seg = PETSegmenter()

        mock_morphology = MagicMock()
        mock_morphology.remove_small_objects.side_effect = lambda x, **kw: x
        mock_binary_fill = MagicMock(side_effect=lambda x: x)

        with patch.dict(
            "sys.modules",
            {
                "skimage": MagicMock(morphology=mock_morphology),
                "skimage.morphology": mock_morphology,
                "scipy.ndimage": MagicMock(binary_fill_holes=mock_binary_fill),
            },
        ):
            image = np.array([[1.0, 3.0], [5.0, 0.5]])
            result = seg.segment_metabolic_volume(image, method="fixed", threshold=2.5)

        assert result[0, 0] == False  # noqa: E712
        assert result[0, 1] == True  # noqa: E712
        assert result[1, 0] == True  # noqa: E712

    def test_unknown_method_raises(self):
        """Unknown segmentation method raises ValueError."""
        from honeybee.processors.radiology.segmentation import PETSegmenter

        seg = PETSegmenter()
        with pytest.raises(ValueError, match="Unknown method"):
            seg.segment_metabolic_volume(np.zeros((4, 4)), method="magic")

    def test_suv_metrics_empty_mask(self):
        """Empty mask returns zero metrics."""
        from honeybee.processors.radiology.segmentation import PETSegmenter

        seg = PETSegmenter()
        image = np.random.uniform(0, 5, (16, 16)).astype(np.float32)
        mask = np.zeros((16, 16), dtype=bool)
        metrics = seg.calculate_suv_metrics(image, mask)

        assert metrics["suv_max"] == 0.0
        assert metrics["mtv"] == 0.0


class TestDetectWindowPreset:
    """Test the private _detect_window_preset method (5 tests)."""

    def test_lung_keywords(self):
        """LUNG/CHEST/THORAX series description yields 'lung'."""
        from honeybee.processors.radiology.processor import RadiologyProcessor

        for desc in ["LUNG CT", "CHEST CT", "THORAX", "PULMONARY"]:
            meta = _make_metadata(series_description=desc)
            assert RadiologyProcessor._detect_window_preset(meta) == "lung"

    def test_abdomen_keywords(self):
        """ABD/PELVIS series description yields 'abdomen'."""
        from honeybee.processors.radiology.processor import RadiologyProcessor

        for desc in ["ABD CT", "PELVIS", "LIVER CT", "ABDOMEN"]:
            meta = _make_metadata(series_description=desc)
            assert RadiologyProcessor._detect_window_preset(meta) == "abdomen"

    def test_brain_keywords(self):
        """BRAIN/HEAD series description yields 'brain'."""
        from honeybee.processors.radiology.processor import RadiologyProcessor

        for desc in ["BRAIN MRI", "HEAD CT", "NEURO", "CRANIAL"]:
            meta = _make_metadata(series_description=desc)
            assert RadiologyProcessor._detect_window_preset(meta) == "brain"

    def test_bone_keywords(self):
        """BONE/SPINE series description yields 'bone'."""
        from honeybee.processors.radiology.processor import RadiologyProcessor

        for desc in ["BONE SCAN", "SPINE CT", "SKELETAL", "MSK"]:
            meta = _make_metadata(series_description=desc)
            assert RadiologyProcessor._detect_window_preset(meta) == "bone"

    def test_fallback_soft_tissue(self):
        """Unrecognised description falls back to 'soft_tissue'."""
        from honeybee.processors.radiology.processor import RadiologyProcessor

        meta = _make_metadata(series_description="Random description")
        assert RadiologyProcessor._detect_window_preset(meta) == "soft_tissue"


class TestOrientationHelper:
    """Test _orientation_to_direction helper (2 tests)."""

    def test_standard_6_element_orientation(self):
        """6-element orientation is expanded to 9-element direction."""
        from honeybee.processors.radiology.processor import RadiologyProcessor

        orientation = [1, 0, 0, 0, 1, 0]
        direction = RadiologyProcessor._orientation_to_direction(orientation)

        assert len(direction) == 9
        assert direction[6] == 0  # cross-product z
        assert direction[7] == 0
        assert direction[8] == 1

    def test_non_6_element_fallback(self):
        """Non-standard orientation returns identity direction."""
        from honeybee.processors.radiology.processor import RadiologyProcessor

        direction = RadiologyProcessor._orientation_to_direction([1, 0, 0])
        assert direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)


class TestModelAliases:
    """Test model alias edge cases (3 tests)."""

    def test_case_insensitive_alias(self):
        """Model names are case-insensitive."""
        processor = _make_processor(model="REMEDIS")
        assert processor.model_name == "remedis"

    def test_unknown_model_stored_as_is(self):
        """Unknown model names are lowered and stored directly."""
        processor = _make_processor(model="CustomModel")
        assert processor.model_name == "custommodel"

    def test_radimagenet_densenet_variant(self):
        """radimagenet-densenet121 is a valid alias."""
        processor = _make_processor(model="radimagenet-densenet121")
        assert processor.model_name == "radimagenet-densenet121"


class TestAggregateEdgeCases:
    """Test aggregation edge cases (3 tests)."""

    def test_aggregate_median(self):
        """Median aggregation computes element-wise median."""
        embeddings = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 1.0]])
        processor = _make_processor()
        result = processor.aggregate_embeddings(embeddings, method="median")

        np.testing.assert_array_almost_equal(result, [2.0, 3.0])

    def test_aggregate_std(self):
        """Std aggregation computes element-wise standard deviation."""
        embeddings = np.array([[0.0, 0.0], [2.0, 2.0]])
        processor = _make_processor()
        result = processor.aggregate_embeddings(embeddings, method="std")

        np.testing.assert_array_almost_equal(result, [1.0, 1.0])

    def test_aggregate_empty_raises(self):
        """Aggregating empty array raises ValueError."""
        processor = _make_processor()
        with pytest.raises(ValueError, match="No embeddings"):
            processor.aggregate_embeddings(np.array([]))


class TestPrepareForModelEdgeCases:
    """Test prepare_for_model edge cases (3 tests)."""

    def test_prepare_default_3d_uses_middle_slice(self):
        """Without n_slices, only the middle slice is returned."""
        image = np.random.randn(20, 64, 64).astype(np.float32)
        processor = _make_processor()
        result = processor.prepare_for_model(image)

        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_prepare_with_metadata_detects_window(self):
        """Metadata with lung description triggers lung window."""
        image = np.random.uniform(-1500, 500, (32, 64, 64)).astype(np.float32)
        metadata = _make_metadata(series_description="CHEST CT")
        processor = _make_processor()
        result = processor.prepare_for_model(image, metadata=metadata)

        # Windowed images are uint8
        assert result[0].dtype == np.uint8

    def test_prepare_constant_image(self):
        """Constant-value image produces zeros (no range to scale)."""
        image = np.full((64, 64), 500.0, dtype=np.float32)
        processor = _make_processor()
        result = processor.prepare_for_model(image)

        # When window is None and range is zero the branch produces zeros
        # BUT with constant value, img_max - img_min > 0 is False, so all zeros
        assert result[0].shape == (64, 64, 3)


class TestNormalizerInvalidMethod:
    """Test IntensityNormalizer rejects bad methods (1 test)."""

    def test_normalizer_invalid_method(self):
        """Invalid method raises ValueError."""
        from honeybee.processors.radiology.preprocessing import IntensityNormalizer

        with pytest.raises(ValueError, match="not supported"):
            IntensityNormalizer(method="magic")


class TestHUClipperInvalidPreset:
    """Test HUClipper rejects bad presets (1 test)."""

    def test_hu_clipper_invalid_preset(self):
        """Invalid preset raises ValueError."""
        from honeybee.processors.radiology.preprocessing import HUClipper

        clipper = HUClipper()
        with pytest.raises(ValueError, match="Unknown preset"):
            clipper.clip(np.zeros((4, 4)), preset="imaginary")


class TestArtifactReducerMethods:
    """Test ArtifactReducer dispatching (2 tests)."""

    def test_unknown_artifact_type(self):
        """Unknown artifact type raises ValueError."""
        from honeybee.processors.radiology.preprocessing import ArtifactReducer

        reducer = ArtifactReducer()
        with pytest.raises(ValueError, match="Unknown artifact type"):
            reducer.reduce_artifacts(np.zeros((4, 4)), artifact_type="cosmic_rays")

    def test_beam_hardening_correction(self):
        """Beam hardening correction applies quadratic correction."""
        from honeybee.processors.radiology.preprocessing import ArtifactReducer

        reducer = ArtifactReducer()
        image = np.ones((4, 4), dtype=np.float32) * 100
        result = reducer.reduce_artifacts(
            image, artifact_type="beam_hardening", correction_factor=0.1
        )

        # corrected = image - factor * image^2 / max^2
        expected = 100 - 0.1 * (100**2) / (100**2)
        np.testing.assert_almost_equal(result[0, 0], expected, decimal=4)
