"""
Radiology Processor

Main processor for radiological imaging data. Provides a unified API for:
- Image loading (DICOM/NIfTI)
- Preprocessing (denoising, normalization, windowing)
- Segmentation (lungs, organs, tumors, metabolic volumes)
- Spatial operations (resampling, registration, reorientation)
- Embedding generation via the model registry
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .metadata import ImageMetadata
from .preprocessing import (
    ArtifactReducer,
    Denoiser,
    IntensityNormalizer,
    WindowLevelAdjuster,
    preprocess_ct,
    preprocess_mri,
    preprocess_pet,
)
from .segmentation import NNUNetSegmenter, PETSegmenter

logger = logging.getLogger(__name__)

# Model alias map for the registry
_MODEL_ALIASES = {
    "remedis": "remedis",
    "radimagenet": "radimagenet-resnet50",
    "radimagenet-resnet50": "radimagenet-resnet50",
    "radimagenet-densenet121": "radimagenet-densenet121",
    "medsiglip": "medsiglip",
    "rad-dino": "rad-dino",
}


class RadiologyProcessor:
    """Unified processor for radiological imaging data.

    Combines image loading, preprocessing, segmentation, and embedding
    generation into a single API matching the HoneyBee docs.

    Args:
        model: Embedding model alias ('remedis', 'radimagenet', 'medsiglip', 'rad-dino')
            or a registry preset name.
        model_path: Path to model weights (required for some models like remedis).
        device: Device for computation ('cuda' or 'cpu'). Auto-detected if None.
        segmentation_backend: Default segmentation backend
            ('lungmask', 'totalsegmentator', 'nnunet').
        segmentation_model_paths: Mapping of task names to nnU-Net model folder paths.
        **model_kwargs: Extra keyword arguments forwarded to the model registry.

    Example:
        >>> processor = RadiologyProcessor(model="remedis", model_path="/path/to/weights")
        >>> image, metadata = processor.load_dicom("ct_series/")
        >>> preprocessed = processor.preprocess(image, metadata)
        >>> embeddings = processor.generate_embeddings(preprocessed)
    """

    def __init__(
        self,
        model: str = "remedis",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        segmentation_backend: str = "lungmask",
        segmentation_model_paths: Optional[Dict[str, str]] = None,
        **model_kwargs,
    ):
        # Resolve model alias
        self.model_name = _MODEL_ALIASES.get(model.lower(), model.lower())
        self.model_path = model_path
        self._model_kwargs = model_kwargs
        self.embedding_model = None  # Lazy loaded

        # Device auto-detection
        if device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Segmentation backend
        self._segmentation_backend = segmentation_backend.lower()
        self._segmentation_model_paths = segmentation_model_paths or {}

        # Lazy-initialized components
        self._lungmask_segmenter = None
        self._totalseg_wrapper = None
        self._nnunet_segmenter = None
        self._pet_segmenter = None
        self._windower = WindowLevelAdjuster()
        self._artifact_reducer = ArtifactReducer()

    # ================================================================
    # Image Loading
    # ================================================================

    def load_dicom(self, folder_or_file: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load DICOM file or series.

        Args:
            folder_or_file: Path to a single DICOM file or directory containing a series.

        Returns:
            Tuple of (image array, ImageMetadata).
        """
        import SimpleITK as sitk

        path = Path(folder_or_file)

        if path.is_dir():
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            if not dicom_names:
                raise FileNotFoundError(f"No DICOM files found in {path}")
            reader.SetFileNames(dicom_names)
            sitk_image = reader.Execute()
        else:
            sitk_image = sitk.ReadImage(str(path))

        image = sitk.GetArrayFromImage(sitk_image)
        metadata = self._sitk_to_metadata(sitk_image, path)
        return image, metadata

    def load_nifti(self, path: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load NIfTI file.

        Args:
            path: Path to NIfTI file (.nii or .nii.gz).

        Returns:
            Tuple of (image array, ImageMetadata).
        """
        import nibabel as nib

        img = nib.load(str(path))
        data = img.get_fdata()
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])

        metadata = ImageMetadata(
            modality="unknown",
            patient_id="",
            study_date="",
            series_description=str(Path(path).stem),
            pixel_spacing=spacing if len(spacing) == 3 else (spacing[0], spacing[1], 1.0),
            image_position=(0.0, 0.0, 0.0),
            image_orientation=[1, 0, 0, 0, 1, 0],
            rows=data.shape[0] if data.ndim >= 2 else None,
            columns=data.shape[1] if data.ndim >= 2 else None,
            number_of_slices=data.shape[2] if data.ndim >= 3 else 1,
        )
        return data, metadata

    def load_atlas(self, path: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load atlas image (alias for load_nifti).

        Args:
            path: Path to atlas NIfTI file.

        Returns:
            Tuple of (atlas array, ImageMetadata).
        """
        return self.load_nifti(path)

    # ================================================================
    # Segmentation
    # ================================================================

    def segment_lungs(
        self, image: np.ndarray, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """Segment lungs from CT scan.

        Uses the configured segmentation backend (default: lungmask).

        Args:
            image: CT volume in HU values.
            spacing: Voxel spacing (z, y, x) in mm.

        Returns:
            Binary lung mask (uint8).
        """
        if self._segmentation_backend == "lungmask":
            if self._lungmask_segmenter is None:
                from .segmentation import LungmaskSegmenter

                self._lungmask_segmenter = LungmaskSegmenter()
            seg = self._lungmask_segmenter.segment(image, spacing=spacing)
            return (seg > 0).astype(np.uint8)
        elif self._segmentation_backend == "nnunet":
            return self._get_nnunet().segment_lungs(image, spacing=spacing)
        elif self._segmentation_backend == "totalsegmentator":
            organs = self.segment_organs(image, spacing=spacing)
            lung_keys = [k for k in organs if "lung" in k.lower()]
            if lung_keys:
                mask = np.zeros_like(image, dtype=np.uint8)
                for k in lung_keys:
                    mask = np.maximum(mask, organs[k])
                return mask
            return np.zeros_like(image, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown segmentation backend: {self._segmentation_backend}")

    def segment_organs(
        self,
        image: np.ndarray,
        organs: Optional[List[str]] = None,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> Dict[str, np.ndarray]:
        """Segment multiple organs from CT image.

        Args:
            image: CT image in HU values.
            organs: List of organs to segment (None returns all).
            spacing: Voxel spacing (z, y, x) in mm.

        Returns:
            Dict mapping organ names to binary masks.
        """
        if self._segmentation_backend == "totalsegmentator":
            if self._totalseg_wrapper is None:
                from .segmentation import TotalSegmentatorWrapper

                self._totalseg_wrapper = TotalSegmentatorWrapper()
            results = self._totalseg_wrapper.segment(image, spacing=spacing)
            if organs is not None:
                results = {k: v for k, v in results.items() if k in organs}
            return results
        else:
            return self._get_nnunet().segment_organs(image, spacing=spacing, organs=organs)

    def segment_tumor(
        self,
        image: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        seed_point: Optional[Tuple[int, ...]] = None,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        task: str = "brain_tumor",
    ) -> np.ndarray:
        """Segment tumor using nnU-Net.

        Args:
            image: Medical image (CT/MRI).
            metadata: Image metadata (unused, kept for API compat).
            seed_point: Deprecated. Ignored.
            spacing: Voxel spacing (z, y, x) in mm.
            task: nnU-Net task name.

        Returns:
            Binary tumor mask.
        """
        if seed_point is not None:
            warnings.warn(
                "seed_point is deprecated and ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self._get_nnunet().segment_tumor(image, spacing=spacing, task=task)

    def segment_metabolic_volume(
        self, image: np.ndarray, threshold: float = 2.5, method: str = "fixed"
    ) -> np.ndarray:
        """Segment metabolically active regions in PET image.

        Args:
            image: PET image (preferably in SUV units).
            threshold: SUV threshold for segmentation.
            method: Segmentation method ('fixed', 'adaptive', 'gradient').

        Returns:
            Binary mask of metabolically active regions.
        """
        if self._pet_segmenter is None:
            self._pet_segmenter = PETSegmenter()
        return self._pet_segmenter.segment_metabolic_volume(
            image, method=method, threshold=threshold
        )

    # ================================================================
    # Denoising
    # ================================================================

    def denoise(self, image: np.ndarray, method: str = "nlm", **kwargs) -> np.ndarray:
        """Apply denoising to medical image.

        Args:
            image: Input image.
            method: Denoising method ('nlm', 'bilateral', 'rician', 'deep', 'pet_specific', etc.)
            **kwargs: Method-specific parameters.

        Returns:
            Denoised image.
        """
        denoiser = Denoiser(method=method)
        return denoiser.denoise(image, **kwargs)

    def reduce_metal_artifacts(self, image: np.ndarray, threshold: float = 3000) -> np.ndarray:
        """Reduce metal artifacts in CT images.

        Args:
            image: CT scan in HU values.
            threshold: HU threshold for metal detection.

        Returns:
            CT image with reduced metal artifacts.
        """
        return self._artifact_reducer.reduce_artifacts(
            image, artifact_type="metal", threshold=threshold
        )

    def correct_bias_field(self, image: np.ndarray, backend: str = "sitk") -> np.ndarray:
        """Apply N4 bias field correction to MRI image.

        Args:
            image: Input MRI image.
            backend: Backend to use ('sitk' or 'ants').

        Returns:
            Bias-corrected image.
        """
        if backend == "ants":
            try:
                import ants

                ants_img = ants.from_numpy(image.astype(np.float32))
                corrected = ants.n4_bias_field_correction(ants_img)
                return corrected.numpy()
            except ImportError:
                logger.warning("ANTsPy not available, falling back to SimpleITK")

        import SimpleITK as sitk

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(sitk_image)
        return sitk.GetArrayFromImage(corrected)

    # ================================================================
    # Spatial Operations
    # ================================================================

    def resample(
        self,
        image: np.ndarray,
        metadata: ImageMetadata,
        new_spacing: Tuple[float, float, float],
        interpolation: str = "linear",
    ) -> np.ndarray:
        """Resample image to new spacing using SimpleITK.

        Args:
            image: Input image.
            metadata: Image metadata with current spacing.
            new_spacing: Target spacing (x, y, z).
            interpolation: Interpolation method ('linear', 'nearest', 'bspline').

        Returns:
            Resampled image.
        """
        import SimpleITK as sitk

        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing(metadata.pixel_spacing[::-1])

        original_size = sitk_image.GetSize()
        original_spacing = sitk_image.GetSpacing()
        new_size = [
            int(round(osz * osp / nsp))
            for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())

        interp_map = {
            "linear": sitk.sitkLinear,
            "nearest": sitk.sitkNearestNeighbor,
            "bspline": sitk.sitkBSpline,
        }
        resampler.SetInterpolator(interp_map.get(interpolation, sitk.sitkLinear))
        resampled = resampler.Execute(sitk_image)
        return sitk.GetArrayFromImage(resampled)

    def reorient(
        self, image: np.ndarray, metadata: ImageMetadata, target_orientation: str = "RAS"
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """Reorient medical image to standard orientation.

        Args:
            image: Input image.
            metadata: Image metadata with orientation information.
            target_orientation: Target orientation code (e.g. 'RAS', 'LPS').

        Returns:
            Tuple of (reoriented image, updated metadata).
        """
        import SimpleITK as sitk

        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing(metadata.pixel_spacing[::-1])
        sitk_image.SetOrigin(metadata.image_position)

        direction = self._orientation_to_direction(metadata.image_orientation)
        sitk_image.SetDirection(direction)

        reoriented = sitk.DICOMOrient(sitk_image, target_orientation)
        reoriented_array = sitk.GetArrayFromImage(reoriented)

        new_metadata = ImageMetadata(
            modality=metadata.modality,
            patient_id=metadata.patient_id,
            study_date=metadata.study_date,
            series_description=metadata.series_description,
            pixel_spacing=tuple(reoriented.GetSpacing()[::-1]),
            image_position=tuple(reoriented.GetOrigin()),
            image_orientation=list(reoriented.GetDirection()),
            window_center=metadata.window_center,
            window_width=metadata.window_width,
            rescale_intercept=metadata.rescale_intercept,
            rescale_slope=metadata.rescale_slope,
            manufacturer=metadata.manufacturer,
            scanner_model=metadata.scanner_model,
            kvp=metadata.kvp,
            exposure=metadata.exposure,
            slice_thickness=reoriented.GetSpacing()[2],
            spacing_between_slices=reoriented.GetSpacing()[2],
            rows=reoriented_array.shape[1] if len(reoriented_array.shape) > 1 else None,
            columns=reoriented_array.shape[2] if len(reoriented_array.shape) > 2 else None,
            number_of_slices=(reoriented_array.shape[0] if len(reoriented_array.shape) > 2 else 1),
            extra_metadata=metadata.extra_metadata,
        )

        return reoriented_array, new_metadata

    def register(
        self, image: np.ndarray, atlas_image: np.ndarray, method: str = "rigid"
    ) -> np.ndarray:
        """Register moving image to fixed image.

        Args:
            image: Image to be transformed (moving).
            atlas_image: Reference image (fixed).
            method: Registration method ('rigid', 'affine', 'syn').

        Returns:
            Registered image.
        """
        import SimpleITK as sitk

        fixed_sitk = sitk.GetImageFromArray(atlas_image)
        moving_sitk = sitk.GetImageFromArray(image)

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        ndim = len(atlas_image.shape)
        if method == "rigid":
            transform_cls = sitk.Euler3DTransform() if ndim == 3 else sitk.Euler2DTransform()
        elif method == "affine":
            transform_cls = sitk.AffineTransform(ndim)
        else:
            transform_cls = sitk.Euler3DTransform() if ndim == 3 else sitk.Euler2DTransform()

        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk,
            moving_sitk,
            transform_cls,
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        try:
            final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)
            registered_sitk = resampler.Execute(moving_sitk)
            return sitk.GetArrayFromImage(registered_sitk)
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return image

    def crop_to_roi(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop image to the bounding box of a binary mask.

        Args:
            image: Input image (2D or 3D).
            mask: Binary mask defining the ROI.

        Returns:
            Cropped image.
        """
        masked = image * mask
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            logger.warning("Empty mask provided, returning original image")
            return image
        slices = tuple(slice(c.min(), c.max() + 1) for c in coords)
        return masked[slices]

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply binary mask to image (zero outside mask).

        Args:
            image: Input image.
            mask: Binary mask.

        Returns:
            Masked image.
        """
        return image * mask

    # ================================================================
    # Normalization
    # ================================================================

    def verify_hounsfield_units(
        self, image: np.ndarray, metadata: Optional[ImageMetadata] = None
    ) -> Dict[str, Any]:
        """Verify that CT image is in Hounsfield Units.

        Args:
            image: CT image array.
            metadata: Image metadata (optional).

        Returns:
            Dictionary with verification results and statistics.
        """
        results = {
            "is_hu": False,
            "min_value": float(image.min()),
            "max_value": float(image.max()),
            "mean_value": float(image.mean()),
            "likely_air_present": False,
            "likely_bone_present": False,
            "warnings": [],
        }

        min_val = image.min()
        max_val = image.max()

        air_voxels = np.sum((image >= -1050) & (image <= -950))
        results["likely_air_present"] = bool(air_voxels > (image.size * 0.001))

        if max_val > 200 and max_val <= 4096:
            results["likely_bone_present"] = True

        if min_val >= -2048 and max_val <= 4096:
            results["is_hu"] = True
        else:
            results["warnings"].append(
                f"Values outside typical HU range: [{min_val:.1f}, {max_val:.1f}]"
            )

        if metadata and hasattr(metadata, "rescale_slope"):
            if metadata.rescale_slope != 1.0 or metadata.rescale_intercept != 0.0:
                if min_val >= 0 and max_val > 4096:
                    results["warnings"].append(
                        f"Values may be raw pixel data (all positive, max={max_val:.1f}). "
                        f"Rescale params: slope={metadata.rescale_slope}, "
                        f"intercept={metadata.rescale_intercept}"
                    )

        return results

    def apply_window(
        self, image: np.ndarray, window: Union[float, str], level: Optional[float] = None
    ) -> np.ndarray:
        """Apply window/level adjustment.

        Args:
            image: Input image.
            window: Window width or preset name.
            level: Window center/level (required if window is numeric).

        Returns:
            Windowed image.
        """
        return self._windower.adjust(image, window=window, level=level)

    def normalize_intensity(
        self, image: np.ndarray, method: str = "zscore", **kwargs
    ) -> np.ndarray:
        """Normalize image intensities.

        Args:
            image: Input image.
            method: Normalization method ('zscore', 'minmax', 'percentile', 'histogram').
            **kwargs: Method-specific parameters.

        Returns:
            Normalized image.
        """
        if method == "z_score":
            method = "zscore"
        normalizer = IntensityNormalizer(method=method)
        return normalizer.normalize(image, **kwargs)

    def calculate_suv(
        self,
        image: np.ndarray,
        patient_weight: float,
        injected_dose: float,
        injection_time: Optional[str] = None,
    ) -> np.ndarray:
        """Calculate Standardized Uptake Value (SUV) from PET image.

        Args:
            image: PET image pixel values.
            patient_weight: Patient weight in kg.
            injected_dose: Injected dose in mCi.
            injection_time: Injection time (reserved for decay correction).

        Returns:
            SUV image.
        """
        # Convert mCi to Bq (1 mCi = 3.7e7 Bq)
        dose_bq = injected_dose * 3.7e7
        suv = image.astype(np.float64) * (patient_weight * 1000) / dose_bq
        return suv

    # ================================================================
    # Pipeline
    # ================================================================

    def preprocess(
        self,
        image: np.ndarray,
        metadata: ImageMetadata,
        denoise: bool = True,
        reduce_artifacts: bool = False,
        resample_spacing: Optional[Tuple[float, float, float]] = None,
        normalize: bool = True,
        window: Optional[str] = None,
    ) -> np.ndarray:
        """Apply full preprocessing pipeline based on modality.

        Args:
            image: Input image array.
            metadata: Image metadata.
            denoise: Apply denoising.
            reduce_artifacts: Apply artifact reduction.
            resample_spacing: Target voxel spacing for resampling.
            normalize: Apply normalization.
            window: Window preset name (for CT).

        Returns:
            Preprocessed image.
        """
        if metadata.is_ct():
            result = preprocess_ct(
                image,
                denoise=denoise,
                normalize=normalize,
                window=window or "lung",
                reduce_artifacts=reduce_artifacts,
            )
        elif metadata.is_mri():
            result = preprocess_mri(
                image, denoise=denoise, bias_correction=True, normalize=normalize
            )
        elif metadata.is_pet():
            result = preprocess_pet(image, denoise=denoise, normalize=normalize)
        else:
            result = image.copy()
            if denoise:
                denoiser = Denoiser(method="bilateral")
                result = denoiser.denoise(result)
            if normalize:
                normalizer = IntensityNormalizer(method="minmax")
                result = normalizer.normalize(result)

        if resample_spacing:
            result = self.resample(result, metadata, resample_spacing)

        return result

    # ================================================================
    # Embeddings
    # ================================================================

    def generate_embeddings(
        self,
        image: np.ndarray,
        mode: str = "2d",
        metadata: Optional[ImageMetadata] = None,
        window: Optional[str] = None,
        n_slices: Optional[int] = None,
    ) -> np.ndarray:
        """Generate embeddings from medical image.

        Args:
            image: Input image (2D or 3D).
            mode: Processing mode ('2d' or '3d').
            metadata: Image metadata for auto-detecting window.
            window: Override window preset.
            n_slices: Number of slices for 3D.

        Returns:
            Embedding vector.
        """
        # Lazy load model via registry
        if self.embedding_model is None:
            from ...models.registry import load_model

            self.embedding_model = load_model(
                model=self.model_name,
                model_path=self.model_path,
                device=self.device,
                **self._model_kwargs,
            )

        # Prepare images for model
        rgb_images = self.prepare_for_model(
            image, metadata=metadata, window=window, n_slices=n_slices
        )
        embeddings = self.embedding_model.generate_embeddings(rgb_images)

        if len(embeddings.shape) > 1 and embeddings.shape[0] > 1:
            embeddings = self.aggregate_embeddings(embeddings, method="mean")
        elif len(embeddings.shape) > 1:
            embeddings = embeddings[0]

        return embeddings

    def aggregate_embeddings(self, embeddings: np.ndarray, method: str = "mean") -> np.ndarray:
        """Aggregate slice-level embeddings to volume-level.

        Args:
            embeddings: Array of shape (num_slices, embedding_dim).
            method: Aggregation method ('mean', 'max', 'median', 'std', 'concat').

        Returns:
            Aggregated embedding vector.
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings to aggregate")

        if method == "mean":
            return np.mean(embeddings, axis=0)
        elif method == "max":
            return np.max(embeddings, axis=0)
        elif method == "median":
            return np.median(embeddings, axis=0)
        elif method == "std":
            return np.std(embeddings, axis=0)
        elif method == "concat":
            mean_emb = np.mean(embeddings, axis=0)
            std_emb = np.std(embeddings, axis=0)
            return np.concatenate([mean_emb, std_emb])
        else:
            raise ValueError(
                f"Unknown aggregation method: {method}. "
                "Supported: mean, max, median, std, concat"
            )

    # ================================================================
    # Utilities
    # ================================================================

    def prepare_for_model(
        self,
        image: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        window: Optional[str] = None,
        n_slices: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Prepare medical image for embedding model input.

        Args:
            image: Input image (2D or 3D volume).
            metadata: Image metadata for auto-detecting window preset.
            window: Override window preset.
            n_slices: Number of slices from 3D volume. None uses middle slice.

        Returns:
            List of RGB uint8 images (H, W, 3).
        """
        if window is None and metadata is not None:
            window = self._detect_window_preset(metadata)

        if window is not None:
            windowed = self._windower.adjust(image, window=window, output_range=(0, 255))
        else:
            img_min, img_max = float(image.min()), float(image.max())
            if img_max - img_min > 0:
                windowed = (image - img_min) / (img_max - img_min) * 255
            else:
                windowed = np.zeros_like(image, dtype=np.float64)

        windowed = np.clip(windowed, 0, 255).astype(np.uint8)

        if windowed.ndim == 3:
            if n_slices is not None:
                indices = np.linspace(0, windowed.shape[0] - 1, n_slices, dtype=int)
                slices = [windowed[i] for i in indices]
            else:
                slices = [windowed[windowed.shape[0] // 2]]
        else:
            slices = [windowed]

        rgb_images = []
        for sl in slices:
            if sl.ndim == 2:
                rgb = np.stack([sl, sl, sl], axis=-1)
            else:
                rgb = sl
            rgb_images.append(rgb)

        return rgb_images

    def process_batch(self, images: List[np.ndarray], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Process batch of images.

        Args:
            images: List of images.
            batch_size: Batch size for processing.
            **kwargs: Additional arguments for generate_embeddings.

        Returns:
            Batch embeddings array.
        """
        embeddings = []
        for img in images:
            emb = self.generate_embeddings(img, **kwargs)
            embeddings.append(emb)
        return np.stack(embeddings)

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the configured embedding model.

        Returns:
            Dict with model_name, embedding_dim, device, is_loaded.
        """
        try:
            from ...models.registry import _PRESET_REGISTRY

            config = _PRESET_REGISTRY.get(self.model_name)
        except ImportError:
            config = None

        return {
            "model_name": self.model_name,
            "embedding_dim": (
                config.embedding_dim
                if config
                else getattr(self.embedding_model, "embedding_dim", None)
            ),
            "device": self.device,
            "is_loaded": self.embedding_model is not None,
        }

    # ================================================================
    # Private helpers
    # ================================================================

    def _get_nnunet(self) -> NNUNetSegmenter:
        """Lazy-load NNUNetSegmenter."""
        if self._nnunet_segmenter is None:
            self._nnunet_segmenter = NNUNetSegmenter(
                model_paths=self._segmentation_model_paths, device=self.device
            )
        return self._nnunet_segmenter

    @staticmethod
    def _detect_window_preset(metadata: ImageMetadata) -> str:
        """Auto-detect optimal window preset from metadata series description."""
        desc = (metadata.series_description or "").upper()

        if any(kw in desc for kw in ("LUNG", "CHEST", "THORAX", "PULM")):
            return "lung"
        if any(kw in desc for kw in ("ABD", "PELVIS", "LIVER", "ABDOMEN")):
            return "abdomen"
        if any(kw in desc for kw in ("BRAIN", "HEAD", "NEURO", "CRANIAL")):
            return "brain"
        if any(kw in desc for kw in ("BONE", "SPINE", "SKELETAL", "MSK")):
            return "bone"
        if "CTA" in desc or "ANGIO" in desc:
            return "cta"
        return "soft_tissue"

    @staticmethod
    def _orientation_to_direction(orientation: List[float]) -> Tuple[float, ...]:
        """Convert DICOM orientation to SimpleITK direction matrix."""
        if len(orientation) == 6:
            row_x, row_y, row_z = orientation[0:3]
            col_x, col_y, col_z = orientation[3:6]
            slice_x = row_y * col_z - row_z * col_y
            slice_y = row_z * col_x - row_x * col_z
            slice_z = row_x * col_y - row_y * col_x
            return (row_x, row_y, row_z, col_x, col_y, col_z, slice_x, slice_y, slice_z)
        else:
            return (1, 0, 0, 0, 1, 0, 0, 0, 1)

    @staticmethod
    def _sitk_to_metadata(sitk_image, path: Path) -> ImageMetadata:
        """Extract ImageMetadata from a SimpleITK image."""
        spacing = sitk_image.GetSpacing()

        # Try to extract DICOM tags
        modality = ""
        patient_id = ""
        study_date = ""
        series_desc = ""
        window_center = None
        window_width = None
        rescale_intercept = 0.0
        rescale_slope = 1.0
        manufacturer = None
        scanner_model = None

        for key in sitk_image.GetMetaDataKeys():
            val = sitk_image.GetMetaData(key)
            if key == "0008|0060":
                modality = val.strip()
            elif key == "0010|0020":
                patient_id = val.strip()
            elif key == "0008|0020":
                study_date = val.strip()
            elif key == "0008|103e":
                series_desc = val.strip()
            elif key == "0028|1050":
                try:
                    window_center = float(val.split("\\")[0])
                except (ValueError, IndexError):
                    pass
            elif key == "0028|1051":
                try:
                    window_width = float(val.split("\\")[0])
                except (ValueError, IndexError):
                    pass
            elif key == "0028|1052":
                try:
                    rescale_intercept = float(val)
                except ValueError:
                    pass
            elif key == "0028|1053":
                try:
                    rescale_slope = float(val)
                except ValueError:
                    pass
            elif key == "0008|0070":
                manufacturer = val.strip()
            elif key == "0008|1090":
                scanner_model = val.strip()

        size = sitk_image.GetSize()
        return ImageMetadata(
            modality=modality or "unknown",
            patient_id=patient_id,
            study_date=study_date,
            series_description=series_desc or str(path.stem),
            pixel_spacing=(spacing[2], spacing[1], spacing[0]) if len(spacing) == 3 else spacing,
            image_position=tuple(sitk_image.GetOrigin()),
            image_orientation=list(sitk_image.GetDirection()),
            window_center=window_center,
            window_width=window_width,
            rescale_intercept=rescale_intercept,
            rescale_slope=rescale_slope,
            manufacturer=manufacturer,
            scanner_model=scanner_model,
            rows=size[1] if len(size) >= 2 else None,
            columns=size[0] if len(size) >= 1 else None,
            number_of_slices=size[2] if len(size) >= 3 else 1,
        )
