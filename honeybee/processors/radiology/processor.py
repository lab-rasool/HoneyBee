"""
Radiology Processor

Main processor for radiological imaging data using modular components.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology

# Import modular components
from ...loaders.Radiology import DicomLoader, ImageMetadata, NiftiLoader, load_medical_image
from ...models import REMEDIS, RadImageNet
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RadiologyProcessor:
    """
    Streamlined processor for radiological imaging data using modular components.

    This processor combines:
    - Unified data loading (DICOM/NIfTI)
    - Modular preprocessing components
    - Advanced segmentation algorithms
    - Embedding generation with multiple models
    """

    def __init__(
        self,
        model: str = "radimagenet",
        model_name: str = "DenseNet121",
        device: Optional[str] = None,
        use_hub: bool = True,
        extract_features: bool = False,
        segmentation_model_paths: Optional[Dict[str, str]] = None,
    ):
        """Initialize the RadiologyProcessor.

        Args:
            model: Embedding model to use ('remedis' or 'radimagenet')
            model_name: Specific model name for RadImageNet
            device: Device for computation ('cuda' or 'cpu')
            use_hub: Whether to use HuggingFace Hub for models
            extract_features: Enable intermediate feature extraction
            segmentation_model_paths: Mapping of task names to nnU-Net model folder
                paths (e.g. ``{"lung": "/path/to/lung_model/"}``)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model.lower()

        # Initialize data loaders
        self.dicom_loader = DicomLoader(lazy_load=True)
        self.nifti_loader = NiftiLoader()

        # Initialize preprocessing components
        self.denoiser = None
        self.normalizer = None
        self.windower = WindowLevelAdjuster()
        self.artifact_reducer = ArtifactReducer()

        # Initialize segmentation components
        self.segmenter = NNUNetSegmenter(
            model_paths=segmentation_model_paths, device=self.device
        )
        self.pet_segmenter = PETSegmenter()

        # Initialize embedding model
        self._initialize_model(model, model_name, use_hub, extract_features)

        logger.info(f"RadiologyProcessor initialized with {model} model on {self.device}")

    def _initialize_model(self, model: str, model_name: str, use_hub: bool, extract_features: bool):
        """Initialize the embedding model."""
        self._registry_model = False

        if self.model_type == "remedis":
            self.model = REMEDIS()
        elif self.model_type == "radimagenet":
            self.model = RadImageNet(
                model_name=model_name, use_hub=use_hub, extract_features=extract_features
            )
        else:
            # Try loading from model registry
            try:
                from ...models.registry import _PRESET_REGISTRY, load_model

                if model.lower() in _PRESET_REGISTRY:
                    self.model = load_model(model.lower(), device=self.device)
                    self._registry_model = True
                    logger.info(f"Loaded '{model}' from model registry")
                else:
                    raise ValueError(f"Unknown model: {model}")
            except ImportError:
                raise ValueError(f"Unknown model: {model}")

    def load_image(self, path: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load medical image from file.

        Args:
            path: Path to image file (DICOM or NIfTI)

        Returns:
            Tuple of (image array, metadata)
        """
        return load_medical_image(path)

    def preprocess(
        self,
        image: np.ndarray,
        metadata: ImageMetadata,
        denoise: bool = True,
        normalize: bool = True,
        window: Optional[str] = None,
        reduce_artifacts: bool = False,
        resample_spacing: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """Apply preprocessing pipeline based on modality.

        Args:
            image: Input image array
            metadata: Image metadata
            denoise: Apply denoising
            normalize: Apply normalization
            window: Window preset name (for CT)
            reduce_artifacts: Apply artifact reduction
            resample_spacing: Target voxel spacing for resampling

        Returns:
            Preprocessed image
        """
        # Apply modality-specific preprocessing
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
            # Generic preprocessing
            result = image.copy()

            if denoise:
                denoiser = Denoiser(method="bilateral")
                result = denoiser.denoise(result)

            if normalize:
                normalizer = IntensityNormalizer(method="minmax")
                result = normalizer.normalize(result)

        # Resample if requested
        if resample_spacing:
            result = self.resample(result, metadata, resample_spacing)

        return result

    def resample(
        self,
        image: np.ndarray,
        metadata: ImageMetadata,
        new_spacing: Tuple[float, float, float],
        interpolation: str = "linear",
    ) -> np.ndarray:
        """Resample image to new spacing.

        Args:
            image: Input image
            metadata: Image metadata with current spacing
            new_spacing: Target spacing (x, y, z)
            interpolation: Interpolation method

        Returns:
            Resampled image
        """
        # Convert to SimpleITK image
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing(metadata.pixel_spacing[::-1])  # SimpleITK uses xyz order

        # Calculate new size
        original_size = sitk_image.GetSize()
        original_spacing = sitk_image.GetSpacing()

        new_size = [
            int(round(osz * osp / nsp))
            for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
        ]

        # Set up resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())

        # Set interpolation
        if interpolation == "linear":
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interpolation == "nearest":
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif interpolation == "bspline":
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)

        # Resample
        resampled = resampler.Execute(sitk_image)

        return sitk.GetArrayFromImage(resampled)

    def segment_lungs(
        self, ct_image: np.ndarray, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """Segment lungs from CT scan using nnU-Net.

        Args:
            ct_image: CT scan in HU values
            spacing: Voxel spacing (z, y, x) in mm

        Returns:
            Binary mask of lung regions (uint8)
        """
        return self.segmenter.segment_lungs(ct_image, spacing=spacing)

    def segment_brain(
        self, mri_image: np.ndarray, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """Segment brain from MRI scan using nnU-Net.

        Args:
            mri_image: MRI scan
            spacing: Voxel spacing (z, y, x) in mm

        Returns:
            Binary mask of brain region (uint8)
        """
        return self.segmenter.extract_brain(mri_image, spacing=spacing)

    def prepare_for_model(
        self,
        image: np.ndarray,
        metadata: Optional[ImageMetadata] = None,
        window: Optional[str] = None,
        n_slices: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Prepare medical image for embedding model input.

        Auto-detects optimal window from metadata, applies windowing to [0, 255],
        and converts to 3-channel RGB. Handles both 2D and 3D inputs.

        Args:
            image: Input image (2D or 3D volume)
            metadata: Image metadata for auto-detecting window preset
            window: Override window preset ('lung', 'abdomen', 'brain', 'bone',
                    'soft_tissue'). If None, auto-detected from metadata.
            n_slices: Number of evenly-spaced slices to extract from 3D volume.
                     If None, uses middle slice for 3D input.

        Returns:
            List of RGB uint8 images (H, W, 3), ready for any model
        """
        # Auto-detect window from metadata if not specified
        if window is None and metadata is not None:
            window = self._detect_window_preset(metadata)

        # Apply windowing to [0, 255]
        if window is not None:
            windowed = self.windower.adjust(image, window=window, output_range=(0, 255))
        else:
            img_min, img_max = float(image.min()), float(image.max())
            if img_max - img_min > 0:
                windowed = (image - img_min) / (img_max - img_min) * 255
            else:
                windowed = np.zeros_like(image, dtype=np.float64)

        windowed = np.clip(windowed, 0, 255).astype(np.uint8)

        # Extract slices from 3D volume
        if windowed.ndim == 3:
            if n_slices is not None:
                indices = np.linspace(0, windowed.shape[0] - 1, n_slices, dtype=int)
                slices = [windowed[i] for i in indices]
            else:
                slices = [windowed[windowed.shape[0] // 2]]
        else:
            slices = [windowed]

        # Convert each slice to 3-channel RGB
        rgb_images = []
        for sl in slices:
            if sl.ndim == 2:
                rgb = np.stack([sl, sl, sl], axis=-1)
            else:
                rgb = sl
            rgb_images.append(rgb)

        return rgb_images

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

    def generate_embeddings(
        self,
        image: np.ndarray,
        mode: str = "2d",
        aggregation: str = "mean",
        preprocess: bool = True,
        metadata: Optional[ImageMetadata] = None,
        window: Optional[str] = None,
        n_slices: Optional[int] = None,
    ) -> np.ndarray:
        """Generate embeddings from medical image.

        Args:
            image: Input image (2D or 3D)
            mode: Processing mode ('2d' or '3d')
            aggregation: How to aggregate 3D embeddings
            preprocess: Whether to apply preprocessing
            metadata: Image metadata for preprocessing
            window: Window preset for prepare_for_model (used with registry models)
            n_slices: Number of slices for 3D (used with registry models)

        Returns:
            Embedding vector
        """
        # Registry models use prepare_for_model for image preparation
        if self._registry_model:
            rgb_images = self.prepare_for_model(
                image, metadata=metadata, window=window, n_slices=n_slices
            )
            embeddings = self.model.generate_embeddings(rgb_images)
            if len(embeddings.shape) > 1 and embeddings.shape[0] > 1:
                embeddings = self.aggregate_embeddings(embeddings, method=aggregation)
            elif len(embeddings.shape) > 1:
                embeddings = embeddings[0]
            return embeddings

        # Preprocess if requested
        if preprocess and metadata:
            image = self.preprocess(image, metadata)

        # Generate embeddings based on model
        if self.model_type == "remedis":
            if len(image.shape) == 3:
                middle = image.shape[0] // 2
                slice_2d = image[middle]
            else:
                slice_2d = image

            slice_norm = (
                (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255
            ).astype(np.uint8)

            slice_rgb = np.stack([slice_norm] * 3, axis=-1)
            embeddings = self.model.predict(slice_rgb[np.newaxis, ...])[0]

        elif self.model_type == "radimagenet":
            embeddings = self.model.generate_embeddings(image, mode=mode, aggregation=aggregation)

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            if len(embeddings.shape) > 1:
                embeddings = embeddings.flatten()

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return embeddings

    def process_batch(self, images: List[np.ndarray], batch_size: int = 32, **kwargs) -> np.ndarray:
        """Process batch of images efficiently.

        Args:
            images: List of images
            batch_size: Batch size for processing
            **kwargs: Additional arguments for generate_embeddings

        Returns:
            Batch embeddings array
        """
        if self.model_type == "radimagenet":
            # Use RadImageNet's batch processing
            return self.model.process_batch(images, batch_size)
        else:
            # Process one by one for other models
            embeddings = []
            for img in images:
                emb = self.generate_embeddings(img, **kwargs)
                embeddings.append(emb)
            return np.stack(embeddings)

    def extract_features(
        self, image: np.ndarray, layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Extract features from multiple layers.

        Args:
            image: Input image
            layer_names: Specific layers to extract from

        Returns:
            Dictionary mapping layer names to features
        """
        if self.model_type == "radimagenet" and hasattr(self.model, "extract_features"):
            # Set up feature extraction if not already enabled
            if not self.model.extract_features:
                self.model.extract_features = True
                self.model._setup_feature_extraction()

            # Generate embeddings with feature extraction
            result = self.model.generate_embeddings(image, return_features=True)

            if isinstance(result, dict):
                return result["features"]
            else:
                return {}
        else:
            logger.warning(f"Feature extraction not supported for {self.model_type}")
            return {}

    # ========== Data Loading Methods ==========

    def load_dicom(self, path: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load DICOM file or series.

        Args:
            path: Path to DICOM file or directory containing series

        Returns:
            Tuple of (image array, metadata)
        """
        path = Path(path)
        if path.is_dir():
            return self.dicom_loader.load_series(path)
        else:
            return self.dicom_loader.load_file(path)

    def load_nifti(self, path: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load NIfTI file.

        Args:
            path: Path to NIfTI file (.nii or .nii.gz)

        Returns:
            Tuple of (image array, metadata)
        """
        return self.nifti_loader.load_file(path)

    # ========== Preprocessing Methods (Public API) ==========

    def denoise(self, image: np.ndarray, method: str = "nlm", **kwargs) -> np.ndarray:
        """Apply denoising to medical image.

        Args:
            image: Input image
            method: Denoising method ('nlm', 'tv', 'bilateral', 'median', 'gaussian', 'rician', 'deep')
            **kwargs: Method-specific parameters

        Returns:
            Denoised image
        """
        # All methods including 'deep' and 'rician' are now supported by Denoiser
        denoiser = Denoiser(method=method)
        return denoiser.denoise(image, **kwargs)

    def reduce_metal_artifacts(self, ct_image: np.ndarray, threshold: float = 3000) -> np.ndarray:
        """Reduce metal artifacts in CT images.

        Args:
            ct_image: CT scan in HU values
            threshold: HU threshold for metal detection

        Returns:
            CT image with reduced metal artifacts
        """
        return self.artifact_reducer.reduce_artifacts(
            ct_image, artifact_type="metal", threshold=threshold
        )

    def apply_window(
        self, image: np.ndarray, window: Union[float, str], level: Optional[float] = None
    ) -> np.ndarray:
        """Apply window/level adjustment to image.

        Args:
            image: Input image
            window: Window width or preset name ('lung', 'bone', 'soft_tissue', etc.)
            level: Window center/level (required if window is numeric)

        Returns:
            Windowed image
        """
        return self.windower.adjust(image, window=window, level=level)

    def normalize_intensity(
        self, image: np.ndarray, method: str = "z_score", **kwargs
    ) -> np.ndarray:
        """Normalize image intensities.

        Args:
            image: Input image
            method: Normalization method ('z_score', 'minmax', 'percentile', 'histogram')
            **kwargs: Method-specific parameters

        Returns:
            Normalized image
        """
        # Handle both z_score and zscore
        if method == "z_score":
            method = "zscore"

        normalizer = IntensityNormalizer(method=method)
        return normalizer.normalize(image, **kwargs)

    def verify_hounsfield_units(
        self, ct_image: np.ndarray, metadata: Optional[ImageMetadata] = None
    ) -> Dict[str, Any]:
        """Verify that CT image is in Hounsfield Units.

        Args:
            ct_image: CT image array
            metadata: Image metadata (optional)

        Returns:
            Dictionary with verification results and statistics
        """
        results = {
            "is_hu": False,
            "min_value": float(ct_image.min()),
            "max_value": float(ct_image.max()),
            "mean_value": float(ct_image.mean()),
            "likely_air_present": False,
            "likely_bone_present": False,
            "warnings": [],
        }

        # Check if values are in typical HU range
        min_val = ct_image.min()
        max_val = ct_image.max()

        # Air is around -1000 HU — check if a meaningful fraction of voxels are in the air range
        air_voxels = np.sum((ct_image >= -1050) & (ct_image <= -950))
        results["likely_air_present"] = bool(air_voxels > (ct_image.size * 0.001))  # >0.1%

        if max_val > 200 and max_val <= 4096:
            results["likely_bone_present"] = True

        # Check if image appears to be in HU (wide range for GE padding and dense bone)
        if min_val >= -2048 and max_val <= 4096:
            results["is_hu"] = True
        else:
            results["warnings"].append(
                f"Values outside typical HU range: [{min_val:.1f}, {max_val:.1f}]"
            )

        # Check metadata if available — only warn when data looks like raw pixel data
        if metadata and hasattr(metadata, "rescale_slope"):
            if metadata.rescale_slope != 1.0 or metadata.rescale_intercept != 0.0:
                # Only warn if data looks like it hasn't been rescaled yet
                if min_val >= 0 and max_val > 4096:
                    results["warnings"].append(
                        f"Values may be raw pixel data (all positive, max={max_val:.1f}). "
                        f"Rescale params: slope={metadata.rescale_slope}, "
                        f"intercept={metadata.rescale_intercept}"
                    )

        return results

    def reorient(
        self, image: np.ndarray, metadata: ImageMetadata, target_orientation: str = "RAS"
    ) -> Tuple[np.ndarray, ImageMetadata]:
        """Reorient medical image to standard orientation.

        Args:
            image: Input image
            metadata: Image metadata with orientation information
            target_orientation: Target orientation code (e.g., 'RAS', 'LPS')

        Returns:
            Tuple of (reoriented image, updated metadata)
        """
        # Convert to SimpleITK for reorientation
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing(metadata.pixel_spacing[::-1])
        sitk_image.SetOrigin(metadata.image_position)

        # Set direction from orientation
        direction = self._orientation_to_direction(metadata.image_orientation)
        sitk_image.SetDirection(direction)

        # Reorient to target
        reoriented = sitk.DICOMOrient(sitk_image, target_orientation)

        # Convert back to numpy
        reoriented_array = sitk.GetArrayFromImage(reoriented)

        # Update metadata
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
            number_of_slices=reoriented_array.shape[0] if len(reoriented_array.shape) > 2 else 1,
            extra_metadata=metadata.extra_metadata,
        )

        return reoriented_array, new_metadata

    def _orientation_to_direction(self, orientation: List[float]) -> Tuple[float, ...]:
        """Convert DICOM orientation to SimpleITK direction matrix."""
        if len(orientation) == 6:
            # DICOM orientation has row and column direction cosines
            # Need to compute the third direction (cross product)
            row_x, row_y, row_z = orientation[0:3]
            col_x, col_y, col_z = orientation[3:6]

            # Cross product for slice direction
            slice_x = row_y * col_z - row_z * col_y
            slice_y = row_z * col_x - row_x * col_z
            slice_z = row_x * col_y - row_y * col_x

            return (row_x, row_y, row_z, col_x, col_y, col_z, slice_x, slice_y, slice_z)
        else:
            # Default to identity
            return (1, 0, 0, 0, 1, 0, 0, 0, 1)

    def register(
        self, moving_image: np.ndarray, fixed_image: np.ndarray, method: str = "rigid"
    ) -> np.ndarray:
        """Register moving image to fixed image.

        Args:
            moving_image: Image to be transformed
            fixed_image: Reference image
            method: Registration method ('rigid', 'affine', 'deformable')

        Returns:
            Registered moving image
        """
        # Convert to SimpleITK
        fixed_sitk = sitk.GetImageFromArray(fixed_image)
        moving_sitk = sitk.GetImageFromArray(moving_image)

        # Initialize registration
        registration_method = sitk.ImageRegistrationMethod()

        # Set metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        # Set interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Set optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Set initial transform
        if method == "rigid":
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.Euler3DTransform() if len(fixed_image.shape) == 3 else sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        elif method == "affine":
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.AffineTransform(3 if len(fixed_image.shape) == 3 else 2),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        elif method in ("syn", "antspy_rigid", "antspy_affine"):
            return self._register_antspy(moving_image, fixed_image, method)
        else:
            logger.warning(f"Method {method} not implemented, using rigid registration")
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk,
                moving_sitk,
                sitk.Euler3DTransform() if len(fixed_image.shape) == 3 else sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )

        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Execute registration
        try:
            final_transform = registration_method.Execute(fixed_sitk, moving_sitk)

            # Apply transform
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(final_transform)

            registered_sitk = resampler.Execute(moving_sitk)
            registered = sitk.GetArrayFromImage(registered_sitk)

            return registered

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            logger.warning("Returning original moving image")
            return moving_image

    def _register_antspy(
        self, moving_image: np.ndarray, fixed_image: np.ndarray, method: str
    ) -> np.ndarray:
        """Register using ANTsPy.

        Falls back to SimpleITK registration if ANTsPy is not installed.
        """
        try:
            import ants

            fixed_ants = ants.from_numpy(fixed_image.astype(np.float32))
            moving_ants = ants.from_numpy(moving_image.astype(np.float32))

            type_map = {
                "antspy_rigid": "Rigid",
                "antspy_affine": "Affine",
                "syn": "SyN",
            }
            reg_type = type_map.get(method, "SyN")

            result = ants.registration(
                fixed=fixed_ants,
                moving=moving_ants,
                type_of_transform=reg_type,
            )

            return result["warpedmovout"].numpy()
        except ImportError:
            logger.warning(f"ANTsPy not available for '{method}', falling back to SimpleITK rigid")
            return self.register(moving_image, fixed_image, method="rigid")

    # ========== Segmentation Methods ==========

    def segment_organs(
        self,
        ct_image: np.ndarray,
        organs: Optional[List[str]] = None,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> Dict[str, np.ndarray]:
        """Segment multiple organs from CT image using nnU-Net.

        Args:
            ct_image: CT image in HU values
            organs: List of organs to segment (None returns all)
            spacing: Voxel spacing (z, y, x) in mm

        Returns:
            Dictionary mapping organ names to binary masks
        """
        return self.segmenter.segment_organs(ct_image, spacing=spacing, organs=organs)

    def segment_tumor(
        self,
        image: np.ndarray,
        metadata: ImageMetadata,
        seed_point: Optional[Tuple[int, ...]] = None,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        task: str = "brain_tumor",
        **kwargs,
    ) -> np.ndarray:
        """Segment tumor using nnU-Net.

        Args:
            image: Medical image (CT/MRI)
            metadata: Image metadata
            seed_point: Deprecated. Ignored if provided.
            spacing: Voxel spacing (z, y, x) in mm
            task: nnU-Net task name (default "brain_tumor")
            **kwargs: Reserved for future use

        Returns:
            Binary tumor mask
        """
        import warnings

        if seed_point is not None:
            warnings.warn(
                "seed_point is deprecated and ignored. NNUNetSegmenter does not "
                "require a seed point. This parameter will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.segmenter.segment_tumor(image, spacing=spacing, task=task)

    def segment_metabolic_volume(
        self, pet_image: np.ndarray, threshold: float = 2.5, method: str = "fixed"
    ) -> np.ndarray:
        """Segment metabolically active regions in PET image.

        Args:
            pet_image: PET image (preferably in SUV units)
            threshold: SUV threshold for segmentation
            method: Segmentation method ('fixed', 'adaptive', 'gradient')

        Returns:
            Binary mask of metabolically active regions
        """
        return self.pet_segmenter.segment_metabolic_volume(
            pet_image, method=method, threshold=threshold
        )

    # ========== Additional Public API Methods ==========

    def crop_to_roi(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop image to the bounding box of a binary mask.

        Args:
            image: Input image (2D or 3D)
            mask: Binary mask defining the ROI

        Returns:
            Cropped image containing only the ROI region
        """
        # Apply mask first
        masked = image * mask

        # Find bounding box of the mask
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            logger.warning("Empty mask provided, returning original image")
            return image

        slices = tuple(slice(c.min(), c.max() + 1) for c in coords)
        return masked[slices]

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply binary mask to image (zero outside mask).

        Args:
            image: Input image
            mask: Binary mask (same shape as image)

        Returns:
            Masked image with zeros outside the mask region
        """
        return image * mask

    def correct_bias_field(self, image: np.ndarray, backend: str = "sitk") -> np.ndarray:
        """Apply N4 bias field correction to MRI image.

        Args:
            image: Input MRI image
            backend: Backend to use ('sitk' or 'ants')

        Returns:
            Bias-corrected image
        """
        if backend == "ants":
            try:
                import ants

                ants_img = ants.from_numpy(image.astype(np.float32))
                corrected = ants.n4_bias_field_correction(ants_img)
                return corrected.numpy()
            except ImportError:
                logger.warning("ANTsPy not available, falling back to SimpleITK")

        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(sitk_image)
        return sitk.GetArrayFromImage(corrected)

    def calculate_suv(
        self,
        image: np.ndarray,
        patient_weight: float,
        injected_dose: float,
        injection_time: Optional[str] = None,
    ) -> np.ndarray:
        """Calculate Standardized Uptake Value (SUV) from PET image.

        Args:
            image: PET image pixel values
            patient_weight: Patient weight in kg
            injected_dose: Injected dose in mCi
            injection_time: Injection time in ISO format (reserved for decay correction)

        Returns:
            SUV image
        """
        # Convert injected dose from mCi to Bq (1 mCi = 3.7e7 Bq)
        dose_bq = injected_dose * 3.7e7

        # SUV = pixel_value * body_weight(g) / injected_dose(Bq)
        suv = image.astype(np.float64) * (patient_weight * 1000) / dose_bq

        return suv

    def aggregate_embeddings(
        self, embeddings: np.ndarray, method: str = "mean"
    ) -> np.ndarray:
        """Aggregate slice-level embeddings to volume-level.

        Args:
            embeddings: Array of shape (num_slices, embedding_dim)
            method: Aggregation method ('mean', 'max', 'concat')

        Returns:
            Aggregated embedding vector
        """
        if method == "mean":
            return embeddings.mean(axis=0)
        elif method == "max":
            result = embeddings.max(axis=0)
            # torch.Tensor.max returns a named tuple (values, indices);
            # numpy ndarray.max returns the array directly.
            return result.values if hasattr(result, "values") else result
        elif method == "concat":
            return embeddings.flatten()
        else:
            raise ValueError(f"Unknown aggregation method: {method}. Choose from: mean, max, concat")

    def load_atlas(self, path: Union[str, Path]) -> Tuple[np.ndarray, 'ImageMetadata']:
        """Load atlas image (alias for load_nifti).

        Args:
            path: Path to atlas NIfTI file

        Returns:
            Tuple of (atlas array, metadata)
        """
        return self.nifti_loader.load_file(path)
