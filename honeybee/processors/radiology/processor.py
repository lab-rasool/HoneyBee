"""
Radiology Processor

Main processor for radiological imaging data using modular components.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from scipy import ndimage
from skimage import morphology, filters, segmentation, measure
from scipy.ndimage import binary_fill_holes
import cv2

# Import modular components
from ...loaders.Radiology import DicomLoader, NiftiLoader, ImageMetadata, load_medical_image
from .preprocessing import (
    Denoiser, IntensityNormalizer, WindowLevelAdjuster, ArtifactReducer,
    preprocess_ct, preprocess_mri, preprocess_pet
)
from .segmentation import CTSegmenter, MRISegmenter, PETSegmenter
from ...models import REMEDIS, RadImageNet

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
    ):
        """Initialize the RadiologyProcessor.
        
        Args:
            model: Embedding model to use ('remedis' or 'radimagenet')
            model_name: Specific model name for RadImageNet
            device: Device for computation ('cuda' or 'cpu')
            use_hub: Whether to use HuggingFace Hub for models
            extract_features: Enable intermediate feature extraction
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
        self.ct_segmenter = CTSegmenter()
        self.mri_segmenter = MRISegmenter()
        self.pet_segmenter = PETSegmenter()

        # Initialize embedding model
        self._initialize_model(model, model_name, use_hub, extract_features)

        logger.info(f"RadiologyProcessor initialized with {model} model on {self.device}")
    
    def _initialize_model(self, model: str, model_name: str, use_hub: bool, extract_features: bool):
        """Initialize the embedding model."""
        if self.model_type == "remedis":
            self.model = REMEDIS()
        elif self.model_type == "radimagenet":
            self.model = RadImageNet(
                model_name=model_name,
                use_hub=use_hub,
                extract_features=extract_features
            )
        else:
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
                window=window or 'lung',
                reduce_artifacts=reduce_artifacts
            )
        elif metadata.is_mri():
            result = preprocess_mri(
                image,
                denoise=denoise,
                bias_correction=True,
                normalize=normalize
            )
        elif metadata.is_pet():
            result = preprocess_pet(
                image,
                denoise=denoise,
                normalize=normalize
            )
        else:
            # Generic preprocessing
            result = image.copy()
            
            if denoise:
                denoiser = Denoiser(method='bilateral')
                result = denoiser.denoise(result)
            
            if normalize:
                normalizer = IntensityNormalizer(method='minmax')
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
        interpolation: str = 'linear'
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
        if interpolation == 'linear':
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interpolation == 'nearest':
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        elif interpolation == 'bspline':
            resampler.SetInterpolator(sitk.sitkBSpline)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
        
        # Resample
        resampled = resampler.Execute(sitk_image)
        
        return sitk.GetArrayFromImage(resampled)
    
    def segment_lungs(self, ct_image: np.ndarray) -> np.ndarray:
        """Segment lungs from CT scan.
        
        Args:
            ct_image: CT scan in HU values
            
        Returns:
            Binary mask of lung regions
        """
        # Threshold for air
        binary_image = ct_image < -400
        
        # Remove artifacts outside body
        labels = measure.label(binary_image)
        
        # Assume largest connected component is background
        props = measure.regionprops(labels)
        if props:
            background_label = max(props, key=lambda x: x.area).label
            binary_image[labels == background_label] = 0
        
        # Fill holes
        for i in range(binary_image.shape[0]):
            binary_image[i] = binary_fill_holes(binary_image[i])
        
        # Get two largest components (left and right lung)
        labels = measure.label(binary_image)
        props = measure.regionprops(labels)
        
        if len(props) >= 2:
            # Sort by area and get two largest
            props_sorted = sorted(props, key=lambda x: x.area, reverse=True)
            lung_labels = [props_sorted[0].label, props_sorted[1].label]
            
            # Create lung mask
            lung_mask = np.zeros_like(binary_image)
            for label in lung_labels:
                lung_mask[labels == label] = 1
        else:
            lung_mask = binary_image
        
        # Morphological operations to smooth
        lung_mask = morphology.binary_closing(lung_mask, morphology.ball(2))
        
        return lung_mask.astype(np.uint8)
    
    def segment_brain(self, mri_image: np.ndarray) -> np.ndarray:
        """Segment brain from MRI scan.
        
        Args:
            mri_image: MRI scan
            
        Returns:
            Binary mask of brain region
        """
        # Simple threshold-based approach
        # More sophisticated methods would use atlas-based or ML approaches
        
        # Otsu thresholding
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(mri_image)
        binary = mri_image > thresh
        
        # Remove small components
        min_size = np.prod(mri_image.shape) // 100
        binary = morphology.remove_small_objects(binary, min_size=min_size)
        
        # Fill holes
        for i in range(binary.shape[0]):
            binary[i] = binary_fill_holes(binary[i])
        
        # Get largest component
        labels = measure.label(binary)
        props = measure.regionprops(labels)
        
        if props:
            largest = max(props, key=lambda x: x.area)
            brain_mask = labels == largest.label
        else:
            brain_mask = binary
        
        return brain_mask.astype(np.uint8)
    
    def generate_embeddings(
        self,
        image: np.ndarray,
        mode: str = '2d',
        aggregation: str = 'mean',
        preprocess: bool = True,
        metadata: Optional[ImageMetadata] = None
    ) -> np.ndarray:
        """Generate embeddings from medical image.
        
        Args:
            image: Input image (2D or 3D)
            mode: Processing mode ('2d' or '3d')
            aggregation: How to aggregate 3D embeddings
            preprocess: Whether to apply preprocessing
            metadata: Image metadata for preprocessing
            
        Returns:
            Embedding vector
        """
        # Preprocess if requested
        if preprocess and metadata:
            image = self.preprocess(image, metadata)
        
        # Generate embeddings based on model
        if self.model_type == "remedis":
            # REMEDIS expects specific preprocessing
            if len(image.shape) == 3:
                # Process middle slice for 3D
                middle = image.shape[0] // 2
                slice_2d = image[middle]
            else:
                slice_2d = image
            
            # Normalize to 0-255
            slice_norm = ((slice_2d - slice_2d.min()) / 
                         (slice_2d.max() - slice_2d.min() + 1e-8) * 255).astype(np.uint8)
            
            # REMEDIS expects RGB
            slice_rgb = np.stack([slice_norm] * 3, axis=-1)
            embeddings = self.model.predict(slice_rgb[np.newaxis, ...])[0]
            
        elif self.model_type == "radimagenet":
            # Use enhanced RadImageNet features
            embeddings = self.model.generate_embeddings(
                image, mode=mode, aggregation=aggregation
            )
            
            # Convert tensor to numpy
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            # Flatten if needed
            if len(embeddings.shape) > 1:
                embeddings = embeddings.flatten()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return embeddings
    
    def process_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
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
        self,
        image: np.ndarray,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Extract features from multiple layers.

        Args:
            image: Input image
            layer_names: Specific layers to extract from

        Returns:
            Dictionary mapping layer names to features
        """
        if self.model_type == "radimagenet" and hasattr(self.model, 'extract_features'):
            # Set up feature extraction if not already enabled
            if not self.model.extract_features:
                self.model.extract_features = True
                self.model._setup_feature_extraction()

            # Generate embeddings with feature extraction
            result = self.model.generate_embeddings(
                image, return_features=True
            )

            if isinstance(result, dict):
                return result['features']
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

    def denoise(self, image: np.ndarray, method: str = 'nlm', **kwargs) -> np.ndarray:
        """Apply denoising to medical image.

        Args:
            image: Input image
            method: Denoising method ('nlm', 'tv', 'bilateral', 'median', 'gaussian', 'deep')
            **kwargs: Method-specific parameters

        Returns:
            Denoised image
        """
        if method == 'deep':
            logger.warning("Deep learning denoising not yet implemented, using NLM instead")
            method = 'nlm'

        denoiser = Denoiser(method=method)
        return denoiser.denoise(image, **kwargs)

    def reduce_metal_artifacts(self, ct_image: np.ndarray,
                              threshold: float = 3000) -> np.ndarray:
        """Reduce metal artifacts in CT images.

        Args:
            ct_image: CT scan in HU values
            threshold: HU threshold for metal detection

        Returns:
            CT image with reduced metal artifacts
        """
        return self.artifact_reducer.reduce_artifacts(
            ct_image, artifact_type='metal', threshold=threshold
        )

    def apply_window(self, image: np.ndarray,
                    window: Union[float, str],
                    level: Optional[float] = None) -> np.ndarray:
        """Apply window/level adjustment to image.

        Args:
            image: Input image
            window: Window width or preset name ('lung', 'bone', 'soft_tissue', etc.)
            level: Window center/level (required if window is numeric)

        Returns:
            Windowed image
        """
        return self.windower.adjust(image, window=window, level=level)

    def normalize_intensity(self, image: np.ndarray,
                          method: str = 'z_score',
                          **kwargs) -> np.ndarray:
        """Normalize image intensities.

        Args:
            image: Input image
            method: Normalization method ('z_score', 'minmax', 'percentile', 'histogram')
            **kwargs: Method-specific parameters

        Returns:
            Normalized image
        """
        # Handle both z_score and zscore
        if method == 'z_score':
            method = 'zscore'

        normalizer = IntensityNormalizer(method=method)
        return normalizer.normalize(image, **kwargs)

    def verify_hounsfield_units(self, ct_image: np.ndarray,
                               metadata: Optional[ImageMetadata] = None) -> Dict[str, Any]:
        """Verify that CT image is in Hounsfield Units.

        Args:
            ct_image: CT image array
            metadata: Image metadata (optional)

        Returns:
            Dictionary with verification results and statistics
        """
        results = {
            'is_hu': False,
            'min_value': float(ct_image.min()),
            'max_value': float(ct_image.max()),
            'mean_value': float(ct_image.mean()),
            'likely_air_present': False,
            'likely_bone_present': False,
            'warnings': []
        }

        # Check if values are in typical HU range
        min_val = ct_image.min()
        max_val = ct_image.max()

        # Air is around -1000 HU, bone is > 200 HU
        if min_val < -900 and min_val > -1100:
            results['likely_air_present'] = True

        if max_val > 200 and max_val < 4000:
            results['likely_bone_present'] = True

        # Check if image appears to be in HU
        if min_val >= -1100 and max_val <= 4000:
            results['is_hu'] = True
        else:
            results['warnings'].append(f"Values outside typical HU range: [{min_val:.1f}, {max_val:.1f}]")

        # Check metadata if available
        if metadata and hasattr(metadata, 'rescale_slope'):
            if metadata.rescale_slope != 1.0 or metadata.rescale_intercept != 0.0:
                results['warnings'].append(
                    f"Rescale parameters suggest raw pixel values: "
                    f"slope={metadata.rescale_slope}, intercept={metadata.rescale_intercept}"
                )

        return results

    def reorient(self, image: np.ndarray,
                metadata: ImageMetadata,
                target_orientation: str = 'RAS') -> Tuple[np.ndarray, ImageMetadata]:
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
            extra_metadata=metadata.extra_metadata
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

    def register(self, moving_image: np.ndarray,
                fixed_image: np.ndarray,
                method: str = 'rigid') -> np.ndarray:
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
            convergenceWindowSize=10
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Set initial transform
        if method == 'rigid':
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk, moving_sitk,
                sitk.Euler3DTransform() if len(fixed_image.shape) == 3 else sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        elif method == 'affine':
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk, moving_sitk,
                sitk.AffineTransform(3 if len(fixed_image.shape) == 3 else 2),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
        else:
            logger.warning(f"Method {method} not implemented, using rigid registration")
            initial_transform = sitk.CenteredTransformInitializer(
                fixed_sitk, moving_sitk,
                sitk.Euler3DTransform() if len(fixed_image.shape) == 3 else sitk.Euler2DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
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

    # ========== Segmentation Methods ==========

    def segment_organs(self, ct_image: np.ndarray,
                      organs: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Segment multiple organs from CT image.

        Args:
            ct_image: CT image in HU values
            organs: List of organs to segment (default: ['liver', 'spleen', 'kidney'])

        Returns:
            Dictionary mapping organ names to binary masks
        """
        return self.ct_segmenter.segment_organs(ct_image, organs=organs)

    def segment_tumor(self, image: np.ndarray,
                     metadata: ImageMetadata,
                     seed_point: Optional[Tuple[int, ...]] = None,
                     **kwargs) -> np.ndarray:
        """Segment tumor using region growing.

        Args:
            image: Medical image (CT/MRI/PET)
            metadata: Image metadata to determine modality
            seed_point: Starting point for region growing (required)
            **kwargs: Additional parameters for segmentation

        Returns:
            Binary tumor mask
        """
        if seed_point is None:
            raise ValueError("seed_point is required for tumor segmentation")

        if metadata.is_ct():
            return self.ct_segmenter.segment_tumor(image, seed_point, **kwargs)
        elif metadata.is_mri():
            return self.mri_segmenter.segment_tumor(image, seed_point, **kwargs)
        else:
            logger.warning(f"Tumor segmentation not optimized for modality {metadata.modality}")
            # Fallback to CT segmenter
            return self.ct_segmenter.segment_tumor(image, seed_point, **kwargs)

    def segment_metabolic_volume(self, pet_image: np.ndarray,
                                threshold: float = 2.5,
                                method: str = 'fixed') -> np.ndarray:
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