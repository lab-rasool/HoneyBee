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