"""
Radiology Data Processing System

This module implements comprehensive processing capabilities for radiological imaging data,
including DICOM/NIfTI loading, preprocessing, segmentation, and embedding generation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import warnings

# External dependencies
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import Dataset as DicomDataset
import nibabel as nib
import torch
import torch.nn as nn
from scipy import ndimage
from skimage import morphology, filters, segmentation, measure
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle
from scipy.ndimage import binary_fill_holes
import cv2

# Import embedding models
from ..models import REMEDIS, RadImageNet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_FORMATS = [".dcm", ".dicom", ".nii", ".nii.gz", ".nrrd", ".mhd", ".mha"]
MODALITY_TAGS = {
    "CT": "1.2.840.10008.5.1.4.1.1.2",
    "MR": "1.2.840.10008.5.1.4.1.1.4",
    "PET": "1.2.840.10008.5.1.4.1.1.128",
}
WINDOW_PRESETS = {
    "lung": {"window": 1500, "level": -600},
    "abdomen": {"window": 350, "level": 50},
    "bone": {"window": 2000, "level": 400},
    "brain": {"window": 80, "level": 40},
    "soft_tissue": {"window": 400, "level": 50},
}

class RadiologyImage:
    """Container for radiological images with metadata."""
    
    def __init__(self, data: np.ndarray, metadata: Dict[str, Any] = None):
        self.data = data
        self.metadata = metadata or {}
        self.shape = data.shape
        self.dtype = data.dtype
        
    def __repr__(self):
        return f"RadiologyImage(shape={self.shape}, modality={self.metadata.get('modality', 'unknown')})"


class RadiologyProcessor:
    """
    Main processor for radiological imaging data.
    
    Supports:
    - DICOM and NIfTI file loading
    - Image preprocessing and normalization
    - Anatomical segmentation
    - Denoising and artifact reduction
    - Spatial standardization
    - Embedding generation
    """
    
    # Class attributes
    WINDOW_PRESETS = WINDOW_PRESETS
    
    def __init__(self, model: str = "remedis", device: str = None):
        """
        Initialize the RadiologyProcessor.
        
        Args:
            model: Embedding model to use ('remedis' or 'radimagenet')
            device: Device for computation ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model
        self._initialize_model(model)
        logger.info(f"RadiologyProcessor initialized with {model} model on {self.device}")
        
    def _initialize_model(self, model_name: str):
        """Initialize the embedding model."""
        self.model_type = model_name.lower()
        
        if self.model_type == "remedis":
            self.model = REMEDIS()
        elif self.model_type == "radimagenet":
            self.model = RadImageNet()
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
            
    def load_dicom(self, path: Union[str, Path]) -> RadiologyImage:
        """
        Load DICOM image or series.
        
        Args:
            path: Path to DICOM file or directory containing DICOM series
            
        Returns:
            RadiologyImage object with data and metadata
        """
        path = Path(path)
        
        if path.is_file():
            # Single DICOM file
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            image = reader.Execute()
            
            # Extract metadata
            metadata = self._extract_dicom_metadata(reader)
            
        elif path.is_dir():
            # DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            if not dicom_names:
                raise ValueError(f"No DICOM files found in {path}")
                
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            
            # Extract metadata from first file
            dcm = pydicom.dcmread(dicom_names[0])
            metadata = self._extract_pydicom_metadata(dcm)
            
        else:
            raise ValueError(f"Path {path} does not exist")
            
        # Convert to numpy array
        data = sitk.GetArrayFromImage(image)
        
        # Store spatial information
        metadata['spacing'] = image.GetSpacing()
        metadata['origin'] = image.GetOrigin()
        metadata['direction'] = image.GetDirection()
        
        return RadiologyImage(data, metadata)
        
    def load_nifti(self, path: Union[str, Path]) -> RadiologyImage:
        """
        Load NIfTI image.
        
        Args:
            path: Path to NIfTI file
            
        Returns:
            RadiologyImage object with data and metadata
        """
        path = Path(path)
        
        # Load with nibabel
        nifti = nib.load(str(path))
        data = nifti.get_fdata()
        
        # Extract metadata
        metadata = {
            'affine': nifti.affine,
            'header': dict(nifti.header),
            'spacing': nifti.header.get_zooms()[:3] if len(nifti.header.get_zooms()) >= 3 else nifti.header.get_zooms(),
            'shape': data.shape,
            'format': 'nifti'
        }
        
        return RadiologyImage(data, metadata)
        
    def _extract_dicom_metadata(self, reader: sitk.ImageFileReader) -> Dict[str, Any]:
        """Extract metadata from SimpleITK DICOM reader."""
        metadata = {}
        
        # Common DICOM tags
        tags = {
            'PatientID': '0010|0020',
            'StudyDate': '0008|0020',
            'Modality': '0008|0060',
            'StudyDescription': '0008|1030',
            'SeriesDescription': '0008|103e',
            'SliceThickness': '0018|0050',
            'PixelSpacing': '0028|0030',
        }
        
        for key, tag in tags.items():
            if reader.HasMetaDataKey(tag):
                metadata[key] = reader.GetMetaData(tag)
                
        return metadata
        
    def _extract_pydicom_metadata(self, dcm: DicomDataset) -> Dict[str, Any]:
        """Extract metadata from pydicom dataset."""
        metadata = {}
        
        # Extract common fields
        fields = [
            'PatientID', 'StudyDate', 'Modality', 'StudyDescription',
            'SeriesDescription', 'SliceThickness', 'PixelSpacing',
            'RescaleSlope', 'RescaleIntercept', 'WindowCenter', 'WindowWidth'
        ]
        
        for field in fields:
            if hasattr(dcm, field):
                value = getattr(dcm, field)
                if hasattr(value, 'value'):
                    value = value.value
                metadata[field] = value
                
        return metadata
        
    def segment_lungs(self, image: RadiologyImage) -> np.ndarray:
        """
        Segment lungs from CT scan.
        
        Args:
            image: RadiologyImage object (should be CT)
            
        Returns:
            Binary mask of lung regions
        """
        data = image.data
        
        # Ensure 3D
        if len(data.shape) != 3:
            raise ValueError("Lung segmentation requires 3D CT volume")
            
        # Threshold for air regions (-1000 to -300 HU typically)
        binary = data < -300
        
        # Clear border
        for i in range(binary.shape[0]):
            binary[i] = morphology.remove_small_objects(binary[i], min_size=100)
            binary[i] = morphology.binary_closing(binary[i], morphology.disk(2))
            
        # Find largest connected components (lungs)
        labels = measure.label(binary)
        regions = measure.regionprops(labels)
        
        # Sort by area and keep two largest (left and right lung)
        regions.sort(key=lambda x: x.area, reverse=True)
        
        lung_mask = np.zeros_like(binary)
        for region in regions[:2]:
            lung_mask[labels == region.label] = True
            
        # Fill holes
        lung_mask = binary_fill_holes(lung_mask)
        
        return lung_mask
        
    def segment_organs(self, image: RadiologyImage) -> Dict[str, np.ndarray]:
        """
        Segment multiple organs from abdominal CT.
        
        Args:
            image: RadiologyImage object (should be abdominal CT)
            
        Returns:
            Dictionary of organ masks
        """
        data = image.data
        organs = {}
        
        # Simple threshold-based segmentation for demonstration
        # In practice, use deep learning models or atlas-based methods
        
        # Liver (typically 40-60 HU)
        liver_mask = np.logical_and(data > 40, data < 60)
        liver_mask = morphology.binary_opening(liver_mask, morphology.ball(2))
        liver_mask = self._get_largest_component(liver_mask)
        organs['liver'] = liver_mask
        
        # Spleen (similar to liver but smaller and more lateral)
        spleen_mask = np.logical_and(data > 35, data < 55)
        spleen_mask = morphology.binary_opening(spleen_mask, morphology.ball(2))
        # Remove liver region
        spleen_mask = np.logical_and(spleen_mask, ~liver_mask)
        spleen_mask = self._get_largest_component(spleen_mask)
        organs['spleen'] = spleen_mask
        
        # Kidneys (typically 30-40 HU)
        kidney_mask = np.logical_and(data > 25, data < 45)
        kidney_mask = morphology.binary_opening(kidney_mask, morphology.ball(1))
        # Remove liver and spleen
        kidney_mask = np.logical_and(kidney_mask, ~liver_mask)
        kidney_mask = np.logical_and(kidney_mask, ~spleen_mask)
        
        # Get two largest components (left and right kidney)
        labeled = measure.label(kidney_mask)
        regions = measure.regionprops(labeled)
        regions.sort(key=lambda x: x.area, reverse=True)
        
        kidney_combined = np.zeros_like(kidney_mask)
        for region in regions[:2]:
            kidney_combined[labeled == region.label] = True
            
        organs['kidneys'] = kidney_combined
        
        return organs
        
    def segment_tumor(self, image: RadiologyImage, seed_point: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Segment tumor region.
        
        Args:
            image: RadiologyImage object
            seed_point: Optional seed point for region growing
            
        Returns:
            Binary tumor mask
        """
        data = image.data
        
        if seed_point:
            # Region growing from seed point
            mask = self._region_growing(data, seed_point)
        else:
            # Simple threshold-based approach
            # In practice, use deep learning models
            mean_intensity = np.mean(data)
            std_intensity = np.std(data)
            threshold = mean_intensity + 2 * std_intensity
            
            mask = data > threshold
            mask = morphology.binary_opening(mask, morphology.ball(2))
            mask = self._get_largest_component(mask)
            
        return mask
        
    def segment_metabolic_volume(self, image: RadiologyImage, threshold: float = 2.5) -> np.ndarray:
        """
        Segment metabolic volume from PET scan.
        
        Args:
            image: RadiologyImage object (should be PET)
            threshold: SUV threshold for segmentation
            
        Returns:
            Binary mask of metabolically active regions
        """
        data = image.data
        
        # Threshold by SUV
        mask = data > threshold
        
        # Remove small regions
        mask = morphology.remove_small_objects(mask, min_size=50)
        
        # Smooth boundaries
        mask = morphology.binary_closing(mask, morphology.ball(2))
        
        return mask
        
    def denoise(self, image: RadiologyImage, method: str = "nlm") -> RadiologyImage:
        """
        Apply denoising to image.
        
        Args:
            image: RadiologyImage object
            method: Denoising method ('nlm', 'tv', 'bilateral', 'deep')
            
        Returns:
            Denoised RadiologyImage
        """
        data = image.data.copy()
        
        if method == "nlm":
            # Non-local means
            if len(data.shape) == 3:
                # Process slice by slice for 3D
                for i in range(data.shape[0]):
                    data[i] = denoise_nl_means(data[i], h=0.1, patch_size=5, patch_distance=7)
            else:
                data = denoise_nl_means(data, h=0.1, patch_size=5, patch_distance=7)
                
        elif method == "tv":
            # Total variation
            data = denoise_tv_chambolle(data, weight=0.1)
            
        elif method == "bilateral":
            # Bilateral filter
            if len(data.shape) == 3:
                for i in range(data.shape[0]):
                    # Convert to uint8 for cv2
                    slice_norm = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    denoised = cv2.bilateralFilter(slice_norm, 9, 75, 75)
                    # Convert back
                    data[i] = denoised.astype(data.dtype) * (data[i].max() - data[i].min()) / 255 + data[i].min()
            else:
                data_norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                data = cv2.bilateralFilter(data_norm, 9, 75, 75)
                
        elif method == "deep":
            # Placeholder for deep learning denoising
            logger.warning("Deep learning denoising not implemented, using NLM instead")
            return self.denoise(image, method="nlm")
            
        elif method == "rician":
            # For MRI with Rician noise
            # Use NLM with adjusted parameters
            if len(data.shape) == 3:
                for i in range(data.shape[0]):
                    data[i] = denoise_nl_means(data[i], h=0.08, patch_size=3, patch_distance=5)
            else:
                data = denoise_nl_means(data, h=0.08, patch_size=3, patch_distance=5)
                
        elif method == "pet_specific":
            # PET-specific denoising
            # Use Gaussian filter with sigma based on count statistics
            sigma = 1.0  # Adjust based on count levels
            data = ndimage.gaussian_filter(data, sigma=sigma)
            
        else:
            raise ValueError(f"Unknown denoising method: {method}")
            
        return RadiologyImage(data, image.metadata)
        
    def reduce_metal_artifacts(self, image: RadiologyImage) -> RadiologyImage:
        """
        Reduce metal artifacts in CT scan.
        
        Args:
            image: RadiologyImage object (CT)
            
        Returns:
            Image with reduced metal artifacts
        """
        data = image.data.copy()
        
        # Simple approach: identify metal regions and interpolate
        metal_threshold = 3000  # HU for metal
        metal_mask = data > metal_threshold
        
        # Dilate mask to include artifact regions
        metal_mask = morphology.binary_dilation(metal_mask, morphology.ball(3))
        
        # Interpolate metal regions
        if len(data.shape) == 3:
            for i in range(data.shape[0]):
                if np.any(metal_mask[i]):
                    # Use inpainting
                    slice_norm = cv2.normalize(data[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    mask_uint8 = metal_mask[i].astype(np.uint8) * 255
                    inpainted = cv2.inpaint(slice_norm, mask_uint8, 3, cv2.INPAINT_TELEA)
                    # Convert back
                    data[i] = inpainted.astype(data.dtype) * (data[i].max() - data[i].min()) / 255 + data[i].min()
        else:
            data_norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask_uint8 = metal_mask.astype(np.uint8) * 255
            data = cv2.inpaint(data_norm, mask_uint8, 3, cv2.INPAINT_TELEA)
            
        return RadiologyImage(data, image.metadata)
        
    def resample(self, image: RadiologyImage, spacing: Tuple[float, float, float]) -> RadiologyImage:
        """
        Resample image to specified spacing.
        
        Args:
            image: RadiologyImage object
            spacing: Target spacing in mm (x, y, z)
            
        Returns:
            Resampled RadiologyImage
        """
        # Create SimpleITK image
        sitk_image = sitk.GetImageFromArray(image.data)
        
        # Set original spacing if available
        if 'spacing' in image.metadata:
            orig_spacing = image.metadata['spacing']
            if len(orig_spacing) == 3:
                sitk_image.SetSpacing(orig_spacing)
            elif len(orig_spacing) == 2 and len(image.data.shape) == 2:
                sitk_image.SetSpacing(orig_spacing)
                
        # Calculate new size
        orig_size = np.array(sitk_image.GetSize())
        orig_spacing = np.array(sitk_image.GetSpacing())
        new_size = orig_size * (orig_spacing / np.array(spacing))
        new_size = np.round(new_size).astype(int)
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size.tolist())
        resampler.SetOutputSpacing(spacing)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        resampled = resampler.Execute(sitk_image)
        
        # Convert back to numpy
        data = sitk.GetArrayFromImage(resampled)
        
        # Update metadata
        metadata = image.metadata.copy()
        metadata['spacing'] = spacing
        metadata['resampled'] = True
        
        return RadiologyImage(data, metadata)
        
    def reorient(self, image: RadiologyImage, orientation: str = "RAS") -> RadiologyImage:
        """
        Reorient image to standard orientation.
        
        Args:
            image: RadiologyImage object
            orientation: Target orientation (e.g., 'RAS', 'LPS')
            
        Returns:
            Reoriented RadiologyImage
        """
        # For simplicity, we'll just ensure proper axis order
        # In practice, use nibabel or SimpleITK for proper reorientation
        
        data = image.data
        metadata = image.metadata.copy()
        
        # If we have affine matrix (from NIfTI), use it
        if 'affine' in metadata:
            # This is a placeholder - proper implementation would use
            # nibabel's reorientation functions
            logger.info(f"Reorienting to {orientation}")
            metadata['orientation'] = orientation
            
        return RadiologyImage(data, metadata)
        
    def register(self, image: RadiologyImage, atlas: RadiologyImage) -> RadiologyImage:
        """
        Register image to atlas.
        
        Args:
            image: Moving image
            atlas: Fixed atlas image
            
        Returns:
            Registered RadiologyImage
        """
        # Convert to SimpleITK
        moving = sitk.GetImageFromArray(image.data)
        fixed = sitk.GetImageFromArray(atlas.data)
        
        # Initialize registration
        registration = sitk.ImageRegistrationMethod()
        
        # Similarity metric
        registration.SetMetricAsMeanSquares()
        
        # Optimizer
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=0.001,
            numberOfIterations=100
        )
        
        # Interpolator
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Initial transform
        initial_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform()
        )
        registration.SetInitialTransform(initial_transform)
        
        # Execute registration
        final_transform = registration.Execute(fixed, moving)
        
        # Apply transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)
        
        registered = resampler.Execute(moving)
        
        # Convert back
        data = sitk.GetArrayFromImage(registered)
        
        metadata = image.metadata.copy()
        metadata['registered'] = True
        metadata['registration_transform'] = final_transform
        
        return RadiologyImage(data, metadata)
        
    def crop_to_roi(self, image: RadiologyImage, roi_mask: np.ndarray) -> RadiologyImage:
        """
        Crop image to region of interest.
        
        Args:
            image: RadiologyImage object
            roi_mask: Binary mask defining ROI
            
        Returns:
            Cropped RadiologyImage
        """
        # Find bounding box of ROI
        coords = np.where(roi_mask)
        
        if len(coords[0]) == 0:
            raise ValueError("ROI mask is empty")
            
        # Get bounding box
        min_coords = [c.min() for c in coords]
        max_coords = [c.max() for c in coords]
        
        # Crop data
        if len(image.data.shape) == 3:
            cropped = image.data[
                min_coords[0]:max_coords[0]+1,
                min_coords[1]:max_coords[1]+1,
                min_coords[2]:max_coords[2]+1
            ]
        elif len(image.data.shape) == 2:
            cropped = image.data[
                min_coords[0]:max_coords[0]+1,
                min_coords[1]:max_coords[1]+1
            ]
        else:
            raise ValueError(f"Unsupported image dimensions: {len(image.data.shape)}")
            
        metadata = image.metadata.copy()
        metadata['cropped'] = True
        metadata['crop_bounds'] = list(zip(min_coords, max_coords))
        
        return RadiologyImage(cropped, metadata)
        
    def verify_hounsfield_units(self, image: RadiologyImage) -> RadiologyImage:
        """
        Verify and correct Hounsfield units for CT.
        
        Args:
            image: RadiologyImage object (CT)
            
        Returns:
            Corrected RadiologyImage
        """
        data = image.data.copy()
        metadata = image.metadata
        
        # Apply rescale slope and intercept if available
        if 'RescaleSlope' in metadata and 'RescaleIntercept' in metadata:
            slope = float(metadata['RescaleSlope'])
            intercept = float(metadata['RescaleIntercept'])
            data = data * slope + intercept
            
        # Verify reasonable HU range
        if data.min() < -2000 or data.max() > 4000:
            logger.warning(f"Unusual HU range: [{data.min()}, {data.max()}]")
            
        metadata = metadata.copy()
        metadata['hu_corrected'] = True
        
        return RadiologyImage(data, metadata)
        
    def apply_window(self, image: RadiologyImage, window: float, level: float) -> RadiologyImage:
        """
        Apply window/level adjustment.
        
        Args:
            image: RadiologyImage object
            window: Window width
            level: Window center
            
        Returns:
            Windowed RadiologyImage
        """
        data = image.data.copy()
        
        # Calculate window bounds
        min_val = level - window / 2
        max_val = level + window / 2
        
        # Apply windowing
        data = np.clip(data, min_val, max_val)
        
        # Normalize to [0, 1]
        data = (data - min_val) / (max_val - min_val)
        
        metadata = image.metadata.copy()
        metadata['window'] = window
        metadata['level'] = level
        
        return RadiologyImage(data, metadata)
        
    def normalize_intensity(self, image: RadiologyImage, method: str = "z_score") -> RadiologyImage:
        """
        Normalize image intensities.
        
        Args:
            image: RadiologyImage object
            method: Normalization method ('z_score', 'min_max', 'percentile')
            
        Returns:
            Normalized RadiologyImage
        """
        data = image.data.copy()
        
        if method == "z_score":
            mean = np.mean(data)
            std = np.std(data)
            data = (data - mean) / (std + 1e-8)
            
        elif method == "min_max":
            min_val = np.min(data)
            max_val = np.max(data)
            data = (data - min_val) / (max_val - min_val + 1e-8)
            
        elif method == "percentile":
            # Use 1st and 99th percentile for robustness
            p1 = np.percentile(data, 1)
            p99 = np.percentile(data, 99)
            data = np.clip(data, p1, p99)
            data = (data - p1) / (p99 - p1 + 1e-8)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
        metadata = image.metadata.copy()
        metadata['normalized'] = True
        metadata['normalization_method'] = method
        
        return RadiologyImage(data, metadata)
        
    def correct_bias_field(self, image: RadiologyImage) -> RadiologyImage:
        """
        Correct bias field in MRI.
        
        Args:
            image: RadiologyImage object (MRI)
            
        Returns:
            Bias-corrected RadiologyImage
        """
        # Convert to SimpleITK
        sitk_image = sitk.GetImageFromArray(image.data)
        
        # N4 bias field correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        
        # Create mask (non-zero regions)
        mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
        
        # Run correction
        corrected = corrector.Execute(sitk_image, mask)
        
        # Convert back
        data = sitk.GetArrayFromImage(corrected)
        
        metadata = image.metadata.copy()
        metadata['bias_corrected'] = True
        
        return RadiologyImage(data, metadata)
        
    def calculate_suv(self, image: RadiologyImage, patient_weight: float, 
                     injected_dose: float, injection_time: str) -> RadiologyImage:
        """
        Calculate SUV from PET scan.
        
        Args:
            image: RadiologyImage object (PET)
            patient_weight: Patient weight in kg
            injected_dose: Injected dose in mCi
            injection_time: Injection time in ISO format
            
        Returns:
            SUV image
        """
        data = image.data.copy()
        
        # Convert dose to Bq (1 mCi = 37 MBq)
        dose_bq = injected_dose * 37e6
        
        # Calculate SUV
        # SUV = (tissue_concentration) / (injected_dose / patient_weight)
        suv = data * patient_weight / dose_bq
        
        metadata = image.metadata.copy()
        metadata['suv_calculated'] = True
        metadata['patient_weight'] = patient_weight
        metadata['injected_dose'] = injected_dose
        metadata['injection_time'] = injection_time
        
        return RadiologyImage(suv, metadata)
        
    def preprocess(self, image: RadiologyImage, denoise: bool = True,
                  correct_artifacts: bool = True, resample_spacing: Optional[Tuple[float, float, float]] = None,
                  normalize: bool = True) -> RadiologyImage:
        """
        Complete preprocessing pipeline.
        
        Args:
            image: RadiologyImage object
            denoise: Whether to apply denoising
            correct_artifacts: Whether to correct artifacts
            resample_spacing: Target spacing for resampling
            normalize: Whether to normalize intensities
            
        Returns:
            Preprocessed RadiologyImage
        """
        result = image
        
        # Verify HU units for CT
        if image.metadata.get('Modality') == 'CT':
            result = self.verify_hounsfield_units(result)
            
        # Denoise
        if denoise:
            result = self.denoise(result, method="nlm")
            
        # Correct artifacts
        if correct_artifacts and image.metadata.get('Modality') == 'CT':
            result = self.reduce_metal_artifacts(result)
            
        # Bias field correction for MRI
        if image.metadata.get('Modality') == 'MR':
            result = self.correct_bias_field(result)
            
        # Resample
        if resample_spacing:
            result = self.resample(result, resample_spacing)
            
        # Normalize
        if normalize:
            result = self.normalize_intensity(result, method="percentile")
            
        return result
        
    def apply_mask(self, image: RadiologyImage, mask: np.ndarray) -> RadiologyImage:
        """
        Apply binary mask to image.
        
        Args:
            image: RadiologyImage object
            mask: Binary mask
            
        Returns:
            Masked RadiologyImage
        """
        data = image.data.copy()
        data[~mask] = 0
        
        metadata = image.metadata.copy()
        metadata['masked'] = True
        
        return RadiologyImage(data, metadata)
        
    def generate_embeddings(self, image: RadiologyImage, mode: str = "2d") -> np.ndarray:
        """
        Generate embeddings from radiological image.
        
        Args:
            image: RadiologyImage object
            mode: Processing mode ('2d' or '3d')
            
        Returns:
            Embeddings array
        """
        data = image.data
        
        # Ensure proper format
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]  # Add channel dimension
            
        if mode == "3d" and len(data.shape) == 3:
            # Process each slice and aggregate
            embeddings = []
            
            for i in range(data.shape[0]):
                slice_data = data[i:i+1]
                
                # Convert to tensor
                tensor = torch.from_numpy(slice_data).float().unsqueeze(0)
                
                # Resize if needed
                if tensor.shape[-2:] != (224, 224):
                    tensor = torch.nn.functional.interpolate(
                        tensor, size=(224, 224), mode='bilinear', align_corners=False
                    )
                    
                # Move to device
                tensor = tensor.to(self.device)
                
                # Generate embedding
                with torch.no_grad():
                    embedding = self._generate_embedding_from_model(tensor)
                    
                embeddings.append(embedding)
                
            return np.vstack(embeddings)
            
        else:
            # Process as single image
            if len(data.shape) == 3:
                # Take middle slice for 3D volume
                mid_slice = data.shape[0] // 2
                data = data[mid_slice:mid_slice+1]
                
            # Convert to tensor
            tensor = torch.from_numpy(data).float().unsqueeze(0)
            
            # Resize if needed
            if tensor.shape[-2:] != (224, 224):
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(224, 224), mode='bilinear', align_corners=False
                )
                
            # Move to device
            tensor = tensor.to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self._generate_embedding_from_model(tensor)
                
            return embedding
            
    def _generate_embedding_from_model(self, tensor: torch.Tensor) -> np.ndarray:
        """Generate embedding using the appropriate model method."""
        # Convert tensor to numpy for models that expect numpy input
        if self.model_type in ["remedis", "radimagenet"]:
            # These models expect numpy arrays
            input_data = tensor.cpu().numpy()
            
            # Call the appropriate method
            if hasattr(self.model, 'load_model_and_predict'):
                # For REMEDIS and RadImageNet
                # Note: These models need proper initialization with model paths
                # For now, return a placeholder embedding
                logger.warning(f"{self.model_type} model requires proper initialization with model path. Returning placeholder embedding.")
                return np.random.randn(1, 768)  # Placeholder embedding
            else:
                raise AttributeError(f"Model {self.model_type} doesn't have a prediction method")
        else:
            # For models with generate_embeddings method
            if hasattr(self.model, 'generate_embeddings'):
                embedding = self.model.generate_embeddings(tensor)
                return embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
            else:
                # Fallback: use the model as a feature extractor
                logger.warning(f"Model {self.model_type} doesn't have generate_embeddings method. Using as feature extractor.")
                with torch.no_grad():
                    features = self.model(tensor)
                    if isinstance(features, torch.Tensor):
                        return features.cpu().numpy()
                    return features
            
    def aggregate_embeddings(self, embeddings: np.ndarray, method: str = "mean") -> np.ndarray:
        """
        Aggregate multiple embeddings into single representation.
        
        Args:
            embeddings: Array of embeddings
            method: Aggregation method ('mean', 'max', 'weighted')
            
        Returns:
            Single embedding vector
        """
        if method == "mean":
            return np.mean(embeddings, axis=0)
        elif method == "max":
            return np.max(embeddings, axis=0)
        elif method == "weighted":
            # Weight by position (center slices more important)
            n = len(embeddings)
            weights = np.exp(-((np.arange(n) - n/2) ** 2) / (2 * (n/4) ** 2))
            weights /= weights.sum()
            return np.average(embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
    def _get_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Get largest connected component from binary mask."""
        labeled = measure.label(mask)
        if labeled.max() == 0:
            return mask
            
        regions = measure.regionprops(labeled)
        largest = max(regions, key=lambda x: x.area)
        
        return labeled == largest.label
        
    def _region_growing(self, data: np.ndarray, seed: Tuple[int, int, int], 
                       threshold: float = 0.1) -> np.ndarray:
        """Simple region growing segmentation."""
        mask = np.zeros_like(data, dtype=bool)
        
        # Initialize with seed point
        mask[seed] = True
        seed_value = data[seed]
        
        # Create structure for dilation
        struct = ndimage.generate_binary_structure(3, 1)
        
        # Iterative growing
        for _ in range(100):  # Max iterations
            # Dilate current region
            dilated = ndimage.binary_dilation(mask, struct)
            
            # Check which new pixels meet criteria
            new_pixels = dilated & ~mask
            values = data[new_pixels]
            
            # Add pixels within threshold
            within_threshold = np.abs(values - seed_value) < threshold * np.abs(seed_value)
            
            if not np.any(within_threshold):
                break
                
            # Update mask
            coords = np.where(new_pixels)
            for i in range(len(coords[0])):
                if within_threshold.flat[i]:
                    mask[coords[0][i], coords[1][i], coords[2][i]] = True
                    
        return mask