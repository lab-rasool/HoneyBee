"""
Spatial Processing Module

Handles spatial transformations and processing for medical images:
- Resampling with multiple interpolation methods
- Registration to atlas/reference
- Cross-scanner harmonization
"""

import numpy as np
from typing import Union, Tuple, Optional, Dict, List
import SimpleITK as sitk
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import cv2
import logging

logger = logging.getLogger(__name__)


class Resampler:
    """Image resampling with various interpolation methods"""
    
    INTERPOLATION_METHODS = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'bspline': sitk.sitkBSpline,
        'gaussian': sitk.sitkGaussian,
        'lanczos': sitk.sitkLanczosWindowedSinc
    }
    
    def __init__(self, method: str = 'linear', anti_aliasing: bool = True):
        """
        Initialize resampler
        
        Args:
            method: Interpolation method
            anti_aliasing: Apply anti-aliasing filter before downsampling
        """
        if method not in self.INTERPOLATION_METHODS:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        self.method = method
        self.anti_aliasing = anti_aliasing
        self.interpolator = self.INTERPOLATION_METHODS[method]
    
    def resample_to_spacing(self, image: np.ndarray,
                          current_spacing: Tuple[float, float, float],
                          target_spacing: Tuple[float, float, float],
                          order: Optional[int] = None) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Resample image to target spacing
        
        Args:
            image: Input image
            current_spacing: Current voxel spacing (z, y, x)
            target_spacing: Target voxel spacing (z, y, x)
            order: Interpolation order (overrides method)
        """
        # Calculate new size
        current_size = np.array(image.shape)
        current_spacing = np.array(current_spacing)
        target_spacing = np.array(target_spacing)
        
        new_size = current_size * current_spacing / target_spacing
        new_size = np.round(new_size).astype(int)
        
        # Resample
        resampled = self.resample_to_size(image, tuple(new_size), order)
        
        return resampled, tuple(target_spacing)
    
    def resample_to_size(self, image: np.ndarray,
                        target_size: Tuple[int, ...],
                        order: Optional[int] = None) -> np.ndarray:
        """
        Resample image to target size
        
        Args:
            image: Input image
            target_size: Target size
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        """
        if self.anti_aliasing and any(t < s for t, s in zip(target_size, image.shape)):
            # Apply Gaussian smoothing before downsampling
            sigma = []
            for orig, target in zip(image.shape, target_size):
                if target < orig:
                    # Standard deviation for Gaussian kernel
                    s = 0.5 * (orig / target - 1)
                    sigma.append(s)
                else:
                    sigma.append(0)
            
            if any(s > 0 for s in sigma):
                image = ndimage.gaussian_filter(image, sigma=sigma)
        
        # Use scipy for resampling if order is specified
        if order is not None:
            zoom_factors = np.array(target_size) / np.array(image.shape)
            return ndimage.zoom(image, zoom_factors, order=order)
        
        # Otherwise use SimpleITK
        sitk_image = sitk.GetImageFromArray(image)
        
        # Set up resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([1.0] * sitk_image.GetDimension())
        resampler.SetSize([int(s) for s in target_size[::-1]])  # SimpleITK uses xyz order
        resampler.SetInterpolator(self.interpolator)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        
        # Resample
        resampled_sitk = resampler.Execute(sitk_image)
        resampled = sitk.GetArrayFromImage(resampled_sitk)
        
        return resampled
    
    def resample_to_isotropic(self, image: np.ndarray,
                            current_spacing: Tuple[float, float, float],
                            target_resolution: Optional[float] = None) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Resample to isotropic voxels
        
        Args:
            image: Input image
            current_spacing: Current spacing
            target_resolution: Target isotropic resolution (if None, use minimum spacing)
        """
        if target_resolution is None:
            target_resolution = min(current_spacing)
        
        target_spacing = (target_resolution, target_resolution, target_resolution)
        return self.resample_to_spacing(image, current_spacing, target_spacing)
    
    def resample_slices(self, image: np.ndarray,
                       scale_factor: float,
                       method: Optional[str] = None) -> np.ndarray:
        """
        Resample 2D slices independently (useful for anisotropic data)
        
        Args:
            image: 3D image
            scale_factor: Scaling factor for x,y dimensions
            method: Override interpolation method
        """
        if len(image.shape) != 3:
            raise ValueError("This method is for 3D images only")
        
        # Calculate new size for slices
        new_height = int(image.shape[1] * scale_factor)
        new_width = int(image.shape[2] * scale_factor)
        
        # Resample each slice
        resampled_slices = []
        
        for i in range(image.shape[0]):
            slice_2d = image[i]
            
            if method == 'nearest' or self.method == 'nearest':
                interpolation = cv2.INTER_NEAREST
            elif method == 'linear' or self.method == 'linear':
                interpolation = cv2.INTER_LINEAR
            elif method == 'cubic' or self.method == 'bspline':
                interpolation = cv2.INTER_CUBIC
            else:
                interpolation = cv2.INTER_LINEAR
            
            resampled_slice = cv2.resize(slice_2d, (new_width, new_height),
                                        interpolation=interpolation)
            resampled_slices.append(resampled_slice)
        
        return np.stack(resampled_slices)


class RegistrationEngine:
    """Image registration and alignment"""
    
    def __init__(self, method: str = 'rigid', metric: str = 'correlation'):
        """
        Initialize registration engine
        
        Args:
            method: Registration method ('rigid', 'affine', 'bspline')
            metric: Similarity metric ('correlation', 'mutual_information', 'mean_squares')
        """
        self.method = method
        self.metric = metric
        
        # Set up metric
        if metric == 'correlation':
            self.metric_obj = sitk.CorrelationImageMetric()
        elif metric == 'mutual_information':
            self.metric_obj = sitk.MattesMutualInformationImageMetric()
        elif metric == 'mean_squares':
            self.metric_obj = sitk.MeanSquaresImageMetric()
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def register(self, moving_image: np.ndarray,
                fixed_image: np.ndarray,
                initial_transform: Optional[sitk.Transform] = None,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, sitk.Transform]:
        """
        Register moving image to fixed image
        
        Args:
            moving_image: Image to be transformed
            fixed_image: Reference image
            initial_transform: Initial transformation
            mask: Binary mask for registration
        """
        # Convert to SimpleITK
        moving_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))
        fixed_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))
        
        # Set up registration method
        registration = sitk.ImageRegistrationMethod()
        
        # Set metric
        registration.SetMetricAsCorrelation()  # Using correlation for simplicity
        
        # Set optimizer
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=0.001,
            numberOfIterations=200,
            gradientMagnitudeTolerance=1e-8
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        
        # Set interpolator
        registration.SetInterpolator(sitk.sitkLinear)
        
        # Set transform
        if self.method == 'rigid':
            if len(moving_image.shape) == 2:
                transform = sitk.Euler2DTransform()
            else:
                transform = sitk.Euler3DTransform()
        elif self.method == 'affine':
            transform = sitk.AffineTransform(len(moving_image.shape))
        elif self.method == 'bspline':
            transform = self._create_bspline_transform(fixed_sitk)
        else:
            raise ValueError(f"Unknown registration method: {self.method}")
        
        if initial_transform is not None:
            transform = initial_transform
        
        # Center transform on image centers
        if hasattr(transform, 'SetCenter'):
            transform.SetCenter(fixed_sitk.TransformContinuousIndexToPhysicalPoint(
                np.array(fixed_sitk.GetSize()) / 2.0))
        
        registration.SetInitialTransform(transform, inPlace=True)
        
        # Set masks if provided
        if mask is not None:
            mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
            registration.SetMetricFixedMask(mask_sitk)
        
        # Execute registration
        try:
            final_transform = registration.Execute(fixed_sitk, moving_sitk)
            
            # Apply transform
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed_sitk)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(final_transform)
            
            registered_sitk = resampler.Execute(moving_sitk)
            registered = sitk.GetArrayFromImage(registered_sitk)
            
            return registered, final_transform
            
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return moving_image, sitk.Transform()
    
    def _create_bspline_transform(self, image: sitk.Image) -> sitk.BSplineTransform:
        """Create B-spline transform for deformable registration"""
        grid_size = [8] * image.GetDimension()
        transform = sitk.BSplineTransform(image.GetDimension(), 3)
        transform.SetTransformDomainOrigin(image.GetOrigin())
        transform.SetTransformDomainDirection(image.GetDirection())
        transform.SetTransformDomainPhysicalDimensions(
            [sz * sp for sz, sp in zip(image.GetSize(), image.GetSpacing())]
        )
        transform.SetTransformDomainMeshSize(grid_size)
        
        return transform
    
    def apply_transform(self, image: np.ndarray,
                       transform: sitk.Transform,
                       reference_image: np.ndarray) -> np.ndarray:
        """Apply existing transform to image"""
        moving_sitk = sitk.GetImageFromArray(image.astype(np.float32))
        reference_sitk = sitk.GetImageFromArray(reference_image.astype(np.float32))
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(transform)
        
        transformed_sitk = resampler.Execute(moving_sitk)
        return sitk.GetArrayFromImage(transformed_sitk)
    
    def register_to_atlas(self, image: np.ndarray,
                         atlas: np.ndarray,
                         atlas_labels: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Register image to atlas space
        
        Args:
            image: Input image
            atlas: Atlas template
            atlas_labels: Optional atlas label map
        """
        # Multi-resolution registration for better results
        scales = [4, 2, 1]  # Downsampling factors
        
        current_transform = None
        registered = image
        
        for scale in scales:
            # Downsample if needed
            if scale > 1:
                scaled_moving = ndimage.zoom(registered, 1/scale, order=1)
                scaled_fixed = ndimage.zoom(atlas, 1/scale, order=1)
            else:
                scaled_moving = registered
                scaled_fixed = atlas
            
            # Register at current scale
            registered_scaled, transform = self.register(
                scaled_moving, scaled_fixed, current_transform
            )
            
            # Upsample if needed
            if scale > 1:
                registered = ndimage.zoom(registered_scaled, scale, order=1)
            else:
                registered = registered_scaled
            
            current_transform = transform
        
        result = {'registered_image': registered, 'transform': current_transform}
        
        # Transform labels if provided
        if atlas_labels is not None:
            labels_transformed = self.apply_transform(
                atlas_labels.astype(np.float32), 
                current_transform.GetInverse(),
                image
            )
            result['transformed_labels'] = np.round(labels_transformed).astype(atlas_labels.dtype)
        
        return result


class HarmonizationProcessor:
    """Cross-scanner harmonization for multi-center studies"""
    
    def __init__(self, method: str = 'histogram_matching'):
        """
        Initialize harmonization processor
        
        Args:
            method: Harmonization method ('histogram_matching', 'zscore', 'combat')
        """
        self.method = method
        self.reference_histogram = None
        self.reference_stats = None
    
    def fit_reference(self, reference_images: List[np.ndarray]):
        """Fit harmonization parameters from reference images"""
        if self.method == 'histogram_matching':
            # Compute reference histogram
            all_values = []
            for img in reference_images:
                all_values.extend(img.flatten())
            
            all_values = np.array(all_values)
            self.reference_histogram, self.reference_bins = np.histogram(
                all_values, bins=256, density=True
            )
            
        elif self.method == 'zscore':
            # Compute reference statistics
            all_values = []
            for img in reference_images:
                all_values.extend(img.flatten())
            
            all_values = np.array(all_values)
            self.reference_stats = {
                'mean': all_values.mean(),
                'std': all_values.std()
            }
    
    def harmonize(self, image: np.ndarray,
                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Harmonize image to reference distribution
        
        Args:
            image: Input image
            mask: Optional mask for harmonization
        """
        if self.method == 'histogram_matching':
            return self._histogram_matching(image, mask)
        elif self.method == 'zscore':
            return self._zscore_harmonization(image, mask)
        elif self.method == 'combat':
            return self._combat_harmonization(image, mask)
        else:
            raise ValueError(f"Unknown harmonization method: {self.method}")
    
    def _histogram_matching(self, image: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Histogram matching harmonization"""
        if self.reference_histogram is None:
            raise ValueError("Reference histogram not fitted")
        
        # Get image values
        if mask is not None:
            values = image[mask]
        else:
            values = image.flatten()
        
        # Compute image histogram
        img_hist, img_bins = np.histogram(values, bins=256, density=True)
        
        # Compute CDFs
        img_cdf = np.cumsum(img_hist)
        img_cdf = img_cdf / img_cdf[-1]
        
        ref_cdf = np.cumsum(self.reference_histogram)
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # Create mapping
        mapping = np.interp(img_cdf, ref_cdf, self.reference_bins[:-1])
        
        # Apply mapping
        result = np.interp(image, img_bins[:-1], mapping)
        
        return result
    
    def _zscore_harmonization(self, image: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Z-score based harmonization"""
        if self.reference_stats is None:
            raise ValueError("Reference statistics not fitted")
        
        # Compute image statistics
        if mask is not None:
            img_mean = image[mask].mean()
            img_std = image[mask].std()
        else:
            img_mean = image.mean()
            img_std = image.std()
        
        # Standardize
        standardized = (image - img_mean) / (img_std + 1e-8)
        
        # Transform to reference distribution
        harmonized = standardized * self.reference_stats['std'] + self.reference_stats['mean']
        
        return harmonized
    
    def _combat_harmonization(self, image: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
        """ComBat harmonization (simplified version)"""
        # This is a placeholder for ComBat harmonization
        # Full implementation would require batch information and more complex statistics
        logger.warning("ComBat harmonization not fully implemented, using z-score instead")
        return self._zscore_harmonization(image, mask)
    
    def harmonize_batch(self, images: List[np.ndarray],
                       masks: Optional[List[np.ndarray]] = None,
                       reference_idx: Optional[List[int]] = None) -> List[np.ndarray]:
        """
        Harmonize a batch of images
        
        Args:
            images: List of images to harmonize
            masks: Optional masks for each image
            reference_idx: Indices of reference images (if None, use all)
        """
        # Fit reference if not already done
        if reference_idx is not None:
            reference_images = [images[i] for i in reference_idx]
        else:
            reference_images = images
        
        self.fit_reference(reference_images)
        
        # Harmonize all images
        harmonized = []
        for i, img in enumerate(images):
            mask = masks[i] if masks is not None else None
            harmonized.append(self.harmonize(img, mask))
        
        return harmonized


# Convenience functions
def resample_image(image: np.ndarray,
                  target_spacing: Tuple[float, float, float] = None,
                  target_size: Tuple[int, ...] = None,
                  current_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                  method: str = 'linear') -> np.ndarray:
    """Resample medical image"""
    resampler = Resampler(method=method)
    
    if target_spacing is not None:
        resampled, _ = resampler.resample_to_spacing(image, current_spacing, target_spacing)
        return resampled
    elif target_size is not None:
        return resampler.resample_to_size(image, target_size)
    else:
        raise ValueError("Either target_spacing or target_size must be provided")


def register_to_atlas(image: np.ndarray, atlas: np.ndarray,
                     method: str = 'affine') -> Dict[str, np.ndarray]:
    """Register image to atlas"""
    engine = RegistrationEngine(method=method)
    return engine.register_to_atlas(image, atlas)


def harmonize_cross_scanner(images: List[np.ndarray],
                          method: str = 'histogram_matching') -> List[np.ndarray]:
    """Harmonize images from different scanners"""
    processor = HarmonizationProcessor(method=method)
    return processor.harmonize_batch(images)


def reorient_to_standard(image: np.ndarray, 
                        current_orientation: str = 'RAS',
                        target_orientation: str = 'LPS') -> np.ndarray:
    """
    Reorient image to standard orientation
    
    Common orientations:
    - RAS: Right-Anterior-Superior (neuroimaging standard)
    - LPS: Left-Posterior-Superior (DICOM standard)
    """
    # Define axis mappings
    orientation_map = {
        'R': 0, 'L': 0,
        'A': 1, 'P': 1,
        'S': 2, 'I': 2
    }
    
    # Determine flips and permutations needed
    flips = []
    permutation = []
    
    for i, (curr, targ) in enumerate(zip(current_orientation, target_orientation)):
        # Determine axis
        axis = orientation_map[curr]
        permutation.append(axis)
        
        # Determine if flip is needed
        if curr != targ:
            flips.append(axis)
    
    # Apply permutation
    reoriented = np.transpose(image, permutation)
    
    # Apply flips
    for axis in flips:
        reoriented = np.flip(reoriented, axis=axis)
    
    return reoriented


if __name__ == "__main__":
    # Example usage
    import sys
    from data_management import load_medical_image
    
    if len(sys.argv) > 2:
        # Load images
        img1, meta1 = load_medical_image(sys.argv[1])
        img2, meta2 = load_medical_image(sys.argv[2])
        
        print(f"Image 1 shape: {img1.shape}, spacing: {meta1.pixel_spacing}")
        print(f"Image 2 shape: {img2.shape}, spacing: {meta2.pixel_spacing}")
        
        # Example: Resample to common spacing
        common_spacing = (1.0, 1.0, 1.0)
        img1_resampled = resample_image(img1, target_spacing=common_spacing,
                                       current_spacing=meta1.pixel_spacing)
        img2_resampled = resample_image(img2, target_spacing=common_spacing,
                                       current_spacing=meta2.pixel_spacing)
        
        print(f"Resampled shapes: {img1_resampled.shape}, {img2_resampled.shape}")
        
        # Example: Register images
        engine = RegistrationEngine(method='rigid')
        registered, transform = engine.register(img2_resampled, img1_resampled)
        print(f"Registration complete, registered shape: {registered.shape}")