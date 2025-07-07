"""
Preprocessing and Enhancement Module

Comprehensive preprocessing capabilities for medical images including:
- Denoising (NLM, TV, bilateral, PET-specific)
- Artifact reduction
- Intensity normalization
- Window/level adjustment
"""

import numpy as np
from typing import Union, Tuple, Optional, Dict, List
import cv2
from scipy import ndimage
from scipy.ndimage import median_filter, gaussian_filter
from skimage import exposure, morphology, filters
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, denoise_bilateral
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)


class Denoiser:
    """Advanced denoising methods for medical images"""
    
    def __init__(self, method: str = 'nlm'):
        """
        Initialize denoiser
        
        Args:
            method: Denoising method ('nlm', 'tv', 'bilateral', 'median', 'gaussian', 'pet_specific')
        """
        self.method = method.lower()
        self.supported_methods = ['nlm', 'tv', 'bilateral', 'median', 'gaussian', 'pet_specific', 'deep']
        
        if self.method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported. Choose from {self.supported_methods}")
    
    def denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply denoising to image"""
        if self.method == 'nlm':
            return self._nlm_denoise(image, **kwargs)
        elif self.method == 'tv':
            return self._tv_denoise(image, **kwargs)
        elif self.method == 'bilateral':
            return self._bilateral_denoise(image, **kwargs)
        elif self.method == 'median':
            return self._median_denoise(image, **kwargs)
        elif self.method == 'gaussian':
            return self._gaussian_denoise(image, **kwargs)
        elif self.method == 'pet_specific':
            return self._pet_specific_denoise(image, **kwargs)
        elif self.method == 'deep':
            return self._deep_denoise(image, **kwargs)
    
    def _nlm_denoise(self, image: np.ndarray, patch_size: int = 5, 
                     patch_distance: int = 6, h: float = 0.1) -> np.ndarray:
        """Non-local means denoising"""
        # Normalize to 0-1 range for skimage
        img_min, img_max = image.min(), image.max()
        img_norm = (image - img_min) / (img_max - img_min + 1e-8)
        
        # Apply NLM
        if len(image.shape) == 2:
            denoised = denoise_nl_means(img_norm, patch_size=patch_size,
                                       patch_distance=patch_distance, h=h)
        else:
            # Process slice by slice for 3D
            denoised = np.zeros_like(img_norm)
            for i in range(image.shape[0]):
                denoised[i] = denoise_nl_means(img_norm[i], patch_size=patch_size,
                                              patch_distance=patch_distance, h=h)
        
        # Restore original range
        return denoised * (img_max - img_min) + img_min
    
    def _tv_denoise(self, image: np.ndarray, weight: float = 0.1) -> np.ndarray:
        """Total variation denoising"""
        # Normalize
        img_min, img_max = image.min(), image.max()
        img_norm = (image - img_min) / (img_max - img_min + 1e-8)
        
        # Apply TV denoising
        if len(image.shape) == 2:
            denoised = denoise_tv_chambolle(img_norm, weight=weight)
        else:
            denoised = np.zeros_like(img_norm)
            for i in range(image.shape[0]):
                denoised[i] = denoise_tv_chambolle(img_norm[i], weight=weight)
        
        return denoised * (img_max - img_min) + img_min
    
    def _bilateral_denoise(self, image: np.ndarray, sigma_color: float = 0.05,
                          sigma_spatial: float = 15) -> np.ndarray:
        """Bilateral filtering"""
        # For 2D images
        if len(image.shape) == 2:
            # Convert to uint8 for OpenCV
            img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            denoised = cv2.bilateralFilter(img_norm, d=9, sigmaColor=75, sigmaSpace=75)
            # Convert back
            return denoised.astype(np.float32) * (image.max() - image.min()) / 255 + image.min()
        else:
            # Process slice by slice
            denoised = np.zeros_like(image)
            for i in range(image.shape[0]):
                img_norm = cv2.normalize(image[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                denoised[i] = cv2.bilateralFilter(img_norm, d=9, sigmaColor=75, sigmaSpace=75)
            return denoised.astype(np.float32) * (image.max() - image.min()) / 255 + image.min()
    
    def _median_denoise(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        """Median filtering"""
        return median_filter(image, size=size)
    
    def _gaussian_denoise(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian filtering"""
        return gaussian_filter(image, sigma=sigma)
    
    def _pet_specific_denoise(self, image: np.ndarray, 
                             iterations: int = 5,
                             kernel_size: int = 3) -> np.ndarray:
        """PET-specific denoising using iterative filtering"""
        denoised = image.copy()
        
        for _ in range(iterations):
            # Apply median filter to reduce noise
            denoised = median_filter(denoised, size=kernel_size)
            
            # Apply edge-preserving smoothing
            denoised = self._anisotropic_diffusion(denoised, iterations=10)
        
        return denoised
    
    def _deep_denoise(self, image: np.ndarray, model_path: Optional[str] = None) -> np.ndarray:
        """Deep learning-based denoising (placeholder)"""
        logger.warning("Deep denoising not implemented, using bilateral filter instead")
        return self._bilateral_denoise(image)
    
    def _anisotropic_diffusion(self, image: np.ndarray, iterations: int = 10,
                              kappa: float = 50, gamma: float = 0.1) -> np.ndarray:
        """Perona-Malik anisotropic diffusion"""
        img = image.copy()
        
        for _ in range(iterations):
            # Calculate gradients
            nabla_n = np.roll(img, -1, axis=0) - img
            nabla_s = np.roll(img, 1, axis=0) - img
            nabla_e = np.roll(img, -1, axis=1) - img
            nabla_w = np.roll(img, 1, axis=1) - img
            
            # Calculate diffusion coefficients
            c_n = np.exp(-(nabla_n/kappa)**2)
            c_s = np.exp(-(nabla_s/kappa)**2)
            c_e = np.exp(-(nabla_e/kappa)**2)
            c_w = np.exp(-(nabla_w/kappa)**2)
            
            # Update image
            img += gamma * (c_n * nabla_n + c_s * nabla_s + 
                           c_e * nabla_e + c_w * nabla_w)
        
        return img


class IntensityNormalizer:
    """Intensity normalization methods"""
    
    def __init__(self, method: str = 'zscore'):
        """
        Initialize normalizer
        
        Args:
            method: Normalization method ('zscore', 'minmax', 'percentile', 'histogram')
        """
        self.method = method.lower()
        self.supported_methods = ['zscore', 'minmax', 'percentile', 'histogram']
        
        if self.method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported")
    
    def normalize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply normalization"""
        if self.method == 'zscore':
            return self._zscore_normalize(image, **kwargs)
        elif self.method == 'minmax':
            return self._minmax_normalize(image, **kwargs)
        elif self.method == 'percentile':
            return self._percentile_normalize(image, **kwargs)
        elif self.method == 'histogram':
            return self._histogram_normalize(image, **kwargs)
    
    def _zscore_normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Z-score normalization"""
        if mask is not None:
            mean = image[mask > 0].mean()
            std = image[mask > 0].std()
        else:
            mean = image.mean()
            std = image.std()
        
        return (image - mean) / (std + 1e-8)
    
    def _minmax_normalize(self, image: np.ndarray, 
                         out_min: float = 0.0, out_max: float = 1.0) -> np.ndarray:
        """Min-max normalization"""
        in_min, in_max = image.min(), image.max()
        normalized = (image - in_min) / (in_max - in_min + 1e-8)
        return normalized * (out_max - out_min) + out_min
    
    def _percentile_normalize(self, image: np.ndarray, 
                            lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
        """Percentile-based normalization"""
        p_lower = np.percentile(image, lower)
        p_upper = np.percentile(image, upper)
        
        # Clip and normalize
        clipped = np.clip(image, p_lower, p_upper)
        return (clipped - p_lower) / (p_upper - p_lower + 1e-8)
    
    def _histogram_normalize(self, image: np.ndarray, nbins: int = 256) -> np.ndarray:
        """Histogram equalization"""
        # For 2D images
        if len(image.shape) == 2:
            return exposure.equalize_hist(image, nbins=nbins)
        else:
            # Process slice by slice
            normalized = np.zeros_like(image)
            for i in range(image.shape[0]):
                normalized[i] = exposure.equalize_hist(image[i], nbins=nbins)
            return normalized


class WindowLevelAdjuster:
    """Window/level adjustment for medical images"""
    
    # Predefined window settings
    PRESETS = {
        'lung': {'center': -600, 'width': 1500},
        'abdomen': {'center': 50, 'width': 350},
        'bone': {'center': 400, 'width': 2000},
        'brain': {'center': 40, 'width': 80},
        'soft_tissue': {'center': 50, 'width': 400},
        'liver': {'center': 60, 'width': 150},
        'mediastinum': {'center': 50, 'width': 350},
        'stroke': {'center': 35, 'width': 40},
        'cta': {'center': 300, 'width': 600},
        'pet': {'center': 2.5, 'width': 5.0}
    }
    
    def __init__(self):
        self.current_window = None
        self.current_level = None
    
    def adjust(self, image: np.ndarray, 
               window: Union[float, str] = None,
               level: float = None,
               output_range: Tuple[float, float] = (0, 255)) -> np.ndarray:
        """
        Apply window/level adjustment
        
        Args:
            image: Input image
            window: Window width or preset name
            level: Window center/level
            output_range: Output intensity range
        """
        # Handle presets
        if isinstance(window, str):
            if window.lower() not in self.PRESETS:
                raise ValueError(f"Unknown preset: {window}")
            preset = self.PRESETS[window.lower()]
            window = preset['width']
            level = preset['center']
        
        # Use defaults if not provided
        if window is None or level is None:
            # Auto window/level
            level = np.median(image)
            window = np.percentile(image, 95) - np.percentile(image, 5)
        
        # Apply windowing
        min_val = level - window / 2
        max_val = level + window / 2
        
        # Clip and scale
        windowed = np.clip(image, min_val, max_val)
        scaled = (windowed - min_val) / (max_val - min_val)
        
        # Scale to output range
        return scaled * (output_range[1] - output_range[0]) + output_range[0]
    
    def get_auto_window(self, image: np.ndarray, 
                       percentile_range: Tuple[float, float] = (5, 95)) -> Dict[str, float]:
        """Calculate automatic window/level from image statistics"""
        p_low, p_high = np.percentile(image, percentile_range)
        
        center = (p_low + p_high) / 2
        width = p_high - p_low
        
        return {'center': center, 'width': width}


class ArtifactReducer:
    """Artifact reduction for medical images"""
    
    def __init__(self):
        self.methods = ['metal', 'motion', 'ring', 'beam_hardening']
    
    def reduce_artifacts(self, image: np.ndarray, 
                        artifact_type: str = 'metal',
                        **kwargs) -> np.ndarray:
        """Apply artifact reduction"""
        if artifact_type == 'metal':
            return self._reduce_metal_artifacts(image, **kwargs)
        elif artifact_type == 'motion':
            return self._reduce_motion_artifacts(image, **kwargs)
        elif artifact_type == 'ring':
            return self._reduce_ring_artifacts(image, **kwargs)
        elif artifact_type == 'beam_hardening':
            return self._reduce_beam_hardening(image, **kwargs)
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    def _reduce_metal_artifacts(self, image: np.ndarray, 
                               threshold: float = 3000,
                               interpolation_method: str = 'linear') -> np.ndarray:
        """Metal artifact reduction for CT"""
        # Identify metal regions
        metal_mask = image > threshold
        
        # Create interpolation mask
        non_metal_mask = ~metal_mask
        
        # Interpolate metal regions
        if len(image.shape) == 2:
            # Use inpainting for 2D
            result = image.copy()
            metal_regions = metal_mask.astype(np.uint8) * 255
            
            # Convert to appropriate dtype for OpenCV
            img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            # Inpaint metal regions
            inpainted = cv2.inpaint(img_norm, metal_regions, 3, cv2.INPAINT_TELEA)
            
            # Convert back and blend
            inpainted_float = inpainted.astype(np.float32) * (image.max() - image.min()) / 255 + image.min()
            result[metal_mask] = inpainted_float[metal_mask]
            
            return result
        else:
            # Process slice by slice for 3D
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = self._reduce_metal_artifacts(image[i], threshold, interpolation_method)
            return result
    
    def _reduce_motion_artifacts(self, image: np.ndarray, 
                                method: str = 'averaging') -> np.ndarray:
        """Motion artifact reduction"""
        if method == 'averaging':
            # Simple temporal averaging (placeholder)
            return gaussian_filter(image, sigma=1.0)
        else:
            # More sophisticated methods would require multiple frames
            logger.warning("Advanced motion correction not implemented")
            return image
    
    def _reduce_ring_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Ring artifact reduction (common in CT)"""
        # For 2D images, apply polar transform
        if len(image.shape) == 2:
            # Convert to polar coordinates
            center = (image.shape[0] // 2, image.shape[1] // 2)
            
            # Create polar image
            max_radius = min(center)
            polar_image = cv2.warpPolar(image, (max_radius, 360), center, 
                                       max_radius, cv2.WARP_FILL_OUTLIERS)
            
            # Apply median filter along angular direction
            filtered_polar = median_filter(polar_image, size=(1, 5))
            
            # Convert back to Cartesian
            result = cv2.warpPolar(filtered_polar, image.shape, center,
                                  max_radius, cv2.WARP_INVERSE_MAP)
            
            return result
        else:
            # Process slice by slice
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = self._reduce_ring_artifacts(image[i])
            return result
    
    def _reduce_beam_hardening(self, image: np.ndarray, 
                              correction_factor: float = 0.1) -> np.ndarray:
        """Beam hardening correction"""
        # Simple polynomial correction
        # More sophisticated methods would use water correction or dual-energy CT
        corrected = image - correction_factor * (image ** 2) / (image.max() ** 2)
        return corrected


def preprocess_ct(image: np.ndarray, 
                 denoise: bool = True,
                 normalize: bool = True,
                 window: str = 'lung',
                 reduce_artifacts: bool = False) -> np.ndarray:
    """Complete CT preprocessing pipeline"""
    result = image.copy()
    
    # Denoise
    if denoise:
        denoiser = Denoiser(method='bilateral')
        result = denoiser.denoise(result)
    
    # Reduce artifacts
    if reduce_artifacts:
        artifact_reducer = ArtifactReducer()
        result = artifact_reducer.reduce_artifacts(result, artifact_type='metal')
    
    # Window/level adjustment
    windower = WindowLevelAdjuster()
    result = windower.adjust(result, window=window)
    
    # Normalize
    if normalize:
        normalizer = IntensityNormalizer(method='minmax')
        result = normalizer.normalize(result, out_min=0, out_max=1)
    
    return result


def preprocess_mri(image: np.ndarray,
                  denoise: bool = True,
                  bias_correction: bool = True,
                  normalize: bool = True) -> np.ndarray:
    """Complete MRI preprocessing pipeline"""
    result = image.copy()
    
    # Denoise
    if denoise:
        denoiser = Denoiser(method='nlm')
        result = denoiser.denoise(result)
    
    # Bias field correction
    if bias_correction:
        # Use SimpleITK for N4 bias correction
        sitk_image = sitk.GetImageFromArray(result)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(sitk_image)
        result = sitk.GetArrayFromImage(corrected)
    
    # Normalize
    if normalize:
        normalizer = IntensityNormalizer(method='zscore')
        result = normalizer.normalize(result)
    
    return result


def preprocess_pet(image: np.ndarray,
                  denoise: bool = True,
                  normalize: bool = True,
                  suv_conversion: bool = False,
                  body_weight: float = None,
                  injected_dose: float = None) -> np.ndarray:
    """Complete PET preprocessing pipeline"""
    result = image.copy()
    
    # SUV conversion if requested
    if suv_conversion and body_weight and injected_dose:
        # SUV = pixel_value * body_weight(g) / injected_dose(Bq)
        result = result * (body_weight * 1000) / injected_dose
    
    # Denoise
    if denoise:
        denoiser = Denoiser(method='pet_specific')
        result = denoiser.denoise(result)
    
    # Normalize
    if normalize:
        normalizer = IntensityNormalizer(method='percentile')
        result = normalizer.normalize(result, lower=5, upper=95)
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    from data_management import load_medical_image
    
    if len(sys.argv) > 1:
        # Load image
        image, metadata = load_medical_image(sys.argv[1])
        
        # Preprocess based on modality
        if metadata.modality == 'CT':
            processed = preprocess_ct(image)
        elif metadata.modality == 'MR':
            processed = preprocess_mri(image)
        elif metadata.modality == 'PT':
            processed = preprocess_pet(image)
        else:
            processed = image
        
        print(f"Original shape: {image.shape}, range: [{image.min():.2f}, {image.max():.2f}]")
        print(f"Processed shape: {processed.shape}, range: [{processed.min():.2f}, {processed.max():.2f}]")