"""
Corrected Stain Normalization Methods for Digital Pathology

Fixed implementations that properly handle H&E stain separation.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Union
import warnings


def rgb_to_od(rgb):
    """Convert RGB to optical density"""
    rgb = rgb.astype(np.float64)
    od = -np.log10((rgb + 1) / 256)
    return od


def od_to_rgb(od):
    """Convert optical density to RGB"""
    rgb = 256 * (10**(-od)) - 1
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def normalize_rows(A):
    """Normalize rows of a matrix"""
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def get_stain_matrix(rgb_image, luminosity_threshold=0.8, angular_percentiles=(1, 99)):
    """
    Extract stain matrix using the method of Macenko et al.
    
    Args:
        rgb_image: RGB image
        luminosity_threshold: Threshold for background removal
        angular_percentiles: Percentiles for robust angle estimation
        
    Returns:
        Stain matrix (3x2) with H and E vectors as columns
    """
    # Convert to OD
    od = rgb_to_od(rgb_image)
    
    # Remove background
    od_flat = od.reshape(-1, 3)
    optical_density = np.sqrt(np.sum(od_flat**2, axis=1))
    mask = optical_density > luminosity_threshold
    
    if np.sum(mask) < 100:
        # Return default H&E stain matrix
        return np.array([[0.65, 0.70, 0.29],  # Hematoxylin
                        [0.07, 0.99, 0.11]]).T  # Eosin
    
    # Get tissue pixels
    od_tissue = od_flat[mask]
    
    # Compute eigenvectors
    cov = np.cov(od_tissue.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Project data onto plane spanned by first two eigenvectors
    projection = od_tissue @ eigenvectors[:, :2]
    
    # Find angular coordinates
    angles = np.arctan2(projection[:, 1], projection[:, 0])
    
    # Find robust min/max angles
    min_angle = np.percentile(angles, angular_percentiles[0])
    max_angle = np.percentile(angles, angular_percentiles[1])
    
    # Convert back to stain vectors
    v1 = eigenvectors[:, 0] * np.cos(min_angle) + eigenvectors[:, 1] * np.sin(min_angle)
    v2 = eigenvectors[:, 0] * np.cos(max_angle) + eigenvectors[:, 1] * np.sin(max_angle)
    
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Order stain vectors - hematoxylin first (has higher optical density)
    if v1[0] < v2[0]:
        v1, v2 = v2, v1
    
    return np.column_stack([v1, v2])


class ReinhardNormalizer:
    """
    Reinhard color normalization method.
    
    Transfers color characteristics by matching mean and standard deviation
    in LAB color space.
    """
    
    def __init__(self):
        self.target_mean = None
        self.target_std = None
    
    def fit(self, target_image: np.ndarray):
        """Fit normalizer to target image."""
        # Convert to LAB color space - OpenCV expects BGR input
        target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Calculate mean and std for each channel
        self.target_mean = np.mean(target_lab.reshape(-1, 3), axis=0)
        self.target_std = np.std(target_lab.reshape(-1, 3), axis=0)
        
        return self
    
    def transform(self, source_image: np.ndarray) -> np.ndarray:
        """Transform source image using fitted parameters."""
        if self.target_mean is None:
            raise ValueError("Normalizer must be fitted first")
        
        # Convert to LAB
        source_bgr = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
        source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Calculate source statistics
        source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
        source_std = np.std(source_lab.reshape(-1, 3), axis=0)
        
        # Normalize each channel
        result_lab = source_lab.copy()
        for i in range(3):
            # Avoid division by zero
            if source_std[i] < 1e-6:
                result_lab[:, :, i] = source_lab[:, :, i] - source_mean[i] + self.target_mean[i]
            else:
                result_lab[:, :, i] = ((source_lab[:, :, i] - source_mean[i]) / source_std[i]) * self.target_std[i] + self.target_mean[i]
        
        # Clip to valid LAB range in OpenCV (0-255 for all channels)
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        return result_rgb


class MacenkoNormalizer:
    """
    Macenko stain normalization method.
    
    Separates and normalizes H&E stains using singular value decomposition.
    """
    
    def __init__(self, luminosity_threshold=0.8, angular_percentiles=(1, 99)):
        self.luminosity_threshold = luminosity_threshold
        self.angular_percentiles = angular_percentiles
        self.target_stain_matrix = None
        self.target_max_conc = None
    
    def fit(self, target_image: np.ndarray):
        """Fit normalizer to target image."""
        self.target_stain_matrix = get_stain_matrix(
            target_image, 
            self.luminosity_threshold, 
            self.angular_percentiles
        )
        
        # Get target concentrations for reference
        od = rgb_to_od(target_image)
        concentrations = self._get_concentrations(od, self.target_stain_matrix)
        self.target_max_conc = np.percentile(concentrations, 99, axis=0)
        
        return self
    
    def transform(self, source_image: np.ndarray) -> np.ndarray:
        """Transform source image using fitted parameters."""
        if self.target_stain_matrix is None:
            raise ValueError("Normalizer must be fitted first")
        
        # Get source stain matrix
        source_stain_matrix = get_stain_matrix(
            source_image,
            self.luminosity_threshold,
            self.angular_percentiles
        )
        
        # Convert to OD
        source_od = rgb_to_od(source_image)
        
        # Get source concentrations
        source_concentrations = self._get_concentrations(source_od, source_stain_matrix)
        
        # Get max concentrations
        source_max_conc = np.percentile(source_concentrations, 99, axis=0)
        
        # Normalize concentrations to match target
        normalized_conc = source_concentrations.copy()
        for i in range(2):
            if source_max_conc[i] > 0:
                normalized_conc[:, i] = normalized_conc[:, i] * (self.target_max_conc[i] / source_max_conc[i])
        
        # Reconstruct using target stain matrix
        od_reconstructed = normalized_conc @ self.target_stain_matrix.T
        od_reconstructed = od_reconstructed.reshape(source_image.shape)
        
        # Convert back to RGB
        rgb_normalized = od_to_rgb(od_reconstructed)
        
        return rgb_normalized
    
    def _get_concentrations(self, od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Get stain concentrations using least squares"""
        od_flat = od.reshape(-1, 3)
        # Use pseudo-inverse for stability
        concentrations = od_flat @ np.linalg.pinv(stain_matrix).T
        # Ensure non-negative concentrations
        concentrations = np.maximum(concentrations, 0)
        return concentrations


class VahadaneNormalizer:
    """
    Vahadane stain normalization - using the same approach as Macenko
    but with different concentration normalization.
    """
    
    def __init__(self, luminosity_threshold=0.8, angular_percentiles=(1, 99)):
        self.luminosity_threshold = luminosity_threshold
        self.angular_percentiles = angular_percentiles
        self.target_stain_matrix = None
        self.target_concentrations_stats = None
    
    def fit(self, target_image: np.ndarray):
        """Fit normalizer to target image."""
        self.target_stain_matrix = get_stain_matrix(
            target_image,
            self.luminosity_threshold,
            self.angular_percentiles
        )
        
        # Get target concentration statistics
        od = rgb_to_od(target_image)
        concentrations = self._get_concentrations(od, self.target_stain_matrix)
        
        # Store multiple percentiles for better matching
        self.target_concentrations_stats = {
            'mean': np.mean(concentrations, axis=0),
            'std': np.std(concentrations, axis=0),
            'percentiles': np.percentile(concentrations, [1, 25, 50, 75, 99], axis=0)
        }
        
        return self
    
    def transform(self, source_image: np.ndarray) -> np.ndarray:
        """Transform source image using fitted parameters."""
        if self.target_stain_matrix is None:
            raise ValueError("Normalizer must be fitted first")
        
        # Get source stain matrix
        source_stain_matrix = get_stain_matrix(
            source_image,
            self.luminosity_threshold,
            self.angular_percentiles
        )
        
        # Convert to OD
        source_od = rgb_to_od(source_image)
        
        # Get source concentrations
        source_concentrations = self._get_concentrations(source_od, source_stain_matrix)
        
        # Get source statistics
        source_stats = {
            'mean': np.mean(source_concentrations, axis=0),
            'std': np.std(source_concentrations, axis=0),
            'percentiles': np.percentile(source_concentrations, [1, 25, 50, 75, 99], axis=0)
        }
        
        # Normalize concentrations using histogram matching approach
        normalized_conc = source_concentrations.copy()
        for i in range(2):
            if source_stats['std'][i] > 0:
                # Z-score normalization followed by rescaling
                normalized_conc[:, i] = (source_concentrations[:, i] - source_stats['mean'][i]) / source_stats['std'][i]
                normalized_conc[:, i] = normalized_conc[:, i] * self.target_concentrations_stats['std'][i] + self.target_concentrations_stats['mean'][i]
                
                # Ensure non-negative
                normalized_conc[:, i] = np.maximum(normalized_conc[:, i], 0)
        
        # Reconstruct using target stain matrix
        od_reconstructed = normalized_conc @ self.target_stain_matrix.T
        od_reconstructed = od_reconstructed.reshape(source_image.shape)
        
        # Convert back to RGB
        rgb_normalized = od_to_rgb(od_reconstructed)
        
        return rgb_normalized
    
    def _get_concentrations(self, od: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Get stain concentrations using least squares"""
        od_flat = od.reshape(-1, 3)
        concentrations = od_flat @ np.linalg.pinv(stain_matrix).T
        concentrations = np.maximum(concentrations, 0)
        return concentrations


# Convenience functions
def normalize_reinhard(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quick Reinhard normalization."""
    normalizer = ReinhardNormalizer()
    normalizer.fit(target)
    return normalizer.transform(source)


def normalize_macenko(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quick Macenko normalization."""
    normalizer = MacenkoNormalizer()
    normalizer.fit(target)
    return normalizer.transform(source)


def normalize_vahadane(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quick Vahadane normalization."""
    normalizer = VahadaneNormalizer()
    normalizer.fit(target)
    return normalizer.transform(source)


# Compatibility classes and functions
class BaseNormalizer:
    """Base class for stain normalizers - for compatibility."""
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = False  # GPU not implemented in this version
        
    def fit(self, target_image: np.ndarray):
        raise NotImplementedError
        
    def transform(self, source_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def fit_transform(self, source_image: np.ndarray, target_image: np.ndarray) -> np.ndarray:
        self.fit(target_image)
        return self.transform(source_image)
        
    def set_target_params(self, params: Dict[str, np.ndarray]):
        """Set target parameters from predefined values."""
        pass


class ColorAugmenter:
    """
    Color augmentation utilities for histopathology images.
    
    Provides controlled color variations for data augmentation while
    preserving tissue structure.
    """
    
    def __init__(self, 
                 hue_shift_range: Tuple[float, float] = (-0.05, 0.05),
                 saturation_range: Tuple[float, float] = (0.95, 1.05),
                 brightness_range: Tuple[float, float] = (0.95, 1.05),
                 contrast_range: Tuple[float, float] = (0.95, 1.05)):
        """
        Initialize color augmenter.
        
        Args:
            hue_shift_range: Range for hue shift in HSV space
            saturation_range: Range for saturation scaling
            brightness_range: Range for brightness scaling
            contrast_range: Range for contrast adjustment
        """
        self.hue_shift_range = hue_shift_range
        self.saturation_range = saturation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def augment(self, image: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply random color augmentation to image.
        
        Args:
            image: Input image in RGB format
            seed: Random seed for reproducibility
            
        Returns:
            Augmented image
        """
        if seed is not None:
            np.random.seed(seed)
        
        augmented = image.copy()
        
        # Hue and saturation in HSV space
        if self.hue_shift_range != (0, 0) or self.saturation_range != (1, 1):
            hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Hue shift
            hue_shift = np.random.uniform(*self.hue_shift_range)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift * 180) % 180
            
            # Saturation scaling
            sat_scale = np.random.uniform(*self.saturation_range)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
            
            augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Brightness
        if self.brightness_range != (1, 1):
            brightness_scale = np.random.uniform(*self.brightness_range)
            augmented = np.clip(augmented.astype(np.float32) * brightness_scale, 0, 255).astype(np.uint8)
        
        # Contrast
        if self.contrast_range != (1, 1):
            contrast_scale = np.random.uniform(*self.contrast_range)
            mean = augmented.mean()
            augmented = np.clip((augmented.astype(np.float32) - mean) * contrast_scale + mean, 0, 255).astype(np.uint8)
        
        return augmented


# Constants for compatibility
STAIN_MATRIX_DEFAULT = np.array([
    [0.65, 0.70, 0.29],  # Hematoxylin
    [0.07, 0.99, 0.11],  # Eosin
    [0.29, 0.11, 0.90]   # DAB/Background
])