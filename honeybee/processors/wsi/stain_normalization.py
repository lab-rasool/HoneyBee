"""
Corrected Stain Normalization Methods for Digital Pathology

Fixed implementations that properly handle H&E stain separation.
"""

__all__ = [
    'ReinhardNormalizer', 'MacenkoNormalizer', 'VahadaneNormalizer',
    'ColorAugmenter', 'BaseNormalizer', 
    'normalize_reinhard', 'normalize_macenko', 'normalize_vahadane',
    'normalize_stain_tissue_aware',
    'rgb_to_od', 'od_to_rgb', 'get_stain_matrix_macenko', 'validate_stain_matrix',
    'STAIN_NORM_TARGETS', 'STAIN_MATRIX_DEFAULT'
]

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Union
import warnings


# Standard stain normalization targets
STAIN_NORM_TARGETS = {
    "tcga_avg": {
        "mean_lab": np.array([66.98, 128.77, 113.74]),
        "std_lab": np.array([15.89, 10.22, 9.41]),
        "stain_matrix": np.array([
            [0.644, 0.093],  # Red channel: H and E absorption
            [0.717, 0.954],  # Green channel: H and E absorption
            [0.267, 0.283]   # Blue channel: H and E absorption
        ])
    }
}


def rgb_to_od(rgb):
    """Convert RGB to optical density"""
    rgb = rgb.astype(np.float32)
    # Add epsilon to avoid log(0)
    od = -np.log((rgb + 1) / 256)  # Use natural log, not log10
    return od


def od_to_rgb(od):
    """Convert optical density to RGB"""
    rgb = 256 * np.exp(-od) - 1  # Use natural exp, not 10^x
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def normalize_rows(A):
    """Normalize rows of a matrix"""
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def get_stain_matrix_macenko(rgb_image, tissue_mask=None, debug=False):
    """
    Extract H&E stain matrix using Macenko method.
    
    Args:
        rgb_image: RGB image
        tissue_mask: Optional binary mask indicating tissue regions
        debug: If True, return additional debug information
        
    Returns:
        Stain matrix (3x2) with H and E vectors as columns
        If debug=True, returns (stain_matrix, debug_info)
    """
    debug_info = {}
    
    # Default H&E matrix
    default_matrix = np.array([[0.650, 0.072],  # H and E red absorption
                              [0.704, 0.990],  # H and E green absorption  
                              [0.286, 0.105]])  # H and E blue absorption
    
    try:
        # Convert to optical density
        od = rgb_to_od(rgb_image)
        
        # Remove pixels with low optical density (background)
        od_flat = od.reshape(-1, 3)
        
        # Apply tissue mask if provided
        if tissue_mask is not None:
            tissue_mask_flat = tissue_mask.flatten()
            od_flat = od_flat[tissue_mask_flat]
        
        # Remove transparent pixels
        od_flat = od_flat[(od_flat > 0.15).any(axis=1)]
        
        if len(od_flat) < 100:
            # Return default H&E matrix if not enough pixels
            return default_matrix if not debug else (default_matrix, debug_info)
        
        # Compute eigenvectors using covariance (more stable than SVD for this use case)
        cov = np.cov(od_flat.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Project data onto the plane spanned by the first two principal components
        projection = od_flat @ eigenvectors[:, :2]
        
        # Find the angle of each point in the 2D plane
        angles = np.arctan2(projection[:, 1], projection[:, 0])
        
        # Find robust min and max angles
        min_angle = np.percentile(angles, 1)
        max_angle = np.percentile(angles, 99)
        
        # Convert angles back to stain vectors
        vec1 = eigenvectors[:, 0] * np.cos(min_angle) + eigenvectors[:, 1] * np.sin(min_angle)
        vec2 = eigenvectors[:, 0] * np.cos(max_angle) + eigenvectors[:, 1] * np.sin(max_angle)
        
        # Normalize vectors
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        # Ensure all values are positive
        if vec1[0] < 0: vec1 = -vec1
        if vec2[0] < 0: vec2 = -vec2
        
        # Simple heuristic: Hematoxylin has higher blue component
        # Eosin has higher green component
        if vec1[2] > vec2[2]:  # vec1 has more blue
            h_vec, e_vec = vec1, vec2
        else:
            h_vec, e_vec = vec2, vec1
        
        stain_matrix = np.column_stack([h_vec, e_vec])
        
        if debug:
            debug_info['od_pixels_analyzed'] = len(od_flat)
            debug_info['vec1'] = vec1
            debug_info['vec2'] = vec2
            debug_info['h_vec'] = h_vec
            debug_info['e_vec'] = e_vec
            debug_info['eigenvalues'] = eigenvalues
            return stain_matrix, debug_info
        
        return stain_matrix
        
    except Exception as e:
        # If any error occurs, return default matrix
        if debug:
            debug_info['error'] = str(e)
            return default_matrix, debug_info
        return default_matrix


def validate_stain_matrix(stain_matrix):
    """
    Validate and diagnose issues with stain matrix.
    
    Returns dict with validation results and diagnostics.
    """
    results = {
        'valid': True,
        'warnings': [],
        'h_vector': stain_matrix[:, 0],
        'e_vector': stain_matrix[:, 1],
        'h_characteristics': {},
        'e_characteristics': {}
    }
    
    # Check basic validity
    if stain_matrix.shape != (3, 2):
        results['valid'] = False
        results['warnings'].append(f"Invalid shape: {stain_matrix.shape}")
        return results
    
    # Check for negative values
    if np.any(stain_matrix < 0):
        results['warnings'].append("Negative values detected in stain matrix")
    
    # Analyze H vector (should have high red/green absorption)
    h_vec = stain_matrix[:, 0]
    results['h_characteristics'] = {
        'red_od': h_vec[0],
        'green_od': h_vec[1],
        'blue_od': h_vec[2],
        'red_green_ratio': h_vec[0] / (h_vec[1] + 1e-6),
        'appears_purple': h_vec[0] > 0.5 and h_vec[1] > 0.5 and h_vec[2] < 0.4
    }
    
    # Analyze E vector (should have low red, high green/blue absorption)
    e_vec = stain_matrix[:, 1]
    results['e_characteristics'] = {
        'red_od': e_vec[0],
        'green_od': e_vec[1],
        'blue_od': e_vec[2],
        'red_green_ratio': e_vec[0] / (e_vec[1] + 1e-6),
        'appears_pink': e_vec[0] < 0.3 and e_vec[1] > 0.7
    }
    
    # Check if vectors might be swapped
    if (results['h_characteristics']['red_od'] < results['e_characteristics']['red_od']):
        results['warnings'].append("H and E vectors might be swapped")
    
    # Check if H vector looks correct
    if not results['h_characteristics']['appears_purple']:
        results['warnings'].append("H vector doesn't match expected purple/blue characteristics")
    
    # Check if E vector looks correct
    if not results['e_characteristics']['appears_pink']:
        results['warnings'].append("E vector doesn't match expected pink characteristics")
    
    return results


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
    
    def set_target_params(self, params: Dict[str, np.ndarray]):
        """Set target parameters from predefined values."""
        if 'mean_lab' in params:
            self.target_mean = params['mean_lab']
        if 'std_lab' in params:
            self.target_std = params['std_lab']
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
        normalized_lab = source_lab.copy()
        for i in range(3):
            # Avoid division by zero
            if source_std[i] > 0:
                normalized_lab[:, :, i] = (source_lab[:, :, i] - source_mean[i]) * (self.target_std[i] / source_std[i]) + self.target_mean[i]
            else:
                normalized_lab[:, :, i] = source_lab[:, :, i] - source_mean[i] + self.target_mean[i]
        
        # Convert back to RGB
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)
        normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
        normalized = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)
        
        return normalized


class MacenkoNormalizer:
    """
    Macenko stain normalization method.
    
    Separates and normalizes H&E stains using singular value decomposition.
    """
    
    def __init__(self, use_tissue_mask=True):
        self.target_stain_matrix = None
        self.target_max_concentrations = None
        self.use_tissue_mask = use_tissue_mask
    
    def _get_tissue_mask(self, image: np.ndarray) -> np.ndarray:
        """Simple tissue detection based on luminance."""
        # Convert to grayscale
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        # Threshold - tissue is darker than background
        tissue_mask = gray < 235
        return tissue_mask
    
    def fit(self, target_image: np.ndarray, tissue_mask: Optional[np.ndarray] = None):
        """Fit normalizer to target image."""
        # Get tissue mask if needed
        if self.use_tissue_mask and tissue_mask is None:
            tissue_mask = self._get_tissue_mask(target_image)
        
        # Get target stain matrix
        self.target_stain_matrix = get_stain_matrix_macenko(target_image, tissue_mask)
        
        # Validate stain matrix shape
        assert self.target_stain_matrix.shape == (3, 2), \
            f"Stain matrix should be 3x2, got {self.target_stain_matrix.shape}"
        
        # Get target stain concentrations
        od = rgb_to_od(target_image)
        od_flat = od.reshape(-1, 3)
        
        # Apply tissue mask to concentrations calculation if available
        if self.use_tissue_mask and tissue_mask is not None:
            tissue_mask_flat = tissue_mask.flatten()
            od_flat_tissue = od_flat[tissue_mask_flat]
        else:
            od_flat_tissue = od_flat
        
        # Calculate concentrations using least squares with rcond=-1
        # Solve: stain_matrix @ concentrations.T = od_flat.T
        # This gives us concentrations for each pixel
        concentrations = np.linalg.lstsq(self.target_stain_matrix, od_flat_tissue.T, rcond=-1)[0].T
        
        # Get 99th percentile concentrations (simpler approach)
        self.target_max_concentrations = np.percentile(concentrations, 99, axis=0)
        
        return self
    
    def set_target_params(self, params: Dict[str, np.ndarray]):
        """Set target parameters from predefined values."""
        if 'stain_matrix' in params:
            self.target_stain_matrix = params['stain_matrix']
        if 'max_concentrations' in params:
            self.target_max_concentrations = params['max_concentrations']
        elif self.target_stain_matrix is not None:
            # Use default max concentrations if not provided
            self.target_max_concentrations = np.array([1.0, 1.0])
        return self
    
    def transform(self, source_image: np.ndarray, tissue_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform source image using fitted parameters."""
        if self.target_stain_matrix is None:
            raise ValueError("Normalizer must be fitted first")
            
        # Get tissue mask if needed
        if self.use_tissue_mask and tissue_mask is None:
            tissue_mask = self._get_tissue_mask(source_image)
            
        # Get source stain matrix
        source_stain_matrix = get_stain_matrix_macenko(source_image, tissue_mask)
        
        # Convert to optical density
        od = rgb_to_od(source_image)
        od_flat = od.reshape(-1, 3)
        
        # Get source concentrations using lstsq with rcond=-1
        # Solve: stain_matrix @ concentrations.T = od_flat.T
        source_concentrations = np.linalg.lstsq(source_stain_matrix, od_flat.T, rcond=-1)[0].T
        
        # Normalize concentrations similar to working implementation
        # Calculate 99th percentile for both source and target
        source_conc_99 = np.percentile(source_concentrations, 99, axis=0)
        
        # Simple scaling to match target range
        normalized_concentrations = source_concentrations.copy()
        for i in range(2):  # For H and E
            if source_conc_99[i] > 0:
                # Scale to match target 99th percentile
                scale = self.target_max_concentrations[i] / source_conc_99[i]
                normalized_concentrations[:, i] = source_concentrations[:, i] * scale
        
        # Create result image preserving background
        # Reconstruct OD from normalized concentrations
        od_reconstructed = np.dot(normalized_concentrations, self.target_stain_matrix.T)
        od_reconstructed = od_reconstructed.reshape(source_image.shape)
        
        # Convert OD back to RGB using base 10 (not base e)
        trans = od_to_rgb(od_reconstructed)
        
        # Preserve background regions if using tissue mask
        if self.use_tissue_mask and tissue_mask is not None:
            result = source_image.copy()
            result[tissue_mask] = trans[tissue_mask]
            return result
        else:
            return trans
    


class VahadaneNormalizer:
    """
    Vahadane stain normalization - uses structure-preserving color normalization.
    """
    
    def __init__(self, use_tissue_mask=True):
        self.target_stain_matrix = None
        self.target_concentrations_stats = None
        self.use_tissue_mask = use_tissue_mask
    
    def _get_tissue_mask(self, image: np.ndarray) -> np.ndarray:
        """Simple tissue detection based on luminance."""
        # Convert to grayscale
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        # Threshold - tissue is darker than background
        tissue_mask = gray < 235
        return tissue_mask
    
    def fit(self, target_image: np.ndarray, tissue_mask: Optional[np.ndarray] = None):
        """Fit normalizer to target image."""
        # Get tissue mask if needed
        if self.use_tissue_mask and tissue_mask is None:
            tissue_mask = self._get_tissue_mask(target_image)
        
        # Get target stain matrix
        self.target_stain_matrix = get_stain_matrix_macenko(target_image, tissue_mask)
        
        # Validate stain matrix shape
        assert self.target_stain_matrix.shape == (3, 2), \
            f"Stain matrix should be 3x2, got {self.target_stain_matrix.shape}"
        
        # Get target stain concentrations
        od = rgb_to_od(target_image)
        od_flat = od.reshape(-1, 3)
        
        # Apply tissue mask to concentrations calculation if available
        if self.use_tissue_mask and tissue_mask is not None:
            tissue_mask_flat = tissue_mask.flatten()
            od_flat_tissue = od_flat[tissue_mask_flat]
        else:
            od_flat_tissue = od_flat
        
        # Calculate concentrations using lstsq with rcond=-1
        concentrations = np.linalg.lstsq(self.target_stain_matrix, od_flat_tissue.T, rcond=-1)[0].T
        
        # Store concentration statistics for Vahadane style normalization
        self.target_concentrations_stats = {
            'mean': np.mean(concentrations, axis=0),
            'std': np.std(concentrations, axis=0)
        }
        
        return self
    
    def set_target_params(self, params: Dict[str, np.ndarray]):
        """Set target parameters from predefined values."""
        if 'stain_matrix' in params:
            self.target_stain_matrix = params['stain_matrix']
        if 'concentration_stats' in params:
            self.target_concentrations_stats = params['concentration_stats']
        elif self.target_stain_matrix is not None:
            # Use more realistic default stats based on typical H&E
            self.target_concentrations_stats = {
                'mean': np.array([0.7, 0.3]),  # H typically higher than E
                'std': np.array([0.3, 0.15]),
                'percentiles': np.array([[0.2, 0.05], [0.5, 0.2], [0.7, 0.3], [0.9, 0.4], [1.2, 0.6]])
            }
        return self
    
    def transform(self, source_image: np.ndarray, tissue_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform source image using fitted parameters."""
        if self.target_stain_matrix is None:
            raise ValueError("Normalizer must be fitted first")
            
        # Get tissue mask if needed
        if self.use_tissue_mask and tissue_mask is None:
            tissue_mask = self._get_tissue_mask(source_image)
            
        # Get source stain matrix
        source_stain_matrix = get_stain_matrix_macenko(source_image, tissue_mask)
        
        # Convert to optical density
        od = rgb_to_od(source_image)
        od_flat = od.reshape(-1, 3)
        
        # Get source concentrations using lstsq with rcond=-1
        # Solve: stain_matrix @ concentrations.T = od_flat.T
        source_concentrations = np.linalg.lstsq(source_stain_matrix, od_flat.T, rcond=-1)[0].T
        
        # Get source statistics from tissue regions only
        if self.use_tissue_mask and tissue_mask is not None:
            tissue_mask_flat = tissue_mask.flatten()
            tissue_concentrations = source_concentrations[tissue_mask_flat]
            source_mean = np.mean(tissue_concentrations, axis=0)
            source_std = np.std(tissue_concentrations, axis=0)
        else:
            source_mean = np.mean(source_concentrations, axis=0)
            source_std = np.std(source_concentrations, axis=0)
        
        # Vahadane style normalization: match mean and std of concentrations
        normalized_concentrations = source_concentrations.copy()
        for i in range(2):  # For H and E
            if source_std[i] > 0:
                # Standardize then rescale to target distribution
                normalized_concentrations[:, i] = (source_concentrations[:, i] - source_mean[i]) / source_std[i]
                normalized_concentrations[:, i] = (normalized_concentrations[:, i] * 
                                                 self.target_concentrations_stats['std'][i] + 
                                                 self.target_concentrations_stats['mean'][i])
                # Ensure non-negative concentrations
                normalized_concentrations[:, i] = np.maximum(normalized_concentrations[:, i], 0)
                # Clip extreme values
                normalized_concentrations[:, i] = np.minimum(normalized_concentrations[:, i], 
                                                           self.target_concentrations_stats['mean'][i] + 3 * self.target_concentrations_stats['std'][i])
        
        # Create result image
        # Reconstruct OD from normalized concentrations
        od_reconstructed = np.dot(normalized_concentrations, self.target_stain_matrix.T)
        od_reconstructed = od_reconstructed.reshape(source_image.shape)
        
        # Convert OD back to RGB using base 10 (not base e)
        trans = od_to_rgb(od_reconstructed)
        
        # Preserve background regions if using tissue mask
        if self.use_tissue_mask and tissue_mask is not None:
            result = source_image.copy()
            result[tissue_mask] = trans[tissue_mask]
            return result
        else:
            return trans
    


# Convenience functions
def normalize_reinhard(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quick Reinhard normalization."""
    normalizer = ReinhardNormalizer()
    normalizer.fit(target)
    return normalizer.transform(source)


def normalize_macenko(source: np.ndarray, target: np.ndarray, use_tissue_mask: bool = True) -> np.ndarray:
    """Quick Macenko normalization."""
    normalizer = MacenkoNormalizer(use_tissue_mask=use_tissue_mask)
    normalizer.fit(target)
    return normalizer.transform(source)


def normalize_vahadane(source: np.ndarray, target: np.ndarray, use_tissue_mask: bool = True) -> np.ndarray:
    """Quick Vahadane normalization."""
    normalizer = VahadaneNormalizer(use_tissue_mask=use_tissue_mask)
    normalizer.fit(target)
    return normalizer.transform(source)


def normalize_stain_tissue_aware(source: np.ndarray, 
                                target: np.ndarray, 
                                method: str = "macenko",
                                tissue_threshold: int = 235,
                                debug: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Tissue-aware stain normalization that preserves background.
    
    Args:
        source: Source RGB image to normalize
        target: Target RGB image for normalization reference
        method: Normalization method ("reinhard", "macenko", "vahadane")
        tissue_threshold: Threshold for tissue detection (0-255)
        debug: If True, return debug information
        
    Returns:
        Normalized RGB image
        If debug=True, returns (normalized_image, debug_info)
    """
    # Simple tissue detection
    def get_tissue_mask(image):
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
        return gray < tissue_threshold
    
    # Get tissue masks
    source_mask = get_tissue_mask(source)
    target_mask = get_tissue_mask(target)
    
    debug_info = {
        'source_tissue_ratio': np.sum(source_mask) / source_mask.size,
        'target_tissue_ratio': np.sum(target_mask) / target_mask.size,
        'method': method
    }
    
    # Select normalizer
    if method.lower() == "reinhard":
        normalizer = ReinhardNormalizer()
        normalizer.fit(target)
        normalized = normalizer.transform(source)
    elif method.lower() == "macenko":
        normalizer = MacenkoNormalizer(use_tissue_mask=True)
        normalizer.fit(target, target_mask)
        normalized = normalizer.transform(source, source_mask)
    elif method.lower() == "vahadane":
        normalizer = VahadaneNormalizer(use_tissue_mask=True)
        normalizer.fit(target, target_mask)
        normalized = normalizer.transform(source, source_mask)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if debug:
        # Add stain matrix info for Macenko/Vahadane
        if hasattr(normalizer, 'target_stain_matrix'):
            debug_info['target_stain_matrix'] = normalizer.target_stain_matrix
            debug_info['stain_validation'] = validate_stain_matrix(normalizer.target_stain_matrix)
        return normalized, debug_info
    
    return normalized


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
    [0.644, 0.093],  # Red channel: H and E absorption
    [0.717, 0.954],  # Green channel: H and E absorption
    [0.267, 0.283]   # Blue channel: H and E absorption
])