"""
Classical Tissue Detection Module for Digital Pathology

Provides traditional computer vision methods for tissue detection in WSI
without requiring deep learning models.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List
from scipy.ndimage import sobel, binary_fill_holes, label
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
from skimage.color import rgb2hsv, rgb2gray
from skimage.measure import regionprops


class ClassicalTissueDetector:
    """
    Classical tissue detection using traditional computer vision techniques.
    
    Provides multiple methods for tissue detection including:
    - Otsu thresholding
    - Multi-Otsu thresholding
    - HSV-based detection
    - Gradient-based edge detection
    - Morphological operations
    """
    
    def __init__(self, 
                 method: str = "otsu_hsv",
                 min_tissue_size: int = 1000,
                 morphology_radius: int = 5):
        """
        Initialize tissue detector.
        
        Args:
            method: Detection method ("otsu", "multi_otsu", "hsv", "gradient", "otsu_hsv")
            min_tissue_size: Minimum tissue region size in pixels
            morphology_radius: Radius for morphological operations
        """
        self.method = method.lower()
        self.min_tissue_size = min_tissue_size
        self.morphology_radius = morphology_radius
        
        # HSV thresholds for tissue detection
        self.hsv_thresholds = {
            'h_min': 0, 'h_max': 180,  # Full hue range
            's_min': 20, 's_max': 255,  # Some saturation
            'v_min': 30, 'v_max': 235   # Not too dark or bright
        }
    
    def detect(self, image: np.ndarray, 
               return_labels: bool = False) -> np.ndarray:
        """
        Detect tissue regions in image.
        
        Args:
            image: RGB image
            return_labels: If True, return labeled regions instead of binary mask
            
        Returns:
            Binary mask or labeled regions
        """
        if self.method == "otsu":
            mask = self._detect_otsu(image)
        elif self.method == "multi_otsu":
            mask = self._detect_multi_otsu(image)
        elif self.method == "hsv":
            mask = self._detect_hsv(image)
        elif self.method == "gradient":
            mask = self._detect_gradient(image)
        elif self.method == "otsu_hsv":
            mask = self._detect_otsu_hsv(image)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
        
        # Post-process mask
        mask = self._postprocess_mask(mask)
        
        if return_labels:
            labeled, num_features = label(mask)
            return labeled
        
        return mask
    
    def _detect_otsu(self, image: np.ndarray) -> np.ndarray:
        """Tissue detection using Otsu's threshold on grayscale."""
        # Convert to grayscale
        gray = rgb2gray(image)
        
        # Apply Otsu's threshold
        threshold = threshold_otsu(gray)
        
        # Create binary mask (tissue is darker than background)
        mask = gray < threshold
        
        return mask
    
    def _detect_multi_otsu(self, image: np.ndarray, n_classes: int = 3) -> np.ndarray:
        """Tissue detection using multi-level Otsu thresholding."""
        # Convert to grayscale
        gray = rgb2gray(image)
        
        # Apply multi-Otsu threshold
        thresholds = threshold_multiotsu(gray, classes=n_classes)
        
        # Use lowest threshold (darkest regions are tissue)
        mask = gray < thresholds[0]
        
        return mask
    
    def _detect_hsv(self, image: np.ndarray) -> np.ndarray:
        """Tissue detection using HSV color space thresholding."""
        # Convert to HSV
        hsv = rgb2hsv(image)
        
        # Extract channels
        h = (hsv[:, :, 0] * 180).astype(np.uint8)  # Convert to 0-180 range
        s = (hsv[:, :, 1] * 255).astype(np.uint8)  # Convert to 0-255 range
        v = (hsv[:, :, 2] * 255).astype(np.uint8)  # Convert to 0-255 range
        
        # Apply thresholds
        mask = (
            (h >= self.hsv_thresholds['h_min']) & (h <= self.hsv_thresholds['h_max']) &
            (s >= self.hsv_thresholds['s_min']) & (s <= self.hsv_thresholds['s_max']) &
            (v >= self.hsv_thresholds['v_min']) & (v <= self.hsv_thresholds['v_max'])
        )
        
        return mask
    
    def _detect_gradient(self, image: np.ndarray) -> np.ndarray:
        """Tissue detection using gradient magnitude."""
        # Convert to grayscale
        gray = np.mean(image.astype(np.float32), axis=2)
        
        # Compute gradients
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold gradient magnitude
        threshold = threshold_otsu(magnitude)
        mask = magnitude > threshold
        
        return mask
    
    def _detect_otsu_hsv(self, image: np.ndarray) -> np.ndarray:
        """Combined Otsu and HSV detection for robustness."""
        # Get both masks
        otsu_mask = self._detect_otsu(image)
        hsv_mask = self._detect_hsv(image)
        
        # Combine masks (intersection for conservative, union for aggressive)
        # Using intersection here for better precision
        mask = otsu_mask & hsv_mask
        
        return mask
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up mask."""
        # Close small gaps
        if self.morphology_radius > 0:
            selem = disk(self.morphology_radius)
            mask = binary_closing(mask, selem)
        
        # Fill holes
        mask = binary_fill_holes(mask)
        
        # Remove small objects
        mask = remove_small_objects(mask, min_size=self.min_tissue_size)
        
        # Optional: opening to remove thin connections
        if self.morphology_radius > 2:
            selem_small = disk(self.morphology_radius // 2)
            mask = binary_opening(mask, selem_small)
        
        return mask
    
    def get_tissue_stats(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute tissue statistics.
        
        Args:
            image: RGB image
            
        Returns:
            Dictionary with tissue statistics
        """
        mask = self.detect(image)
        
        total_pixels = mask.size
        tissue_pixels = np.sum(mask)
        
        # Get region properties
        labeled, num_regions = label(mask)
        regions = regionprops(labeled)
        
        # Compute statistics
        stats = {
            'tissue_ratio': tissue_pixels / total_pixels,
            'tissue_pixels': int(tissue_pixels),
            'total_pixels': int(total_pixels),
            'num_regions': num_regions,
            'largest_region_area': max([r.area for r in regions]) if regions else 0,
            'mean_region_area': np.mean([r.area for r in regions]) if regions else 0
        }
        
        return stats
    
    def detect_at_multiple_scales(self, image: np.ndarray, 
                                 scales: List[float] = [1.0, 0.5, 0.25]) -> Dict[int, np.ndarray]:
        """
        Detect tissue at multiple scales for multi-resolution analysis.
        
        Args:
            image: RGB image
            scales: List of scale factors
            
        Returns:
            Dictionary mapping scale to tissue mask
        """
        results = {}
        
        for scale in scales:
            # Resize image
            if scale != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                scaled_image = image
            
            # Detect tissue
            mask = self.detect(scaled_image)
            
            # Resize mask back to original size if needed
            if scale != 1.0:
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), 
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            
            results[scale] = mask
        
        return results
    
    def refine_with_color_filtering(self, image: np.ndarray, 
                                   initial_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Refine tissue detection using color-based filtering.
        
        Args:
            image: RGB image
            initial_mask: Initial tissue mask (if None, compute it)
            
        Returns:
            Refined tissue mask
        """
        if initial_mask is None:
            initial_mask = self.detect(image)
        
        # Extract tissue regions
        tissue_pixels = image[initial_mask]
        
        if len(tissue_pixels) < 100:
            return initial_mask
        
        # Compute color statistics in tissue regions
        mean_color = np.mean(tissue_pixels, axis=0)
        std_color = np.std(tissue_pixels, axis=0)
        
        # Create refined mask based on color similarity
        # Using Mahalanobis-like distance
        diff = image.astype(np.float32) - mean_color
        normalized_diff = diff / (std_color + 1e-6)
        distance = np.sqrt(np.sum(normalized_diff**2, axis=2))
        
        # Threshold based on standard deviations
        refined_mask = distance < 3.0  # Within 3 standard deviations
        
        # Combine with initial mask
        refined_mask = refined_mask & initial_mask
        
        # Clean up
        refined_mask = self._postprocess_mask(refined_mask)
        
        return refined_mask


def detect_tissue(image: np.ndarray, 
                 method: str = "otsu_hsv",
                 min_tissue_size: int = 1000) -> np.ndarray:
    """
    Convenience function for tissue detection.
    
    Args:
        image: RGB image
        method: Detection method
        min_tissue_size: Minimum tissue region size
        
    Returns:
        Binary tissue mask
    """
    detector = ClassicalTissueDetector(method=method, min_tissue_size=min_tissue_size)
    return detector.detect(image)


def get_tissue_bounding_boxes(image: np.ndarray, 
                             min_tissue_size: int = 1000) -> List[Tuple[int, int, int, int]]:
    """
    Get bounding boxes for tissue regions.
    
    Args:
        image: RGB image
        min_tissue_size: Minimum tissue region size
        
    Returns:
        List of bounding boxes (x, y, width, height)
    """
    detector = ClassicalTissueDetector(min_tissue_size=min_tissue_size)
    labeled = detector.detect(image, return_labels=True)
    
    regions = regionprops(labeled)
    bboxes = []
    
    for region in regions:
        y1, x1, y2, x2 = region.bbox
        bboxes.append((x1, y1, x2 - x1, y2 - y1))
    
    return bboxes


def tissue_mask_to_contours(mask: np.ndarray, 
                           min_contour_area: int = 100) -> List[np.ndarray]:
    """
    Convert tissue mask to contours.
    
    Args:
        mask: Binary tissue mask
        min_contour_area: Minimum contour area to keep
        
    Returns:
        List of contour arrays
    """
    # Ensure uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            filtered_contours.append(contour)
    
    return filtered_contours