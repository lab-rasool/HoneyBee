"""
Fixed stain normalization implementation based on the working version
"""
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Union


def normalize_macenko_fixed(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Macenko stain normalization - simplified version that works correctly.
    Based on the working implementation from results/staining.
    """
    
    def get_stain_matrix(image):
        """Extract stain matrix using Macenko method."""
        # Convert to optical density
        od = -np.log((image.astype(np.float32) + 1) / 256)
        
        # Remove transparent pixels
        od_flat = od.reshape(-1, 3)
        od_flat = od_flat[(od_flat > 0.15).any(axis=1)]
        
        if len(od_flat) < 100:
            # Default stain matrix
            return np.array([[0.650, 0.072],
                           [0.704, 0.990],
                           [0.286, 0.105]])
        
        # Apply SVD
        _, _, V = np.linalg.svd(od_flat.T, full_matrices=False)
        
        # Project data onto the plane
        projection = od_flat @ V[:2].T
        
        # Find angle of each point
        angles = np.arctan2(projection[:, 1], projection[:, 0])
        
        # Find robust min and max angles
        min_angle = np.percentile(angles, 1)
        max_angle = np.percentile(angles, 99)
        
        # Convert angles back to stain vectors
        vec1 = V[0] * np.cos(min_angle) + V[1] * np.sin(min_angle)
        vec2 = V[0] * np.cos(max_angle) + V[1] * np.sin(max_angle)
        
        # Normalize
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        # Make positive
        if vec1[0] < 0: vec1 = -vec1
        if vec2[0] < 0: vec2 = -vec2
        
        # Order by blue channel (hematoxylin has more blue)
        if vec1[2] > vec2[2]:
            return np.column_stack([vec1, vec2])
        else:
            return np.column_stack([vec2, vec1])
    
    # Get stain matrices
    stain_matrix_target = get_stain_matrix(target)
    stain_matrix_source = get_stain_matrix(source)
    
    # Convert source to OD
    od_source = -np.log((source.astype(np.float32) + 1) / 256)
    
    # Get source concentrations
    source_concentrations = np.linalg.lstsq(
        stain_matrix_source, 
        od_source.reshape(-1, 3).T, 
        rcond=None
    )[0].T
    
    # Get concentration statistics for normalization
    conc_max_source = np.percentile(source_concentrations, 99, axis=0)
    
    # Convert target to OD
    od_target = -np.log((target.astype(np.float32) + 1) / 256)
    
    # Get target concentrations
    target_concentrations = np.linalg.lstsq(
        stain_matrix_target,
        od_target.reshape(-1, 3).T,
        rcond=None
    )[0].T
    
    # Get target statistics
    conc_max_target = np.percentile(target_concentrations, 99, axis=0)
    
    # Normalize concentrations
    normalized_concentrations = source_concentrations.copy()
    for i in range(2):
        if conc_max_source[i] > 0:
            normalized_concentrations[:, i] = (
                source_concentrations[:, i] * conc_max_target[i] / conc_max_source[i]
            )
    
    # Transform using target stain matrix
    od_normalized = normalized_concentrations @ stain_matrix_target.T
    
    # Convert back to RGB
    normalized = np.exp(-od_normalized) * 256 - 1
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    normalized = normalized.reshape(source.shape)
    
    return normalized


def normalize_with_tissue_mask(source: np.ndarray, 
                              target: np.ndarray,
                              threshold: int = 235) -> np.ndarray:
    """
    Normalize with tissue mask to preserve background.
    """
    # Simple tissue detection
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    tissue_mask = gray < threshold
    
    # Normalize
    normalized = normalize_macenko_fixed(source, target)
    
    # Preserve background
    result = source.copy()
    result[tissue_mask] = normalized[tissue_mask]
    
    return result


class MacenkoNormalizerFixed:
    """Fixed Macenko normalizer that works correctly."""
    
    def __init__(self, use_tissue_mask=True):
        self.target_image = None
        self.use_tissue_mask = use_tissue_mask
        
    def fit(self, target: np.ndarray):
        """Fit to target image."""
        self.target_image = target
        return self
        
    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform source image."""
        if self.target_image is None:
            raise ValueError("Must fit before transform")
            
        if self.use_tissue_mask:
            return normalize_with_tissue_mask(source, self.target_image)
        else:
            return normalize_macenko_fixed(source, self.target_image)