"""
Working stain normalization implementation that correctly normalizes tissue colors.
Based on the Macenko method with proper color space handling.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class WorkingMacenkoNormalizer:
    """
    A working implementation of Macenko stain normalization that properly handles color normalization.
    """
    
    def __init__(self):
        self.target_stain_matrix = None
        self.target_maxC = None
        
    def get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        """
        Get stain matrix using the Macenko method.
        """
        # Convert to float
        image = image.astype(np.float64)
        
        # Calculate optical density
        # Use log with small epsilon to avoid log(0)
        OD = -np.log((image + 1) / 256)
        
        # Remove data with OD below threshold
        ODhat = OD.reshape((-1, 3))
        mask = np.any(ODhat > 0.15, axis=1)
        ODhat = ODhat[mask]
        
        if len(ODhat) < 100:
            # Return default matrix if not enough pixels
            return np.array([[0.650, 0.072],
                           [0.704, 0.990],
                           [0.286, 0.105]])
        
        # Calculate eigenvectors
        cov = np.cov(ODhat.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Create projection matrix
        if eigenvectors[0, 0] < 0: eigenvectors[:, 0] *= -1
        if eigenvectors[0, 1] < 0: eigenvectors[:, 1] *= -1
        
        # Project data onto the first two PCs
        That = ODhat @ eigenvectors[:, :2]
        
        # Find angle of each point
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        # Find min and max angles
        minPhi = np.percentile(phi, 1)
        maxPhi = np.percentile(phi, 99)
        
        # Convert back to vectors
        vMin = eigenvectors[:, 0] * np.cos(minPhi) + eigenvectors[:, 1] * np.sin(minPhi)
        vMax = eigenvectors[:, 0] * np.cos(maxPhi) + eigenvectors[:, 1] * np.sin(maxPhi)
        
        # Normalize
        if vMin[0] < 0: vMin *= -1
        if vMax[0] < 0: vMax *= -1
        
        # Form matrix
        HE = np.array([vMin, vMax]).T
        
        # Order H and E properly
        # H should have higher blue component
        if HE[2, 0] < HE[2, 1]:
            HE = HE[:, [1, 0]]
            
        return HE
    
    def fit(self, target: np.ndarray):
        """Fit to target image."""
        self.target_stain_matrix = self.get_stain_matrix(target)
        
        # Get target concentrations
        target_od = -np.log((target.astype(np.float64) + 1) / 256)
        C = np.linalg.lstsq(self.target_stain_matrix, target_od.reshape(-1, 3).T, rcond=None)[0].T
        self.target_maxC = np.percentile(C, 99, axis=0)
        
        return self
        
    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transform source image to match target staining."""
        if self.target_stain_matrix is None:
            raise ValueError("Must fit before transform")
            
        # Get source stain matrix
        source_stain_matrix = self.get_stain_matrix(source)
        
        # Convert source to OD and get concentrations
        source_od = -np.log((source.astype(np.float64) + 1) / 256)
        source_C = np.linalg.lstsq(source_stain_matrix, source_od.reshape(-1, 3).T, rcond=None)[0].T
        
        # Get 99th percentile of source concentrations
        source_maxC = np.percentile(source_C, 99, axis=0)
        
        # Normalize concentrations
        normalized_C = source_C.copy()
        for i in range(2):
            normalized_C[:, i] *= (self.target_maxC[i] / source_maxC[i])
            
        # Transform using target stain matrix
        normalized_od = (self.target_stain_matrix @ normalized_C.T).T
        
        # Convert back to RGB
        normalized_rgb = np.exp(-normalized_od) * 256 - 1
        normalized_rgb = np.clip(normalized_rgb, 0, 255)
        normalized_rgb = normalized_rgb.reshape(source.shape).astype(np.uint8)
        
        return normalized_rgb


def working_macenko_normalize(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Simple function interface for Macenko normalization.
    """
    normalizer = WorkingMacenkoNormalizer()
    normalizer.fit(target)
    return normalizer.transform(source)