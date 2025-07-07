"""
Radiology Dataset Management

Dataset class for managing collections of medical images with lazy loading support.
"""

import os
import numpy as np
import pydicom
from typing import Union, List, Tuple, Optional
from pathlib import Path
import logging

from .loader import DicomLoader, NiftiLoader
from .metadata import ImageMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RadiologyDataset:
    """Dataset class for managing collections of medical images
    
    Features:
        - Automatic discovery of medical images
        - Support for DICOM and NIfTI formats
        - Lazy loading for memory efficiency
        - Filtering by modality
        - Batch loading capabilities
    """
    
    def __init__(self, root_dir: Union[str, Path], 
                 modality: Optional[str] = None,
                 lazy_load: bool = True,
                 cache_size: int = 10):
        """Initialize dataset
        
        Args:
            root_dir: Root directory containing medical images
            modality: Filter by specific modality (optional)
            lazy_load: Use memory-efficient lazy loading
            cache_size: Number of images to cache in memory
        """
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.lazy_load = lazy_load
        self.cache_size = cache_size
        
        self.dicom_loader = DicomLoader(lazy_load=lazy_load)
        self.nifti_loader = NiftiLoader()
        
        self._index = []
        self._cache = {}
        self._cache_order = []
        
        self._build_index()
    
    def _build_index(self):
        """Build index of all medical images"""
        logger.info(f"Building dataset index from {self.root_dir}")
        
        # Find all potential medical images
        for file in self.root_dir.rglob("*"):
            if file.is_file():
                # Check file extension
                if file.suffix.lower() in ['.dcm', '.dicom']:
                    self._index.append(('dicom', file))
                elif file.suffix.lower() in ['.nii', '.nii.gz']:
                    self._index.append(('nifti', file))
                else:
                    # Try to detect DICOM without extension
                    try:
                        pydicom.dcmread(str(file), stop_before_pixels=True)
                        self._index.append(('dicom', file))
                    except:
                        pass
        
        # Apply modality filter if specified
        if self.modality:
            self._filter_by_modality()
        
        logger.info(f"Found {len(self._index)} medical images")
    
    def _filter_by_modality(self):
        """Filter index by modality"""
        filtered_index = []
        
        for file_type, filepath in self._index:
            try:
                if file_type == 'dicom':
                    ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
                    if hasattr(ds, 'Modality') and ds.Modality == self.modality:
                        filtered_index.append((file_type, filepath))
                # NIfTI files don't store modality, so skip filtering
                elif file_type == 'nifti' and self.modality is None:
                    filtered_index.append((file_type, filepath))
            except:
                pass
        
        self._index = filtered_index
    
    def __len__(self):
        """Get number of images in dataset"""
        return len(self._index)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ImageMetadata]:
        """Get image and metadata by index
        
        Args:
            idx: Index of image to load
            
        Returns:
            Tuple of (image array, metadata)
        """
        if idx < 0 or idx >= len(self._index):
            raise IndexError(f"Index {idx} out of range [0, {len(self._index)})")
        
        # Check cache
        if idx in self._cache:
            # Move to end of cache order (LRU)
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            return self._cache[idx]
        
        # Load image
        file_type, filepath = self._index[idx]
        
        if file_type == 'dicom':
            data, metadata = self.dicom_loader.load_file(filepath)
        elif file_type == 'nifti':
            data, metadata = self.nifti_loader.load_file(filepath)
        
        # Add to cache
        self._add_to_cache(idx, (data, metadata))
        
        return data, metadata
    
    def _add_to_cache(self, idx: int, data: Tuple[np.ndarray, ImageMetadata]):
        """Add item to cache with LRU eviction"""
        if len(self._cache) >= self.cache_size:
            # Remove oldest item
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
        
        self._cache[idx] = data
        self._cache_order.append(idx)
    
    def get_metadata(self, idx: int) -> ImageMetadata:
        """Get metadata without loading pixel data
        
        Args:
            idx: Index of image
            
        Returns:
            ImageMetadata object
        """
        file_type, filepath = self._index[idx]
        
        if file_type == 'dicom':
            ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
            return self.dicom_loader._extract_metadata(ds)
        elif file_type == 'nifti':
            import nibabel as nib
            nii = nib.load(str(filepath))
            return self.nifti_loader._extract_metadata(nii)
    
    def get_batch(self, indices: List[int]) -> Tuple[List[np.ndarray], List[ImageMetadata]]:
        """Load multiple images as batch
        
        Args:
            indices: List of indices to load
            
        Returns:
            Tuple of (list of arrays, list of metadata)
        """
        images = []
        metadata_list = []
        
        for idx in indices:
            image, metadata = self[idx]
            images.append(image)
            metadata_list.append(metadata)
        
        return images, metadata_list
    
    def filter_by_modality(self, modality: str) -> 'RadiologyDataset':
        """Create filtered dataset by modality
        
        Args:
            modality: Modality to filter by (CT, MR, PT, etc.)
            
        Returns:
            New filtered dataset
        """
        filtered = RadiologyDataset(self.root_dir, modality, self.lazy_load)
        return filtered
    
    def get_modalities(self) -> List[str]:
        """Get list of unique modalities in dataset
        
        Returns:
            List of modality strings
        """
        modalities = set()
        
        for file_type, filepath in self._index:
            try:
                if file_type == 'dicom':
                    ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
                    if hasattr(ds, 'Modality'):
                        modalities.add(ds.Modality)
            except:
                pass
        
        return sorted(list(modalities))
    
    def get_file_paths(self) -> List[Path]:
        """Get all file paths in dataset
        
        Returns:
            List of file paths
        """
        return [filepath for _, filepath in self._index]
    
    def summary(self) -> dict:
        """Get dataset summary statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_images': len(self._index),
            'dicom_files': sum(1 for ft, _ in self._index if ft == 'dicom'),
            'nifti_files': sum(1 for ft, _ in self._index if ft == 'nifti'),
            'modalities': self.get_modalities(),
            'root_directory': str(self.root_dir)
        }
        
        return stats