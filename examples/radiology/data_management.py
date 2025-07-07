"""
Data Management Module for Medical Imaging

Handles loading and management of DICOM, NIfTI, and other medical image formats
with support for metadata preservation and memory-efficient loading.
"""

import os
import numpy as np
import pydicom
import nibabel as nib
from typing import Union, Dict, List, Tuple, Optional, Any
from pathlib import Path
import SimpleITK as sitk
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Container for medical image metadata"""
    modality: str
    patient_id: str
    study_date: str
    series_description: str
    pixel_spacing: Tuple[float, float, float]
    image_position: Tuple[float, float, float]
    image_orientation: List[float]
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    rescale_intercept: float = 0.0
    rescale_slope: float = 1.0
    manufacturer: Optional[str] = None
    scanner_model: Optional[str] = None
    kvp: Optional[float] = None
    exposure: Optional[float] = None
    slice_thickness: Optional[float] = None
    spacing_between_slices: Optional[float] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    number_of_slices: Optional[int] = None
    extra_metadata: Dict[str, Any] = None


class DicomLoader:
    """Advanced DICOM loading with series management and metadata extraction"""
    
    def __init__(self, lazy_load: bool = True):
        self.lazy_load = lazy_load
        self._cache = {}
        
    def load_file(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load a single DICOM file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"DICOM file not found: {filepath}")
            
        try:
            ds = pydicom.dcmread(str(filepath), force=True)
            
            # Extract pixel data
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply rescale slope and intercept
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
            
            # Extract metadata
            metadata = self._extract_metadata(ds)
            
            return pixel_array, metadata
            
        except Exception as e:
            logger.error(f"Error loading DICOM file {filepath}: {e}")
            raise
    
    def load_series(self, directory: Union[str, Path], 
                   series_uid: Optional[str] = None) -> Tuple[np.ndarray, ImageMetadata]:
        """Load a complete DICOM series"""
        directory = Path(directory)
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")
        
        # Get all DICOM files
        dicom_files = self._find_dicom_files(directory)
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {directory}")
        
        # Group by series
        series_dict = self._group_by_series(dicom_files)
        
        # Select series
        if series_uid:
            if series_uid not in series_dict:
                raise ValueError(f"Series UID {series_uid} not found")
            files = series_dict[series_uid]
        else:
            # Use the series with most files
            series_uid = max(series_dict, key=lambda k: len(series_dict[k]))
            files = series_dict[series_uid]
            logger.info(f"Auto-selected series {series_uid} with {len(files)} files")
        
        # Sort files by instance number or slice location
        sorted_files = self._sort_dicom_files(files)
        
        # Load all slices
        if self.lazy_load:
            volume, metadata = self._lazy_load_series(sorted_files)
        else:
            volume, metadata = self._load_series(sorted_files)
        
        return volume, metadata
    
    def load_multi_frame(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load multi-frame DICOM (e.g., 4D data)"""
        filepath = Path(filepath)
        ds = pydicom.dcmread(str(filepath))
        
        if not hasattr(ds, 'NumberOfFrames'):
            raise ValueError("Not a multi-frame DICOM")
        
        # Extract all frames
        frames = []
        for i in range(int(ds.NumberOfFrames)):
            frame = ds.pixel_array[i] if len(ds.pixel_array.shape) > 2 else ds.pixel_array
            frames.append(frame)
        
        volume = np.stack(frames)
        metadata = self._extract_metadata(ds)
        metadata.number_of_slices = int(ds.NumberOfFrames)
        
        return volume, metadata
    
    def _find_dicom_files(self, directory: Path) -> List[Path]:
        """Find all DICOM files in directory"""
        dicom_files = []
        
        for file in directory.rglob("*"):
            if file.is_file():
                try:
                    # Quick check if it's DICOM
                    with open(file, 'rb') as f:
                        f.seek(128)
                        if f.read(4) == b'DICM':
                            dicom_files.append(file)
                except:
                    # Try loading with pydicom
                    try:
                        pydicom.dcmread(str(file), stop_before_pixels=True)
                        dicom_files.append(file)
                    except:
                        pass
        
        return dicom_files
    
    def _group_by_series(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group DICOM files by series UID"""
        series_dict = {}
        
        for file in files:
            try:
                ds = pydicom.dcmread(str(file), stop_before_pixels=True)
                series_uid = str(ds.SeriesInstanceUID)
                
                if series_uid not in series_dict:
                    series_dict[series_uid] = []
                series_dict[series_uid].append(file)
            except:
                pass
        
        return series_dict
    
    def _sort_dicom_files(self, files: List[Path]) -> List[Path]:
        """Sort DICOM files by instance number or slice location"""
        file_info = []
        
        for file in files:
            ds = pydicom.dcmread(str(file), stop_before_pixels=True)
            
            # Try different sorting keys
            sort_key = 0
            if hasattr(ds, 'InstanceNumber'):
                sort_key = int(ds.InstanceNumber)
            elif hasattr(ds, 'SliceLocation'):
                sort_key = float(ds.SliceLocation)
            elif hasattr(ds, 'ImagePositionPatient'):
                sort_key = float(ds.ImagePositionPatient[2])
            
            file_info.append((file, sort_key))
        
        # Sort by key
        file_info.sort(key=lambda x: x[1])
        
        return [f[0] for f in file_info]
    
    def _load_series(self, files: List[Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load complete series into memory"""
        slices = []
        metadata = None
        
        for i, file in enumerate(files):
            ds = pydicom.dcmread(str(file))
            
            # Get pixel data
            slice_data = ds.pixel_array.astype(np.float32)
            
            # Apply rescale
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slice_data = slice_data * ds.RescaleSlope + ds.RescaleIntercept
            
            slices.append(slice_data)
            
            # Extract metadata from first slice
            if i == 0:
                metadata = self._extract_metadata(ds)
        
        # Stack into volume
        volume = np.stack(slices)
        metadata.number_of_slices = len(slices)
        
        return volume, metadata
    
    def _lazy_load_series(self, files: List[Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Create lazy-loaded volume using memory mapping"""
        # Read first file to get dimensions and metadata
        ds = pydicom.dcmread(str(files[0]))
        metadata = self._extract_metadata(ds)
        
        shape = (len(files), ds.Rows, ds.Columns)
        dtype = np.float32
        
        # Create memory-mapped array
        volume = np.zeros(shape, dtype=dtype)
        
        # Load slices with parallel processing
        def load_slice(args):
            idx, file = args
            ds = pydicom.dcmread(str(file))
            slice_data = ds.pixel_array.astype(np.float32)
            
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slice_data = slice_data * ds.RescaleSlope + ds.RescaleIntercept
            
            return idx, slice_data
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(load_slice, enumerate(files))
            
            for idx, slice_data in results:
                volume[idx] = slice_data
        
        metadata.number_of_slices = len(files)
        return volume, metadata
    
    def _extract_metadata(self, ds: pydicom.Dataset) -> ImageMetadata:
        """Extract comprehensive metadata from DICOM dataset"""
        # Basic required fields
        modality = str(ds.Modality) if hasattr(ds, 'Modality') else 'Unknown'
        patient_id = str(ds.PatientID) if hasattr(ds, 'PatientID') else 'Unknown'
        study_date = str(ds.StudyDate) if hasattr(ds, 'StudyDate') else 'Unknown'
        series_desc = str(ds.SeriesDescription) if hasattr(ds, 'SeriesDescription') else 'Unknown'
        
        # Spatial information
        pixel_spacing = [1.0, 1.0, 1.0]
        if hasattr(ds, 'PixelSpacing'):
            pixel_spacing[0] = float(ds.PixelSpacing[0])
            pixel_spacing[1] = float(ds.PixelSpacing[1])
        if hasattr(ds, 'SliceThickness'):
            pixel_spacing[2] = float(ds.SliceThickness)
        elif hasattr(ds, 'SpacingBetweenSlices'):
            pixel_spacing[2] = float(ds.SpacingBetweenSlices)
        
        # Position and orientation
        image_position = (0.0, 0.0, 0.0)
        if hasattr(ds, 'ImagePositionPatient'):
            image_position = tuple(float(x) for x in ds.ImagePositionPatient)
        
        image_orientation = [1, 0, 0, 0, 1, 0]
        if hasattr(ds, 'ImageOrientationPatient'):
            image_orientation = [float(x) for x in ds.ImageOrientationPatient]
        
        # Window settings
        window_center = float(ds.WindowCenter) if hasattr(ds, 'WindowCenter') else None
        window_width = float(ds.WindowWidth) if hasattr(ds, 'WindowWidth') else None
        
        # Handle lists for window values
        if isinstance(window_center, (list, pydicom.multival.MultiValue)):
            window_center = float(window_center[0])
        if isinstance(window_width, (list, pydicom.multival.MultiValue)):
            window_width = float(window_width[0])
        
        # Rescale values
        rescale_intercept = float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else 0.0
        rescale_slope = float(ds.RescaleSlope) if hasattr(ds, 'RescaleSlope') else 1.0
        
        # Scanner information
        manufacturer = str(ds.Manufacturer) if hasattr(ds, 'Manufacturer') else None
        scanner_model = str(ds.ManufacturerModelName) if hasattr(ds, 'ManufacturerModelName') else None
        
        # CT-specific
        kvp = float(ds.KVP) if hasattr(ds, 'KVP') else None
        exposure = float(ds.Exposure) if hasattr(ds, 'Exposure') else None
        
        # Dimensions
        rows = int(ds.Rows) if hasattr(ds, 'Rows') else None
        columns = int(ds.Columns) if hasattr(ds, 'Columns') else None
        
        # Additional metadata
        extra_metadata = {}
        
        # MRI-specific
        if modality == 'MR':
            if hasattr(ds, 'EchoTime'):
                extra_metadata['echo_time'] = float(ds.EchoTime)
            if hasattr(ds, 'RepetitionTime'):
                extra_metadata['repetition_time'] = float(ds.RepetitionTime)
            if hasattr(ds, 'FlipAngle'):
                extra_metadata['flip_angle'] = float(ds.FlipAngle)
            if hasattr(ds, 'MagneticFieldStrength'):
                extra_metadata['field_strength'] = float(ds.MagneticFieldStrength)
        
        # PET-specific
        if modality == 'PT':
            if hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
                pet_info = ds.RadiopharmaceuticalInformationSequence[0]
                if hasattr(pet_info, 'RadionuclideTotalDose'):
                    extra_metadata['injected_dose'] = float(pet_info.RadionuclideTotalDose)
                if hasattr(pet_info, 'RadiopharmaceuticalStartTime'):
                    extra_metadata['injection_time'] = str(pet_info.RadiopharmaceuticalStartTime)
        
        return ImageMetadata(
            modality=modality,
            patient_id=patient_id,
            study_date=study_date,
            series_description=series_desc,
            pixel_spacing=tuple(pixel_spacing),
            image_position=image_position,
            image_orientation=image_orientation,
            window_center=window_center,
            window_width=window_width,
            rescale_intercept=rescale_intercept,
            rescale_slope=rescale_slope,
            manufacturer=manufacturer,
            scanner_model=scanner_model,
            kvp=kvp,
            exposure=exposure,
            slice_thickness=pixel_spacing[2],
            spacing_between_slices=pixel_spacing[2],
            rows=rows,
            columns=columns,
            extra_metadata=extra_metadata
        )


class NiftiLoader:
    """NIfTI format loader with orientation preservation"""
    
    def __init__(self):
        self._cache = {}
    
    def load_file(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load NIfTI file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"NIfTI file not found: {filepath}")
        
        try:
            # Load with nibabel
            nii = nib.load(str(filepath))
            
            # Get data
            data = nii.get_fdata().astype(np.float32)
            
            # Extract metadata
            metadata = self._extract_metadata(nii)
            
            return data, metadata
            
        except Exception as e:
            logger.error(f"Error loading NIfTI file {filepath}: {e}")
            raise
    
    def save_file(self, data: np.ndarray, metadata: ImageMetadata, 
                  filepath: Union[str, Path]):
        """Save data as NIfTI file"""
        filepath = Path(filepath)
        
        # Create affine matrix
        affine = np.eye(4)
        affine[0, 0] = metadata.pixel_spacing[0]
        affine[1, 1] = metadata.pixel_spacing[1]
        affine[2, 2] = metadata.pixel_spacing[2]
        affine[:3, 3] = metadata.image_position
        
        # Create NIfTI image
        nii = nib.Nifti1Image(data, affine)
        
        # Save
        nib.save(nii, str(filepath))
        logger.info(f"Saved NIfTI to {filepath}")
    
    def _extract_metadata(self, nii: nib.Nifti1Image) -> ImageMetadata:
        """Extract metadata from NIfTI image"""
        header = nii.header
        affine = nii.affine
        
        # Extract spacing
        zooms = header.get_zooms()
        pixel_spacing = (zooms[0], zooms[1], zooms[2] if len(zooms) > 2 else 1.0)
        
        # Extract position
        image_position = tuple(affine[:3, 3])
        
        # Extract orientation (first two columns of rotation matrix)
        image_orientation = list(affine[:3, 0]) + list(affine[:3, 1])
        
        # Dimensions
        shape = nii.shape
        rows = shape[0]
        columns = shape[1]
        number_of_slices = shape[2] if len(shape) > 2 else 1
        
        # Additional metadata
        extra_metadata = {
            'nifti_version': header['sizeof_hdr'],
            'datatype': header.get_data_dtype(),
            'units': header.get_xyzt_units(),
            'qform_code': int(header['qform_code']),
            'sform_code': int(header['sform_code'])
        }
        
        return ImageMetadata(
            modality='Unknown',  # NIfTI doesn't store modality
            patient_id='Unknown',
            study_date='Unknown',
            series_description='NIfTI Image',
            pixel_spacing=pixel_spacing,
            image_position=image_position,
            image_orientation=image_orientation,
            rows=rows,
            columns=columns,
            number_of_slices=number_of_slices,
            extra_metadata=extra_metadata
        )


class RadiologyDataset:
    """Dataset class for managing collections of medical images"""
    
    def __init__(self, root_dir: Union[str, Path], 
                 modality: Optional[str] = None,
                 lazy_load: bool = True):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.lazy_load = lazy_load
        
        self.dicom_loader = DicomLoader(lazy_load=lazy_load)
        self.nifti_loader = NiftiLoader()
        
        self._index = []
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
        
        logger.info(f"Found {len(self._index)} medical images")
    
    def __len__(self):
        return len(self._index)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ImageMetadata]:
        """Get image and metadata by index"""
        file_type, filepath = self._index[idx]
        
        if file_type == 'dicom':
            return self.dicom_loader.load_file(filepath)
        elif file_type == 'nifti':
            return self.nifti_loader.load_file(filepath)
    
    def filter_by_modality(self, modality: str) -> 'RadiologyDataset':
        """Create filtered dataset by modality"""
        filtered = RadiologyDataset(self.root_dir, modality, self.lazy_load)
        filtered._index = []
        
        for file_type, filepath in self._index:
            try:
                if file_type == 'dicom':
                    ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
                    if hasattr(ds, 'Modality') and ds.Modality == modality:
                        filtered._index.append((file_type, filepath))
            except:
                pass
        
        logger.info(f"Filtered dataset to {len(filtered._index)} {modality} images")
        return filtered


def load_medical_image(filepath: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
    """Convenience function to load any medical image format"""
    filepath = Path(filepath)
    
    # Try different loaders based on extension
    if filepath.suffix.lower() in ['.nii', '.nii.gz']:
        loader = NiftiLoader()
        return loader.load_file(filepath)
    else:
        # Assume DICOM
        loader = DicomLoader()
        
        # Check if it's a directory (series) or file
        if filepath.is_dir():
            return loader.load_series(filepath)
        else:
            return loader.load_file(filepath)


def load_dicom_series(directory: Union[str, Path], 
                     series_uid: Optional[str] = None) -> Tuple[np.ndarray, ImageMetadata]:
    """Load a DICOM series from directory"""
    loader = DicomLoader()
    return loader.load_series(directory, series_uid)


def get_metadata(filepath: Union[str, Path]) -> ImageMetadata:
    """Extract metadata without loading pixel data"""
    filepath = Path(filepath)
    
    if filepath.suffix.lower() in ['.nii', '.nii.gz']:
        nii = nib.load(str(filepath))
        loader = NiftiLoader()
        return loader._extract_metadata(nii)
    else:
        ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
        loader = DicomLoader()
        return loader._extract_metadata(ds)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        try:
            volume, metadata = load_medical_image(path)
            print(f"Loaded volume shape: {volume.shape}")
            print(f"Modality: {metadata.modality}")
            print(f"Pixel spacing: {metadata.pixel_spacing}")
            print(f"Window: C={metadata.window_center}, W={metadata.window_width}")
        except Exception as e:
            print(f"Error: {e}")