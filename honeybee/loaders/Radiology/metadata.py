"""
Medical Image Metadata

Standardized metadata container for medical images.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


@dataclass
class ImageMetadata:
    """Container for medical image metadata
    
    Attributes:
        modality: Imaging modality (CT, MR, PT, etc.)
        patient_id: Patient identifier
        study_date: Date of study acquisition
        series_description: Description of the series
        pixel_spacing: Physical spacing between pixels (x, y, z)
        image_position: 3D position of first voxel
        image_orientation: Direction cosines of first row and column
        window_center: Default window center for display
        window_width: Default window width for display
        rescale_intercept: Value to add after rescale slope
        rescale_slope: Multiplicative factor for pixel values
        manufacturer: Scanner manufacturer name
        scanner_model: Scanner model name
        kvp: Peak kilovoltage (CT)
        exposure: Exposure value (CT)
        slice_thickness: Thickness of each slice
        spacing_between_slices: Distance between slice centers
        rows: Number of rows in image matrix
        columns: Number of columns in image matrix
        number_of_slices: Number of slices in volume
        extra_metadata: Additional modality-specific metadata
    """
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
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_voxel_spacing(self) -> Tuple[float, float, float]:
        """Get voxel spacing as tuple"""
        return self.pixel_spacing
    
    def get_image_size(self) -> Tuple[int, int, int]:
        """Get image dimensions"""
        return (
            self.rows or 0,
            self.columns or 0,
            self.number_of_slices or 1
        )
    
    def get_window_settings(self) -> Tuple[Optional[float], Optional[float]]:
        """Get window center and width"""
        return self.window_center, self.window_width
    
    def is_ct(self) -> bool:
        """Check if modality is CT"""
        return self.modality.upper() == 'CT'
    
    def is_mri(self) -> bool:
        """Check if modality is MRI"""
        return self.modality.upper() in ['MR', 'MRI']
    
    def is_pet(self) -> bool:
        """Check if modality is PET"""
        return self.modality.upper() in ['PT', 'PET']
    
    def has_window_settings(self) -> bool:
        """Check if window settings are available"""
        return self.window_center is not None and self.window_width is not None