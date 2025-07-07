# Legacy imports (kept for backward compatibility)
try:
    from .radiology.radiology import DICOMPreprocessor
except ImportError:
    # If old file doesn't exist, provide None
    DICOMPreprocessor = None

from .Reader.mindsDBreader import manifest_to_df
from .Reader.reader import PDF
from .Scans.scan import Scan
from .Slide.slide import Slide

# New imports
from .radiology import DicomLoader, NiftiLoader, RadiologyDataset, ImageMetadata
