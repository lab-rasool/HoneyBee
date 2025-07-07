"""
HoneyBee Radiology Processing Module

Comprehensive radiology processing capabilities including:
- DICOM/NIfTI data management
- Preprocessing and enhancement
- Segmentation algorithms
- Spatial processing
- AI integration with RadImageNet
"""

from .data_management import (
    DicomLoader,
    NiftiLoader,
    RadiologyDataset,
    load_medical_image,
    load_dicom_series,
    get_metadata
)

from .preprocessing import (
    Denoiser,
    IntensityNormalizer,
    WindowLevelAdjuster,
    ArtifactReducer,
    preprocess_ct,
    preprocess_mri,
    preprocess_pet
)

from .segmentation import (
    CTSegmenter,
    MRISegmenter,
    PETSegmenter,
    segment_lungs,
    segment_organs,
    detect_nodules,
    extract_brain,
    segment_metabolic_volume
)

from .spatial_processing import (
    Resampler,
    RegistrationEngine,
    HarmonizationProcessor,
    resample_image,
    register_to_atlas,
    harmonize_cross_scanner
)

from .ai_integration import (
    RadImageNetProcessor,
    create_embedding_model,
    generate_embeddings,
    load_pretrained_model,
    process_2d_slices,
    process_3d_volume
)

from .utils import (
    visualize_slices,
    save_processed_image,
    create_montage,
    calculate_metrics,
    export_results
)

__all__ = [
    # Data Management
    'DicomLoader',
    'NiftiLoader',
    'RadiologyDataset',
    'load_medical_image',
    'load_dicom_series',
    'get_metadata',
    
    # Preprocessing
    'Denoiser',
    'IntensityNormalizer',
    'WindowLevelAdjuster',
    'ArtifactReducer',
    'preprocess_ct',
    'preprocess_mri',
    'preprocess_pet',
    
    # Segmentation
    'CTSegmenter',
    'MRISegmenter',
    'PETSegmenter',
    'segment_lungs',
    'segment_organs',
    'detect_nodules',
    'extract_brain',
    'segment_metabolic_volume',
    
    # Spatial Processing
    'Resampler',
    'RegistrationEngine',
    'HarmonizationProcessor',
    'resample_image',
    'register_to_atlas',
    'harmonize_cross_scanner',
    
    # AI Integration
    'RadImageNetProcessor',
    'create_embedding_model',
    'generate_embeddings',
    'load_pretrained_model',
    'process_2d_slices',
    'process_3d_volume',
    
    # Utils
    'visualize_slices',
    'save_processed_image',
    'create_montage',
    'calculate_metrics',
    'export_results'
]