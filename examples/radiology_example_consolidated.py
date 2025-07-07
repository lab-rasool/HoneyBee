"""
Example: Using Consolidated Radiology Components

This example demonstrates how to use the new consolidated radiology components
in the HoneyBee library for medical image processing and embedding generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import consolidated components from HoneyBee
from honeybee.loaders.radiology import (
    DicomLoader, NiftiLoader, RadiologyDataset, 
    ImageMetadata, load_medical_image
)
from honeybee.processors.radiology import (
    RadiologyProcessor,
    Denoiser, IntensityNormalizer, WindowLevelAdjuster, ArtifactReducer,
    preprocess_ct, preprocess_mri, preprocess_pet
)
from honeybee.models.RadImageNet import RadImageNet


def example_1_basic_loading():
    """Example 1: Basic Image Loading"""
    print("\n" + "="*60)
    print("Example 1: Basic Image Loading")
    print("="*60)
    
    # Create example paths (replace with actual paths)
    dicom_path = Path("path/to/dicom/file.dcm")
    nifti_path = Path("path/to/nifti/file.nii.gz")
    
    # Initialize loaders
    dicom_loader = DicomLoader(lazy_load=True)
    nifti_loader = NiftiLoader()
    
    print("✓ Loaders initialized")
    
    # Example: Load medical image (auto-detects format)
    # image, metadata = load_medical_image(dicom_path)
    
    # Create sample metadata
    metadata = ImageMetadata(
        modality='CT',
        patient_id='EXAMPLE001',
        study_date='20240101',
        series_description='Chest CT',
        pixel_spacing=(1.0, 1.0, 2.5),
        image_position=(0.0, 0.0, 0.0),
        image_orientation=[1, 0, 0, 0, 1, 0],
        window_center=-600,
        window_width=1500
    )
    
    print(f"Modality: {metadata.modality}")
    print(f"Is CT: {metadata.is_ct()}")
    print(f"Voxel spacing: {metadata.get_voxel_spacing()}")
    print(f"Window settings: {metadata.get_window_settings()}")


def example_2_preprocessing():
    """Example 2: Modular Preprocessing"""
    print("\n" + "="*60)
    print("Example 2: Modular Preprocessing")
    print("="*60)
    
    # Create synthetic CT image
    ct_image = np.random.randn(64, 64, 32) * 1000 - 500  # HU range
    
    # 1. Denoising
    print("\n1. Denoising methods:")
    for method in ['nlm', 'tv', 'bilateral']:
        denoiser = Denoiser(method=method)
        denoised = denoiser.denoise(ct_image[:, :, 16])  # Single slice
        print(f"  - {method}: shape={denoised.shape}, "
              f"range=[{denoised.min():.1f}, {denoised.max():.1f}]")
    
    # 2. Window/Level Adjustment
    print("\n2. Window presets:")
    windower = WindowLevelAdjuster()
    for preset in ['lung', 'bone', 'abdomen']:
        windowed = windower.adjust(ct_image[:, :, 16], window=preset)
        print(f"  - {preset}: range=[{windowed.min():.1f}, {windowed.max():.1f}]")
    
    # 3. Normalization
    print("\n3. Normalization methods:")
    for method in ['zscore', 'minmax', 'percentile']:
        normalizer = IntensityNormalizer(method=method)
        normalized = normalizer.normalize(ct_image[:, :, 16])
        print(f"  - {method}: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    
    # 4. Complete CT preprocessing pipeline
    print("\n4. Complete CT pipeline:")
    processed = preprocess_ct(
        ct_image,
        denoise=True,
        normalize=True,
        window='lung',
        reduce_artifacts=False
    )
    print(f"  Output shape: {processed.shape}")
    print(f"  Output range: [{processed.min():.3f}, {processed.max():.3f}]")


def example_3_embedding_generation():
    """Example 3: Embedding Generation with RadImageNet"""
    print("\n" + "="*60)
    print("Example 3: Embedding Generation")
    print("="*60)
    
    # Initialize processor with RadImageNet
    print("Note: Skipping actual model initialization in this example")
    print("Set use_hub=True to download models from HuggingFace")
    
    # For demonstration, we'll show the interface
    print("Processor interface:")
    
    # Create test images
    test_2d = np.random.rand(512, 512).astype(np.float32)
    test_3d = np.random.rand(32, 512, 512).astype(np.float32)
    
    # Generate 2D embeddings
    print("\nGenerating 2D embeddings...")
    # embeddings_2d = processor.generate_embeddings(test_2d, mode='2d')
    # print(f"  2D embedding shape: {embeddings_2d.shape}")
    
    # Generate 3D embeddings with different aggregations
    print("\nGenerating 3D embeddings:")
    for aggregation in ['mean', 'max']:
        # embeddings_3d = processor.generate_embeddings(
        #     test_3d, mode='3d', aggregation=aggregation
        # )
        # print(f"  {aggregation} aggregation shape: {embeddings_3d.shape}")
        print(f"  {aggregation} aggregation: [would generate embeddings]")


def example_4_advanced_features():
    """Example 4: Advanced RadImageNet Features"""
    print("\n" + "="*60)
    print("Example 4: Advanced Features")
    print("="*60)
    
    # Initialize model with feature extraction
    print("RadImageNet features:")
    print("  • Multi-scale feature extraction")
    print("  • Batch processing")
    print("  • Intermediate layer features")
    print("  • Support for DenseNet121, ResNet50, InceptionV3")
    print("  • Input sizes: 224x224 (DenseNet/ResNet), 299x299 (Inception)")
    
    # Test multi-scale feature extraction
    test_image = np.random.rand(256, 256).astype(np.float32)
    
    # Multi-scale features
    print("\nMulti-scale feature extraction:")
    # scales = [0.5, 1.0, 1.5]
    # multi_scale_features = model.extract_multi_scale_features(test_image, scales)
    # for scale, features in multi_scale_features.items():
    #     print(f"  Scale {scale}: {features.shape}")
    print("  [Would extract features at multiple scales]")
    
    # Batch processing
    print("\nBatch processing:")
    batch_images = [np.random.rand(128, 128) for _ in range(10)]
    # batch_embeddings = model.process_batch(batch_images, batch_size=5)
    # print(f"  Batch embeddings shape: {batch_embeddings.shape}")
    print("  [Would process batch of 10 images]")


def example_5_dataset_management():
    """Example 5: Dataset Management"""
    print("\n" + "="*60)
    print("Example 5: Dataset Management")
    print("="*60)
    
    # Create dataset from directory (example)
    # dataset = RadiologyDataset(
    #     root_dir=Path("/path/to/medical/images"),
    #     modality="CT",
    #     lazy_load=True
    # )
    
    # Example dataset operations
    print("Dataset operations:")
    print("  - Automatic file discovery")
    print("  - Modality filtering")
    print("  - Lazy loading for memory efficiency")
    print("  - Batch loading capabilities")
    print("  - Metadata extraction without loading pixels")
    
    # Example usage:
    # print(f"Dataset size: {len(dataset)}")
    # print(f"Modalities: {dataset.get_modalities()}")
    # 
    # # Load specific image
    # image, metadata = dataset[0]
    # 
    # # Load batch
    # batch_images, batch_metadata = dataset.get_batch([0, 1, 2, 3, 4])


def example_6_complete_pipeline():
    """Example 6: Complete Processing Pipeline"""
    print("\n" + "="*60)
    print("Example 6: Complete Pipeline")
    print("="*60)
    
    # Initialize processor (would use actual model in practice)
    print("Pipeline demonstration (without model weights):")
    
    # Create synthetic CT scan
    ct_scan = np.random.randn(32, 512, 512) * 1000 - 500
    
    # Create metadata
    metadata = ImageMetadata(
        modality='CT',
        patient_id='PIPE001',
        study_date='20240101',
        series_description='Chest CT - Example',
        pixel_spacing=(1.0, 1.0, 2.5),
        image_position=(0.0, 0.0, 0.0),
        image_orientation=[1, 0, 0, 0, 1, 0]
    )
    
    print("Pipeline steps:")
    
    # 1. Preprocessing
    print("\n1. Preprocessing...")
    from honeybee.processors.radiology import preprocess_ct
    preprocessed = preprocess_ct(
        ct_scan,
        denoise=True,
        normalize=True,
        window='lung',
        reduce_artifacts=False
    )
    print(f"   ✓ Preprocessed shape: {preprocessed.shape}")
    print(f"   ✓ Output range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # 2. Segmentation (if needed)
    print("\n2. Segmentation...")
    print("   ✓ Lung segmentation available")
    print("   ✓ Brain segmentation available") 
    print("   ✓ Multi-organ segmentation (with full processor)")
    
    # 3. Embedding generation
    print("\n3. Embedding generation...")
    # embeddings = processor.generate_embeddings(
    #     preprocessed,
    #     mode='3d',
    #     aggregation='mean',
    #     metadata=metadata
    # )
    # print(f"   ✓ Embeddings shape: {embeddings.shape}")
    print("   ✓ [Would generate embeddings]")
    
    print("\n✓ Pipeline complete!")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" HONEYBEE RADIOLOGY COMPONENTS - CONSOLIDATED EXAMPLES")
    print("="*80)
    
    print("\nThis example demonstrates the new consolidated radiology components.")
    print("The library now provides:")
    print("  • Unified data loaders for DICOM and NIfTI")
    print("  • Modular preprocessing components")
    print("  • Enhanced RadImageNet model with advanced features")
    print("  • Streamlined RadiologyProcessor using modular components")
    
    # Run examples
    example_1_basic_loading()
    example_2_preprocessing()
    example_3_embedding_generation()
    example_4_advanced_features()
    example_5_dataset_management()
    example_6_complete_pipeline()
    
    print("\n" + "="*80)
    print(" All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()