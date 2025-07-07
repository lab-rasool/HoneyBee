"""
Example CT Processing Script

Demonstrates comprehensive CT processing using HoneyBee radiology module
with actual CT samples from the examples/samples/CT directory.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import logging

# Import radiology modules
from radiology.data_management import DicomLoader, load_dicom_series
from radiology.preprocessing import preprocess_ct, Denoiser, WindowLevelAdjuster
from radiology.segmentation import CTSegmenter, detect_nodules
from radiology.spatial_processing import resample_image
from radiology.ai_integration import RadImageNetProcessor
from radiology.utils import (
    visualize_slices, save_processed_image, calculate_metrics,
    export_results, plot_intensity_histogram, annotate_nodules
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main processing pipeline for CT data"""
    
    # Path to CT samples
    ct_dir = Path("../samples/CT")
    
    if not ct_dir.exists():
        logger.error(f"CT samples directory not found: {ct_dir}")
        return
    
    logger.info(f"Processing CT data from: {ct_dir}")
    
    # Step 1: Load DICOM series
    logger.info("Step 1: Loading DICOM series...")
    try:
        ct_volume, metadata = load_dicom_series(ct_dir)
        logger.info(f"Loaded CT volume shape: {ct_volume.shape}")
        logger.info(f"Modality: {metadata.modality}")
        logger.info(f"Patient ID: {metadata.patient_id}")
        logger.info(f"Pixel spacing: {metadata.pixel_spacing}")
        logger.info(f"Window center/width: {metadata.window_center}/{metadata.window_width}")
    except Exception as e:
        logger.error(f"Failed to load DICOM series: {e}")
        return
    
    # Create results directory
    results_dir = Path("results_ct_processing")
    results_dir.mkdir(exist_ok=True)
    
    # Step 2: Visualize original data
    logger.info("Step 2: Visualizing original CT data...")
    visualize_slices(
        ct_volume,
        window=(metadata.window_center, metadata.window_width),
        save_path=results_dir / "original_slices.png"
    )
    
    # Plot intensity histogram
    plot_intensity_histogram(
        ct_volume,
        title="Original CT Intensity Distribution",
        save_path=results_dir / "intensity_histogram.png"
    )
    
    # Step 3: Preprocessing
    logger.info("Step 3: Preprocessing CT data...")
    
    # Apply different window presets
    window_presets = ['lung', 'bone', 'soft_tissue']
    windowed_images = {}
    
    for preset in window_presets:
        logger.info(f"  Applying {preset} window...")
        windowed = preprocess_ct(
            ct_volume,
            denoise=True,
            normalize=True,
            window=preset,
            reduce_artifacts=False
        )
        windowed_images[preset] = windowed
    
    # Visualize different windows
    middle_slice = ct_volume.shape[0] // 2
    comparison_dict = {
        'Original': ct_volume[middle_slice],
        'Lung Window': windowed_images['lung'][middle_slice],
        'Bone Window': windowed_images['bone'][middle_slice],
        'Soft Tissue': windowed_images['soft_tissue'][middle_slice]
    }
    
    # Save windowed images
    for name, img in comparison_dict.items():
        save_processed_image(
            img,
            {'window_preset': name, **metadata.__dict__},
            results_dir / f"windowed_{name.lower().replace(' ', '_')}",
            format='numpy'
        )
    
    # Step 4: Denoising comparison
    logger.info("Step 4: Comparing denoising methods...")
    
    denoising_methods = ['nlm', 'bilateral', 'median', 'tv']
    denoised_images = {}
    
    for method in denoising_methods:
        logger.info(f"  Applying {method} denoising...")
        denoiser = Denoiser(method=method)
        denoised = denoiser.denoise(ct_volume)
        denoised_images[method] = denoised
    
    # Step 5: Lung segmentation
    logger.info("Step 5: Performing lung segmentation...")
    
    segmenter = CTSegmenter()
    
    # Basic lung segmentation
    lung_mask = segmenter.segment_lungs(ct_volume, enhanced=False)
    logger.info(f"  Basic lung segmentation: {lung_mask.sum()} voxels")
    
    # Enhanced lung segmentation (with airway removal)
    lung_mask_enhanced = segmenter.segment_lungs(ct_volume, enhanced=True)
    logger.info(f"  Enhanced lung segmentation: {lung_mask_enhanced.sum()} voxels")
    
    # Visualize segmentation results
    visualize_slices(
        ct_volume,
        masks={
            'Basic Segmentation': lung_mask,
            'Enhanced Segmentation': lung_mask_enhanced
        },
        window=(-1000, 0),  # Lung window
        save_path=results_dir / "lung_segmentation.png"
    )
    
    # Step 6: Nodule detection
    logger.info("Step 6: Detecting lung nodules...")
    
    nodules = detect_nodules(ct_volume, lung_mask_enhanced, min_size=3.0, max_size=20.0)
    logger.info(f"  Detected {len(nodules)} potential nodules")
    
    if nodules:
        # Annotate nodules on middle slice
        annotate_nodules(
            ct_volume,
            nodules,
            slice_idx=middle_slice,
            save_path=results_dir / "detected_nodules.png"
        )
        
        # Export nodule information
        nodule_data = []
        for i, nodule in enumerate(nodules):
            nodule_data.append({
                'nodule_id': i + 1,
                'position_z': nodule['position'][0] if len(nodule['position']) == 3 else 0,
                'position_y': nodule['position'][1] if len(nodule['position']) == 3 else nodule['position'][0],
                'position_x': nodule['position'][2] if len(nodule['position']) == 3 else nodule['position'][1],
                'diameter_mm': nodule['diameter'],
                'intensity_hu': nodule['intensity']
            })
        
        export_results({'nodules': nodule_data}, results_dir / "nodule_detection", format='csv')
    
    # Step 7: Multi-organ segmentation
    logger.info("Step 7: Performing multi-organ segmentation...")
    
    organs_to_segment = ['bone', 'muscle']  # Liver/kidney might not be visible in chest CT
    organ_masks = segmenter.segment_organs(ct_volume, organs=organs_to_segment)
    
    for organ, mask in organ_masks.items():
        volume_ml = mask.sum() * np.prod(metadata.pixel_spacing) / 1000  # Convert to ml
        logger.info(f"  {organ.capitalize()} volume: {volume_ml:.2f} ml")
    
    # Step 8: Spatial resampling
    logger.info("Step 8: Resampling to isotropic voxels...")
    
    # Resample to 1mm isotropic voxels
    target_spacing = (1.0, 1.0, 1.0)
    resampled_volume = resample_image(
        ct_volume,
        target_spacing=target_spacing,
        current_spacing=metadata.pixel_spacing,
        method='linear'
    )
    logger.info(f"  Resampled volume shape: {resampled_volume.shape}")
    logger.info(f"  Original spacing: {metadata.pixel_spacing}")
    logger.info(f"  New spacing: {target_spacing}")
    
    # Step 9: AI-based embedding generation
    logger.info("Step 9: Generating RadImageNet embeddings...")
    
    # Initialize RadImageNet processor
    radimagenet = RadImageNetProcessor(model_name='densenet121', pretrained=True)
    
    # Generate 2D embedding from middle slice
    embedding_2d = radimagenet.generate_embeddings(ct_volume, mode='2d')
    logger.info(f"  2D embedding shape: {embedding_2d.shape}")
    
    # Generate 3D embedding (aggregated)
    embedding_3d = radimagenet.generate_embeddings(ct_volume, mode='3d', aggregation='mean')
    logger.info(f"  3D embedding shape: {embedding_3d.shape}")
    
    # Extract multi-scale features
    features = radimagenet.extract_features(ct_volume)
    for layer_name, feat in features.items():
        logger.info(f"  {layer_name} features shape: {feat.shape}")
    
    # Step 10: Export comprehensive results
    logger.info("Step 10: Exporting comprehensive results...")
    
    # Compile all results
    comprehensive_results = {
        'metadata': {
            'patient_id': metadata.patient_id,
            'study_date': metadata.study_date,
            'modality': metadata.modality,
            'original_shape': ct_volume.shape,
            'pixel_spacing': metadata.pixel_spacing,
            'scanner': metadata.manufacturer,
            'model': metadata.scanner_model
        },
        'preprocessing': {
            'denoising_methods': list(denoising_methods),
            'window_presets': list(window_presets)
        },
        'segmentation': {
            'lung_volume_ml': lung_mask_enhanced.sum() * np.prod(metadata.pixel_spacing) / 1000,
            'lung_voxels': int(lung_mask_enhanced.sum()),
            'organ_volumes': {
                organ: float(mask.sum() * np.prod(metadata.pixel_spacing) / 1000)
                for organ, mask in organ_masks.items()
            }
        },
        'nodule_detection': {
            'total_nodules': len(nodules),
            'nodule_sizes': [n['diameter'] for n in nodules] if nodules else []
        },
        'spatial_processing': {
            'original_spacing': metadata.pixel_spacing,
            'resampled_spacing': target_spacing,
            'resampled_shape': resampled_volume.shape
        },
        'ai_embeddings': {
            '2d_embedding_dim': embedding_2d.shape[0],
            '3d_embedding_dim': embedding_3d.shape[0],
            'feature_layers': list(features.keys())
        }
    }
    
    # Export as JSON
    export_results(comprehensive_results, results_dir / "comprehensive_results", format='json')
    
    # Save processed volumes
    logger.info("Saving processed volumes...")
    
    # Save lung mask
    save_processed_image(
        lung_mask_enhanced.astype(np.uint8),
        {'description': 'Enhanced lung segmentation mask', **metadata.__dict__},
        results_dir / "lung_mask_enhanced",
        format='nifti'
    )
    
    # Save resampled volume
    save_processed_image(
        resampled_volume,
        {'description': 'Isotropic resampled CT', 'spacing': target_spacing, **metadata.__dict__},
        results_dir / "ct_resampled",
        format='nifti'
    )
    
    logger.info("Processing complete! Results saved to: " + str(results_dir))
    
    # Generate summary report
    generate_summary_report(comprehensive_results, results_dir)


def generate_summary_report(results: dict, output_dir: Path):
    """Generate a text summary report"""
    report_path = output_dir / "processing_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("CT PROCESSING SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Metadata
        f.write("SCAN INFORMATION:\n")
        f.write(f"  Patient ID: {results['metadata']['patient_id']}\n")
        f.write(f"  Study Date: {results['metadata']['study_date']}\n")
        f.write(f"  Scanner: {results['metadata']['scanner']} {results['metadata']['model']}\n")
        f.write(f"  Original Size: {results['metadata']['original_shape']}\n")
        f.write(f"  Voxel Spacing: {results['metadata']['pixel_spacing']} mm\n\n")
        
        # Segmentation results
        f.write("SEGMENTATION RESULTS:\n")
        f.write(f"  Lung Volume: {results['segmentation']['lung_volume_ml']:.2f} ml\n")
        for organ, volume in results['segmentation']['organ_volumes'].items():
            f.write(f"  {organ.capitalize()} Volume: {volume:.2f} ml\n")
        f.write("\n")
        
        # Nodule detection
        f.write("NODULE DETECTION:\n")
        f.write(f"  Total Nodules Found: {results['nodule_detection']['total_nodules']}\n")
        if results['nodule_detection']['nodule_sizes']:
            f.write(f"  Size Range: {min(results['nodule_detection']['nodule_sizes']):.1f} - "
                   f"{max(results['nodule_detection']['nodule_sizes']):.1f} mm\n")
        f.write("\n")
        
        # AI embeddings
        f.write("AI ANALYSIS:\n")
        f.write(f"  2D Embedding Dimension: {results['ai_embeddings']['2d_embedding_dim']}\n")
        f.write(f"  3D Embedding Dimension: {results['ai_embeddings']['3d_embedding_dim']}\n")
        f.write(f"  Feature Layers: {', '.join(results['ai_embeddings']['feature_layers'])}\n")
    
    logger.info(f"Summary report saved to: {report_path}")


if __name__ == "__main__":
    main()