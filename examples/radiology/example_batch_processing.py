"""
Example Batch Processing Script

Demonstrates batch processing of multiple medical images with
parallel processing and harmonization for multi-center studies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
import pandas as pd
import time

# Import radiology modules
from radiology.data_management import RadiologyDataset, load_medical_image
from radiology.preprocessing import preprocess_ct, preprocess_mri, preprocess_pet
from radiology.segmentation import segment_lungs, extract_brain, segment_metabolic_volume
from radiology.spatial_processing import HarmonizationProcessor, resample_image
from radiology.ai_integration import RadImageNetProcessor
from radiology.utils import export_results, calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_image(file_path: Path, 
                        output_dir: Path,
                        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict:
    """Process a single medical image"""
    
    logger.info(f"Processing: {file_path}")
    
    results = {
        'file_path': str(file_path),
        'status': 'processing',
        'errors': []
    }
    
    try:
        # Load image
        image, metadata = load_medical_image(file_path)
        results['modality'] = metadata.modality
        results['original_shape'] = image.shape
        results['original_spacing'] = metadata.pixel_spacing
        
        # Preprocessing based on modality
        if metadata.modality == 'CT':
            processed = preprocess_ct(image, window='lung')
            # Segment lungs
            mask = segment_lungs(processed)
            segmentation_target = 'lungs'
            
        elif metadata.modality == 'MR':
            processed = preprocess_mri(image)
            # Extract brain
            mask = extract_brain(processed)
            segmentation_target = 'brain'
            
        elif metadata.modality == 'PT':
            processed = preprocess_pet(image)
            # Segment metabolic volume
            mask = segment_metabolic_volume(processed)
            segmentation_target = 'metabolic_volume'
            
        else:
            results['status'] = 'skipped'
            results['errors'].append(f"Unsupported modality: {metadata.modality}")
            return results
        
        # Calculate segmentation volume
        volume_ml = mask.sum() * np.prod(metadata.pixel_spacing) / 1000
        results[f'{segmentation_target}_volume_ml'] = float(volume_ml)
        
        # Resample to target spacing
        resampled = resample_image(
            processed,
            target_spacing=target_spacing,
            current_spacing=metadata.pixel_spacing
        )
        results['resampled_shape'] = resampled.shape
        
        # Generate embeddings
        processor = RadImageNetProcessor('densenet121')
        embedding = processor.generate_embeddings(resampled, mode='2d')
        results['embedding_dim'] = embedding.shape[0]
        
        # Save results
        patient_id = metadata.patient_id.replace('/', '_')  # Sanitize filename
        
        # Save embedding
        np.save(output_dir / f"{patient_id}_embedding.npy", embedding)
        
        # Save processed image
        np.save(output_dir / f"{patient_id}_processed.npy", resampled)
        
        # Save mask
        np.save(output_dir / f"{patient_id}_mask.npy", mask)
        
        results['status'] = 'completed'
        results['output_files'] = [
            f"{patient_id}_embedding.npy",
            f"{patient_id}_processed.npy",
            f"{patient_id}_mask.npy"
        ]
        
    except Exception as e:
        results['status'] = 'failed'
        results['errors'].append(str(e))
        logger.error(f"Failed to process {file_path}: {e}")
    
    return results


def batch_process_dataset(dataset_path: Path,
                         output_dir: Path,
                         max_workers: int = 4,
                         modality_filter: str = None) -> pd.DataFrame:
    """Batch process entire dataset with parallel processing"""
    
    logger.info(f"Starting batch processing of dataset: {dataset_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    dataset = RadiologyDataset(dataset_path, modality=modality_filter)
    logger.info(f"Found {len(dataset)} images to process")
    
    # Collect all file paths
    file_paths = [dataset._index[i][1] for i in range(len(dataset))]
    
    # Process in parallel
    results_list = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_image, file_path, output_dir): file_path
            for file_path in file_paths
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results_list.append(result)
                
                # Progress update
                completed = len(results_list)
                logger.info(f"Progress: {completed}/{len(file_paths)} "
                          f"({100*completed/len(file_paths):.1f}%)")
                
            except Exception as e:
                logger.error(f"Exception processing {file_path}: {e}")
                results_list.append({
                    'file_path': str(file_path),
                    'status': 'failed',
                    'errors': [str(e)]
                })
    
    # Calculate processing time
    total_time = time.time() - start_time
    logger.info(f"Batch processing completed in {total_time:.2f} seconds")
    logger.info(f"Average time per image: {total_time/len(file_paths):.2f} seconds")
    
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    # Save results summary
    results_df.to_csv(output_dir / "batch_processing_results.csv", index=False)
    
    # Print summary
    logger.info("\nProcessing Summary:")
    logger.info(f"  Total images: {len(results_df)}")
    logger.info(f"  Completed: {(results_df['status'] == 'completed').sum()}")
    logger.info(f"  Failed: {(results_df['status'] == 'failed').sum()}")
    logger.info(f"  Skipped: {(results_df['status'] == 'skipped').sum()}")
    
    if 'modality' in results_df.columns:
        logger.info("\nModality breakdown:")
        for modality, count in results_df['modality'].value_counts().items():
            logger.info(f"  {modality}: {count}")
    
    return results_df


def harmonize_multicenter_data(data_dirs: List[Path],
                              output_dir: Path,
                              reference_center: int = 0) -> None:
    """Harmonize data from multiple centers/scanners"""
    
    logger.info("Starting multi-center harmonization...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images from each center
    center_images = {}
    center_metadata = {}
    
    for i, data_dir in enumerate(data_dirs):
        logger.info(f"Loading data from center {i+1}: {data_dir}")
        
        images = []
        metadata = []
        
        # Load all processed images from center
        for img_file in data_dir.glob("*_processed.npy"):
            img = np.load(img_file)
            images.append(img)
            
            # Extract patient ID from filename
            patient_id = img_file.stem.replace('_processed', '')
            metadata.append({
                'patient_id': patient_id,
                'center': i + 1,
                'file': img_file.name
            })
        
        center_images[i] = images
        center_metadata[i] = metadata
        logger.info(f"  Loaded {len(images)} images from center {i+1}")
    
    # Flatten all images for harmonization
    all_images = []
    all_metadata = []
    
    for center_idx, images in center_images.items():
        all_images.extend(images)
        all_metadata.extend(center_metadata[center_idx])
    
    # Initialize harmonization
    harmonizer = HarmonizationProcessor(method='histogram_matching')
    
    # Use reference center for harmonization
    reference_images = center_images[reference_center]
    harmonizer.fit_reference(reference_images)
    
    # Harmonize all images
    logger.info("Harmonizing images...")
    harmonized_images = []
    
    for i, img in enumerate(all_images):
        harmonized = harmonizer.harmonize(img)
        harmonized_images.append(harmonized)
        
        # Save harmonized image
        meta = all_metadata[i]
        output_file = output_dir / f"center{meta['center']}_{meta['patient_id']}_harmonized.npy"
        np.save(output_file, harmonized)
    
    # Generate harmonization report
    report = {
        'total_images': len(all_images),
        'reference_center': reference_center + 1,
        'centers': len(data_dirs),
        'harmonization_method': 'histogram_matching',
        'images_per_center': {
            f'center_{i+1}': len(imgs) for i, imgs in center_images.items()
        }
    }
    
    export_results(report, output_dir / "harmonization_report", format='json')
    logger.info(f"Harmonization complete. Results saved to: {output_dir}")


def compare_embeddings_across_modalities(multimodal_dir: Path) -> None:
    """Compare embeddings from different modalities for same patients"""
    
    logger.info("Comparing embeddings across modalities...")
    
    # Group embeddings by patient
    patient_embeddings = {}
    
    for emb_file in multimodal_dir.glob("*_embedding.npy"):
        # Extract patient ID and modality from filename
        parts = emb_file.stem.split('_')
        patient_id = parts[0]
        
        if patient_id not in patient_embeddings:
            patient_embeddings[patient_id] = {}
        
        # Load embedding
        embedding = np.load(emb_file)
        
        # Determine modality from associated metadata
        # This is simplified - in practice you'd track this properly
        patient_embeddings[patient_id][emb_file.name] = embedding
    
    # Calculate similarities between modalities
    logger.info(f"Found {len(patient_embeddings)} patients with embeddings")
    
    # Perform analysis (simplified example)
    for patient_id, embeddings in patient_embeddings.items():
        if len(embeddings) > 1:
            logger.info(f"\nPatient {patient_id}:")
            emb_list = list(embeddings.values())
            
            # Calculate cosine similarities
            for i in range(len(emb_list)):
                for j in range(i+1, len(emb_list)):
                    similarity = np.dot(emb_list[i], emb_list[j]) / (
                        np.linalg.norm(emb_list[i]) * np.linalg.norm(emb_list[j])
                    )
                    logger.info(f"  Similarity between modalities: {similarity:.3f}")


def main():
    """Main function demonstrating batch processing capabilities"""
    
    # Example 1: Batch process CT dataset
    logger.info("=" * 50)
    logger.info("EXAMPLE 1: Batch Processing CT Dataset")
    logger.info("=" * 50)
    
    ct_dataset_path = Path("../samples")  # This would contain multiple CT series
    batch_output_dir = Path("results_batch_processing")
    
    if ct_dataset_path.exists():
        results_df = batch_process_dataset(
            ct_dataset_path,
            batch_output_dir,
            max_workers=2,  # Adjust based on system
            modality_filter='CT'
        )
        
        # Analyze results
        if not results_df.empty and 'lungs_volume_ml' in results_df.columns:
            lung_volumes = results_df[results_df['lungs_volume_ml'].notna()]['lungs_volume_ml']
            if not lung_volumes.empty:
                logger.info(f"\nLung volume statistics:")
                logger.info(f"  Mean: {lung_volumes.mean():.2f} ml")
                logger.info(f"  Std: {lung_volumes.std():.2f} ml")
                logger.info(f"  Range: {lung_volumes.min():.2f} - {lung_volumes.max():.2f} ml")
    
    # Example 2: Multi-center harmonization (simulated)
    logger.info("\n" + "=" * 50)
    logger.info("EXAMPLE 2: Multi-Center Harmonization")
    logger.info("=" * 50)
    
    # Simulate multiple centers (in practice, these would be different directories)
    center_dirs = [
        batch_output_dir,  # Use results from batch processing as "center 1"
        # Path("center2_data"),  # Would be another center's data
        # Path("center3_data"),  # And another
    ]
    
    # Only run if we have data
    if batch_output_dir.exists() and any(batch_output_dir.glob("*_processed.npy")):
        harmonization_output = Path("results_harmonization")
        
        # For demo, just use one center
        logger.info("Note: Using single center data for demonstration")
        harmonize_multicenter_data(
            [batch_output_dir],
            harmonization_output,
            reference_center=0
        )
    
    # Example 3: Quality control metrics
    logger.info("\n" + "=" * 50)
    logger.info("EXAMPLE 3: Quality Control Metrics")
    logger.info("=" * 50)
    
    if batch_output_dir.exists():
        # Calculate quality metrics for processed images
        qc_metrics = []
        
        for proc_file in batch_output_dir.glob("*_processed.npy"):
            img = np.load(proc_file)
            mask_file = proc_file.parent / proc_file.name.replace('_processed', '_mask')
            
            if mask_file.exists():
                mask = np.load(mask_file)
                
                metrics = {
                    'file': proc_file.name,
                    'image_mean': float(img.mean()),
                    'image_std': float(img.std()),
                    'mask_coverage': float(mask.sum() / mask.size),
                    'snr': float(img.mean() / (img.std() + 1e-8))
                }
                qc_metrics.append(metrics)
        
        if qc_metrics:
            qc_df = pd.DataFrame(qc_metrics)
            qc_df.to_csv(batch_output_dir / "quality_control_metrics.csv", index=False)
            
            logger.info("Quality control summary:")
            logger.info(f"  Average SNR: {qc_df['snr'].mean():.2f}")
            logger.info(f"  Average mask coverage: {qc_df['mask_coverage'].mean():.2%}")
    
    logger.info("\nBatch processing examples completed!")


if __name__ == "__main__":
    main()