#!/usr/bin/env python3
"""
Example 02: Tissue Detection

This example demonstrates different methods for detecting tissue regions in WSI,
including both classical computer vision approaches and deep learning methods.

Key Features Demonstrated:
- Classical Otsu-based tissue detection
- Deep learning tissue detection with pre-trained model
- Comparison of detection methods
- Tissue mask visualization
- Performance benchmarking
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import cv2
import torch
from PIL import Image
from scipy.ndimage import sobel
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk
from honeybee.loaders.Slide.slide import Slide
from honeybee.models.TissueDetector.tissue_detector import TissueDetector
from utils import (
    print_example_header, print_section_header, get_sample_wsi_path,
    get_tissue_detector_path, create_output_dir, timer_decorator, 
    validate_paths, save_numpy_compressed, create_results_summary
)
from visualizations import (
    plot_tissue_detection_comparison, plot_patch_grid_on_wsi,
    save_figure
)
import matplotlib.pyplot as plt


def classical_tissue_detection(image: np.ndarray, 
                             min_object_size: int = 1000) -> np.ndarray:
    """
    Perform classical tissue detection using Otsu thresholding.
    
    Args:
        image: RGB image
        min_object_size: Minimum size of objects to keep
        
    Returns:
        Binary tissue mask
    """
    # Convert to grayscale
    gray_image = np.mean(image.astype(np.float32), axis=2)
    
    # Compute gradients
    grad_x = sobel(gray_image, axis=1)
    grad_y = sobel(gray_image, axis=0)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Apply Otsu's threshold
    threshold_value = threshold_otsu(magnitude)
    tissue_mask = magnitude > threshold_value
    
    # Morphological operations to clean up
    tissue_mask = binary_closing(tissue_mask, disk(5))
    tissue_mask = remove_small_objects(tissue_mask, min_size=min_object_size)
    
    return tissue_mask


def main():
    # Setup
    print_example_header(
        "Example 02: Tissue Detection",
        "Learn different methods for detecting tissue regions in WSI"
    )
    
    # Configuration
    wsi_path = get_sample_wsi_path()
    tissue_detector_path = get_tissue_detector_path()
    output_dir = create_output_dir("/mnt/f/Projects/HoneyBee/examples/wsi/tmp", "02_tissue_detection")
    
    # Validate paths
    if not validate_paths(wsi_path, tissue_detector_path):
        print("Error: Required files not found!")
        return
    
    print(f"WSI Path: {wsi_path}")
    print(f"Tissue Detector Path: {tissue_detector_path}")
    print(f"Output Directory: {output_dir}")
    
    # =============================
    # 1. Load WSI at Low Resolution
    # =============================
    print_section_header("1. Loading WSI for Tissue Detection")
    
    # First, load without tissue detection to get a baseline
    slide_no_detection = Slide(wsi_path, visualize=False, max_patches=1000, verbose=False)
    
    # Get low resolution image for classical detection
    # Use level 2 or the last level for faster processing
    resolutions = slide_no_detection.img.resolutions
    detection_level = min(2, resolutions['level_count'] - 1)
    level_dims = resolutions['level_dimensions'][detection_level]
    
    print(f"Using level {detection_level} for tissue detection")
    print(f"Level dimensions: {level_dims}")
    
    @timer_decorator
    def get_low_res_image():
        """Get low resolution image for detection."""
        low_res = slide_no_detection.img.read_region([0, 0], level_dims, detection_level)
        return np.asarray(low_res)
    
    low_res_image = get_low_res_image()
    print(f"Low resolution image shape: {low_res_image.shape}")
    
    # =============================
    # 2. Classical Tissue Detection
    # =============================
    print_section_header("2. Classical Tissue Detection (Otsu)")
    
    @timer_decorator
    def classical_detection():
        """Perform classical tissue detection."""
        return classical_tissue_detection(low_res_image, min_object_size=100)
    
    classical_mask = classical_detection()
    
    # Calculate tissue percentage
    tissue_pixels = np.sum(classical_mask)
    total_pixels = classical_mask.size
    tissue_percentage = (tissue_pixels / total_pixels) * 100
    
    print(f"Tissue pixels: {tissue_pixels:,}")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Tissue percentage: {tissue_percentage:.2f}%")
    
    # =============================
    # 3. Deep Learning Tissue Detection
    # =============================
    print_section_header("3. Deep Learning Tissue Detection")
    
    # Initialize tissue detector
    print("Loading tissue detector model...")
    tissue_detector = TissueDetector(model_path=tissue_detector_path)
    
    # First, let's test the tissue detector directly on a patch
    print("\nTesting tissue detector on sample patches...")
    
    # Extract a few test patches
    test_coords = [(5000, 5000), (10000, 10000), (15000, 15000)]
    for coord in test_coords:
        try:
            # Read a patch from the slide
            patch = slide_no_detection.img.read_region(coord, [512, 512], 0)
            patch_array = np.asarray(patch)[:, :, :3]
            
            # Resize for tissue detection
            patch_resized = cv2.resize(patch_array, (224, 224))
            patch_pil = Image.fromarray(patch_resized)
            
            # Apply tissue detector
            patch_transformed = tissue_detector.transforms(patch_pil)
            patch_batch = patch_transformed.unsqueeze(0).to(tissue_detector.device)
            
            with torch.no_grad():
                prediction = tissue_detector.model(patch_batch)
                prob = torch.nn.functional.softmax(prediction, dim=1).cpu().numpy()[0]
                tissue_class = np.argmax(prob)
            
            print(f"  Patch at {coord}: Class {tissue_class} (0=artifact, 1=background, 2=tissue)")
            print(f"    Probabilities: artifact={prob[0]:.3f}, background={prob[1]:.3f}, tissue={prob[2]:.3f}")
        except Exception as e:
            print(f"  Error testing patch at {coord}: {e}")
    
    # Load slide with tissue detection
    # Use parameters from reference implementation for better detection
    @timer_decorator
    def load_with_detection():
        """Load slide with deep learning tissue detection."""
        return Slide(
            wsi_path, 
            visualize=False, 
            tile_size=512,  # Standard tile size from reference
            max_patches=1000,  # More patches for better coverage
            tissue_detector=tissue_detector,
            verbose=True
        )
    
    slide_with_detection = load_with_detection()
    
    # Extract tissue detection results
    print("\nExtracting tissue detection results...")
    
    # Count tissue tiles
    tissue_tiles = 0
    background_tiles = 0
    artifact_tiles = 0
    
    tissue_coords = []
    tissue_confidence_threshold = 0.8  # Using 80% threshold as in reference
    
    # The tissue detector outputs probabilities for 3 classes:
    # Index 0: Artifact
    # Index 1: Background  
    # Index 2: Tissue
    
    for address in slide_with_detection.iterateTiles():
        tile_info = slide_with_detection.tileDictionary[address]
        
        # The Slide class stores both raw predictions and derived levels
        if 'tissueLevel' in tile_info:
            # Use the derived tissue level (already computed by Slide class)
            tissue_level = tile_info['tissueLevel']
            background_level = tile_info.get('backgroundLevel', 0)
            artifact_level = tile_info.get('artifactLevel', 0)
            
            # Find which class has highest probability
            levels = [artifact_level, background_level, tissue_level]
            pred_class = np.argmax(levels)
            
            # Check if tissue class has high enough confidence (regardless of which class wins)
            # This matches the reference implementation logic
            if tissue_level >= tissue_confidence_threshold:
                tissue_tiles += 1
                tissue_coords.append((tile_info['x'], tile_info['y']))
            elif pred_class == 1:  # Background class
                background_tiles += 1
            else:  # Artifact class or low confidence tissue
                artifact_tiles += 1
    
    total_tiles = len(slide_with_detection.tileDictionary)
    print(f"\nTile Classification Results:")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Tissue tiles (>={tissue_confidence_threshold:.0%} confidence): {tissue_tiles} ({tissue_tiles/total_tiles*100:.1f}%)")
    print(f"  Background tiles: {background_tiles} ({background_tiles/total_tiles*100:.1f}%)")
    print(f"  Artifact tiles: {artifact_tiles} ({artifact_tiles/total_tiles*100:.1f}%)")
    
    # =============================
    # 4. Create Deep Learning Prediction Map
    # =============================
    print_section_header("4. Creating Deep Learning Prediction Map")
    
    # Create prediction map with class probabilities (like the reference visualization)
    # This will store the actual predictions for visualization
    prediction_map = np.zeros((slide_with_detection.numTilesInY, slide_with_detection.numTilesInX, 3))
    
    # Also create a simple binary mask for tissue
    deep_mask = np.zeros((slide_with_detection.numTilesInY, slide_with_detection.numTilesInX))
    
    for address in slide_with_detection.iterateTiles():
        tile_info = slide_with_detection.tileDictionary[address]
        if 'tissueLevel' in tile_info:
            tissue_level = tile_info['tissueLevel']
            background_level = tile_info.get('backgroundLevel', 0)
            artifact_level = tile_info.get('artifactLevel', 0)
            
            # Store the predictions in the prediction map
            prediction_map[address[1], address[0], 0] = artifact_level
            prediction_map[address[1], address[0], 1] = background_level
            prediction_map[address[1], address[0], 2] = tissue_level
            
            # Mark as tissue if tissue confidence is high enough
            if tissue_level >= tissue_confidence_threshold:
                deep_mask[address[1], address[0]] = 1
    
    # Store prediction map in slide object for visualization
    slide_with_detection.predictionMap = prediction_map
    
    # Resize deep mask to match classical mask size for comparison
    from skimage.transform import resize
    deep_mask_resized = resize(deep_mask, classical_mask.shape, order=0, 
                              anti_aliasing=False, preserve_range=True).astype(bool)
    
    # =============================
    # 5. Compare Detection Methods
    # =============================
    print_section_header("5. Comparing Detection Methods")
    
    # Calculate overlap between methods
    intersection = np.logical_and(classical_mask, deep_mask_resized)
    union = np.logical_or(classical_mask, deep_mask_resized)
    
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    dice = 2 * np.sum(intersection) / (np.sum(classical_mask) + np.sum(deep_mask_resized))
    
    print(f"Intersection over Union (IoU): {iou:.3f}")
    print(f"Dice coefficient: {dice:.3f}")
    
    # Visualize comparison
    plot_tissue_detection_comparison(
        low_res_image, 
        classical_mask, 
        deep_mask_resized,
        save_path=os.path.join(output_dir, "detection_comparison.png")
    )
    
    # =============================
    # 6. Visualize Patch Grid
    # =============================
    print_section_header("6. Visualizing Detected Tissue Patches")
    
    # Get thumbnail for visualization
    thumbnail = np.asarray(slide_with_detection.slide)
    
    # Plot patch grid with tissue patches highlighted
    all_coords = []
    tissue_indices = []
    
    for idx, address in enumerate(slide_with_detection.iterateTiles()):
        tile_info = slide_with_detection.tileDictionary[address]
        all_coords.append((tile_info['x'], tile_info['y']))
        
        if 'tissueLevel' in tile_info:
            tissue_level = tile_info['tissueLevel']
            background_level = tile_info.get('backgroundLevel', 0)
            artifact_level = tile_info.get('artifactLevel', 0)
            
            levels = [artifact_level, background_level, tissue_level]
            pred_class = np.argmax(levels)
            
            if tissue_level >= tissue_confidence_threshold:
                tissue_indices.append(idx)
    
    # Show first 100 patches for visualization
    if len(all_coords) > 100:
        all_coords = all_coords[:100]
        tissue_indices = [i for i in tissue_indices if i < 100]
    
    plot_patch_grid_on_wsi(
        thumbnail, 
        all_coords, 
        slide_with_detection.tileSize,
        highlight_indices=tissue_indices,
        save_path=os.path.join(output_dir, "patch_grid_visualization.png")
    )
    
    # =============================
    # 7. Save Results
    # =============================
    print_section_header("7. Saving Results")
    
    # Save masks
    save_numpy_compressed(classical_mask, os.path.join(output_dir, "classical_mask"))
    save_numpy_compressed(deep_mask_resized, os.path.join(output_dir, "deep_mask"))
    
    # Use the Slide class's built-in visualization method
    # This creates the exact visualization shown in the reference images
    from honeybee.loaders.Slide.slide import Slide as SlideClass
    
    # Check if the visualization method exists and call it
    if hasattr(slide_with_detection, 'visualize') and callable(getattr(SlideClass, 'visualize', None)):
        slide_with_detection.path_to_store_visualization = output_dir
        SlideClass.visualize(slide_with_detection)
        print(f"Slide visualization saved to: {output_dir}")
    
    # Create additional visualization matching the reference style
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original slide thumbnail
    axes[0].imshow(np.asarray(slide_with_detection.slide))
    axes[0].set_title('original', fontsize=16)
    axes[0].axis('on')
    
    # Show deep tissue detection prediction map
    axes[1].imshow(slide_with_detection.predictionMap)
    axes[1].set_title('deep tissue detection', fontsize=16)
    axes[1].axis('on')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "tissue_detection_visualization.png"))
    plt.close()
    
    # Also save the comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(low_res_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(classical_mask, cmap='gray')
    axes[1].set_title('Classical Detection')
    axes[1].axis('off')
    
    axes[2].imshow(deep_mask_resized, cmap='gray')
    axes[2].set_title('Deep Learning Detection')
    axes[2].axis('off')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "tissue_detection_comparison.png"))
    plt.close()
    
    # Save statistics matching the reference format
    stats_file = os.path.join(output_dir, "tissue_detection_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("Tissue Detection Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Slide: {wsi_path}\n")
        f.write(f"Tile size: {slide_with_detection.tileSize}\n")
        f.write(f"Max patches: {total_tiles}\n\n")
        f.write(f"Total tiles analyzed: {total_tiles}\n")
        f.write(f"Tissue tiles (>80% conf): {tissue_tiles} ({tissue_tiles/total_tiles*100:.1f}%)\n")
        f.write(f"Background tiles: {background_tiles} ({background_tiles/total_tiles*100:.1f}%)\n")
        f.write(f"Artifact tiles: {artifact_tiles} ({artifact_tiles/total_tiles*100:.1f}%)\n")
        f.write(f"Patches extracted: {tissue_tiles}\n")
        f.write(f"Patch dimensions: ({tissue_tiles}, 224, 224, 3)\n")
    
    print(f"Statistics saved to: {stats_file}")
    
    # =============================
    # Summary
    # =============================
    print_section_header("Summary")
    
    results = {
        "wsi_path": wsi_path,
        "detection_level": detection_level,
        "detection_resolution": f"{level_dims[0]} x {level_dims[1]}",
        "classical_detection": {
            "tissue_percentage": f"{tissue_percentage:.2f}%",
            "tissue_pixels": tissue_pixels,
            "total_pixels": total_pixels
        },
        "deep_learning_detection": {
            "total_tiles": total_tiles,
            "tissue_tiles": tissue_tiles,
            "background_tiles": background_tiles,
            "artifact_tiles": artifact_tiles,
            "tile_size": slide_with_detection.tileSize
        },
        "comparison_metrics": {
            "iou": f"{iou:.3f}",
            "dice": f"{dice:.3f}"
        },
        "output_directory": output_dir
    }
    
    create_results_summary(results, output_dir)
    
    print("\nTissue Detection Summary:")
    print(f"  Classical method: {tissue_percentage:.2f}% tissue")
    print(f"  Deep learning: {tissue_tiles}/{total_tiles} tiles contain tissue")
    print(f"  Agreement (IoU): {iou:.3f}")
    print(f"\nExample completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()