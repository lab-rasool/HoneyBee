#!/usr/bin/env python3
"""
Example 05: Patch Extraction

This example demonstrates comprehensive patch extraction strategies for WSI processing,
including tissue-based filtering, multi-scale extraction, and efficient batch processing.

Key Features Demonstrated:
- Grid-based patch extraction
- Random patch sampling
- Tissue-based filtering with confidence thresholds
- Multi-scale patch extraction
- Patch quality assessment
- Memory-efficient batch processing
- GPU-accelerated extraction with CuCIM
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import cv2
import torch
from PIL import Image
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from honeybee.loaders.Slide.slide import Slide
from honeybee.models.TissueDetector.tissue_detector import TissueDetector
from utils import (
    print_example_header, print_section_header, get_sample_wsi_path,
    get_tissue_detector_path, create_output_dir, timer_decorator,
    validate_paths, save_numpy_compressed, create_results_summary,
    ProgressTracker
)
from visualizations import (
    plot_patch_grid_on_wsi, plot_patch_samples, save_figure
)
import matplotlib.pyplot as plt
from cucim import CuImage


class PatchExtractor:
    """
    Advanced patch extractor with multiple strategies.
    """
    
    def __init__(self, 
                 slide_path: str,
                 tile_size: int = 512,
                 target_size: int = 224,
                 tissue_detector: Optional[TissueDetector] = None,
                 device: str = 'cuda'):
        """
        Initialize patch extractor.
        
        Args:
            slide_path: Path to WSI file
            tile_size: Size of tiles to extract
            target_size: Target size for patches (after resizing)
            tissue_detector: Optional tissue detector model
            device: Device for computation
        """
        self.slide_path = slide_path
        self.tile_size = tile_size
        self.target_size = target_size
        self.tissue_detector = tissue_detector
        self.device = device
        
        # Load slide with CuCIM for GPU acceleration
        self.img = CuImage(slide_path)
        self.resolutions = self.img.resolutions
        
    def extract_grid_patches(self, 
                           level: int = 0,
                           stride: Optional[int] = None,
                           max_patches: int = 1000) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract patches in a regular grid pattern.
        
        Args:
            level: Pyramid level to extract from
            stride: Stride between patches (defaults to tile_size)
            max_patches: Maximum number of patches to extract
            
        Returns:
            Tuple of (patches, coordinates)
        """
        if stride is None:
            stride = self.tile_size
            
        width, height = self.resolutions['level_dimensions'][level]
        
        patches = []
        coordinates = []
        
        # Calculate grid positions
        for y in range(0, height - self.tile_size + 1, stride):
            for x in range(0, width - self.tile_size + 1, stride):
                if len(patches) >= max_patches:
                    break
                    
                # Extract patch
                patch = self.img.read_region(
                    location=(x, y),
                    size=(self.tile_size, self.tile_size),
                    level=level
                )
                patch_array = np.asarray(patch)[:, :, :3]  # RGB only
                
                # Check if patch contains tissue
                if self.tissue_detector is not None:
                    if not self._is_tissue_patch(patch_array):
                        continue
                
                # Resize to target size
                if self.target_size != self.tile_size:
                    patch_array = cv2.resize(patch_array, (self.target_size, self.target_size))
                
                patches.append(patch_array)
                coordinates.append((x, y))
                
            if len(patches) >= max_patches:
                break
                
        return patches, coordinates
    
    def extract_random_patches(self,
                             n_patches: int = 100,
                             level: int = 0,
                             tissue_only: bool = True) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract patches at random locations.
        
        Args:
            n_patches: Number of patches to extract
            level: Pyramid level to extract from
            tissue_only: Whether to only extract tissue patches
            
        Returns:
            Tuple of (patches, coordinates)
        """
        width, height = self.resolutions['level_dimensions'][level]
        
        patches = []
        coordinates = []
        attempts = 0
        max_attempts = n_patches * 10  # Allow multiple attempts to find tissue
        
        while len(patches) < n_patches and attempts < max_attempts:
            attempts += 1
            
            # Random location
            x = np.random.randint(0, width - self.tile_size)
            y = np.random.randint(0, height - self.tile_size)
            
            # Extract patch
            patch = self.img.read_region(
                location=(x, y),
                size=(self.tile_size, self.tile_size),
                level=level
            )
            patch_array = np.asarray(patch)[:, :, :3]
            
            # Check tissue if required
            if tissue_only and self.tissue_detector is not None:
                if not self._is_tissue_patch(patch_array, confidence_threshold=0.8):
                    continue
            
            # Resize to target size
            if self.target_size != self.tile_size:
                patch_array = cv2.resize(patch_array, (self.target_size, self.target_size))
            
            patches.append(patch_array)
            coordinates.append((x, y))
        
        return patches, coordinates
    
    def extract_multiscale_patches(self,
                                 coordinate: Tuple[int, int],
                                 scales: List[int] = [1, 2, 4]) -> Dict[int, np.ndarray]:
        """
        Extract patches at multiple scales from the same location.
        
        Args:
            coordinate: Center coordinate for extraction
            scales: List of scale factors
            
        Returns:
            Dictionary mapping scale to patch
        """
        x, y = coordinate
        multiscale_patches = {}
        
        for scale in scales:
            # Calculate size at this scale
            scaled_size = self.tile_size * scale
            
            # Adjust coordinates to center the patch
            x_start = max(0, x - scaled_size // 2)
            y_start = max(0, y - scaled_size // 2)
            
            # Extract patch
            patch = self.img.read_region(
                location=(x_start, y_start),
                size=(scaled_size, scaled_size),
                level=0
            )
            patch_array = np.asarray(patch)[:, :, :3]
            
            # Always resize to target size for consistency
            patch_resized = cv2.resize(patch_array, (self.target_size, self.target_size))
            
            multiscale_patches[scale] = patch_resized
        
        return multiscale_patches
    
    def extract_with_context(self,
                           coordinate: Tuple[int, int],
                           context_size: int = 128) -> Dict[str, np.ndarray]:
        """
        Extract patch with surrounding context.
        
        Args:
            coordinate: Center coordinate
            context_size: Size of context border
            
        Returns:
            Dictionary with 'patch' and 'context'
        """
        x, y = coordinate
        
        # Extract main patch
        main_patch = self.img.read_region(
            location=(x, y),
            size=(self.tile_size, self.tile_size),
            level=0
        )
        main_array = np.asarray(main_patch)[:, :, :3]
        
        # Extract larger context patch
        context_start_x = max(0, x - context_size)
        context_start_y = max(0, y - context_size)
        context_size_total = self.tile_size + 2 * context_size
        
        context_patch = self.img.read_region(
            location=(context_start_x, context_start_y),
            size=(context_size_total, context_size_total),
            level=0
        )
        context_array = np.asarray(context_patch)[:, :, :3]
        
        # Resize both to target size
        main_resized = cv2.resize(main_array, (self.target_size, self.target_size))
        context_resized = cv2.resize(context_array, (self.target_size, self.target_size))
        
        return {
            'patch': main_resized,
            'context': context_resized
        }
    
    def _is_tissue_patch(self, patch: np.ndarray, confidence_threshold: float = 0.5) -> bool:
        """
        Check if patch contains tissue using tissue detector.
        
        Args:
            patch: Patch array
            confidence_threshold: Minimum confidence for tissue class
            
        Returns:
            True if patch contains tissue
        """
        if self.tissue_detector is None:
            return True
            
        # Resize for tissue detection
        patch_resized = cv2.resize(patch, (224, 224))
        patch_pil = Image.fromarray(patch_resized)
        
        # Apply tissue detector
        patch_transformed = self.tissue_detector.transforms(patch_pil)
        patch_batch = patch_transformed.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.tissue_detector.model(patch_batch)
            prob = torch.nn.functional.softmax(prediction, dim=1).cpu().numpy()[0]
            
        # Check if tissue class (index 2) has high confidence
        return prob[2] >= confidence_threshold
    
    def assess_patch_quality(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Assess quality metrics of a patch.
        
        Args:
            patch: Patch array
            
        Returns:
            Dictionary of quality metrics
        """
        # Convert to grayscale for some metrics
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Calculate metrics
        metrics = {
            'brightness': np.mean(patch),
            'contrast': np.std(patch),
            'saturation': np.std(patch, axis=2).mean(),
            'edge_density': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'focus_measure': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'tissue_percentage': (gray < 230).sum() / gray.size
        }
        
        return metrics


def main():
    # Setup
    print_example_header(
        "Example 05: Patch Extraction",
        "Learn advanced patch extraction strategies for WSI processing"
    )
    
    # Configuration
    wsi_path = get_sample_wsi_path()
    tissue_detector_path = get_tissue_detector_path()
    output_dir = create_output_dir("/mnt/f/Projects/HoneyBee/examples/wsi/tmp", "05_patch_extraction")
    
    # Validate paths
    if not validate_paths(wsi_path, tissue_detector_path):
        print("Error: Required files not found!")
        return
    
    print(f"WSI Path: {wsi_path}")
    print(f"Tissue Detector Path: {tissue_detector_path}")
    print(f"Output Directory: {output_dir}")
    
    # =============================
    # 1. Initialize Components
    # =============================
    print_section_header("1. Initializing Patch Extractor")
    
    # Load tissue detector
    tissue_detector = TissueDetector(model_path=tissue_detector_path)
    print("Tissue detector loaded")
    
    # Initialize patch extractor
    extractor = PatchExtractor(
        slide_path=wsi_path,
        tile_size=512,
        target_size=224,
        tissue_detector=tissue_detector
    )
    
    print(f"Slide loaded with {extractor.resolutions['level_count']} levels")
    for level in range(extractor.resolutions['level_count']):
        dims = extractor.resolutions['level_dimensions'][level]
        print(f"  Level {level}: {dims[0]} x {dims[1]}")
    
    # =============================
    # 2. Grid-based Extraction
    # =============================
    print_section_header("2. Grid-based Patch Extraction")
    
    @timer_decorator
    def extract_grid():
        return extractor.extract_grid_patches(
            level=0,
            stride=1024,  # Half overlap
            max_patches=50
        )
    
    grid_patches, grid_coords = extract_grid()
    print(f"Extracted {len(grid_patches)} patches in grid pattern")
    
    # =============================
    # 3. Random Patch Extraction
    # =============================
    print_section_header("3. Random Patch Extraction")
    
    @timer_decorator
    def extract_random():
        return extractor.extract_random_patches(
            n_patches=30,
            level=0,
            tissue_only=True
        )
    
    random_patches, random_coords = extract_random()
    print(f"Extracted {len(random_patches)} random tissue patches")
    
    # =============================
    # 4. Multi-scale Extraction
    # =============================
    print_section_header("4. Multi-scale Patch Extraction")
    
    # Select a few coordinates for multi-scale extraction
    sample_coords = random_coords[:3] if len(random_coords) >= 3 else random_coords
    
    multiscale_patches = []
    for coord in sample_coords:
        ms_patches = extractor.extract_multiscale_patches(coord, scales=[1, 2, 4])
        multiscale_patches.append(ms_patches)
    
    print(f"Extracted multi-scale patches from {len(sample_coords)} locations")
    print(f"Scales: {list(multiscale_patches[0].keys()) if multiscale_patches else []}")
    
    # =============================
    # 5. Context-aware Extraction
    # =============================
    print_section_header("5. Context-aware Patch Extraction")
    
    context_patches = []
    for coord in sample_coords:
        ctx_patch = extractor.extract_with_context(coord, context_size=128)
        context_patches.append(ctx_patch)
    
    print(f"Extracted {len(context_patches)} patches with context")
    
    # =============================
    # 6. Patch Quality Assessment
    # =============================
    print_section_header("6. Patch Quality Assessment")
    
    # Assess quality of random patches
    quality_metrics = []
    for patch in random_patches[:10]:
        metrics = extractor.assess_patch_quality(patch)
        quality_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    for key in quality_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in quality_metrics])
    
    print("Average quality metrics (10 patches):")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # =============================
    # 7. Efficient Batch Processing
    # =============================
    print_section_header("7. Efficient Batch Processing")
    
    # Demonstrate parallel extraction
    def extract_patch_batch(coords_batch):
        """Extract a batch of patches."""
        patches = []
        for x, y in coords_batch:
            patch = extractor.img.read_region(
                location=(x, y),
                size=(extractor.tile_size, extractor.tile_size),
                level=0
            )
            patch_array = np.asarray(patch)[:, :, :3]
            patches.append(patch_array)
        return patches
    
    # Generate test coordinates
    test_coords = [(i * 1000, j * 1000) for i in range(5) for j in range(5)]
    
    # Split into batches
    batch_size = 5
    coord_batches = [test_coords[i:i+batch_size] for i in range(0, len(test_coords), batch_size)]
    
    @timer_decorator
    def parallel_extraction():
        """Parallel patch extraction."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(extract_patch_batch, coord_batches))
        return [patch for batch in results for patch in batch]
    
    parallel_patches = parallel_extraction()
    print(f"Extracted {len(parallel_patches)} patches in parallel")
    
    # =============================
    # 8. Visualizations
    # =============================
    print_section_header("8. Creating Visualizations")
    
    # Load slide for visualization
    slide = Slide(wsi_path, visualize=False, max_patches=100, tile_size=1024)
    thumbnail = np.asarray(slide.slide)
    
    # Visualize grid extraction pattern
    plot_patch_grid_on_wsi(
        thumbnail,
        grid_coords[:20],  # Show first 20
        patch_size=512,
        save_path=os.path.join(output_dir, "grid_extraction_pattern.png")
    )
    
    # Visualize random extraction
    plot_patch_grid_on_wsi(
        thumbnail,
        random_coords[:20],  # Show first 20
        patch_size=512,
        highlight_indices=list(range(20)),  # Highlight all as tissue
        save_path=os.path.join(output_dir, "random_extraction_pattern.png")
    )
    
    # Visualize patch samples
    if len(random_patches) > 0:
        plot_patch_samples(
            random_patches[:12],
            labels=[f"Patch {i+1}" for i in range(min(12, len(random_patches)))],
            n_samples=12,
            save_path=os.path.join(output_dir, "extracted_patches_sample.png")
        )
    
    # Visualize multi-scale patches
    if multiscale_patches:
        fig, axes = plt.subplots(len(multiscale_patches), 3, figsize=(12, 4*len(multiscale_patches)))
        if len(multiscale_patches) == 1:
            axes = axes.reshape(1, -1)
        
        for i, ms_patch_dict in enumerate(multiscale_patches):
            for j, (scale, patch) in enumerate(ms_patch_dict.items()):
                axes[i, j].imshow(patch)
                axes[i, j].set_title(f'Location {i+1}, Scale {scale}x')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, "multiscale_patches.png"))
        plt.close()
    
    # Visualize context patches
    if context_patches:
        fig, axes = plt.subplots(len(context_patches), 2, figsize=(8, 4*len(context_patches)))
        if len(context_patches) == 1:
            axes = axes.reshape(1, -1)
        
        for i, ctx_dict in enumerate(context_patches):
            axes[i, 0].imshow(ctx_dict['patch'])
            axes[i, 0].set_title(f'Patch {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(ctx_dict['context'])
            axes[i, 1].set_title(f'With Context {i+1}')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, "context_patches.png"))
        plt.close()
    
    # =============================
    # 9. Save Results
    # =============================
    print_section_header("9. Saving Results")
    
    # Save extracted patches
    if grid_patches:
        save_numpy_compressed(
            np.array(grid_patches),
            os.path.join(output_dir, "grid_patches")
        )
    
    if random_patches:
        save_numpy_compressed(
            np.array(random_patches),
            os.path.join(output_dir, "random_patches")
        )
    
    # Save coordinates
    np.save(os.path.join(output_dir, "grid_coordinates.npy"), grid_coords)
    np.save(os.path.join(output_dir, "random_coordinates.npy"), random_coords)
    
    # =============================
    # Summary
    # =============================
    print_section_header("Summary")
    
    results = {
        "wsi_path": wsi_path,
        "extraction_methods": {
            "grid": {
                "patches_extracted": len(grid_patches),
                "stride": 1024,
                "overlap": "50%"
            },
            "random": {
                "patches_extracted": len(random_patches),
                "tissue_only": True,
                "confidence_threshold": 0.8
            },
            "multiscale": {
                "locations": len(multiscale_patches),
                "scales": [1, 2, 4]
            },
            "context": {
                "patches": len(context_patches),
                "context_size": 128
            }
        },
        "quality_metrics": avg_metrics,
        "performance": {
            "parallel_extraction": f"{len(parallel_patches)} patches",
            "workers": 4
        },
        "output_directory": output_dir
    }
    
    create_results_summary(results, output_dir)
    
    print("\nPatch Extraction Summary:")
    print(f"  Grid extraction: {len(grid_patches)} patches")
    print(f"  Random extraction: {len(random_patches)} tissue patches")
    print(f"  Multi-scale: {len(multiscale_patches)} locations with 3 scales each")
    print(f"  Quality assessment: Average tissue percentage {avg_metrics.get('tissue_percentage', 0):.2%}")
    print(f"\nExample completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()