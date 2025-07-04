#!/usr/bin/env python3
"""
Example 01: WSI Loading Basics

This example demonstrates the fundamental operations for loading and exploring
Whole Slide Images (WSI) using HoneyBee's Slide loader with CuCIM backend.

Key Features Demonstrated:
- Loading WSI files with GPU acceleration
- Exploring metadata and properties
- Navigating multi-resolution pyramids
- Memory-efficient region reading
- Thumbnail generation
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from honeybee.loaders.Slide.slide import Slide
from utils import (
    print_example_header, print_section_header, get_sample_wsi_path,
    create_output_dir, timer_decorator, format_size, get_device
)
from visualizations import plot_wsi_overview, plot_multi_resolution_pyramid
import json


def main():
    # Setup
    print_example_header(
        "Example 01: WSI Loading Basics",
        "Learn how to load and explore Whole Slide Images with HoneyBee"
    )
    
    # Configuration
    wsi_path = get_sample_wsi_path()
    output_dir = create_output_dir("/mnt/f/Projects/HoneyBee/examples/wsi/tmp", "01_wsi_loading")
    
    # Check if WSI exists
    if not os.path.exists(wsi_path):
        print(f"Error: WSI file not found at {wsi_path}")
        return
    
    print(f"WSI Path: {wsi_path}")
    print(f"Output Directory: {output_dir}")
    
    # =============================
    # 1. Basic WSI Loading
    # =============================
    print_section_header("1. Basic WSI Loading")
    
    @timer_decorator
    def load_wsi():
        """Load WSI with timing."""
        slide = Slide(wsi_path, visualize=False, max_patches=100, verbose=True)
        return slide
    
    slide = load_wsi()
    print(f"WSI loaded successfully!")
    
    # =============================
    # 2. Explore Metadata
    # =============================
    print_section_header("2. WSI Metadata")
    
    # Basic properties from CuImage
    resolutions = slide.img.resolutions
    metadata = slide.img.metadata
    
    print(f"Dimensions: {slide.slide.width} x {slide.slide.height} (at level {slide.selected_level})")
    print(f"Number of levels: {resolutions['level_count']}")
    print(f"Magnification levels: {resolutions['level_dimensions']}")
    print(f"Downsampling factors: {resolutions['level_downsamples']}")
    
    # Full WSI dimensions (level 0)
    level0_dims = resolutions['level_dimensions'][0]
    print(f"Full resolution: {level0_dims[0]} x {level0_dims[1]}")
    
    # Vendor-specific metadata
    print("\nVendor-specific metadata:")
    if 'aperio' in metadata:
        aperio_meta = metadata['aperio']
        print(f"  - Scanner: {aperio_meta.get('ScanScope ID', 'N/A')}")
        print(f"  - Magnification: {aperio_meta.get('AppMag', 'N/A')}x")
        print(f"  - Date: {aperio_meta.get('Date', 'N/A')}")
        print(f"  - MPP: {aperio_meta.get('MPP', 'N/A')} microns/pixel")
    
    # Save full metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nFull metadata saved to: {metadata_path}")
    
    # =============================
    # 3. Multi-Resolution Pyramid
    # =============================
    print_section_header("3. Multi-Resolution Pyramid")
    
    # Load images from different pyramid levels
    pyramid_images = []
    level_info = []
    
    for level in range(min(3, resolutions['level_count'])):  # Show first 3 levels
        # Get level dimensions
        level_dims = resolutions['level_dimensions'][level]
        downsample = resolutions['level_downsamples'][level]
        
        print(f"\nLevel {level}:")
        print(f"  Dimensions: {level_dims}")
        print(f"  Downsample factor: {downsample:.2f}")
        
        # For higher levels, we can read the entire image
        # For level 0, just read a small region to avoid memory issues
        if level == 0:
            # Read a 1000x1000 region from center for level 0
            center = (level_dims[0] // 2, level_dims[1] // 2)
            region_size = min(1000, level_dims[0]), min(1000, level_dims[1])
            location = (center[0] - region_size[0]//2, center[1] - region_size[1]//2)
            
            @timer_decorator
            def read_level_thumbnail():
                region = slide.img.read_region(location, region_size, level)
                return np.asarray(region)
        else:
            # For lower resolution levels, read the entire level
            @timer_decorator
            def read_level_thumbnail():
                region = slide.img.read_region([0, 0], level_dims, level)
                return np.asarray(region)
        
        thumbnail = read_level_thumbnail()
        pyramid_images.append(thumbnail)
        level_info.append(level_dims)
        
        # Memory usage
        print(f"  Thumbnail shape: {thumbnail.shape}")
        print(f"  Memory usage: {format_size(thumbnail.nbytes)}")
    
    # Visualize pyramid
    plot_multi_resolution_pyramid(
        pyramid_images[:3], 
        level_info[:3],
        save_path=os.path.join(output_dir, "multi_resolution_pyramid.png")
    )
    
    # =============================
    # 4. Region Reading
    # =============================
    print_section_header("4. Efficient Region Reading")
    
    # Read specific regions at different magnifications
    region_size = (1000, 1000)
    center_x, center_y = level0_dims[0] // 2, level0_dims[1] // 2
    
    print(f"Reading {region_size[0]}x{region_size[1]} region from center...")
    
    # Read at different levels
    regions = {}
    for level in [0, 1]:  # Level 0 (highest res) and level 1
        location = (
            center_x - region_size[0] // 2,
            center_y - region_size[1] // 2
        )
        
        @timer_decorator
        def read_region():
            return np.asarray(slide.img.read_region(location, region_size, level))
        
        region = read_region()
        regions[f"level_{level}"] = region
        
        print(f"  Level {level} region shape: {region.shape}")
    
    # =============================
    # 5. Thumbnail Generation
    # =============================
    print_section_header("5. Thumbnail Generation")
    
    # Generate overview thumbnail
    @timer_decorator
    def generate_thumbnail():
        # Use the lowest resolution level for thumbnail
        last_level = resolutions['level_count'] - 1
        last_dims = resolutions['level_dimensions'][last_level]
        thumb = slide.img.read_region([0, 0], last_dims, last_level)
        thumb_np = np.asarray(thumb)
        # Resize to target size
        from skimage.transform import resize
        if thumb_np.shape[0] > 800 or thumb_np.shape[1] > 800:
            scale = min(800 / thumb_np.shape[0], 800 / thumb_np.shape[1])
            new_size = (int(thumb_np.shape[0] * scale), int(thumb_np.shape[1] * scale), 3)
            thumb_np = resize(thumb_np, new_size, anti_aliasing=True, 
                            preserve_range=True).astype(np.uint8)
        return thumb_np
    
    thumbnail = generate_thumbnail()
    print(f"Thumbnail shape: {thumbnail.shape}")
    
    # Create metadata dict for visualization
    viz_metadata = {
        'dimensions': f"{level0_dims[0]} x {level0_dims[1]}",
        'level_count': resolutions['level_count'],
        'objective_power': metadata.get('aperio', {}).get('AppMag', 'N/A'),
        'mpp': metadata.get('aperio', {}).get('MPP', 'N/A'),
        'vendor': 'Aperio' if 'aperio' in metadata else 'Unknown'
    }
    
    # Visualize overview
    plot_wsi_overview(
        thumbnail, 
        viz_metadata,
        save_path=os.path.join(output_dir, "wsi_overview.png")
    )
    
    # =============================
    # 6. Memory Management
    # =============================
    print_section_header("6. Memory Management")
    
    # Estimate memory requirements
    print("Memory requirements for full resolution:")
    full_res_bytes = level0_dims[0] * level0_dims[1] * 3  # RGB
    print(f"  Full image size: {format_size(full_res_bytes)}")
    
    print("\nMemory-efficient strategies:")
    print("  - Use appropriate pyramid level for task")
    print("  - Read only required regions")
    print("  - Process in tiles/patches")
    print("  - Use GPU memory when available")
    
    # =============================
    # 7. File Format Information
    # =============================
    print_section_header("7. File Format Information")
    
    # Get file information
    file_stats = os.stat(wsi_path)
    print(f"File size: {format_size(file_stats.st_size)}")
    print(f"File format: {os.path.splitext(wsi_path)[1]}")
    
    # Compression info if available
    if 'tiff' in metadata:
        tiff_meta = metadata['tiff']
        print(f"Compression: {tiff_meta.get('compression', 'Unknown')}")
    
    # =============================
    # Summary
    # =============================
    print_section_header("Summary")
    
    summary = {
        "file_path": wsi_path,
        "file_size": format_size(file_stats.st_size),
        "dimensions": level0_dims,
        "pyramid_levels": resolutions['level_count'],
        "objective_magnification": metadata.get('aperio', {}).get('AppMag', 'N/A'),
        "microns_per_pixel": metadata.get('aperio', {}).get('MPP', 'N/A'),
        "estimated_full_memory": format_size(full_res_bytes),
        "output_directory": output_dir
    }
    
    print("WSI Loading Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExample completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()