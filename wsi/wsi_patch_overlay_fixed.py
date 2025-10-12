#!/usr/bin/env python3
"""
WSI Patch-wise Embedding Visualization with UMAP
This version properly uses tissue detection and creates RGB overlay from UMAP embeddings
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import torch
from PIL import Image
from skimage.transform import resize
import cv2
import warnings
warnings.filterwarnings('ignore')

# UMAP for dimensionality reduction
try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.decomposition import PCA
    HAS_UMAP = False
    print("UMAP not available, using PCA for dimensionality reduction")

# Add HoneyBee to path
honeybee_path = Path(__file__).parent.parent
sys.path.insert(0, str(honeybee_path))

# HoneyBee imports
from honeybee.loaders.Slide.slide import Slide
from honeybee.models.TissueDetector.tissue_detector import TissueDetector
from honeybee.models.UNI.uni import UNI


def create_colored_patch(patch, color_rgb):
    """Create a colored version of a patch based on its embedding color."""
    # Convert to grayscale to get structure
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    
    # Create RGB from grayscale
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    
    # Apply color with blending
    color_float = color_rgb.astype(np.float32) / 255.0
    
    # Blend original colors with embedding colors
    original_float = patch.astype(np.float32) / 255.0
    colored = original_float * 0.3 + (gray_3ch * color_float) * 0.7
    
    # Convert back
    colored = (colored * 255).astype(np.uint8)
    
    return colored


def create_patch_embedding_overlay(
    wsi_path="/mnt/f/Projects/HoneyBee/examples/samples/sample.svs",
    uni_model_path="/mnt/d/Models/UNI/pytorch_model.bin",
    tissue_detector_path=None,  # Optional tissue detector
    tile_size=256,
    max_patches=1000,
    batch_size=16,
    output_dir="./wsi_overlay_umap",
    overlay_alpha=0.7,
    force_level=None
):
    """Create WSI visualization with patch overlay colored by UMAP embeddings."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("WSI Patch Embedding Overlay with UMAP")
    print("="*60)
    
    # Initialize models
    print("\nInitializing models...")
    
    # Tissue detector (optional)
    tissue_detector = None
    if tissue_detector_path and os.path.exists(tissue_detector_path):
        try:
            tissue_detector = TissueDetector(model_path=tissue_detector_path)
            print(f"✓ Tissue detector loaded from {tissue_detector_path}")
        except Exception as e:
            print(f"⚠ Failed to load tissue detector: {e}")
            print("  Will use all patches")
    else:
        print("⚠ No tissue detector, will process all patches")
    
    # UNI model
    uni_model = UNI(model_path=uni_model_path)
    print(f"✓ UNI model loaded on {uni_model.device}")
    
    # Load WSI
    print(f"\nAnalyzing WSI: {wsi_path}")
    
    # Use cucim for efficient loading
    try:
        from cucim import CuImage
        img = CuImage(str(wsi_path))
        resolutions = img.resolutions
    except ImportError:
        print("CuCIM not available, using openslide")
        import openslide
        slide_openslide = openslide.OpenSlide(str(wsi_path))
        resolutions = {
            'level_count': slide_openslide.level_count,
            'level_dimensions': slide_openslide.level_dimensions,
            'level_downsamples': slide_openslide.level_downsamples
        }
        img = slide_openslide
    
    print("\nWSI Structure:")
    for level in range(resolutions['level_count']):
        dims = resolutions['level_dimensions'][level]
        downsample = resolutions['level_downsamples'][level]
        print(f"  Level {level}: {dims[0]:,} x {dims[1]:,} (downsample: {downsample:.1f}x)")
    
    # Choose extraction level
    if force_level is not None:
        extraction_level = force_level
    else:
        # Choose level where we get reasonable number of patches
        for level in range(resolutions['level_count']):
            dims = resolutions['level_dimensions'][level]
            estimated_patches = (dims[0] // tile_size) * (dims[1] // tile_size)
            if estimated_patches <= max_patches * 5:
                extraction_level = level
                break
        else:
            extraction_level = min(1, resolutions['level_count'] - 1)
    
    print(f"\nSelected extraction level: {extraction_level}")
    
    # Load slide
    slide = Slide(
        str(wsi_path),
        tile_size=tile_size,
        max_patches=max_patches * 10,
        tissue_detector=tissue_detector,
        verbose=False
    )
    
    # Extract patches
    print("\nExtracting patches...")
    patches = []
    patch_coords = []
    
    # Get dimensions at extraction level
    level_dims = resolutions['level_dimensions'][extraction_level]
    level_downsample = resolutions['level_downsamples'][extraction_level]
    
    # Calculate grid
    grid_x = level_dims[0] // tile_size
    grid_y = level_dims[1] // tile_size
    
    # Sample patches evenly across the slide
    step_x = max(1, grid_x // int(np.sqrt(max_patches)))
    step_y = max(1, grid_y // int(np.sqrt(max_patches)))
    
    for y in tqdm(range(0, grid_y, step_y), desc="Extracting rows"):
        for x in range(0, grid_x, step_x):
            if len(patches) >= max_patches:
                break
                
            # Calculate coordinates at level 0
            x_coord = int(x * tile_size * level_downsample)
            y_coord = int(y * tile_size * level_downsample)
            
            try:
                # Read patch at extraction level
                if hasattr(img, 'read_region'):
                    patch = img.read_region(
                        location=[x_coord, y_coord],
                        size=[tile_size, tile_size],
                        level=extraction_level
                    )
                else:
                    # OpenSlide
                    patch = img.read_region(
                        (x_coord, y_coord),
                        extraction_level,
                        (tile_size, tile_size)
                    )
                
                patch = np.asarray(patch)
                
                # Remove alpha channel if present
                if patch.shape[2] > 3:
                    patch = patch[:, :, :3]
                
                # Simple tissue check if no detector
                if tissue_detector is None:
                    # Check if patch has tissue (not too white)
                    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                    if np.mean(gray) < 235:  # Not pure white
                        patches.append(patch)
                        patch_coords.append((x * tile_size, y * tile_size))
                else:
                    # Use tissue detector if available
                    patches.append(patch)
                    patch_coords.append((x * tile_size, y * tile_size))
                    
            except Exception as e:
                continue
    
    print(f"Extracted {len(patches)} patches")
    
    if len(patches) == 0:
        print("No patches found!")
        return
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = []
    
    for i in tqdm(range(0, len(patches), batch_size), desc="Embedding batches"):
        batch = patches[i:i + batch_size]
        
        # Prepare batch for UNI
        batch_array = []
        for patch in batch:
            # Resize to 224x224 for UNI
            if patch.shape[0] != 224:
                resized = resize(patch, (224, 224, 3), preserve_range=True).astype(np.uint8)
            else:
                resized = patch
            
            # Normalize
            resized = resized.astype(np.float32) / 255.0
            batch_array.append(resized)
        
        batch_array = np.array(batch_array)
        
        # Generate embeddings
        with torch.no_grad():
            batch_embeddings = uni_model.load_model_and_predict(batch_array)
            embeddings.append(batch_embeddings.cpu().numpy())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings: {embeddings.shape}")
    
    # Dimensionality reduction with UMAP
    print("\nApplying UMAP to reduce to 3D...")
    
    if HAS_UMAP and len(embeddings) > 15:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, len(embeddings)-1),
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        embeddings_3d = reducer.fit_transform(embeddings)
    else:
        print("Using PCA instead of UMAP")
        pca = PCA(n_components=3, random_state=42)
        embeddings_3d = pca.fit_transform(embeddings)
    
    # Normalize to RGB range
    min_vals = embeddings_3d.min(axis=0)
    max_vals = embeddings_3d.max(axis=0)
    embeddings_norm = (embeddings_3d - min_vals) / (max_vals - min_vals + 1e-8)
    patch_colors = (embeddings_norm * 255).astype(np.uint8)
    
    # Create colored patches
    print("\nCreating colored patches...")
    colored_patches = []
    
    for patch, color in tqdm(zip(patches, patch_colors), total=len(patches), desc="Coloring"):
        colored_patch = create_colored_patch(patch, color)
        colored_patches.append(colored_patch)
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Get image at extraction level for visualization
    if hasattr(img, 'read_region'):
        viz_image = img.read_region(
            location=[0, 0],
            size=level_dims,
            level=extraction_level
        )
    else:
        viz_image = img.read_region((0, 0), extraction_level, level_dims)
    
    viz_image = np.asarray(viz_image)
    if viz_image.shape[2] > 3:
        viz_image = viz_image[:, :, :3]
    
    print(f"Visualization at level {extraction_level}: {viz_image.shape}")
    
    # Create overlay
    overlay_canvas = viz_image.copy()
    color_only_canvas = np.ones_like(viz_image) * 255
    
    # Place patches
    for (x, y), colored_patch in tqdm(zip(patch_coords, colored_patches), 
                                      total=len(patch_coords), 
                                      desc="Placing patches"):
        
        y_end = min(y + tile_size, viz_image.shape[0])
        x_end = min(x + tile_size, viz_image.shape[1])
        
        h = y_end - y
        w = x_end - x
        
        if h > 0 and w > 0:
            patch_region = colored_patch[:h, :w]
            
            # Place on canvases
            color_only_canvas[y:y_end, x:x_end] = patch_region
            
            # Blend
            original_region = overlay_canvas[y:y_end, x:x_end]
            overlay_canvas[y:y_end, x:x_end] = (
                overlay_alpha * patch_region + 
                (1 - overlay_alpha) * original_region
            ).astype(np.uint8)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    
    # Layout
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(viz_image)
    ax1.set_title(f'Original WSI (Level {extraction_level})', fontsize=14)
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(color_only_canvas)
    ax2.set_title('UMAP-Colored Patches', fontsize=14)
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(overlay_canvas)
    ax3.set_title(f'Blended Overlay (α={overlay_alpha})', fontsize=14)
    ax3.axis('off')
    
    # Example patches
    ax4 = plt.subplot(2, 3, 4)
    n_examples = min(9, len(patches))
    grid_size = 3
    example_grid = np.ones((grid_size * 100, grid_size * 100, 3), dtype=np.uint8) * 255
    
    for i in range(n_examples):
        row = i // grid_size
        col = i % grid_size
        patch_small = cv2.resize(colored_patches[i], (100, 100))
        example_grid[row*100:(row+1)*100, col*100:(col+1)*100] = patch_small
    
    ax4.imshow(example_grid)
    ax4.set_title('Example UMAP-Colored Patches', fontsize=14)
    ax4.axis('off')
    
    # 3D UMAP scatter
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    scatter = ax5.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=patch_colors/255.0,
        s=30,
        alpha=0.6
    )
    ax5.set_xlabel('UMAP 1')
    ax5.set_ylabel('UMAP 2')
    ax5.set_zlabel('UMAP 3')
    ax5.set_title('3D UMAP Embedding Space', fontsize=14)
    
    # 2D UMAP projection
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1],
                c=patch_colors/255.0, s=30, alpha=0.6)
    ax6.set_xlabel('UMAP Dimension 1')
    ax6.set_ylabel('UMAP Dimension 2')
    ax6.set_title('2D UMAP Projection', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'WSI Patch Embedding Visualization with UMAP\n'
                 f'{Path(wsi_path).name} - {len(patches)} patches at level {extraction_level}',
                 fontsize=16)
    
    plt.tight_layout()
    
    # Save outputs
    output_path = output_dir / f"{Path(wsi_path).stem}_umap_viz.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Save individual outputs
    Image.fromarray(overlay_canvas).save(output_dir / "overlay_umap.png")
    Image.fromarray(color_only_canvas).save(output_dir / "patches_umap.png")
    
    # Save embeddings
    np.save(output_dir / "embeddings_1024d.npy", embeddings)
    np.save(output_dir / "embeddings_3d_umap.npy", embeddings_3d)
    np.save(output_dir / "patch_colors_rgb.npy", patch_colors)
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"- Extracted and visualized at level: {extraction_level}")
    print(f"- Processed {len(patches)} patches of {tile_size}x{tile_size}")
    print(f"- UNI embeddings: {embeddings.shape}")
    print(f"- UMAP reduction: {embeddings_3d.shape}")
    print(f"- Output directory: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate WSI patch embedding overlay with UMAP visualization"
    )
    parser.add_argument("--wsi_path", type=str, 
                       default="/mnt/f/Projects/HoneyBee/examples/samples/sample.svs",
                       help="Path to WSI file")
    parser.add_argument("--uni_model_path", type=str,
                       default="/mnt/d/Models/UNI/pytorch_model.bin",
                       help="Path to UNI model")
    parser.add_argument("--tissue_detector_path", type=str,
                       default=None,
                       help="Path to tissue detector model (optional)")
    parser.add_argument("--tile_size", type=int, default=256,
                       help="Size of patches to extract")
    parser.add_argument("--max_patches", type=int, default=1000,
                       help="Maximum number of patches to process")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for embedding generation")
    parser.add_argument("--output_dir", type=str, default="./wsi_overlay_umap",
                       help="Output directory")
    parser.add_argument("--overlay_alpha", type=float, default=0.7,
                       help="Alpha for overlay blending")
    parser.add_argument("--force_level", type=int, default=None,
                       help="Force extraction at specific pyramid level")
    
    args = parser.parse_args()
    
    create_patch_embedding_overlay(
        wsi_path=args.wsi_path,
        uni_model_path=args.uni_model_path,
        tissue_detector_path=args.tissue_detector_path,
        tile_size=args.tile_size,
        max_patches=args.max_patches,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        overlay_alpha=args.overlay_alpha,
        force_level=args.force_level
    )


if __name__ == "__main__":
    main()