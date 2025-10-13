#!/usr/bin/env python3
"""
WSI Patch-wise Embedding Visualization with UMAP using CuCIM
Enhanced version using CuCIM for faster loading and better tissue detection
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
from honeybee.models.UNI.uni import UNI

# CuCIM for fast WSI loading
from cucim import CuImage


def simple_tissue_detection(patch, threshold=0.8):
    """Simple tissue detection based on color statistics."""
    # Convert to grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    
    # Calculate tissue score (lower values = more tissue)
    white_pixels = np.sum(gray > 230) / gray.size
    
    # Also check for variance (tissue has more texture)
    variance = np.var(gray)
    
    # Combined score
    is_tissue = (white_pixels < threshold) and (variance > 100)
    
    return is_tissue, 1.0 - white_pixels


def create_enhanced_colored_patch(patch, color_rgb, enhancement=1.5):
    """Create an enhanced colored version of a patch based on its embedding color."""
    # Convert to LAB for better color manipulation
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance contrast in lightness channel
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    
    # Apply color based on embedding
    color_hsv = cv2.cvtColor(color_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
    
    # Create colored version
    colored_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Shift hue towards embedding color
    hue_shift = color_hsv[0] - colored_hsv[:, :, 0].mean()
    colored_hsv[:, :, 0] = (colored_hsv[:, :, 0] + hue_shift * enhancement) % 180
    
    # Enhance saturation based on embedding
    colored_hsv[:, :, 1] = np.clip(colored_hsv[:, :, 1] * (1 + enhancement * (color_hsv[1] / 255)), 0, 255)
    
    # Convert back to RGB
    colored = cv2.cvtColor(colored_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Blend with original
    result = cv2.addWeighted(patch, 0.3, colored, 0.7, 0)
    
    return result


def create_patch_embedding_overlay_cuimage(
    wsi_path="/mnt/f/Projects/HoneyBee/examples/samples/sample.svs",
    uni_model_path="/mnt/d/Models/UNI/pytorch_model.bin",
    tile_size=256,
    max_patches=1000,
    batch_size=32,
    output_dir="./wsi_overlay_cuimage",
    overlay_alpha=0.6,
    extraction_level=None,
    tissue_threshold=0.8,
    min_tissue_area=0.3
):
    """Create WSI visualization with patch overlay using CuCIM for speed."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("WSI Patch Embedding Overlay with UMAP (CuCIM Enhanced)")
    print("="*60)
    
    # Initialize UNI model
    print("\nInitializing UNI model...")
    uni_model = UNI(model_path=uni_model_path)
    print(f"✓ UNI model loaded on {uni_model.device}")
    
    # Load WSI with CuCIM
    print(f"\nLoading WSI with CuCIM: {wsi_path}")
    start_time = time.time()
    
    img = CuImage(str(wsi_path))
    resolutions = img.resolutions
    
    load_time = time.time() - start_time
    print(f"✓ WSI loaded in {load_time:.2f}s")
    
    print("\nWSI Properties:")
    print(f"  Dimensions: {img.shape}")
    print(f"  Number of levels: {resolutions['level_count']}")
    print(f"  Spacing: {img.spacing}")
    
    print("\nPyramid levels:")
    for level in range(resolutions['level_count']):
        dims = resolutions['level_dimensions'][level]
        downsample = resolutions['level_downsamples'][level]
        print(f"  Level {level}: {dims[0]:,} x {dims[1]:,} (downsample: {downsample:.1f}x)")
    
    # Choose extraction level
    if extraction_level is None:
        # Auto-select based on desired patch count
        for level in range(resolutions['level_count']):
            dims = resolutions['level_dimensions'][level]
            estimated_patches = (dims[0] // tile_size) * (dims[1] // tile_size)
            if estimated_patches <= max_patches * 4:
                extraction_level = level
                break
        else:
            extraction_level = min(1, resolutions['level_count'] - 1)
    
    print(f"\nUsing extraction level: {extraction_level}")
    level_dims = resolutions['level_dimensions'][extraction_level]
    level_downsample = resolutions['level_downsamples'][extraction_level]
    
    # Extract patches with tissue detection
    print("\nExtracting patches with tissue detection...")
    patches = []
    patch_coords = []
    tissue_scores = []
    
    # Calculate grid
    grid_x = level_dims[0] // tile_size
    grid_y = level_dims[1] // tile_size
    total_tiles = grid_x * grid_y
    
    print(f"  Grid size: {grid_x} x {grid_y} = {total_tiles} potential patches")
    
    # Sample patches with stride for efficiency
    stride = max(1, int(np.sqrt(total_tiles / max_patches)))
    
    # Use CuCIM's efficient region reading
    extracted = 0
    skipped = 0
    
    pbar = tqdm(total=min(max_patches, (grid_y // stride) * (grid_x // stride)), 
                desc="Extracting patches")
    
    for y in range(0, grid_y, stride):
        for x in range(0, grid_x, stride):
            if extracted >= max_patches:
                break
                
            # Calculate coordinates
            x_coord = x * tile_size
            y_coord = y * tile_size
            
            # Read patch using CuCIM
            try:
                # CuCIM read_region with level
                location = [int(x_coord * level_downsample), 
                           int(y_coord * level_downsample)]
                
                patch = img.read_region(
                    location=location,
                    size=[tile_size, tile_size],
                    level=extraction_level
                )
                patch = np.asarray(patch)[:, :, :3]  # Remove alpha
                
                # Tissue detection
                is_tissue, tissue_score = simple_tissue_detection(patch, tissue_threshold)
                
                if is_tissue and tissue_score > min_tissue_area:
                    patches.append(patch)
                    patch_coords.append((x_coord, y_coord))
                    tissue_scores.append(tissue_score)
                    extracted += 1
                    pbar.update(1)
                else:
                    skipped += 1
                    
            except Exception as e:
                continue
    
    pbar.close()
    
    print(f"\n✓ Extracted {len(patches)} tissue patches")
    print(f"  Skipped {skipped} non-tissue patches")
    print(f"  Average tissue score: {np.mean(tissue_scores):.3f}")
    
    if len(patches) == 0:
        print("No tissue patches found!")
        return
    
    # Generate embeddings
    print("\nGenerating UNI embeddings...")
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Embedding batches"):
            batch = patches[i:i + batch_size]
            
            # Prepare batch
            batch_array = []
            for patch in batch:
                # Resize to 224x224 for UNI
                if patch.shape[0] != 224:
                    resized = resize(patch, (224, 224, 3), preserve_range=True).astype(np.uint8)
                else:
                    resized = patch
                
                # Normalize to [0, 1]
                normalized = resized.astype(np.float32) / 255.0
                batch_array.append(normalized)
            
            batch_array = np.array(batch_array)
            
            # Generate embeddings
            batch_embeddings = uni_model.load_model_and_predict(batch_array)
            embeddings.append(batch_embeddings.cpu().numpy())
            
            # Clear cache periodically
            if torch.cuda.is_available() and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
    
    embeddings = np.vstack(embeddings)
    print(f"✓ Generated embeddings: {embeddings.shape}")
    
    # Apply UMAP
    print("\nApplying UMAP for dimensionality reduction...")
    
    if HAS_UMAP and len(embeddings) > 15:
        # Configure UMAP for better results
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(30, len(embeddings)-1),
            min_dist=0.1,
            spread=1.0,
            metric='cosine',
            random_state=42,
            n_epochs=500,
            init='spectral',
            verbose=True
        )
        embeddings_3d = reducer.fit_transform(embeddings)
        
        # Additional 2D projection for visualization
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, len(embeddings)-1),
            min_dist=0.1,
            spread=1.0,
            metric='cosine',
            random_state=42,
            n_epochs=300,
            init='spectral'
        )
        embeddings_2d = reducer_2d.fit_transform(embeddings)
    else:
        print("Using PCA (install umap-learn for better results)")
        pca_3d = PCA(n_components=3, random_state=42)
        embeddings_3d = pca_3d.fit_transform(embeddings)
        
        pca_2d = PCA(n_components=2, random_state=42)
        embeddings_2d = pca_2d.fit_transform(embeddings)
    
    # Normalize to RGB
    min_vals = embeddings_3d.min(axis=0)
    max_vals = embeddings_3d.max(axis=0)
    embeddings_norm = (embeddings_3d - min_vals) / (max_vals - min_vals + 1e-8)
    patch_colors = (embeddings_norm * 255).astype(np.uint8)
    
    # Create enhanced colored patches
    print("\nCreating enhanced colored patches...")
    colored_patches = []
    
    for patch, color, tissue_score in tqdm(zip(patches, patch_colors, tissue_scores), 
                                           total=len(patches), desc="Coloring"):
        # Enhance coloring based on tissue score
        enhancement = 0.5 + tissue_score
        colored_patch = create_enhanced_colored_patch(patch, color, enhancement)
        colored_patches.append(colored_patch)
    
    # Create visualization at extraction level
    print("\nCreating final visualization...")
    
    # Read full image at extraction level
    viz_image = img.read_region(
        location=[0, 0],
        size=level_dims,
        level=extraction_level
    )
    viz_image = np.asarray(viz_image)[:, :, :3]
    
    # Create canvases
    overlay_canvas = viz_image.copy()
    color_only_canvas = np.ones_like(viz_image) * 240  # Light gray background
    heatmap_canvas = np.zeros((viz_image.shape[0], viz_image.shape[1]), dtype=np.float32)
    
    # Place patches with smooth blending
    for (x, y), colored_patch, tissue_score in tqdm(zip(patch_coords, colored_patches, tissue_scores), 
                                                    total=len(patch_coords), 
                                                    desc="Placing patches"):
        
        y_end = min(y + tile_size, viz_image.shape[0])
        x_end = min(x + tile_size, viz_image.shape[1])
        
        h = y_end - y
        w = x_end - x
        
        if h > 0 and w > 0:
            patch_region = colored_patch[:h, :w]
            
            # Update canvases
            color_only_canvas[y:y_end, x:x_end] = patch_region
            heatmap_canvas[y:y_end, x:x_end] = tissue_score
            
            # Smooth blending for overlay
            original_region = overlay_canvas[y:y_end, x:x_end]
            alpha = overlay_alpha * tissue_score  # Vary alpha by tissue content
            overlay_canvas[y:y_end, x:x_end] = (
                alpha * patch_region + 
                (1 - alpha) * original_region
            ).astype(np.uint8)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(24, 18))
    
    # Main visualizations
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(viz_image)
    ax1.set_title(f'Original WSI (Level {extraction_level})', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(color_only_canvas)
    ax2.set_title('UMAP-Colored Patches', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(overlay_canvas)
    ax3.set_title(f'Blended Overlay', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Tissue heatmap
    ax4 = plt.subplot(3, 3, 4)
    im = ax4.imshow(heatmap_canvas, cmap='hot', vmin=0, vmax=1)
    ax4.set_title('Tissue Density Heatmap', fontsize=14, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    # 3D UMAP scatter
    ax5 = plt.subplot(3, 3, 5, projection='3d')
    scatter = ax5.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=patch_colors/255.0,
        s=50,
        alpha=0.7,
        edgecolors='gray',
        linewidth=0.5
    )
    ax5.set_xlabel('UMAP 1', fontsize=12)
    ax5.set_ylabel('UMAP 2', fontsize=12)
    ax5.set_zlabel('UMAP 3', fontsize=12)
    ax5.set_title('3D UMAP Embedding Space', fontsize=14, fontweight='bold')
    ax5.view_init(elev=20, azim=45)
    
    # 2D UMAP projection
    ax6 = plt.subplot(3, 3, 6)
    scatter2d = ax6.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=patch_colors/255.0, s=60, alpha=0.7,
                           edgecolors='gray', linewidth=0.5)
    ax6.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax6.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax6.set_title('2D UMAP Projection', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Example patches grid
    ax7 = plt.subplot(3, 3, 7)
    n_examples = min(16, len(patches))
    grid_size = int(np.sqrt(n_examples))
    example_size = 80
    example_grid = np.ones((grid_size * example_size, grid_size * example_size, 3), dtype=np.uint8) * 255
    
    for i in range(n_examples):
        row = i // grid_size
        col = i % grid_size
        patch_small = cv2.resize(colored_patches[i], (example_size, example_size))
        y_start = row * example_size
        x_start = col * example_size
        example_grid[y_start:y_start+example_size, x_start:x_start+example_size] = patch_small
    
    ax7.imshow(example_grid)
    ax7.set_title('Example Colored Patches', fontsize=14, fontweight='bold')
    ax7.axis('off')
    
    # Statistics
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(tissue_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax8.set_xlabel('Tissue Score', fontsize=12)
    ax8.set_ylabel('Count', fontsize=12)
    ax8.set_title('Tissue Score Distribution', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Embedding statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.text(0.1, 0.9, f'WSI: {Path(wsi_path).name}', fontsize=12, transform=ax9.transAxes)
    ax9.text(0.1, 0.8, f'Patches extracted: {len(patches)}', fontsize=12, transform=ax9.transAxes)
    ax9.text(0.1, 0.7, f'Extraction level: {extraction_level}', fontsize=12, transform=ax9.transAxes)
    ax9.text(0.1, 0.6, f'Patch size: {tile_size}x{tile_size}', fontsize=12, transform=ax9.transAxes)
    ax9.text(0.1, 0.5, f'Embedding dim: {embeddings.shape[1]}', fontsize=12, transform=ax9.transAxes)
    ax9.text(0.1, 0.4, f'UMAP components: 3D + 2D', fontsize=12, transform=ax9.transAxes)
    ax9.text(0.1, 0.3, f'Avg tissue score: {np.mean(tissue_scores):.3f}', fontsize=12, transform=ax9.transAxes)
    ax9.text(0.1, 0.2, f'Processing time: {time.time() - start_time:.1f}s', fontsize=12, transform=ax9.transAxes)
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    ax9.set_title('Summary Statistics', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'WSI Patch Embedding Analysis with UMAP\n{Path(wsi_path).name}',
                 fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    # Save outputs
    output_path = output_dir / f"{Path(wsi_path).stem}_cuimage_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved comprehensive analysis to: {output_path}")
    
    # Save individual components
    Image.fromarray(overlay_canvas).save(output_dir / "overlay_cuimage.png")
    Image.fromarray(color_only_canvas).save(output_dir / "patches_cuimage.png")
    
    # Save heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_canvas, cmap='hot')
    plt.colorbar(label='Tissue Score')
    plt.title('Tissue Density Heatmap')
    plt.axis('off')
    plt.savefig(output_dir / "tissue_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save data
    np.save(output_dir / "embeddings_1024d.npy", embeddings)
    np.save(output_dir / "embeddings_3d_umap.npy", embeddings_3d)
    np.save(output_dir / "embeddings_2d_umap.npy", embeddings_2d)
    np.save(output_dir / "patch_colors_rgb.npy", patch_colors)
    np.save(output_dir / "tissue_scores.npy", np.array(tissue_scores))
    
    # Save metadata
    metadata = {
        'wsi_path': str(wsi_path),
        'n_patches': len(patches),
        'extraction_level': extraction_level,
        'tile_size': tile_size,
        'level_dimensions': level_dims,
        'level_downsample': level_downsample,
        'processing_time': time.time() - start_time
    }
    
    import json
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    print("  - Comprehensive analysis figure")
    print("  - Individual overlay images")
    print("  - Tissue density heatmap")
    print("  - Embeddings (1024D, 3D, 2D)")
    print("  - Metadata JSON")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate WSI patch embedding overlay with UMAP using CuCIM"
    )
    parser.add_argument("--wsi_path", type=str, 
                       default="/mnt/f/Projects/HoneyBee/examples/samples/sample.svs",
                       help="Path to WSI file")
    parser.add_argument("--uni_model_path", type=str,
                       default="/mnt/d/Models/UNI/pytorch_model.bin",
                       help="Path to UNI model")
    parser.add_argument("--tile_size", type=int, default=64,
                       help="Size of patches to extract")
    parser.add_argument("--max_patches", type=int, default=10000,
                       help="Maximum number of patches to process")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for embedding generation")
    parser.add_argument("--output_dir", type=str, default="./wsi_overlay_cuimage",
                       help="Output directory")
    parser.add_argument("--overlay_alpha", type=float, default=0.6,
                       help="Alpha for overlay blending")
    parser.add_argument("--extraction_level", type=int, default=None,
                       help="WSI pyramid level for extraction")
    parser.add_argument("--tissue_threshold", type=float, default=0.8,
                       help="Threshold for tissue detection")
    parser.add_argument("--min_tissue_area", type=float, default=0.3,
                       help="Minimum tissue area fraction")
    
    args = parser.parse_args()
    
    create_patch_embedding_overlay_cuimage(
        wsi_path=args.wsi_path,
        uni_model_path=args.uni_model_path,
        tile_size=args.tile_size,
        max_patches=args.max_patches,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        overlay_alpha=args.overlay_alpha,
        extraction_level=args.extraction_level,
        tissue_threshold=args.tissue_threshold,
        min_tissue_area=args.min_tissue_area
    )


if __name__ == "__main__":
    main()