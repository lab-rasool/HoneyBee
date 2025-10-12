#!/usr/bin/env python3
"""
WSI Patch-wise Embedding Visualization - Final Version

This version ensures patches are extracted and visualized at compatible resolutions.
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
    
    # Apply color
    color_float = color_rgb.astype(np.float32) / 255.0
    colored = gray_3ch * color_float
    
    # Convert back
    colored = (colored * 255).astype(np.uint8)
    
    return colored


def create_patch_embedding_overlay_final(
    wsi_path="/mnt/f/Projects/HoneyBee/examples/samples/sample.svs",
    uni_model_path="/mnt/d/Models/UNI/pytorch_model.bin",
    tissue_detector_path="/mnt/d/Models/TissueDetector/HnE.pt",
    tile_size=256,
    max_patches=500,
    batch_size=16,
    output_dir="./wsi_overlay_final",
    overlay_alpha=0.7,
    force_level=None  # Force a specific level for extraction
):
    """Create WSI visualization with patch overlay at appropriate resolution."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("WSI Patch Embedding Overlay - Final Version")
    print("="*60)
    
    # Initialize models
    print("\nInitializing models...")
    
    # Tissue detector
    if tissue_detector_path and tissue_detector_path.lower() != "none" and os.path.exists(tissue_detector_path):
        tissue_detector = TissueDetector(model_path=tissue_detector_path)
        print(f"✓ Tissue detector loaded")
    else:
        tissue_detector = None
        print("⚠ No tissue detector, will process all patches")
    
    # UNI model
    uni_model = UNI(model_path=uni_model_path)
    print(f"✓ UNI model loaded on {uni_model.device}")
    
    # Analyze WSI structure first
    print(f"\nAnalyzing WSI: {wsi_path}")
    
    # Load minimal slide to check structure
    from cucim import CuImage
    img = CuImage(str(wsi_path))
    resolutions = img.resolutions
    
    print("\nWSI Structure:")
    for level in range(resolutions['level_count']):
        dims = resolutions['level_dimensions'][level]
        downsample = resolutions['level_downsamples'][level]
        print(f"  Level {level}: {dims[0]:,} x {dims[1]:,} (downsample: {downsample:.1f}x)")
    
    # Choose extraction level based on patch count desired
    # We want patches to be meaningful in the visualization
    if force_level is not None:
        extraction_level = force_level
    else:
        # Choose level where we get reasonable number of patches
        for level in range(resolutions['level_count']):
            dims = resolutions['level_dimensions'][level]
            estimated_patches = (dims[0] // tile_size) * (dims[1] // tile_size)
            if estimated_patches <= max_patches * 10:  # Some headroom
                extraction_level = level
                break
        else:
            extraction_level = resolutions['level_count'] - 1
    
    print(f"\nSelected extraction level: {extraction_level}")
    
    # Load slide at chosen level
    slide = Slide(
        str(wsi_path),
        tile_size=tile_size,
        max_patches=max_patches * 10,  # Load more initially
        tissue_detector=tissue_detector,
        verbose=False
    )
    
    # Force slide to use our chosen level
    slide.selected_level = extraction_level
    slide.slide = slide.img.read_region(location=[0, 0], level=extraction_level)
    slide.slide.height = int(resolutions['level_dimensions'][extraction_level][1])
    slide.slide.width = int(resolutions['level_dimensions'][extraction_level][0])
    
    # Recalculate grid
    slide.numTilesInX = slide.slide.width // (tile_size - slide.tileOverlap)
    slide.numTilesInY = slide.slide.height // (tile_size - slide.tileOverlap)
    slide.tileDictionary = slide._generate_tile_dictionary()
    
    print(f"\nExtracting from level {extraction_level}:")
    print(f"  Dimensions: {slide.slide.width} x {slide.slide.height}")
    print(f"  Tile size: {tile_size}")
    print(f"  Grid: {slide.numTilesInX} x {slide.numTilesInY}")
    
    # If we have tissue detector, run it
    if tissue_detector is not None:
        print("\nRunning tissue detection...")
        slide.detectTissue(batchSize=8)
    
    # Extract patches
    print("\nExtracting patches...")
    patches = []
    patch_addresses = []
    patch_coords = []
    
    tissue_threshold = 0.5
    count = 0
    
    # Get all tile addresses
    all_addresses = list(slide.iterateTiles())
    
    # Sample evenly if we have too many
    if len(all_addresses) > max_patches:
        import random
        random.seed(42)
        sampled_addresses = random.sample(all_addresses, max_patches)
    else:
        sampled_addresses = all_addresses[:max_patches]
    
    for address in tqdm(sampled_addresses, desc="Extracting"):
        tile_info = slide.tileDictionary[address]
        
        # Check tissue
        include = tissue_detector is None or tile_info.get('tissueLevel', 1.0) >= tissue_threshold
        
        if include:
            try:
                # Extract at extraction level
                start_x = tile_info['x']
                start_y = tile_info['y']
                
                patch = slide.img.read_region(
                    location=[start_x, start_y],
                    size=[tile_size, tile_size],
                    level=extraction_level
                )
                patch = np.asarray(patch)
                
                if patch.shape[2] > 3:
                    patch = patch[:, :, :3]
                
                patches.append(patch)
                patch_addresses.append(address)
                patch_coords.append((start_x, start_y))
                count += 1
                
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
        
        # Resize for UNI
        resized_batch = []
        for patch in batch:
            if patch.shape[0] != 224:
                resized = resize(patch, (224, 224, 3), preserve_range=True).astype(np.uint8)
            else:
                resized = patch
            resized_batch.append(resized)
        
        resized_batch = np.array(resized_batch)
        
        # Generate embeddings
        with torch.no_grad():
            batch_embeddings = uni_model.load_model_and_predict(resized_batch)
            embeddings.append(batch_embeddings.cpu().numpy())
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    embeddings = np.vstack(embeddings)
    print(f"Generated embeddings: {embeddings.shape}")
    
    # Dimensionality reduction
    print("\nReducing to 3D for RGB mapping...")
    
    if HAS_UMAP and len(embeddings) > 15:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, len(embeddings)-1),
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        embeddings_3d = reducer.fit_transform(embeddings)
    else:
        pca = PCA(n_components=3, random_state=42)
        embeddings_3d = pca.fit_transform(embeddings)
    
    # Normalize to RGB
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
    
    # Create visualization at extraction level
    print("\nCreating visualization...")
    
    # Get image at extraction level
    viz_dims = resolutions['level_dimensions'][extraction_level]
    viz_image = np.asarray(slide.slide)
    
    print(f"Visualization at level {extraction_level}: {viz_image.shape}")
    
    # Create overlay
    overlay_canvas = viz_image.copy()
    color_only_canvas = np.ones_like(viz_image) * 255
    
    # Place patches directly (no scaling needed - same level)
    for (x, y), colored_patch in tqdm(zip(patch_coords, colored_patches), 
                                      total=len(patch_coords), 
                                      desc="Placing patches"):
        
        # Direct placement
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
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    
    # Layout
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(viz_image)
    ax1.set_title(f'Original WSI (Level {extraction_level})', fontsize=14)
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(color_only_canvas)
    ax2.set_title('Colored Patches', fontsize=14)
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(overlay_canvas)
    ax3.set_title(f'Overlay (α={overlay_alpha})', fontsize=14)
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
    ax4.set_title('Example Patches', fontsize=14)
    ax4.axis('off')
    
    # 3D scatter
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    scatter = ax5.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        c=patch_colors/255.0,
        s=30,
        alpha=0.6
    )
    ax5.set_xlabel('Dim 1')
    ax5.set_ylabel('Dim 2')
    ax5.set_zlabel('Dim 3')
    ax5.set_title('3D Embedding Space', fontsize=14)
    
    # 2D projection
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1],
                c=patch_colors/255.0, s=30, alpha=0.6)
    ax6.set_xlabel('Dimension 1')
    ax6.set_ylabel('Dimension 2')
    ax6.set_title('2D Projection', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'WSI Patch Embedding Visualization\n'
                 f'{Path(wsi_path).name} - {len(patches)} patches at level {extraction_level}',
                 fontsize=16)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f"{Path(wsi_path).stem}_final_viz.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    # Save individual outputs
    Image.fromarray(overlay_canvas).save(output_dir / "overlay_final.png")
    Image.fromarray(color_only_canvas).save(output_dir / "patches_final.png")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"- Extracted and visualized at level: {extraction_level}")
    print(f"- Processed {len(patches)} patches of {tile_size}x{tile_size}")
    print(f"- Dimensions at this level: {viz_dims[0]} x {viz_dims[1]}")
    print(f"- Output: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate final WSI patch embedding overlay"
    )
    parser.add_argument("--wsi_path", type=str, 
                       default="/mnt/f/Projects/HoneyBee/examples/samples/sample.svs")
    parser.add_argument("--uni_model_path", type=str,
                       default="/mnt/d/Models/UNI/pytorch_model.bin")
    parser.add_argument("--tissue_detector_path", type=str,
                       default="/mnt/d/Models/TissueDetector/HnE.pt")
    parser.add_argument("--tile_size", type=int, default=256,
                       help="Size of patches to extract")
    parser.add_argument("--max_patches", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./wsi_overlay_final")
    parser.add_argument("--overlay_alpha", type=float, default=0.7)
    parser.add_argument("--force_level", type=int, default=None,
                       help="Force extraction at specific pyramid level")
    
    args = parser.parse_args()
    
    create_patch_embedding_overlay_final(
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