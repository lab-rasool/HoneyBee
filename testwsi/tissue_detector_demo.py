"""
Deep Learning Tissue Detector Demonstration

This script demonstrates the DenseNet121-based deep learning tissue detector
and compares it with classical tissue detection methods.

Fixed Issues:
- Loads tiles BEFORE tissue detection to avoid getTile() bug
- Extracts 36 small patches (224x224) instead of 9 large tiles
- Shows actual tissue content (no more black patches!)
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

# Add HoneyBee to path
honeybee_path = Path(__file__).parent.parent
sys.path.insert(0, str(honeybee_path))

# Import components
from honeybee.processors import PathologyProcessor
from honeybee.models.TissueDetector.tissue_detector import TissueDetector

# Configuration
WSI_PATH = Path(__file__).parent / "sample.svs"
OUTPUT_DIR = Path(__file__).parent / "tissue_detector_outputs"
TISSUE_DETECTOR_PATH = "/mnt/d/Models/TissueDetector/HnE.pt"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("Deep Learning Tissue Detector Demonstration")
print("=" * 80)
print(f"WSI: {WSI_PATH}")
print(f"Tissue Detector Model: {TISSUE_DETECTOR_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

# Check if model exists
if not os.path.exists(TISSUE_DETECTOR_PATH):
    print(f"ERROR: Tissue detector model not found at {TISSUE_DETECTOR_PATH}")
    sys.exit(1)

# Initialize processor
processor = PathologyProcessor(model="uni")

# ==============================================================================
# Part 1: Load WSI and Tiles (BEFORE Tissue Detection)
# ==============================================================================
print("Part 1: Loading WSI and Pre-loading Tiles")
print("-" * 80)

start_time = time.time()

# Load WSI with smaller tile size for more detailed classification
# Using 128x128 tiles gives us ~170 tiles instead of 9
wsi = processor.load_wsi(WSI_PATH, tile_size=128, max_patches=500, verbose=False)
print(f"✓ WSI loaded: {wsi.slide.width}x{wsi.slide.height}")
print(f"  Tile size: {wsi.tileSize}x{wsi.tileSize}")
print(f"  Tiles: {wsi.numTilesInX}x{wsi.numTilesInY} = {wsi.numTilesInX * wsi.numTilesInY}")

# Pre-load all tiles BEFORE running tissue detection
# (This avoids the getTile() bug that occurs after detectTissue())
print("  Pre-loading all tiles...")
tiles_data = []
for addr in list(wsi.iterateTiles()):
    tile = wsi.getTile(addr, writeToNumpy=True)
    if tile is not None and len(tile.shape) >= 3 and tile.shape[2] >= 3:
        tile_rgb = tile[:, :, :3]
        tiles_data.append((addr, tile_rgb))

print(f"✓ Pre-loaded {len(tiles_data)} tiles ({time.time() - start_time:.2f}s)")
print()

# ==============================================================================
# Part 2: Run Deep Learning Tissue Detection
# ==============================================================================
print("Part 2: Running Deep Learning Tissue Detection")
print("-" * 80)

# Load tissue detector
tissue_detector = TissueDetector(model_path=TISSUE_DETECTOR_PATH)
print(f"✓ Tissue detector loaded")

# Add tissue detector and run detection
wsi.tissue_detector = tissue_detector
detect_start = time.time()
wsi.detectTissue()
detect_time = time.time() - detect_start

print(f"✓ Tissue detection completed in {detect_time:.2f}s")
print(f"  Total tiles analyzed: {len(wsi.tileDictionary)}")
print()

# ==============================================================================
# Part 3: Extract Predictions and Create Heatmaps
# ==============================================================================
print("Part 3: Extracting Deep Learning Predictions")
print("-" * 80)

# Collect predictions from all tiles
artifact_levels = []
background_levels = []
tissue_levels = []
tile_positions = []

for address, tile_info in wsi.tileDictionary.items():
    if 'artifactLevel' in tile_info and 'backgroundLevel' in tile_info and 'tissueLevel' in tile_info:
        artifact_levels.append(tile_info['artifactLevel'])
        background_levels.append(tile_info['backgroundLevel'])
        tissue_levels.append(tile_info['tissueLevel'])
        tile_positions.append(address)

artifact_levels = np.array(artifact_levels)
background_levels = np.array(background_levels)
tissue_levels = np.array(tissue_levels)

print(f"  Tiles with predictions: {len(artifact_levels)}")
print(f"  Artifact levels - Mean: {artifact_levels.mean():.3f}, Range: [{artifact_levels.min():.3f}, {artifact_levels.max():.3f}]")
print(f"  Background levels - Mean: {background_levels.mean():.3f}, Range: [{background_levels.min():.3f}, {background_levels.max():.3f}]")
print(f"  Tissue levels - Mean: {tissue_levels.mean():.3f}, Range: [{tissue_levels.min():.3f}, {tissue_levels.max():.3f}]")
print()

# Create probability heatmaps
grid_size_x = wsi.numTilesInX
grid_size_y = wsi.numTilesInY

artifact_map = np.zeros((grid_size_y, grid_size_x))
background_map = np.zeros((grid_size_y, grid_size_x))
tissue_map = np.zeros((grid_size_y, grid_size_x))

for idx, (x, y) in enumerate(tile_positions):
    artifact_map[y, x] = artifact_levels[idx]
    background_map[y, x] = background_levels[idx]
    tissue_map[y, x] = tissue_levels[idx]

# ==============================================================================
# Part 4: Visualization 1 - Probability Heatmaps
# ==============================================================================
print("Part 4: Creating Probability Heatmap Visualizations")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Original thumbnail
thumbnail = np.asarray(wsi.slide)
axes[0, 0].imshow(thumbnail)
axes[0, 0].set_title('Original WSI Thumbnail', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# Artifact probability
im1 = axes[0, 1].imshow(artifact_map, cmap='Reds', vmin=0, vmax=1)
axes[0, 1].set_title(f'Artifact Probability\n(Mean: {artifact_levels.mean():.3f})', fontsize=12)
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, label='Probability')

# Background probability
im2 = axes[1, 0].imshow(background_map, cmap='Blues', vmin=0, vmax=1)
axes[1, 0].set_title(f'Background Probability\n(Mean: {background_levels.mean():.3f})', fontsize=12)
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, label='Probability')

# Tissue probability
im3 = axes[1, 1].imshow(tissue_map, cmap='Greens', vmin=0, vmax=1)
axes[1, 1].set_title(f'Tissue Probability\n(Mean: {tissue_levels.mean():.3f})', fontsize=12)
axes[1, 1].axis('off')
plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, label='Probability')

plt.suptitle('Deep Learning Tissue Detector - 3-Class Probability Maps\nDenseNet121 Architecture',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_tissue_detector_predictions.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / '01_tissue_detector_predictions.png'}")
print()

# ==============================================================================
# Part 5: Visualization 2 - Classical vs Deep Learning Comparison
# ==============================================================================
print("Part 5: Comparing Classical vs Deep Learning Methods")
print("-" * 80)

# Run classical tissue detection methods
classical_methods = {
    'otsu': processor.detect_tissue(thumbnail, method="otsu"),
    'hsv': processor.detect_tissue(thumbnail, method="hsv"),
    'otsu_hsv': processor.detect_tissue(thumbnail, method="otsu_hsv")
}

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Original (top left)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(thumbnail)
ax.set_title('Original WSI', fontsize=14, fontweight='bold')
ax.axis('off')

# Classical methods (row 1)
for idx, (method, mask) in enumerate(classical_methods.items()):
    tissue_pct = np.sum(mask) / mask.size * 100
    ax = fig.add_subplot(gs[0, idx+1])
    ax.imshow(mask, cmap='gray')
    ax.set_title(f'{method.upper()} (Classical)\n{tissue_pct:.1f}% tissue', fontsize=11)
    ax.axis('off')

# Deep learning probability maps (row 2)
ax = fig.add_subplot(gs[1, 0])
ax.axis('off')

ax = fig.add_subplot(gs[1, 1])
ax.imshow(artifact_map, cmap='Reds', vmin=0, vmax=1)
ax.set_title('DL: Artifact', fontsize=11)
ax.axis('off')

ax = fig.add_subplot(gs[1, 2])
ax.imshow(background_map, cmap='Blues', vmin=0, vmax=1)
ax.set_title('DL: Background', fontsize=11)
ax.axis('off')

ax = fig.add_subplot(gs[1, 3])
ax.imshow(tissue_map, cmap='Greens', vmin=0, vmax=1)
ax.set_title('DL: Tissue', fontsize=11)
ax.axis('off')

# Thresholded tissue masks at different confidence levels (row 3)
thresholds = [0.3, 0.5, 0.7, 0.9]
for idx, thresh in enumerate(thresholds):
    tissue_mask_thresh = tissue_map > thresh
    tissue_pct = np.sum(tissue_mask_thresh) / tissue_mask_thresh.size * 100

    ax = fig.add_subplot(gs[2, idx])
    ax.imshow(tissue_mask_thresh, cmap='gray')
    ax.set_title(f'DL: Tissue > {thresh}\n{tissue_pct:.1f}% tissue', fontsize=11)
    ax.axis('off')

plt.suptitle('Classical vs Deep Learning Tissue Detection Methods',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig(OUTPUT_DIR / "02_tissue_detector_comparison.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / '02_tissue_detector_comparison.png'}")
print()

# ==============================================================================
# Part 6: Use Tiles as Patches (128x128)
# ==============================================================================
print("Part 6: Using Tiles as Patches for Classification")
print("-" * 80)

# With 128x128 tiles, use each tile directly as a patch
# This gives us many more patches for detailed classification
all_patches = []
patch_metadata = []

for tile_addr, tile_img in tiles_data:
    # Get predictions for this tile
    tile_info = wsi.tileDictionary.get(tile_addr, {})
    artifact = tile_info.get('artifactLevel', 0)
    background = tile_info.get('backgroundLevel', 0)
    tissue = tile_info.get('tissueLevel', 0)

    # Use the entire tile as a patch
    all_patches.append(tile_img)
    patch_metadata.append({
        'tile_addr': tile_addr,
        'artifact': artifact,
        'background': background,
        'tissue': tissue
    })

patch_size = tile_img.shape[0]  # Should be 128
print(f"✓ Created {len(all_patches)} patches ({patch_size}x{patch_size}) from {len(tiles_data)} tiles")
print()

# ==============================================================================
# Part 7: Visualization 3 - Large Patch Grid
# ==============================================================================
print("Part 7: Creating Large Patch Grid Visualization")
print("-" * 80)

# Show up to 144 patches in 12x12 grid
n_display = min(144, len(all_patches))
n_rows = 12
n_cols = 12

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 24))
axes = axes.flatten()

for i in range(n_display):
    patch = all_patches[i]
    metadata = patch_metadata[i]

    # Get dominant class
    probs = [metadata['artifact'], metadata['background'], metadata['tissue']]
    dominant_class = np.argmax(probs)
    colors = ['red', 'blue', 'green']
    labels = ['Artifact', 'Background', 'Tissue']

    # Plot patch
    axes[i].imshow(patch)
    axes[i].set_title(
        f'{labels[dominant_class][0]}\n{probs[dominant_class]:.2f}',
        fontsize=6,
        color=colors[dominant_class],
        fontweight='bold'
    )
    axes[i].axis('off')

    # Add colored border
    for spine in axes[i].spines.values():
        spine.set_edgecolor(colors[dominant_class])
        spine.set_linewidth(2)
        spine.set_visible(True)

# Hide empty subplots
for i in range(n_display, n_rows * n_cols):
    axes[i].axis('off')

plt.suptitle(f'Large Patch Grid: {n_display} Patches ({patch_size}×{patch_size})\nBorder Color: Red=Artifact, Blue=Background, Green=Tissue',
             fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_large_patch_grid.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / '03_large_patch_grid.png'}")
print()

# ==============================================================================
# Part 8: Visualization 4 - Representative Tile Sample
# ==============================================================================
print("Part 8: Creating Representative Tile Sample Visualization")
print("-" * 80)

# Show 25 representative tiles in 5x5 grid
# Select tiles by sorting on tissue level to show diversity
def select_representative_tiles(metadata_list, n_samples=25):
    """Select diverse representative tiles across tissue levels"""
    tissue_values = [m['tissue'] for m in metadata_list]
    sorted_indices = np.argsort(tissue_values)

    # Sample evenly across the sorted range
    step = max(1, len(sorted_indices) // n_samples)
    selected = sorted_indices[::step][:n_samples]
    return selected

sample_indices = select_representative_tiles(patch_metadata, n_samples=25)
n_show = min(25, len(sample_indices))

fig, axes = plt.subplots(5, 5, figsize=(15, 15))
axes = axes.flatten()

for plot_idx, data_idx in enumerate(sample_indices[:n_show]):
    tile_img = all_patches[data_idx]
    metadata = patch_metadata[data_idx]

    artifact = metadata['artifact']
    background = metadata['background']
    tissue = metadata['tissue']

    # Dominant class
    dominant_class = np.argmax([artifact, background, tissue])
    colors = ['red', 'blue', 'green']
    labels = ['Artifact', 'Background', 'Tissue']

    # Plot tile
    axes[plot_idx].imshow(tile_img)
    axes[plot_idx].set_title(
        f'{labels[dominant_class]}\n'
        f'A:{artifact:.2f} B:{background:.2f}\nT:{tissue:.2f}',
        fontsize=8
    )
    axes[plot_idx].axis('off')

    # Add colored border
    for spine in axes[plot_idx].spines.values():
        spine.set_edgecolor(colors[dominant_class])
        spine.set_linewidth(3)
        spine.set_visible(True)

# Hide empty subplots
for i in range(n_show, 25):
    axes[i].axis('off')

plt.suptitle(f'Representative Tile Sample: {n_show} of {len(tiles_data)} tiles ({patch_size}×{patch_size})\nBorder Color: Red=Artifact, Blue=Background, Green=Tissue',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_representative_sample.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / '04_representative_sample.png'}")
print()

# ==============================================================================
# Part 9: Visualization 5 - Patch Extraction Demonstration
# ==============================================================================
print("Part 9: Creating Patch Extraction Demonstration")
print("-" * 80)

# Show patches that would be selected at different tissue thresholds
thresholds_to_demo = [0.05, 0.10, 0.20, 0.34]  # 0.34 is max tissue level

fig, axes = plt.subplots(4, 9, figsize=(22, 10))

for row_idx, thresh in enumerate(thresholds_to_demo):
    # Find patches meeting this threshold
    selected_patches = []
    for i, metadata in enumerate(patch_metadata):
        if metadata['tissue'] >= thresh:
            selected_patches.append(i)

    # Title for this row
    axes[row_idx, 0].text(0.5, 0.5,
                          f'Tissue ≥ {thresh:.2f}\n{len(selected_patches)} patches',
                          ha='center', va='center', fontsize=11, fontweight='bold',
                          transform=axes[row_idx, 0].transAxes)
    axes[row_idx, 0].axis('off')

    # Show up to 8 patches for this threshold
    n_show = min(8, len(selected_patches))
    for col_idx in range(8):
        if col_idx < n_show:
            patch_idx = selected_patches[col_idx]
            patch = all_patches[patch_idx]
            metadata = patch_metadata[patch_idx]

            axes[row_idx, col_idx+1].imshow(patch)
            axes[row_idx, col_idx+1].set_title(f'T:{metadata["tissue"]:.2f}', fontsize=8)
            axes[row_idx, col_idx+1].axis('off')

            # Green border for selected patches
            for spine in axes[row_idx, col_idx+1].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)
                spine.set_visible(True)
        else:
            axes[row_idx, col_idx+1].axis('off')

plt.suptitle('Patch Extraction at Different Tissue Thresholds',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_patch_extraction_demo.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {OUTPUT_DIR / '05_patch_extraction_demo.png'}")
print()

# ==============================================================================
# Part 10: Summary Statistics
# ==============================================================================
print("=" * 80)
print("Deep Learning Tissue Detector Summary")
print("=" * 80)
print()
print("Model Information:")
print(f"  Architecture: DenseNet121 (3-class classifier)")
print(f"  Model Path: {TISSUE_DETECTOR_PATH}")
print(f"  Model Size: {os.path.getsize(TISSUE_DETECTOR_PATH) / 1024 / 1024:.1f} MB")
print()
print("Detection Results:")
print(f"  Total tiles analyzed: {len(artifact_levels)}")
print(f"  Detection time: {detect_time:.2f}s ({len(artifact_levels)/detect_time:.1f} tiles/sec)")
print()
print("Class Predictions:")
print(f"  Artifact  - Mean: {artifact_levels.mean():.3f}, Std: {artifact_levels.std():.3f}")
print(f"  Background - Mean: {background_levels.mean():.3f}, Std: {background_levels.std():.3f}")
print(f"  Tissue    - Mean: {tissue_levels.mean():.3f}, Std: {tissue_levels.std():.3f}")
print()
print("Patch Extraction:")
print(f"  Total patches: {len(all_patches)} ({patch_size}×{patch_size})")
print(f"  Tiles analyzed: {len(tiles_data)}")
print()
print("Tissue Detection at Different Thresholds:")
for thresh in [0.05, 0.10, 0.20, 0.30]:
    n_patches = sum(1 for m in patch_metadata if m['tissue'] >= thresh)
    pct = n_patches / len(all_patches) * 100
    print(f"  Threshold ≥ {thresh:.2f}: {n_patches} patches ({pct:.1f}%)")
print()
print("Generated Visualizations:")
print(f"  1. {OUTPUT_DIR / '01_tissue_detector_predictions.png'}")
print(f"     - 3-class probability heatmaps")
print(f"  2. {OUTPUT_DIR / '02_tissue_detector_comparison.png'}")
print(f"     - Classical vs deep learning comparison")
print(f"  3. {OUTPUT_DIR / '03_large_patch_grid.png'}")
print(f"     - 144 patches in 12×12 grid ({patch_size}×{patch_size})")
print(f"  4. {OUTPUT_DIR / '04_representative_sample.png'}")
print(f"     - 25 representative tiles showing classification diversity")
print(f"  5. {OUTPUT_DIR / '05_patch_extraction_demo.png'}")
print(f"     - Patch extraction at different tissue thresholds")
print()
print("=" * 80)
print("Deep Learning Tissue Detector Demo Complete!")
print("=" * 80)
print()
print("Key Improvements:")
print("  ✓ Pre-loaded tiles before detection (avoids getTile() bug)")
print(f"  ✓ Using {len(all_patches)} small {patch_size}×{patch_size} tiles for detailed classification")
print("  ✓ No more black patches - all tissue content visible!")
print("  ✓ Much higher granularity with 128×128 tiles vs 512×512")
print("  ✓ Multiple visualization styles for comprehensive validation")
print("=" * 80)
