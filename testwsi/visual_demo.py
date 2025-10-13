"""
Visual Demo for PathologyProcessor

This script creates comprehensive visualizations of all PathologyProcessor
capabilities for visual confirmation and documentation.
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

# Import PathologyProcessor
from honeybee.processors import PathologyProcessor

# Configuration
WSI_PATH = Path(__file__).parent / "sample.svs"
OUTPUT_DIR = Path(__file__).parent / "visual_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("PathologyProcessor Visual Demo")
print("=" * 80)
print(f"WSI: {WSI_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

# Initialize processor
processor = PathologyProcessor(model="uni")

# Load WSI
print("Loading WSI...")
wsi = processor.load_wsi(WSI_PATH, tile_size=512, max_patches=100, verbose=False)
thumbnail = np.asarray(wsi.slide)
print(f"✓ Loaded: {thumbnail.shape}")
print()

# ==============================================================================
# 1. Tissue Detection Comparison
# ==============================================================================
print("1. Creating Tissue Detection Visualizations...")
methods = ["otsu", "hsv", "otsu_hsv"]
masks = {}

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

# Original
ax = fig.add_subplot(gs[0, 0])
ax.imshow(thumbnail)
ax.set_title('Original Image', fontsize=14, fontweight='bold')
ax.axis('off')

# Detection methods
for idx, method in enumerate(methods):
    mask = processor.detect_tissue(thumbnail, method=method)
    masks[method] = mask
    tissue_pct = np.sum(mask) / mask.size * 100

    # Mask visualization
    ax = fig.add_subplot(gs[0, idx+1])
    ax.imshow(mask, cmap='gray')
    ax.set_title(f'{method.upper()}\n{tissue_pct:.1f}% tissue', fontsize=12)
    ax.axis('off')

    # Overlay visualization
    ax = fig.add_subplot(gs[1, idx+1])
    overlay = thumbnail.copy()
    overlay[mask] = (overlay[mask] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
    ax.imshow(overlay)
    ax.set_title(f'{method.upper()} Overlay', fontsize=12)
    ax.axis('off')

plt.suptitle('Tissue Detection Methods Comparison', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(OUTPUT_DIR / "01_tissue_detection.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / '01_tissue_detection.png'}")
print()

# ==============================================================================
# 2. Stain Normalization Comparison
# ==============================================================================
print("2. Creating Stain Normalization Visualizations...")

# Get a patch for normalization
patch_address = list(wsi.iterateTiles())[min(5, len(list(wsi.iterateTiles()))-1)]
original_patch = wsi.getTile(patch_address, writeToNumpy=True)[:, :, :3]

normalization_methods = ["reinhard", "macenko", "vahadane"]
normalized_patches = {}

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Original patch (top and bottom left)
axes[0, 0].imshow(original_patch)
axes[0, 0].set_title('Original Patch', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

axes[1, 0].imshow(original_patch)
axes[1, 0].set_title('Original (Reference)', fontsize=12)
axes[1, 0].axis('off')

# Normalized patches
for idx, method in enumerate(normalization_methods):
    try:
        normalized = processor.normalize_stain(original_patch, method=method, use_target_params=True)
        normalized_patches[method] = normalized

        # Top row: Normalized result
        axes[0, idx+1].imshow(normalized)
        axes[0, idx+1].set_title(f'{method.capitalize()} Normalized', fontsize=12)
        axes[0, idx+1].axis('off')

        # Bottom row: Side-by-side comparison
        comparison = np.hstack([original_patch[:, :original_patch.shape[1]//2, :],
                               normalized[:, normalized.shape[1]//2:, :]])
        axes[1, idx+1].imshow(comparison)
        axes[1, idx+1].set_title(f'{method.capitalize()} (Left: Original, Right: Normalized)', fontsize=10)
        axes[1, idx+1].axis('off')

        print(f"  ✓ {method.capitalize()}")
    except Exception as e:
        print(f"  ✗ {method.capitalize()}: {e}")
        axes[0, idx+1].text(0.5, 0.5, f'Error:\n{method}', ha='center', va='center', transform=axes[0, idx+1].transAxes)
        axes[0, idx+1].axis('off')
        axes[1, idx+1].axis('off')

plt.suptitle('Stain Normalization Methods Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_stain_normalization.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / '02_stain_normalization.png'}")
print()

# ==============================================================================
# 3. Stain Separation (H&E Deconvolution)
# ==============================================================================
print("3. Creating Stain Separation Visualizations...")

stains = processor.separate_stains(original_patch, method="hed")

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

# Original
ax = fig.add_subplot(gs[0, 0])
ax.imshow(original_patch)
ax.set_title('Original H&E Image', fontsize=14, fontweight='bold')
ax.axis('off')

# RGB visualizations
ax = fig.add_subplot(gs[0, 1])
ax.imshow(stains['rgb_h'])
ax.set_title('Hematoxylin (RGB)', fontsize=12)
ax.axis('off')

ax = fig.add_subplot(gs[0, 2])
ax.imshow(stains['rgb_e'])
ax.set_title('Eosin (RGB)', fontsize=12)
ax.axis('off')

ax = fig.add_subplot(gs[0, 3])
ax.imshow(stains['rgb_d'])
ax.set_title('DAB/Background (RGB)', fontsize=12)
ax.axis('off')

# Intensity maps (row 2)
ax = fig.add_subplot(gs[1, 0])
ax.axis('off')  # Empty

ax = fig.add_subplot(gs[1, 1])
im1 = ax.imshow(stains['hematoxylin'], cmap='Blues')
ax.set_title('Hematoxylin Intensity', fontsize=12)
ax.axis('off')
plt.colorbar(im1, ax=ax, fraction=0.046)

ax = fig.add_subplot(gs[1, 2])
im2 = ax.imshow(stains['eosin'], cmap='Reds')
ax.set_title('Eosin Intensity', fontsize=12)
ax.axis('off')
plt.colorbar(im2, ax=ax, fraction=0.046)

ax = fig.add_subplot(gs[1, 3])
im3 = ax.imshow(stains['dab'], cmap='Greys')
ax.set_title('DAB Intensity', fontsize=12)
ax.axis('off')
plt.colorbar(im3, ax=ax, fraction=0.046)

# Statistics (row 3)
ax = fig.add_subplot(gs[2, :])
ax.axis('off')
stats_text = f"""
Stain Separation Statistics:
• Hematoxylin: Range [{stains['hematoxylin'].min():.3f}, {stains['hematoxylin'].max():.3f}], Mean {stains['hematoxylin'].mean():.3f}
• Eosin: Range [{stains['eosin'].min():.3f}, {stains['eosin'].max():.3f}], Mean {stains['eosin'].mean():.3f}
• DAB/Background: Range [{stains['dab'].min():.3f}, {stains['dab'].max():.3f}], Mean {stains['dab'].mean():.3f}

Hematoxylin stains cell nuclei (blue/purple), Eosin stains cytoplasm and extracellular matrix (pink)
"""
ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11, family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('H&E Stain Separation (Color Deconvolution)', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(OUTPUT_DIR / "03_stain_separation.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / '03_stain_separation.png'}")
print()

# ==============================================================================
# 4. Patch Extraction
# ==============================================================================
print("4. Creating Patch Extraction Visualizations...")

# Extract patches from multiple tiles
patches = []
for i, address in enumerate(list(wsi.iterateTiles())[:12]):
    patch = wsi.getTile(address, writeToNumpy=True)
    if patch is not None and patch.shape[2] >= 3:
        patch = patch[:, :, :3]
        # Resize to standard size
        from skimage.transform import resize
        patch = resize(patch, (224, 224, 3), preserve_range=True).astype(np.uint8)
        patches.append(patch)

print(f"  Extracted {len(patches)} patches")

# Create visualization
n_patches = min(len(patches), 12)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for i in range(n_patches):
    axes[i].imshow(patches[i])
    axes[i].set_title(f'Patch {i+1} (224x224)', fontsize=10)
    axes[i].axis('off')

for i in range(n_patches, 12):
    axes[i].axis('off')

plt.suptitle(f'Extracted Patches from WSI (Total: {len(patches)} patches)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_extracted_patches.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / '04_extracted_patches.png'}")
print()

# ==============================================================================
# 5. Complete Pipeline Overview
# ==============================================================================
print("5. Creating Complete Pipeline Overview...")

fig = plt.figure(figsize=(24, 14))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

# Step 1: Original WSI Thumbnail
ax = fig.add_subplot(gs[0, 0])
ax.imshow(thumbnail)
ax.set_title('1. Load WSI\n(Multi-resolution pyramid)', fontsize=12, fontweight='bold')
ax.axis('off')
ax.text(0.5, -0.1, f'Dimensions: {thumbnail.shape[1]}x{thumbnail.shape[0]}',
        ha='center', transform=ax.transAxes, fontsize=9)

# Step 2: Tissue Detection
ax = fig.add_subplot(gs[0, 1])
best_mask = masks['otsu_hsv']
ax.imshow(best_mask, cmap='gray')
ax.set_title('2. Tissue Detection\n(Classical: Otsu+HSV)', fontsize=12, fontweight='bold')
ax.axis('off')
ax.text(0.5, -0.1, f'Tissue: {np.sum(best_mask)/best_mask.size*100:.1f}%',
        ha='center', transform=ax.transAxes, fontsize=9)

# Step 3: Stain Normalization
ax = fig.add_subplot(gs[0, 2])
if 'macenko' in normalized_patches:
    ax.imshow(normalized_patches['macenko'])
else:
    ax.imshow(original_patch)
ax.set_title('3. Stain Normalization\n(Macenko method)', fontsize=12, fontweight='bold')
ax.axis('off')
ax.text(0.5, -0.1, 'Consistent color appearance',
        ha='center', transform=ax.transAxes, fontsize=9)

# Step 4: Stain Separation
ax = fig.add_subplot(gs[0, 3])
ax.imshow(stains['rgb_h'])
ax.set_title('4. Stain Separation\n(H&E deconvolution)', fontsize=12, fontweight='bold')
ax.axis('off')
ax.text(0.5, -0.1, 'Hematoxylin channel shown',
        ha='center', transform=ax.transAxes, fontsize=9)

# Step 5: Patch Extraction (show grid)
ax = fig.add_subplot(gs[1, :2])
if len(patches) >= 6:
    patch_grid = np.hstack([patches[i] for i in range(6)])
    ax.imshow(patch_grid)
else:
    ax.imshow(patches[0] if patches else original_patch)
ax.set_title('5. Extract Tissue Patches\n(224x224 patches from tissue regions)',
             fontsize=12, fontweight='bold')
ax.axis('off')

# Step 6: Embedding Generation (conceptual)
ax = fig.add_subplot(gs[1, 2:])
ax.axis('off')
ax.text(0.5, 0.7, '6. Generate Embeddings', ha='center', va='center',
        fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.5, 'Foundation Models:', ha='center', va='center',
        fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.35, '• UNI (ViT-L): Universal pathology encoder',
        ha='center', va='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.25, '• UNI2 (ViT-H): Enhanced UNI model',
        ha='center', va='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.15, '• Virchow2 (DINOv2): 3.1M slides training',
        ha='center', va='center', fontsize=10, transform=ax.transAxes)
ax.text(0.5, 0.05, '• REMEDIS: Medical image embeddings',
        ha='center', va='center', fontsize=10, transform=ax.transAxes)
ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                           edgecolor='navy', linewidth=2, transform=ax.transAxes))

# Step 7: Aggregation (conceptual)
ax = fig.add_subplot(gs[2, :])
ax.axis('off')
ax.text(0.5, 0.7, '7. Aggregate to Slide-Level Representation',
        ha='center', va='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.45, 'Aggregation Methods: mean, max, median, std, concat',
        ha='center', va='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.25, 'Output: Single embedding vector representing entire slide',
        ha='center', va='center', fontsize=11, transform=ax.transAxes)
ax.text(0.5, 0.05, '→ Ready for downstream tasks: Classification, Survival Analysis, Retrieval',
        ha='center', va='center', fontsize=10, style='italic', transform=ax.transAxes)
ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                           edgecolor='darkgreen', linewidth=2, transform=ax.transAxes))

plt.suptitle('PathologyProcessor Complete Pipeline', fontsize=18, fontweight='bold', y=0.98)
plt.savefig(OUTPUT_DIR / "05_complete_pipeline.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR / '05_complete_pipeline.png'}")
print()

# ==============================================================================
# Summary
# ==============================================================================
print("=" * 80)
print("Visual Demo Complete!")
print("=" * 80)
print(f"All visualizations saved to: {OUTPUT_DIR}")
print()
print("Generated files:")
print("  1. 01_tissue_detection.png - Tissue detection methods comparison")
print("  2. 02_stain_normalization.png - Stain normalization comparison")
print("  3. 03_stain_separation.png - H&E stain separation")
print("  4. 04_extracted_patches.png - Extracted tissue patches")
print("  5. 05_complete_pipeline.png - Complete pipeline overview")
print()
print("Open these files to visually confirm PathologyProcessor capabilities!")
print("=" * 80)
