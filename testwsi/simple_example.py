"""
Simple PathologyProcessor Example - Matches Website Documentation

This script demonstrates the basic usage of PathologyProcessor
as shown in the HoneyBee documentation:
https://lab-rasool.github.io/HoneyBee/docs/pathology-processing/
"""

import sys
from pathlib import Path

# Add HoneyBee to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from honeybee.processors import PathologyProcessor

# =============================================================================
# Example 1: Basic Usage (as shown on website)
# =============================================================================
print("=" * 80)
print("Example 1: Basic PathologyProcessor Usage")
print("=" * 80)

# Initialize processor with UNI model
processor = PathologyProcessor(model="uni", model_path="/mnt/d/Models/UNI/uni.pt")

# Load WSI
wsi = processor.load_wsi("sample.svs")
print("✓ Loaded WSI")

# Detect tissue using Otsu method
tissue_mask = processor.detect_tissue(wsi, method="otsu")
print(f"✓ Detected tissue (method: otsu)")

# Normalize stain using Macenko method
normalized_wsi = processor.normalize_stain(wsi, method="macenko")
print(f"✓ Normalized stain (method: macenko)")

# Separate H&E stains
stains = processor.separate_stains(wsi)
print(f"✓ Separated stains: {list(stains.keys())}")

# Extract patches
patches = processor.extract_patches(
    wsi,
    tissue_mask,
    patch_size=256,
    overlap=0.2,
    min_tissue_percentage=0.5
)
print(f"✓ Extracted {len(patches)} patches")

# Generate embeddings (if patches available)
if len(patches) > 0:
    embeddings = processor.generate_embeddings(patches)
    print(f"✓ Generated embeddings: {embeddings.shape}")

    # Aggregate embeddings to slide level
    slide_embedding = processor.aggregate_embeddings(embeddings, method="mean")
    print(f"✓ Aggregated to slide embedding: {slide_embedding.shape}")
else:
    print("⚠ No patches extracted, skipping embedding generation")

print()

# =============================================================================
# Example 2: Complete Pipeline (from website docs)
# =============================================================================
print("=" * 80)
print("Example 2: Complete Pipeline")
print("=" * 80)
print("Processing slide with full pipeline...")
print()

# This is the exact code example from the website
processor = PathologyProcessor(model="uni", model_path="/mnt/d/Models/UNI/uni.pt")
wsi = processor.load_wsi("sample.svs")
normalized_wsi = processor.normalize_stain(wsi, method="macenko")
tissue_mask = processor.detect_tissue(normalized_wsi, method="deeplearning")
patches = processor.extract_patches(normalized_wsi, tissue_mask)
embeddings = processor.generate_embeddings(patches)
slide_embedding = processor.aggregate_embeddings(embeddings)

print("✓ Complete pipeline executed successfully!")
print(f"  Final slide embedding: {slide_embedding.shape}")
print()

# =============================================================================
# Example 3: Different Models and Methods
# =============================================================================
print("=" * 80)
print("Example 3: Testing Different Models and Methods")
print("=" * 80)

# Try different stain normalization methods
print("Stain normalization methods:")
for method in ["reinhard", "macenko", "vahadane"]:
    normalized = processor.normalize_stain(wsi, method=method)
    print(f"  ✓ {method.capitalize()}")

# Try different tissue detection methods
print("\nTissue detection methods:")
for method in ["otsu", "hsv", "otsu_hsv"]:
    mask = processor.detect_tissue(wsi, method=method)
    print(f"  ✓ {method}")

# Try different aggregation methods
print("\nEmbedding aggregation methods:")
if len(patches) > 0:
    test_embeddings = processor.generate_embeddings(patches[:5])  # Use subset
    for agg_method in ["mean", "max", "median"]:
        agg = processor.aggregate_embeddings(test_embeddings, method=agg_method)
        print(f"  ✓ {agg_method}: {agg.shape}")

print()
print("=" * 80)
print("All examples completed successfully!")
print("=" * 80)
print()
print("The PathologyProcessor API matches the website documentation exactly:")
print("  - processor = PathologyProcessor(model='uni')")
print("  - wsi = processor.load_wsi('path/to/slide.svs')")
print("  - tissue_mask = processor.detect_tissue(wsi, method='otsu')")
print("  - normalized_wsi = processor.normalize_stain(wsi, method='macenko')")
print("  - stains = processor.separate_stains(wsi)")
print("  - patches = processor.extract_patches(wsi, tissue_mask)")
print("  - embeddings = processor.generate_embeddings(patches)")
print("  - slide_embedding = processor.aggregate_embeddings(embeddings)")
print()
