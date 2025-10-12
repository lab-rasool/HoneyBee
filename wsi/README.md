# WSI Patch Embedding Visualization

This directory contains scripts for generating patch-wise embedding visualizations of Whole Slide Images (WSI). The visualization colors each tissue patch based on its embedding vector projected to 3D RGB space using UMAP.

## Overview

The scripts create a unique visualization where:
- Each tissue patch from the WSI is processed through the UNI model to generate a 1024-dimensional embedding
- UMAP reduces these embeddings to 3D coordinates
- The 3D coordinates are mapped to RGB colors
- Similar tissue patterns appear in similar colors

## Scripts

### 1. `wsi_patch_embedding_viz.py`
Full-featured script with all options and configurations.

```bash
python wsi_patch_embedding_viz.py \
    --wsi_path /path/to/slide.svs \
    --uni_model_path /path/to/uni.pt \
    --tile_size 512 \
    --max_patches 2000 \
    --output_dir ./output
```

### 2. `simple_patch_viz.py`
Simplified version for quick testing and debugging.

```bash
python simple_patch_viz.py
```

## Requirements

### Dependencies
- HoneyBee framework (parent directory)
- PyTorch with CUDA support (recommended)
- UMAP for dimensionality reduction: `pip install umap-learn`
- cuCIM for fast WSI loading (via HoneyBee)

### Models
- **UNI Model**: Download from [HuggingFace](https://huggingface.co/MahmoodLab/UNI)
- **Tissue Detector**: Will auto-download or use HoneyBee's default

## Output

The scripts generate:
1. **Visualization Image** - 4-panel figure showing:
   - Original WSI
   - Patch colors based on UMAP coordinates
   - Blended overlay
   - 3D embedding space scatter plot

2. **Data Files**:
   - `patch_embeddings.npy` - Raw 1024D embeddings
   - `umap_embeddings.npy` - 3D UMAP coordinates
   - `patch_colors.npy` - RGB colors for each patch
   - `patch_coordinates.npy` - Patch locations in WSI

## Example Usage

### Basic Usage
```python
from simple_patch_viz import create_patch_embedding_viz

create_patch_embedding_viz(
    wsi_path="path/to/slide.svs",
    uni_model_path="path/to/uni.pt",
    max_patches=100
)
```

### Advanced Usage
```python
from wsi_patch_embedding_viz import WSIPatchEmbeddingVisualizer

visualizer = WSIPatchEmbeddingVisualizer(
    wsi_path="slide.svs",
    uni_model_path="uni.pt",
    tile_size=512,
    max_patches=1000,
    batch_size=32
)

visualizer.run()
```

## Customization

### Adjusting UMAP Parameters
In the scripts, modify the UMAP initialization:
```python
reducer = umap.UMAP(
    n_components=3,      # Keep at 3 for RGB
    n_neighbors=15,      # Increase for more global structure
    min_dist=0.1,        # Decrease for tighter clusters
    metric='cosine'      # Try 'euclidean' for different results
)
```

### Using Different Models
Replace the UNI model with UNI2 or Virchow2:
```python
# For UNI2
from honeybee.models.UNI2.uni2 import UNI2
model = UNI2(model_path="path/to/uni2.bin")

# For Virchow2
from honeybee.models.Virchow2.virchow2 import Virchow2
model = Virchow2()  # Auto-downloads from HuggingFace
```

### Memory Management
For large WSIs or limited GPU memory:
- Reduce `max_patches` to process fewer patches
- Decrease `batch_size` for embedding generation
- Use `tile_size=256` for smaller patches

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size
- Process fewer patches
- Use CPU mode by setting device='cpu'

### No Tissue Detected
- Check tissue_threshold (default 0.8)
- Verify tissue detector model path
- Try without tissue detection (will process all patches)

### UMAP Not Installed
The script will fall back to PCA if UMAP is not available:
```bash
pip install umap-learn
```

## Interpretation

The resulting visualization helps identify:
- **Tissue heterogeneity**: Different colors indicate different tissue patterns
- **Similar regions**: Patches with similar colors have similar features
- **Anomalies**: Unique colors may indicate unusual tissue patterns
- **Batch effects**: Check if colors correlate with scanning artifacts

## Performance

Typical processing times:
- 100 patches: ~30 seconds
- 1000 patches: ~5 minutes
- 5000 patches: ~20 minutes

(Times vary based on GPU, batch size, and WSI size)