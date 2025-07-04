# HoneyBee WSI Processing Examples

This directory contains comprehensive examples demonstrating the Whole Slide Image (WSI) processing capabilities of HoneyBee. These examples showcase various techniques for loading, processing, and analyzing digital pathology images.

## Overview

HoneyBee provides state-of-the-art WSI processing capabilities including:
- **GPU-accelerated I/O** through CuCIM and NVIDIA GPUDirect Storage
- **Multi-resolution pyramid** support for efficient processing
- **Tissue detection** using both classical and deep learning methods
- **Stain normalization** with multiple algorithms (Reinhard, Macenko, Vahadane)
- **Stain separation** for quantitative analysis
- **Embedding generation** using pre-trained pathology models (UNI, UNI2, Virchow2)
- **Batch processing** capabilities for large cohorts

## Prerequisites

### Required Dependencies
```bash
pip install cucim
pip install torch torchvision
pip install scikit-image scipy
pip install matplotlib seaborn
pip install numpy pandas
pip install albumentations
```

### Optional Dependencies
```bash
pip install cupy-cuda11x  # For GPU-accelerated stain normalization
pip install opencv-python  # For additional image processing
```

### Required Files
- Sample WSI: `/mnt/f/Projects/HoneyBee/examples/samples/sample.svs`
- Tissue detector model: `/mnt/d/Models/TissueDetector/HnE.pt`
- Embedding models (optional): UNI, UNI2, Virchow2 weights

## Examples

### 01. WSI Loading Basics
**File**: `01_wsi_loading_basics.py`

Learn the fundamentals of loading and exploring WSI files:
- Loading WSI with CuCIM backend
- Exploring metadata and properties
- Navigating multi-resolution pyramids
- Memory-efficient region reading
- Thumbnail generation

```bash
python 01_wsi_loading_basics.py
```

**Key concepts**:
- Understanding WSI file structure
- Resolution levels and downsampling
- Memory management strategies
- Vendor-specific metadata extraction

### 02. Tissue Detection
**File**: `02_tissue_detection.py`

Compare different methods for detecting tissue regions:
- Classical Otsu-based detection
- Deep learning tissue detection
- Performance comparison
- Tissue mask visualization

```bash
python 02_tissue_detection.py
```

**Key concepts**:
- Gradient-based tissue detection
- Deep learning classification (tissue/background/artifact)
- IoU and Dice coefficient metrics
- Patch-based processing

### 03. Stain Normalization
**File**: `03_stain_normalization.py`

Normalize stain variations across WSI images:
- Reinhard color normalization
- Macenko stain normalization
- Vahadane stain normalization
- TCGA average targets
- Batch processing performance

```bash
python 03_stain_normalization.py
```

**Key concepts**:
- Color space transformations
- Statistical matching techniques
- Stain matrix estimation
- Performance optimization

### 04. Stain Separation
**File**: `04_stain_separation.py`

Separate and analyze individual stain components:
- H&E stain separation
- HED color space conversion
- Channel visualization
- Stain reconstruction
- Quantitative analysis

```bash
python 04_stain_separation.py
```

**Key concepts**:
- Color deconvolution
- Beer-Lambert law
- Stain intensity modification
- Reconstruction quality metrics

### 05. Patch Extraction (Planned)
**File**: `05_patch_extraction.py`

Extract patches for downstream analysis:
- Grid-based extraction
- Tissue-aware filtering
- Overlap handling
- Coordinate tracking
- Memory-efficient loading

### 06. Embedding Generation (Planned)
**File**: `06_embedding_generation.py`

Generate embeddings using pre-trained models:
- UNI model integration
- UNI2 and Virchow2 examples
- Batch processing
- Feature aggregation
- GPU memory management

### 07. Complete Pipeline (Planned)
**File**: `07_complete_pipeline.py`

End-to-end WSI processing pipeline:
- Complete workflow integration
- Configuration management
- Result saving and loading
- Performance optimization

### 08. Batch Processing (Planned)
**File**: `08_batch_processing.py`

Process multiple WSIs efficiently:
- Parallel processing strategies
- Progress tracking
- Error handling
- Result aggregation

## Utility Modules

### utils.py
Common utility functions including:
- Configuration management
- Timing decorators
- Path validation
- Progress tracking
- Result saving

### visualizations.py
Comprehensive visualization functions:
- WSI overview plots
- Tissue detection comparisons
- Stain normalization results
- Patch grid visualization
- Processing timelines

## Performance Tips

1. **GPU Acceleration**
   - Ensure CUDA is available for CuCIM
   - Use GPU memory efficiently
   - Monitor memory usage with `nvidia-smi`

2. **Memory Management**
   - Use appropriate pyramid levels
   - Process in patches/tiles
   - Clear GPU cache regularly

3. **Batch Processing**
   - Use concurrent loading
   - Optimize batch sizes
   - Implement proper error handling

## Output Structure

Each example creates timestamped output directories:
```
tmp/
├── 01_wsi_loading_YYYYMMDD_HHMMSS/
│   ├── metadata.json
│   ├── config.json
│   └── results_summary.txt
├── 02_tissue_detection_YYYYMMDD_HHMMSS/
│   ├── classical_mask.npz
│   ├── deep_mask.npz
│   └── tissue_detection_comparison.png
└── ...
```

## Troubleshooting

### Common Issues

1. **CUDA/CuCIM not available**
   ```
   Error: CuCIM requires CUDA
   Solution: Install CUDA toolkit and cuCIM with GPU support
   ```

2. **Memory errors**
   ```
   Error: CUDA out of memory
   Solution: Reduce batch size or use lower resolution levels
   ```

3. **Missing models**
   ```
   Error: Tissue detector model not found
   Solution: Download pre-trained models to specified paths
   ```

4. **Tissue detection sensitivity**
   ```
   Issue: Deep learning model classifies most tiles as artifacts
   Solution: Adjust tissue_threshold parameter (default 0.5, try 0.3)
   Note: Different models/datasets may require different thresholds
   ```

### Debug Mode
Enable verbose output in examples:
```python
slide = Slide(wsi_path, verbose=True)
```

## Advanced Usage

### Custom Stain Matrices
```python
custom_matrix = np.array([
    [0.650, 0.704, 0.286],  # Custom H
    [0.268, 0.570, 0.776],  # Custom E
    [0.0, 0.0, 0.0]
])
```

### GPU Memory Optimization
```python
import torch
torch.cuda.empty_cache()  # Clear cache
torch.cuda.set_per_process_memory_fraction(0.8)  # Limit GPU usage
```

### Custom Tissue Detection
```python
def custom_tissue_detector(patch):
    # Your custom detection logic
    return tissue_mask
```

## Contributing

To add new examples:
1. Follow the existing naming convention
2. Include comprehensive docstrings
3. Use utility functions for common operations
4. Add visualization of results
5. Update this README

## License

These examples are part of the HoneyBee project and follow the same license terms.

## References

- CuCIM Documentation: https://docs.rapids.ai/api/cucim/
- scikit-image: https://scikit-image.org/
- PyTorch: https://pytorch.org/

## Contact

For questions or issues with these examples, please open an issue in the HoneyBee repository.