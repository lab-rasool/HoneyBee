# HoneyBee Radiology Processing - Examples

This directory contains comprehensive examples and demonstrations of the HoneyBee radiology processing API.

## Main Notebook

**[radiology_processing_examples.ipynb](radiology_processing_examples.ipynb)** - Complete interactive examples

This notebook contains:

1. **Quick Start** - Simple 6-step workflow for beginners
2. **Visual Demo** - Complete visual demonstration with outputs
3. **Clinical Workflow** - Realistic clinical radiology analysis
4. **API Tests** - Comprehensive validation of all 11 API methods

## Features Demonstrated

- Data Loading (DICOM/NIfTI)
- Hounsfield Unit Verification
- Denoising (NLM, Bilateral, Median)
- Window/Level Adjustment (Lung, Bone, Soft Tissue, Abdomen)
- Intensity Normalization (Z-Score, MinMax, Percentile)
- Spatial Resampling
- Lung Segmentation
- Multi-Organ Segmentation
- Embedding Generation (RadImageNet)
- Metal Artifact Reduction
- Clinical Report Generation

## Quick Start

```bash
# Open the notebook
jupyter notebook radiology_processing_examples.ipynb
```

Or use JupyterLab:
```bash
jupyter lab radiology_processing_examples.ipynb
```

## Test Data

- **Source:** TCGA-61-1740 CT series
- **Format:** DICOM
- **Location:** `CT/` directory

## Generated Outputs

Visualizations are saved to `demo_results/` directory:
- Lung segmentation overlays
- Denoising comparisons
- Window preset comparisons
- Normalization visualizations
- Embedding distributions
- Complete pipeline overview

## Documentation

- **Website:** https://lab-rasool.github.io/HoneyBee/
- **API Docs:** https://lab-rasool.github.io/HoneyBee/docs/radiology-processing/

## Status

✅ All API methods implemented and tested
✅ All examples working correctly
✅ Visual outputs generated successfully
