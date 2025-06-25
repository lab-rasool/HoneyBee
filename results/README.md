# HoneyBee Results Directory

This directory contains organized analysis results for the HoneyBee multimodal oncology AI framework.

## Directory Structure

```
results/
├── shared_data/          # Shared embeddings and data files
│   └── embeddings/       # Pre-computed embeddings for all modalities
├── survival/             # Disease-specific survival analysis
│   ├── models/           # Survival model implementations
│   ├── utils/            # Utility functions
│   └── configs/          # Configuration files
├── classification/       # Cancer type classification
│   ├── utils/            # Data loading utilities
│   └── classification_results/  # Results and visualizations
├── retrieval/            # Patient similarity retrieval
│   ├── utils/            # Data loading utilities
│   └── retrieval_results/       # Results and visualizations
└── old/                  # Legacy code (can be removed)
```

## Analysis Modules

### 1. Survival Analysis (`survival/`)
- **Purpose**: Train Cox, Random Survival Forest, and DeepSurv models by disease type
- **Key Script**: `train_by_disease.py`
- **Output**: Disease-specific survival predictions and model comparisons

### 2. Classification (`classification/`)
- **Purpose**: Classify cancer types using Random Forest on multimodal embeddings
- **Key Script**: `cancer_classification.py`
- **Output**: Confusion matrices, t-SNE visualizations, performance metrics

### 3. Retrieval (`retrieval/`)
- **Purpose**: Patient-to-patient similarity search based on cancer type
- **Key Script**: `retrieval_benchmark.py`
- **Output**: Precision@k curves, performance heatmaps

## Shared Data

All analysis modules use the `shared_data/` directory for embeddings, ensuring:
- No dependency on the old directory structure
- Consistent data across all analyses
- Easy portability of results

## Running Analyses

Each module can be run independently:

```bash
# Survival analysis
cd survival
python train_by_disease.py

# Classification
cd classification
python cancer_classification.py

# Retrieval
cd retrieval
python retrieval_benchmark.py
```

## Key Results Summary

### Classification Performance
- Clinical: 90.21% accuracy
- Pathology: 49.16% accuracy
- Multimodal: 88.94% accuracy

### Retrieval Performance (Precision@10)
- Clinical: 82.63%
- Pathology: 42.95%
- Multimodal: 58.02%

### Survival Analysis
- Best performing: TCGA-LIHC with pathology DeepSurv (C-index: 0.8056)
- Clinical embeddings show consistent performance across diseases

## Notes

1. The `old/` directory is no longer needed and can be safely removed
2. All paths are relative to ensure portability
3. Each analysis module has its own README with detailed information
4. Results are saved in module-specific subdirectories