# Cancer Classification Results

This directory contains the cancer classification code and results using embeddings from multiple modalities.

## Overview

The classification system performs cancer type classification using Random Forest classifiers on embeddings from different modalities:
- Clinical text embeddings (from Gatortron)
- Pathology report embeddings 
- Multimodal embeddings (concatenation of available modalities)

## Key Results

Based on the experiments run on 2025-06-24:

| Modality | Accuracy | F1-Score | Classes |
|----------|----------|----------|---------|
| Clinical | 90.21% (±0.53%) | 90.01% (±0.54%) | 33 |
| Pathology | 49.16% (±0.79%) | 46.50% (±0.78%) | 32 |
| Multimodal | 88.94% (±0.61%) | 88.68% (±0.64%) | 32 |

### Key Findings:
1. **Clinical embeddings** perform best with over 90% accuracy
2. **Pathology embeddings** show lower performance, likely due to data quality or embedding representation issues
3. **Multimodal embeddings** perform well but slightly below clinical-only, suggesting that pathology may be adding noise

## Directory Structure

```
classification/
├── cancer_classification.py    # Main classification script
├── utils/                      # Utility modules
│   ├── data_loader.py         # Data loading functions
│   └── __init__.py
├── classification_results/     # Output directory
│   ├── figures/               # Visualizations
│   │   ├── *_confusion_matrix.*  # Confusion matrices
│   │   ├── *_tsne.*              # t-SNE visualizations
│   │   └── performance_comparison.*  # Performance comparison charts
│   ├── results/               # Numerical results
│   │   ├── classification_results.json
│   │   ├── confusion_matrices.json
│   │   └── *_tsne_coords.csv
│   └── reports/               # Text reports
│       └── classification_summary.txt
└── README.md                  # This file
```

## Usage

To run the classification experiments:

```bash
cd /mnt/f/Projects/HoneyBee/results/classification
python cancer_classification.py
```

The script will:
1. Load pre-computed embeddings from the old results directory
2. Run 10 classification experiments with different random seeds
3. Generate confusion matrices and t-SNE visualizations
4. Create performance comparison charts
5. Save detailed results and reports

## Configuration

Key parameters in `cancer_classification.py`:
- `N_RUNS = 10`: Number of runs for each modality
- `TEST_SIZE = 0.2`: Train-test split ratio
- `N_ESTIMATORS = 100`: Number of trees in Random Forest

## Visualizations

The system generates several visualizations:
1. **Confusion Matrices**: Show prediction accuracy for each cancer type
2. **t-SNE Plots**: 2D visualization of embedding space colored by cancer type
3. **Performance Comparison**: Bar chart comparing metrics across modalities

## Notes

- The embeddings are loaded from pre-computed pickle files in the old results directory
- Each run uses a different random seed for train-test splitting
- Results show mean and standard deviation across all runs
- The multimodal embeddings are created by concatenating embeddings from patients who have data in multiple modalities