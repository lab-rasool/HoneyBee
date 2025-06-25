# Cancer Type Retrieval Benchmarking

This directory contains the retrieval benchmarking code and results for cancer type similarity search using embeddings from multiple modalities.

## Overview

The retrieval system performs patient-to-patient similarity search based on embedding distances. Given a query patient, it retrieves the k most similar patients and evaluates how many share the same cancer type (Precision@k).

## Key Results

Based on the experiments run on 2025-06-24:

| Modality | Precision@5 | Precision@10 | Precision@20 |
|----------|-------------|--------------|--------------|
| Clinical | 0.8669 | 0.8263 | 0.7710 |
| Pathology | 0.4950 | 0.4295 | 0.3564 |
| Multimodal | 0.6562 | 0.5802 | 0.4901 |

### Key Findings:
1. **Clinical embeddings** consistently perform best across all k values
2. **Pathology embeddings** show significantly lower retrieval performance
3. **Multimodal embeddings** perform between clinical and pathology, suggesting pathology adds noise
4. Performance drops as k increases (expected behavior)

## Directory Structure

```
retrieval/
├── retrieval_benchmark.py    # Main retrieval experiment script
├── utils/                    # Utility modules
│   ├── data_loader.py       # Data loading functions
│   └── __init__.py
├── retrieval_results/       # Output directory
│   ├── figures/            # Visualizations
│   │   ├── retrieval_precision_at_k.*    # Precision curves
│   │   ├── retrieval_heatmap.*           # Performance heatmap
│   │   └── precision_at_*_bar.*          # Bar charts for specific k values
│   ├── results/            # Numerical results
│   │   └── retrieval_results.json
│   └── reports/            # Text reports
│       └── retrieval_summary.txt
└── README.md               # This file
```

## Usage

To run the retrieval experiments:

```bash
cd /mnt/f/Projects/HoneyBee/results/retrieval
python retrieval_benchmark.py
```

The script will:
1. Load pre-computed embeddings from the old results directory
2. Perform patient-to-patient similarity search using cosine similarity
3. Evaluate Precision@k for k=1 to 50
4. Generate visualizations and reports

## Configuration

Key parameters in `retrieval_benchmark.py`:
- `N_RUNS = 3`: Number of runs with random shuffling
- `K_VALUES = list(range(1, 51))`: k values to evaluate
- Cosine similarity is used as the distance metric

## Visualizations

The system generates several visualizations:
1. **Precision@k Curves**: Shows how precision changes with k for each modality
2. **Performance Heatmap**: Matrix view of precision values at key k values
3. **Bar Charts**: Comparison of modalities at specific k values (5, 10, 20)

## Methodology

1. **Normalization**: All embeddings are L2-normalized for cosine similarity
2. **Self-exclusion**: Each patient is excluded from being their own nearest neighbor
3. **Evaluation**: Precision@k = (# of retrieved patients with same cancer type) / k
4. **Averaging**: Results are averaged across multiple runs with random shuffling

## Performance Analysis

- **Clinical superiority**: Clinical embeddings capture cancer type information most effectively
- **Performance degradation**: All modalities show expected drop in precision as k increases
- **Multimodal compromise**: Combining modalities doesn't improve over clinical alone

This suggests that clinical text contains the most discriminative features for cancer type identification in the retrieval task.