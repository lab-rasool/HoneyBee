# Retrieval Analysis

This directory contains the unified retrieval analysis for HoneyBee embeddings.

## Quick Start

Run all retrieval analyses with a single command:

```bash
python run_all_retrieval_analyses.py
```

## What it does

The unified script performs comprehensive retrieval evaluation:

1. **Basic Retrieval** - Precision@k metrics with multiple runs
2. **Stable Analysis** - Bootstrap confidence intervals for reliability
3. **AMI Analysis** - Clustering quality assessment
4. **Failure Analysis** - Identifies commonly confused cancer types
5. **Visualizations** - Comprehensive plots and comparisons

## Output

All results are saved in a single directory: `unified_retrieval_results/`

- `data/` - JSON files with all numerical results
- `figures/` - Visualization plots
- `reports/` - Human-readable summary reports

## Modalities Analyzed

- Clinical (text embeddings)
- Pathology (report embeddings)
- Radiology (image embeddings)
- Molecular (genomic embeddings)

## Archived Scripts

Previous individual analysis scripts have been moved to `archived_scripts/` for reference.
The unified script combines all their functionality.