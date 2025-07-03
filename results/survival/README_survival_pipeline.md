# Survival Analysis Pipeline

This directory contains a modular pipeline for survival analysis with separate scripts for training, visualization, and reporting.

## Overview

The pipeline has been refactored to separate model training from figure generation and reporting, allowing for:
- Reusable trained models
- Faster figure regeneration without retraining
- Independent report generation
- Better reproducibility

## Scripts

### 1. `train_survival_models.py`
Trains survival models and saves them for later use.

**Usage:**
```bash
# Train all models (default)
python train_survival_models.py

# Train specific configurations
python train_survival_models.py --modalities clinical pathology --models cox rsf --projects TCGA-BLCA TCGA-KIRC

# Custom paths
python train_survival_models.py --data-path /path/to/data --output-path /path/to/output
```

**Output:**
- Individual fold models saved in `models/` directory
- Best fold models saved as `*_best.pkl`
- Training summary CSV with timestamps

### 2. `generate_survival_figures.py`
Generates risk stratification curves and summary plots from saved models.

**Usage:**
```bash
# Generate all figures
python generate_survival_figures.py

# Regenerate existing figures
python generate_survival_figures.py --regenerate

# Only generate summary plots
python generate_survival_figures.py --summary-only

# Only generate risk curves
python generate_survival_figures.py --curves-only
```

**Output:**
- Risk stratification curves in `risk_curves/` directory
- Summary plots (heatmaps, comparisons) in main directory

### 3. `generate_survival_reports.py`
Creates detailed performance reports and tables.

**Usage:**
```bash
# Generate all report formats
python generate_survival_reports.py

# Generate specific formats
python generate_survival_reports.py --format markdown csv

# Available formats: markdown, csv, latex, all
```

**Output:**
- `survival_analysis_detailed_report.md` - Comprehensive markdown report
- `survival_analysis_full_results.csv` - All metrics in CSV format
- `survival_analysis_pivot_table.csv` - Pivot table of results
- `survival_performance_ci_table.tex` - LaTeX table with confidence intervals

### 4. `test_model_pipeline.py`
Tests the model saving and loading functionality.

**Usage:**
```bash
python test_model_pipeline.py
```

## Model Storage Format

Models are saved as pickle files with the following structure:
```python
{
    'model': trained_model_object,
    'scaler': StandardScaler_object,
    'model_type': 'cox'|'rsf'|'deepsurv',
    'modality': modality_name,
    'project': project_name,
    'fold': fold_number,
    'train_c_index': float,
    'test_c_index': float,
    'n_samples_train': int,
    'n_samples_test': int,
    'n_features': int,
    'pca': PCA_object (for Cox models if used),
    'device': device_string (for DeepSurv),
    'input_dim': int (for DeepSurv)
}
```

## Workflow Example

1. **Train models** (one-time or when updating):
   ```bash
   python train_survival_models.py
   ```

2. **Generate figures** (can be run multiple times):
   ```bash
   python generate_survival_figures.py
   ```

3. **Generate reports** (can be run independently):
   ```bash
   python generate_survival_reports.py
   ```

## Directory Structure

```
survival/
├── models/                    # Saved trained models
│   ├── *_fold0.pkl           # Individual fold models
│   ├── *_fold1.pkl
│   ├── ...
│   └── *_best.pkl            # Best performing fold
├── cv_results/               # Cross-validation results
├── risk_curves/              # Risk stratification plots
│   └── TCGA-*/              # Per-project curves
├── train_survival_models.py
├── generate_survival_figures.py
├── generate_survival_reports.py
├── test_model_pipeline.py
└── survival_analysis.py     # Core analysis module
```

## Notes

- Models are automatically saved during training in the `models/` directory
- The best performing fold (highest C-index) is saved separately for easy access
- All scripts use the same data loading and preprocessing from `survival_analysis.py`
- Figures can be regenerated without retraining by using the `--regenerate` flag
- Reports aggregate results from all CV folds for comprehensive statistics