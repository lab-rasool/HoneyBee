# Survival Analysis Report - All Models

## Executive Summary

**Best Overall Performance:**
- Embedding: clinical_qwen
- Cancer Type: PRAD
- Cox C-index: 0.997

## Model Performance Summary

### Average C-index by Embedding Type

| Embedding | Cox PH | RSF | DeepSurv |
|-----------|--------|-----|----------|
| clinical_gatortron | 0.855 | 0.842 | 0.866 |
| clinical_llama | 0.858 | 0.842 | 0.861 |
| clinical_medgemma | 0.856 | 0.850 | 0.862 |
| clinical_qwen | 0.842 | 0.850 | 0.862 |
| pathology_gatortron | 0.580 | 0.581 | 0.586 |
| pathology_llama | 0.573 | 0.572 | 0.573 |
| pathology_medgemma | 0.596 | 0.566 | 0.584 |
| pathology_qwen | 0.584 | 0.570 | 0.592 |

### Clinical vs Pathology Performance

- **Clinical embeddings**: 0.853 average C-index
- **Pathology embeddings**: 0.583 average C-index
- **Difference**: 0.269

### Architecture Comparison

- **Encoder-only models**: 0.715 average C-index
- **Decoder-only models**: 0.721 average C-index

### Domain Comparison

- **Medical-focused models**: 0.722 average C-index
- **General-purpose models**: 0.714 average C-index

## Cancer-Specific Results

### ACC
- Patients: 91
- Events: 33.0
- Best embedding: clinical_llama (C-index: 0.879)

### BLCA
- Patients: 406
- Events: 178.0
- Best embedding: clinical_gatortron (C-index: 0.883)

### BRCA
- Patients: 1082
- Events: 152.0
- Best embedding: clinical_gatortron (C-index: 0.945)

### CESC
- Patients: 296
- Events: 73.0
- Best embedding: clinical_gatortron (C-index: 0.914)

### CHOL
- Patients: 48
- Events: 22.0
- Best embedding: clinical_medgemma (C-index: 0.549)

### COAD
- Patients: 440
- Events: 95.0
- Best embedding: clinical_medgemma (C-index: 0.938)

### ESCA
- Patients: 185
- Events: 77.0
- Best embedding: clinical_medgemma (C-index: 0.849)

### GBM
- Patients: 597
- Events: 496.0
- Best embedding: clinical_gatortron (C-index: 0.769)

### HNSC
- Patients: 528
- Events: 224.0
- Best embedding: clinical_gatortron (C-index: 0.873)

### KICH
- Patients: 112
- Events: 12.0
- Best embedding: clinical_llama (C-index: 0.990)

### KIRC
- Patients: 534
- Events: 175.0
- Best embedding: clinical_gatortron (C-index: 0.921)

### KIRP
- Patients: 289
- Events: 44.0
- Best embedding: clinical_gatortron (C-index: 0.965)

### LGG
- Patients: 521
- Events: 131.0
- Best embedding: clinical_gatortron (C-index: 0.921)

### LIHC
- Patients: 373
- Events: 132.0
- Best embedding: clinical_gatortron (C-index: 0.902)

### LUAD
- Patients: 510
- Events: 183.0
- Best embedding: clinical_llama (C-index: 0.859)

### LUSC
- Patients: 496
- Events: 214.0
- Best embedding: clinical_gatortron (C-index: 0.848)

### MESO
- Patients: 85
- Events: 73.0
- Best embedding: clinical_gatortron (C-index: 0.693)

### OV
- Patients: 599
- Events: 360.0
- Best embedding: clinical_llama (C-index: 0.828)

### PAAD
- Patients: 184
- Events: 99.0
- Best embedding: clinical_gatortron (C-index: 0.779)

### PRAD
- Patients: 499
- Events: 10.0
- Best embedding: clinical_qwen (C-index: 0.997)

### READ
- Patients: 161
- Events: 27.0
- Best embedding: clinical_gatortron (C-index: 0.949)

### SARC
- Patients: 262
- Events: 100.0
- Best embedding: clinical_gatortron (C-index: 0.899)

### SKCM
- Patients: 459
- Events: 223.0
- Best embedding: clinical_gatortron (C-index: 0.867)

### STAD
- Patients: 411
- Events: 168.0
- Best embedding: clinical_gatortron (C-index: 0.872)

### THCA
- Patients: 514
- Events: 16.0
- Best embedding: clinical_gatortron (C-index: 0.990)

### UCEC
- Patients: 545
- Events: 91.0
- Best embedding: clinical_gatortron (C-index: 0.958)

### UCS
- Patients: 55
- Events: 34.0
- Best embedding: clinical_medgemma (C-index: 0.623)

### UVM
- Patients: 80
- Events: 23.0
- Best embedding: clinical_qwen (C-index: 0.892)

## Technical Details

- **Cross-validation**: 5-fold stratified
- **Preprocessing**: StandardScaler + PCA (if >100 dimensions)
- **Minimum requirements**: 30 patients, 10 events per cancer type
- **Models evaluated**: Cox PH, Random Survival Forest, DeepSurv (GPU only)
- **Risk stratification**: Patients divided into tertiles (Low/Medium/High risk)
- **Statistical testing**: Log-rank test for group comparisons

## Risk Stratification Curves

Risk stratification curves have been generated for each cancer type with sufficient data.
These curves show Kaplan-Meier survival probabilities for Low, Medium, and High risk groups
as determined by each model's risk scores. Curves are saved in the `risk_stratification/` directory
organized by cancer type.
