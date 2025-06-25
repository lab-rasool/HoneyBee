# t-SNE Visualizations for HoneyBee Multimodal Embeddings

This directory contains t-SNE visualizations for both individual modalities and multimodal fusion embeddings from the HoneyBee project.

## Usage

Run the unified visualization script:

```bash
# Create all visualizations (default)
python create_tsne_visualizations.py

# Create only individual modality visualizations
python create_tsne_visualizations.py --individual

# Create only multimodal visualizations
python create_tsne_visualizations.py --multimodal

# Specify custom output directory
python create_tsne_visualizations.py --output-dir /path/to/output
```

## Output Structure

The script generates PDF files with separate legend files for better readability:

```
stratified_tsne_visualizations/
├── clinical/                    # Individual modality results
│   ├── clinical_tsne_by_cancer_type_plot.pdf
│   ├── clinical_tsne_by_cancer_type_legend.pdf
│   ├── clinical_tsne_by_sex_plot.pdf
│   ├── clinical_tsne_by_age_group_plot.pdf
│   └── clinical_tsne_by_organ_site_plot.pdf
├── molecular/                   # Similar structure for each modality
├── pathology/
├── wsi/
├── radiology/
├── multimodal_concat/          # Multimodal fusion results
│   ├── concat_tsne_by_cancer_type_plot.pdf
│   ├── concat_tsne_by_cancer_type_legend.pdf
│   ├── concat_tsne_by_modality_combo_plot.pdf
│   ├── concat_tsne_by_modality_combo_legend.pdf
│   └── concat_modality_combination_stats.csv
├── multimodal_mean_pool/       # Similar for mean pooling
└── multimodal_kronecker/       # Similar for Kronecker product
```

## Features

- **Individual Modalities**: Visualizations for clinical, molecular, pathology, WSI, and radiology embeddings
- **Multimodal Fusion**: Three fusion methods (concatenation, mean pooling, Kronecker product)
- **Stratifications**: By cancer type, sex, age group, organ site, and modality combinations
- **Separate Legends**: Created for all visualizations for better readability
- **Visual Differentiation**:
  - TCGA cancer types: Custom color scheme grouping related cancers (e.g., kidney cancers in greens, lung cancers in oranges) with different markers
  - TCGA legend includes full cancer type descriptions
  - Other categories: Different colors for clear distinction
- **Batch Processing**: Efficient handling of large datasets (11,000+ patients)
- **Flexible Patient Selection**: Includes all patients with 2+ modalities

## Data Requirements

The script expects embeddings to be available in:
- `/mnt/f/Projects/HoneyBee/results/shared_data/embeddings/`
  - `patient_data_with_embeddings.csv`
  - `{modality}_embeddings.pkl` files
  - `radiology_embeddings.npy`

## Statistics

Current dataset includes:
- Total patients with embeddings: 11,428
- Patients with 2+ modalities: 11,424
- Most common combination: clinical+molecular+pathology (71.07%)

## Notes

- All visualizations have separate legend files (`*_legend.pdf`) for better readability
- Legend file size is automatically adjusted based on the number of categories
- The script samples 3,000 patients for multimodal visualizations to maintain reasonable processing time
- All visualizations use 300 DPI for publication quality
- Random seed is fixed (42) for reproducibility