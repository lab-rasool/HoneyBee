# Multimodal Cancer Data Embeddings

## Data Structure

- **patient_data_with_embeddings.csv**: Contains ALL patient clinical data and references to embedding files
- **patient_index_mapping.csv**: Simple mapping between array indices and patient IDs
- **[modality]_embeddings.npy**: NumPy arrays containing embeddings for each modality
- **multimodal_embeddings.npy**: Combined multimodal embeddings created by concatenating scaled modality embeddings

## Dataset Statistics

- Total patients: 1105
- Cancer types: TCGA-BLCA, TCGA-BRCA, TCGA-CESC, TCGA-COAD, TCGA-ESCA, TCGA-KICH, TCGA-KIRC, TCGA-KIRP, TCGA-LIHC, TCGA-LUAD, TCGA-LUSC, TCGA-OV, TCGA-PRAD, TCGA-READ, TCGA-SARC, TCGA-STAD, TCGA-THCA, TCGA-UCEC
- Embedding dimensions:
  - clinical: 1
  - pathology: 1024
  - radiology: 1000
  - molecular: 48
  - multimodal: 3096
