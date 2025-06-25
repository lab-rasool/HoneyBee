# Shared Data Directory

This directory contains shared data files used by the survival, classification, and retrieval analysis folders.

## Contents

### Embeddings Directory (`embeddings/`)

Contains pre-computed embeddings for different modalities:

- **clinical_embeddings.pkl**: Clinical text embeddings from Gatortron model
- **pathology_embeddings.pkl**: Pathology report embeddings
- **radiology_embeddings.npy**: Radiology image embeddings
- **molecular_embeddings.npy**: Molecular/genomic embeddings
- **multimodal_embeddings.npy**: Combined multimodal embeddings
- **patient_data_with_embeddings.csv**: Patient metadata with embedding references
- **patient_index_mapping.csv**: Mapping between patient IDs and indices

## Usage

All analysis folders (survival, classification, retrieval) reference this shared data directory to:
- Avoid data duplication
- Ensure consistency across analyses
- Make the results folders independent of the old directory structure

## Data Format

### Pickle Files (.pkl)
Contains dictionaries with:
- `X`: Embedding matrix (numpy array)
- `y`: Labels (cancer types)
- `patient_ids`: Patient identifiers

### Numpy Files (.npy)
Raw embedding arrays that need to be loaded with corresponding metadata

### CSV Files
Metadata and mapping information for patient data

## Important Notes

1. This directory should be kept with the analysis folders when moving or copying results
2. The embeddings are pre-computed and should not be modified
3. All paths in the analysis scripts are relative to ensure portability