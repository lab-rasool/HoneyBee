"""Per-modality preprocessing for the molecular pillar.

Each function ports one of the scripts in
``lab-rasool/SeNMo/package_classes/`` to a callable that accepts a
path or DataFrame and returns a DataFrame in the SeNMo-expected shape.
The :func:`combine_modalities` helper concatenates them into the
final 80,697-dim vector consumed by SeNMo.
"""

from .clinical import preprocess_clinical_covariates
from .combine import SENMO_INPUT_DIM, combine_modalities
from .dna_methylation import preprocess_dna_methylation
from .dna_mutation import preprocess_dna_mutation
from .gene_expression import preprocess_gene_expression
from .mirna import preprocess_mirna
from .protein import preprocess_protein

__all__ = [
    "SENMO_INPUT_DIM",
    "combine_modalities",
    "preprocess_clinical_covariates",
    "preprocess_dna_methylation",
    "preprocess_dna_mutation",
    "preprocess_gene_expression",
    "preprocess_mirna",
    "preprocess_protein",
]