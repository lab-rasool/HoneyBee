"""
Local embedding backend — wraps HuggingFaceEmbedder for biomedical models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Preset biomedical models (copied from legacy BIOMEDICAL_MODELS)
BIOMEDICAL_MODELS: Dict[str, Dict[str, Any]] = {
    "bioclinicalbert": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "max_length": 512,
        "description": "Clinical BERT trained on MIMIC-III clinical notes",
        "gated": False,
    },
    "pubmedbert": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "max_length": 512,
        "description": "BERT trained on PubMed abstracts and full-text articles",
        "gated": False,
    },
    "gatortron": {
        "model_name": "UFNLP/gatortron-base",
        "max_length": 512,
        "description": "Clinical foundation model from University of Florida",
        "gated": True,
    },
    "clinicalt5": {
        "model_name": "razent/SciFive-base-PMC",
        "max_length": 512,
        "description": "T5 model fine-tuned on PubMed Central articles",
        "gated": False,
    },
    "biobert": {
        "model_name": "dmis-lab/biobert-v1.1",
        "max_length": 512,
        "description": "BioBERT for biomedical text mining",
        "gated": False,
    },
    "scibert": {
        "model_name": "allenai/scibert_scivocab_uncased",
        "max_length": 512,
        "description": "BERT model trained on scientific publications",
        "gated": False,
    },
    "sentence-transformers": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_length": 256,
        "description": "Fast lightweight sentence embeddings (general purpose)",
        "gated": False,
    },
}


class LocalEmbeddingBackend:
    """Generate embeddings using a local HuggingFace transformer model."""

    def __init__(
        self,
        model_name: str = "bioclinicalbert",
        pooling_method: str = "mean",
        max_length: Optional[int] = None,
        **kwargs: Any,
    ):
        # Resolve preset name to HF model ID
        preset = BIOMEDICAL_MODELS.get(model_name)
        if preset:
            self._hf_model_name = preset["model_name"]
            self._max_length = max_length or preset["max_length"]
        else:
            self._hf_model_name = model_name
            self._max_length = max_length or 512

        self._pooling_method = pooling_method
        self._kwargs = kwargs
        self._embedder = None

    def _load(self):
        if self._embedder is not None:
            return
        from honeybee.models import HuggingFaceEmbedder

        self._embedder = HuggingFaceEmbedder(
            model_name=self._hf_model_name,
            pooling_method=self._pooling_method,
            max_length=self._max_length,
            **self._kwargs,
        )

    def embed(self, texts: Union[str, List[str]], batch_size: int = 32, **kwargs: Any) -> np.ndarray:
        """Generate embeddings for one or more texts."""
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        return self._embedder.generate_embeddings(texts, batch_size=batch_size)
