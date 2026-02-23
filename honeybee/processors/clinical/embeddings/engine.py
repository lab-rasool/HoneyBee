"""
Embedding engine — dispatches to local HuggingFace or API-based backend.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..types import ClinicalDocument

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Unified embedding interface supporting local and API modes."""

    def __init__(self, mode: str = "local", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._mode = mode or config.get("mode", "local")
        self._config = config
        self._backend = None

    def _load(self):
        if self._backend is not None:
            return
        if self._mode == "local":
            from .local_backend import LocalEmbeddingBackend

            self._backend = LocalEmbeddingBackend(
                model_name=self._config.get("model", "bioclinicalbert"),
                pooling_method=self._config.get("pooling_method", "mean"),
                max_length=self._config.get("max_length"),
            )
        elif self._mode == "api":
            from .api_backend import APIEmbeddingBackend

            self._backend = APIEmbeddingBackend(
                model=self._config.get("model", "ollama/nomic-embed-text"),
                api_base=self._config.get("api_base"),
            )
        else:
            raise ValueError(f"Unknown embedding mode: {self._mode!r}. Use 'local' or 'api'.")

    def embed(self, texts: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        """Generate embeddings for text(s)."""
        self._load()
        return self._backend.embed(texts, **kwargs)

    def embed_document(self, document: ClinicalDocument, **kwargs: Any) -> np.ndarray:
        """Embed the full document text."""
        return self.embed(document.text, **kwargs)
