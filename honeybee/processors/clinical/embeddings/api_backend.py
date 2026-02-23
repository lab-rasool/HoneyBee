"""
API embedding backend — wraps litellm for remote embedding services
(OpenAI, Ollama, vLLM, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class APIEmbeddingBackend:
    """Generate embeddings via a remote API using litellm."""

    def __init__(
        self,
        model: str = "ollama/nomic-embed-text",
        api_base: Optional[str] = None,
        **kwargs: Any,
    ):
        self._model = model
        self._api_base = api_base
        self._kwargs = kwargs

    def embed(self, texts: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        """Synchronous embedding call via litellm."""
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "API embedding backend requires litellm. "
                "Install with: pip install litellm"
            )

        if isinstance(texts, str):
            texts = [texts]

        call_kwargs: Dict[str, Any] = {"model": self._model, "input": texts}
        if self._api_base:
            call_kwargs["api_base"] = self._api_base
        call_kwargs.update(self._kwargs)
        call_kwargs.update(kwargs)

        response = litellm.embedding(**call_kwargs)
        embeddings = [item["embedding"] for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    async def aembed(self, texts: Union[str, List[str]], **kwargs: Any) -> np.ndarray:
        """Asynchronous embedding call via litellm."""
        try:
            import litellm
        except ImportError:
            raise ImportError("API embedding backend requires litellm")

        if isinstance(texts, str):
            texts = [texts]

        call_kwargs: Dict[str, Any] = {"model": self._model, "input": texts}
        if self._api_base:
            call_kwargs["api_base"] = self._api_base
        call_kwargs.update(self._kwargs)
        call_kwargs.update(kwargs)

        response = await litellm.aembedding(**call_kwargs)
        embeddings = [item["embedding"] for item in response.data]
        return np.array(embeddings, dtype=np.float32)
