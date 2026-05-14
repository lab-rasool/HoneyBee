"""Result dataclass for the molecular pillar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np


@dataclass
class MolecularResult:
    """Output of :meth:`MolecularProcessor.process`.

    Attributes:
        embedding: 48-dim patient embedding from the SeNMo encoder.
            Shape ``(48,)`` for single-patient input, ``(N, 48)`` for
            batched input.
        hazard_score: Cox-style hazard score from the SeNMo regression
            head. A Python ``float`` for single-patient input, an
            ``np.ndarray`` of shape ``(N,)`` for batched input. Higher
            values indicate worse prognosis.
        input_features: The multi-omics feature vector fed into the
            model. Stored on the result so downstream code can refer
            back to it (e.g., for caching).
    """

    embedding: np.ndarray
    hazard_score: Union[float, np.ndarray]
    input_features: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of the result."""
        hazard = (
            self.hazard_score.tolist()
            if isinstance(self.hazard_score, np.ndarray)
            else float(self.hazard_score)
        )
        return {
            "embedding": self.embedding.tolist(),
            "hazard_score": hazard,
            "input_features_shape": list(self.input_features.shape),
        }