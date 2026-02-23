"""Transformer NER backend — HuggingFace token classification pipeline."""

from __future__ import annotations

import logging
from typing import List

from ..types import ClinicalEntity
from .base import NERBackend

logger = logging.getLogger(__name__)

# d4data/biomedical-ner-all entity label mapping
_LABEL_MAP = {
    "Disease_disorder": "condition",
    "Sign_symptom": "condition",
    "Medication": "medication",
    "Therapeutic_procedure": "procedure",
    "Diagnostic_procedure": "procedure",
    "Biological_structure": "anatomy",
    "Lab_value": "measurement",
    "Dosage": "dosage",
    "Duration": "temporal",
    "Date": "temporal",
    "Age": "temporal",
    "Clinical_event": "condition",
}


class TransformerNER(NERBackend):
    """NER using a HuggingFace token-classification pipeline."""

    @property
    def name(self) -> str:
        return "transformer"

    def __init__(self, model: str = "d4data/biomedical-ner-all"):
        self._model_name = model
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "ner",
                model=self._model_name,
                aggregation_strategy="simple",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load transformer NER model '{self._model_name}': {exc}"
            ) from exc

    def extract(self, text: str) -> List[ClinicalEntity]:
        self._load()
        results = self._pipeline(text)
        entities: List[ClinicalEntity] = []
        for r in results:
            # Strip B-/I- prefixes
            label = r.get("entity_group", r.get("entity", ""))
            label = label.lstrip("B-").lstrip("I-")
            etype = _LABEL_MAP.get(label)
            if etype is None:
                continue
            start = r.get("start", 0)
            end = r.get("end", 0)
            entities.append(
                ClinicalEntity(
                    text=text[start:end],
                    type=etype,
                    start=start,
                    end=end,
                    confidence=float(r.get("score", 0.0)),
                    properties={"backend": "transformer", "label": label},
                )
            )
        return entities
