"""SciSpacy NER backend — biomedical named entity recognition via spaCy."""

from __future__ import annotations

import logging
from typing import List

from ..types import ClinicalEntity
from .base import NERBackend

logger = logging.getLogger(__name__)

# spaCy label → HoneyBee entity type
_LABEL_MAP = {
    "DISEASE": "condition",
    "CHEMICAL": "medication",
    "ENTITY": "condition",
    "DATE": "temporal",
    "GENE_OR_GENE_PRODUCT": "biomarker",
}


class SciSpacyNER(NERBackend):
    """NER using a scispacy model (e.g. ``en_core_sci_lg``)."""

    @property
    def name(self) -> str:
        return "scispacy"

    def __init__(self, model: str = "en_core_sci_lg"):
        self._model_name = model
        self._nlp = None

    def _load(self):
        if self._nlp is not None:
            return
        try:
            import spacy

            self._nlp = spacy.load(self._model_name)
        except ImportError:
            raise ImportError(
                "scispacy backend requires spacy and a scispacy model. "
                "Install with: pip install scispacy && "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz --no-deps"
            )
        except OSError:
            raise ImportError(
                f"spaCy model '{self._model_name}' not found. "
                "Install with: pip install https://s3-us-west-2.amazonaws.com/"
                "ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz --no-deps"
            )

    def extract(self, text: str) -> List[ClinicalEntity]:
        self._load()
        doc = self._nlp(text)
        entities: List[ClinicalEntity] = []
        for ent in doc.ents:
            etype = _LABEL_MAP.get(ent.label_)
            if etype is None:
                continue
            entities.append(
                ClinicalEntity(
                    text=ent.text,
                    type=etype,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.80,
                    properties={"backend": "scispacy", "label": ent.label_},
                )
            )
        return entities
