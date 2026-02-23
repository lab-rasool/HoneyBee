"""MedCAT NER backend — UMLS concept recognition via MedCAT model packs."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..types import ClinicalEntity, OntologyCode
from .base import NERBackend

logger = logging.getLogger(__name__)


class MedCATNER(NERBackend):
    """NER using a MedCAT model pack (optional, heavyweight)."""

    @property
    def name(self) -> str:
        return "medcat"

    def __init__(self, model_pack_path: Optional[str] = None):
        self._model_pack_path = model_pack_path
        self._cat = None

    def _load(self):
        if self._cat is not None:
            return
        try:
            from medcat.cat import CAT

            if self._model_pack_path:
                self._cat = CAT.load_model_pack(self._model_pack_path)
            else:
                logger.warning(
                    "MedCAT backend initialized without a model pack. "
                    "Provide model_pack_path for best results."
                )
                self._cat = CAT.load_model_pack("umls_sm_v1.0")
        except ImportError:
            raise ImportError(
                "medcat backend requires medcat. "
                "Install with: pip install medcat"
            )

    def extract(self, text: str) -> List[ClinicalEntity]:
        self._load()
        result = self._cat.get_entities(text)
        entities: List[ClinicalEntity] = []

        ents = result.get("entities", {})
        if isinstance(ents, dict):
            ents = ents.values()

        for ent in ents:
            cui = ent.get("cui", "")
            name = ent.get("detected_name", ent.get("source_value", ""))
            pretty_name = ent.get("pretty_name", name)

            ontology_codes = []
            if cui:
                ontology_codes.append(
                    OntologyCode(
                        system="umls",
                        code=cui,
                        display=pretty_name,
                        source_api="medcat",
                    )
                )

            entities.append(
                ClinicalEntity(
                    text=name,
                    type=self._map_type(ent.get("types", [])),
                    start=ent.get("start", 0),
                    end=ent.get("end", 0),
                    confidence=ent.get("acc", 0.0),
                    properties={"backend": "medcat", "cui": cui},
                    ontology_codes=ontology_codes,
                )
            )
        return entities

    @staticmethod
    def _map_type(types: list) -> str:
        if not types:
            return "condition"
        # MedCAT semantic types to HoneyBee types
        mapping = {
            "T047": "condition",  # Disease or Syndrome
            "T191": "condition",  # Neoplastic Process
            "T121": "medication",  # Pharmacologic Substance
            "T059": "procedure",  # Laboratory Procedure
            "T060": "procedure",  # Diagnostic Procedure
            "T061": "procedure",  # Therapeutic Procedure
            "T201": "measurement",  # Clinical Attribute
            "T033": "condition",  # Finding
            "T184": "condition",  # Sign or Symptom
        }
        for t in types:
            if t in mapping:
                return mapping[t]
        return "condition"
