"""MedSpacy NER backend — clinical context detection (negation, uncertainty)."""

from __future__ import annotations

import logging
from typing import List, Optional

from ..types import ClinicalEntity
from .base import NERBackend

logger = logging.getLogger(__name__)

# Default oncology-focused target rules covering the categories that _map_category supports.
# Users can override by passing target_rules=[] (empty) or a custom list to __init__.
_DEFAULT_RULES: List[dict] = [
    # PROBLEM — conditions, diseases, findings
    {"literal": "breast cancer", "category": "PROBLEM"},
    {"literal": "invasive ductal carcinoma", "category": "PROBLEM"},
    {"literal": "ductal carcinoma", "category": "PROBLEM"},
    {"literal": "adenocarcinoma", "category": "PROBLEM"},
    {"literal": "lung cancer", "category": "PROBLEM"},
    {"literal": "non-small cell lung cancer", "category": "PROBLEM"},
    {"literal": "carcinoma", "category": "PROBLEM"},
    {"literal": "metastasis", "category": "PROBLEM"},
    {"literal": "hypertension", "category": "PROBLEM"},
    {"literal": "diabetes mellitus", "category": "PROBLEM"},
    {"literal": "diabetes", "category": "PROBLEM"},
    {"literal": "hyperlipidemia", "category": "PROBLEM"},
    {"literal": "atrial fibrillation", "category": "PROBLEM"},
    {"literal": "pneumonia", "category": "PROBLEM"},
    {"literal": "tumor", "category": "PROBLEM"},
    {"literal": "lymphoma", "category": "PROBLEM"},
    {"literal": "melanoma", "category": "PROBLEM"},
    {"literal": "leukemia", "category": "PROBLEM"},
    {"literal": "sarcoma", "category": "PROBLEM"},
    {"literal": "anemia", "category": "PROBLEM"},
    {"literal": "nausea", "category": "PROBLEM"},
    {"literal": "fatigue", "category": "PROBLEM"},
    {"literal": "pain", "category": "PROBLEM"},
    {"literal": "edema", "category": "PROBLEM"},
    # MEDICATION — drugs and therapies
    {"literal": "tamoxifen", "category": "MEDICATION"},
    {"literal": "metformin", "category": "MEDICATION"},
    {"literal": "lisinopril", "category": "MEDICATION"},
    {"literal": "atorvastatin", "category": "MEDICATION"},
    {"literal": "ondansetron", "category": "MEDICATION"},
    {"literal": "doxorubicin", "category": "MEDICATION"},
    {"literal": "cyclophosphamide", "category": "MEDICATION"},
    {"literal": "paclitaxel", "category": "MEDICATION"},
    {"literal": "carboplatin", "category": "MEDICATION"},
    {"literal": "pembrolizumab", "category": "MEDICATION"},
    {"literal": "cisplatin", "category": "MEDICATION"},
    {"literal": "gemcitabine", "category": "MEDICATION"},
    {"literal": "ibuprofen", "category": "MEDICATION"},
    {"literal": "warfarin", "category": "MEDICATION"},
    {"literal": "aspirin", "category": "MEDICATION"},
    # TEST — diagnostic procedures and labs
    {"literal": "mammogram", "category": "TEST"},
    {"literal": "biopsy", "category": "TEST"},
    {"literal": "CT scan", "category": "TEST"},
    {"literal": "bone scan", "category": "TEST"},
    {"literal": "echocardiogram", "category": "TEST"},
    {"literal": "MRI", "category": "TEST"},
    {"literal": "PET scan", "category": "TEST"},
    {"literal": "blood test", "category": "TEST"},
    {"literal": "ultrasound", "category": "TEST"},
    {"literal": "x-ray", "category": "TEST"},
    {"literal": "colonoscopy", "category": "TEST"},
    {"literal": "endoscopy", "category": "TEST"},
    # TREATMENT — therapeutic interventions
    {"literal": "chemotherapy", "category": "TREATMENT"},
    {"literal": "radiation", "category": "TREATMENT"},
    {"literal": "immunotherapy", "category": "TREATMENT"},
    {"literal": "surgery", "category": "TREATMENT"},
    {"literal": "transplant", "category": "TREATMENT"},
]


class MedSpacyNER(NERBackend):
    """NER using medspacy (pyrush, target_matcher, context)."""

    @property
    def name(self) -> str:
        return "medspacy"

    def __init__(self, target_rules: Optional[List] = None):
        self._target_rules = target_rules if target_rules is not None else _DEFAULT_RULES
        self._nlp = None

    def _load(self):
        if self._nlp is not None:
            return
        try:
            import medspacy

            # Suppress verbose PyRuSH DEBUG logging (uses loguru, not stdlib logging)
            logging.getLogger("PyRuSH").setLevel(logging.WARNING)
            try:
                from loguru import logger as _loguru_logger

                _loguru_logger.disable("PyRuSH")
            except ImportError:
                pass

            self._nlp = medspacy.load(enable=["medspacy_pyrush", "medspacy_target_matcher", "medspacy_context"])
            if self._target_rules:
                from medspacy.target_matcher import TargetRule

                rules = [
                    TargetRule(literal=r["literal"], category=r["category"])
                    for r in self._target_rules
                ]
                self._nlp.get_pipe("medspacy_target_matcher").add(rules)
        except ImportError:
            raise ImportError(
                "medspacy backend requires medspacy. "
                "Install with: pip install medspacy"
            )

    def extract(self, text: str) -> List[ClinicalEntity]:
        self._load()
        doc = self._nlp(text)
        entities: List[ClinicalEntity] = []
        for ent in doc.ents:
            entity = ClinicalEntity(
                text=ent.text,
                type=self._map_category(ent.label_),
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.85,
                properties={"backend": "medspacy", "label": ent.label_},
                is_negated=getattr(ent._, "is_negated", False),
                is_uncertain=getattr(ent._, "is_uncertain", False),
                is_historical=getattr(ent._, "is_historical", False),
                is_family=getattr(ent._, "is_family", False),
            )
            entities.append(entity)
        return entities

    @staticmethod
    def _map_category(label: str) -> str:
        label_lower = label.lower()
        mapping = {
            "problem": "condition",
            "treatment": "medication",
            "test": "procedure",
            "medication": "medication",
        }
        return mapping.get(label_lower, "condition")
