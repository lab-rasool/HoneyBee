"""
NER Engine — orchestrates multiple NER backends and merges their results.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..types import ClinicalEntity
from .base import NERBackend

logger = logging.getLogger(__name__)

# Registry of backend name → class path (lazy imports)
_BACKEND_REGISTRY = {
    "scispacy": ".scispacy_backend:SciSpacyNER",
    "medspacy": ".medspacy_backend:MedSpacyNER",
    "medcat": ".medcat_backend:MedCATNER",
    "transformer": ".transformer_backend:TransformerNER",
}


def _import_backend(name: str, config: Dict[str, Any]) -> NERBackend:
    """Instantiate a backend by name."""
    if name == "scispacy":
        from .scispacy_backend import SciSpacyNER

        return SciSpacyNER(model=config.get("scispacy_model", "en_core_sci_lg"))
    elif name == "medspacy":
        from .medspacy_backend import MedSpacyNER

        return MedSpacyNER(target_rules=config.get("medspacy_rules"))
    elif name == "medcat":
        from .medcat_backend import MedCATNER

        return MedCATNER(model_pack_path=config.get("medcat_model_pack"))
    elif name == "transformer":
        from .transformer_backend import TransformerNER

        return TransformerNER(model=config.get("transformer_model", "d4data/biomedical-ner-all"))
    else:
        raise ValueError(
            f"Unknown NER backend: {name!r}. "
            f"Available: {list(_BACKEND_REGISTRY)}"
        )


class NEREngine:
    """Run one or more NER backends, merge and deduplicate results."""

    def __init__(
        self,
        backends: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        backend_names = backends or config.get("backends", [])
        self._backends: List[NERBackend] = []
        for name in backend_names:
            try:
                self._backends.append(_import_backend(name, config))
            except (ImportError, RuntimeError, ValueError) as exc:
                logger.warning("Skipping NER backend %r: %s", name, exc)

        if not self._backends:
            logger.warning("No NER backends available; extract() will return empty results")

    def extract(self, text: str) -> List[ClinicalEntity]:
        """Run all backends on *text*, merge, and return deduplicated entities."""
        all_entities: List[ClinicalEntity] = []
        for backend in self._backends:
            try:
                ents = backend.extract(text)
                all_entities.extend(ents)
            except Exception as exc:
                logger.warning("Backend %s failed: %s", backend.name, exc)
        return self._merge_entities(all_entities)

    @property
    def backend_names(self) -> List[str]:
        return [b.name for b in self._backends]

    # ------------------------------------------------------------------
    # Merge / deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_entities(entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """Deduplicate overlapping entities.

        Rules:
        - Same type, overlapping span → keep the longer (or higher confidence) one.
        - Different type, overlapping span → keep both.
        """
        if not entities:
            return []

        # Sort by start position, then by span length descending
        entities.sort(key=lambda e: (e.start, -(e.end - e.start)))

        merged: List[ClinicalEntity] = []
        for entity in entities:
            is_duplicate = False
            for existing in merged:
                # Check overlap
                if entity.start < existing.end and entity.end > existing.start:
                    if entity.type == existing.type:
                        # Same type: keep longer or higher confidence
                        if (entity.end - entity.start) > (existing.end - existing.start):
                            merged.remove(existing)
                            merged.append(entity)
                        elif (entity.end - entity.start) == (
                            existing.end - existing.start
                        ) and entity.confidence > existing.confidence:
                            merged.remove(existing)
                            merged.append(entity)
                        is_duplicate = True
                        break
                    # Different type: keep both (not a duplicate)
            if not is_duplicate:
                merged.append(entity)

        # Re-sort by position
        merged.sort(key=lambda e: e.start)
        return merged
