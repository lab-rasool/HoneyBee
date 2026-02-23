"""
Shared data types for the clinical processing pipeline.

All stages of the pipeline (ingestion, NER, ontology, temporal, embeddings)
use these dataclasses to pass data between each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class OntologyCode:
    """A single ontology code linked to a clinical entity."""

    system: str  # "snomed_ct", "rxnorm", "loinc", "icd10cm"
    code: str
    display: str
    source_api: str = "local"  # "umls", "bioportal", "snowstorm", "local"


@dataclass
class ClinicalEntity:
    """A clinical entity extracted from text."""

    text: str
    type: str  # condition, medication, biomarker, staging, measurement, temporal, procedure, tumor
    start: int
    end: int
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    # Context attributes (populated by medspacy or context-aware backends)
    is_negated: bool = False
    is_uncertain: bool = False
    is_historical: bool = False
    is_family: bool = False
    # Ontology links (populated by OntologyResolver)
    ontology_codes: List[OntologyCode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        d = {
            "text": self.text,
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "properties": self.properties,
            "is_negated": self.is_negated,
            "is_uncertain": self.is_uncertain,
            "is_historical": self.is_historical,
            "is_family": self.is_family,
            "ontology_codes": [
                {
                    "system": oc.system,
                    "code": oc.code,
                    "display": oc.display,
                    "source_api": oc.source_api,
                }
                for oc in self.ontology_codes
            ],
        }
        return d


@dataclass
class ClinicalDocument:
    """A clinical document with text, sections, and metadata."""

    text: str
    sections: Dict[str, str] = field(default_factory=dict)  # section_name -> content
    metadata: Dict[str, Any] = field(default_factory=dict)  # source_type, extraction_method, etc.
    source_path: Optional[Path] = None


@dataclass
class TimelineEvent:
    """A temporal event extracted from clinical text."""

    date: Optional[datetime] = None
    date_text: str = ""
    sentence: str = ""
    related_entities: List[int] = field(default_factory=list)  # indices into entity list

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "date": self.date.isoformat() if self.date else None,
            "date_text": self.date_text,
            "sentence": self.sentence,
            "related_entities": self.related_entities,
        }


@dataclass
class ClinicalResult:
    """The full output of the clinical processing pipeline."""

    document: ClinicalDocument
    entities: List[ClinicalEntity] = field(default_factory=list)
    relationships: List[Dict] = field(default_factory=list)
    timeline: List[TimelineEvent] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None

    @property
    def text(self) -> str:
        """Shortcut to document text."""
        return self.document.text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        d: Dict[str, Any] = {
            "text": self.document.text,
            "sections": self.document.sections,
            "metadata": self.document.metadata,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": self.relationships,
            "timeline": [t.to_dict() for t in self.timeline],
        }
        if self.embeddings is not None:
            d["embeddings_shape"] = list(self.embeddings.shape)
        return d
