"""
HoneyBee Clinical Processor subpackage.

Modular pipeline: Ingestion → NER → Ontology → Temporal → Embeddings
"""

from .embeddings import EmbeddingEngine
from .ingestion import DocumentIngester
from .ner import NEREngine
from .ontology import OntologyResolver
from .processor import ClinicalProcessor
from .temporal import TimelineExtractor
from .types import (
    ClinicalDocument,
    ClinicalEntity,
    ClinicalResult,
    OntologyCode,
    TimelineEvent,
)

__all__ = [
    "ClinicalProcessor",
    "ClinicalDocument",
    "ClinicalEntity",
    "ClinicalResult",
    "OntologyCode",
    "TimelineEvent",
    "DocumentIngester",
    "NEREngine",
    "OntologyResolver",
    "TimelineExtractor",
    "EmbeddingEngine",
]
