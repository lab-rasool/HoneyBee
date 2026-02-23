"""
ClinicalProcessor — thin orchestrator wiring the modular pipeline stages.

Stages: DocumentIngester → NEREngine → OntologyResolver → TimelineExtractor → EmbeddingEngine
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .embeddings import EmbeddingEngine
from .ingestion import DocumentIngester
from .ner import NEREngine
from .ontology import OntologyResolver
from .temporal import TimelineExtractor
from .types import ClinicalDocument, ClinicalEntity, ClinicalResult

logger = logging.getLogger(__name__)

# Default configuration — sensible out-of-the-box with no external services
DEFAULT_CONFIG: Dict[str, Any] = {
    "ingestion": {
        "strategy": "hi_res",
        "ocr_languages": ["eng"],
    },
    "ner": {
        "backends": ["transformer"],
    },
    "ontology": {
        "backends": [],
        "umls_api_key": None,
        "bioportal_api_key": None,
        "snowstorm_base_url": "https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct",
        "snowstorm_max_retries": 3,
        "snowstorm_base_delay": 0.25,
        "snowstorm_request_delay": 0.15,
        "cache_size": 1000,
    },
    "temporal": {
        "enabled": True,
    },
    "embeddings": {
        "mode": "local",
        "model": "bioclinicalbert",
        "api_base": None,
        "pooling_method": "mean",
    },
}


def _deep_merge(default: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *default*."""
    result = default.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ClinicalProcessor:
    """
    Modular clinical NLP pipeline for oncology.

    Example::

        proc = ClinicalProcessor()
        result = proc.process_text("Patient diagnosed with stage III breast cancer.")
        print(result.entities)

    Stages can also be used independently::

        from honeybee.processors.clinical import DocumentIngester, NEREngine
        doc = DocumentIngester().ingest_text("some clinical text")
        entities = NEREngine(backends=["scispacy"]).extract(doc.text)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = _deep_merge(DEFAULT_CONFIG, config or {})
        self.logger = logging.getLogger(__name__)

        # Instantiate stages lazily-but-eagerly for the lightweight ones
        self.ingester = DocumentIngester(self.config.get("ingestion"))
        self.ner_engine = NEREngine(config=self.config.get("ner"))
        self.ontology_resolver = OntologyResolver(config=self.config.get("ontology"))
        self.timeline_extractor = TimelineExtractor(self.config.get("temporal"))
        # Embedding engine is deferred (loads model weights)
        self._embedding_engine: Optional[EmbeddingEngine] = None

    # ------------------------------------------------------------------
    # Embedding engine (lazy)
    # ------------------------------------------------------------------

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        if self._embedding_engine is None:
            emb_cfg = self.config.get("embeddings", {})
            self._embedding_engine = EmbeddingEngine(
                mode=emb_cfg.get("mode", "local"),
                config=emb_cfg,
            )
        return self._embedding_engine

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(self, source: Union[str, Path], **kwargs: Any) -> ClinicalResult:
        """Full pipeline: ingest → NER → ontology → timeline."""
        document = self.ingester.ingest(source)
        return self._run_pipeline(document, **kwargs)

    def process_text(self, text: str, **kwargs: Any) -> ClinicalResult:
        """Process raw text through the pipeline."""
        document = self.ingester.ingest_text(text)
        return self._run_pipeline(document, **kwargs)

    def process_batch(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*",
        save_output: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> List[ClinicalResult]:
        """Process all files matching *file_pattern* in *input_dir*."""
        input_dir = Path(input_dir)
        results: List[ClinicalResult] = []
        for path in sorted(input_dir.glob(file_pattern)):
            if path.is_file():
                try:
                    result = self.process(path, **kwargs)
                    if save_output:
                        self._save_output(result, path, output_dir)
                    results.append(result)
                except Exception as exc:
                    self.logger.warning("Failed to process %s: %s", path, exc)
        return results

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate embeddings (delegates to EmbeddingEngine).

        Args:
            text: Single string or list of strings.
            model_name: Override the default model for this call.
            **kwargs: Passed to the embedding backend (batch_size, etc.).
                Keys like ``pooling_method`` and ``max_length`` are
                engine-config params and will be extracted automatically.
        """
        # Keys that are engine config, not per-call params
        _ENGINE_CONFIG_KEYS = ("pooling_method", "max_length")

        if model_name:
            # One-off engine for a specific model
            emb_cfg = dict(self.config.get("embeddings", {}))
            emb_cfg["model"] = model_name
            for key in _ENGINE_CONFIG_KEYS:
                if key in kwargs:
                    emb_cfg[key] = kwargs.pop(key)
            engine = EmbeddingEngine(mode=emb_cfg.get("mode", "local"), config=emb_cfg)
            return engine.embed(text, **kwargs)

        # Also handle config overrides for the default engine path
        cfg_overrides = {k: kwargs.pop(k) for k in _ENGINE_CONFIG_KEYS if k in kwargs}
        if cfg_overrides:
            emb_cfg = dict(self.config.get("embeddings", {}))
            emb_cfg.update(cfg_overrides)
            engine = EmbeddingEngine(mode=emb_cfg.get("mode", "local"), config=emb_cfg)
            return engine.embed(text, **kwargs)

        return self.embedding_engine.embed(text, **kwargs)

    # ------------------------------------------------------------------
    # Interop helpers
    # ------------------------------------------------------------------

    def to_fhir(self, result: ClinicalResult, patient_id: Optional[str] = None) -> Dict:
        """Convert a ClinicalResult to a FHIR R4 Bundle."""
        from .interop.fhir_converter import FHIRConverter

        return FHIRConverter().to_fhir_bundle(result, patient_id=patient_id)

    def process_fhir(self, bundle_json: Dict) -> ClinicalResult:
        """Ingest a FHIR Bundle and run the NLP pipeline on its narrative text."""
        from .interop.fhir_converter import FHIRConverter

        parsed = FHIRConverter().from_fhir_bundle(bundle_json)
        text = parsed.get("text", "")
        return self.process_text(text)

    def process_hl7(self, message: str) -> ClinicalResult:
        """Parse an HL7 v2 message and run the NLP pipeline on its text."""
        from .interop.hl7_parser import HL7Parser

        parsed = HL7Parser().parse(message)
        text = parsed.get("text", "")
        return self.process_text(text)

    # ------------------------------------------------------------------
    # Summary stats (kept for backward compat)
    # ------------------------------------------------------------------

    def get_summary_statistics(self, result: ClinicalResult) -> Dict[str, Any]:
        """Return summary statistics for a processing result."""
        entity_types: Dict[str, int] = {}
        for e in result.entities:
            entity_types[e.type] = entity_types.get(e.type, 0) + 1
        return {
            "text_length": len(result.text),
            "num_entities": len(result.entities),
            "entity_types": entity_types,
            "num_timeline_events": len(result.timeline),
            "num_relationships": len(result.relationships),
        }

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, document: ClinicalDocument, **kwargs: Any) -> ClinicalResult:
        # 1. NER
        entities = self.ner_engine.extract(document.text)

        # 2. Ontology resolution
        entities = self.ontology_resolver.resolve(entities)

        # 3. Relationships (proximity-based, same as legacy)
        relationships = self._extract_relationships(document.text, entities)

        # 4. Timeline
        timeline = []
        if self.config.get("temporal", {}).get("enabled", True):
            timeline = self.timeline_extractor.extract(document, entities)

        return ClinicalResult(
            document=document,
            entities=entities,
            relationships=relationships,
            timeline=timeline,
        )

    @staticmethod
    def _extract_relationships(
        text: str, entities: List[ClinicalEntity]
    ) -> List[Dict[str, Any]]:
        """Simple proximity-based relationship extraction."""
        relationships: List[Dict[str, Any]] = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if j <= i:
                    continue
                # Within 50 characters of each other
                gap = min(abs(e1.end - e2.start), abs(e2.end - e1.start))
                if gap > 50:
                    continue
                rel_type = _determine_relationship(e1.type, e2.type)
                if rel_type:
                    relationships.append({
                        "type": rel_type,
                        "source_idx": i,
                        "target_idx": j,
                        "source_text": e1.text,
                        "target_text": e2.text,
                    })
        return relationships

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def _save_output(
        result: ClinicalResult,
        source_path: Path,
        output_dir: Optional[Union[str, Path]],
    ) -> None:
        out_dir = Path(output_dir) if output_dir else source_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{source_path.stem}_result.json"
        out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))


# ------------------------------------------------------------------
# Relationship helpers
# ------------------------------------------------------------------

_RELATIONSHIP_MAP = {
    ("tumor", "staging"): "has_stage",
    ("tumor", "biomarker"): "has_biomarker",
    ("tumor", "measurement"): "has_measurement",
    ("condition", "medication"): "treated_with",
    ("medication", "condition"): "treats",
    ("condition", "temporal"): "temporal_relation",
    ("tumor", "temporal"): "temporal_relation",
    ("medication", "temporal"): "temporal_relation",
    ("procedure", "temporal"): "temporal_relation",
    ("staging", "temporal"): "temporal_relation",
    ("biomarker", "temporal"): "temporal_relation",
    ("measurement", "temporal"): "temporal_relation",
    ("response", "temporal"): "temporal_relation",
    ("condition", "procedure"): "investigated_by",
    ("procedure", "measurement"): "has_result",
    ("response", "medication"): "response_to",
    ("response", "procedure"): "response_to",
    ("staging", "staging"): "progression",
}


def _determine_relationship(type1: str, type2: str) -> Optional[str]:
    """Map entity type pair to relationship type."""
    return _RELATIONSHIP_MAP.get((type1, type2)) or _RELATIONSHIP_MAP.get((type2, type1))
