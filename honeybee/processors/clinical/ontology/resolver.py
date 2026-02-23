"""
Ontology resolver — queries configured API backends to attach ontology codes
to ClinicalEntity objects.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from ..types import ClinicalEntity, OntologyCode

logger = logging.getLogger(__name__)


class OntologyResolver:
    """Resolve clinical entities against ontology APIs (UMLS, BioPortal, Snowstorm)."""

    def __init__(
        self,
        backends: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        backend_names = backends or config.get("backends", [])
        self._cache_size = config.get("cache_size", 1000)
        self._clients: Dict[str, Any] = {}

        for name in backend_names:
            try:
                self._clients[name] = self._init_client(name, config)
            except (ImportError, ValueError) as exc:
                logger.warning("Skipping ontology backend %r: %s", name, exc)

        if backend_names and not self._clients:
            self._warn_missing_keys(backend_names, config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """Attach ontology codes to each entity (mutates in place, also returns)."""
        if not self._clients:
            return entities

        for entity in entities:
            codes = self._lookup(entity.text)
            entity.ontology_codes.extend(codes)
        return entities

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lookup(self, term: str) -> List[OntologyCode]:
        """Query all configured backends for *term*."""
        # Use the cached version
        return self._cached_lookup(term)

    @lru_cache(maxsize=1000)
    def _cached_lookup(self, term: str) -> List[OntologyCode]:
        codes: List[OntologyCode] = []
        for name, client in self._clients.items():
            try:
                results = client.search(term)
                codes.extend(results)
            except Exception as exc:
                logger.warning("Ontology backend %s failed for %r: %s", name, term, exc)
        return codes

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------

    @staticmethod
    def _init_client(name: str, config: Dict) -> Any:
        if name == "umls":
            api_key = config.get("umls_api_key") or os.environ.get("UMLS_API_KEY")
            if not api_key:
                raise ValueError(
                    "UMLS requires an API key. Set 'umls_api_key' in config "
                    "or UMLS_API_KEY env var. Sign up: https://uts.nlm.nih.gov/uts/"
                )
            from .umls_client import UMLSClient

            return UMLSClient(api_key=api_key)
        elif name == "bioportal":
            api_key = config.get("bioportal_api_key") or os.environ.get("BIOPORTAL_API_KEY")
            if not api_key:
                raise ValueError(
                    "BioPortal requires an API key. Set 'bioportal_api_key' in config "
                    "or BIOPORTAL_API_KEY env var. Sign up: https://bioportal.bioontology.org/accounts/new"
                )
            from .bioportal_client import BioPortalClient

            return BioPortalClient(api_key=api_key)
        elif name == "snowstorm":
            from .snowstorm_client import SnowstormClient

            base_url = config.get(
                "snowstorm_base_url",
                "https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct",
            )
            return SnowstormClient(
                base_url=base_url,
                max_retries=config.get("snowstorm_max_retries", 3),
                base_delay=config.get("snowstorm_base_delay", 0.25),
                request_delay=config.get("snowstorm_request_delay", 0.15),
            )
        else:
            raise ValueError(
                f"Unknown ontology backend: {name!r}. "
                "Available: umls, bioportal, snowstorm"
            )

    @staticmethod
    def _warn_missing_keys(names: List[str], config: Dict) -> None:
        msgs = []
        if "umls" in names:
            msgs.append(
                "UMLS: set UMLS_API_KEY env var or umls_api_key in config "
                "(https://uts.nlm.nih.gov/uts/)"
            )
        if "bioportal" in names:
            msgs.append(
                "BioPortal: set BIOPORTAL_API_KEY env var or bioportal_api_key in config "
                "(https://bioportal.bioontology.org/accounts/new)"
            )
        if msgs:
            logger.warning(
                "No ontology backends could be initialized. "
                "Required API keys:\n  " + "\n  ".join(msgs)
            )
