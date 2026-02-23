"""BioPortal API client for ontology lookup and annotation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..types import OntologyCode

logger = logging.getLogger(__name__)

_BASE_URL = "https://data.bioontology.org"


class BioPortalClient:
    """Thin wrapper around the NCBO BioPortal REST API."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        try:
            import requests  # noqa: F401

            self._requests = requests
        except ImportError:
            raise ImportError("BioPortal client requires requests: pip install requests")

    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"apikey token={self._api_key}"}

    def search(
        self,
        term: str,
        ontologies: Optional[List[str]] = None,
        max_results: int = 5,
    ) -> List[OntologyCode]:
        """Search BioPortal for a term."""
        params: Dict = {"q": term, "pagesize": max_results}
        if ontologies:
            params["ontologies"] = ",".join(ontologies)

        try:
            resp = self._requests.get(
                f"{_BASE_URL}/search",
                params=params,
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("BioPortal search failed for %r: %s", term, exc)
            return []

        codes: List[OntologyCode] = []
        for item in data.get("collection", [])[:max_results]:
            ontology_id = item.get("links", {}).get("ontology", "").split("/")[-1]
            codes.append(
                OntologyCode(
                    system=ontology_id.lower(),
                    code=item.get("@id", "").split("/")[-1],
                    display=item.get("prefLabel", ""),
                    source_api="bioportal",
                )
            )
        return codes

    def annotate(
        self,
        text: str,
        ontologies: Optional[List[str]] = None,
    ) -> List[OntologyCode]:
        """Auto-annotate text using BioPortal's Annotator."""
        params: Dict = {"text": text}
        if ontologies:
            params["ontologies"] = ",".join(ontologies)

        try:
            resp = self._requests.post(
                f"{_BASE_URL}/annotator",
                data=params,
                headers=self._headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("BioPortal annotate failed: %s", exc)
            return []

        codes: List[OntologyCode] = []
        for ann in data:
            cls = ann.get("annotatedClass", {})
            ontology_id = cls.get("links", {}).get("ontology", "").split("/")[-1]
            codes.append(
                OntologyCode(
                    system=ontology_id.lower(),
                    code=cls.get("@id", "").split("/")[-1],
                    display=cls.get("prefLabel", ann.get("text", "")),
                    source_api="bioportal",
                )
            )
        return codes
