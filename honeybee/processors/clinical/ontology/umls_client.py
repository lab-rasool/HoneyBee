"""UMLS Terminology Services API client."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..types import OntologyCode

logger = logging.getLogger(__name__)

_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"


class UMLSClient:
    """Thin wrapper around the UMLS REST API."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        try:
            import requests  # noqa: F401

            self._requests = requests
        except ImportError:
            raise ImportError("UMLS client requires requests: pip install requests")

    def search(
        self,
        term: str,
        sabs: Optional[str] = None,
        max_results: int = 5,
    ) -> List[OntologyCode]:
        """Search UMLS for a term, optionally restricting to source vocabularies."""
        params: Dict = {
            "apiKey": self._api_key,
            "string": term,
            "pageSize": max_results,
        }
        if sabs:
            params["sabs"] = sabs

        try:
            resp = self._requests.get(
                f"{_BASE_URL}/search/current",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("UMLS search failed for %r: %s", term, exc)
            return []

        results = data.get("result", {}).get("results", [])
        codes: List[OntologyCode] = []
        for r in results:
            codes.append(
                OntologyCode(
                    system="umls",
                    code=r.get("ui", ""),
                    display=r.get("name", ""),
                    source_api="umls",
                )
            )
        return codes

    def crosswalk(
        self,
        source: str,
        code: str,
        target: str,
    ) -> List[OntologyCode]:
        """Cross-vocabulary mapping (e.g. SNOMEDCT_US → ICD10CM)."""
        try:
            resp = self._requests.get(
                f"{_BASE_URL}/crosswalk/current/source/{source}/{code}",
                params={"apiKey": self._api_key, "targetSource": target},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("UMLS crosswalk failed: %s", exc)
            return []

        results = data.get("result", [])
        return [
            OntologyCode(
                system=target.lower(),
                code=r.get("ui", ""),
                display=r.get("name", ""),
                source_api="umls",
            )
            for r in results
        ]

    def get_concept(self, cui: str) -> Dict:
        """Get CUI details."""
        try:
            resp = self._requests.get(
                f"{_BASE_URL}/content/current/CUI/{cui}",
                params={"apiKey": self._api_key},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("result", {})
        except Exception as exc:
            logger.warning("UMLS concept lookup failed for %s: %s", cui, exc)
            return {}
