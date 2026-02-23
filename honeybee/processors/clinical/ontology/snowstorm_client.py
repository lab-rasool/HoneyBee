"""Snowstorm SNOMED CT Terminology Server client."""

from __future__ import annotations

import logging
import time
from typing import Dict, List

from ..types import OntologyCode

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct"


class SnowstormClient:
    """Thin wrapper around the SNOMED International Snowstorm REST API."""

    def __init__(
        self,
        base_url: str = _DEFAULT_URL,
        *,
        max_retries: int = 3,
        base_delay: float = 0.25,
        request_delay: float = 0.15,
    ):
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._request_delay = request_delay
        try:
            import requests  # noqa: F401

            self._requests = requests
        except ImportError:
            raise ImportError("Snowstorm client requires requests: pip install requests")

    def _request_with_retry(self, method: str, url: str, **kwargs) -> "requests.Response":
        """HTTP request with per-request throttle and exponential backoff on 429."""
        time.sleep(self._request_delay)
        kwargs.setdefault("timeout", 10)
        for attempt in range(self._max_retries + 1):
            resp = self._requests.request(method, url, **kwargs)
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp
            delay = self._base_delay * (2 ** attempt)
            logger.info(
                "Snowstorm 429 for %s, retrying in %.1fs (attempt %d/%d)",
                url, delay, attempt + 1, self._max_retries,
            )
            time.sleep(delay)
        resp.raise_for_status()
        return resp  # unreachable but satisfies type checker

    def search(self, term: str, max_results: int = 5) -> List[OntologyCode]:
        """Search for SNOMED CT concepts."""
        try:
            resp = self._request_with_retry(
                "GET",
                f"{self._base_url}/MAIN/concepts",
                params={"term": term, "limit": max_results, "activeFilter": True},
                headers={"Accept": "application/json"},
            )
            data = resp.json()
        except Exception as exc:
            logger.warning("Snowstorm search failed for %r: %s", term, exc)
            return []

        codes: List[OntologyCode] = []
        for item in data.get("items", []):
            codes.append(
                OntologyCode(
                    system="snomed_ct",
                    code=item.get("conceptId", ""),
                    display=item.get("fsn", {}).get("term", item.get("id", "")),
                    source_api="snowstorm",
                )
            )
        return codes

    def get_concept(self, sctid: str) -> Dict:
        """Get details for a SNOMED CT concept by ID."""
        try:
            resp = self._request_with_retry(
                "GET",
                f"{self._base_url}/MAIN/concepts/{sctid}",
                headers={"Accept": "application/json"},
            )
            return resp.json()
        except Exception as exc:
            logger.warning("Snowstorm concept lookup failed for %s: %s", sctid, exc)
            return {}

    def fhir_lookup(self, code: str) -> Dict:
        """FHIR CodeSystem/$lookup for a SNOMED CT code."""
        try:
            resp = self._request_with_retry(
                "GET",
                f"{self._base_url}/fhir/CodeSystem/$lookup",
                params={"system": "http://snomed.info/sct", "code": code},
                headers={"Accept": "application/fhir+json"},
            )
            return resp.json()
        except Exception as exc:
            logger.warning("Snowstorm FHIR lookup failed for %s: %s", code, exc)
            return {}
