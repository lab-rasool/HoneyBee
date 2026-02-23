"""
Timeline extraction — finds dates in clinical text and links them to entities.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .._text_utils import find_dates
from ..types import ClinicalDocument, ClinicalEntity, TimelineEvent

logger = logging.getLogger(__name__)


class TimelineExtractor:
    """Extract timeline events from clinical text and link them to entities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._reference_date = config.get("reference_date")
        self._prefer_dates_from = config.get("prefer_dates_from", "past")

    def extract(
        self,
        document: ClinicalDocument,
        entities: List[ClinicalEntity],
    ) -> List[TimelineEvent]:
        """Extract timeline events from *document* text, linking nearby entities."""
        text = document.text
        if not text or not text.strip():
            return []

        events: List[TimelineEvent] = []

        # Try dateparser.search first (richer parsing)
        try:
            from dateparser.search import search_dates

            settings = {"PREFER_DATES_FROM": self._prefer_dates_from}
            if self._reference_date:
                settings["RELATIVE_BASE"] = self._reference_date
            found = search_dates(text, settings=settings) or []
            for date_text, dt in found:
                sentence = self._enclosing_sentence(text, date_text)
                related = self._find_nearby_entities(text, date_text, entities)
                events.append(
                    TimelineEvent(
                        date=dt,
                        date_text=date_text,
                        sentence=sentence,
                        related_entities=related,
                    )
                )
        except ImportError:
            logger.debug("dateparser not installed, using string-scan fallback")
            events = self._scan_fallback(text, entities)

        # Sort chronologically (None dates go last)
        events.sort(key=lambda e: e.date or datetime.max)
        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scan_fallback(
        self, text: str, entities: List[ClinicalEntity]
    ) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []
        for _start, _end, date_text in find_dates(text):
            dt = self._try_parse(date_text)
            sentence = self._enclosing_sentence(text, date_text)
            related = self._find_nearby_entities(text, date_text, entities)
            events.append(
                TimelineEvent(
                    date=dt,
                    date_text=date_text,
                    sentence=sentence,
                    related_entities=related,
                )
            )
        return events

    @staticmethod
    def _try_parse(text: str) -> Optional[datetime]:
        try:
            import dateutil.parser

            return dateutil.parser.parse(text, fuzzy=True)
        except Exception:
            return None

    @staticmethod
    def _enclosing_sentence(text: str, fragment: str) -> str:
        idx = text.find(fragment)
        if idx < 0:
            return ""
        # Walk backwards to sentence start
        start = max(0, text.rfind(".", 0, idx) + 1)
        # Walk forward to sentence end
        end_dot = text.find(".", idx + len(fragment))
        end = end_dot + 1 if end_dot >= 0 else len(text)
        return text[start:end].strip()

    @staticmethod
    def _find_nearby_entities(
        text: str,
        date_text: str,
        entities: List[ClinicalEntity],
        window: int = 150,
    ) -> List[int]:
        """Return indices of entities within *window* chars of *date_text*."""
        idx = text.find(date_text)
        if idx < 0:
            return []
        date_start = idx
        date_end = idx + len(date_text)
        related: List[int] = []
        for i, ent in enumerate(entities):
            if ent.type == "temporal":
                continue
            # Check if entity is within window of the date mention
            if abs(ent.start - date_end) <= window or abs(date_start - ent.end) <= window:
                related.append(i)
        return related
