"""Abstract base class for NER backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..types import ClinicalEntity


class NERBackend(ABC):
    """Interface that every NER backend must implement."""

    @abstractmethod
    def extract(self, text: str) -> List[ClinicalEntity]:
        """Extract clinical entities from *text*."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (e.g. ``"rule_based"``)."""
