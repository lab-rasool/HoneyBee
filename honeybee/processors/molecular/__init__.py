"""HoneyBee Molecular Processor subpackage.

Multi-omics analysis pillar backed by SeNMo (Waqas et al. 2025).
"""

from .processor import MolecularProcessor
from .result import MolecularResult

__all__ = [
    "MolecularProcessor",
    "MolecularResult",
]