"""
HoneyBee Processors Module

Provides unified interfaces for different data modality processors.
"""

from .clinical_processor import ClinicalProcessor

__all__ = ["ClinicalProcessor"]