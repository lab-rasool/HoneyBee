"""
HoneyBee: Multimodal AI Framework for Cancer Research

Main module providing unified interface for multimodal data processing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Import processors
from .processors import ClinicalProcessor, RadiologyProcessor
from .processors.clinical import (
    ClinicalDocument,
    ClinicalEntity,
    ClinicalResult,
    DocumentIngester,
    EmbeddingEngine,
    NEREngine,
    OntologyCode,
    OntologyResolver,
    TimelineEvent,
    TimelineExtractor,
)


class HoneyBee:
    """
    Main HoneyBee interface for multimodal cancer research applications

    Example:
        >>> honeybee = HoneyBee()
        >>> embeddings = honeybee.generate_embeddings("Patient diagnosed with cancer", modality="clinical")
        >>> result = honeybee.process_clinical(text="Patient with lung cancer")
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize processors with appropriate configurations
        clinical_config = self.config.get("clinical", {})
        self.clinical_processor = ClinicalProcessor(config=clinical_config)

        # Radiology processor (lazy — created on first use)
        self._radiology_processor = None

    def generate_embeddings(
        self, data: Union[str, List[str], np.ndarray], modality: str = "clinical", **kwargs
    ) -> np.ndarray:
        """
        Generate embeddings for different data modalities

        Args:
            data: Input data (text/list of texts for clinical, array for other modalities)
            modality: Data modality ("clinical", "pathology", "molecular")
            **kwargs: Additional arguments passed to the modality-specific processor

        Returns:
            Generated embeddings as numpy array
        """
        if modality == "clinical":
            if isinstance(data, (str, list)):
                return self.clinical_processor.generate_embeddings(data, **kwargs)
            else:
                raise ValueError("Clinical modality requires text input (str or list of str)")
        elif modality == "radiology":
            if not isinstance(data, np.ndarray):
                raise ValueError("Radiology modality requires numpy array input")
            if self._radiology_processor is None:
                rad_config = self.config.get("radiology", {})
                self._radiology_processor = RadiologyProcessor(**rad_config)
            return self._radiology_processor.generate_embeddings(data, **kwargs)
        else:
            # For other modalities, return placeholder embeddings
            self.logger.warning(f"Modality {modality} not fully implemented, returning placeholder")
            return np.random.randn(1, 768)  # Placeholder embedding

    def integrate_embeddings(self, embeddings_list: List[np.ndarray]) -> np.ndarray:
        """
        Integrate embeddings from multiple modalities

        Args:
            embeddings_list: List of embedding arrays from different modalities

        Returns:
            Integrated multimodal embeddings
        """
        if not embeddings_list:
            raise ValueError("No embeddings provided")

        integrated = np.concatenate(embeddings_list, axis=-1)

        self.logger.info(
            f"Integrated {len(embeddings_list)} modalities into shape {integrated.shape}"
        )
        return integrated

    def predict_survival(self, embeddings: np.ndarray) -> Dict:
        """
        Predict survival outcomes from multimodal embeddings

        Args:
            embeddings: Integrated multimodal embeddings

        Returns:
            Survival prediction results
        """
        self.logger.info("Generating survival predictions (placeholder)")

        return {
            "survival_probability": 0.75,
            "risk_score": 0.25,
            "confidence": 0.85,
            "time_to_event": 365,
        }

    def process_clinical(
        self,
        document_path: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        save_output: bool = False,
    ) -> ClinicalResult:
        """
        Process clinical documents or text.

        Returns a ClinicalResult dataclass with entities, timeline, etc.
        """
        if document_path is not None:
            return self.clinical_processor.process(document_path)
        elif text is not None:
            return self.clinical_processor.process_text(text)
        else:
            raise ValueError("Either document_path or text must be provided")

    def process_radiology(
        self,
        dicom_path: Optional[Union[str, Path]] = None,
        nifti_path: Optional[Union[str, Path]] = None,
        image: Optional[np.ndarray] = None,
        preprocess: bool = True,
        generate_embeddings: bool = False,
        **kwargs,
    ) -> Dict:
        """Process radiology images."""
        if self._radiology_processor is None:
            rad_config = self.config.get("radiology", {})
            self._radiology_processor = RadiologyProcessor(**rad_config)

        result = {}

        if dicom_path is not None:
            img, metadata = self._radiology_processor.load_dicom(str(dicom_path))
            result["image"] = img
            result["metadata"] = metadata
        elif nifti_path is not None:
            img, metadata = self._radiology_processor.load_nifti(str(nifti_path))
            result["image"] = img
            result["metadata"] = metadata
        elif image is not None:
            img = image
            metadata = None
            result["image"] = img
            result["metadata"] = metadata
        else:
            raise ValueError("One of dicom_path, nifti_path, or image must be provided")

        if preprocess and metadata is not None:
            img = self._radiology_processor.preprocess(img, metadata, **kwargs)
            result["image"] = img

        if generate_embeddings:
            emb = self._radiology_processor.generate_embeddings(img, **kwargs)
            result["embeddings"] = emb

        return result

    def process_clinical_batch(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*",
        save_output: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[ClinicalResult]:
        """Process multiple clinical documents in batch."""
        return self.clinical_processor.process_batch(
            input_dir=input_dir,
            file_pattern=file_pattern,
            save_output=save_output,
            output_dir=output_dir,
        )


# Re-export commonly used classes

__all__ = [
    "HoneyBee",
    "ClinicalProcessor",
    "RadiologyProcessor",
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
