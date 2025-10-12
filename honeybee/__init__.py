"""
HoneyBee: Multimodal AI Framework for Cancer Research

Main module providing unified interface for multimodal data processing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np

# Import processors
from .processors import ClinicalProcessor

class HoneyBee:
    """
    Main HoneyBee interface for multimodal cancer research applications

    Example:
        >>> honeybee = HoneyBee()
        >>> embeddings = honeybee.generate_embeddings("Patient diagnosed with cancer", modality="clinical")
        >>> result = honeybee.process_clinical(document_path="report.pdf")
    """

    def __init__(self, config: Dict = None):
        """
        Initialize HoneyBee framework

        Args:
            config: Configuration dictionary with processor-specific settings.
                   Example:
                   {
                       "clinical": {
                           "tokenization": {"model": "gatortron"},
                           "entity_recognition": {"use_rules": True}
                       }
                   }
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize processors with appropriate configurations
        clinical_config = self.config.get("clinical", {})
        self.clinical_processor = ClinicalProcessor(config=clinical_config)

    def generate_embeddings(
        self,
        data: Union[str, List[str], np.ndarray],
        modality: str = "clinical",
        **kwargs
    ) -> np.ndarray:
        """
        Generate embeddings for different data modalities

        Args:
            data: Input data (text/list of texts for clinical, array for other modalities)
            modality: Data modality ("clinical", "pathology", "radiology", "molecular")
            **kwargs: Additional arguments passed to the modality-specific processor
                     For clinical modality:
                         - model_name: Biomedical model to use (gatortron, pubmedbert, etc.)
                         - pooling_method: How to pool embeddings (mean, cls, max, pooler_output)
                         - batch_size: Batch size for processing

        Returns:
            Generated embeddings as numpy array

        Example:
            >>> honeybee = HoneyBee()
            >>> # Single text
            >>> emb = honeybee.generate_embeddings("Patient with lung cancer", modality="clinical")
            >>> # Multiple texts
            >>> texts = ["Text 1", "Text 2"]
            >>> emb = honeybee.generate_embeddings(texts, modality="clinical", model_name="gatortron")
        """
        if modality == "clinical":
            if isinstance(data, (str, list)):
                return self.clinical_processor.generate_embeddings(data, **kwargs)
            else:
                raise ValueError("Clinical modality requires text input (str or list of str)")
        else:
            # For other modalities, return placeholder embeddings
            # In a real implementation, you would have dedicated processors
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
        
        # Simple concatenation strategy
        # In a real implementation, you might use attention mechanisms or learned fusion
        integrated = np.concatenate(embeddings_list, axis=-1)
        
        self.logger.info(f"Integrated {len(embeddings_list)} modalities into shape {integrated.shape}")
        return integrated
    
    def predict_survival(self, embeddings: np.ndarray) -> Dict:
        """
        Predict survival outcomes from multimodal embeddings

        Args:
            embeddings: Integrated multimodal embeddings

        Returns:
            Survival prediction results
        """
        # Placeholder implementation
        # In a real system, this would use a trained survival model
        self.logger.info("Generating survival predictions (placeholder)")

        return {
            "survival_probability": 0.75,
            "risk_score": 0.25,
            "confidence": 0.85,
            "time_to_event": 365  # days
        }

    def process_clinical(
        self,
        document_path: Optional[Union[str, Path]] = None,
        text: Optional[str] = None,
        save_output: bool = False
    ) -> Dict:
        """
        Process clinical documents or text

        Convenience method that wraps ClinicalProcessor functionality.

        Args:
            document_path: Path to clinical document (PDF, image, EHR file)
            text: Raw clinical text (used if document_path is None)
            save_output: Whether to save processing results to file

        Returns:
            Processing results including entities, timeline, tokenization, etc.

        Example:
            >>> honeybee = HoneyBee()
            >>> # Process PDF
            >>> result = honeybee.process_clinical(document_path="report.pdf")
            >>> # Process text directly
            >>> result = honeybee.process_clinical(text="Patient with lung cancer")
        """
        if document_path is not None:
            return self.clinical_processor.process(document_path, save_output=save_output)
        elif text is not None:
            return self.clinical_processor.process_text(text)
        else:
            raise ValueError("Either document_path or text must be provided")

    def process_clinical_batch(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*",
        save_output: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict]:
        """
        Process multiple clinical documents in batch

        Convenience method that wraps ClinicalProcessor batch functionality.

        Args:
            input_dir: Directory containing clinical documents
            file_pattern: File pattern to match (e.g., "*.pdf")
            save_output: Whether to save processing results
            output_dir: Directory to save outputs (defaults to input_dir)

        Returns:
            List of processing results for each document

        Example:
            >>> honeybee = HoneyBee()
            >>> results = honeybee.process_clinical_batch(
            ...     input_dir="./clinical_reports",
            ...     file_pattern="*.pdf"
            ... )
        """
        return self.clinical_processor.process_batch(
            input_dir=input_dir,
            file_pattern=file_pattern,
            save_output=save_output,
            output_dir=output_dir
        )

# Re-export commonly used classes
from .processors import ClinicalProcessor

__all__ = ["HoneyBee", "ClinicalProcessor"]