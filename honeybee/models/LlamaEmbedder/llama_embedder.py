"""
Llama-based embedder for clinical and pathology text processing

This module extends HoneyBee's embedding capabilities with Llama models,
providing enhanced performance for long clinical documents and few-shot scenarios.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Union
import numpy as np


class LlamaEmbedder:
    """
    Llama model wrapper for generating embeddings from clinical text.
    
    Supports various Llama model sizes and provides optimized processing
    for medical domain text.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: Optional[str] = None,
        max_length: int = 2048,
        use_flash_attention: bool = True,
        load_in_8bit: bool = False
    ):
        """
        Initialize Llama embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (cuda/cpu)
            max_length: Maximum sequence length
            use_flash_attention: Whether to use Flash Attention 2
            load_in_8bit: Whether to load model in 8-bit precision
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        model_kwargs = {
            'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
            'device_map': 'auto' if self.device == 'cuda' else None
        }
        
        if use_flash_attention and self.device == 'cuda':
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            
        if load_in_8bit and self.device == 'cuda':
            model_kwargs['load_in_8bit'] = True
            
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        
    def _chunk_text(self, text: str, chunk_size: int = 1024, overlap: int = 128) -> List[str]:
        """
        Split long text into overlapping chunks for processing.
        
        Args:
            text: Input text
            chunk_size: Size of each chunk in tokens
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        # Tokenize to get accurate chunk boundaries
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            start += chunk_size - overlap
            
        return chunks
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 4,
        pooling_strategy: str = 'mean',
        handle_long_texts: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for clinical texts.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            pooling_strategy: How to pool token embeddings ('mean', 'max', 'cls')
            handle_long_texts: Whether to chunk and aggregate long texts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                # Handle long texts by chunking
                if handle_long_texts and len(self.tokenizer.encode(text)) > self.max_length:
                    chunks = self._chunk_text(text, chunk_size=self.max_length - 50)
                    chunk_embeddings = []
                    
                    for chunk in chunks:
                        emb = self._encode_text(chunk, pooling_strategy)
                        chunk_embeddings.append(emb)
                    
                    # Aggregate chunk embeddings (mean pooling)
                    text_embedding = np.mean(chunk_embeddings, axis=0)
                else:
                    text_embedding = self._encode_text(text, pooling_strategy)
                    
                batch_embeddings.append(text_embedding)
            
            all_embeddings.extend(batch_embeddings)
            
        return np.vstack(all_embeddings)
    
    def _encode_text(self, text: str, pooling_strategy: str) -> np.ndarray:
        """Encode a single text to embedding."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        if self.device == 'cuda':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Apply pooling strategy
        if pooling_strategy == 'cls':
            embedding = outputs.last_hidden_state[:, 0, :]
        elif pooling_strategy == 'max':
            embedding = torch.max(outputs.last_hidden_state, dim=1)[0]
        else:  # mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
        return embedding.squeeze().cpu().numpy()
    
    def extract_clinical_entities(self, text: str, use_few_shot: bool = True) -> Dict[str, List[str]]:
        """
        Extract clinical entities using Llama's few-shot capabilities.
        
        Args:
            text: Clinical text
            use_few_shot: Whether to use few-shot prompting
            
        Returns:
            Dictionary of extracted entities
        """
        if use_few_shot:
            prompt = self._create_few_shot_ner_prompt(text)
        else:
            prompt = f"Extract medical entities from this text:\n{text}"
            
        # This is a placeholder - in practice, you would use the generative
        # capabilities of Llama for entity extraction
        entities = {
            'medications': [],
            'diagnoses': [],
            'procedures': [],
            'lab_values': [],
            'anatomical_sites': []
        }
        
        # Simple pattern matching as demonstration
        # In practice, use Llama's generation capabilities
        text_lower = text.lower()
        
        # Medication patterns
        if any(term in text_lower for term in ['mg', 'ml', 'daily', 'bid', 'tid']):
            entities['medications'].append('detected')
            
        # Diagnosis patterns  
        if any(term in text_lower for term in ['cancer', 'carcinoma', 'tumor', 'metastasis']):
            entities['diagnoses'].append('detected')
            
        return entities
    
    def _create_few_shot_ner_prompt(self, text: str) -> str:
        """Create few-shot prompt for clinical NER."""
        few_shot_examples = """
Extract medical entities from the clinical text.

Example 1:
Text: "Patient started on metformin 500mg BID for diabetes mellitus type 2."
Medications: metformin 500mg BID
Diagnoses: diabetes mellitus type 2
Procedures: none
Lab values: none

Example 2:
Text: "CT scan revealed 3cm mass in right upper lobe. Biopsy confirmed adenocarcinoma."
Medications: none
Diagnoses: adenocarcinoma, mass in right upper lobe
Procedures: CT scan, biopsy
Lab values: none

Now extract entities from:
Text: "{}"
""".format(text)
        
        return few_shot_examples


class ClinicalLlamaProcessor:
    """
    High-level processor for clinical text using Llama models.
    Integrates with HoneyBee's existing pipeline.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):
        self.embedder = LlamaEmbedder(model_name)
        
    def process_clinical_document(
        self,
        document: str,
        extract_entities: bool = True,
        generate_summary: bool = False
    ) -> Dict:
        """
        Process a clinical document comprehensively.
        
        Args:
            document: Clinical document text
            extract_entities: Whether to extract medical entities
            generate_summary: Whether to generate document summary
            
        Returns:
            Dictionary with embeddings and extracted information
        """
        results = {}
        
        # Generate embeddings
        embeddings = self.embedder.generate_embeddings(document)
        results['embeddings'] = embeddings
        
        # Extract entities if requested
        if extract_entities:
            entities = self.embedder.extract_clinical_entities(document)
            results['entities'] = entities
            
        # Generate summary if requested
        if generate_summary:
            # Placeholder for summary generation
            results['summary'] = "Clinical document processed with Llama"
            
        return results
    
    def process_pathology_report(self, report: str) -> Dict:
        """
        Specialized processing for pathology reports.
        
        Args:
            report: Pathology report text
            
        Returns:
            Dictionary with structured pathology information
        """
        # Generate embeddings
        embeddings = self.embedder.generate_embeddings(report)
        
        # Extract pathology-specific entities
        entities = self.embedder.extract_clinical_entities(report)
        
        # Additional pathology-specific processing
        pathology_info = {
            'embeddings': embeddings,
            'entities': entities,
            'cancer_grade': 'extracted_grade',  # Placeholder
            'tumor_size': 'extracted_size',      # Placeholder
            'margins': 'extracted_margins'       # Placeholder
        }
        
        return pathology_info