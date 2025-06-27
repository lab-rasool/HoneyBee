"""
Clinical Processor for HoneyBee - Fixed Version

Complete implementation for processing clinical oncology data with OCR, tokenization,
and entity recognition capabilities. This version includes proper error handling and
default values for configuration options.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

# External dependencies
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from PyPDF2 import PdfReader
import pdf2image
import cv2
import torch
from transformers import AutoTokenizer
import nltk
import dateutil.parser
import xml.etree.ElementTree as ET

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except:
        pass

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        pass

from nltk.tokenize import sent_tokenize, word_tokenize

# Import PDF loader from HoneyBee
try:
    from ..loaders import PDF
except ImportError:
    PDF = None

# Constants
SUPPORTED_IMAGE_FORMATS = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
SUPPORTED_EHR_FORMATS = [".xml", ".json", ".csv", ".xlsx"]
CLINICAL_DOCUMENT_TYPES = [
    "operative_report",
    "pathology_report",
    "consultation_note",
    "progress_note",
    "discharge_summary",
]

BIOMEDICAL_MODELS = {
    "bioclinicalbert": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "max_length": 512,
    },
    "pubmedbert": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "max_length": 512,
    },
    "gatortron": {
        "model_name": "UFNLP/gatortron-base",
        "max_length": 512,
    },
    "clinicalt5": {
        "model_name": "razent/SciFive-base-PMC",
        "max_length": 512,
    }
}

# Cancer-specific patterns
CANCER_PATTERNS = {
    "tumor_type": r"(?i)(adenocarcinoma|carcinoma|sarcoma|lymphoma|leukemia|melanoma|glioma|mesothelioma)",
    "tumor_location": r"(?i)(lung|breast|colon|prostate|pancreatic|liver|brain|ovarian|stomach|kidney)",
    "tumor_grade": r"(?i)(grade\s*[1-4]|grade\s*I{1,3}V?|well[\s-]differentiated|moderately[\s-]differentiated|poorly[\s-]differentiated)",
    "tumor_size": r"(\d+\.?\d*)\s*(cm|mm|centimeter|millimeter)",
    "tnm_stage": r"(?i)(T[0-4][a-c]?|N[0-3][a-c]?|M[0-1][a-c]?)",
    "stage_group": r"(?i)(stage\s*[0-4]|stage\s*I{1,3}V?[A-C]?)",
    "biomarker_status": r"(?i)(ER|PR|HER2|EGFR|ALK|ROS1|BRAF|KRAS|PD-L1)\s*(positive|negative|mutant|wild[\s-]?type|\+|-|\d+%)",
    "treatment_response": r"(?i)(complete response|partial response|stable disease|progressive disease|CR|PR|SD|PD)"
}

# Medical abbreviation expansions
MEDICAL_ABBREVIATIONS = {
    "ca": "cancer",
    "mets": "metastasis",
    "chemo": "chemotherapy",
    "rads": "radiation",
    "bx": "biopsy",
    "sx": "surgery",
    "dx": "diagnosis",
    "tx": "treatment",
    "hx": "history",
    "rx": "prescription"
}

# Ontology mappings (simplified for demonstration)
ONTOLOGY_MAPPINGS = {
    "snomed_ct": {
        "breast cancer": {"id": "254838004", "name": "Carcinoma of breast"},
        "lung cancer": {"id": "254637007", "name": "Non-small cell lung cancer"},
        "chemotherapy": {"id": "367336001", "name": "Chemotherapy"},
        "tamoxifen": {"id": "387207008", "name": "Tamoxifen"},
    },
    "rxnorm": {
        "tamoxifen": {"id": "10324", "name": "Tamoxifen"},
        "carboplatin": {"id": "1736854", "name": "Carboplatin"},
        "paclitaxel": {"id": "7053", "name": "Paclitaxel"},
    }
}


class ClinicalProcessor:
    """
    Main processor for clinical oncology data that integrates document processing,
    tokenization, and entity recognition.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Set defaults with safe nested dictionary access
        default_config = {
            "document_processor": {
                "use_ocr": True,
                "use_ehr": True,
                "confidence_threshold": 60,
                "preprocessing": True
            },
            "tokenization": {
                "model": "gatortron",
                "max_length": 512,
                "segment_strategy": "sentence",
                "long_document_strategy": "sliding_window",
                "stride": 128
            },
            "entity_recognition": {
                "use_rules": True,
                "use_patterns": True,
                "cancer_specific_extraction": True,
                "temporal_extraction": True,
                "abbreviation_expansion": True,
                "ontologies": ["snomed_ct", "rxnorm"]
            },
            "processing_pipeline": ["document", "tokenization", "entity_recognition"],
            "output": {
                "include_raw_text": True,
                "include_tokens": True,
                "include_entities": True,
                "include_document_structure": True,
                "include_temporal_timeline": True
            }
        }
        
        # Deep merge config with defaults
        self.config = self._deep_merge(default_config, self.config)
        
        # Initialize components
        self._init_components()
        
    def _deep_merge(self, default: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = default.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
        
    def _init_components(self):
        """Initialize processing components"""
        # Initialize tokenizer
        self.tokenizer = None
        self.embedder = None
        if "tokenization" in self.config["processing_pipeline"]:
            model_name = self.config.get("tokenization", {}).get("model", "gatortron")
            model_config = BIOMEDICAL_MODELS.get(model_name)
            if model_config:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
                    # Don't load the full model by default to save memory
                    # self.embedder = AutoModel.from_pretrained(model_config["model_name"])
                    self.model_max_length = min(
                        model_config["max_length"],
                        self.config.get("tokenization", {}).get("max_length", 512)
                    )
                except Exception as e:
                    self.logger.warning(f"Could not load model {model_config['model_name']}: {e}")
                    
        # Compile regex patterns
        self.compiled_patterns = {}
        for pattern_name, pattern in CANCER_PATTERNS.items():
            self.compiled_patterns[pattern_name] = re.compile(pattern)
            
        # Initialize medical terms for OCR verification
        self.medical_terms = set([
            "patient", "diagnosis", "treatment", "medication", "history",
            "cancer", "tumor", "carcinoma", "adenocarcinoma", "lymphoma",
            "metastasis", "stage", "grade", "biopsy", "surgery", "chemotherapy",
            "radiation", "therapy", "EGFR", "HER2", "ER", "PR", "PD-L1"
        ])
        
    def process(self, input_path: Union[str, Path], save_output: bool = False) -> Dict:
        """
        Process a clinical document through the full pipeline
        
        Args:
            input_path: Path to the clinical document
            save_output: Whether to save the output to a file
            
        Returns:
            Dictionary containing all processing results
        """
        input_path = Path(input_path)
        
        # Initialize result
        result = {
            "file_path": str(input_path),
            "file_name": input_path.name,
            "processing_timestamp": datetime.now().isoformat(),
            "processing_pipeline": self.config["processing_pipeline"]
        }
        
        try:
            # Step 1: Document Processing
            if "document" in self.config["processing_pipeline"]:
                self.logger.info(f"Processing document: {input_path}")
                document_result = self._process_document(input_path)
                
                if self.config.get("output", {}).get("include_raw_text", True):
                    result["text"] = document_result.get("text", document_result.get("full_text", ""))
                    
                result["document_processing"] = {
                    "method": document_result.get("extraction_method", "unknown"),
                    "confidence": document_result.get("avg_confidence", document_result.get("confidence", 0))
                }
                
                # Analyze document structure
                if self.config.get("output", {}).get("include_document_structure", True):
                    structure = self._analyze_document_structure(result.get("text", ""))
                    result["document_structure"] = structure
                    
            # Step 2: Tokenization
            if "tokenization" in self.config["processing_pipeline"] and result.get("text") and self.tokenizer:
                self.logger.info("Tokenizing text")
                tokenization_result = self._tokenize_text(
                    result["text"],
                    result.get("document_structure")
                )
                
                if self.config.get("output", {}).get("include_tokens", True):
                    result["tokenization"] = tokenization_result
                    
            # Step 3: Entity Recognition
            if "entity_recognition" in self.config["processing_pipeline"] and result.get("text"):
                self.logger.info("Extracting entities")
                entity_result = self._extract_entities(result["text"])
                
                if self.config.get("output", {}).get("include_entities", True):
                    result["entities"] = entity_result["entities"]
                    result["entity_relationships"] = entity_result["entity_relationships"]
                    
                # Extract temporal timeline
                if self.config.get("output", {}).get("include_temporal_timeline", True) and entity_result["entities"]:
                    timeline = self._extract_timeline(entity_result["entities"], result["text"])
                    result["temporal_timeline"] = timeline
                    
            # Generate embeddings if model is loaded
            if self.embedder and result.get("text"):
                embeddings = self._generate_embeddings(result["text"])
                if embeddings is not None:
                    result["embeddings"] = embeddings
                    
            # Save output if requested
            if save_output:
                output_path = input_path.with_suffix('.clinical.json')
                self._save_output(result, output_path)
                result["output_saved"] = str(output_path)
                
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {e}")
            result["error"] = str(e)
            
        return result
        
    def process_text(self, text: str, document_type: str = "unknown") -> Dict:
        """
        Process raw clinical text directly
        
        Args:
            text: Clinical text to process
            document_type: Type of clinical document
            
        Returns:
            Dictionary containing all processing results
        """
        # Initialize result
        result = {
            "text": text,
            "document_type": document_type,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Analyze structure
            if self.config.get("output", {}).get("include_document_structure", True):
                structure = self._analyze_document_structure(text)
                result["document_structure"] = structure
                
            # Tokenization
            if "tokenization" in self.config["processing_pipeline"] and self.tokenizer:
                tokenization_result = self._tokenize_text(text, result.get("document_structure"))
                if self.config.get("output", {}).get("include_tokens", True):
                    result["tokenization"] = tokenization_result
                    
            # Entity Recognition
            if "entity_recognition" in self.config["processing_pipeline"]:
                entity_result = self._extract_entities(text)
                
                if self.config.get("output", {}).get("include_entities", True):
                    result["entities"] = entity_result["entities"]
                    result["entity_relationships"] = entity_result["entity_relationships"]
                    
                # Timeline
                if self.config.get("output", {}).get("include_temporal_timeline", True) and entity_result["entities"]:
                    timeline = self._extract_timeline(entity_result["entities"], text)
                    result["temporal_timeline"] = timeline
                    
            # Generate embeddings
            if self.embedder:
                embeddings = self._generate_embeddings(text)
                if embeddings is not None:
                    result["embeddings"] = embeddings
                    
        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            result["error"] = str(e)
            
        return result
        
    def process_batch(self, input_dir: Union[str, Path], 
                     file_pattern: str = "*", 
                     save_output: bool = True,
                     output_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
        """Process multiple clinical documents in batch"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir
        
        # Find matching files
        files = list(input_dir.glob(file_pattern))
        self.logger.info(f"Found {len(files)} files to process")
        
        results = []
        for file_path in files:
            self.logger.info(f"Processing {file_path.name}")
            result = self.process(file_path, save_output=False)
            
            if save_output and not result.get("error"):
                output_path = output_dir / f"{file_path.stem}.clinical.json"
                self._save_output(result, output_path)
                result["output_saved"] = str(output_path)
                
            results.append(result)
            
        return results
        
    def _process_document(self, file_path: Path) -> Dict:
        """Process document based on file type"""
        suffix = file_path.suffix.lower()
        
        # Image formats (including PDF)
        if suffix in SUPPORTED_IMAGE_FORMATS:
            return self._process_image_document(file_path)
        # EHR formats
        elif suffix in SUPPORTED_EHR_FORMATS:
            return self._process_ehr_document(file_path)
        else:
            # Try to read as text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return {
                    "text": text,
                    "extraction_method": "direct_read"
                }
            except Exception as e:
                raise ValueError(f"Unsupported file format: {suffix}")
                
    def _process_image_document(self, file_path: Path) -> Dict:
        """Process image-based documents with OCR"""
        if file_path.suffix.lower() == ".pdf":
            return self._process_pdf(file_path)
        else:
            return self._process_image(file_path)
            
    def _process_pdf(self, file_path: Path) -> Dict:
        """Process PDF document"""
        result = {
            "file_path": str(file_path),
            "file_type": "pdf",
            "pages": [],
            "full_text": "",
            "confidence_scores": []
        }
        
        try:
            # Try direct text extraction first
            pdf_reader = PdfReader(str(file_path))
            direct_text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    direct_text += page_text + "\n"
                    
            # If direct extraction worked, use it
            if len(direct_text.strip()) > 100:
                result["full_text"] = direct_text
                result["extraction_method"] = "direct"
                return result
                
            # Otherwise, OCR the PDF
            if self.config.get("document_processor", {}).get("use_ocr", True):
                try:
                    images = pdf2image.convert_from_path(str(file_path))
                    
                    for i, image in enumerate(images):
                        page_result = self._ocr_image(image)
                        result["pages"].append(page_result)
                        result["full_text"] += page_result["text"] + "\n"
                        if page_result.get("confidence"):
                            result["confidence_scores"].append(page_result["confidence"])
                            
                    result["extraction_method"] = "ocr"
                    result["avg_confidence"] = np.mean(result["confidence_scores"]) if result["confidence_scores"] else 0
                except Exception as e:
                    self.logger.warning(f"OCR failed for {file_path}: {e}")
                    result["full_text"] = direct_text
                    result["extraction_method"] = "direct_partial"
                    
        except Exception as e:
            self.logger.error(f"Error processing PDF {file_path}: {e}")
            result["error"] = str(e)
            
        return result
        
    def _process_image(self, file_path: Path) -> Dict:
        """Process image file with OCR"""
        result = {
            "file_path": str(file_path),
            "file_type": "image",
            "text": "",
            "confidence": 0
        }
        
        try:
            image = Image.open(file_path)
            ocr_result = self._ocr_image(image)
            result.update(ocr_result)
            result["extraction_method"] = "ocr"
            
        except Exception as e:
            self.logger.error(f"Error processing image {file_path}: {e}")
            result["error"] = str(e)
            
        return result
        
    def _ocr_image(self, image: Image.Image) -> Dict:
        """Perform OCR on an image"""
        # Preprocess if enabled
        if self.config.get("document_processor", {}).get("preprocessing", True):
            image = self._preprocess_image(image)
            
        # Perform OCR
        try:
            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text with confidence filtering
            text_parts = []
            confidences = []
            threshold = self.config.get("document_processor", {}).get("confidence_threshold", 60)
            
            for i, conf in enumerate(ocr_data["conf"]):
                if int(conf) > threshold:
                    text = ocr_data["text"][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(int(conf))
                        
            # Join and post-process
            full_text = " ".join(text_parts)
            full_text = self._post_process_ocr_text(full_text)
            
            return {
                "text": full_text,
                "confidence": np.mean(confidences) if confidences else 0,
                "word_count": len(text_parts)
            }
            
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}
            
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
            
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Apply slight blur
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        # Convert to numpy for OpenCV operations
        img_array = np.array(image)
        
        # Apply threshold
        _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        img_array = cv2.medianBlur(img_array, 3)
        
        return Image.fromarray(img_array)
        
    def _post_process_ocr_text(self, text: str) -> str:
        """Post-process OCR text with medical terminology"""
        # Fix common OCR errors
        replacements = {
            "ml_": "mL",
            "mg_": "mg",
            "_": "",
            "  ": " ",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Expand abbreviations if enabled
        if self.config.get("entity_recognition", {}).get("abbreviation_expansion", True):
            words = text.split()
            expanded_words = []
            for word in words:
                lower_word = word.lower()
                if lower_word in MEDICAL_ABBREVIATIONS:
                    expanded_words.append(MEDICAL_ABBREVIATIONS[lower_word])
                else:
                    expanded_words.append(word)
            text = " ".join(expanded_words)
            
        return text
        
    def _process_ehr_document(self, file_path: Path) -> Dict:
        """Process structured EHR data"""
        suffix = file_path.suffix.lower()
        
        if suffix == ".json":
            return self._process_json_ehr(file_path)
        elif suffix == ".xml":
            return self._process_xml_ehr(file_path)
        elif suffix in [".csv", ".xlsx"]:
            return self._process_tabular_ehr(file_path)
            
    def _process_json_ehr(self, file_path: Path) -> Dict:
        """Process JSON EHR data"""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Convert to text
        text_parts = []
        self._flatten_json(data, text_parts)
        
        return {
            "text": "\n".join(text_parts),
            "extraction_method": "json_parse",
            "structured_data": data
        }
        
    def _flatten_json(self, obj, result: List[str], prefix: str = ""):
        """Flatten JSON to key-value pairs"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    self._flatten_json(value, result, new_key)
                else:
                    result.append(f"{new_key}: {value}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._flatten_json(item, result, f"{prefix}[{i}]")
        else:
            result.append(f"{prefix}: {obj}")
            
    def _process_xml_ehr(self, file_path: Path) -> Dict:
        """Process XML EHR data"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        text_parts = []
        self._parse_xml_element(root, text_parts)
        
        return {
            "text": "\n".join(text_parts),
            "extraction_method": "xml_parse"
        }
        
    def _parse_xml_element(self, element, result: List[str], prefix: str = ""):
        """Parse XML element recursively"""
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        new_prefix = f"{prefix}.{tag}" if prefix else tag
        
        if element.text and element.text.strip():
            result.append(f"{new_prefix}: {element.text.strip()}")
            
        for child in element:
            self._parse_xml_element(child, result, new_prefix)
            
    def _process_tabular_ehr(self, file_path: Path) -> Dict:
        """Process CSV/Excel EHR data"""
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        # Convert to text
        text_parts = []
        for idx, row in df.iterrows():
            text_parts.append(f"Record {idx + 1}:")
            for col, value in row.items():
                if pd.notna(value):
                    text_parts.append(f"  {col}: {value}")
                    
        return {
            "text": "\n".join(text_parts),
            "extraction_method": "tabular_parse",
            "num_records": len(df)
        }
        
    def _analyze_document_structure(self, text: str) -> Dict:
        """Analyze document structure"""
        section_patterns = {
            "chief_complaint": r"(?i)(chief complaint|cc|presenting complaint)",
            "history_present_illness": r"(?i)(history of present illness|hpi|present illness)",
            "past_medical_history": r"(?i)(past medical history|pmh|medical history)",
            "medications": r"(?i)(medications|current medications|medication list)",
            "allergies": r"(?i)(allergies|drug allergies|allergy)",
            "physical_exam": r"(?i)(physical exam|physical examination|pe)",
            "assessment_plan": r"(?i)(assessment and plan|a&p|assessment|plan)",
            "laboratory": r"(?i)(laboratory|lab results|labs)",
            "imaging": r"(?i)(imaging|radiology|ct|mri|xray|x-ray)",
            "pathology": r"(?i)(pathology|biopsy|cytology)",
        }
        
        lines = text.split('\n')
        sections = {}
        current_section = "unstructured"
        current_content = []
        
        for line in lines:
            # Check if line matches section pattern
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    # Start new section
                    current_section = section_name
                    current_content = [line]
                    section_found = True
                    break
                    
            if not section_found and line.strip():
                current_content.append(line)
                
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        return {
            "sections": sections,
            "headers": list(sections.keys()),
            "num_sections": len(sections)
        }
        
    def _tokenize_text(self, text: str, document_structure: Dict = None) -> Dict:
        """Tokenize clinical text"""
        if not self.tokenizer:
            return {"error": "Tokenizer not initialized"}
            
        # Clean text
        text = self._clean_text(text)
        
        # Segment text
        segments = self._segment_text(text)
        
        # Check if long document handling needed
        if self._needs_long_document_handling(segments):
            return self._handle_long_document(segments)
        else:
            return self._tokenize_segments(segments)
            
    def _clean_text(self, text: str) -> str:
        """Clean clinical text"""
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Handle clinical abbreviations
        abbreviations = {
            "pt": "patient",
            "hx": "history",
            "dx": "diagnosis",
            "tx": "treatment",
            "rx": "prescription"
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(f" {abbr} ", f" {expansion} ")
            
        return text
        
    def _segment_text(self, text: str) -> List[str]:
        """Segment text based on strategy"""
        strategy = self.config.get("tokenization", {}).get("segment_strategy", "sentence")
        
        if strategy == "sentence":
            try:
                return sent_tokenize(text)
            except:
                # Fallback to simple splitting
                return text.split('. ')
        elif strategy == "paragraph":
            return [p.strip() for p in text.split('\n\n') if p.strip()]
        elif strategy == "fixed":
            try:
                words = word_tokenize(text)
            except:
                words = text.split()
            chunk_size = 100
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunks.append(' '.join(words[i:i + chunk_size]))
            return chunks
        else:
            return [text]
            
    def _needs_long_document_handling(self, segments: List[str]) -> bool:
        """Check if document needs special handling"""
        # Quick check
        for segment in segments[:3]:
            tokens = self.tokenizer.tokenize(segment)
            if len(tokens) > self.model_max_length - 2:
                return True
                
        # Total length check
        total_text = ' '.join(segments)
        tokens = self.tokenizer.tokenize(total_text)
        return len(tokens) > self.model_max_length - 2
        
    def _handle_long_document(self, segments: List[str]) -> Dict:
        """Handle long documents"""
        strategy = self.config.get("tokenization", {}).get("long_document_strategy", "sliding_window")
        
        if strategy == "sliding_window":
            return self._sliding_window_tokenization(segments)
        elif strategy == "hierarchical":
            return self._hierarchical_tokenization(segments)
        elif strategy == "important_segments":
            return self._important_segments_tokenization(segments)
        else:
            return self._tokenize_segments(segments)
            
    def _sliding_window_tokenization(self, segments: List[str]) -> Dict:
        """Tokenize with sliding window"""
        full_text = ' '.join(segments)
        
        # Tokenize full text
        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True
        )
        
        all_input_ids = full_encoding['input_ids']
        stride = self.config.get("tokenization", {}).get("stride", 128)
        max_length = self.model_max_length - 2
        
        windows = []
        for i in range(0, len(all_input_ids), max_length - stride):
            end_idx = min(i + max_length, len(all_input_ids))
            
            # Get window
            window_ids = [self.tokenizer.cls_token_id] + all_input_ids[i:end_idx] + [self.tokenizer.sep_token_id]
            window_mask = [1] * len(window_ids)
            
            # Pad
            padding_length = self.model_max_length - len(window_ids)
            if padding_length > 0:
                window_ids += [self.tokenizer.pad_token_id] * padding_length
                window_mask += [0] * padding_length
                
            windows.append({
                "input_ids": window_ids,
                "attention_mask": window_mask,
                "start_token_idx": i,
                "end_token_idx": end_idx
            })
            
            if end_idx >= len(all_input_ids):
                break
                
        return {
            "input_ids": [w["input_ids"] for w in windows],
            "attention_mask": [w["attention_mask"] for w in windows],
            "num_windows": len(windows),
            "tokenization_strategy": "sliding_window"
        }
        
    def _hierarchical_tokenization(self, segments: List[str]) -> Dict:
        """Tokenize preserving hierarchy"""
        results = {
            "sections": {},
            "tokenization_strategy": "hierarchical"
        }
        
        for i, segment in enumerate(segments):
            section_name = f"segment_{i}"
            section_result = self._tokenize_single_text(segment)
            results["sections"][section_name] = section_result
            
        return results
        
    def _important_segments_tokenization(self, segments: List[str]) -> Dict:
        """Tokenize focusing on important segments"""
        # Score segments by clinical term density
        clinical_terms = {
            "diagnosis", "cancer", "tumor", "stage", "grade", "treatment",
            "chemotherapy", "radiation", "surgery", "metastasis", "response"
        }
        
        segment_scores = []
        for segment in segments:
            lower_segment = segment.lower()
            score = sum(1 for term in clinical_terms if term in lower_segment)
            segment_scores.append((score, segment))
            
        # Sort by importance
        segment_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Select top segments
        selected_segments = []
        total_tokens = 0
        
        for score, segment in segment_scores:
            tokens = self.tokenizer.tokenize(segment)
            if total_tokens + len(tokens) < self.model_max_length - 100:
                selected_segments.append(segment)
                total_tokens += len(tokens)
            else:
                break
                
        # Tokenize
        combined_text = ' '.join(selected_segments)
        result = self._tokenize_single_text(combined_text)
        result["tokenization_strategy"] = "important_segments"
        result["num_segments_selected"] = len(selected_segments)
        result["num_segments_total"] = len(segments)
        
        return result
        
    def _tokenize_segments(self, segments: List[str]) -> Dict:
        """Tokenize segments normally"""
        combined_text = ' '.join(segments)
        return self._tokenize_single_text(combined_text)
        
    def _tokenize_single_text(self, text: str) -> Dict:
        """Tokenize a single text"""
        encoding = self.tokenizer(
            text,
            max_length=self.model_max_length,
            truncation=True,
            padding=True,
            return_tensors="np"
        )
        
        return {
            "input_ids": encoding["input_ids"][0].tolist(),
            "attention_mask": encoding["attention_mask"][0].tolist(),
            "num_tokens": int(np.sum(encoding["attention_mask"])),
            "tokenizer_name": self.config.get("tokenization", {}).get("model", "unknown")
        }
        
    def _extract_entities(self, text: str) -> Dict:
        """Extract entities from text"""
        entities = []
        
        # Rule-based extraction
        if self.config.get("entity_recognition", {}).get("use_rules", True):
            rule_entities = self._extract_rule_based_entities(text)
            entities.extend(rule_entities)
            
        # Pattern-based extraction
        if self.config.get("entity_recognition", {}).get("use_patterns", True):
            pattern_entities = self._extract_pattern_based_entities(text)
            entities.extend(pattern_entities)
            
        # Cancer-specific extraction
        if self.config.get("entity_recognition", {}).get("cancer_specific_extraction", True):
            cancer_entities = self._extract_cancer_specific_entities(text)
            entities.extend(cancer_entities)
            
        # Temporal extraction
        if self.config.get("entity_recognition", {}).get("temporal_extraction", True):
            temporal_entities = self._extract_temporal_entities(text)
            entities.extend(temporal_entities)
            
        # Merge and normalize
        entities = self._merge_entities(entities)
        
        if self.config.get("entity_recognition", {}).get("ontologies", []):
            self._normalize_entities(entities)
            
        # Extract relationships
        entity_relationships = self._extract_entity_relationships(entities, text)
        
        return {
            "entities": entities,
            "entity_relationships": entity_relationships,
            "num_entities": len(entities)
        }
        
    def _extract_rule_based_entities(self, text: str) -> List[Dict]:
        """Extract entities using rules"""
        entities = []
        
        # Medication patterns
        med_pattern = r"(?i)(\w+)\s+(\d+)\s*(mg|mcg|g|ml|units?)\s*(daily|bid|tid|qid|prn|po|iv|im|sq)"
        for match in re.finditer(med_pattern, text):
            entities.append({
                "text": match.group(0),
                "type": "medication",
                "start": match.start(),
                "end": match.end(),
                "properties": {
                    "drug_name": match.group(1),
                    "dose": match.group(2),
                    "unit": match.group(3),
                    "frequency": match.group(4),
                    "source": "rule-based"
                }
            })
            
        # Lab values
        lab_pattern = r"(\w+[\s\w]*?):\s*(\d+\.?\d*)\s*([a-zA-Z/%]+)"
        for match in re.finditer(lab_pattern, text):
            entities.append({
                "text": match.group(0),
                "type": "measurement",
                "start": match.start(),
                "end": match.end(),
                "properties": {
                    "test_name": match.group(1).strip(),
                    "value": match.group(2),
                    "unit": match.group(3),
                    "source": "rule-based"
                }
            })
            
        return entities
        
    def _extract_pattern_based_entities(self, text: str) -> List[Dict]:
        """Extract entities using patterns"""
        entities = []
        
        # Condition patterns
        condition_pattern = r"(?i)(diagnosed with|history of|presents with|suffering from)\s+([a-zA-Z\s]+?)(?=\.|,|\s+and|\s+with)"
        for match in re.finditer(condition_pattern, text):
            entities.append({
                "text": match.group(2).strip(),
                "type": "condition",
                "start": match.start(2),
                "end": match.end(2),
                "properties": {
                    "context": match.group(1),
                    "source": "pattern-based"
                }
            })
            
        return entities
        
    def _extract_cancer_specific_entities(self, text: str) -> List[Dict]:
        """Extract cancer-specific entities"""
        entities = []
        
        for pattern_name, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                entity_type = self._get_entity_type_from_pattern(pattern_name)
                
                entity = {
                    "text": match.group(0),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "properties": {
                        "pattern": pattern_name,
                        "source": "cancer-specific"
                    }
                }
                
                # Add specific properties
                if pattern_name == "tumor_size":
                    entity["properties"]["size"] = match.group(1)
                    entity["properties"]["unit"] = match.group(2)
                elif pattern_name == "biomarker_status":
                    entity["properties"]["biomarker"] = match.group(1)
                    entity["properties"]["status"] = match.group(2) if match.lastindex >= 2 else "unknown"
                    
                entities.append(entity)
                
        return entities
        
    def _extract_temporal_entities(self, text: str) -> List[Dict]:
        """Extract temporal entities"""
        entities = []
        
        # Date patterns
        date_patterns = [
            r"\d{1,2}/\d{1,2}/\d{2,4}",
            r"\d{1,2}-\d{1,2}-\d{2,4}",
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    parsed_date = dateutil.parser.parse(match.group(0))
                    entities.append({
                        "text": match.group(0),
                        "type": "temporal",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {
                            "temporal_type": "date",
                            "normalized_date": parsed_date.isoformat(),
                            "source": "temporal"
                        }
                    })
                except:
                    pass
                    
        return entities
        
    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge overlapping entities"""
        if not entities:
            return []
            
        # Sort by start position
        entities.sort(key=lambda x: (x["start"], -x["end"]))
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            # Check overlap
            if entity["start"] < current["end"]:
                # Keep longer or more specific
                if entity["end"] > current["end"] or self._is_more_specific(entity["type"], current["type"]):
                    current = entity
            else:
                merged.append(current)
                current = entity
                
        merged.append(current)
        
        return merged
        
    def _is_more_specific(self, type1: str, type2: str) -> bool:
        """Check if type1 is more specific than type2"""
        specificity_order = [
            "tumor", "biomarker", "staging", "response",
            "medication", "dosage", "procedure",
            "condition", "measurement", "anatomy",
            "temporal"
        ]
        
        try:
            return specificity_order.index(type1) < specificity_order.index(type2)
        except ValueError:
            return False
            
    def _normalize_entities(self, entities: List[Dict]):
        """Normalize entities to ontologies"""
        for entity in entities:
            normalized_text = entity["text"].lower().strip()
            
            # Look up in ontologies
            ontology_links = []
            for ontology in self.config.get("entity_recognition", {}).get("ontologies", []):
                if ontology in ONTOLOGY_MAPPINGS:
                    mapping = ONTOLOGY_MAPPINGS[ontology].get(normalized_text)
                    if mapping:
                        ontology_links.append({
                            "ontology": ontology,
                            "concept_id": mapping["id"],
                            "concept_name": mapping["name"]
                        })
                        
            if ontology_links:
                entity["properties"]["ontology_links"] = ontology_links
                
    def _extract_entity_relationships(self, entities: List[Dict], text: str) -> List[Dict]:
        """Extract relationships between entities"""
        relationships = []
        
        # Proximity-based relationships
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check proximity
                if entity2["start"] - entity1["end"] < 50:
                    rel_type = self._determine_relationship(entity1, entity2)
                    if rel_type:
                        relationships.append({
                            "source": i,
                            "target": j,
                            "type": rel_type
                        })
                        
        return relationships
        
    def _determine_relationship(self, entity1: Dict, entity2: Dict) -> Optional[str]:
        """Determine relationship type"""
        # Medication-dosage
        if entity1["type"] == "medication" and entity2["type"] == "dosage":
            return "has_dosage"
            
        # Condition-treatment
        if entity1["type"] == "condition" and entity2["type"] in ["medication", "procedure"]:
            return "treated_with"
            
        # Tumor-staging
        if entity1["type"] == "tumor" and entity2["type"] == "staging":
            return "has_stage"
            
        return None
        
    def _get_entity_type_from_pattern(self, pattern_name: str) -> str:
        """Map pattern to entity type"""
        mapping = {
            "tumor_type": "tumor",
            "tumor_location": "tumor",
            "tumor_grade": "staging",
            "tumor_size": "measurement",
            "tnm_stage": "staging",
            "stage_group": "staging",
            "biomarker_status": "biomarker",
            "treatment_response": "response"
        }
        
        return mapping.get(pattern_name, "condition")
        
    def _extract_timeline(self, entities: List[Dict], text: str) -> List[Dict]:
        """Extract temporal timeline"""
        timeline_events = []
        
        # Get temporal entities
        temporal_entities = [e for e in entities if e["type"] == "temporal"]
        
        # Find related entities
        for temporal in temporal_entities:
            related_entities = []
            
            for i, entity in enumerate(entities):
                if entity["type"] != "temporal":
                    # Check proximity
                    if abs(entity["start"] - temporal["start"]) < 100:
                        related_entities.append(i)
                        
            event = {
                "temporal_text": temporal["text"],
                "temporal_type": temporal["properties"].get("temporal_type", "unknown"),
                "normalized_date": temporal["properties"].get("normalized_date"),
                "related_entities": related_entities
            }
            
            timeline_events.append(event)
            
        # Sort by date
        timeline_events.sort(key=lambda x: x.get("normalized_date", "9999"))
        
        return timeline_events
        
    def _generate_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Generate embeddings for clinical text"""
        if not self.embedder or not self.tokenizer:
            return None
            
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=self.model_max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embedder(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                
            return embeddings.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return None
            
    def _save_output(self, result: Dict, output_path: Path):
        """Save results to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            def convert_arrays(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_arrays(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_arrays(v) for v in obj]
                return obj
                
            result_serializable = convert_arrays(result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_serializable, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved output to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving output: {e}")
            
    def get_summary_statistics(self, result: Dict) -> Dict:
        """Get summary statistics"""
        stats = {
            "text_length": len(result.get("text", "")),
            "num_entities": len(result.get("entities", [])),
            "num_relationships": len(result.get("entity_relationships", [])),
            "num_timeline_events": len(result.get("temporal_timeline", [])),
            "entity_types": {}
        }
        
        # Count by type
        for entity in result.get("entities", []):
            entity_type = entity["type"]
            stats["entity_types"][entity_type] = stats["entity_types"].get(entity_type, 0) + 1
            
        return stats