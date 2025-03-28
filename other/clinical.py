"""
Clinical Oncology Data Processing System

This package implements comprehensive processing capabilities for oncology data,
including text extraction, document processing, tokenization and entity recognition.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# External dependencies
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertTokenizer,
    BertForTokenClassification,
    T5Tokenizer,
)
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize

from honeybee.loaders import PDF

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
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "gatortron": "UFNLP/gatortron-base",
    "clinicalt5": "healx/gpt-t5-clinical",
}
MEDICAL_ONTOLOGIES = {
    "snomed_ct": "SNOMEDCT_US",
    "rxnorm": "RXNORM",
    "loinc": "LNC",
    "icd_o_3": "ICD10CM",
}

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# MODULE 1: Text Extraction and Document Processing
# -----------------------------------------------------------------------------


class DocumentProcessor:
    """Base class for document processing capabilities"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def process(self, input_path: Union[str, Path]) -> Dict:
        """Process a document and return extracted data"""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        # Determine file type and process accordingly
        if input_path.suffix.lower() in SUPPORTED_IMAGE_FORMATS:
            return self._process_image_document(input_path)
        elif input_path.suffix.lower() in SUPPORTED_EHR_FORMATS:
            return self._process_ehr_document(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

    def _process_image_document(self, file_path: Path) -> Dict:
        """Process image-based documents like PDFs and scanned images"""
        self.logger.info(f"Processing image document: {file_path}")
        # This would be implemented by subclasses
        raise NotImplementedError

    def _process_ehr_document(self, file_path: Path) -> Dict:
        """Process structured EHR data exports"""
        self.logger.info(f"Processing EHR document: {file_path}")
        # This would be implemented by subclasses
        raise NotImplementedError

    def calculate_confidence_score(self, extracted_data: Dict) -> float:
        """Calculate confidence score for extraction quality"""
        # Implementation would depend on specific metrics
        return 0.0


class OCRProcessor(DocumentProcessor):
    """Handles OCR processing for image-based documents with medical terminology verification"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.config = {
            "tesseract_path": r"/usr/bin/tesseract",
            "lang": "eng+osd",
            "config": "--psm 1 --oem 3",
            "medical_dict_path": "path/to/medical_dictionary",
            "pdf_chunk_size": 512,
            "pdf_chunk_overlap": 10,
            **self.config,
        }

        # Initialize Tesseract with medical dictionaries
        if self.config.get("tesseract_path"):
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_path"]

        # Load medical dictionary for post-processing
        self.medical_terms = self._load_medical_dictionary()

        # Initialize PDF loader from HoneyBee
        self.pdf_loader = PDF(
            chunk_size=self.config["pdf_chunk_size"],
            chunk_overlap=self.config["pdf_chunk_overlap"],
        )

    def _load_medical_dictionary(self) -> set:
        """Load medical dictionary for term verification"""
        try:
            with open(self.config["medical_dict_path"], "r") as f:
                return set(line.strip() for line in f)
        except Exception as e:
            self.logger.warning(f"Could not load medical dictionary: {e}")
            return set()

    def _process_image_document(self, file_path: Path) -> Dict:
        """Process an image document using OCR pipeline"""
        # Check if the file is a PDF
        if file_path.suffix.lower() == ".pdf":
            return self._process_pdf_document(file_path)
        else:
            # Process other image types with standard OCR
            return self._process_standard_image(file_path)

    def _process_pdf_document(self, file_path: Path) -> Dict:
        """Process PDF document using HoneyBee PDF loader"""
        self.logger.info(f"Processing PDF document using HoneyBee loader: {file_path}")

        try:
            # Extract text using HoneyBee PDF loader
            full_text = self.pdf_loader.read(str(file_path))

            # Split text into pages (approximate by chunks)
            # This is an approximation as the PDF loader may not maintain page boundaries
            text_chunks = full_text.split("\n\n")
            pages = []

            # Create page entries from chunks
            for i, chunk in enumerate(text_chunks):
                if chunk.strip():  # Skip empty chunks
                    pages.append(
                        {
                            "page_num": i + 1,
                            "text": chunk,
                            "confidence": 0.9,  # Assume high confidence for PDF loader
                            "layout": {
                                "paragraphs": [{"text": chunk}],
                                "columns": [],
                                "tables": [],
                            },
                        }
                    )

            # Identify document type
            document_type = self._identify_document_type(pages)

            # Analyze document structure
            document_structure = self._analyze_document_structure(pages)

            return {
                "document_type": document_type,
                "pages": pages,
                "structure": document_structure,
                "full_text": full_text,
                "overall_confidence": 0.9,  # Assume high confidence for PDF loader
                "extraction_method": "honeybee_pdf_loader",
            }

        except Exception as e:
            self.logger.error(f"Error processing PDF with HoneyBee loader: {e}")
            self.logger.info("Falling back to standard OCR processing")
            # Fall back to standard image processing
            return self._process_standard_image(file_path)

    def _process_standard_image(self, file_path: Path) -> Dict:
        """Process an image document using standard OCR pipeline"""
        # 1. Load image
        if file_path.suffix.lower() == ".pdf":
            images = self._extract_images_from_pdf(file_path)
        else:
            images = [Image.open(file_path)]

        # 2. Process each image
        results = []
        for i, img in enumerate(images):
            self.logger.info(f"Processing image {i + 1}/{len(images)}")

            # Apply preprocessing
            preprocessed_img = self._preprocess_image(img)

            # Perform OCR
            ocr_text = pytesseract.image_to_string(
                preprocessed_img, lang=self.config["lang"], config=self.config["config"]
            )

            # Get layout information
            layout_info = self._analyze_layout(preprocessed_img)

            # Perform post-processing with medical terminology verification
            processed_text = self._post_process_text(ocr_text)

            # Calculate confidence
            confidence = self._calculate_ocr_confidence(ocr_text, preprocessed_img)

            results.append(
                {
                    "page_num": i + 1,
                    "text": processed_text,
                    "layout": layout_info,
                    "confidence": confidence,
                }
            )

        # 3. Integrate results
        document_structure = self._analyze_document_structure(results)

        # Check if we have any results before calculating average confidence
        overall_confidence = 0.0
        if results:
            overall_confidence = sum(r["confidence"] for r in results) / len(results)

        return {
            "document_type": self._identify_document_type(results),
            "pages": results,
            "structure": document_structure,
            "full_text": "\n\n".join(r["text"] for r in results),
            "overall_confidence": overall_confidence,
            "extraction_method": "standard_ocr",
        }

    def _extract_images_from_pdf(self, pdf_path: Path) -> List[Image.Image]:
        """Extract images from PDF files"""
        # Would use a library like pdf2image
        # Placeholder implementation
        return []

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """Preprocess image for improved OCR"""
        # Apply deskewing
        img = self._deskew_image(img)

        # Convert to grayscale
        if img.mode != "L":
            img = img.convert("L")

        # Apply noise reduction
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)

        # Apply binarization
        img = img.point(lambda p: p > 128 and 255)

        return img

    def _deskew_image(self, img: Image.Image) -> Image.Image:
        """Deskew image for better OCR results"""
        # Convert PIL Image to cv2 format
        cv_img = np.array(img)

        # Implementation would detect and correct skew angle
        # Placeholder for actual implementation

        # Convert back to PIL
        return Image.fromarray(cv_img)

    def _analyze_layout(self, img: Image.Image) -> Dict:
        """Analyze document layout"""
        # Use pytesseract to get bounding boxes and layout
        layout_data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT
        )

        # Extract columns, paragraphs, etc.
        # Placeholder for actual implementation

        return {"columns": [], "paragraphs": [], "tables": []}

    def _post_process_text(self, text: str) -> str:
        """Apply post-processing with medical terminology verification"""
        # Apply basic cleanup
        text = re.sub(r"\n+", "\n", text)

        # Medical terminology verification
        words = text.split()
        corrected_words = []

        for word in words:
            # Check if word is in medical dictionary or apply correction
            if word.lower() in self.medical_terms:
                corrected_words.append(word)
            else:
                # Apply fuzzy matching or correction
                # Placeholder for actual implementation
                corrected_words.append(word)

        return " ".join(corrected_words)

    def _calculate_ocr_confidence(self, text: str, img: Image.Image) -> float:
        """Calculate confidence score for OCR quality"""
        # Get confidence scores from Tesseract
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            if "conf" in data:
                # Filter out -1 values (which indicate no confidence available)
                conf_values = [c for c in data["conf"] if c >= 0]
                if conf_values:
                    return sum(conf_values) / len(conf_values) / 100.0
        except Exception as e:
            self.logger.warning(f"Error calculating OCR confidence: {e}")

        # Default confidence if calculation fails
        return 0.5

    def _analyze_document_structure(self, pages: List[Dict]) -> Dict:
        """Analyze hierarchical document structure"""
        # Identify sections, headers, etc.
        return {"sections": [], "headers": []}

    def _identify_document_type(self, pages: List[Dict]) -> str:
        """Identify clinical document type"""
        full_text = " ".join(page["text"] for page in pages)

        # Simple keyword-based approach
        keywords = {
            "operative_report": [
                "operation",
                "procedure",
                "surgeon",
                "preoperative",
                "postoperative",
            ],
            "pathology_report": ["specimen", "microscopic", "diagnosis", "pathology"],
            "consultation_note": [
                "consultation",
                "reason for consultation",
                "impression",
                "plan",
            ],
            "progress_note": [
                "progress",
                "subjective",
                "objective",
                "assessment",
                "plan",
            ],
            "discharge_summary": [
                "discharge",
                "hospital course",
                "follow up",
                "medications",
            ],
        }

        scores = {}
        for doc_type, words in keywords.items():
            scores[doc_type] = sum(
                1
                for word in words
                if re.search(r"\b" + word + r"\b", full_text, re.IGNORECASE)
            )

        # Return the document type with the highest score
        if not scores:
            return "unknown"
        return max(scores.items(), key=lambda x: x[1])[0]


class EHRProcessor(DocumentProcessor):
    """Handles processing of structured EHR data"""

    def __init__(self, config: Dict = None):
        super().__init__(config)

    def _process_ehr_document(self, file_path: Path) -> Dict:
        """Process structured EHR data exports"""
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            data = self._process_json_ehr(file_path)
        elif suffix == ".xml":
            data = self._process_xml_ehr(file_path)
        elif suffix in [".csv", ".xlsx"]:
            data = self._process_tabular_ehr(file_path)
        else:
            raise ValueError(f"Unsupported EHR format: {suffix}")

        # Convert to standardized key-value pairs
        standardized_data = self._standardize_ehr_data(data)

        # Preserve semantic relationships
        semantic_structure = self._preserve_semantic_relationships(standardized_data)

        # Document structure analysis
        document_structure = self._analyze_document_structure(semantic_structure)

        return {
            "raw_data": data,
            "standardized_data": standardized_data,
            "semantic_structure": semantic_structure,
            "document_structure": document_structure,
            "document_type": self._identify_document_type(standardized_data),
            "confidence": self.calculate_confidence_score(standardized_data),
        }

    def _process_json_ehr(self, file_path: Path) -> Dict:
        """Process JSON EHR data"""
        with open(file_path, "r") as f:
            return json.load(f)

    def _process_xml_ehr(self, file_path: Path) -> Dict:
        """Process XML EHR data"""
        import xml.etree.ElementTree as ET

        tree = ET.parse(file_path)
        root = tree.getroot()

        # Convert XML to dictionary
        # This is a simplified implementation - actual XML parsing
        # would likely need to handle namespaces and complex structures
        result = self._xml_to_dict(root)
        return result

    def _xml_to_dict(self, element):
        """Convert XML element to dictionary recursively"""
        result = {}

        # Add attributes
        for key, value in element.attrib.items():
            result[f"@{key}"] = value

        # Add child elements
        for child in element:
            child_dict = self._xml_to_dict(child)

            # Handle multiple elements with same tag
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = child_dict

        # Add text content if element has no children
        if not result and element.text and element.text.strip():
            return element.text.strip()

        return result

    def _process_tabular_ehr(self, file_path: Path) -> Dict:
        """Process tabular EHR data (CSV, Excel)"""
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".xlsx":
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported tabular format: {suffix}")

        # Convert DataFrame to dictionary
        records = df.to_dict(orient="records")

        # Group by any identifiable categories
        grouped_data = {}

        # Try to identify patient ID or similar grouping factor
        id_columns = [
            col for col in df.columns if "id" in col.lower() or "patient" in col.lower()
        ]

        if id_columns:
            primary_key = id_columns[0]
            for record in records:
                key = record[primary_key]
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(record)
            return grouped_data
        else:
            # If no grouping factor found, return as is
            return {"records": records}

    def _standardize_ehr_data(self, data: Dict) -> Dict:
        """Convert EHR data into standardized key-value pair representations"""
        # Flatten and standardize terminology
        standardized = {}

        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k

                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    if all(isinstance(x, dict) for x in v):
                        for i, item in enumerate(v):
                            items.extend(
                                flatten_dict(item, f"{new_key}[{i}]", sep=sep).items()
                            )
                    else:
                        items.append((new_key, v))
                else:
                    items.append((new_key, v))
            return dict(items)

        # Standardize keys
        flattened = flatten_dict(data)

        # Apply standardization rules
        for key, value in flattened.items():
            # Convert keys to snake_case
            std_key = re.sub(r"([A-Z])", r"_\1", key).lower()
            std_key = re.sub(r"[^a-z0-9_.]", "_", std_key)
            std_key = re.sub(r"_{2,}", "_", std_key)

            # Standardize common terms
            for old, new in [
                ("patient_id", "patient_identifier"),
                ("dob", "date_of_birth"),
                ("gender", "biological_sex"),
                ("medication", "drug"),
                ("rx", "prescription"),
                ("dx", "diagnosis"),
            ]:
                std_key = re.sub(rf"\b{old}\b", new, std_key)

            standardized[std_key] = value

        return standardized

    def _preserve_semantic_relationships(self, data: Dict) -> Dict:
        """Preserve semantic relationships in data structure conversion"""
        # Identify related fields
        relationships = {}

        # Group related fields by common prefixes
        prefixes = {}
        for key in data.keys():
            parts = key.split(".")
            for i in range(1, len(parts)):
                prefix = ".".join(parts[:i])
                if prefix not in prefixes:
                    prefixes[prefix] = []
                prefixes[prefix].append(key)

        # Create relationship groups
        for prefix, keys in prefixes.items():
            if len(keys) > 1:  # Only create group if multiple related fields
                relationships[prefix] = {k: data[k] for k in keys}

        # Build a tree structure for hierarchical relationships
        tree = {}
        for key, value in data.items():
            parts = key.split(".")
            current = tree
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        return {"flat": data, "relationships": relationships, "hierarchical": tree}

    def _analyze_document_structure(self, data: Dict) -> Dict:
        """Identify hierarchical organization in the document"""
        # Extract sections from hierarchical data
        sections = []

        def find_sections(d, path=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    sections.append(
                        {
                            "name": key,
                            "path": f"{path}.{key}" if path else key,
                            "depth": len(path.split(".")) + 1 if path else 1,
                        }
                    )
                    find_sections(value, f"{path}.{key}" if path else key)

        find_sections(data.get("hierarchical", {}))

        # Sort sections by path for consistent order
        sections.sort(key=lambda x: x["path"])

        return {
            "sections": sections,
            "max_depth": max(s["depth"] for s in sections) if sections else 0,
            "section_count": len(sections),
        }

    def _identify_document_type(self, data: Dict) -> str:
        """Identify clinical document type from standardized data"""
        # Extract all keys and values to a single string for simple analysis
        text = " ".join(
            [f"{k} {v}" if isinstance(v, str) else k for k, v in data.items()]
        )

        # Simple keyword-based approach
        keywords = {
            "operative_report": [
                "operation",
                "procedure",
                "surgeon",
                "preoperative",
                "postoperative",
            ],
            "pathology_report": ["specimen", "microscopic", "diagnosis", "pathology"],
            "consultation_note": [
                "consultation",
                "reason for consultation",
                "impression",
                "plan",
            ],
            "progress_note": [
                "progress",
                "subjective",
                "objective",
                "assessment",
                "plan",
            ],
            "discharge_summary": [
                "discharge",
                "hospital course",
                "follow up",
                "medications",
            ],
        }

        scores = {}
        for doc_type, words in keywords.items():
            scores[doc_type] = sum(
                1
                for word in words
                if re.search(r"\b" + word + r"\b", text, re.IGNORECASE)
            )

        # Return the document type with the highest score
        if not scores:
            return "unknown"
        return max(scores.items(), key=lambda x: x[1])[0]

    def calculate_confidence_score(self, data: Dict) -> float:
        """Calculate confidence score for extraction quality"""
        # Check data completeness and consistency
        if not data:
            return 0.0

        # A simple scoring mechanism based on completeness
        total_fields = 0
        non_empty_fields = 0

        def count_fields(d):
            nonlocal total_fields, non_empty_fields

            for k, v in d.items():
                if isinstance(v, dict):
                    count_fields(v)
                else:
                    total_fields += 1
                    if v is not None and (not isinstance(v, str) or v.strip()):
                        non_empty_fields += 1

        count_fields(data)

        if total_fields == 0:
            return 0.0

        # Basic completeness score
        completeness = non_empty_fields / total_fields

        # Could be expanded with consistency checks
        return completeness


# -----------------------------------------------------------------------------
# MODULE 2: Tokenization and Language Model Integration
# -----------------------------------------------------------------------------


class TokenizationProcessor:
    """
    Handles tokenization and language model integration for clinical text
    with support for multiple biomedical models and strategies for long documents.
    """

    def __init__(self, model_name: str = "bioclinicalbert", config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Set defaults
        self.config = {
            "max_length": 512,
            "stride": 128,
            "batch_size": 16,
            "dynamic_batch_sizing": True,
            "segment_min_length": 10,
            "segment_strategy": "sentence",  # 'sentence', 'paragraph', 'fixed'
            "special_token_handling": True,
            "long_document_strategy": "sliding_window",  # 'sliding_window', 'hierarchical', 'important_segments', 'summarize'
            **self.config,
        }

        self.model_name = model_name
        if model_name not in BIOMEDICAL_MODELS:
            self.logger.warning(
                f"Unknown model: {model_name}, defaulting to bioclinicalbert"
            )
            self.model_name = "bioclinicalbert"

        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """Load the specified biomedical tokenizer"""
        model_path = BIOMEDICAL_MODELS[self.model_name]
        self.logger.info(f"Loading tokenizer: {model_path}")

        try:
            if "t5" in self.model_name.lower():
                return T5Tokenizer.from_pretrained(model_path)
            elif "bert" in self.model_name.lower():
                return BertTokenizer.from_pretrained(model_path)
            else:
                return AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            self.logger.info("Falling back to default tokenizer")
            return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def tokenize_document(self, document: Union[str, Dict]) -> Dict:
        """
        Tokenize a clinical document with a multi-stage tokenization pipeline

        Args:
            document: Either a string (raw text) or a dictionary with processed text

        Returns:
            Dictionary with tokenization results
        """
        # Extract text from document if it's a dictionary
        if isinstance(document, dict):
            text = document.get("full_text", "")
            if not text and "pages" in document:
                text = "\n\n".join(page.get("text", "") for page in document["pages"])
            document_type = document.get("document_type", "unknown")
        else:
            text = document
            document_type = "unknown"

        # Step 1: Preliminary text cleaning
        cleaned_text = self._clean_text(text)

        # Step 2: Text segmentation
        segments = self._segment_text(cleaned_text)

        # Step 3: Apply tokenization strategy based on document length
        if (
            len(segments) * self.config["segment_min_length"]
            > self.config["max_length"]
        ):
            # Long document
            tokenized_result = self._handle_long_document(segments, document_type)
        else:
            # Short document - standard tokenization
            tokenized_result = self._tokenize_segments(segments)

        return {
            "tokens": tokenized_result["tokens"],
            "token_ids": tokenized_result["token_ids"],
            "segment_mapping": tokenized_result["segment_mapping"],
            "attention_mask": tokenized_result["attention_mask"],
            "special_tokens": tokenized_result.get("special_tokens", []),
            "truncated": tokenized_result.get("truncated", False),
            "num_segments": len(segments),
            "document_type": document_type,
        }

    def _clean_text(self, text: str) -> str:
        """Perform preliminary text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Fix common OCR errors
        text = re.sub(r"rnl", "ml", text)  # Fix common OCR error for "ml"
        text = re.sub(r"I<eukocytes", "Leukocytes", text)  # Fix common OCR error

        # Remove non-printable characters
        text = "".join(c if c.isprintable() or c in ["\n", "\t"] else " " for c in text)

        # Normalize medical abbreviations
        abbreviations = {
            r"\bpt\b": "patient",
            r"\bDx\b": "diagnosis",
            r"\bTx\b": "treatment",
            r"\bHx\b": "history",
            r"\bRx\b": "prescription",
        }

        for abbr, expansion in abbreviations.items():
            text = re.sub(abbr, expansion, text)

        return text

    def _segment_text(self, text: str) -> List[str]:
        """Segment text into appropriate units"""
        strategy = self.config["segment_strategy"]

        if strategy == "sentence":
            # Split by sentences
            segments = sent_tokenize(text)
        elif strategy == "paragraph":
            # Split by paragraphs (double newlines)
            segments = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        elif strategy == "fixed":
            # Split by fixed character length
            fixed_length = self.config.get("fixed_segment_length", 200)
            segments = [
                text[i : i + fixed_length] for i in range(0, len(text), fixed_length)
            ]
        else:
            # Default to sentence
            segments = sent_tokenize(text)

        # Remove empty segments and ensure minimum length
        min_length = self.config["segment_min_length"]
        segments = [s for s in segments if len(s) >= min_length]

        return segments

    def _tokenize_segments(self, segments: List[str]) -> Dict:
        """Tokenize a list of text segments"""
        all_tokens = []
        all_token_ids = []
        segment_mapping = []  # Maps tokens to original segments

        for i, segment in enumerate(segments):
            # Tokenize the segment
            encoding = self.tokenizer(
                segment,
                add_special_tokens=self.config["special_token_handling"],
                return_attention_mask=True,
                return_token_type_ids=False,
                return_offsets_mapping=False,
                truncation=True,
                max_length=self.config["max_length"],
            )

            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

            # Add to result
            all_tokens.extend(tokens)
            all_token_ids.extend(encoding["input_ids"])
            segment_mapping.extend([i] * len(tokens))

        # Create attention mask (1 for all tokens)
        attention_mask = [1] * len(all_token_ids)

        return {
            "tokens": all_tokens,
            "token_ids": all_token_ids,
            "segment_mapping": segment_mapping,
            "attention_mask": attention_mask,
            "truncated": False,
        }

    def _handle_long_document(self, segments: List[str], document_type: str) -> Dict:
        """Handle tokenization for long documents"""
        strategy = self.config["long_document_strategy"]

        if strategy == "sliding_window":
            return self._sliding_window_tokenization(segments)
        elif strategy == "hierarchical":
            return self._hierarchical_tokenization(segments, document_type)
        elif strategy == "important_segments":
            return self._important_segments_tokenization(segments, document_type)
        elif strategy == "summarize":
            return self._summarize_and_tokenize(segments)
        else:
            # Default to sliding window
            return self._sliding_window_tokenization(segments)

    def _sliding_window_tokenization(self, segments: List[str]) -> Dict:
        """Apply sliding window tokenization for long documents"""
        max_length = self.config["max_length"]
        stride = self.config["stride"]

        # Combine segments into text
        text = " ".join(segments)

        # Tokenize with sliding window
        encoding = self.tokenizer(
            text,
            return_overflowing_tokens=True,
            stride=stride,
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            padding="max_length",
        )

        # Extract token information
        all_tokens = []
        all_token_ids = []
        all_attention_masks = []
        window_mapping = []  # Maps tokens to window

        for i, window_ids in enumerate(encoding["input_ids"]):
            tokens = self.tokenizer.convert_ids_to_tokens(window_ids)
            all_tokens.extend(tokens)
            all_token_ids.extend(window_ids)
            all_attention_masks.extend(encoding["attention_mask"][i])
            window_mapping.extend([i] * len(tokens))

        return {
            "tokens": all_tokens,
            "token_ids": all_token_ids,
            "segment_mapping": window_mapping,  # Maps to window instead of original segment
            "attention_mask": all_attention_masks,
            "windows": len(encoding["input_ids"]),
            "window_size": max_length,
            "stride": stride,
            "truncated": True,  # Indicates that the document was processed in windows
        }

    def _hierarchical_tokenization(
        self, segments: List[str], document_type: str
    ) -> Dict:
        """Apply hierarchical tokenization preserving document boundaries"""
        # Group segments into sections based on document type
        sections = self._identify_document_sections(segments, document_type)

        all_tokens = []
        all_token_ids = []
        all_attention_masks = []
        section_mapping = []  # Maps tokens to section
        segment_mapping = []  # Maps tokens to original segments

        for section_idx, section in enumerate(sections):
            # Tokenize each section
            section_text = " ".join(segments[i] for i in section["segment_indices"])

            encoding = self.tokenizer(
                section_text,
                add_special_tokens=True,
                return_attention_mask=True,
                truncation=True,
                max_length=self.config["max_length"],
            )

            tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

            # Add to result
            all_tokens.extend(tokens)
            all_token_ids.extend(encoding["input_ids"])
            all_attention_masks.extend(encoding["attention_mask"])
            section_mapping.extend([section_idx] * len(tokens))

            # Map to original segments (approximate)
            token_count = len(tokens)
            segment_count = len(section["segment_indices"])

            if segment_count == 0:
                continue

            tokens_per_segment = token_count / segment_count

            for i, seg_idx in enumerate(section["segment_indices"]):
                start = int(i * tokens_per_segment)
                end = int((i + 1) * tokens_per_segment)
                segment_mapping.extend([seg_idx] * (end - start))

            # Adjust if lengths don't match
            while len(segment_mapping) < len(all_tokens):
                segment_mapping.append(section["segment_indices"][-1])

        return {
            "tokens": all_tokens,
            "token_ids": all_token_ids,
            "segment_mapping": segment_mapping,
            "section_mapping": section_mapping,
            "attention_mask": all_attention_masks,
            "sections": sections,
            "hierarchical": True,
            "truncated": False,
        }

    def _identify_document_sections(
        self, segments: List[str], document_type: str
    ) -> List[Dict]:
        """Identify document sections based on document type and content"""
        sections = []

        # Simple heuristic section identification
        section_headers = {
            "operative_report": [
                "preoperative diagnosis",
                "procedure",
                "findings",
                "postoperative diagnosis",
            ],
            "pathology_report": [
                "clinical history",
                "gross description",
                "microscopic description",
                "diagnosis",
            ],
            "consultation_note": [
                "chief complaint",
                "history",
                "physical examination",
                "assessment",
                "plan",
            ],
            "progress_note": ["subjective", "objective", "assessment", "plan"],
            "discharge_summary": [
                "admission diagnosis",
                "hospital course",
                "discharge diagnosis",
                "follow up",
            ],
        }

        # Get section headers for document type
        headers = section_headers.get(document_type, [])
        if not headers:
            # Generic section detection
            headers = [h for sublist in section_headers.values() for h in sublist]

        # Find sections based on headers
        current_section = {"name": "preamble", "segment_indices": []}
        sections.append(current_section)

        for i, segment in enumerate(segments):
            # Check if segment starts a new section
            is_header = False
            for header in headers:
                if re.search(rf"\b{re.escape(header)}\b", segment.lower()):
                    # Start new section
                    current_section = {"name": header, "segment_indices": []}
                    sections.append(current_section)
                    is_header = True
                    break

            # Add segment to current section
            current_section["segment_indices"].append(i)

        return sections

    def _important_segments_tokenization(
        self, segments: List[str], document_type: str
    ) -> Dict:
        """Tokenize document focusing on important segments using term density"""
        # Identify important segments using term density heuristics
        important_segments = self._identify_important_segments(segments, document_type)

        # Tokenize only important segments
        selected_segments = [segments[i] for i in important_segments]

        # Standard tokenization for selected segments
        result = self._tokenize_segments(selected_segments)

        # Map back to original segment indices
        adjusted_mapping = [
            important_segments[idx] for idx in result["segment_mapping"]
        ]
        result["segment_mapping"] = adjusted_mapping
        result["important_segments"] = important_segments

        return result

    def _identify_important_segments(
        self, segments: List[str], document_type: str
    ) -> List[int]:
        """Identify important segments using term density heuristics"""
        # Terms to look for based on document type
        important_terms = {
            "operative_report": [
                "performed",
                "surgery",
                "incision",
                "procedure",
                "finding",
            ],
            "pathology_report": [
                "tumor",
                "lesion",
                "malignant",
                "benign",
                "margin",
                "diagnosis",
            ],
            "consultation_note": ["assessment", "recommend", "impression", "plan"],
            "progress_note": ["assessment", "plan", "change", "improve", "worsen"],
            "discharge_summary": [
                "discharge",
                "follow-up",
                "medication",
                "instruction",
            ],
            "unknown": ["diagnosis", "result", "finding", "treatment", "plan"],
        }

        terms = important_terms.get(document_type, important_terms["unknown"])

        # Score segments by term density
        scores = []
        for i, segment in enumerate(segments):
            text = segment.lower()
            # Count term occurrences
            count = sum(text.count(term) for term in terms)
            # Calculate term density
            density = count / (len(text.split()) + 1)  # +1 to avoid division by zero
            scores.append((i, density))

        # Sort by score and take top segments up to max_length
        scores.sort(key=lambda x: x[1], reverse=True)

        # Take top segments (limit to 70% of max_length)
        max_segments = int(
            self.config["max_length"] * 0.7 / self.config["segment_min_length"]
        )

        # Always include first and last segments (often contain key information)
        top_segments = [idx for idx, _ in scores[:max_segments]]
        if 0 not in top_segments and segments:
            top_segments.append(0)
        if len(segments) - 1 not in top_segments and len(segments) > 1:
            top_segments.append(len(segments) - 1)

        # Sort by original order
        top_segments.sort()

        return top_segments

    def _summarize_and_tokenize(self, segments: List[str]) -> Dict:
        """Summarize document and tokenize the summary for extremely long documents"""
        # Simple extractive summarization
        summary_segments = self._extractive_summarize(segments)

        # Tokenize summary
        result = self._tokenize_segments(summary_segments)

        result["summarized"] = True
        result["summary_ratio"] = len(summary_segments) / len(segments)

        return result

    def _extractive_summarize(self, segments: List[str]) -> List[str]:
        """Simple extractive summarization"""
        if len(segments) <= 10:
            return segments

        # Include first two and last two segments
        summary = [segments[0], segments[1]]

        # Add some segments from the middle
        mid_segments = segments[2:-2]

        # Take every nth segment based on length
        n = max(1, len(mid_segments) // 6)
        summary.extend(mid_segments[::n])

        # Add last segments
        if len(segments) > 3:
            summary.append(segments[-2])
        if len(segments) > 2:
            summary.append(segments[-1])

        return summary

    def prepare_batch(self, documents: List[Dict]) -> Dict:
        """
        Prepare a batch of documents for processing with dynamic batch sizing

        Args:
            documents: List of tokenized documents

        Returns:
            Dictionary with batched input tensors
        """
        if not documents:
            return None

        batch_size = self.config["batch_size"]
        dynamic_sizing = self.config["dynamic_batch_sizing"]

        if dynamic_sizing:
            # Adjust batch size based on document length
            avg_length = sum(len(doc["token_ids"]) for doc in documents) / len(
                documents
            )
            adjusted_size = max(1, int(batch_size * (512 / avg_length)))
            batch_size = min(adjusted_size, len(documents))

        # Take a subset of documents for current batch
        batch_docs = documents[:batch_size]

        # Determine max sequence length in batch
        max_length = max(len(doc["token_ids"]) for doc in batch_docs)

        # Prepare input tensors
        input_ids = []
        attention_masks = []
        segment_ids = []

        for doc in batch_docs:
            # Pad to max length
            ids = doc["token_ids"]
            attn = doc.get("attention_mask", [1] * len(ids))

            # Padding
            padding_length = max_length - len(ids)
            ids = ids + [self.tokenizer.pad_token_id] * padding_length
            attn = attn + [0] * padding_length

            input_ids.append(ids)
            attention_masks.append(attn)
            segment_ids.append(doc.get("segment_mapping", [0] * len(ids)))

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "segment_ids": torch.tensor(segment_ids),
        }


# -----------------------------------------------------------------------------
# MODULE 3: Clinical Entity Recognition and Normalization
# -----------------------------------------------------------------------------


class EntityRecognitionProcessor:
    """
    Handles clinical entity recognition and normalization,
    specializing in oncology-specific entities and temporal information.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Set defaults
        self.config = {
            "use_spacy": True,
            "use_rules": True,
            "use_deep_learning": True,
            "ontologies": ["snomed_ct", "rxnorm", "loinc", "icd_o_3"],
            "entity_types": [
                "condition",
                "medication",
                "procedure",
                "observation",
                "biomarker",
                "tumor",
                "staging",
            ],
            "abbreviation_expansion": True,
            "term_disambiguation": True,
            "cancer_specific_extraction": True,
            "temporal_extraction": True,
            **self.config,
        }

        # Initialize NLP components
        self.nlp = self._initialize_nlp()

        # Load ontologies
        self.ontologies = self._load_ontologies()

        # Load cancer-specific extractors
        self.cancer_extractors = self._initialize_cancer_extractors()

    def _initialize_nlp(self):
        """Initialize NLP components"""
        if not self.config["use_spacy"]:
            return None

        try:
            # Load spaCy model
            import spacy

            try:
                # Try to load clinical model first
                nlp = spacy.load("en_core_sci_md")
            except OSError:
                # Fall back to standard model
                nlp = spacy.load("en_core_web_md")

            self.logger.info(f"Loaded spaCy model: {nlp.meta['name']}")
            return nlp
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}")
            return None

    def _load_ontologies(self) -> Dict:
        """Load medical ontologies for normalization"""
        ontologies = {}

        for ontology_name in self.config["ontologies"]:
            ontology_key = ontology_name.lower()

            if ontology_key not in MEDICAL_ONTOLOGIES:
                self.logger.warning(f"Unknown ontology: {ontology_name}")
                continue

            ontology_id = MEDICAL_ONTOLOGIES[ontology_key]

            # This would normally load from UMLS or other sources
            # For the purpose of this implementation, we'll create placeholders
            ontologies[ontology_key] = {
                "id": ontology_id,
                "concepts": self._load_ontology_concepts(ontology_key),
                "synonyms": self._load_ontology_synonyms(ontology_key),
                "relations": self._load_ontology_relations(ontology_key),
            }

            self.logger.info(f"Loaded ontology: {ontology_key}")

        return ontologies

    def _load_ontology_concepts(self, ontology_name: str) -> Dict:
        """Load ontology concepts (placeholder)"""
        # This would normally load from a database or file
        # For demonstration, return minimal sample data
        if ontology_name == "snomed_ct":
            return {
                "254837009": {
                    "name": "Malignant neoplasm of breast",
                    "type": "disorder",
                },
                "13979006": {"name": "Lung cancer", "type": "disorder"},
                "118576006": {
                    "name": "Pathologic TNM stage",
                    "type": "observable entity",
                },
            }
        elif ontology_name == "rxnorm":
            return {
                "1309226": {"name": "Tamoxifen", "type": "substance"},
                "1309209": {"name": "Docetaxel", "type": "substance"},
                "1551504": {"name": "Trastuzumab", "type": "substance"},
            }
        # Add more cases for other ontologies
        return {}

    def _load_ontology_synonyms(self, ontology_name: str) -> Dict:
        """Load ontology synonyms (placeholder)"""
        # This would normally load from a database or file
        if ontology_name == "snomed_ct":
            return {
                "breast cancer": ["254837009"],
                "carcinoma of breast": ["254837009"],
                "mammary carcinoma": ["254837009"],
                "lung cancer": ["13979006"],
                "carcinoma of lung": ["13979006"],
                "bronchogenic carcinoma": ["13979006"],
                "tnm staging": ["118576006"],
                "tnm classification": ["118576006"],
            }
        elif ontology_name == "rxnorm":
            return {
                "tamoxifen": ["1309226"],
                "nolvadex": ["1309226"],
                "docetaxel": ["1309209"],
                "taxotere": ["1309209"],
                "trastuzumab": ["1551504"],
                "herceptin": ["1551504"],
            }
        # Add more cases for other ontologies
        return {}

    def _load_ontology_relations(self, ontology_name: str) -> Dict:
        """Load ontology relationships (placeholder)"""
        # This would normally load from a database or file
        if ontology_name == "snomed_ct":
            return {
                # Relation format: (source_id, type, target_id)
                (
                    "254837009",
                    "is_a",
                    "363346000",
                ),  # Breast cancer is_a malignant neoplasm
                (
                    "13979006",
                    "is_a",
                    "363346000",
                ),  # Lung cancer is_a malignant neoplasm
            }
        # Add more cases for other ontologies
        return {}

    def _initialize_cancer_extractors(self) -> Dict:
        """Initialize specialized cancer-specific entity extractors"""
        extractors = {}

        if self.config["cancer_specific_extraction"]:
            extractors = {
                "tumor": TumorExtractor(),
                "staging": StagingExtractor(),
                "biomarker": BiomarkerExtractor(),
                "treatment": TreatmentExtractor(),
                "response": ResponseExtractor(),
            }

        return extractors

    def process(self, document: Union[str, Dict]) -> Dict:
        """
        Process clinical document for entity recognition and normalization

        Args:
            document: Either raw text or a dictionary with processed text

        Returns:
            Dictionary with recognized entities and their properties
        """
        # Extract text from document if it's a dictionary
        if isinstance(document, dict):
            text = document.get("full_text", "")
            if not text and "pages" in document:
                text = "\n\n".join(page.get("text", "") for page in document["pages"])
            document_type = document.get("document_type", "unknown")
        else:
            text = document
            document_type = "unknown"

        # Initialize results
        entities = []

        # 1. Rule-based extraction
        if self.config["use_rules"]:
            rule_entities = self._extract_with_rules(text, document_type)
            entities.extend(rule_entities)

        # 2. spaCy-based extraction
        if self.config["use_spacy"] and self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            entities.extend(spacy_entities)

        # 3. Deep learning extraction
        if self.config["use_deep_learning"]:
            dl_entities = self._extract_with_deep_learning(text, document_type)
            entities.extend(dl_entities)

        # 4. Cancer-specific extraction
        if self.config["cancer_specific_extraction"]:
            cancer_entities = self._extract_cancer_specific(text, document_type)
            entities.extend(cancer_entities)

        # 5. Temporal information extraction
        if self.config["temporal_extraction"]:
            temporal_entities = self._extract_temporal_information(text)
            entities.extend(temporal_entities)

        # 6. Entity normalization
        normalized_entities = self._normalize_entities(entities)

        # 7. Link to ontologies
        linked_entities = self._link_to_ontologies(normalized_entities)

        # 8. Resolve entity relationships
        entity_relationships = self._resolve_entity_relationships(linked_entities)

        # 9. Deduplicate entities
        unique_entities = self._deduplicate_entities(linked_entities)

        return {
            "entities": unique_entities,
            "relationships": entity_relationships,
            "document_type": document_type,
            "entity_counts": self._count_entities_by_type(unique_entities),
            "temporal_timeline": self._construct_timeline(unique_entities),
        }

    def _extract_with_rules(self, text: str, document_type: str) -> List[Dict]:
        """Extract entities using rule-based approaches"""
        entities = []

        # Simple pattern matching for common entities
        # This would typically include more sophisticated rules

        # Medications
        medication_patterns = [
            r"\b(\w+)\s+(\d+\s*(?:mg|mcg|g|ml))\b",  # e.g., "tamoxifen 20 mg"
            r"\b(\w+)\s+(\d+)\s*(mg|mcg|g|ml)\b",  # e.g., "docetaxel 75 mg"
        ]

        for pattern in medication_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                med_name = match.group(1)
                dose = match.group(2)

                entities.append(
                    {
                        "text": match.group(0),
                        "type": "medication",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {
                            "name": med_name,
                            "dosage": dose if len(match.groups()) > 1 else None,
                            "source": "rule-based",
                        },
                    }
                )

        # Tumor location patterns
        tumor_patterns = [
            r"\btumor\s+in\s+(?:the\s+)?(\w+)\b",
            r"\b(\w+)\s+(?:tumor|mass|lesion|cancer|carcinoma)\b",
            r"\bmalignant\s+(?:tumor|mass|lesion)\s+(?:in|of)\s+(?:the\s+)?(\w+)\b",
        ]

        for pattern in tumor_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                location = match.group(1)

                entities.append(
                    {
                        "text": match.group(0),
                        "type": "tumor",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {"location": location, "source": "rule-based"},
                    }
                )

        # TNM staging patterns
        staging_patterns = [
            r"\b(T\d+[a-z]?)(N\d+[a-z]?)(M\d+[a-z]?)\b",  # e.g., "T2N0M0"
            r"\bstage\s+([0IV]+[a-z]?)\b",  # e.g., "stage IIb"
            r"\b(T\d+[a-z]?)[\s,]+(N\d+[a-z]?)[\s,]+(M\d+[a-z]?)\b",  # e.g., "T2, N0, M0"
        ]

        for pattern in staging_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if "stage" in pattern:
                    entities.append(
                        {
                            "text": match.group(0),
                            "type": "staging",
                            "start": match.start(),
                            "end": match.end(),
                            "properties": {
                                "stage": match.group(1),
                                "source": "rule-based",
                            },
                        }
                    )
                else:
                    entities.append(
                        {
                            "text": match.group(0),
                            "type": "staging",
                            "start": match.start(),
                            "end": match.end(),
                            "properties": {
                                "T": match.group(1),
                                "N": match.group(2),
                                "M": match.group(3),
                                "source": "rule-based",
                            },
                        }
                    )

        # Add more rule patterns as needed

        return entities

    def _extract_with_spacy(self, text: str) -> List[Dict]:
        """Extract entities using spaCy"""
        entities = []

        doc = self.nlp(text)

        for ent in doc.ents:
            # Map spaCy entity types to our schema
            entity_type = self._map_spacy_entity_type(ent.label_)

            if entity_type:
                entities.append(
                    {
                        "text": ent.text,
                        "type": entity_type,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "properties": {"spacy_label": ent.label_, "source": "spacy"},
                    }
                )

        return entities

    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity type to our schema"""
        # Default mapping from spaCy to our types
        mapping = {
            "DISEASE": "condition",
            "CHEMICAL": "medication",
            "PROCEDURE": "procedure",
            "TEST": "observation",
            "BODY_PART": "anatomy",
        }

        return mapping.get(spacy_label)

    def _extract_with_deep_learning(self, text: str, document_type: str) -> List[Dict]:
        """
        Extract entities using deep learning models

        This is a placeholder that would normally use a trained NER model
        """
        # In a real implementation, this would use a pretrained or fine-tuned
        # clinical NER model specific to oncology

        # For the purpose of this implementation, return empty
        return []

    def _extract_cancer_specific(self, text: str, document_type: str) -> List[Dict]:
        """Extract cancer-specific entities"""
        entities = []

        for extractor_name, extractor in self.cancer_extractors.items():
            extracted = extractor.extract(text, document_type)

            for entity in extracted:
                # Ensure consistent format
                entity["properties"] = entity.get("properties", {})
                entity["properties"]["source"] = f"cancer-{extractor_name}"
                entities.append(entity)

        return entities

    def _extract_temporal_information(self, text: str) -> List[Dict]:
        """Extract temporal information for patient timelines"""
        entities = []

        # Date patterns (various formats)
        date_patterns = [
            r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b",  # MM/DD/YYYY or DD/MM/YYYY
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",  # Month DD, YYYY
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{4})\b",  # DD Month YYYY
        ]

        # Time expressions
        time_expressions = [
            r"\b(\d+)\s+(?:day|week|month|year)s?\s+(?:ago|before|prior)\b",
            r"\b(?:last|previous)\s+(?:day|week|month|year)\b",
            r"\b(?:next|following)\s+(?:day|week|month|year)\b",
            r"\bin\s+(\d+)\s+(?:day|week|month|year)s?\b",
            r"\b(?:since|after|before)\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
        ]

        # Relative time markers
        relative_markers = [
            r"\bcurrently\b",
            r"\bpresently\b",
            r"\bat\s+presentation\b",
            r"\bon\s+admission\b",
            r"\bduring\s+(?:the\s+)?(?:procedure|operation|treatment|hospitalization)\b",
            r"\bpost[- ](?:operative|procedure|treatment|therapy)\b",
            r"\bpre[- ](?:operative|procedure|treatment|therapy)\b",
        ]

        # Extract dates
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    {
                        "text": match.group(0),
                        "type": "temporal",
                        "subtype": "date",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {"value": match.group(0), "source": "temporal"},
                    }
                )

        # Extract time expressions
        for pattern in time_expressions:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    {
                        "text": match.group(0),
                        "type": "temporal",
                        "subtype": "time_expression",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {"value": match.group(0), "source": "temporal"},
                    }
                )

        # Extract relative markers
        for pattern in relative_markers:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(
                    {
                        "text": match.group(0),
                        "type": "temporal",
                        "subtype": "relative_marker",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {"value": match.group(0), "source": "temporal"},
                    }
                )

        return entities

    def _normalize_entities(self, entities: List[Dict]) -> List[Dict]:
        """Normalize clinical entities"""
        normalized = []

        for entity in entities:
            # Create a copy to avoid modifying original
            norm_entity = entity.copy()

            # Handle abbreviations if enabled
            if self.config["abbreviation_expansion"] and "text" in entity:
                expanded = self._expand_abbreviation(entity["text"])
                if expanded != entity["text"]:
                    if "properties" not in norm_entity:
                        norm_entity["properties"] = {}
                    norm_entity["properties"]["expanded"] = expanded

            # Apply normalization based on entity type
            if entity.get("type") == "medication":
                self._normalize_medication(norm_entity)
            elif entity.get("type") == "condition":
                self._normalize_condition(norm_entity)
            elif entity.get("type") == "procedure":
                self._normalize_procedure(norm_entity)
            elif entity.get("type") == "tumor":
                self._normalize_tumor(norm_entity)
            elif entity.get("type") == "staging":
                self._normalize_staging(norm_entity)
            elif entity.get("type") == "biomarker":
                self._normalize_biomarker(norm_entity)
            elif entity.get("type") == "temporal":
                self._normalize_temporal(norm_entity)

            normalized.append(norm_entity)

        # Apply term disambiguation if enabled
        if self.config["term_disambiguation"]:
            normalized = self._disambiguate_terms(normalized)

        return normalized

    def _expand_abbreviation(self, text: str) -> str:
        """Expand medical abbreviations"""
        # Common oncology abbreviations
        abbreviations = {
            # General medical
            "pt": "patient",
            "hx": "history",
            "dx": "diagnosis",
            "tx": "treatment",
            "rx": "prescription",
            "sx": "symptom",
            "fx": "fracture",
            # Oncology specific
            "ca": "cancer",
            "mets": "metastasis",
            "chemo": "chemotherapy",
            "rt": "radiation therapy",
            "xrt": "radiation therapy",
            "gy": "gray",
            "cr": "complete response",
            "pr": "partial response",
            "sd": "stable disease",
            "pd": "progressive disease",
            # Cancer types
            "nsclc": "non-small cell lung cancer",
            "sclc": "small cell lung cancer",
            "crc": "colorectal cancer",
            "hcc": "hepatocellular carcinoma",
        }

        # For exact matches only
        if text.lower() in abbreviations:
            return abbreviations[text.lower()]

        # For more context-aware expansion, we would need more complex algorithms
        return text

    def _normalize_medication(self, entity: Dict) -> None:
        """Normalize medication entities"""
        if "properties" not in entity:
            entity["properties"] = {}

        # Extract and normalize dosage information
        if "text" in entity:
            # Extract dosage patterns
            dosage_match = re.search(
                r"(\d+\.?\d*)\s*(mg|mcg|g|ml|mg/m2)", entity["text"], re.IGNORECASE
            )
            if dosage_match:
                entity["properties"]["dosage"] = dosage_match.group(1)
                entity["properties"]["unit"] = dosage_match.group(2).lower()

            # Normalize drug name (lowercase for consistency)
            if "name" in entity["properties"]:
                entity["properties"]["name"] = entity["properties"]["name"].lower()

    def _normalize_condition(self, entity: Dict) -> None:
        """Normalize condition entities"""
        if "properties" not in entity:
            entity["properties"] = {}

        # Extract laterality (left/right)
        if "text" in entity:
            laterality_match = re.search(
                r"\b(left|right)\b", entity["text"], re.IGNORECASE
            )
            if laterality_match:
                entity["properties"]["laterality"] = laterality_match.group(1).lower()

    def _normalize_tumor(self, entity: Dict) -> None:
        """Normalize tumor entities"""
        if "properties" not in entity:
            entity["properties"] = {}

        # Extract and normalize size information
        if "text" in entity:
            size_match = re.search(
                r"(\d+\.?\d*)\s*(mm|cm|centimeter|millimeter)",
                entity["text"],
                re.IGNORECASE,
            )
            if size_match:
                entity["properties"]["size"] = float(size_match.group(1))
                unit = size_match.group(2).lower()
                # Convert to standard unit (cm)
                if unit.startswith("mm"):
                    entity["properties"]["size"] /= 10
                entity["properties"]["size_unit"] = "cm"

            # Extract location (if not already present)
            if "location" not in entity["properties"]:
                for organ in [
                    "breast",
                    "lung",
                    "liver",
                    "colon",
                    "prostate",
                    "kidney",
                    "bladder",
                    "pancreas",
                    "brain",
                ]:
                    if re.search(r"\b" + organ + r"\b", entity["text"], re.IGNORECASE):
                        entity["properties"]["location"] = organ
                        break

    def _normalize_staging(self, entity: Dict) -> None:
        """Normalize staging entities"""
        if "properties" not in entity:
            entity["properties"] = {}

        # Standardize TNM format
        for prop in ["T", "N", "M"]:
            if prop in entity["properties"]:
                # Ensure uppercase and strip spaces
                entity["properties"][prop] = entity["properties"][prop].upper().strip()

        # For clinical stage (I, II, III, IV)
        if "stage" in entity["properties"]:
            # Ensure uppercase roman numerals
            stage = entity["properties"]["stage"]
            entity["properties"]["stage"] = stage.upper()

    def _normalize_biomarker(self, entity: Dict) -> None:
        """Normalize biomarker entities"""
        if "properties" not in entity:
            entity["properties"] = {}

        # Standardize biomarker status
        if "status" in entity["properties"]:
            status = entity["properties"]["status"].lower()

            # Map various expressions to standard values
            if status in ["positive", "pos", "+", "overexpressed"]:
                entity["properties"]["status"] = "positive"
            elif status in ["negative", "neg", "-", "not expressed"]:
                entity["properties"]["status"] = "negative"
            elif status in ["equivocal", "borderline", "indeterminate"]:
                entity["properties"]["status"] = "equivocal"

    def _normalize_temporal(self, entity: Dict) -> None:
        """Normalize temporal entities"""
        if "properties" not in entity:
            entity["properties"] = {}

        # Standardize date formats if possible
        if entity.get("subtype") == "date" and "value" in entity["properties"]:
            date_text = entity["properties"]["value"]

            # Try to parse with various formats
            from datetime import datetime

            date_formats = [
                "%m/%d/%Y",
                "%d/%m/%Y",  # MM/DD/YYYY or DD/MM/YYYY
                "%m-%d-%Y",
                "%d-%m-%Y",  # MM-DD-YYYY or DD-MM-YYYY
                "%B %d, %Y",
                "%b %d, %Y",  # Month DD, YYYY
                "%d %B %Y",
                "%d %b %Y",  # DD Month YYYY
            ]

            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_text, fmt)
                    entity["properties"]["normalized_date"] = parsed_date.strftime(
                        "%Y-%m-%d"
                    )
                    break
                except ValueError:
                    continue

    def _disambiguate_terms(self, entities: List[Dict]) -> List[Dict]:
        """Disambiguate terms with multiple possible meanings"""
        # For this implementation, focus on common ambiguous terms in oncology

        # Example: "cold" could be temperature or common cold
        # Example: "discharge" could be a substance or the act of leaving hospital

        # This would normally use context information and ML models
        # For simplicity, return the original entities
        return entities

    def _link_to_ontologies(self, entities: List[Dict]) -> List[Dict]:
        """Link entities to ontology concepts"""
        linked_entities = []

        for entity in entities:
            # Copy entity to avoid modifying the original
            linked_entity = entity.copy()

            # Add ontology_links field if not exists
            if "properties" not in linked_entity:
                linked_entity["properties"] = {}

            if "ontology_links" not in linked_entity["properties"]:
                linked_entity["properties"]["ontology_links"] = []

            entity_text = entity.get("text", "").lower()
            entity_type = entity.get("type", "")

            # Attempt to match with ontologies based on entity type
            for ontology_name, ontology in self.ontologies.items():
                # Check entity text against synonyms dictionary
                if entity_text in ontology["synonyms"]:
                    concept_ids = ontology["synonyms"][entity_text]

                    for concept_id in concept_ids:
                        if concept_id in ontology["concepts"]:
                            concept = ontology["concepts"][concept_id]

                            # Add link to ontology
                            linked_entity["properties"]["ontology_links"].append(
                                {
                                    "ontology": ontology_name,
                                    "concept_id": concept_id,
                                    "concept_name": concept["name"],
                                    "concept_type": concept.get("type", ""),
                                }
                            )

            linked_entities.append(linked_entity)

        return linked_entities

    def _resolve_entity_relationships(self, entities: List[Dict]) -> List[Dict]:
        """Identify relationships between entities"""
        relationships = []

        # Map entities by type for easier access
        entity_by_type = {}
        for i, entity in enumerate(entities):
            entity_type = entity.get("type", "")
            if entity_type not in entity_by_type:
                entity_by_type[entity_type] = []
            entity_by_type[entity_type].append((i, entity))

        # Relate biomarkers to tumors
        if "biomarker" in entity_by_type and "tumor" in entity_by_type:
            for biomarker_idx, biomarker in entity_by_type["biomarker"]:
                for tumor_idx, tumor in entity_by_type["tumor"]:
                    # Check if they might be related (e.g., by proximity in text)
                    if (
                        abs(biomarker.get("start", 0) - tumor.get("start", 0)) < 500
                    ):  # Within 500 chars
                        relationships.append(
                            {
                                "source": biomarker_idx,
                                "target": tumor_idx,
                                "type": "associated_with",
                            }
                        )

        # Relate staging to tumors
        if "staging" in entity_by_type and "tumor" in entity_by_type:
            for staging_idx, staging in entity_by_type["staging"]:
                for tumor_idx, tumor in entity_by_type["tumor"]:
                    if abs(staging.get("start", 0) - tumor.get("start", 0)) < 500:
                        relationships.append(
                            {
                                "source": staging_idx,
                                "target": tumor_idx,
                                "type": "classifies",
                            }
                        )

        # Relate treatments to conditions
        if "medication" in entity_by_type and "condition" in entity_by_type:
            for med_idx, med in entity_by_type["medication"]:
                for cond_idx, cond in entity_by_type["condition"]:
                    if abs(med.get("start", 0) - cond.get("start", 0)) < 500:
                        relationships.append(
                            {"source": med_idx, "target": cond_idx, "type": "treats"}
                        )

        # Relate temporal information to other entities
        if "temporal" in entity_by_type:
            for temp_idx, temp in entity_by_type["temporal"]:
                # Find entities that are mentioned close to this temporal marker
                for other_type, other_entities in entity_by_type.items():
                    if other_type != "temporal":
                        for other_idx, other in other_entities:
                            if (
                                abs(temp.get("start", 0) - other.get("start", 0)) < 200
                            ):  # Closer proximity
                                relationships.append(
                                    {
                                        "source": other_idx,
                                        "target": temp_idx,
                                        "type": "occurred_at",
                                    }
                                )

        return relationships

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities"""
        unique_entities = []
        seen_spans = set()

        for entity in entities:
            # Create a key based on start, end and type
            span_key = (entity.get("start"), entity.get("end"), entity.get("type"))

            if span_key not in seen_spans:
                seen_spans.add(span_key)
                unique_entities.append(entity)

        return unique_entities

    def _count_entities_by_type(self, entities: List[Dict]) -> Dict:
        """Count entities by type"""
        counts = {}

        for entity in entities:
            entity_type = entity.get("type", "unknown")
            if entity_type not in counts:
                counts[entity_type] = 0
            counts[entity_type] += 1

        return counts

    def _construct_timeline(self, entities: List[Dict]) -> List[Dict]:
        """Construct a patient timeline from temporal and related entities"""
        timeline = []

        # Filter temporal entities
        temporal_entities = [e for e in entities if e.get("type") == "temporal"]

        # Create timeline events
        for i, entity in enumerate(entities):
            if entity.get("type") == "temporal":
                # Find related non-temporal entities
                related_entities = []

                for j, other in enumerate(entities):
                    if other.get("type") != "temporal":
                        # Check if they are close in the text
                        if abs(entity.get("start", 0) - other.get("start", 0)) < 200:
                            related_entities.append(j)

                # Create timeline event
                timeline_event = {
                    "temporal_entity": i,
                    "temporal_text": entity.get("text", ""),
                    "temporal_value": entity.get("properties", {}).get("value", ""),
                    "normalized_date": entity.get("properties", {}).get(
                        "normalized_date", ""
                    ),  # Add default empty string
                    "related_entities": related_entities,
                }

                timeline.append(timeline_event)

        # Sort timeline by date if possible
        # Use a helper function to provide a key that handles None values
        def sort_key(x):
            date_val = x.get("normalized_date", "")
            return date_val if date_val else ""  # Return empty string for None values

        timeline.sort(key=sort_key)

        return timeline


# Custom extractors for cancer-specific entity types


class TumorExtractor:
    """Extracts tumor characteristics"""

    def extract(self, text: str, document_type: str) -> List[Dict]:
        entities = []

        # Tumor location
        location_patterns = [
            r"\b(tumor|mass|lesion|carcinoma)\s+(?:in|of|on|at)\s+(?:the\s+)?(\w+)\b",
            r"\b(\w+)\s+(tumor|mass|lesion|carcinoma)\b",
            r"\b(\w+)\s+cancer\b",
        ]

        for pattern in location_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if "tumor|mass|lesion|carcinoma" in pattern:
                    tumor_type = match.group(1)
                    location = match.group(2)
                else:
                    tumor_type = "cancer"
                    location = match.group(1)

                entities.append(
                    {
                        "text": match.group(0),
                        "type": "tumor",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {"location": location, "tumor_type": tumor_type},
                    }
                )

        # Tumor size
        size_patterns = [
            r"(tumor|mass|lesion)\s+(?:size|measuring)\s+(\d+\.?\d*)\s*(mm|cm)",
            r"(\d+\.?\d*)\s*(mm|cm)\s+(tumor|mass|lesion)",
            r"(tumor|mass|lesion)\s+(\d+\.?\d*)\s*[×x]\s*(\d+\.?\d*)\s*[×x]?\s*(?:(\d+\.?\d*)\s*)?(mm|cm)",  # 3D measurements
        ]

        for pattern in size_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Handle different pattern matches
                if "[×x]" in pattern:  # 3D measurement
                    tumor_type = match.group(1)
                    length = match.group(2)
                    width = match.group(3)
                    depth = match.group(4) if match.group(4) else None
                    unit = match.group(5)

                    entities.append(
                        {
                            "text": match.group(0),
                            "type": "tumor",
                            "start": match.start(),
                            "end": match.end(),
                            "properties": {
                                "tumor_type": tumor_type,
                                "length": float(length),
                                "width": float(width),
                                "depth": float(depth) if depth else None,
                                "unit": unit,
                            },
                        }
                    )
                else:  # Simple size
                    try:
                        if match.group(1) in ["tumor", "mass", "lesion"]:
                            tumor_type = match.group(1)
                            size = match.group(2)
                            unit = match.group(3)
                        else:
                            size = match.group(1)
                            unit = match.group(2)
                            tumor_type = match.group(3)

                        entities.append(
                            {
                                "text": match.group(0),
                                "type": "tumor",
                                "start": match.start(),
                                "end": match.end(),
                                "properties": {
                                    "tumor_type": tumor_type,
                                    "size": float(size),
                                    "unit": unit,
                                },
                            }
                        )
                    except (IndexError, ValueError):
                        # Skip if pattern matching doesn't work as expected
                        pass

        return entities


class StagingExtractor:
    """Extracts cancer staging information"""

    def extract(self, text: str, document_type: str) -> List[Dict]:
        entities = []

        # TNM staging
        tnm_patterns = [
            r"\b(c|p|y|r|a|u)?T(\d+[a-z]?)N(\d+[a-z]?)M(\d+[a-z]?)\b",  # Combined format: T2N0M0
            r"\b(?:clinical|pathologic|post-treatment)?\s*(?:stage|classification)?\s*:?\s*(T(\d+[a-z]?))\s*,?\s*(N(\d+[a-z]?))\s*,?\s*(M(\d+[a-z]?))\b",  # Separated: T2, N0, M0
        ]

        for pattern in tnm_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if "c|p|y|r|a|u" in pattern:  # Combined format with prefix
                    prefix = match.group(1) or ""
                    t_stage = match.group(2)
                    n_stage = match.group(3)
                    m_stage = match.group(4)
                else:  # Separated format
                    prefix = ""
                    t_stage = match.group(2)
                    n_stage = match.group(4)
                    m_stage = match.group(6)

                entities.append(
                    {
                        "text": match.group(0),
                        "type": "staging",
                        "subtype": "tnm",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {
                            "prefix": prefix.upper() if prefix else None,
                            "T": t_stage,
                            "N": n_stage,
                            "M": m_stage,
                        },
                    }
                )

        # General stage
        stage_patterns = [
            r"\b(?:clinical|pathologic|pathological)?\s*stage\s*:?\s*(0|[IVX]+[a-z]?)\b",
            r"\bstage\s+(?:is|as)\s+(0|[IVX]+[a-z]?)\b",
            r"\b(?:AJCC|FIGO)\s+stage\s+(0|[IVX]+[a-z]?)\b",
        ]

        for pattern in stage_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                stage = match.group(1)

                entities.append(
                    {
                        "text": match.group(0),
                        "type": "staging",
                        "subtype": "stage",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {
                            "stage": stage.upper(),
                        },
                    }
                )

        # Grade information
        grade_patterns = [
            r"\bgrade\s+(\d+)\b",
            r"\bwell\s+differentiated\b",
            r"\bmoderately\s+differentiated\b",
            r"\bpoorly\s+differentiated\b",
            r"\bundifferentiated\b",
        ]

        for pattern in grade_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if "grade" in pattern:
                    grade = match.group(1)
                    grade_text = f"Grade {grade}"
                else:
                    grade_text = match.group(0)
                    grade = {
                        "well differentiated": "1",
                        "moderately differentiated": "2",
                        "poorly differentiated": "3",
                        "undifferentiated": "4",
                    }.get(grade_text.lower(), None)

                entities.append(
                    {
                        "text": match.group(0),
                        "type": "staging",
                        "subtype": "grade",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {"grade": grade, "grade_text": grade_text},
                    }
                )

        return entities


class BiomarkerExtractor:
    """Extracts biomarker status information"""

    def extract(self, text: str, document_type: str) -> List[Dict]:
        entities = []

        # Common biomarkers in oncology
        biomarkers = {
            "breast": [
                "er",
                "estrogen receptor",
                "pr",
                "progesterone receptor",
                "her2",
                "her-2",
                "brca1",
                "brca2",
            ],
            "lung": ["egfr", "alk", "ros1", "pd-l1", "pdl1", "kras", "braf"],
            "colorectal": [
                "kras",
                "nras",
                "braf",
                "msi",
                "microsatellite instability",
                "msi-h",
                "mss",
            ],
            "melanoma": ["braf", "nras", "c-kit"],
            "general": [
                "pd-l1",
                "pdl1",
                "tmb",
                "tumor mutational burden",
                "msi",
                "microsatellite instability",
            ],
        }

        # Flatten biomarker list
        all_biomarkers = []
        for category, markers in biomarkers.items():
            all_biomarkers.extend(markers)

        # Remove duplicates
        all_biomarkers = list(set(all_biomarkers))

        # Biomarker status patterns
        for biomarker in all_biomarkers:
            # Use word boundary for shorter biomarkers
            if len(biomarker) <= 3:
                biomarker_pattern = r"\b" + re.escape(biomarker) + r"\b"
            else:
                biomarker_pattern = re.escape(biomarker)

            status_patterns = [
                biomarker_pattern
                + r"\s+(?:is|was)?\s*(positive|negative|equivocal|overexpressed|not expressed|borderline)",
                biomarker_pattern
                + r"(?:\s+status)?(?:\s+is|\:)?\s*(positive|negative|\+|\-|equivocal|overexpressed|not expressed|borderline)",
                r"(positive|negative|equivocal|overexpressed|not expressed|borderline)\s+(?:for)?\s+"
                + biomarker_pattern,
                biomarker_pattern
                + r"\s+expression(?:\s+is)?\s*(positive|negative|high|low|equivocal|borderline)",
                r"(high|low|positive|negative)\s+expression\s+of\s+"
                + biomarker_pattern,
            ]

            for pattern in status_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Identify which group contains the status
                    status_group = None
                    for i in range(1, match.lastindex + 1 if match.lastindex else 1):
                        if match.group(i) and match.group(i).lower() in [
                            "positive",
                            "negative",
                            "equivocal",
                            "+",
                            "-",
                            "overexpressed",
                            "not expressed",
                            "borderline",
                            "high",
                            "low",
                        ]:
                            status_group = i
                            break

                    if status_group is None:
                        continue

                    status = match.group(status_group)

                    # Map status values to standard terms
                    if status.lower() in ["+", "overexpressed", "high"]:
                        status = "positive"
                    elif status.lower() in ["-", "not expressed", "low"]:
                        status = "negative"

                    # Determine which category this biomarker belongs to
                    category = "general"
                    for cat, markers in biomarkers.items():
                        if biomarker in markers:
                            category = cat
                            break

                    entities.append(
                        {
                            "text": match.group(0),
                            "type": "biomarker",
                            "start": match.start(),
                            "end": match.end(),
                            "properties": {
                                "name": biomarker,
                                "status": status.lower(),
                                "category": category,
                            },
                        }
                    )

        # Percentage patterns (e.g., 'PD-L1 90%')
        percentage_patterns = [
            r"("
            + "|".join(re.escape(b) for b in all_biomarkers)
            + r")\s+(\d+(?:\.\d+)?)%",
            r"(\d+(?:\.\d+)?)%\s+("
            + "|".join(re.escape(b) for b in all_biomarkers)
            + r")",
        ]

        for pattern in percentage_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.group(1) in all_biomarkers:
                    biomarker = match.group(1)
                    percentage = float(match.group(2))
                else:
                    biomarker = match.group(2)
                    percentage = float(match.group(1))

                # Determine category
                category = "general"
                for cat, markers in biomarkers.items():
                    if biomarker in markers:
                        category = cat
                        break

                entities.append(
                    {
                        "text": match.group(0),
                        "type": "biomarker",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {
                            "name": biomarker,
                            "percentage": percentage,
                            "status": "positive" if percentage > 0 else "negative",
                            "category": category,
                        },
                    }
                )

        return entities


class TreatmentExtractor:
    """Extracts treatment details"""

    def extract(self, text: str, document_type: str) -> List[Dict]:
        entities = []

        # Common cancer treatments
        treatments = {
            "chemotherapy": [
                "doxorubicin",
                "cyclophosphamide",
                "paclitaxel",
                "docetaxel",
                "carboplatin",
                "cisplatin",
                "gemcitabine",
                "5-fu",
                "5-fluorouracil",
                "capecitabine",
                "adriamycin",
                "taxol",
                "taxotere",
                "folfox",
                "folfiri",
                "ac-t",
                "tc",
                "capeox",
            ],
            "targeted_therapy": [
                "trastuzumab",
                "herceptin",
                "pertuzumab",
                "perjeta",
                "lapatinib",
                "tykerb",
                "imatinib",
                "gleevec",
                "erlotinib",
                "tarceva",
                "gefitinib",
                "iressa",
                "bevacizumab",
                "avastin",
                "rituximab",
                "rituxan",
                "cetuximab",
                "erbitux",
            ],
            "immunotherapy": [
                "pembrolizumab",
                "keytruda",
                "nivolumab",
                "opdivo",
                "atezolizumab",
                "tecentriq",
                "durvalumab",
                "imfinzi",
                "ipilimumab",
                "yervoy",
                "immune checkpoint inhibitor",
                "pd-1 inhibitor",
                "pd-l1 inhibitor",
                "ctla-4 inhibitor",
            ],
            "radiation": [
                "radiation therapy",
                "radiotherapy",
                "external beam radiation",
                "imrt",
                "sbrt",
                "brachytherapy",
                "gamma knife",
                "stereotactic radiosurgery",
                "proton therapy",
                "xrt",
                "radiation",
            ],
            "surgery": [
                "surgery",
                "resection",
                "mastectomy",
                "lumpectomy",
                "partial mastectomy",
                "lobectomy",
                "radical mastectomy",
                "colectomy",
                "hemicolectomy",
                "prostatectomy",
                "orchiectomy",
                "hysterectomy",
                "oophorectomy",
                "nephrectomy",
                "pneumonectomy",
            ],
            "hormone_therapy": [
                "tamoxifen",
                "nolvadex",
                "aromatase inhibitor",
                "letrozole",
                "femara",
                "anastrozole",
                "arimidex",
                "exemestane",
                "aromasin",
                "fulvestrant",
                "faslodex",
                "bicalutamide",
                "casodex",
                "leuprolide",
                "lupron",
            ],
        }

        # Flatten treatment lists
        all_treatments = {}
        for category, drugs in treatments.items():
            for drug in drugs:
                all_treatments[drug] = category

        # Treatment patterns
        for treatment, category in all_treatments.items():
            # Short treatments need word boundaries
            if len(treatment) <= 3:
                treatment_pattern = r"\b" + re.escape(treatment) + r"\b"
            else:
                treatment_pattern = re.escape(treatment)

            # Look for treatment mentions with dosage
            dosage_patterns = [
                treatment_pattern + r"\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|mg/m2|mg/kg)",
                treatment_pattern
                + r"\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|mg/m2|mg/kg)(?:/(?:day|week|cycle))?",
                r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|mg/m2|mg/kg)\s+(?:of\s+)?"
                + treatment_pattern,
            ]

            for pattern in dosage_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Extract treatment name and dosage
                    if match.group(1) and match.group(1).replace(".", "", 1).isdigit():
                        # First group is dosage
                        dosage = match.group(1)
                        unit = match.group(2)
                        treatment_name = treatment
                    else:
                        # First group is treatment
                        treatment_name = treatment
                        try:
                            dosage = match.group(2)
                            unit = match.group(3)
                        except (IndexError, AttributeError):
                            dosage = None
                            unit = None

                    entities.append(
                        {
                            "text": match.group(0),
                            "type": "treatment",
                            "subtype": category,
                            "start": match.start(),
                            "end": match.end(),
                            "properties": {
                                "name": treatment_name,
                                "dosage": float(dosage)
                                if dosage and dosage.replace(".", "", 1).isdigit()
                                else None,
                                "unit": unit,
                                "category": category,
                            },
                        }
                    )

            # Simple treatment mentions
            simple_pattern = treatment_pattern
            for match in re.finditer(simple_pattern, text, re.IGNORECASE):
                # Check if this match overlaps with any existing entities
                overlap = False
                for entity in entities:
                    if (
                        match.start() >= entity["start"]
                        and match.start() < entity["end"]
                    ) or (
                        match.end() > entity["start"] and match.end() <= entity["end"]
                    ):
                        overlap = True
                        break

                if not overlap:
                    entities.append(
                        {
                            "text": match.group(0),
                            "type": "treatment",
                            "subtype": category,
                            "start": match.start(),
                            "end": match.end(),
                            "properties": {"name": treatment, "category": category},
                        }
                    )

        # Treatment regimen patterns
        regimen_patterns = [
            r"\b(AC-T|TC|FOLFOX|FOLFIRI|FEC|CMF|TAC|TCH|CHOP|R-CHOP|ABVD|BEP|MVAC|ECF)\b",
            r"\b((?:neo)?adjuvant\s+(?:chemo|radio|hormone)?therapy)\b",
            r"\b(systemic\s+therapy)\b",
            r"\b((?:neo)?adjuvant\s+treatment)\b",
        ]

        for pattern in regimen_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                regimen = match.group(1)

                # Determine category based on regimen name
                category = "regimen"
                if any(
                    chemo in regimen.lower()
                    for chemo in ["chemo", "folfox", "folfiri", "ac-t", "tc"]
                ):
                    category = "chemotherapy"
                elif any(rad in regimen.lower() for rad in ["radio"]):
                    category = "radiation"
                elif any(hormone in regimen.lower() for hormone in ["hormone"]):
                    category = "hormone_therapy"

                entities.append(
                    {
                        "text": match.group(0),
                        "type": "treatment",
                        "subtype": "regimen",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {"name": regimen, "category": category},
                    }
                )

        return entities


class ResponseExtractor:
    """Extracts treatment response assessments"""

    def extract(self, text: str, document_type: str) -> List[Dict]:
        entities = []

        # Response patterns (RECIST criteria)
        response_patterns = [
            r"\b(complete\s+(?:response|remission))\b",
            r"\b(partial\s+response)\b",
            r"\b(stable\s+disease)\b",
            r"\b(progressive\s+disease|progression)\b",
            r"\bcr\b",
            r"\bpr\b",
            r"\bsd\b",
            r"\bpd\b",
            r"\bno\s+evidence\s+of\s+(?:disease|recurrence)\b",
            r"\bned\b",
            r"\bresponse\s+to\s+(?:treatment|therapy)\s+(?:was|is)?\s+(complete|partial|minimal|none)\b",
        ]

        for pattern in response_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                response_text = match.group(0)

                # Determine response category
                if re.search(
                    r"complete|cr\b|no\s+evidence|ned", response_text, re.IGNORECASE
                ):
                    response_type = "complete response"
                elif re.search(r"partial|pr\b", response_text, re.IGNORECASE):
                    response_type = "partial response"
                elif re.search(r"stable|sd\b", response_text, re.IGNORECASE):
                    response_type = "stable disease"
                elif re.search(r"progress|pd\b", response_text, re.IGNORECASE):
                    response_type = "progressive disease"
                else:
                    response_type = "unspecified response"

                entities.append(
                    {
                        "text": response_text,
                        "type": "response",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {
                            "response_type": response_type,
                        },
                    }
                )

        # Measurement-based responses
        measurement_patterns = [
            r"(tumor|mass|lesion)\s+(?:size|diameter|measurement)?\s+(?:has|is)?\s+(decreased|increased|unchanged)",
            r"(tumor|mass|lesion)\s+(?:has|shows)\s+(decreased|increased)\s+(?:in\s+)?(?:size|diameter)",
            r"((\d+)%)\s+(reduction|increase|decrease)\s+in\s+(tumor|mass|lesion)",
            r"(tumor|mass|lesion)\s+(reduction|increase|decrease)\s+(?:of|by)\s+((\d+)%)",
        ]

        for pattern in measurement_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Handle different pattern types
                if "%" in pattern:
                    if match.group(1) and "%" in match.group(1):
                        percent = match.group(2)
                        direction = match.group(3)
                        subject = match.group(4)
                    else:
                        subject = match.group(1)
                        direction = match.group(2)
                        percent = match.group(4)
                else:
                    subject = match.group(1)
                    direction = match.group(2)
                    percent = None

                # Map direction to response type
                if direction.lower() in ["decreased", "reduction", "decrease"]:
                    response_type = (
                        "partial response"
                        if percent and int(percent) >= 30
                        else "minor response"
                    )
                elif direction.lower() in ["increased", "increase"]:
                    response_type = (
                        "progressive disease"
                        if percent and int(percent) >= 20
                        else "minor progression"
                    )
                else:
                    response_type = "stable disease"

                properties = {
                    "response_type": response_type,
                    "subject": subject,
                    "direction": direction,
                }

                if percent:
                    properties["percent_change"] = int(percent)

                entities.append(
                    {
                        "text": match.group(0),
                        "type": "response",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": properties,
                    }
                )

        return entities


# -----------------------------------------------------------------------------
# Main Processor: Combines all modules
# -----------------------------------------------------------------------------


class ClinicalOncologyProcessor:
    """
    Main processor that combines all modules for end-to-end processing of
    clinical oncology data.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Set defaults
        self.config = {
            "document_processor": {"use_ocr": True, "use_ehr": True},
            "tokenization": {
                "model": "gatortron",
                "max_length": 512,
                "segment_strategy": "sentence",
                "long_document_strategy": "sliding_window",
            },
            "entity_recognition": {
                "use_rules": True,
                "use_spacy": True,
                "use_deep_learning": False,
                "cancer_specific_extraction": True,
                "temporal_extraction": True,
                "ontologies": ["snomed_ct", "rxnorm"],
            },
            "processing_pipeline": ["document", "tokenization", "entity_recognition"],
            "output": {
                "include_raw_text": True,
                "include_tokens": True,
                "include_entities": True,
                "include_document_structure": True,
                "include_temporal_timeline": True,
            },
            **self.config,
        }

        # Initialize processors
        self._init_processors()

    def _init_processors(self):
        """Initialize all processor components"""
        # Document processors
        self.ocr_processor = (
            OCRProcessor(self.config.get("document_processor", {}))
            if self.config["document_processor"]["use_ocr"]
            else None
        )
        self.ehr_processor = (
            EHRProcessor(self.config.get("document_processor", {}))
            if self.config["document_processor"]["use_ehr"]
            else None
        )

        # Tokenization processor
        self.tokenization_processor = TokenizationProcessor(
            self.config["tokenization"].get("model", "bioclinicalbert"),
            self.config.get("tokenization", {}),
        )

        # Entity recognition processor
        self.entity_processor = EntityRecognitionProcessor(
            self.config.get("entity_recognition", {})
        )

    def process(
        self,
        input_path: Union[str, Path],
        save_output: bool = False,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Process a clinical oncology document.

        Args:
            input_path: Path to input document
            save_output: Whether to save processing output to file
            output_path: Path to save output (if None, use input_path with .json extension)

        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.logger.info(f"Processing file: {input_path}")

        # Initialize results
        results = {
            "file_path": str(input_path),
            "file_name": input_path.name,
            "file_extension": input_path.suffix,
            "processing_timestamp": pd.Timestamp.now().isoformat(),
        }

        # Step 1: Document Processing
        if "document" in self.config["processing_pipeline"]:
            document_result = self._process_document(input_path)
            results["document"] = document_result

            # Include raw text if configured
            if self.config["output"]["include_raw_text"]:
                results["text"] = document_result.get("full_text", "")

            # Include document structure if configured
            if self.config["output"]["include_document_structure"]:
                results["document_structure"] = document_result.get("structure", {})

        # Step 2: Tokenization
        if "tokenization" in self.config["processing_pipeline"]:
            tokenization_result = self._process_tokenization(
                document_result
                if "document" in self.config["processing_pipeline"]
                else input_path
            )

            # Include tokens if configured
            if self.config["output"]["include_tokens"]:
                results["tokenization"] = tokenization_result

        # Step 3: Entity Recognition
        if "entity_recognition" in self.config["processing_pipeline"]:
            entity_result = self._process_entity_recognition(
                document_result
                if "document" in self.config["processing_pipeline"]
                else input_path
            )

            # Include entities if configured
            if self.config["output"]["include_entities"]:
                results["entities"] = entity_result["entities"]
                results["entity_relationships"] = entity_result["relationships"]

            # Include temporal timeline if configured
            if self.config["output"]["include_temporal_timeline"]:
                results["temporal_timeline"] = entity_result["temporal_timeline"]

        # Save output if requested
        if save_output:
            if output_path is None:
                output_path = input_path.with_suffix(".json")
            else:
                output_path = Path(output_path)

            self._save_results(results, output_path)

        return results

    def _process_document(self, input_path: Path) -> Dict:
        """Process document with appropriate processor"""
        suffix = input_path.suffix.lower()

        if suffix in SUPPORTED_IMAGE_FORMATS and self.ocr_processor:
            return self.ocr_processor.process(input_path)
        elif suffix in SUPPORTED_EHR_FORMATS and self.ehr_processor:
            return self.ehr_processor.process(input_path)
        else:
            raise ValueError(f"No suitable processor for file type: {suffix}")

    def _process_tokenization(self, document: Union[Path, Dict]) -> Dict:
        """Process document with tokenization processor"""
        return self.tokenization_processor.tokenize_document(document)

    def _process_entity_recognition(self, document: Union[Path, Dict]) -> Dict:
        """Process document with entity recognition processor"""
        return self.entity_processor.process(document)

    def _save_results(self, results: Dict, output_path: Path) -> None:
        """Save processing results to file"""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {output_path}")

    def process_batch(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*",
        save_output: bool = False,
        output_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        Process a batch of clinical oncology documents.

        Args:
            input_dir: Directory containing input documents
            file_pattern: Glob pattern for selecting files
            save_output: Whether to save processing output to files
            output_dir: Directory to save output (if None, use input_dir)

        Returns:
            List of dictionaries with processing results
        """
        input_dir = Path(input_dir)

        if not input_dir.exists() or not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        # Get list of files matching pattern
        files = list(input_dir.glob(file_pattern))

        if not files:
            self.logger.warning(f"No files found matching pattern: {file_pattern}")
            return []

        self.logger.info(f"Processing {len(files)} files")

        # Process each file
        results = []

        for file_path in files:
            try:
                if output_dir:
                    output_path = Path(output_dir) / file_path.with_suffix(".json").name
                else:
                    output_path = file_path.with_suffix(".json")

                result = self.process(
                    file_path, save_output=save_output, output_path=output_path
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")

        return results

    def process_text(self, text: str, document_type: str = "unknown") -> Dict:
        """
        Process raw text.

        Args:
            text: Raw text to process
            document_type: Type of clinical document

        Returns:
            Dictionary with processing results
        """
        # Create a document-like structure
        document = {"full_text": text, "document_type": document_type}

        # Initialize results
        results = {
            "document_type": document_type,
            "processing_timestamp": pd.Timestamp.now().isoformat(),
        }

        # Include raw text if configured
        if self.config["output"]["include_raw_text"]:
            results["text"] = text

        # Step 1: Tokenization
        if "tokenization" in self.config["processing_pipeline"]:
            tokenization_result = self.tokenization_processor.tokenize_document(
                document
            )

            # Include tokens if configured
            if self.config["output"]["include_tokens"]:
                results["tokenization"] = tokenization_result

        # Step 2: Entity Recognition
        if "entity_recognition" in self.config["processing_pipeline"]:
            entity_result = self.entity_processor.process(document)

            # Include entities if configured
            if self.config["output"]["include_entities"]:
                results["entities"] = entity_result["entities"]
                results["entity_relationships"] = entity_result["relationships"]

            # Include temporal timeline if configured
            if self.config["output"]["include_temporal_timeline"]:
                results["temporal_timeline"] = entity_result["temporal_timeline"]

        return results


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        return json.load(f)


def initialize_processor(
    config_path: Optional[Union[str, Path]] = None,
) -> ClinicalOncologyProcessor:
    """Initialize processor with optional configuration"""
    if config_path:
        config = load_config(config_path)
    else:
        config = {}

    return ClinicalOncologyProcessor(config)


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------


def main():
    """Example usage of the Clinical Oncology Data Processing System"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration (optional)
    config = {
        "document_processor": {"use_ocr": True, "use_ehr": True},
        "tokenization": {
            "model": "bioclinicalbert",
            "max_length": 512,
            "segment_strategy": "sentence",
            "long_document_strategy": "sliding_window",
        },
        "entity_recognition": {
            "use_rules": True,
            "use_spacy": True,
            "use_deep_learning": True,
            "cancer_specific_extraction": True,
            "temporal_extraction": True,
            "ontologies": ["snomed_ct", "rxnorm"],
        },
    }

    # Initialize processor
    processor = ClinicalOncologyProcessor(config)

    # Example 1: Process a single file
    result = processor.process(
        "/mnt/d/TCGA/raw/TCGA-ACC/raw/TCGA-OR-A5JL/Pathology Report/c9f9dc8b-68ca-4a7e-be69-4d23df5a51a1/TCGA-OR-A5JL.BD8435C3-C525-472C-8B43-EC2CAE22B785.PDF",
        save_output=True,
    )

    # Example 2: Process a batch of files
    # results = processor.process_batch('path/to/documents', file_pattern='*.pdf', save_output=True)

    # Example 3: Process raw text
    sample_text = """
    Patient is a 58-year-old female with stage IIB (T2N1M0) infiltrating ductal carcinoma of the left breast.
    Diagnosis was made on 04/15/2023. Initial surgical management included a left mastectomy on 05/02/2023.
    Pathology showed a 3.2 cm tumor with 2/15 positive lymph nodes. Biomarker testing showed ER positive (95%),
    PR positive (80%), and HER2 negative. Patient started adjuvant chemotherapy with AC-T regimen on 06/01/2023.
    Completed 4 cycles of AC followed by 12 weeks of paclitaxel 80 mg/m2. Post-treatment imaging on 09/28/2023
    showed no evidence of disease. Patient started tamoxifen 20 mg daily on 10/10/2023 and will continue for 5 years.
    Follow-up appointment is scheduled for 01/15/2024.
    """
    # result = processor.process_text(sample_text, document_type='progress_note')

    # Print summary of results
    print(f"Document type: {result.get('document_type')}")
    print(f"Entities found: {len(result.get('entities', []))}")

    # Print identified entities by type
    entity_types = {}
    for entity in result.get("entities", []):
        entity_type = entity.get("type", "unknown")
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(entity)

    print("\nEntities by type:")
    for entity_type, entities in entity_types.items():
        print(f"  {entity_type}: {len(entities)}")

    # Print first few entities of each type
    print("\nSample entities:")
    for entity_type, entities in entity_types.items():
        print(f"\n  {entity_type.upper()}:")
        for i, entity in enumerate(entities[:3]):  # Show first 3 of each type
            print(f"    - {entity.get('text')}")

    # Print timeline events
    print("\nTemporal timeline:")
    for i, event in enumerate(
        result.get("temporal_timeline", [])[:5]
    ):  # Show first 5 events
        print(f"  Event {i + 1}: {event.get('temporal_text')}")
        related_entities = []
        for idx in event.get("related_entities", []):
            if idx < len(result.get("entities", [])):
                related_entities.append(result["entities"][idx].get("text", ""))
        if related_entities:
            print(f"    Related: {', '.join(related_entities)}")


if __name__ == "__main__":
    main()
