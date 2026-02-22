import json
import logging
import re
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import dateutil.parser
import nltk
import numpy as np
import pandas as pd
import pdf2image
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from PyPDF2 import PdfReader
from transformers import AutoTokenizer

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt")
    except Exception:
        pass

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab")
    except Exception:
        pass

from nltk.tokenize import sent_tokenize, word_tokenize

# Import PDF loader from HoneyBee
try:
    from ...loaders import PDF
except ImportError:
    PDF = None

# Optional: spaCy for biomedical NER
try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    _SPACY_AVAILABLE = False

# Ontology mappings from subpackage
from .ontologies import ONTOLOGY_MAPPINGS

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
        "description": "Clinical BERT trained on MIMIC-III clinical notes",
        "gated": False,
    },
    "pubmedbert": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "max_length": 512,
        "description": "BERT trained on PubMed abstracts and full-text articles",
        "gated": False,
    },
    "gatortron": {
        "model_name": "UFNLP/gatortron-base",
        "max_length": 512,
        "description": "Clinical foundation model from University of Florida",
        "gated": True,
    },
    "clinicalt5": {
        "model_name": "razent/SciFive-base-PMC",
        "max_length": 512,
        "description": "T5 model fine-tuned on PubMed Central articles",
        "gated": False,
    },
    "biobert": {
        "model_name": "dmis-lab/biobert-v1.1",
        "max_length": 512,
        "description": "BioBERT for biomedical text mining",
        "gated": False,
    },
    "scibert": {
        "model_name": "allenai/scibert_scivocab_uncased",
        "max_length": 512,
        "description": "BERT model trained on scientific publications",
        "gated": False,
    },
    "sentence-transformers": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "max_length": 256,
        "description": "Fast lightweight sentence embeddings (general purpose)",
        "gated": False,
    },
}

# Cancer-specific patterns
CANCER_PATTERNS = {
    "tumor_type": (
        r"(?i)(adenocarcinoma|carcinoma|sarcoma|lymphoma"
        r"|leukemia|melanoma|glioma|mesothelioma)"
    ),
    "tumor_location": (
        r"(?i)(lung|breast|colon|prostate|pancreatic"
        r"|liver|brain|ovarian|stomach|kidney)"
    ),
    "tumor_grade": (
        r"(?i)(grade\s*[1-4]|grade\s*I{1,3}V?"
        r"|well[\s-]differentiated|moderately[\s-]differentiated"
        r"|poorly[\s-]differentiated)"
    ),
    "tumor_size": r"(\d+\.?\d*)\s*(cm|mm|centimeter|millimeter)",
    "tnm_stage": r"(?i)\b(T[0-4][a-c]?|N[0-3][a-c]?|M[0-1][a-c]?)\b",
    "stage_group": r"(?i)(stage\s*[0-4]|stage\s*I{1,3}V?[A-C]?)",
    "biomarker_status": (
        r"(?i)(ER|PR|HER2|EGFR|ALK|ROS1|BRAF|KRAS|PD-L1)"
        r"(?:\s*:\s*|\s+)(positive|negative|mutant|wild[\s-]?type|\+|-|\d+%)"
    ),
    "treatment_response": (
        r"(?i)(complete response|partial response"
        r"|stable disease|progressive disease|CR|PR|SD|PD)"
    ),
    "symptom": (
        r"(?i)\b(pain|fatigue|nausea|vomiting|fever|dyspnea"
        r"|cough|headache|diarrhea|constipation|weight loss)\b"
    ),
    "procedure": (
        r"(?i)\b(mastectomy|lobectomy|colectomy|biopsy"
        r"|nephrectomy|prostatectomy|hysterectomy"
        r"|radiation therapy|radiotherapy|immunotherapy)\b"
    ),
    "test": (
        r"(?i)\b(CT scan|MRI|PET scan|X-ray|ultrasound"
        r"|mammogram|colonoscopy|endoscopy|echocardiogram)\b"
    ),
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
    "rx": "prescription",
}

# spaCy entity label to HoneyBee type mapping
SPACY_ENTITY_TYPE_MAP = {
    "DISEASE": "condition",
    "CHEMICAL": "medication",
    "ENTITY": "condition",
    "PERSON": None,
    "DATE": "temporal",
    "ORG": None,
    "GPE": None,
    "CARDINAL": None,
}

# Deep learning NER entity label mapping (d4data/biomedical-ner-all)
DL_NER_ENTITY_TYPE_MAP = {
    "Disease_disorder": "condition",
    "Sign_symptom": "condition",
    "Medication": "medication",
    "Therapeutic_procedure": "procedure",
    "Diagnostic_procedure": "procedure",
    "Biological_structure": "anatomy",
    "Lab_value": "measurement",
    "Dosage": "dosage",
    "Duration": "temporal",
    "Date": "temporal",
    "Age": "temporal",
    "Clinical_event": "condition",
    "Nonbiological_location": None,
    "Detailed_description": None,
}

# Ambiguous clinical terms for disambiguation
AMBIGUOUS_TERMS = {
    "CA": {
        "cancer": {
            "type": "condition",
            "context_clues": [
                "diagnosed", "stage", "tumor", "metast", "oncol",
                "malignant", "carcinoma", "neoplasm", "chemotherapy",
                "radiation", "biopsy", "prognosis",
            ],
        },
        "calcium": {
            "type": "measurement",
            "context_clues": [
                "level", "mg/dL", "serum", "lab", "mmol",
                "hypercalcemia", "hypocalcemia", "ionized",
                "corrected", "blood",
            ],
        },
    },
    "PE": {
        "pulmonary embolism": {
            "type": "condition",
            "context_clues": [
                "pulmonary", "embolism", "clot", "DVT", "anticoagul",
                "heparin", "warfarin", "d-dimer", "CT angiography",
                "chest pain", "dyspnea",
            ],
        },
        "physical exam": {
            "type": "procedure",
            "context_clues": [
                "exam", "examination", "findings", "vitals",
                "auscultation", "palpation", "inspection",
                "normal", "unremarkable",
            ],
        },
    },
    "MS": {
        "multiple sclerosis": {
            "type": "condition",
            "context_clues": [
                "sclerosis", "demyelinat", "neurolog", "lesion",
                "relapsing", "remitting", "MRI brain", "oligoclonal",
                "interferon",
            ],
        },
        "mitral stenosis": {
            "type": "condition",
            "context_clues": [
                "mitral", "valve", "stenosis", "cardiac", "murmur",
                "atrial fibrillation", "rheumatic", "echocardiogram",
            ],
        },
    },
    "RA": {
        "rheumatoid arthritis": {
            "type": "condition",
            "context_clues": [
                "rheumatoid", "arthritis", "joint", "swelling",
                "methotrexate", "autoimmune", "RF", "anti-CCP",
                "inflammation", "synovitis",
            ],
        },
        "right atrium": {
            "type": "anatomy",
            "context_clues": [
                "atrium", "atrial", "cardiac", "heart", "chamber",
                "catheter", "pressure", "echocardiogram", "right",
            ],
        },
    },
    "MI": {
        "myocardial infarction": {
            "type": "condition",
            "context_clues": [
                "myocardial", "infarction", "heart attack", "troponin",
                "chest pain", "STEMI", "NSTEMI", "coronary",
                "angioplasty", "stent",
            ],
        },
        "mitral insufficiency": {
            "type": "condition",
            "context_clues": [
                "mitral", "insufficiency", "regurgitation", "valve",
                "murmur", "echocardiogram", "prolapse",
            ],
        },
    },
}

# Relationship type definitions for richer entity relationships
RELATIONSHIP_PATTERNS = {
    "has_location": {
        "type_pairs": [
            ("tumor", "anatomy"), ("condition", "anatomy"),
        ],
        "syntactic_patterns": [
            r"{entity1}\s+(?:of|in|at|on)\s+(?:the\s+)?{entity2}",
            r"{entity2}\s+{entity1}",
        ],
    },
    "has_biomarker": {
        "type_pairs": [
            ("tumor", "biomarker"), ("condition", "biomarker"),
        ],
        "syntactic_patterns": [
            r"{entity1}\s+.*?{entity2}",
        ],
    },
    "causes": {
        "type_pairs": [
            ("condition", "condition"), ("tumor", "condition"),
            ("medication", "condition"),
        ],
        "syntactic_patterns": [
            r"{entity1}\s+(?:caused?|causing|leads?\s+to|resulting\s+in)\s+.*?{entity2}",
            r"{entity2}\s+(?:caused?\s+by|due\s+to|secondary\s+to)\s+.*?{entity1}",
        ],
    },
    "contraindicates": {
        "type_pairs": [
            ("medication", "condition"),
        ],
        "syntactic_patterns": [
            r"{entity1}\s+(?:contraindicated\s+in|not\s+recommended\s+for)\s+.*?{entity2}",
        ],
    },
    "has_result": {
        "type_pairs": [
            ("procedure", "measurement"), ("condition", "measurement"),
        ],
        "syntactic_patterns": [
            r"{entity1}\s+(?:showed?|reveals?|demonstrates?|indicates?)\s+.*?{entity2}",
        ],
    },
    "temporal_relation": {
        "type_pairs": [
            ("condition", "temporal"), ("medication", "temporal"),
            ("procedure", "temporal"), ("tumor", "temporal"),
            ("staging", "temporal"), ("biomarker", "temporal"),
            ("measurement", "temporal"), ("response", "temporal"),
        ],
        "syntactic_patterns": [
            r"{entity1}\s+(?:on|at|since|from|until|during)\s+.*?{entity2}",
            r"{entity2}\s*[,:]\s*{entity1}",
        ],
    },
    "progression": {
        "type_pairs": [
            ("staging", "staging"),
        ],
        "syntactic_patterns": [
            r"{entity1}\s+(?:progressed?\s+to|advanced?\s+to)\s+.*?{entity2}",
        ],
    },
    "response_to": {
        "type_pairs": [
            ("response", "medication"), ("response", "procedure"),
        ],
        "syntactic_patterns": [
            r"{entity1}\s+(?:to|after|following|with)\s+.*?{entity2}",
        ],
    },
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
                "preprocessing": True,
            },
            "tokenization": {
                "model": "gatortron",
                "max_length": 512,
                "segment_strategy": "sentence",
                "long_document_strategy": "sliding_window",
                "stride": 128,
            },
            "entity_recognition": {
                "use_rules": True,
                "use_patterns": True,
                "cancer_specific_extraction": True,
                "temporal_extraction": True,
                "abbreviation_expansion": True,
                "ontologies": ["snomed_ct", "rxnorm"],
                "spacy_model": "en_core_sci_md",
                "dl_ner_model": "d4data/biomedical-ner-all",
            },
            "processing_pipeline": [
                "document", "tokenization", "entity_recognition",
            ],
            "output": {
                "include_raw_text": True,
                "include_tokens": True,
                "include_entities": True,
                "include_document_structure": True,
                "include_temporal_timeline": True,
            },
        }

        # Deep merge config with defaults
        self.config = self._deep_merge(default_config, self.config)

        # Lazy-loaded models
        self._spacy_nlp = None
        self._ner_pipeline = None

        # Initialize components
        self._init_components()

    def _deep_merge(self, default: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = default.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
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
            model_name = self.config.get("tokenization", {}).get(
                "model", "gatortron"
            )
            model_config = BIOMEDICAL_MODELS.get(model_name)
            if model_config:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_config["model_name"]
                    )
                    self.model_max_length = min(
                        model_config["max_length"],
                        self.config.get("tokenization", {}).get(
                            "max_length", 512
                        ),
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Could not load model "
                        f"{model_config['model_name']}: {e}"
                    )

        # Warn about spaCy if requested but not installed
        er_config = self.config.get("entity_recognition", {})
        if er_config.get("use_spacy", False) and not _SPACY_AVAILABLE:
            warnings.warn(
                "use_spacy=True but spaCy is not installed. "
                "Install with: pip install spacy scispacy. "
                "spaCy-based NER will be skipped.",
                stacklevel=2,
            )

        # Warn about DL NER if requested but transformers pipeline unavailable
        if er_config.get("use_deep_learning", False):
            try:
                from transformers import pipeline  # noqa: F401
            except ImportError:
                warnings.warn(
                    "use_deep_learning=True but transformers pipeline "
                    "is not available. DL NER will be skipped.",
                    stacklevel=2,
                )

        ontologies = er_config.get("ontologies", [])
        unsupported = [
            o for o in ontologies if o not in ONTOLOGY_MAPPINGS
        ]
        if unsupported:
            warnings.warn(
                f"Ontologies not yet supported and will be skipped: "
                f"{unsupported}. "
                f"Supported: {list(ONTOLOGY_MAPPINGS.keys())}",
                stacklevel=2,
            )

        # Compile regex patterns
        self.compiled_patterns = {}
        for pattern_name, pattern in CANCER_PATTERNS.items():
            self.compiled_patterns[pattern_name] = re.compile(pattern)

        # Initialize medical terms for OCR verification
        self.medical_terms = set(
            [
                "patient", "diagnosis", "treatment", "medication",
                "history", "cancer", "tumor", "carcinoma",
                "adenocarcinoma", "lymphoma", "metastasis", "stage",
                "grade", "biopsy", "surgery", "chemotherapy",
                "radiation", "therapy", "EGFR", "HER2", "ER", "PR",
                "PD-L1",
            ]
        )

    # ================================================================
    # spaCy NER (Feature #1)
    # ================================================================

    def _load_spacy_model(self):
        """Lazy-load spaCy model, cache in self._spacy_nlp."""
        if self._spacy_nlp is not None:
            return self._spacy_nlp
        if not _SPACY_AVAILABLE:
            return None
        model_name = self.config.get("entity_recognition", {}).get(
            "spacy_model", "en_core_sci_md"
        )
        try:
            self._spacy_nlp = spacy.load(model_name)
        except OSError:
            self.logger.warning(
                f"spaCy model '{model_name}' not found. "
                f"Install with: python -m spacy download {model_name}"
            )
            self._spacy_nlp = None
        return self._spacy_nlp

    def _extract_spacy_entities(self, text: str) -> List[Dict]:
        """Extract entities using spaCy NER."""
        nlp = self._load_spacy_model()
        if nlp is None:
            return []

        entities = []
        # Handle long texts by chunking
        max_len = nlp.max_length
        chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]

        for chunk_offset_idx, chunk in enumerate(chunks):
            offset = chunk_offset_idx * max_len
            doc = nlp(chunk)
            for ent in doc.ents:
                mapped_type = SPACY_ENTITY_TYPE_MAP.get(ent.label_)
                if mapped_type is None:
                    continue
                entities.append({
                    "text": ent.text,
                    "type": mapped_type,
                    "start": ent.start_char + offset,
                    "end": ent.end_char + offset,
                    "properties": {
                        "source": "spacy",
                        "spacy_label": ent.label_,
                    },
                })
        return entities

    # ================================================================
    # Deep Learning NER (Feature #2)
    # ================================================================

    def _load_ner_pipeline(self):
        """Lazy-load HuggingFace NER pipeline."""
        if self._ner_pipeline is not None:
            return self._ner_pipeline
        try:
            from transformers import pipeline

            model_name = self.config.get("entity_recognition", {}).get(
                "dl_ner_model", "d4data/biomedical-ner-all"
            )
            self._ner_pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",
            )
        except Exception as e:
            self.logger.warning(f"Could not load NER pipeline: {e}")
            self._ner_pipeline = None
        return self._ner_pipeline

    def _extract_dl_entities(self, text: str) -> List[Dict]:
        """Extract entities using deep learning NER pipeline."""
        pipe = self._load_ner_pipeline()
        if pipe is None:
            return []

        entities = []
        # Chunk for long texts (pipeline has token limit)
        chunk_size = 512
        chunks = []
        chunk_offsets = []

        # Build chunks by character position
        for i in range(0, len(text), chunk_size):
            end = min(i + chunk_size, len(text))
            chunks.append(text[i:end])
            chunk_offsets.append(i)

        for chunk, offset in zip(chunks, chunk_offsets):
            try:
                results = pipe(chunk)
            except Exception:
                continue
            for ent in results:
                # Strip BIO prefix from entity_group
                label = ent.get("entity_group", "")
                label = re.sub(r"^[BIO]-", "", label)
                mapped_type = DL_NER_ENTITY_TYPE_MAP.get(label)
                if mapped_type is None:
                    continue
                entities.append({
                    "text": ent.get("word", "").strip(),
                    "type": mapped_type,
                    "start": ent.get("start", 0) + offset,
                    "end": ent.get("end", 0) + offset,
                    "properties": {
                        "source": "deep_learning",
                        "dl_label": label,
                        "dl_confidence": round(
                            ent.get("score", 0.0), 4
                        ),
                    },
                })
        return entities

    # ================================================================
    # Term Disambiguation (Feature #4)
    # ================================================================

    def _disambiguate_entities(
        self, entities: List[Dict], text: str
    ) -> List[Dict]:
        """Disambiguate ambiguous clinical terms using context clues."""
        for entity in entities:
            term_upper = entity["text"].strip().upper()
            if term_upper not in AMBIGUOUS_TERMS:
                continue

            candidates = AMBIGUOUS_TERMS[term_upper]
            # Extract context window (200 chars before/after)
            ctx_start = max(0, entity["start"] - 200)
            ctx_end = min(len(text), entity["end"] + 200)
            context = text[ctx_start:ctx_end].lower()

            best_meaning = None
            best_score = 0

            for meaning, info in candidates.items():
                score = sum(
                    1 for clue in info["context_clues"]
                    if clue.lower() in context
                )
                if score > best_score:
                    best_score = score
                    best_meaning = meaning
                    best_type = info["type"]

            if best_score > 0 and best_meaning is not None:
                entity["type"] = best_type
                entity["properties"]["disambiguated"] = True
                entity["properties"]["disambiguated_to"] = best_meaning
                entity["properties"]["disambiguation_score"] = best_score
            else:
                entity["properties"]["ambiguous"] = True

        return entities

    # ================================================================
    # Abbreviation Expansion Tracking (Feature #6)
    # ================================================================

    def _expand_with_tracking(
        self, text: str
    ) -> Tuple[str, List[Dict]]:
        """Expand abbreviations and track where expansions occurred.

        Returns:
            Tuple of (expanded_text, expansion_map) where each entry in
            expansion_map is {start, end, original, expansion}.
        """
        if not self.config.get("entity_recognition", {}).get(
            "abbreviation_expansion", True
        ):
            return text, []

        expansion_map = []
        offset = 0

        for abbr, expansion in MEDICAL_ABBREVIATIONS.items():
            pattern = re.compile(
                rf"\b{re.escape(abbr)}\b", re.IGNORECASE
            )
            for match in pattern.finditer(text):
                orig_start = match.start()
                orig_end = match.end()
                # Calculate position in the expanded text
                new_start = orig_start + offset
                new_end = new_start + len(expansion)
                expansion_map.append({
                    "start": new_start,
                    "end": new_end,
                    "original": match.group(0),
                    "expansion": expansion,
                })
                offset += len(expansion) - (orig_end - orig_start)

        # Actually perform expansion
        expanded = text
        for abbr, expansion in MEDICAL_ABBREVIATIONS.items():
            expanded = re.sub(
                rf"\b{re.escape(abbr)}\b",
                expansion,
                expanded,
                flags=re.IGNORECASE,
            )

        return expanded, expansion_map

    def _annotate_expanded_entities(
        self,
        entities: List[Dict],
        expansion_map: List[Dict],
    ) -> List[Dict]:
        """Annotate entities that overlap with abbreviation expansions."""
        for entity in entities:
            for exp in expansion_map:
                # Check overlap
                if entity["start"] < exp["end"] and entity["end"] > exp["start"]:
                    entity["properties"]["expanded"] = {
                        "original": exp["original"],
                        "expanded_to": exp["expansion"],
                    }
                    break
        return entities

    # ================================================================
    # Summarize Strategy (Feature #5)
    # ================================================================

    def _summarize_tokenization(self, segments: List[str]) -> Dict:
        """TextRank-based extractive summarization with clinical boosting."""
        full_text = " ".join(segments)
        try:
            sentences = sent_tokenize(full_text)
        except Exception:
            sentences = full_text.split(". ")

        if not sentences:
            return self._tokenize_segments(segments)

        # Score sentences
        textrank_scores = self._score_sentences_textrank(sentences)
        clinical_scores = self._compute_clinical_relevance_scores(
            sentences
        )

        # Combine scores (0.6 textrank + 0.4 clinical)
        combined = {}
        for i in range(len(sentences)):
            combined[i] = (
                0.6 * textrank_scores.get(i, 0.0)
                + 0.4 * clinical_scores.get(i, 0.0)
            )

        # Select sentences within token budget
        token_budget = self.model_max_length - 10
        selected_indices = self._select_sentences_within_budget(
            sentences, combined, token_budget
        )

        # Preserve original order
        selected_indices.sort()
        summary_sentences = [sentences[i] for i in selected_indices]
        summary_text = " ".join(summary_sentences)

        # Tokenize the summary
        result = self._tokenize_single_text(summary_text)
        result["tokenization_strategy"] = "summarize"
        result["num_sentences_selected"] = len(selected_indices)
        result["num_sentences_total"] = len(sentences)
        result["summary_text"] = summary_text
        return result

    def _score_sentences_textrank(
        self, sentences: List[str]
    ) -> Dict[int, float]:
        """Score sentences using TextRank (PageRank on word-overlap graph)."""
        n = len(sentences)
        if n <= 1:
            return {0: 1.0} if n == 1 else {}

        # Build word sets
        word_sets = []
        for s in sentences:
            words = set(w.lower() for w in s.split() if len(w) > 2)
            word_sets.append(words)

        # Build similarity matrix
        similarity = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if not word_sets[i] or not word_sets[j]:
                    continue
                overlap = len(word_sets[i] & word_sets[j])
                denom = np.log(len(word_sets[i])) + np.log(
                    len(word_sets[j])
                )
                if denom > 0:
                    similarity[i][j] = overlap / denom
                    similarity[j][i] = similarity[i][j]

        # Try networkx PageRank, fall back to simple scoring
        try:
            import networkx as nx

            graph = nx.from_numpy_array(similarity)
            scores_dict = nx.pagerank(graph, max_iter=100)
            # Normalize
            max_score = max(scores_dict.values()) if scores_dict else 1.0
            if max_score > 0:
                return {
                    k: v / max_score for k, v in scores_dict.items()
                }
            return scores_dict
        except ImportError:
            # Fallback: sum of similarities as score
            raw = {i: float(similarity[i].sum()) for i in range(n)}
            max_score = max(raw.values()) if raw else 1.0
            if max_score > 0:
                return {k: v / max_score for k, v in raw.items()}
            return raw

    def _compute_clinical_relevance_scores(
        self, sentences: List[str]
    ) -> Dict[int, float]:
        """Score sentences by clinical term density."""
        scores = {}
        for i, s in enumerate(sentences):
            lower = s.lower()
            words = lower.split()
            if not words:
                scores[i] = 0.0
                continue
            count = sum(
                1 for t in self.medical_terms if t.lower() in lower
            )
            scores[i] = count / len(words)

        # Normalize
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            return {k: v / max_score for k, v in scores.items()}
        return scores

    def _select_sentences_within_budget(
        self,
        sentences: List[str],
        scores: Dict[int, float],
        token_budget: int,
    ) -> List[int]:
        """Greedily select highest-scoring sentences within token budget."""
        ranked = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )
        selected = []
        total_tokens = 0

        for idx, score in ranked:
            sent = sentences[idx]
            if self.tokenizer:
                n_tokens = len(self.tokenizer.tokenize(sent))
            else:
                n_tokens = len(sent.split())
            if total_tokens + n_tokens <= token_budget:
                selected.append(idx)
                total_tokens += n_tokens

        return selected

    # ================================================================
    # Richer Entity Relationships (Feature #8)
    # ================================================================

    def _extract_syntactic_relationships(
        self, entities: List[Dict], text: str
    ) -> List[Dict]:
        """Extract relationships using syntactic patterns."""
        relationships = []
        try:
            sents = sent_tokenize(text)
        except Exception:
            sents = text.split(". ")

        for sent in sents:
            # Find entities in this sentence
            sent_start = text.find(sent)
            if sent_start < 0:
                continue
            sent_end = sent_start + len(sent)

            ents_in_sent = []
            for idx, ent in enumerate(entities):
                if ent["start"] >= sent_start and ent["end"] <= sent_end:
                    ents_in_sent.append((idx, ent))

            # Check all pairs
            for i_idx, (gi, ent_i) in enumerate(ents_in_sent):
                for gj, ent_j in ents_in_sent[i_idx + 1:]:
                    rel = self._check_syntactic_pattern(
                        ent_i, ent_j, sent, text
                    )
                    if rel:
                        relationships.append({
                            "source": gi,
                            "target": gj,
                            "type": rel,
                            "confidence": 0.9,
                            "evidence": "syntactic_pattern",
                        })

        return relationships

    def _check_syntactic_pattern(
        self,
        ent1: Dict,
        ent2: Dict,
        sentence: str,
        full_text: str,
    ) -> Optional[str]:
        """Check if a syntactic pattern matches between two entities."""
        for rel_type, rel_info in RELATIONSHIP_PATTERNS.items():
            type_pairs = rel_info["type_pairs"]
            patterns = rel_info["syntactic_patterns"]

            # Check if entity types match any pair
            pair_match = False
            for t1, t2 in type_pairs:
                if ent1["type"] == t1 and ent2["type"] == t2:
                    pair_match = True
                    break
                if ent1["type"] == t2 and ent2["type"] == t1:
                    pair_match = True
                    break

            if not pair_match:
                continue

            # Check syntactic patterns
            e1_text = re.escape(ent1["text"])
            e2_text = re.escape(ent2["text"])
            for pat in patterns:
                concrete = pat.replace(
                    "{entity1}", e1_text
                ).replace("{entity2}", e2_text)
                try:
                    if re.search(concrete, sentence, re.IGNORECASE):
                        return rel_type
                except re.error:
                    continue

        return None

    def _extract_sentence_cooccurrence_relationships(
        self, entities: List[Dict], text: str
    ) -> List[Dict]:
        """Extract relationships from sentence co-occurrence."""
        relationships = []
        try:
            sents = sent_tokenize(text)
        except Exception:
            sents = text.split(". ")

        # Build compatible type pairs set
        compatible_pairs = set()
        for rel_info in RELATIONSHIP_PATTERNS.values():
            for t1, t2 in rel_info["type_pairs"]:
                compatible_pairs.add((t1, t2))
                compatible_pairs.add((t2, t1))

        for sent in sents:
            sent_start = text.find(sent)
            if sent_start < 0:
                continue
            sent_end = sent_start + len(sent)

            ents_in_sent = []
            for idx, ent in enumerate(entities):
                if ent["start"] >= sent_start and ent["end"] <= sent_end:
                    ents_in_sent.append((idx, ent))

            for i_idx, (gi, ent_i) in enumerate(ents_in_sent):
                for gj, ent_j in ents_in_sent[i_idx + 1:]:
                    pair = (ent_i["type"], ent_j["type"])
                    if pair in compatible_pairs:
                        # Determine relationship type
                        rel_type = self._determine_relationship(
                            ent_i, ent_j
                        )
                        if rel_type:
                            relationships.append({
                                "source": gi,
                                "target": gj,
                                "type": rel_type,
                                "confidence": 0.7,
                                "evidence": "sentence_cooccurrence",
                            })

        return relationships

    def _extract_proximity_relationships(
        self, entities: List[Dict], text: str
    ) -> List[Dict]:
        """Extract relationships based on proximity (within 50 chars)."""
        relationships = []
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1:], i + 1):
                if entity2["start"] - entity1["end"] < 50:
                    rel_type = self._determine_relationship(
                        entity1, entity2
                    )
                    if rel_type:
                        relationships.append({
                            "source": i,
                            "target": j,
                            "type": rel_type,
                            "confidence": 0.6,
                            "evidence": "proximity",
                        })
        return relationships

    # ================================================================
    # Public API methods
    # ================================================================

    def process(
        self, input_path: Union[str, Path], save_output: bool = False
    ) -> Dict:
        """Process a clinical document through the full pipeline."""
        input_path = Path(input_path)

        result = {
            "file_path": str(input_path),
            "file_name": input_path.name,
            "processing_timestamp": datetime.now().isoformat(),
            "processing_pipeline": self.config["processing_pipeline"],
        }

        try:
            if "document" in self.config["processing_pipeline"]:
                self.logger.info(f"Processing document: {input_path}")
                document_result = self._process_document(input_path)

                if self.config.get("output", {}).get(
                    "include_raw_text", True
                ):
                    result["text"] = document_result.get(
                        "text", document_result.get("full_text", "")
                    )

                result["document_processing"] = {
                    "method": document_result.get(
                        "extraction_method", "unknown"
                    ),
                    "confidence": document_result.get(
                        "avg_confidence",
                        document_result.get("confidence", 0),
                    ),
                }

                if self.config.get("output", {}).get(
                    "include_document_structure", True
                ):
                    structure = self._analyze_document_structure(
                        result.get("text", "")
                    )
                    result["document_structure"] = structure

            if (
                "tokenization" in self.config["processing_pipeline"]
                and result.get("text")
                and self.tokenizer
            ):
                self.logger.info("Tokenizing text")
                tokenization_result = self._tokenize_text(
                    result["text"],
                    result.get("document_structure"),
                )
                if self.config.get("output", {}).get(
                    "include_tokens", True
                ):
                    result["tokenization"] = tokenization_result

            if (
                "entity_recognition"
                in self.config["processing_pipeline"]
                and result.get("text")
            ):
                self.logger.info("Extracting entities")
                entity_result = self._extract_entities(result["text"])

                if self.config.get("output", {}).get(
                    "include_entities", True
                ):
                    result["entities"] = entity_result["entities"]
                    result["entity_relationships"] = entity_result[
                        "entity_relationships"
                    ]

                if (
                    self.config.get("output", {}).get(
                        "include_temporal_timeline", True
                    )
                    and entity_result["entities"]
                ):
                    timeline = self._extract_timeline(
                        entity_result["entities"], result["text"]
                    )
                    result["temporal_timeline"] = timeline

            if save_output:
                output_path = input_path.with_suffix(".clinical.json")
                self._save_output(result, output_path)
                result["output_saved"] = str(output_path)

        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {e}")
            result["error"] = str(e)

        return result

    def process_text(
        self, text: str, document_type: str = "unknown"
    ) -> Dict:
        """Process raw clinical text directly."""
        result = {
            "text": text,
            "document_type": document_type,
            "processing_timestamp": datetime.now().isoformat(),
        }

        try:
            if self.config.get("output", {}).get(
                "include_document_structure", True
            ):
                structure = self._analyze_document_structure(text)
                result["document_structure"] = structure

            if (
                "tokenization" in self.config["processing_pipeline"]
                and self.tokenizer
            ):
                tokenization_result = self._tokenize_text(
                    text, result.get("document_structure")
                )
                if self.config.get("output", {}).get(
                    "include_tokens", True
                ):
                    result["tokenization"] = tokenization_result

            if (
                "entity_recognition"
                in self.config["processing_pipeline"]
            ):
                entity_result = self._extract_entities(text)

                if self.config.get("output", {}).get(
                    "include_entities", True
                ):
                    result["entities"] = entity_result["entities"]
                    result["entity_relationships"] = entity_result[
                        "entity_relationships"
                    ]

                if (
                    self.config.get("output", {}).get(
                        "include_temporal_timeline", True
                    )
                    and entity_result["entities"]
                ):
                    timeline = self._extract_timeline(
                        entity_result["entities"], text
                    )
                    result["temporal_timeline"] = timeline

        except Exception as e:
            self.logger.error(f"Error processing text: {e}")
            result["error"] = str(e)

        return result

    def process_batch(
        self,
        input_dir: Union[str, Path],
        file_pattern: str = "*",
        save_output: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[Dict]:
        """Process multiple clinical documents in batch."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir

        files = list(input_dir.glob(file_pattern))
        self.logger.info(f"Found {len(files)} files to process")

        results = []
        for file_path in files:
            self.logger.info(f"Processing {file_path.name}")
            result = self.process(file_path, save_output=False)

            if save_output and not result.get("error"):
                output_path = (
                    output_dir / f"{file_path.stem}.clinical.json"
                )
                self._save_output(result, output_path)
                result["output_saved"] = str(output_path)

            results.append(result)

        return results

    # ================================================================
    # FHIR/HL7 convenience methods (Feature #9)
    # ================================================================

    def to_fhir(
        self, result: Dict, patient_id: Optional[str] = None
    ) -> Dict:
        """Convert processor output to FHIR R4 Bundle.

        Args:
            result: Output from process_text() or process().
            patient_id: Optional patient reference ID.

        Returns:
            FHIR Bundle as JSON-serializable dict.
        """
        from .interop.fhir_converter import FHIRConverter

        converter = FHIRConverter()
        return converter.to_fhir_bundle(result, patient_id)

    def process_fhir(self, bundle_json: Dict) -> Dict:
        """Parse FHIR Bundle and process extracted text.

        Args:
            bundle_json: FHIR Bundle as dict.

        Returns:
            Processing result from process_text().
        """
        from .interop.fhir_converter import FHIRConverter

        converter = FHIRConverter()
        parsed = converter.from_fhir_bundle(bundle_json)
        if parsed.get("text"):
            return self.process_text(parsed["text"])
        return {"text": "", "error": "No text extracted from bundle"}

    def process_hl7(self, message: str) -> Dict:
        """Parse HL7 v2 message and process extracted text.

        Args:
            message: Raw HL7 v2 message string.

        Returns:
            Processing result from process_text() with HL7 metadata.
        """
        from .interop.hl7_parser import HL7Parser

        parser = HL7Parser()
        parsed = parser.parse(message)
        result = self.process_text(parsed.get("text", ""))
        result["hl7_metadata"] = {
            "message_type": parsed.get("message_type", ""),
            "patient": parsed.get("patient", {}),
        }
        return result

    # ================================================================
    # Document processing methods
    # ================================================================

    def _process_document(self, file_path: Path) -> Dict:
        """Process document based on file type"""
        suffix = file_path.suffix.lower()

        if suffix in SUPPORTED_IMAGE_FORMATS:
            return self._process_image_document(file_path)
        elif suffix in SUPPORTED_EHR_FORMATS:
            return self._process_ehr_document(file_path)
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                return {"text": text, "extraction_method": "direct_read"}
            except Exception:
                raise ValueError(
                    f"Unsupported file format: {suffix}"
                )

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
            "confidence_scores": [],
        }

        try:
            pdf_reader = PdfReader(str(file_path))
            direct_text = ""

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    direct_text += page_text + "\n"

            if len(direct_text.strip()) > 100:
                result["full_text"] = direct_text
                result["extraction_method"] = "direct"
                return result

            if self.config.get("document_processor", {}).get(
                "use_ocr", True
            ):
                try:
                    images = pdf2image.convert_from_path(
                        str(file_path)
                    )
                    for i, image in enumerate(images):
                        page_result = self._ocr_image(image)
                        result["pages"].append(page_result)
                        result["full_text"] += (
                            page_result["text"] + "\n"
                        )
                        if page_result.get("confidence"):
                            result["confidence_scores"].append(
                                page_result["confidence"]
                            )

                    result["extraction_method"] = "ocr"
                    result["avg_confidence"] = (
                        np.mean(result["confidence_scores"])
                        if result["confidence_scores"]
                        else 0
                    )
                except Exception as e:
                    self.logger.warning(
                        f"OCR failed for {file_path}: {e}"
                    )
                    result["full_text"] = direct_text
                    result["extraction_method"] = "direct_partial"

        except Exception as e:
            self.logger.error(
                f"Error processing PDF {file_path}: {e}"
            )
            result["error"] = str(e)

        return result

    def _process_image(self, file_path: Path) -> Dict:
        """Process image file with OCR"""
        result = {
            "file_path": str(file_path),
            "file_type": "image",
            "text": "",
            "confidence": 0,
        }

        try:
            image = Image.open(file_path)
            ocr_result = self._ocr_image(image)
            result.update(ocr_result)
            result["extraction_method"] = "ocr"
        except Exception as e:
            self.logger.error(
                f"Error processing image {file_path}: {e}"
            )
            result["error"] = str(e)

        return result

    def _ocr_image(self, image: Image.Image) -> Dict:
        """Perform OCR on an image"""
        if self.config.get("document_processor", {}).get(
            "preprocessing", True
        ):
            image = self._preprocess_image(image)

        try:
            ocr_data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )
            text_parts = []
            confidences = []
            threshold = self.config.get("document_processor", {}).get(
                "confidence_threshold", 60
            )

            for i, conf in enumerate(ocr_data["conf"]):
                if int(conf) > threshold:
                    text = ocr_data["text"][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(int(conf))

            full_text = " ".join(text_parts)
            full_text = self._post_process_ocr_text(full_text)

            return {
                "text": full_text,
                "confidence": (
                    np.mean(confidences) if confidences else 0
                ),
                "word_count": len(text_parts),
            }

        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return {"text": "", "confidence": 0, "error": str(e)}

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR"""
        if image.mode != "L":
            image = image.convert("L")

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        image = image.filter(ImageFilter.MedianFilter(size=3))

        img_array = np.array(image)
        _, img_array = cv2.threshold(
            img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        img_array = cv2.medianBlur(img_array, 3)

        return Image.fromarray(img_array)

    def _post_process_ocr_text(self, text: str) -> str:
        """Post-process OCR text with medical terminology"""
        replacements = {
            "ml_": "mL",
            "mg_": "mg",
            "_": "",
            "  ": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        if self.config.get("entity_recognition", {}).get(
            "abbreviation_expansion", True
        ):
            for abbr, expansion in MEDICAL_ABBREVIATIONS.items():
                text = re.sub(
                    rf"\b{re.escape(abbr)}\b",
                    expansion,
                    text,
                    flags=re.IGNORECASE,
                )

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
        else:
            raise ValueError(
                f"Unsupported EHR format: {suffix}. "
                f"Supported formats: {SUPPORTED_EHR_FORMATS}"
            )

    def _process_json_ehr(self, file_path: Path) -> Dict:
        """Process JSON EHR data"""
        with open(file_path, "r") as f:
            data = json.load(f)

        text_parts = []
        self._flatten_json(data, text_parts)

        return {
            "text": "\n".join(text_parts),
            "extraction_method": "json_parse",
            "structured_data": data,
        }

    def _flatten_json(
        self, obj, result: List[str], prefix: str = ""
    ):
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
            "extraction_method": "xml_parse",
        }

    def _parse_xml_element(
        self, element, result: List[str], prefix: str = ""
    ):
        """Parse XML element recursively"""
        tag = (
            element.tag.split("}")[-1]
            if "}" in element.tag
            else element.tag
        )
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

        text_parts = []
        for idx, row in df.iterrows():
            text_parts.append(f"Record {idx + 1}:")
            for col, value in row.items():
                if pd.notna(value):
                    text_parts.append(f"  {col}: {value}")

        return {
            "text": "\n".join(text_parts),
            "extraction_method": "tabular_parse",
            "num_records": len(df),
        }

    # ================================================================
    # Document structure analysis
    # ================================================================

    def _analyze_document_structure(self, text: str) -> Dict:
        """Analyze document structure"""
        section_patterns = {
            "chief_complaint": (
                r"(?i)(chief complaint|cc|presenting complaint)"
            ),
            "history_present_illness": (
                r"(?i)(history of present illness|hpi|present illness)"
            ),
            "past_medical_history": (
                r"(?i)(past medical history|pmh|medical history)"
            ),
            "medications": (
                r"(?i)(medications|current medications|medication list)"
            ),
            "allergies": r"(?i)(allergies|drug allergies|allergy)",
            "physical_exam": (
                r"(?i)(physical exam|physical examination|pe)"
            ),
            "assessment_plan": (
                r"(?i)(assessment and plan|a&p|assessment|plan)"
            ),
            "laboratory": r"(?i)(laboratory|lab results|labs)",
            "imaging": (
                r"(?i)(imaging|radiology|ct|mri|xray|x-ray)"
            ),
            "pathology": r"(?i)(pathology|biopsy|cytology)",
        }

        lines = text.split("\n")
        sections = {}
        current_section = "unstructured"
        current_content = []

        for line in lines:
            section_found = False
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line):
                    if current_content:
                        sections[current_section] = "\n".join(
                            current_content
                        )
                    current_section = section_name
                    current_content = [line]
                    section_found = True
                    break

            if not section_found and line.strip():
                current_content.append(line)

        if current_content:
            sections[current_section] = "\n".join(current_content)

        return {
            "sections": sections,
            "headers": list(sections.keys()),
            "num_sections": len(sections),
        }

    # ================================================================
    # Tokenization
    # ================================================================

    def _tokenize_text(
        self, text: str, document_structure: Dict = None
    ) -> Dict:
        """Tokenize clinical text"""
        if not self.tokenizer:
            return {"error": "Tokenizer not initialized"}

        text = self._clean_text(text)
        segments = self._segment_text(text)

        if self._needs_long_document_handling(segments):
            return self._handle_long_document(segments)
        else:
            return self._tokenize_segments(segments)

    def _clean_text(self, text: str) -> str:
        """Clean clinical text"""
        text = " ".join(text.split())

        if self.config.get("entity_recognition", {}).get(
            "abbreviation_expansion", True
        ):
            for abbr, expansion in MEDICAL_ABBREVIATIONS.items():
                text = re.sub(
                    rf"\b{re.escape(abbr)}\b",
                    expansion,
                    text,
                    flags=re.IGNORECASE,
                )

        return text

    def _segment_text(self, text: str) -> List[str]:
        """Segment text based on strategy"""
        strategy = self.config.get("tokenization", {}).get(
            "segment_strategy", "sentence"
        )

        if strategy == "sentence":
            try:
                return sent_tokenize(text)
            except Exception:
                return text.split(". ")
        elif strategy == "paragraph":
            return [p.strip() for p in text.split("\n\n") if p.strip()]
        elif strategy == "fixed":
            try:
                words = word_tokenize(text)
            except Exception:
                words = text.split()
            chunk_size = 100
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i: i + chunk_size]))
            return chunks
        else:
            return [text]

    def _needs_long_document_handling(
        self, segments: List[str]
    ) -> bool:
        """Check if document needs special handling"""
        for segment in segments[:3]:
            tokens = self.tokenizer.tokenize(segment)
            if len(tokens) > self.model_max_length - 2:
                return True

        total_text = " ".join(segments)
        tokens = self.tokenizer.tokenize(total_text)
        return len(tokens) > self.model_max_length - 2

    def _handle_long_document(self, segments: List[str]) -> Dict:
        """Handle long documents"""
        strategy = self.config.get("tokenization", {}).get(
            "long_document_strategy", "sliding_window"
        )

        if strategy == "sliding_window":
            return self._sliding_window_tokenization(segments)
        elif strategy == "hierarchical":
            return self._hierarchical_tokenization(segments)
        elif strategy == "important_segments":
            return self._important_segments_tokenization(segments)
        elif strategy == "summarize":
            return self._summarize_tokenization(segments)
        else:
            return self._tokenize_segments(segments)

    def _sliding_window_tokenization(
        self, segments: List[str]
    ) -> Dict:
        """Tokenize with sliding window"""
        full_text = " ".join(segments)

        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )

        all_input_ids = full_encoding["input_ids"]
        stride = self.config.get("tokenization", {}).get("stride", 128)

        bos_id = self.tokenizer.cls_token_id
        eos_id = self.tokenizer.sep_token_id
        if eos_id is None:
            eos_id = self.tokenizer.eos_token_id

        prefix_ids = [bos_id] if bos_id is not None else []
        suffix_ids = [eos_id] if eos_id is not None else []
        num_special = len(prefix_ids) + len(suffix_ids)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        max_length = self.model_max_length - num_special

        windows = []
        for i in range(0, len(all_input_ids), max_length - stride):
            end_idx = min(i + max_length, len(all_input_ids))

            window_ids = (
                prefix_ids + all_input_ids[i:end_idx] + suffix_ids
            )
            window_mask = [1] * len(window_ids)

            padding_length = self.model_max_length - len(window_ids)
            if padding_length > 0:
                window_ids += [pad_id] * padding_length
                window_mask += [0] * padding_length

            windows.append({
                "input_ids": window_ids,
                "attention_mask": window_mask,
                "start_token_idx": i,
                "end_token_idx": end_idx,
            })

            if end_idx >= len(all_input_ids):
                break

        return {
            "input_ids": [w["input_ids"] for w in windows],
            "attention_mask": [w["attention_mask"] for w in windows],
            "num_windows": len(windows),
            "tokenization_strategy": "sliding_window",
        }

    def _hierarchical_tokenization(
        self, segments: List[str]
    ) -> Dict:
        """Tokenize preserving hierarchy"""
        results = {
            "sections": {},
            "tokenization_strategy": "hierarchical",
        }

        for i, segment in enumerate(segments):
            section_name = f"segment_{i}"
            section_result = self._tokenize_single_text(segment)
            results["sections"][section_name] = section_result

        return results

    def _important_segments_tokenization(
        self, segments: List[str]
    ) -> Dict:
        """Tokenize focusing on important segments"""
        clinical_terms = {
            "diagnosis", "cancer", "tumor", "stage", "grade",
            "treatment", "chemotherapy", "radiation", "surgery",
            "metastasis", "response",
        }

        segment_scores = []
        for segment in segments:
            lower_segment = segment.lower()
            score = sum(
                1 for term in clinical_terms
                if term in lower_segment
            )
            segment_scores.append((score, segment))

        segment_scores.sort(reverse=True, key=lambda x: x[0])

        selected_segments = []
        total_tokens = 0

        for score, segment in segment_scores:
            tokens = self.tokenizer.tokenize(segment)
            if total_tokens + len(tokens) < self.model_max_length - 100:
                selected_segments.append(segment)
                total_tokens += len(tokens)
            else:
                break

        combined_text = " ".join(selected_segments)
        result = self._tokenize_single_text(combined_text)
        result["tokenization_strategy"] = "important_segments"
        result["num_segments_selected"] = len(selected_segments)
        result["num_segments_total"] = len(segments)

        return result

    def _tokenize_segments(self, segments: List[str]) -> Dict:
        """Tokenize segments normally"""
        combined_text = " ".join(segments)
        return self._tokenize_single_text(combined_text)

    def _tokenize_single_text(self, text: str) -> Dict:
        """Tokenize a single text"""
        encoding = self.tokenizer(
            text,
            max_length=self.model_max_length,
            truncation=True,
            padding=True,
            return_tensors="np",
        )

        return {
            "input_ids": encoding["input_ids"][0].tolist(),
            "attention_mask": encoding["attention_mask"][0].tolist(),
            "num_tokens": int(np.sum(encoding["attention_mask"])),
            "tokenizer_name": self.config.get("tokenization", {}).get(
                "model", "unknown"
            ),
        }

    # ================================================================
    # Entity extraction
    # ================================================================

    def _extract_entities(self, text: str) -> Dict:
        """Extract entities from text"""
        er_config = self.config.get("entity_recognition", {})

        # Track abbreviation expansions
        expanded_text, expansion_map = self._expand_with_tracking(text)

        entities = []

        # Rule-based extraction
        if er_config.get("use_rules", True):
            rule_entities = self._extract_rule_based_entities(
                expanded_text
            )
            entities.extend(rule_entities)

        # Pattern-based extraction
        if er_config.get("use_patterns", True):
            pattern_entities = self._extract_pattern_based_entities(
                expanded_text
            )
            entities.extend(pattern_entities)

        # Cancer-specific extraction
        if er_config.get("cancer_specific_extraction", True):
            cancer_entities = self._extract_cancer_specific_entities(
                expanded_text
            )
            entities.extend(cancer_entities)

        # Temporal extraction
        if er_config.get("temporal_extraction", True):
            temporal_entities = self._extract_temporal_entities(
                expanded_text
            )
            entities.extend(temporal_entities)

        # spaCy NER
        if er_config.get("use_spacy", False) and _SPACY_AVAILABLE:
            spacy_entities = self._extract_spacy_entities(expanded_text)
            entities.extend(spacy_entities)

        # Deep Learning NER
        if er_config.get("use_deep_learning", False):
            dl_entities = self._extract_dl_entities(expanded_text)
            entities.extend(dl_entities)

        # Merge
        entities = self._merge_entities(entities)

        # Term disambiguation
        if er_config.get("term_disambiguation", False):
            entities = self._disambiguate_entities(
                entities, expanded_text
            )

        # Annotate expanded entities
        if expansion_map:
            entities = self._annotate_expanded_entities(
                entities, expansion_map
            )

        # Normalize to ontologies
        if er_config.get("ontologies", []):
            self._normalize_entities(entities)

        # Extract relationships (richer version)
        entity_relationships = self._extract_entity_relationships(
            entities, expanded_text
        )

        result = {
            "entities": entities,
            "entity_relationships": entity_relationships,
            "num_entities": len(entities),
        }

        if expansion_map:
            result["expanded_text"] = expanded_text

        return result

    def _extract_rule_based_entities(
        self, text: str
    ) -> List[Dict]:
        """Extract entities using rules"""
        entities = []

        # Medication patterns
        med_pattern = (
            r"(?i)(\w+)\s+(\d+)\s*(mg|mcg|g|ml|units?)"
            r"\s*(daily|bid|tid|qid|prn|po|iv|im|sq)"
        )
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
                    "source": "rule-based",
                },
            })

        # Lab values
        lab_pattern = (
            r"(?<!\w)([A-Za-z][A-Za-z\s]{0,30}?):\s*(\d+\.?\d*)\s*"
            r"(mg/dL|g/dL|mL|mg|mcg|mmol/L|mEq/L|U/L|IU/L|ng/mL"
            r"|pg/mL|µg/dL|%|cells/µL|mm3|x10[³⁹]?/[µu]?L|mmHg"
            r"|bpm|K/uL|M/uL|fL|pg)"
        )
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
                    "source": "rule-based",
                },
            })

        return entities

    def _extract_pattern_based_entities(
        self, text: str
    ) -> List[Dict]:
        """Extract entities using patterns"""
        entities = []

        condition_pattern = (
            r"(?i)(diagnosed with|history of|presents with"
            r"|suffering from)\s+([a-zA-Z\s]+?)"
            r"(?=\.|,|\s+and|\s+with|$)"
        )
        for match in re.finditer(condition_pattern, text):
            entities.append({
                "text": match.group(2).strip(),
                "type": "condition",
                "start": match.start(2),
                "end": match.end(2),
                "properties": {
                    "context": match.group(1),
                    "source": "pattern-based",
                },
            })

        return entities

    def _extract_cancer_specific_entities(
        self, text: str
    ) -> List[Dict]:
        """Extract cancer-specific entities"""
        entities = []

        for pattern_name, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                entity_type = self._get_entity_type_from_pattern(
                    pattern_name
                )

                entity = {
                    "text": match.group(0),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "properties": {
                        "pattern": pattern_name,
                        "source": "cancer-specific",
                    },
                }

                if pattern_name == "tumor_size":
                    entity["properties"]["size"] = match.group(1)
                    entity["properties"]["unit"] = match.group(2)
                elif pattern_name == "biomarker_status":
                    entity["properties"]["biomarker"] = match.group(1)
                    entity["properties"]["status"] = (
                        match.group(2)
                        if match.lastindex >= 2
                        else "unknown"
                    )

                entities.append(entity)

        return entities

    def _extract_temporal_entities(
        self, text: str
    ) -> List[Dict]:
        """Extract temporal entities"""
        entities = []

        date_patterns = [
            r"\d{1,2}/\d{1,2}/\d{2,4}",
            r"\d{1,2}-\d{1,2}-\d{2,4}",
            (
                r"(?:January|February|March|April|May|June|July"
                r"|August|September|October|November|December)"
                r"\s+\d{1,2},?\s+\d{4}"
            ),
        ]

        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    parsed_date = dateutil.parser.parse(
                        match.group(0)
                    )
                    entities.append({
                        "text": match.group(0),
                        "type": "temporal",
                        "start": match.start(),
                        "end": match.end(),
                        "properties": {
                            "temporal_type": "date",
                            "normalized_date": (
                                parsed_date.isoformat()
                            ),
                            "source": "temporal",
                        },
                    })
                except Exception:
                    pass

        return entities

    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge overlapping entities, preserving different types"""
        if not entities:
            return []

        entities.sort(key=lambda x: (x["start"], -x["end"]))

        merged = []
        current = entities[0]

        for entity in entities[1:]:
            if entity["start"] < current["end"]:
                if entity["type"] != current["type"]:
                    merged.append(current)
                    current = entity
                else:
                    if entity["end"] > current["end"] or (
                        self._is_more_specific(
                            entity["type"], current["type"]
                        )
                    ):
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
            "medication", "dosage", "procedure", "condition",
            "measurement", "anatomy", "temporal",
            "symptom", "test",
        ]

        try:
            return specificity_order.index(type1) < (
                specificity_order.index(type2)
            )
        except ValueError:
            return False

    def _normalize_entities(self, entities: List[Dict]):
        """Normalize entities to ontologies"""
        for entity in entities:
            normalized_text = entity["text"].lower().strip()

            ontology_links = []
            for ontology in self.config.get(
                "entity_recognition", {}
            ).get("ontologies", []):
                if ontology in ONTOLOGY_MAPPINGS:
                    mapping = ONTOLOGY_MAPPINGS[ontology].get(
                        normalized_text
                    )
                    if mapping:
                        ontology_links.append({
                            "ontology": ontology,
                            "concept_id": mapping["id"],
                            "concept_name": mapping["name"],
                            "match_type": "exact",
                        })

            if ontology_links:
                entity["properties"]["ontology_links"] = ontology_links

    def _extract_entity_relationships(
        self, entities: List[Dict], text: str
    ) -> List[Dict]:
        """Extract relationships using multiple strategies."""
        all_relationships = []

        # Strategy 1: Syntactic patterns (highest confidence)
        syntactic = self._extract_syntactic_relationships(
            entities, text
        )
        all_relationships.extend(syntactic)

        # Strategy 2: Sentence co-occurrence
        cooccurrence = (
            self._extract_sentence_cooccurrence_relationships(
                entities, text
            )
        )
        all_relationships.extend(cooccurrence)

        # Strategy 3: Proximity-based (lowest confidence)
        proximity = self._extract_proximity_relationships(
            entities, text
        )
        all_relationships.extend(proximity)

        # Deduplicate: keep highest-confidence for each (source, target)
        seen = {}
        for rel in all_relationships:
            key = (rel["source"], rel["target"])
            if key not in seen or rel["confidence"] > seen[key]["confidence"]:
                seen[key] = rel

        return list(seen.values())

    def _determine_relationship(
        self, entity1: Dict, entity2: Dict
    ) -> Optional[str]:
        """Determine relationship type between two entities."""
        t1, t2 = entity1["type"], entity2["type"]

        # Medication-dosage
        if t1 == "medication" and t2 == "dosage":
            return "has_dosage"
        # Condition-treatment
        if t1 == "condition" and t2 in ["medication", "procedure"]:
            return "treated_with"
        # Tumor-staging
        if t1 == "tumor" and t2 == "staging":
            return "has_stage"
        # Tumor/condition-anatomy
        if t1 in ("tumor", "condition") and t2 == "anatomy":
            return "has_location"
        # Tumor/condition-biomarker
        if t1 in ("tumor", "condition") and t2 == "biomarker":
            return "has_biomarker"
        # Any-temporal
        if t2 == "temporal" and t1 != "temporal":
            return "temporal_relation"
        # Response-medication/procedure
        if t1 == "response" and t2 in ("medication", "procedure"):
            return "response_to"
        # Staging-staging
        if t1 == "staging" and t2 == "staging":
            return "progression"
        # Procedure/test-measurement
        if t1 in ("procedure", "test") and t2 == "measurement":
            return "has_result"
        # Medication-condition (contraindicates checked in syntactic)
        if t1 == "medication" and t2 == "condition":
            return "contraindicates"

        return None

    def _get_entity_type_from_pattern(
        self, pattern_name: str
    ) -> str:
        """Map pattern to entity type"""
        mapping = {
            "tumor_type": "tumor",
            "tumor_location": "anatomy",
            "tumor_grade": "staging",
            "tumor_size": "measurement",
            "tnm_stage": "staging",
            "stage_group": "staging",
            "biomarker_status": "biomarker",
            "treatment_response": "response",
            "symptom": "condition",
            "procedure": "procedure",
            "test": "procedure",
        }

        return mapping.get(pattern_name, "condition")

    def _extract_timeline(
        self, entities: List[Dict], text: str
    ) -> List[Dict]:
        """Extract temporal timeline"""
        timeline_events = []

        temporal_entities = [
            e for e in entities if e["type"] == "temporal"
        ]

        for temporal in temporal_entities:
            related_entities = []

            for i, entity in enumerate(entities):
                if entity["type"] != "temporal":
                    if abs(entity["start"] - temporal["start"]) < 100:
                        related_entities.append(i)

            event = {
                "temporal_text": temporal["text"],
                "temporal_type": temporal["properties"].get(
                    "temporal_type", "unknown"
                ),
                "normalized_date": temporal["properties"].get(
                    "normalized_date"
                ),
                "related_entities": related_entities,
            }

            timeline_events.append(event)

        timeline_events.sort(
            key=lambda x: x.get("normalized_date", "9999")
        )

        return timeline_events

    # ================================================================
    # Output and statistics
    # ================================================================

    def _save_output(self, result: Dict, output_path: Path):
        """Save results to file"""
        try:
            def convert_arrays(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {
                        k: convert_arrays(v) for k, v in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_arrays(v) for v in obj]
                return obj

            result_serializable = convert_arrays(result)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    result_serializable,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            self.logger.info(f"Saved output to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving output: {e}")

    def get_summary_statistics(self, result: Dict) -> Dict:
        """Get summary statistics"""
        stats = {
            "text_length": len(result.get("text", "")),
            "num_entities": len(result.get("entities", [])),
            "num_relationships": len(
                result.get("entity_relationships", [])
            ),
            "num_timeline_events": len(
                result.get("temporal_timeline", [])
            ),
            "entity_types": {},
        }

        for entity in result.get("entities", []):
            entity_type = entity["type"]
            stats["entity_types"][entity_type] = (
                stats["entity_types"].get(entity_type, 0) + 1
            )

        return stats

    def generate_embeddings(
        self,
        text: Union[str, List[str]],
        model_name: Optional[str] = None,
        pooling_method: str = "mean",
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        """Generate embeddings for clinical text using any HuggingFace model."""
        try:
            from ...models import (
                HuggingFaceEmbedder,
                ModelAccessError,
                ModelNotFoundError,
            )
        except ImportError:
            raise ImportError(
                "HuggingFaceEmbedder not available. "
                "Please check installation."
            )

        if model_name is None:
            model_name = self.config.get("tokenization", {}).get(
                "model", "bioclinicalbert"
            )

        if model_name in BIOMEDICAL_MODELS:
            model_config = BIOMEDICAL_MODELS[model_name]
            hf_model_id = model_config["model_name"]
            default_max_length = model_config.get("max_length", 512)

            if model_config.get("gated", False):
                self.logger.warning(
                    f"Model '{model_name}' ({hf_model_id}) requires "
                    f"access approval."
                )

            self.logger.info(
                f"Using preset '{model_name}': "
                f"{model_config.get('description', '')}"
            )
        else:
            hf_model_id = model_name
            default_max_length = 512
            self.logger.info(
                f"Using HuggingFace model: {hf_model_id}"
            )

        final_max_length = (
            max_length if max_length is not None
            else default_max_length
        )

        try:
            self.logger.info(f"Loading model: {hf_model_id}")
            embedder = HuggingFaceEmbedder(
                model_name=hf_model_id,
                pooling_method=pooling_method,
                max_length=final_max_length,
            )
        except (ModelAccessError, ModelNotFoundError):
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to load model '{hf_model_id}': {e}"
            )
            recommended = (
                HuggingFaceEmbedder.get_recommended_open_models()
            )
            suggestion_msg = (
                "\n\nTry these open-access biomedical models:\n"
            )
            for model in recommended["biomedical"][:3]:
                suggestion_msg += f"  - {model['name']}\n"

            raise RuntimeError(
                f"Failed to load model '{hf_model_id}': "
                f"{type(e).__name__}: {e}" + suggestion_msg
            ) from e

        try:
            n = len(text) if isinstance(text, list) else 1
            self.logger.info(
                f"Generating embeddings for {n} text(s)"
            )
            embeddings = embedder.generate_embeddings(
                text, batch_size=batch_size
            )
            self.logger.info(
                f"Embeddings shape: {embeddings.shape}"
            )
            return embeddings

        except Exception as e:
            self.logger.error(
                f"Error during embedding generation: {e}"
            )
            raise RuntimeError(
                f"Failed to generate embeddings: "
                f"{type(e).__name__}: {e}\n"
                f"Model: {hf_model_id}, Pooling: {pooling_method}"
            ) from e
