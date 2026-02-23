"""
Document ingestion: converts files (PDF, images, EHR formats) or raw text
into a ClinicalDocument suitable for downstream NLP processing.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union

from ..types import ClinicalDocument

logger = logging.getLogger(__name__)

# File extensions routed to each handler
_IMAGE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
_EHR_EXTENSIONS = {".json", ".xml", ".csv", ".xlsx"}


class DocumentIngester:
    """Ingest clinical documents into ClinicalDocument objects."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.strategy = config.get("strategy", "hi_res")
        self.ocr_languages = config.get("ocr_languages", ["eng"])
        self.fallback_to_pymupdf = config.get("fallback_to_pymupdf", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, source: Union[str, Path, TextIO]) -> ClinicalDocument:
        """Ingest a document from a file path, file object, or raw text string."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                return self._ingest_file(path)
            # If it does not look like a path, treat it as raw text
            if not any(str(source).endswith(ext) for ext in _IMAGE_EXTENSIONS | _EHR_EXTENSIONS):
                return self.ingest_text(str(source))
            raise FileNotFoundError(f"File not found: {source}")
        # TextIO
        text = source.read()
        return self.ingest_text(text, document_type="stream")

    def ingest_text(self, text: str, document_type: str = "unknown") -> ClinicalDocument:
        """Wrap raw text into a ClinicalDocument with section detection."""
        sections = {}
        return ClinicalDocument(
            text=text,
            sections=sections,
            metadata={"source_type": "text", "document_type": document_type},
        )

    # ------------------------------------------------------------------
    # File dispatch
    # ------------------------------------------------------------------

    def _ingest_file(self, path: Path) -> ClinicalDocument:
        suffix = path.suffix.lower()
        if suffix in _IMAGE_EXTENSIONS:
            return self._ingest_image_or_pdf(path)
        if suffix in _EHR_EXTENSIONS:
            return self._ingest_ehr(path)
        # Fallback: read as plain text
        text = path.read_text(errors="replace")
        doc = self.ingest_text(text, document_type="text_file")
        doc.source_path = path
        return doc

    # ------------------------------------------------------------------
    # PDF / Image ingestion
    # ------------------------------------------------------------------

    def _ingest_image_or_pdf(self, path: Path) -> ClinicalDocument:
        suffix = path.suffix.lower()

        # Try unstructured first
        try:
            return self._ingest_with_unstructured(path)
        except ImportError:
            logger.debug("unstructured not installed, trying fallbacks")
        except Exception as exc:
            logger.warning("unstructured failed for %s: %s", path, exc)

        # Fallback: pymupdf for PDFs
        if suffix == ".pdf" and self.fallback_to_pymupdf:
            try:
                return self._ingest_pdf_pymupdf(path)
            except ImportError:
                logger.debug("pymupdf not installed")
            except Exception as exc:
                logger.warning("pymupdf failed for %s: %s", path, exc)

        # Fallback: PyPDF2 for PDFs
        if suffix == ".pdf":
            try:
                return self._ingest_pdf_pypdf2(path)
            except ImportError:
                logger.debug("PyPDF2 not installed")
            except Exception as exc:
                logger.warning("PyPDF2 failed for %s: %s", path, exc)

        # Fallback: pytesseract OCR for images
        if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
            try:
                return self._ingest_image_ocr(path)
            except ImportError:
                pass

        raise RuntimeError(
            f"Cannot ingest {path}. Install one of: "
            "unstructured[pdf,image], pymupdf, PyPDF2, pytesseract"
        )

    def _ingest_with_unstructured(self, path: Path) -> ClinicalDocument:
        from unstructured.partition.auto import partition

        elements = partition(str(path), strategy=self.strategy)
        text = "\n\n".join(str(el) for el in elements)
        sections = {}
        return ClinicalDocument(
            text=text,
            sections=sections,
            metadata={"source_type": "file", "extraction_method": "unstructured"},
            source_path=path,
        )

    def _ingest_pdf_pymupdf(self, path: Path) -> ClinicalDocument:
        import pymupdf

        doc = pymupdf.open(str(path))
        pages = [page.get_text() for page in doc]
        text = "\n\n".join(pages)
        sections = {}
        return ClinicalDocument(
            text=text,
            sections=sections,
            metadata={
                "source_type": "pdf",
                "extraction_method": "pymupdf",
                "num_pages": len(pages),
            },
            source_path=path,
        )

    def _ingest_pdf_pypdf2(self, path: Path) -> ClinicalDocument:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        sections = {}
        return ClinicalDocument(
            text=text,
            sections=sections,
            metadata={
                "source_type": "pdf",
                "extraction_method": "pypdf2",
                "num_pages": len(pages),
            },
            source_path=path,
        )

    def _ingest_image_ocr(self, path: Path) -> ClinicalDocument:
        import pytesseract
        from PIL import Image

        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang="+".join(self.ocr_languages))
        sections = {}
        return ClinicalDocument(
            text=text,
            sections=sections,
            metadata={"source_type": "image", "extraction_method": "tesseract"},
            source_path=path,
        )

    # ------------------------------------------------------------------
    # EHR ingestion
    # ------------------------------------------------------------------

    def _ingest_ehr(self, path: Path) -> ClinicalDocument:
        suffix = path.suffix.lower()
        if suffix == ".json":
            return self._ingest_json(path)
        elif suffix == ".xml":
            return self._ingest_xml(path)
        elif suffix in {".csv", ".xlsx"}:
            return self._ingest_tabular(path)
        raise ValueError(f"Unsupported EHR format: {suffix}")

    def _ingest_json(self, path: Path) -> ClinicalDocument:
        data = json.loads(path.read_text())
        text = self._flatten_dict(data)
        sections = {}
        return ClinicalDocument(
            text=text,
            sections=sections,
            metadata={"source_type": "ehr", "format": "json"},
            source_path=path,
        )

    def _ingest_xml(self, path: Path) -> ClinicalDocument:
        tree = ET.parse(str(path))
        texts = [elem.text for elem in tree.iter() if elem.text and elem.text.strip()]
        text = " ".join(texts)
        sections = {}
        return ClinicalDocument(
            text=text,
            sections=sections,
            metadata={"source_type": "ehr", "format": "xml"},
            source_path=path,
        )

    def _ingest_tabular(self, path: Path) -> ClinicalDocument:
        import pandas as pd

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        text = df.to_string(index=False)
        sections = {}
        return ClinicalDocument(
            text=text,
            sections=sections,
            metadata={"source_type": "ehr", "format": path.suffix.lower().strip(".")},
            source_path=path,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_dict(d: Any, prefix: str = "") -> str:
        """Recursively flatten a dict/list into readable text."""
        parts: list[str] = []
        if isinstance(d, dict):
            for k, v in d.items():
                parts.append(DocumentIngester._flatten_dict(v, prefix=k))
        elif isinstance(d, list):
            for item in d:
                parts.append(DocumentIngester._flatten_dict(item, prefix=prefix))
        else:
            label = f"{prefix}: " if prefix else ""
            parts.append(f"{label}{d}")
        return " ".join(parts)
