import re

import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from PIL import Image
from PyPDF2 import PdfReader


class PDF:
    """
    A class to handle reading and processing PDF reports.

    Attributes:
    -----------
    chunk_size : int
        The size of each text chunk.
    chunk_overlap : int
        The overlap size between text chunks.

    Methods:
    --------
    __init__(chunk_size=1000, chunk_overlap=200):
        Initializes the PDFreport with specified chunk size and overlap.

    read(pdf_file):
        Reads the text from a PDF file. If text extraction fails, it attempts to extract text from images in the PDF.

    chunk(text):
        Splits the given text into chunks based on the specified chunk size and overlap.

    load(pdf_file):
        Reads the PDF file, processes the text, and returns the text chunks.
    """

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read(self, pdf_file):
        text = ""
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            try:
                text += page.extract_text() or ""
            except:
                pass

        if not text:
            images = convert_from_path(pdf_file)
            for image in images:
                # Pre-process image (e.g., grayscale conversion, resizing, thresholding)
                processed_image = image.convert("L")
                processed_image = processed_image.resize(
                    tuple(2 * s for s in processed_image.size), Image.Resampling.LANCZOS
                )
                processed_image = processed_image.point(lambda p: p > 128 and 255)

                text += pytesseract.image_to_string(processed_image)

        # Clean up extracted text
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,;:!?\n-]", "", text)

        return text

    def chunk(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def load(self, pdf_file):
        report_text = self.read(pdf_file)
        if len(report_text) > 0:
            report_chunks = self.chunk(report_text)
            return report_chunks
        else:
            return None
