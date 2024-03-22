from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pytesseract


class PDFreport:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read(self, pdf_file):
        text = ""
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()

        if len(text) == 0:
            images = convert_from_path(pdf_file)
            for image in images:
                text += pytesseract.image_to_string(image)

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
