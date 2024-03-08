from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pytesseract

def get_pdf_text(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()

    if len(text) == 0:
        images = convert_from_path(pdf_file)
        for image in images:
            text += pytesseract.image_to_string(image)

    return text


def get_chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_file_paths(report_df, DATA_DIR):
    file_paths = []
    for index, row in report_df.iterrows():
        file_path = f"{DATA_DIR}/raw/{row['PatientID']}/Pathology Report/{row['id']}/{row['file_name']}"
        file_paths.append(file_path)
    return file_paths

class PDFreport:
    def __init__(self) -> None:
        pass
