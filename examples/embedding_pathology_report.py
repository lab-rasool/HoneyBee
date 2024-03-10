import json
import logging

import numpy as np
import pandas as pd
import pytesseract
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# log in a log file
logging.basicConfig(filename="./logs/pathology_report.log", level=logging.INFO)


class ClinicalEmbeddings:
    def __init__(self, model_name, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)

    def generate_embeddings(self, sentences):
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state
        return embeddings.cpu().numpy()


def manifest_to_df(manifest_path, modality):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Initialize an empty DataFrame for the modality
    modality_df = pd.DataFrame()

    # Process each patient in the manifest
    for patient in manifest:
        patient_id = patient["PatientID"]
        gdc_case_id = patient["gdc_case_id"]

        # Check if the current patient has the requested modality
        if modality in patient:
            # Convert the list of dictionaries into a DataFrame
            df = pd.DataFrame(patient[modality])
            # Add 'PatientID' and 'gdc_case_id' columns
            df["PatientID"] = patient_id
            df["gdc_case_id"] = gdc_case_id

            # Append the new data to the existing DataFrame for this modality
            modality_df = pd.concat([modality_df, df], ignore_index=True)

    # Check if the modality DataFrame is not empty before returning
    if not modality_df.empty:
        return modality_df
    else:
        return None


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


def main():
    DATA_DIR = "D:\\TCGA-LUAD"
    MANIFEST_PATH = "D:\\TCGA-LUAD\\manifest.json"
    embedding_model = ClinicalEmbeddings(model_name="UFNLP/gatortron-base")

    report_df = manifest_to_df(MANIFEST_PATH, "Pathology Report")

    fail_cases = 0
    report_texts = []
    report_embeddings_binary = []
    pdf_file_paths = get_file_paths(report_df, DATA_DIR)
    for pdf_file_path in tqdm(pdf_file_paths):
        
        #---------------------------------------------
        report_text = get_pdf_text(pdf_file_path)
        if len(report_text) > 0:
            report_texts.append(report_text)
            report_chunks = get_chunk_text(report_text)
            chunk_embeddings = []
            for chunk in report_chunks:
                chunk_embedding = embedding_model.generate_embeddings([chunk])
                chunk_embeddings.append(chunk_embedding)
            report_embedding = np.vstack(chunk_embeddings)
            report_embedding = np.array(pd.Series(list(report_embedding), dtype=object))
            report_embeddings_binary.append(report_embedding)
        else:
            fail_cases += 1
            report_texts.append(None)
            report_embeddings_binary.append(None)

    report_df["report_text"] = report_texts
    report_df["report_embedding"] = report_embeddings_binary
    report_df.to_parquet("./data/parquet/pathology_reports.parquet", index=False)

    if fail_cases > 0:
        print(f"Failed to process {fail_cases} files.")

    # report_df = pd.read_parquet("pathology_reports.parquet")
    # embeddings_series = report_df["report_embedding"]
    # for i in range(len(report_df)):
    #     embeddings_objects = np.array(embeddings_series[i].tolist(), dtype=object)
    #     print(embeddings_objects.shape)


if __name__ == "__main__":
    main()
