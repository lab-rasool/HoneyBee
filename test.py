import gc
import json
import os

import datasets
import minds
import numpy as np
import ollama
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from honeybee.loaders import (
    PDFreport,
    Scan,
    Slide,
    generate_summary_from_json,
    get_chunk_text,
    get_clinical_json_from_minds,
)
from honeybee.models import REMEDIS, HuggingFaceEmbedder, TissueDetector

load_dotenv()

max_patches = 500
embedding_dim = 7 * 7 * 2048
num_samples = 10


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


def PATHOLOGY_REPORT():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "Pathology Report"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)

    # --- CONFIGURATION ---
    embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-base")
    pdf_report = PDFreport(chunk_size=512, chunk_overlap=10)

    report_texts = []
    df["report_text"] = None
    df["embedding"] = None
    df["embedding_shape"] = None

    writer = None
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            file_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['id']}/{row['file_name']}"
            report_text = pdf_report.load(file_path)
            report_texts.append(report_text)

            if len(report_text) > 0:
                embeddings = embedding_model.generate_embeddings(report_text)
                df.at[index, "embedding_shape"] = embeddings.shape
                embeddings = embeddings.reshape(-1)
                embeddings = embeddings.tobytes()
                df.at[index, "embedding"] = embeddings
            else:
                df.at[index, "embedding"] = None
            df.at[index, "report_text"] = report_text

        except Exception as e:
            print(f"Error: {e}")
            report_texts.append(None)
            df.at[index, "embedding"] = None

        if writer is None:
            table = pa.Table.from_pandas(df.iloc[[index]])
            writer = pq.ParquetWriter(f"data/parquet/{MODALITY}.parquet", table.schema)
        else:
            table = pa.Table.from_pandas(df.iloc[[index]])
            writer.write_table(table)

    if writer is not None:
        writer.close()

    gc.collect()
    torch.cuda.empty_cache()

    dataset = datasets.load_dataset(
        "parquet",
        data_files=f"data/parquet/{MODALITY}.parquet",
        split="train",
    )
    dataset.save_to_disk(f"hf_dataset/{MODALITY}")


def WSI():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "Slide Image"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)

    # --- CONFIGURATION ---
    tissue_detector_model_path = "/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt"
    tissue_detector = TissueDetector(model_path=tissue_detector_model_path)
    embedding_model_path = "/mnt/d/Models/REMEDIS/onnx/path-50x1-remedis-s.onnx"

    df["embedding"] = None
    df["embedding_shape"] = None
    writer = None
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            slide_image_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['id']}/{row['file_name']}"
            slide = Slide(
                slide_image_path,
                tileSize=512,
                max_patches=500,
                visualize=False,
                tissue_detector=tissue_detector,
            )
            patches = slide.load_patches_concurrently(target_patch_size=224)
            embedding = REMEDIS.load_model_and_predict(embedding_model_path, patches)
            df.at[index, "embedding_shape"] = embedding.shape
            embedding = embedding.reshape(-1)
            embedding = embedding.tobytes()
            df.at[index, "embedding"] = embedding
        except Exception as e:
            print(f"Error: {e}")
            df.at[index, "embedding"] = None

        if writer is None:
            table = pa.Table.from_pandas(df.iloc[[index]])
            writer = pq.ParquetWriter(f"data/parquet/{MODALITY}.parquet", table.schema)
        else:
            table = pa.Table.from_pandas(df.iloc[[index]])
            writer.write_table(table)

        # --- CLEANUP ---
        del slide, patches, embedding, table
        gc.collect()
        torch.cuda.empty_cache()

    if writer is not None:
        writer.close()

    dataset = datasets.load_dataset(
        "parquet",
        data_files=f"data/parquet/{MODALITY}.parquet",
        split="train",
    )
    dataset.save_to_disk(f"hf_dataset/{MODALITY}")


class CustomDataset(Dataset):
    def __init__(self, hf_dataset, max_length=None):
        self.hf_dataset = hf_dataset
        self.max_length = max_length
        if not max_length:
            self.max_length = self.calculate_max_length()

    def calculate_max_length(self):
        embedding_shapes = [item["embedding_shape"] for item in self.hf_dataset]
        shapes_array = np.array(embedding_shapes)
        lengths = shapes_array[:, 0]
        max_len = lengths.max()
        return max_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        embedding = np.frombuffer(item["embedding"], dtype=np.float32)
        embedding = embedding.reshape(item["embedding_shape"])
        padding_size = self.max_length - embedding.shape[0]
        # Pad the embedding to the maximum length
        if padding_size > 0:
            embedding = F.pad(torch.tensor(embedding), (0, 0, 0, padding_size))
        else:
            embedding = torch.tensor(embedding)

        return {"embedding": embedding}


def main():
    # MODALITY = "Pathology Report"
    # PATHOLOGY_REPORT()
    MODALITY = "Slide Image"
    # WSI()

    # --- LOAD THE DATASET FROM HUGGING FACE ---
    dataset = datasets.load_from_disk(f"hf_dataset/{MODALITY}")
    # dataset = datasets.load_dataset(
    #     "parquet",
    #     data_files=f"data/parquet/{MODALITY}.parquet",
    #     split="train",
    # )
    # dataset = datasets.load_dataset(
    #     "Aakash-Tripathi/luad",
    #     split="train",
    # )

    for i in range(3):
        embedding = np.frombuffer(dataset["embedding"][i], dtype=np.float32)
        embedding = embedding.reshape(dataset["embedding_shape"][i])
        print(embedding.shape)

    # # --- Hugging Face dataset -> PyTorch DataLoader ---
    # torch_dataset = CustomDataset(dataset)
    # data_loader = DataLoader(torch_dataset, batch_size=64, shuffle=True)
    # for data in data_loader:
    #     print(data["embedding"].shape)


if __name__ == "__main__":
    main()
