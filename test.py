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
from huggingface_hub import HfApi, HfFolder, login
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
from honeybee.models import REMEDIS, UNI, HuggingFaceEmbedder, TissueDetector

load_dotenv()


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

        # Ensure the writer is initialized with the correct schema
        if writer is None:
            schema = pa.Table.from_pandas(df.iloc[[index]]).schema
            writer = pq.ParquetWriter(
                f"/mnt/d/TCGA-LUAD/parquet/{MODALITY}.parquet", schema
            )

        table = pa.Table.from_pandas(df.iloc[[index]], schema=schema)
        try:
            writer.write_table(table)
        except ValueError as e:
            print(f"Schema mismatch error: {e}")
            # Re-initialize writer with new schema if needed
            schema = table.schema
            writer = pq.ParquetWriter(
                f"/mnt/d/TCGA-LUAD/parquet/{MODALITY}.parquet", schema
            )
            writer.write_table(table)

    if writer is not None:
        writer.close()

    gc.collect()
    torch.cuda.empty_cache()

    dataset = datasets.load_dataset(
        "parquet",
        data_files=f"/mnt/d/TCGA-LUAD/parquet/{MODALITY}.parquet",
        split="train",
    )
    dataset.save_to_disk(f"/mnt/d/TCGA-LUAD/hf_dataset/{MODALITY}")


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
            print(slide_image_path)
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
            writer = pq.ParquetWriter(
                f"/mnt/d/TCGA-LUAD/parquet/{MODALITY}.parquet", table.schema
            )
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
        data_files=f"/mnt/d/TCGA-LUAD/parquet/{MODALITY}.parquet",
        split="train",
    )
    dataset.save_to_disk(f"/mnt/d/TCGA-LUAD/hf_dataset/{MODALITY}")


def CT():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA/raw/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA/raw/TCGA-LUAD/manifest.json"
    MODALITY = "CT"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)
    # Ensure the output directory exists
    output_dir = "/mnt/d/TCGA-LUAD/parquet/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- CONFIGURATION ---
    embedding_model_path = "/mnt/d/Models/REMEDIS/onnx/cxr-50x1-remedis-m.onnx"

    # Define a consistent schema
    schema = pa.schema(
        [
            ("StudyInstanceUID", pa.string()),
            ("SeriesInstanceUID", pa.string()),
            ("SeriesDate", pa.string()),
            ("BodyPartExamined", pa.string()),
            ("SeriesNumber", pa.string()),
            ("Collection", pa.string()),
            ("Manufacturer", pa.string()),
            ("ManufacturerModelName", pa.string()),
            ("SoftwareVersions", pa.string()),
            ("Visibility", pa.string()),
            ("ImageCount", pa.int64()),
            ("PatientID", pa.string()),
            ("gdc_case_id", pa.string()),
            ("ProtocolName", pa.string()),
            ("SeriesDescription", pa.string()),
            ("embedding", pa.binary()),
            ("embedding_shape", pa.list_(pa.int64())),
            ("__index_level_0__", pa.int64()),
        ]
    )

    df["embedding"] = None
    df["embedding_shape"] = None
    writer = None
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            file_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['SeriesInstanceUID']}/{row['SeriesInstanceUID']}"
            scanner = Scan(file_path, modality="CT")
            patches = scanner.load_patches(target_patch_size=448)
            embedding = REMEDIS.load_model_and_predict(embedding_model_path, patches)
            df.at[index, "embedding_shape"] = embedding.shape
            embedding = embedding.reshape(-1)
            embedding = embedding.tobytes()
            df.at[index, "embedding"] = embedding
        except Exception as e:
            print(f"\033[91mError: {e}\033[0m")
            df.at[index, "embedding"] = None
            scanner = None
            patches = None
            embedding = None
            table = None

        if writer is None:
            table = pa.Table.from_pandas(df.iloc[[index]])
            writer = pq.ParquetWriter(
                f"/mnt/d/TCGA-LUAD/parquet/{MODALITY}.parquet", schema
            )
        else:
            table = pa.Table.from_pandas(df.iloc[[index]], schema=schema)
            writer.write_table(table)

        # --- CLEANUP ---
        del scanner, patches, embedding, table
        gc.collect()
        torch.cuda.empty_cache()

    if writer is not None:
        writer.close()

    dataset = datasets.load_dataset(
        "parquet",
        data_files=f"/mnt/d/TCGA-LUAD/parquet/{MODALITY}.parquet",
        split="train",
    )
    dataset.save_to_disk(f"/mnt/d/TCGA-LUAD/hf_dataset/{MODALITY}")


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


def uni_wsi():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "Slide Image"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)

    # --- CONFIGURATION ---
    tissue_detector_model_path = "/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt"
    tissue_detector = TissueDetector(model_path=tissue_detector_model_path)
    embedding_model_path = (
        "/mnt/d/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
    )
    uni = UNI()

    df["embedding"] = None
    df["embedding_shape"] = None
    writer = None
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            slide_image_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['id']}/{row['file_name']}"
            slide = Slide(
                slide_image_path,
                tileSize=512,
                max_patches=100,
                visualize=False,
                tissue_detector=tissue_detector,
            )
            patches = slide.load_patches_concurrently(target_patch_size=224)

            if patches.shape[0] == 0:
                # try to extract patches again with a larger max_patches
                slide = Slide(
                    slide_image_path,
                    tileSize=512,
                    max_patches=1000,
                    visualize=True,
                    tissue_detector=tissue_detector,
                )
                patches = slide.load_patches_concurrently(target_patch_size=224)

                if patches.shape[0] == 0:
                    # log the slide_image_path in a file
                    with open("empty_patches.txt", "a") as f:
                        f.write(f"{slide_image_path}\n")
                    raise ValueError("No patches extracted.")

            embedding = uni.load_model_and_predict(embedding_model_path, patches)
            df.at[index, "embedding_shape"] = embedding.shape
            embedding = embedding.reshape(-1)
            embedding = np.array(embedding, dtype=np.float32)
            embedding = embedding.tobytes()
            df.at[index, "embedding"] = embedding
        except Exception as e:
            print(f"Error: {e}")
            df.at[index, "embedding"] = None
            continue

        if writer is None:
            table = pa.Table.from_pandas(df.iloc[[index]])
            writer = pq.ParquetWriter(
                "/mnt/d/TCGA-LUAD/parquet/uni_Slide Image.parquet", table.schema
            )
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
        data_files="/mnt/d/TCGA-LUAD/parquet/uni_Slide Image.parquet",
        split="train",
    )
    dataset.save_to_disk("/mnt/d/TCGA-LUAD/hf_dataset/uni_Slide Image")


def CLINICAL_example():
    embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-base")
    json_objects = get_clinical_json_from_minds()
    df = []
    for case_id, patient_data in tqdm(json_objects.items()):
        summary = generate_summary_from_json(patient_data)

        if len(summary) > 0:
            summary_chunks = get_chunk_text(summary)
            chunk_embeddings = []
            for chunk in summary_chunks:
                chunk_embedding = embedding_model.generate_embeddings([chunk])
                chunk_embeddings.append(chunk_embedding)
            clinical_embedding = np.array(chunk_embeddings)
        else:
            clinical_embedding = None
        patient_data["text"] = summary
        patient_data["embedding_shape"] = clinical_embedding.shape

        clinical_embedding = clinical_embedding.reshape(-1)
        clinical_embedding = np.array(clinical_embedding, dtype=np.float32)
        clinical_embedding = clinical_embedding.tobytes()
        patient_data["embedding"] = clinical_embedding

        # Create a new dictionary for DataFrame conversion, excluding lists
        patient_data_for_df = {
            key: value
            for key, value in patient_data.items()
            if not isinstance(value, list)
        }
        df.append(patient_data_for_df)

    clinical_df = pd.DataFrame(df)
    clinical_df.to_parquet(
        "/mnt/d/TCGA-LUAD/parquet/Clinical Data.parquet", index=False
    )

    dataset = datasets.load_dataset(
        "parquet",
        data_files="/mnt/d/TCGA-LUAD/parquet/Clinical Data.parquet",
        split="train",
    )
    dataset.save_to_disk("/mnt/d/TCGA-LUAD/hf_dataset/Clinical Data")


def main():
    # MODALITY = "Pathology Report"
    # MODALITY = "Slide Image"
    # PATHOLOGY_REPORT()
    # WSI()
    # CT()
    # uni_wsi()
    # CLINICAL_example()
    pass

    # api = HfApi()
    # api.upload_folder(
    #     folder_path="/mnt/d/TCGA-LUAD/parquet/",
    #     repo_id="aakashtripathi/TCGA-LUAD",
    #     repo_type="dataset",
    #     multi_commits=True,
    #     multi_commits_verbose=True,
    # )

    # # --- LOAD THE DATASET FROM HUGGING FACE ---
    # dataset = datasets.load_dataset(
    #     "aakashtripathi/TCGA-LUAD", "pathology_report", split="train"
    # )
    # print(dataset)

    # --- LOAD THE DATASET FROM HUGGING FACE ---
    # dataset = datasets.load_from_disk("/mnt/d/TCGA-LUAD/hf_dataset/Slide Image")

    # dataset = datasets.load_dataset(
    #     "parquet",
    #     data_files=f"data/parquet/{MODALITY}.parquet",
    #     split="train",
    # )

    # # --- LOAD PARQUET IN CHUNKS ---
    # parquet_file = pq.ParquetFile("/mnt/d/TCGA-LUAD/parquet/uni_Slide Image.parquet")

    # parquet = pq.read_table(f'data/parquet/{MODALITY}.parquet')
    # df = parquet.to_pandas()
    # # delete the embedding column
    # del df["embedding"]
    # # save the dataframe to a csv file
    # df.to_csv(f'{MODALITY}.csv', index=False)

    # for batch in parquet_file.iter_batches(batch_size=1):
    #     batch_df = batch.to_pandas()
    #     embedding = np.frombuffer(batch_df["embedding"].iloc[0], dtype=np.float32)
    #     embedding = embedding.reshape(batch_df["embedding_shape"].iloc[0])
    #     print(embedding.shape)

    # # --- Hugging Face dataset directly ---
    # for i in range(10):
    #     embedding = np.frombuffer(dataset["embedding"][i], dtype=np.float32)
    #     embedding = embedding.reshape(dataset["embedding_shape"][i])
    #     print(embedding.shape)

    # --- Hugging Face dataset -> PyTorch DataLoader ---
    # torch_dataset = CustomDataset(dataset)
    # data_loader = DataLoader(torch_dataset, batch_size=32, shuffle=True)
    # for data in data_loader:
    #     print(data["embedding"].shape)


if __name__ == "__main__":
    main()
