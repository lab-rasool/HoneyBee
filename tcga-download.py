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

# from honeybee.loaders import (
#     PDFreport,
#     Scan,
#     Slide,
#     get_chunk_text,
# )
# from honeybee.models import REMEDIS, UNI, HuggingFaceEmbedder, TissueDetector

load_dotenv()


def download():
    query = "SELECT * FROM minds.clinical WHERE project_id like '%%TCGA%%'"
    df = minds.query(query)
    # get all unique project_ids and make a folder for each under /mnt/d/TCGA/
    project_ids = df["project_id"].unique()
    for project_id in project_ids:
        project_folder = f"D:\\TCGA\\raw\\{project_id}"
        manifest_file = os.path.join(project_folder, "manifest.json")
        os.makedirs(project_folder, exist_ok=True)

        # if the data is already downloaded, skip
        if os.path.exists(project_folder + "/raw/"):
            print(f"Skipping {project_id} as it is already downloaded.")
            continue
        query = f"SELECT * FROM minds.clinical WHERE project_id='{project_id}'"
        query_cohort = minds.build_cohort(
            query=query,
            output_dir=project_folder,
            manifest=manifest_file if os.path.exists(manifest_file) else None,
        )
        query_cohort.download(threads=8, exclude=["Slide Image"])


# def manifest_to_df(manifest_path, modality):
#     with open(manifest_path, "r") as f:
#         manifest = json.load(f)

#     # Initialize an empty DataFrame for the modality
#     modality_df = pd.DataFrame()

#     # Process each patient in the manifest
#     for patient in manifest:
#         patient_id = patient["PatientID"]
#         gdc_case_id = patient["gdc_case_id"]

#         # Check if the current patient has the requested modality
#         if modality in patient:
#             # Convert the list of dictionaries into a DataFrame
#             df = pd.DataFrame(patient[modality])
#             # Add 'PatientID' and 'gdc_case_id' columns
#             df["PatientID"] = patient_id
#             df["gdc_case_id"] = gdc_case_id

#             # Append the new data to the existing DataFrame for this modality
#             modality_df = pd.concat([modality_df, df], ignore_index=True)

#     # Check if the modality DataFrame is not empty before returning
#     if not modality_df.empty:
#         return modality_df
#     else:
#         return None


# def PATHOLOGY_REPORT(DATA_DIR, MANIFEST_PATH, parquet_path):
#     # --- THIS CAN BE IGNORED ---
#     MODALITY = "Pathology Report"
#     df = manifest_to_df(MANIFEST_PATH, MODALITY)

#     if df is None:
#         print(f"No {MODALITY} found for {DATA_DIR}")
#         return

#     # --- CONFIGURATION ---
#     embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-base")
#     pdf_report = PDFreport(chunk_size=512, chunk_overlap=10)

#     report_texts = []
#     df["report_text"] = None
#     df["embedding"] = None
#     df["embedding_shape"] = None
#     writer = None
#     for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
#         try:
#             file_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['id']}/{row['file_name']}"
#             report_text = pdf_report.load(file_path)
#             report_texts.append(report_text)

#             if len(report_text) > 0:
#                 if not isinstance(report_text, list):
#                     report_text = [report_text]
#                 df.at[index, "report_text"] = report_text
#                 embeddings = embedding_model.generate_embeddings(report_text)
#                 df.at[index, "embedding_shape"] = embeddings.shape
#                 embeddings = embeddings.reshape(-1).tobytes()
#                 df.at[index, "embedding"] = embeddings

#             else:
#                 print(f"Empty report or error processing: {file_path}")
#                 continue

#             if writer is None:
#                 table = pa.Table.from_pandas(df.iloc[[index]])
#                 writer = pq.ParquetWriter(parquet_path, table.schema)
#             else:
#                 table = pa.Table.from_pandas(df.iloc[[index]])
#                 writer.write_table(table)

#         except Exception as e:
#             print(f"Error processing: {file_path} - {e}")
#             continue

#     if writer is not None:
#         writer.close()

#     gc.collect()
#     torch.cuda.empty_cache()

#     # dataset = datasets.load_dataset(
#     #     "parquet",
#     #     data_files=parquet_path,
#     #     split="train",
#     # )
#     # dataset.save_to_disk(hf_dataset_path)


# def WSI():
#     # --- THIS CAN BE IGNORED ---
#     DATA_DIR = "/mnt/d/TCGA"
#     MANIFEST_PATH = "/mnt/d/TCGA/manifest.json"
#     MODALITY = "Slide Image"
#     df = manifest_to_df(MANIFEST_PATH, MODALITY)

#     # --- CONFIGURATION ---
#     tissue_detector_model_path = "/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt"
#     tissue_detector = TissueDetector(model_path=tissue_detector_model_path)
#     embedding_model_path = "/mnt/d/Models/REMEDIS/onnx/path-50x1-remedis-s.onnx"

#     df["embedding"] = None
#     df["embedding_shape"] = None
#     writer = None
#     for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
#         try:
#             slide_image_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['id']}/{row['file_name']}"
#             slide = Slide(
#                 slide_image_path,
#                 tileSize=512,
#                 max_patches=500,
#                 visualize=False,
#                 tissue_detector=tissue_detector,
#             )
#             patches = slide.load_patches_concurrently(target_patch_size=224)
#             embedding = REMEDIS.load_model_and_predict(embedding_model_path, patches)
#             df.at[index, "embedding_shape"] = embedding.shape
#             embedding = embedding.reshape(-1)
#             embedding = embedding.tobytes()
#             df.at[index, "embedding"] = embedding
#         except Exception as e:
#             print(f"Error: {e}")
#             df.at[index, "embedding"] = None

#         if writer is None:
#             table = pa.Table.from_pandas(df.iloc[[index]])
#             writer = pq.ParquetWriter(
#                 f"/mnt/d/TCGA/parquet/{MODALITY}.parquet", table.schema
#             )
#         else:
#             table = pa.Table.from_pandas(df.iloc[[index]])
#             writer.write_table(table)

#         # --- CLEANUP ---
#         del slide, patches, embedding, table
#         gc.collect()
#         torch.cuda.empty_cache()

#     if writer is not None:
#         writer.close()

#     dataset = datasets.load_dataset(
#         "parquet",
#         data_files=f"/mnt/d/TCGA/parquet/{MODALITY}.parquet",
#         split="train",
#     )
#     dataset.save_to_disk(f"/mnt/d/TCGA/hf_dataset/{MODALITY}")


# def CT():
#     # --- THIS CAN BE IGNORED ---
#     DATA_DIR = "/mnt/d/TCGA"
#     MANIFEST_PATH = "/mnt/d/TCGA/manifest.json"
#     MODALITY = "CT"
#     df = manifest_to_df(MANIFEST_PATH, MODALITY)

#     # --- CONFIGURATION ---
#     embedding_model_path = "/mnt/d/Models/REMEDIS/onnx/cxr-50x1-remedis-s.onnx"

#     # Define a consistent schema
#     schema = pa.schema(
#         [
#             ("StudyInstanceUID", pa.string()),
#             ("SeriesInstanceUID", pa.string()),
#             ("SeriesDate", pa.string()),
#             ("BodyPartExamined", pa.string()),
#             ("SeriesNumber", pa.string()),
#             ("Collection", pa.string()),
#             ("Manufacturer", pa.string()),
#             ("ManufacturerModelName", pa.string()),
#             ("SoftwareVersions", pa.string()),
#             ("Visibility", pa.string()),
#             ("ImageCount", pa.int64()),
#             ("PatientID", pa.string()),
#             ("gdc_case_id", pa.string()),
#             ("ProtocolName", pa.string()),
#             ("SeriesDescription", pa.string()),
#             ("embedding", pa.binary()),
#             ("embedding_shape", pa.list_(pa.int64())),
#             ("__index_level_0__", pa.int64()),
#         ]
#     )

#     df["embedding"] = None
#     df["embedding_shape"] = None
#     writer = None
#     for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
#         try:
#             file_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['SeriesInstanceUID']}/{row['SeriesInstanceUID']}"
#             scanner = Scan(file_path, modality="CT")
#             patches = scanner.load_patches(target_patch_size=224)
#             embedding = REMEDIS.load_model_and_predict(embedding_model_path, patches)
#             df.at[index, "embedding_shape"] = embedding.shape
#             embedding = embedding.reshape(-1)
#             embedding = embedding.tobytes()
#             df.at[index, "embedding"] = embedding
#         except Exception as e:
#             print(f"Error: {e}")
#             df.at[index, "embedding"] = None
#             scanner = None
#             patches = None
#             embedding = None
#             table = None

#         if writer is None:
#             table = pa.Table.from_pandas(df.iloc[[index]])
#             writer = pq.ParquetWriter(f"/mnt/d/TCGA/parquet/{MODALITY}.parquet", schema)
#         else:
#             try:
#                 table = pa.Table.from_pandas(df.iloc[[index]], schema=schema)
#                 writer.write_table(table)
#             except Exception as e:
#                 print(f"Error writing to Parquet: {e}")

#         # --- CLEANUP ---
#         del scanner, patches, embedding, table
#         gc.collect()
#         torch.cuda.empty_cache()

#     if writer is not None:
#         writer.close()

#     dataset = datasets.load_dataset(
#         "parquet",
#         data_files=f"/mnt/d/TCGA/parquet/{MODALITY}.parquet",
#         split="train",
#     )
#     dataset.save_to_disk(f"/mnt/d/TCGA/hf_dataset/{MODALITY}")


# class CustomDataset(Dataset):
#     def __init__(self, hf_dataset, max_length=None):
#         self.hf_dataset = hf_dataset
#         self.max_length = max_length
#         if not max_length:
#             self.max_length = self.calculate_max_length()

#     def calculate_max_length(self):
#         embedding_shapes = [item["embedding_shape"] for item in self.hf_dataset]
#         shapes_array = np.array(embedding_shapes)
#         lengths = shapes_array[:, 0]
#         max_len = lengths.max()
#         return max_len

#     def __len__(self):
#         return len(self.hf_dataset)

#     def __getitem__(self, idx):
#         item = self.hf_dataset[idx]
#         embedding = np.frombuffer(item["embedding"], dtype=np.float32)
#         embedding = embedding.reshape(item["embedding_shape"])
#         padding_size = self.max_length - embedding.shape[0]
#         # Pad the embedding to the maximum length
#         if padding_size > 0:
#             embedding = F.pad(torch.tensor(embedding), (0, 0, 0, padding_size))
#         else:
#             embedding = torch.tensor(embedding)

#         return {"embedding": embedding}


# def uni_wsi():
#     # --- THIS CAN BE IGNORED ---
#     DATA_DIR = "/mnt/d/TCGA"
#     MANIFEST_PATH = "/mnt/d/TCGA/manifest.json"
#     MODALITY = "Slide Image"
#     df = manifest_to_df(MANIFEST_PATH, MODALITY)

#     # --- CONFIGURATION ---
#     tissue_detector_model_path = "/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt"
#     tissue_detector = TissueDetector(model_path=tissue_detector_model_path)
#     embedding_model_path = (
#         "/mnt/d/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
#     )
#     uni = UNI()

#     df["embedding"] = None
#     df["embedding_shape"] = None
#     writer = None
#     for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
#         try:
#             slide_image_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['id']}/{row['file_name']}"
#             slide = Slide(
#                 slide_image_path,
#                 tileSize=512,
#                 max_patches=100,
#                 visualize=False,
#                 tissue_detector=tissue_detector,
#             )
#             patches = slide.load_patches_concurrently(target_patch_size=224)

#             if patches.shape[0] == 0:
#                 # try to extract patches again with a larger max_patches
#                 slide = Slide(
#                     slide_image_path,
#                     tileSize=512,
#                     max_patches=1000,
#                     visualize=True,
#                     tissue_detector=tissue_detector,
#                 )
#                 patches = slide.load_patches_concurrently(target_patch_size=224)

#                 if patches.shape[0] == 0:
#                     # log the slide_image_path in a file
#                     with open("empty_patches.txt", "a") as f:
#                         f.write(f"{slide_image_path}\n")
#                     raise ValueError("No patches extracted.")

#             embedding = uni.load_model_and_predict(embedding_model_path, patches)
#             df.at[index, "embedding_shape"] = embedding.shape
#             embedding = embedding.reshape(-1)
#             embedding = np.array(embedding, dtype=np.float32)
#             embedding = embedding.tobytes()
#             df.at[index, "embedding"] = embedding
#         except Exception as e:
#             print(f"Error: {e}")
#             df.at[index, "embedding"] = None
#             continue

#         if writer is None:
#             table = pa.Table.from_pandas(df.iloc[[index]])
#             writer = pq.ParquetWriter(
#                 "/mnt/d/TCGA/parquet/uni_Slide Image.parquet", table.schema
#             )
#         else:
#             table = pa.Table.from_pandas(df.iloc[[index]])
#             writer.write_table(table)

#         # --- CLEANUP ---
#         del slide, patches, embedding, table
#         gc.collect()
#         torch.cuda.empty_cache()

#     if writer is not None:
#         writer.close()

#     dataset = datasets.load_dataset(
#         "parquet",
#         data_files="/mnt/d/TCGA/parquet/uni_Slide Image.parquet",
#         split="train",
#     )
#     dataset.save_to_disk("/mnt/d/TCGA/parquet/uni_Slide Image")


# def process_group(group):
#     common_fields = {}
#     nested_objects = []
#     for col in group.columns:
#         unique_values = group[col].dropna().unique()
#         if len(unique_values) == 1:
#             # If only one unique value exists, it's a common field
#             common_fields[col] = unique_values[0]

#     # Create nested objects for fields that are not common
#     for idx, row in group.iterrows():
#         nested_object = {
#             col: row[col]
#             for col in group.columns
#             if col not in common_fields and pd.notna(row[col])
#         }
#         if nested_object:  # Only add if the nested object is not empty
#             nested_objects.append(nested_object)

#     return common_fields, nested_objects


# def generate_summary_from_json(patient_data):
#     # Initialize an empty list to store sentences
#     summary_sentences = []

#     # Iterate through each key-value pair in the JSON object
#     for key, value in patient_data.items():
#         # if the key is "case_id" then skip it
#         if key == "case_id" or key == "pathology_report_uuid":
#             continue

#         # remove all _ from the key
#         key = key.replace("_", " ")
#         sentence = f"{key}: {value};"

#         # if the value is a list, then skip it
#         if isinstance(value, list):
#             continue

#         summary_sentences.append(sentence)

#     # Compile all sentences into a single summary string
#     summary = " ".join(summary_sentences)

#     return summary


# def CLINICAL(project_id):
#     embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-base")
#     tables = minds.get_tables()
#     json_objects = {}
#     for table in tqdm(tables, desc="Getting data from tables"):
#         query = f"SELECT * FROM minds.{table} WHERE project_id='{project_id}'"
#         df = minds.query(query)
#         for case_id, group in tqdm(df.groupby("case_submitter_id"), leave=False):
#             if case_id not in json_objects:
#                 json_objects[case_id] = {}
#             common_fields, nested_objects = process_group(group)
#             json_objects[case_id].update(common_fields)
#             json_objects[case_id][table] = nested_objects

#     df = []
#     for case_id, patient_data in tqdm(json_objects.items()):
#         summary = generate_summary_from_json(patient_data)

#         if len(summary) > 0:
#             summary_chunks = get_chunk_text(summary)
#             chunk_embeddings = []
#             for chunk in summary_chunks:
#                 chunk_embedding = embedding_model.generate_embeddings([chunk])
#                 chunk_embeddings.append(chunk_embedding)
#             clinical_embedding = np.array(chunk_embeddings)
#         else:
#             clinical_embedding = None
#         patient_data["text"] = summary
#         patient_data["embedding_shape"] = clinical_embedding.shape

#         clinical_embedding = clinical_embedding.reshape(-1)
#         clinical_embedding = np.array(clinical_embedding, dtype=np.float32)
#         clinical_embedding = clinical_embedding.tobytes()
#         patient_data["embedding"] = clinical_embedding

#         # Create a new dictionary for DataFrame conversion, excluding lists
#         patient_data_for_df = {
#             key: value
#             for key, value in patient_data.items()
#             if not isinstance(value, list)
#         }
#         df.append(patient_data_for_df)

#     return df


def main():
    download()

    # query = "SELECT * FROM minds.clinical WHERE project_id like '%%TCGA%%'"
    # df = minds.query(query)
    # # get all unique project_ids and make a folder for each under /mnt/d/TCGA/
    # project_ids = df["project_id"].unique()
    # for project_id in tqdm(project_ids, desc="Processing", leave=False):
    #     project_data_folder = f"/mnt/d/TCGA-raw/{project_id}"
    #     manifest_file = os.path.join(project_data_folder, "manifest.json")

    # # --- CLINICAL DATA ---
    # df = CLINICAL(project_id)
    # clinical_df = pd.DataFrame(df)
    # os.makedirs(f"/mnt/d/TCGA-parquet/{project_id}", exist_ok=True)
    # clinical_df.to_parquet(
    #     f"/mnt/d/TCGA-parquet/{project_id}/Clinical Data.parquet", index=False
    # )

    # # --- PATHOLOGY REPORT ---
    # parquet_path = f"/mnt/d/TCGA-parquet/{project_id}/Pathology Report.parquet"
    # PATHOLOGY_REPORT(project_data_folder, manifest_file, parquet_path)

    # try:
    #     modality = "Clinical Data"
    #     parquet_path = f"/mnt/d/TCGA-parquet/{project_id}/{modality}.parquet"
    #     # convert the parquet to HF dataset
    #     dataset = datasets.load_dataset(
    #         "parquet",
    #         data_files=parquet_path,
    #         split="train",
    #     )
    #     dataset.save_to_disk(f"/mnt/d/TCGA/{project_id}/{modality}")
    #     # upload the dataset to the Hugging Face Hub
    #     api = HfApi()
    #     dataset_path = f"/mnt/d/TCGA/{project_id}/{modality}"
    #     api.upload_folder(
    #         repo_type="dataset",
    #         repo_id="Lab-Rasool/TCGA",
    #         folder_path=dataset_path,
    #         path_in_repo=f"{project_id}/{modality}",
    #     )
    # except Exception as e:
    #     print(f"Error: {e}")

    # CT()
    # uni_wsi()


if __name__ == "__main__":
    main()

    # # try to load the dataset from the Hugging Face Hub and print a sample
    # dataset = datasets.load_dataset(
    #     "Lab-Rasool/TCGA",
    # )
    # print(dataset[0])
