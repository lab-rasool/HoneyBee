import gc
import json
import os
import warnings

import datasets
import minds
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
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
)
from honeybee.models import REMEDIS, UNI, HuggingFaceEmbedder, TissueDetector

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
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


def process_group(group):
    common_fields = {}
    nested_objects = []
    for col in group.columns:
        unique_values = group[col].dropna().unique()
        if len(unique_values) == 1:
            # If only one unique value exists, it's a common field
            common_fields[col] = unique_values[0]

    # Create nested objects for fields that are not common
    for idx, row in group.iterrows():
        nested_object = {
            col: row[col]
            for col in group.columns
            if col not in common_fields and pd.notna(row[col])
        }
        if nested_object:  # Only add if the nested object is not empty
            nested_objects.append(nested_object)

    return common_fields, nested_objects


PROJECTS = [
    "TCGA-ACC",
    "TCGA-COAD",
    "TCGA-KICH",
    "TCGA-LIHC",
    "TCGA-PAAD",
    "TCGA-SKCM",
    "TCGA-UCEC",
    "TCGA-BLCA",
    "TCGA-DLBC",
    "TCGA-KIRC",
    "TCGA-LUAD",
    "TCGA-PCPG",
    "TCGA-STAD",
    "TCGA-UCS",
    "TCGA-BRCA",
    "TCGA-ESCA",
    "TCGA-KIRP",
    "TCGA-LUSC",
    "TCGA-PRAD",
    "TCGA-TGCT",
    "TCGA-UVM",
    "TCGA-CESC",
    "TCGA-GBM",
    "TCGA-LAML",
    "TCGA-MESO",
    "TCGA-READ",
    "TCGA-THCA",
    "TCGA-CHOL",
    "TCGA-HNSC",
    "TCGA-LGG",
    "TCGA-OV",
    "TCGA-SARC",
    "TCGA-THYM",
]


def generate_pathology_report_embeddings():
    for PROJECT in tqdm(PROJECTS, desc="Projects", total=len(PROJECTS), leave=False):
        DATA_DIR = f"/mnt/d/TCGA/raw/{PROJECT}"
        MANIFEST_PATH = DATA_DIR + "/manifest.json"
        MODALITY = "Pathology Report"
        PARQUET = f"/mnt/d/TCGA/parquet/{MODALITY} (gatortron-base).parquet"

        df = manifest_to_df(MANIFEST_PATH, MODALITY)
        if df is None or df.empty:
            continue

        embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-base")
        pdf_report = PDFreport(chunk_size=512, chunk_overlap=10)

        df["report_text"] = None
        df["embedding"] = None
        df["embedding_shape"] = None

        new_rows = []
        for index, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing {PROJECT}", leave=False
        ):
            try:
                file_path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['id']}/{row['file_name']}"
                report_text = pdf_report.load(file_path)

                if len(report_text) > 0:
                    embeddings = embedding_model.generate_embeddings(report_text)
                    embeddings = embeddings.reshape(-1)
                    row["embedding"] = embeddings.tobytes()
                    row["report_text"] = report_text
                    row["embedding_shape"] = embeddings.shape
                else:
                    row["report_text"] = None
                    row["embedding"] = None
                    row["embedding_shape"] = None

                new_rows.append(row)

            except Exception as e:
                print(f"Error processing {row['PatientID']}: {e}")

        new_df = pd.DataFrame(new_rows)

        if os.path.exists(PARQUET):
            existing_df = pq.read_table(PARQUET).to_pandas()
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=["PatientID", "id", "file_name"]
            )
        else:
            combined_df = new_df

        table = pa.Table.from_pandas(combined_df)
        pq.write_table(table, PARQUET)

        gc.collect()
        torch.cuda.empty_cache()


def generate_clinical_embeddings():
    embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-base")

    for PROJECT in tqdm(PROJECTS, desc="Projects", total=len(PROJECTS), leave=False):
        MODALITY = "Clinical Data"
        PARQUET = f"/mnt/d/TCGA/parquet/{MODALITY} (gatortron-base).parquet"

        tables = minds.get_tables()
        json_objects = {}
        for table in tqdm(tables, desc="Getting data from tables", leave=False):
            query = f"SELECT * FROM minds.{table} WHERE project_id='{PROJECT}'"
            df = minds.query(query)
            for case_id, group in tqdm(df.groupby("case_submitter_id"), leave=False):
                if case_id not in json_objects:
                    json_objects[case_id] = {}
                common_fields, nested_objects = process_group(group)
                json_objects[case_id].update(common_fields)
                json_objects[case_id][table] = nested_objects

        df = []
        for case_id, patient_data in tqdm(json_objects.items(), leave=False):
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

        # Append to existing Parquet file if it exists
        if os.path.exists(PARQUET):
            existing_df = pq.read_table(PARQUET).to_pandas()
            combined_df = pd.concat([existing_df, clinical_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=["case_submitter_id", "project_id"]
            )
        else:
            combined_df = clinical_df

        table = pa.Table.from_pandas(combined_df)
        pq.write_table(table, PARQUET)


def generate_slide_embeddings():
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "Slide Image"
    PARQUET = f"/mnt/d/TCGA/parquet/{MODALITY} (UNI).parquet"
    HE_DETECTOR_PATH = "/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt"
    EMBEDDING_MODEL_PATH = "/mnt/d/Models/pytorch_model.bin"

    df = manifest_to_df(MANIFEST_PATH, MODALITY)
    tissue_detector = TissueDetector(model_path=HE_DETECTOR_PATH)
    embedding_model_path = EMBEDDING_MODEL_PATH
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
                slide = Slide(
                    slide_image_path,
                    tileSize=512,
                    max_patches=1000,
                    visualize=True,
                    tissue_detector=tissue_detector,
                )
                patches = slide.load_patches_concurrently(target_patch_size=224)

                if patches.shape[0] == 0:
                    with open("errors.txt", "a") as f:
                        f.write(f"{slide_image_path} | No patches extracted.\n")
                    raise ValueError("No patches extracted.")

            embedding = uni.load_model_and_predict(embedding_model_path, patches)
            df.at[index, "embedding_shape"] = embedding.shape
            embedding = embedding.reshape(-1)
            embedding = np.array(embedding, dtype=np.float32)
            embedding = embedding.tobytes()
            df.at[index, "embedding"] = embedding
        except Exception as e:
            with open("errors.txt", "a") as f:
                f.write(f"{slide_image_path} | {e}\n")
            df.at[index, "embedding"] = None
            continue

        if writer is None:
            table = pa.Table.from_pandas(df.iloc[[index]])
            writer = pq.ParquetWriter(PARQUET, table.schema)
        else:
            table = pa.Table.from_pandas(df.iloc[[index]])
            writer.write_table(table)

        del slide, patches, embedding, table
        gc.collect()
        torch.cuda.empty_cache()

    if writer is not None:
        writer.close()


def convert_parquet_to_dataset():
    modalities = [
        "Pathology Report (gatortron-base)",
        "Clinical Data (gatortron-base)",
        "Slide Image (UNI)",
    ]
    for MODALITY in modalities:
        PARQUET = f"/mnt/d/TCGA/parquet/{MODALITY}.parquet"
        HF_DATASET = f"/mnt/d/TCGA/huggingface/{MODALITY}"
        try:
            dataset = datasets.load_dataset(
                "parquet",
                data_files=PARQUET,
                split="train",
            )
            dataset.save_to_disk(HF_DATASET)
        except Exception as e:
            print(f"Error: {e}")
            continue


def process_all_slide_images():
    HE_DETECTOR_PATH = "/mnt/d/Models/TissueDetector/HnE.pt"
    EMBEDDING_MODEL_PATH = "/mnt/d/Models/UNI/pytorch_model.bin"
    MODALITY = "Slide Image"

    for PROJECT in tqdm(PROJECTS, desc="Projects", total=len(PROJECTS), leave=False):
        DATA_DIR = f"/mnt/d/TCGA-slides/raw/{PROJECT}"
        os.makedirs(DATA_DIR, exist_ok=True)
        MANIFEST_PATH = DATA_DIR + "/manifest.json"
        PARQUET = f"/mnt/d/TCGA/parquet/{PROJECT} {MODALITY} (UNI).parquet"

        # ---------------------------------------------------------------------
        # DOWNLOAD THE SLIDE IMAGES
        # ---------------------------------------------------------------------
        query = f"SELECT * FROM minds.clinical WHERE project_id = '{PROJECT}'"
        query_cohort = minds.build_cohort(
            query=query,
            output_dir=DATA_DIR,
            manifest=MANIFEST_PATH if os.path.exists(MANIFEST_PATH) else None,
        )
        query_cohort.stats()
        query_cohort.download(threads=4, include=["Slide Image"])

        # ---------------------------------------------------------------------
        # GENERATE EMBEDDINGS FOR SLIDE IMAGES USING UNI
        # ---------------------------------------------------------------------
        df = manifest_to_df(MANIFEST_PATH, MODALITY)
        tissue_detector = TissueDetector(model_path=HE_DETECTOR_PATH)
        embedding_model_path = EMBEDDING_MODEL_PATH
        uni = UNI()

        df["embedding"] = None
        df["embedding_shape"] = None
        writer = None
        for index, row in tqdm(
            df.iterrows(), total=len(df), desc="Processing", leave=False
        ):
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
                    slide = Slide(
                        slide_image_path,
                        tileSize=512,
                        max_patches=1000,
                        visualize=True,
                        tissue_detector=tissue_detector,
                    )
                    patches = slide.load_patches_concurrently(target_patch_size=224)

                    if patches.shape[0] == 0:
                        with open("errors.txt", "a") as f:
                            f.write(f"{slide_image_path} | No patches extracted.\n")
                        raise ValueError("No patches extracted.")

                embedding = uni.load_model_and_predict(embedding_model_path, patches)
                df.at[index, "embedding_shape"] = embedding.shape
                embedding = embedding.reshape(-1)
                embedding = np.array(embedding, dtype=np.float32)
                embedding = embedding.tobytes()
                df.at[index, "embedding"] = embedding
            except Exception as e:
                with open("errors.txt", "a") as f:
                    f.write(f"{slide_image_path} | {e}\n")
                df.at[index, "embedding"] = None
                continue

            if writer is None:
                table = pa.Table.from_pandas(df.iloc[[index]])
                writer = pq.ParquetWriter(PARQUET, table.schema)
            else:
                table = pa.Table.from_pandas(df.iloc[[index]])
                writer.write_table(table)

            del slide, patches, embedding, table
            gc.collect()
            torch.cuda.empty_cache()

        if writer is not None:
            writer.close()

        # ---------------------------------------------------------------------
        # DELETE THE RAW DATA
        # ---------------------------------------------------------------------
        os.system(f"rm -rf {DATA_DIR}/raw/")
        gc.collect()


if __name__ == "__main__":
    # generate_pathology_report_embeddings()
    # generate_clinical_embeddings()
    # generate_slide_embeddings()
    # convert_parquet_to_dataset()
    # process_all_slide_images()

    # tcga_query = """SELECT * FROM minds.clinical WHERE project_id LIKE '%%TCGA-%%'"""
    # print(tcga_query)
    # tcga_cohort = minds.build_cohort(
    #     query=tcga_query,
    #     output_dir="/mnt/d/allTCGA",
    #     manifest="/mnt/d/allTCGA/manifest.json"
    #     if os.path.exists("/mnt/d/allTCGA/manifest.json")
    #     else None,
    # )
    # tcga_cohort.stats()

    api = HfApi(token=os.getenv("HF_API_KEY"))
    api.upload_folder(
        folder_path="/mnt/d/TCGA/parquet/Radiology (REMEDIS)",
        path_in_repo="Radiology (REMEDIS)",
        repo_id="Lab-Rasool/TCGA",
        repo_type="dataset",
        multi_commits=True,
        multi_commits_verbose=True,
    )
    # dataset = load_dataset("Lab-Rasool/TCGA", "slide_image", split="train")
    # print(dataset)
