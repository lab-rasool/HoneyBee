import gc
import json
import os

import datasets
import numpy as np
import ollama
import pandas as pd
import torch
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


def get_paths(slide_df, DATA_DIR, MODALITY):
    paths = []
    for index, row in slide_df.iterrows():
        if MODALITY == "CT":
            path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['SeriesInstanceUID']}/{row['SeriesInstanceUID']}/"
        else:
            path = f"{DATA_DIR}/raw/{row['PatientID']}/{MODALITY}/{row['id']}/{row['file_name']}"
        paths.append(path)
    return paths


def upload_dataset_to_hf(dataset_path, repo_id, organization=None):
    """
    Upload a dataset to Hugging Face using HTTP-based methods.

    Parameters:
    - dataset_path: Path to the dataset directory.
    - repo_id: ID of the repository (name of your dataset).
    - organization: (Optional) The organization in which to create the dataset. Use None for your user.
    """
    # Initialize HfApi and get token
    api = HfApi()
    token = HfFolder.get_token()
    if token is None:
        raise ValueError(
            "Hugging Face token not found. Make sure you're logged in with `huggingface-cli login`."
        )

    # Define repository name
    namespace = organization if organization else api.whoami(token)["name"]
    full_repo_name = f"{namespace}/{repo_id}"

    # Create or get repository
    if not api.repo_exists(repo_id=full_repo_name):
        api.create_repo(
            token=token,
            repo_id=repo_id,
            private=True,  # Change to True if you want a private repository
            exist_ok=True,
        )

    # Upload files to the repository
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            file_path = os.path.join(root, name)
            # relative_path = os.path.relpath(file_path, dataset_path)
            # path in repo should be the same as the file path in the dataset
            # for example, if the dataset has a file at "./data/embeddings/embedding_0.npz"
            # the path in the repo should also be "./embeddings/embedding_0.npz"
            relative_path = file_path.replace(dataset_path, "")
            api.upload_file(
                token=token,
                path_or_fileobj=file_path,
                path_in_repo=relative_path,
                repo_id=full_repo_name,
                repo_type="dataset",
            )

    print(f"Dataset uploaded to: https://huggingface.co/{full_repo_name}")


def WSI_example():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "Slide Image"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)
    df = df.head(5)
    paths = get_paths(df, DATA_DIR, MODALITY)
    print(f"Total files: {len(paths)}")

    # --- CONFIGURATION ---
    tissue_detector_model_path = "/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt"
    embedding_model_path = "/mnt/d/Models/REMEDIS/onnx/path-50x1-remedis-s.onnx"

    embeddings = []
    for idx, slide_image_path in enumerate(tqdm(paths)):
        try:
            tissue_detector = TissueDetector(model_path=tissue_detector_model_path)
            slide = Slide(
                slide_image_path,
                tileSize=512,
                max_patches=500,
                visualize=False,
                tissue_detector=tissue_detector,
            )
            patches = slide.load_patches_concurrently(target_patch_size=224)
            embedding = REMEDIS.load_model_and_predict(embedding_model_path, patches)
            print(f"Embedding shape: {embedding.shape}")
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error: {e}")
            embeddings.append(None)

    # --- CLEANUP ---
    gc.collect()
    torch.cuda.empty_cache()


def RAD_example():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "CT"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)
    df = df.head(5)
    paths = get_paths(df, DATA_DIR, MODALITY)
    print(f"Total files: {len(paths)}")

    # --- CONFIGURATION ---
    embedding_model_path = "/mnt/d/Models/REMEDIS/onnx/cxr-50x1-remedis-s.onnx"

    embeddings = []
    for ct_image_path in tqdm(paths):
        try:
            scanner = Scan(ct_image_path, modality="CT")
            patches = scanner.load_patches(target_patch_size=224)
            embedding = REMEDIS.load_model_and_predict(embedding_model_path, patches)
            print(f"Embedding shape: {embedding.shape}")
            embeddings.append(pd.Series(list(embedding), dtype=object))
        except Exception as e:
            print(f"Error: {e}")
            embeddings.append(None)

    # --- CLEANUP ---
    gc.collect()
    torch.cuda.empty_cache()


def PATHOLOGY_REPORT_example():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "Pathology Report"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)
    df = df.head(5)
    file_paths = get_paths(df, DATA_DIR, MODALITY)
    print(f"Total files: {len(file_paths)}")

    # --- CONFIGURATION ---
    embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-base")
    pdf_report = PDFreport(chunk_size=512, chunk_overlap=10)

    report_texts = []
    report_embeddings = []
    for file_path in tqdm(file_paths):
        try:
            # --- PROCESS THE PDF REPORT ---
            report_text = pdf_report.load(file_path)
            report_texts.append(report_text)
            # --- GENERATE EMBEDDINGS ---
            if len(report_text) > 0:
                embeddings = embedding_model.generate_embeddings(report_text)
                report_embeddings.append(
                    np.array(pd.Series(list(embeddings), dtype=object))
                )
            else:
                report_embeddings.append(None)
            print(f"Embedding shape: {embeddings.shape}")
        except Exception as e:
            print(f"Error: {e}")
            report_texts.append(None)
            report_embeddings.append(None)

    # --- CLEANUP ---
    gc.collect()
    torch.cuda.empty_cache()


def CLINICAL_example():
    embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-base")
    json_objects = get_clinical_json_from_minds()
    clinical_df = []
    for case_id, patient_data in tqdm(json_objects.items()):
        summary = generate_summary_from_json(patient_data)

        if len(summary) > 0:
            summary_chunks = get_chunk_text(summary)
            chunk_embeddings = []
            for chunk in summary_chunks:
                chunk_embedding = embedding_model.generate_embeddings([chunk])
                chunk_embeddings.append(chunk_embedding)
            clinical_embedding = np.vstack(chunk_embeddings)
            clinical_embedding = np.array(
                pd.Series(list(clinical_embedding), dtype=object)
            )
        else:
            clinical_embedding = None
        patient_data["summary"] = summary
        patient_data["clinical_embedding"] = clinical_embedding

        # Create a new dictionary for DataFrame conversion, excluding lists
        patient_data_for_df = {
            key: value
            for key, value in patient_data.items()
            if not isinstance(value, list)
        }
        clinical_df.append(patient_data_for_df)

    clinical_df = pd.DataFrame(clinical_df)
    clinical_df.to_parquet("./data/parquet/clinical.parquet", index=False)


if __name__ == "__main__":
    print(f"\033c\n{'='*40}\n WSI Example \n{'='*40}")
    WSI_example()

    print(f"\n{'='*40}\n RAD Example \n{'='*40}")
    RAD_example()

    print(f"\n{'='*40}\n Pathology Report Example \n{'='*40}")
    PATHOLOGY_REPORT_example()

    print(f"\n{'='*40}\n Clinical Example \n{'='*40}")
    CLINICAL_example()

    print(f"\n{'='*40}\n All Done! \n{'='*40}")
