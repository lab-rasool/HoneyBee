import gc
import json

import numpy as np
import pandas as pd
import torch

# Models to include:
# - TissueDetector (H&E, IHC, etc.)
# - REMEDIS
# - HuggingFace models like GatorTron & ClinicalT5
# - SeNMo
# - KimiaNet
# Loaders to include:
# - Slide (WSI: SVS, NIFTI, etc.)
# - Scan (CT, MRI, etc.)

from honeybee.loaders import Slide, Scan, PDFreport
from honeybee.models import REMEDIS, TissueDetector, HuggingFaceEmbedder


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


def WSI_example():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "Slide Image"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)
    paths = get_paths(df, DATA_DIR, MODALITY)
    print(f"Total files: {len(paths)}")

    # --- CONFIGURATION ---
    slide_image_path = np.random.choice(paths)
    tissue_detector_model_path = "/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt"
    embedding_model_path = "/mnt/d/Models/REMEDIS/onnx/path-50x1-remedis-s.onnx"

    # --- PROCESS THE SLIDE & GET PATCHES ---
    tissue_detector = TissueDetector(model_path=tissue_detector_model_path)
    slide = Slide(
        slide_image_path,
        tileSize=512,
        max_patches=500,
        visualize=False,
        tissue_detector=tissue_detector,
    )
    patches = slide.load_patches_concurrently(target_patch_size=224)

    # --- GENERATE EMBEDDINGS FOR THE PATCHES ---
    pred_onnx = REMEDIS.load_model_and_predict(embedding_model_path, patches)
    print(patches.shape, "->", pred_onnx.shape)

    # --- CLEANUP ---
    gc.collect()
    torch.cuda.empty_cache()


def RAD_example():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "CT"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)
    paths = get_paths(df, DATA_DIR, MODALITY)
    print(f"Total files: {len(paths)}")

    # --- CONFIGURATION ---
    ct_image_path = np.random.choice(paths)
    print(ct_image_path)
    embedding_model_path = "/mnt/d/Models/REMEDIS/onnx/cxr-50x1-remedis-s.onnx"

    # --- PROCESS THE CT IMAGE & GET PATCHES ---
    scanner = Scan(ct_image_path, modality="CT")
    patches = scanner.load_patches(target_patch_size=224)

    # --- GENERATE EMBEDDINGS FOR THE PATCHES ---
    pred_onnx = REMEDIS.load_model_and_predict(embedding_model_path, patches)
    print(patches.shape, "->", pred_onnx.shape)

    # --- CLEANUP ---
    gc.collect()
    torch.cuda.empty_cache()


def PATHOLOGY_REPORT_example():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    MODALITY = "Pathology Report"
    df = manifest_to_df(MANIFEST_PATH, MODALITY)
    file_paths = get_paths(df, DATA_DIR, MODALITY)
    print(f"Total files: {len(file_paths)}")

    # --- CONFIGURATION ---
    pdf_file_path = np.random.choice(file_paths)
    print(pdf_file_path)
    embedding_model = HuggingFaceEmbedder(model_name="UFNLP/gatortron-base")

    # --- PROCESS THE PDF REPORT & GET CHUNKS ---
    pdf_report = PDFreport(chunk_size=512, chunk_overlap=10)
    report_text = pdf_report.load(pdf_file_path)

    # --- GENERATE EMBEDDINGS FOR THE CHUNKS ---
    embeddings = embedding_model.generate_embeddings(report_text)
    print(len(report_text), "->", embeddings.shape)

    # --- CLEANUP ---
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # WSI_example()
    # RAD_example()
    PATHOLOGY_REPORT_example()
