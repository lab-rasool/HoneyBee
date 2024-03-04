import gc
import json

import numpy as np
import pandas as pd
import torch

from honeybee.loaders.Slide import Slide
from honeybee.models.REMEDIS import REMEDIS
from honeybee.models.TissueDetector import TissueDetector


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


def get_svs_paths(slide_df, DATA_DIR):
    svs_paths = []
    for index, row in slide_df.iterrows():
        svs_path = f"{DATA_DIR}/raw/{row['PatientID']}/Slide Image/{row['id']}/{row['file_name']}"
        svs_paths.append(svs_path)
    return svs_paths


def main():
    # --- THIS CAN BE IGNORED ---
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    slide_df = manifest_to_df(MANIFEST_PATH, "Slide Image")
    svs_paths = get_svs_paths(slide_df, DATA_DIR)
    print(f"Total slides: {len(svs_paths)}")

    # --- CONFIGURATION ---
    slide_image_path = np.random.choice(svs_paths)
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


if __name__ == "__main__":
    main()
