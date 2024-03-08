import json
import logging
import os

import imageio.v3 as iio
import numpy as np
import onnxruntime as ort
import pandas as pd
from skimage.color import gray2rgb
from skimage.transform import resize

log_file = "./logs/CT_embeddings.log"
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s:%(message)s",
)


class RemedisEmbeddings:
    def __init__(self, module_path):
        self.model = ort.InferenceSession(module_path)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def prepare_volume_or_image(self, vol, target_size=(224, 224)):
        # Normalize the volume or image first
        vol = vol / np.max(vol)

        if vol.ndim == 3:  # If the input is a volume
            # Resize each slice and convert to RGB
            resized_and_colored = np.array(
                [
                    gray2rgb(resize(slice, target_size, anti_aliasing=True))
                    for slice in vol
                ]
            )
        elif vol.ndim == 2:  # If the input is a single image
            # Resize the image, convert to RGB, and add a new axis to simulate a batch (or volume) dimension
            resized = resize(vol, target_size, anti_aliasing=True)
            resized_and_colored = gray2rgb(resized)
            resized_and_colored = np.expand_dims(
                resized_and_colored, axis=0
            )  # Add the batch dimension
        else:
            raise ValueError("Unsupported volume/image shape")

        return resized_and_colored

    def get_embeddings(self, image):
        image = self.prepare_volume_or_image(image)
        embedding_of_image = self.model.run(
            [self.output_name], {self.input_name: image}
        )[0]
        return embedding_of_image


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


def get_modality_paths(df, modality, DATA_DIR):
    paths = []
    for index, row in df.iterrows():
        path = f"{DATA_DIR}/raw/{row['PatientID']}/{modality}/{row['id']}/{row['file_name']}"
        paths.append(path)
    return paths


def main():
    manifest_df = pd.read_csv("/mnt/d/TCGA-LUAD/manifest.csv")
    base_path = "/mnt/d/TCGA-LUAD/"  # Replace with your DICOM files path
    rad_embeddings_model = RemedisEmbeddings(
        "/mnt/d/Models/REMEDIS/onnx/cxr-50x1-remedis-s.onnx"
    )
    for index, row in manifest_df.iterrows():
        series_path = os.path.join(
            base_path, row["PatientID"], row["Modality"], row["SeriesInstanceUID"]
        )
        try:
            vol = iio.imread(series_path, plugin="DICOM")
            print(
                f"Processing series {row['SeriesInstanceUID']} with shape {vol.shape} {vol.ndim}"
            )
            embeddings = rad_embeddings_model.get_embeddings(vol.astype(np.float32))
            print(
                f"Embeddings shape for series {row['SeriesInstanceUID']}: {embeddings.shape}"
            )
        except Exception as e:
            logging.error(f"Error processing series {row['SeriesInstanceUID']}: {e}")


if __name__ == "__main__":
    main()
