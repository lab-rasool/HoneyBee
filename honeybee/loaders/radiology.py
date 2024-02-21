import logging

import imageio.v3 as iio
import numpy as np
import onnxruntime as ort
from skimage.color import gray2rgb
from skimage.transform import resize

log_file = "dicom_conversion.log"
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


# def main():
#     manifest_df = pd.read_csv("/mnt/d/TCGA-LUAD/manifest.csv")
#     base_path = "/mnt/d/TCGA-LUAD/"  # Replace with your DICOM files path
#     rad_embeddings_model = RemedisEmbeddings(
#         "/mnt/d/Models/REMEDIS/onnx/cxr-50x1-remedis-s.onnx"
#     )
#     for index, row in manifest_df.iterrows():
#         series_path = os.path.join(
#             base_path, row["PatientID"], row["Modality"], row["SeriesInstanceUID"]
#         )
#         try:
#             vol = iio.imread(series_path, plugin="DICOM")
#             print(
#                 f"Processing series {row['SeriesInstanceUID']} with shape {vol.shape} {vol.ndim}"
#             )
#             embeddings = rad_embeddings_model.get_embeddings(vol.astype(np.float32))
#             print(
#                 f"Embeddings shape for series {row['SeriesInstanceUID']}: {embeddings.shape}"
#             )
#         except Exception as e:
#             logging.error(f"Error processing series {row['SeriesInstanceUID']}: {e}")
