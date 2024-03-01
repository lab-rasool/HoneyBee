import gc
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from albumentations import Compose, Resize
from cucim import CuImage
from matplotlib import pyplot as plt
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import models, transforms
from tqdm import tqdm

# def convert_models_to_onnx(source_dir, target_dir):
#     import subprocess
#     os.makedirs(target_dir, exist_ok=True)
#     for model_name in os.listdir(source_dir):
#         model_path = os.path.join(source_dir, model_name)
#         onnx_model_path = os.path.join(target_dir, f"{model_name}.onnx")
#         if os.path.exists(onnx_model_path):
#             print(f"Skipping {model_name} as it already exists.")
#             continue
#         if os.path.isdir(model_path):
#             subprocess.run(["python", "-m", "tf2onnx.convert",
#                             "--saved-model", model_path,
#                             "--output", onnx_model_path])
#             print(f"Converted {model_name} to ONNX format.")
# convert_models_to_onnx(source_dir="/mnt/d/Models/REMEDIS/Pretrained-Weights",
#                        target_dir="/mnt/d/Models/REMEDIS/onnx")


class TissueDetector:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self, model_path):
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device).eval()


class WholeSlideImageDataset(Dataset):
    def __init__(self, slideClass, transform=None):
        self.slideClass = slideClass
        self.transform = transform
        self.suitableTileAddresses = self.slideClass.suitableTileAddresses()

    def __len__(self):
        return len(self.suitableTileAddresses)

    def __getitem__(self, idx):
        tileAddress = self.suitableTileAddresses[idx]
        img = self.slideClass.getTile(tileAddress, writeToNumpy=True)[..., :3]
        img = self.transform(Image.fromarray(img).convert("RGB"))
        return {"image": img, "tileAddress": tileAddress}


class Slide:
    def __init__(
        self,
        slide_image_path,
        tileSize=512,
        tileOverlap=0,
        max_patches=500,
        visualize=False,
        tissue_detector=None,
    ):
        self.slide_image_path = slide_image_path
        self.slideFileName = Path(self.slide_image_path).stem
        self.tileSize = tileSize
        self.tileOverlap = round(tileOverlap * tileSize)
        self.tileDictionary = {}
        self.tissue_detector = tissue_detector
        if self.tissue_detector is None:
            raise ValueError("Model path is required for tissue detection.")

        self.img = CuImage(slide_image_path)
        resolutions = self.img.resolutions
        level_dimensions = resolutions["level_dimensions"]
        level_count = resolutions["level_count"]
        print(f"Resolutions: {resolutions}")

        selected_level = 0
        for level in range(level_count):
            width, height = level_dimensions[level]
            numTilesInX = width // tileSize
            numTilesInY = height // tileSize
            print(
                f"Level {level}: {numTilesInX}x{numTilesInY} ({numTilesInX*numTilesInY}) \t Resolution: {width}x{height}"
            )
            if numTilesInX * numTilesInY <= max_patches:
                selected_level = level
                break

        self.slide = self.img.read_region(location=[0, 0], level=selected_level)
        self.slide.height = int(self.slide.metadata["cucim"]["shape"][0])
        self.slide.width = int(self.slide.metadata["cucim"]["shape"][1])
        print(
            f"Selected level {selected_level} with dimensions: {self.slide.height}x{self.slide.width}"
        )

        self.numTilesInX = self.slide.width // (self.tileSize - self.tileOverlap)
        self.numTilesInY = self.slide.height // (self.tileSize - self.tileOverlap)
        self.tileDictionary = self._generate_tile_dictionary()

        self.detectTissue()
        if visualize:
            self.visualize()

    def _generate_tile_dictionary(self):
        tile_dict = {}
        for y in range(self.numTilesInY):
            for x in range(self.numTilesInX):
                tile_dict[(x, y)] = {
                    "x": x * (self.tileSize - self.tileOverlap),
                    "y": y * (self.tileSize - self.tileOverlap),
                    "width": self.tileSize,
                    "height": self.tileSize,
                }
        return tile_dict

    def suitableTileAddresses(self):
        suitableTileAddresses = []
        for tA in self.iterateTiles():
            suitableTileAddresses.append(tA)
        return suitableTileAddresses

    def getTile(self, tileAddress, writeToNumpy=False):
        if len(tileAddress) == 2 and isinstance(tileAddress, tuple):
            if (
                self.numTilesInX >= tileAddress[0]
                and self.numTilesInY >= tileAddress[1]
            ):
                tmpTile = self.slide.read_region(
                    (
                        self.tileDictionary[tileAddress]["x"],
                        self.tileDictionary[tileAddress]["y"],
                    ),
                    (
                        self.tileDictionary[tileAddress]["width"],
                        self.tileDictionary[tileAddress]["height"],
                    ),
                    0,
                )
                if writeToNumpy:
                    return np.asarray(tmpTile)
                else:
                    return tmpTile

    def iterateTiles(
        self, tileDictionary=False, includeImage=False, writeToNumpy=False
    ):
        tileDictionaryIterable = (
            self.tileDictionary if not tileDictionary else tileDictionary
        )
        for key, _ in tileDictionaryIterable.items():
            if includeImage:
                yield key, self.getTile(key, writeToNumpy=writeToNumpy)
            else:
                yield key

    def applyModel(self, batch_size, predictionKey="prediction", numWorkers=16):
        detector = TissueDetector(self.tissue_detector)
        device = detector.device
        model = detector.model
        data_transforms = detector.transforms
        pathSlideDataset = WholeSlideImageDataset(self, transform=data_transforms)
        pathSlideDataloader = torch.utils.data.DataLoader(
            pathSlideDataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=numWorkers,
        )

        for batch_index, inputs in enumerate(pathSlideDataloader):
            inputTile = inputs["image"].to(device)
            output = model(inputTile)
            batch_prediction = (
                torch.nn.functional.softmax(output, dim=1).cpu().data.numpy()
            )
            for index in range(len(inputTile)):
                tileAddress = (
                    inputs["tileAddress"][0][index].item(),
                    inputs["tileAddress"][1][index].item(),
                )
                # self.appendTag(tileAddress, predictionKey, batch_prediction[index, ...])
                self.tileDictionary[tileAddress][predictionKey] = batch_prediction[
                    index, ...
                ]

    def adoptKeyFromTileDictionary(self, upsampleFactor=1):
        for orphanTileAddress in self.iterateTiles():
            self.tileDictionary[orphanTileAddress].update(
                {
                    "x": self.tileDictionary[orphanTileAddress]["x"] * upsampleFactor,
                    "y": self.tileDictionary[orphanTileAddress]["y"] * upsampleFactor,
                    "width": self.tileDictionary[orphanTileAddress]["width"]
                    * upsampleFactor,
                    "height": self.tileDictionary[orphanTileAddress]["height"]
                    * upsampleFactor,
                }
            )

    def detectTissue(
        self,
        tissueDetectionUpsampleFactor=4,
        batchSize=20,
        numWorkers=1,
    ):
        self.applyModel(
            batch_size=batchSize,
            predictionKey="tissue_detector",
            numWorkers=numWorkers,
        )
        self.adoptKeyFromTileDictionary(upsampleFactor=tissueDetectionUpsampleFactor)

        self.predictionMap = np.zeros([self.numTilesInY, self.numTilesInX, 3])
        for address in self.iterateTiles():
            if "tissue_detector" in self.tileDictionary[address]:
                self.predictionMap[address[1], address[0], :] = self.tileDictionary[
                    address
                ]["tissue_detector"]

        predictionMap2 = np.zeros([self.numTilesInY, self.numTilesInX])
        predictionMap1res = resize(
            self.predictionMap, predictionMap2.shape, order=0, anti_aliasing=False
        )

        for address in self.iterateTiles():
            self.tileDictionary[address].update(
                {"artifactLevel": predictionMap1res[address[1], address[0]][0]}
            )
            self.tileDictionary[address].update(
                {"backgroundLevel": predictionMap1res[address[1], address[0]][1]}
            )
            self.tileDictionary[address].update(
                {"tissueLevel": predictionMap1res[address[1], address[0]][2]}
            )

    def visualize(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.slide)
        ax[0].set_title("original")
        ax[1].imshow(self.predictionMap)
        ax[1].set_title("deep tissue detection")
        plt.savefig(f"{self.slideFileName}.png", dpi=300)

    def get_tissue_coordinates(self, threshold=0.8):
        tissue_coordinates = []
        for address in self.iterateTiles():
            if self.tileDictionary[address]["tissueLevel"] > threshold:
                tissue_coordinates.append(
                    (
                        self.tileDictionary[address]["x"],
                        self.tileDictionary[address]["y"],
                    )
                )
        return tissue_coordinates

    def load_tile_thread(self, start_loc, patch_size, target_size):
        try:
            tile = np.asarray(
                self.img.read_region(start_loc, [patch_size, patch_size], 0)
            )
            if tile.ndim == 3 and tile.shape[2] == 3:  # Corrected condition
                transform = Compose([Resize(height=target_size, width=target_size)])
                tile = transform(image=tile)["image"]
                return tile
            else:
                return np.zeros((target_size, target_size, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error reading tile at {start_loc}: {e}")
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    def load_patches_concurrently(self, target_patch_size):
        tissue_coordinates = self.get_tissue_coordinates()
        num_patches = len(tissue_coordinates)
        patches = np.zeros(
            (num_patches, target_patch_size, target_patch_size, 3), dtype=np.uint8
        )

        def load_and_store_patch(index):
            start_loc = tissue_coordinates[index]
            patches[index] = self.load_tile_thread(
                start_loc, self.tileSize, target_patch_size
            )

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            executor.map(load_and_store_patch, range(num_patches))

        return patches.astype(np.float32)


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


def load_model_and_predict(model_path, patches):
    sess = ort.InferenceSession(
        model_path,
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    # "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 24 * 1024 * 1024 * 1024,  # 24GB
                },
            ),
            "CPUExecutionProvider",
        ],
    )
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onnx = sess.run([label_name], {input_name: patches})[0]
    return pred_onnx


def main():
    DATA_DIR = "/mnt/d/TCGA-LUAD"
    MANIFEST_PATH = "/mnt/d/TCGA-LUAD/manifest.json"
    slide_df = manifest_to_df(MANIFEST_PATH, "Slide Image")
    svs_paths = get_svs_paths(slide_df, DATA_DIR)
    # slide_image_path = np.random.choice(svs_paths)
    print(f"Total slides: {len(svs_paths)}")

    for slide_image_path in tqdm(svs_paths):
        slide = Slide(
            slide_image_path,
            tileSize=512,
            max_patches=500,
            visualize=False,
            tissue_detector="/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt",
        )
        patches = slide.load_patches_concurrently(target_patch_size=224)
        model_path = "/mnt/d/Models/REMEDIS/onnx/path-50x1-remedis-s.onnx"
        pred_onnx = load_model_and_predict(model_path, patches)
        print(patches.shape, "->", pred_onnx.shape)

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
