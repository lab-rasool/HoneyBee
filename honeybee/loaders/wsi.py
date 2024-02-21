import concurrent.futures
import gc
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from cucim import CuImage
from matplotlib import pyplot as plt
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import models, transforms

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


def get_random_svs_file(
    manifest_path="/mnt/d/TCGA-LUAD/manifest.json", data_path="/mnt/d/TCGA-LUAD/raw"
):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    while True:
        case = random.choice(manifest)
        if "Slide Image" in case:
            slide_uuids = case.get("Slide Image", [])
            uuid = random.choice(slide_uuids)
            print(f"Selected slide: {uuid}")


def tissueDetector(
    modelStateDictPath="/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt",
):
    data_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 3)
    model.load_state_dict(
        torch.load(modelStateDictPath, map_location=torch.device("cuda"))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    return device, model.to(device).eval(), data_transforms


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
    ):
        self.slide_image_path = slide_image_path
        self.slideFileName = Path(self.slide_image_path).stem
        self.tileSize = tileSize
        self.tileOverlap = round(tileOverlap * tileSize)
        self.tileDictionary = {}

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

    def appendTag(self, tileAddress, key, val):
        self.tileDictionary[tileAddress][key] = val

    def applyModel(
        self, modelZip, batch_size, predictionKey="prediction", numWorkers=16
    ):
        device, model, data_transforms = modelZip
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
                self.appendTag(tileAddress, predictionKey, batch_prediction[index, ...])

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
        modelStateDictPath="/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt",
    ):
        modelZip = tissueDetector(modelStateDictPath=modelStateDictPath)
        self.applyModel(
            modelZip,
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
            if tile.ndim == 3 and tile.shape[2] == 3:
                return resize(tile, (target_size, target_size), anti_aliasing=True)
            else:
                return np.zeros((target_size, target_size, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Error reading tile at {start_loc}: {e}")
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    def load_patches(self, target_patch_size):
        tissue_coordinates = self.get_tissue_coordinates()
        patches = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [
                executor.submit(
                    self.load_tile_thread, loc, self.tileSize, target_patch_size
                )
                for loc in tissue_coordinates
            ]
            for future in concurrent.futures.as_completed(futures):
                patches.append(future.result())
        patches = np.array(patches, dtype=np.float32)
        return patches


class RemedisEmbeddings:
    def __init__(self, model_path, memory_limit=24):
        # check if GPU is available and set the providers accordingly
        if torch.cuda.is_available():
            total_memory = memory_limit * 1024 * 1024 * 1024  # 24GB
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": total_memory,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        self.model = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name

    def prepare_slide(self):
        pass

    def get_embeddings(self):
        pass


def main():
    slide_image_path = get_random_svs_file()

    slide = Slide(
        slide_image_path,
        tileSize=512,
        max_patches=500,
        visualize=False,
    )
    patches = slide.load_patches(target_patch_size=224)

    # Load the ONNX model
    model_path = "/mnt/d/Models/REMEDIS/onnx/path-50x1-remedis-s.onnx"
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 24 * 1024 * 1024 * 1024,  # 24GB
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onnx = sess.run([label_name], {input_name: patches})[0]
    print(patches.shape, "->", pred_onnx.shape)

    del sess
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
