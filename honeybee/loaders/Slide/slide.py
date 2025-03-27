import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import Compose, Resize
from cucim import CuImage
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset


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
        tile_size=512,
        tileOverlap=0,
        max_patches=500,
        visualize=False,
        tissue_detector=None,
        path_to_store_visualization="./visualizations",
        verbose=False,
    ):
        self.verbose = verbose
        self.slide_image_path = slide_image_path
        self.tileSize = tile_size
        self.tileOverlap = round(tileOverlap * tile_size)
        self.tileDictionary = {}
        self.tissue_detector = tissue_detector
        self.img = CuImage(slide_image_path)
        self.path_to_store_visualization = path_to_store_visualization

        self.slideFileName = Path(slide_image_path).stem
        self.slideFilePath = Path(slide_image_path)

        # Select the level with the most suitable number of patches
        self.selected_level = self._select_level(max_patches)

        # Read the slide at the selected level
        self.slide = self.img.read_region(location=[0, 0], level=self.selected_level)
        self.slide.height = int(self.slide.metadata["cucim"]["shape"][0])
        self.slide.width = int(self.slide.metadata["cucim"]["shape"][1])
        if self.verbose:
            print(
                f"Selected level {self.selected_level} with dimensions: {self.slide.height}x{self.slide.width}"
            )

        # Generate tile dictionary
        self.numTilesInX = self.slide.width // (self.tileSize - self.tileOverlap)
        self.numTilesInY = self.slide.height // (self.tileSize - self.tileOverlap)
        self.tileDictionary = self._generate_tile_dictionary()

        if self.tissue_detector is not None:
            self.detectTissue()

        # Visualize
        if visualize:
            self.visualize()

    def _select_level(self, max_patches):
        resolutions = self.img.resolutions
        level_dimensions = resolutions["level_dimensions"]
        level_count = resolutions["level_count"]
        if self.verbose:
            print(f"Resolutions: {resolutions}")

        selected_level = 0
        for level in range(level_count):
            width, height = level_dimensions[level]
            numTilesInX = width // self.tileSize
            numTilesInY = height // self.tileSize
            if self.verbose:
                print(
                    f"Level {level}: {numTilesInX}x{numTilesInY} ({numTilesInX * numTilesInY}) \t Resolution: {width}x{height}"
                )
            if numTilesInX * numTilesInY <= max_patches:
                selected_level = level
                break

        return selected_level

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

    def detectTissue(
        self,
        tissueDetectionUpsampleFactor=4,
        batchSize=20,
        numWorkers=1,
    ):
        detector = self.tissue_detector
        predictionKey = "tissue_detector"
        device = detector.device
        model = detector.model
        data_transforms = detector.transforms
        pathSlideDataset = WholeSlideImageDataset(self, transform=data_transforms)
        pathSlideDataloader = torch.utils.data.DataLoader(
            pathSlideDataset,
            batch_size=batchSize,
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
                self.tileDictionary[tileAddress][predictionKey] = batch_prediction[
                    index, ...
                ]

        upsampleFactor = tissueDetectionUpsampleFactor
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
        threshold = 0.8
        tissue_coordinates = []
        for address in self.iterateTiles():
            if self.tileDictionary[address]["tissueLevel"] > threshold:
                tissue_coordinates.append(
                    (
                        self.tileDictionary[address]["x"],
                        self.tileDictionary[address]["y"],
                    )
                )
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

    def get_patch_coords(self):
        # Extract coordinates of all patches
        coords = []
        for address in self.suitableTileAddresses():
            coords.append(
                (self.tileDictionary[address]["x"], self.tileDictionary[address]["y"])
            )
        return np.array(coords)

    def visualize(self):
        os.makedirs(self.path_to_store_visualization, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.slide)
        ax[0].set_title("original")
        ax[1].imshow(self.predictionMap)
        ax[1].set_title("deep tissue detection")
        plt.savefig(
            f"{self.path_to_store_visualization}/{Path(self.slide_image_path).stem}.png",
            dpi=300,
        )
