from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import models, transforms


class TissueDetector:
    """DenseNet121-based tissue detector for whole-slide images.

    Classifies image patches into three classes:
    ``[artifact, background, tissue]``.

    Parameters
    ----------
    model_path : str, optional
        Path to pre-trained weights.  Downloaded automatically if ``None``.
    device : str
        ``"cuda"`` or ``"cpu"``.
    patch_size : int
        Tile side length used when tiling a slide for inference.
    batch_size : int
        GPU batch size for ``predict_batch``.
    """

    CLASS_NAMES = ("artifact", "background", "tissue")

    def __init__(self, model_path=None, device="cuda", patch_size=224, batch_size=32):
        self.device = torch.device(device)
        self.patch_size = patch_size
        self.batch_size = batch_size

        # If no model path provided, download the weights automatically
        if model_path is None:
            model_path = self._download_weights()

        self.model = self._load_model(model_path)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _download_weights(self):
        """Download model weights from HuggingFace Hub if not available locally."""
        local_weights_dir = Path(__file__).parent / "weights"
        local_weights_path = local_weights_dir / "deep-tissue-detector_densenet_state-dict.pt"

        if local_weights_path.exists():
            print(f"Using existing local weights from {local_weights_path}")
            return str(local_weights_path)

        print("Downloading tissue detector weights from HuggingFace Hub...")
        try:
            try:
                model_path = hf_hub_download(
                    repo_id="Lab-Rasool/tissue-detector",
                    filename="model.safetensors",
                    cache_dir=local_weights_dir.parent / ".cache",
                )
                print("Downloaded SafeTensors weights from HuggingFace Hub")
                return model_path
            except Exception as e:
                print(f"SafeTensors not available, trying PyTorch format: {e}")

            model_path = hf_hub_download(
                repo_id="Lab-Rasool/tissue-detector",
                filename="deep-tissue-detector_densenet_state-dict.pt",
                cache_dir=local_weights_dir.parent / ".cache",
            )
            print("Downloaded PyTorch weights from HuggingFace Hub")
            return model_path

        except Exception as e:
            raise RuntimeError(
                f"Failed to download model weights from HuggingFace Hub: {e}\n"
                f"Please ensure you have internet connectivity and huggingface_hub is installed.\n"
                f"You can install it with: pip install huggingface_hub"
            )

    def _load_model(self, model_path):
        """Load DenseNet121 model with custom classifier."""
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 3)

        model_path_str = str(model_path)
        if model_path_str.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file

                state_dict = load_file(model_path)
                model.load_state_dict(state_dict)
                print("Loaded model from SafeTensors format")
            except ImportError:
                raise RuntimeError(
                    "SafeTensors format detected but safetensors package not installed.\n"
                    "Install it with: pip install safetensors"
                )
        else:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded model from PyTorch format")

        return model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Classify a batch of image tiles.

        Parameters
        ----------
        images : np.ndarray
            ``(N, H, W, 3)`` uint8 RGB array.

        Returns
        -------
        np.ndarray
            ``(N, 3)`` float32 softmax probabilities
            ``[artifact, background, tissue]``.
        """
        tensor_list = []
        for img in images:
            pil = Image.fromarray(img)
            tensor_list.append(self.transforms(pil))

        batch = torch.stack(tensor_list).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs.astype(np.float32)

    # ------------------------------------------------------------------
    # Slide-level detection
    # ------------------------------------------------------------------

    def detect(
        self,
        slide,
        level=None,
        magnification=None,
        thumbnail_size=(2048, 2048),
        threshold=0.5,
        num_workers=4,
    ):
        """Run tissue detection on a whole slide.

        Tiles the slide at a pyramid level, reads patches in parallel,
        runs batched GPU inference, and returns a binary mask plus the
        full 3-class prediction map.

        Parameters
        ----------
        slide : Slide
            An open :class:`~honeybee.loaders.Slide.slide.Slide`.
        level : int, optional
            Pyramid level to tile.  Ignored when *magnification* is set.
        magnification : float, optional
            Target magnification (e.g. ``10.0``).  Overrides *level*.
        thumbnail_size : tuple
            ``(width, height)`` of the thumbnail used for mask upscaling.
        threshold : float
            Tissue-class probability threshold for the binary mask.
        num_workers : int
            Thread-pool size for parallel tile reading.

        Returns
        -------
        binary_mask : np.ndarray
            Boolean mask at thumbnail resolution.
        pred_map : np.ndarray
            ``(rows, cols, 3)`` float32 class probabilities.
        """
        # Resolve target level
        if magnification is not None:
            target_level = slide.get_best_level_for_magnification(magnification)
        elif level is not None:
            target_level = level
        else:
            target_level = slide.level_count - 1  # lowest-res

        level_w, level_h = slide.level_dimensions[target_level]
        downsample = slide.level_downsamples[target_level]
        ps = self.patch_size

        # Build grid at target level
        n_cols = level_w // ps
        n_rows = level_h // ps
        tile_coords = []  # (row, col, x_l0, y_l0)
        for r in range(n_rows):
            for c in range(n_cols):
                x_l0 = int(c * ps * downsample)
                y_l0 = int(r * ps * downsample)
                tile_coords.append((r, c, x_l0, y_l0))

        if len(tile_coords) == 0:
            thumb = slide.get_thumbnail(thumbnail_size)
            th, tw = thumb.shape[:2]
            return np.zeros((th, tw), dtype=bool), np.zeros((0, 0, 3), dtype=np.float32)

        # Parallel read via CuCIM/OpenSlide backend
        def _read_tile(coord):
            _, _, x_l0, y_l0 = coord
            return slide.read_region(
                location=(x_l0, y_l0), size=(ps, ps), level=target_level
            )

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            tiles = list(pool.map(_read_tile, tile_coords))

        tiles_array = np.array(tiles, dtype=np.uint8)

        # Batched inference
        pred_map = np.zeros((n_rows, n_cols, 3), dtype=np.float32)
        for i in range(0, len(tiles_array), self.batch_size):
            batch = tiles_array[i : i + self.batch_size]
            probs = self.predict_batch(batch)
            for j, prob in enumerate(probs):
                r, c, _, _ = tile_coords[i + j]
                pred_map[r, c] = prob

        # Binary mask at thumbnail resolution
        tissue_prob = pred_map[:, :, 2]
        mask_small = (tissue_prob > threshold).astype(np.uint8)

        thumb = slide.get_thumbnail(thumbnail_size)
        th, tw = thumb.shape[:2]
        binary_mask = cv2.resize(
            mask_small, (tw, th), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        return binary_mask, pred_map

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def prediction_map_to_rgb(pred_map: np.ndarray) -> np.ndarray:
        """Convert a ``(rows, cols, 3)`` prediction map to an RGB image.

        Maps ``[artifact, background, tissue]`` to ``[red, green, blue]``.

        Parameters
        ----------
        pred_map : np.ndarray
            ``(rows, cols, 3)`` float32 probabilities.

        Returns
        -------
        np.ndarray
            ``(rows, cols, 3)`` uint8 RGB image.
        """
        rgb = np.zeros_like(pred_map)
        rgb[:, :, 0] = pred_map[:, :, 0]  # R = artifact
        rgb[:, :, 1] = pred_map[:, :, 1]  # G = background
        rgb[:, :, 2] = pred_map[:, :, 2]  # B = tissue
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
