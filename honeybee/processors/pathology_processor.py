"""
PathologyProcessor - Unified interface for whole slide image processing

This module provides a high-level API for pathology image analysis, combining
tissue detection, stain normalization, patch extraction, and embedding generation.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

# Lazy imports - loaded when needed to avoid requiring all dependencies at import time
# These will be imported when the corresponding methods are called


class _LegacyEmbedder:
    """Adapter wrapping legacy model objects as :class:`EmbeddingModel`.

    Handles objects that expose ``generate_embeddings`` or
    ``load_model_and_predict`` but do not conform to the new protocol.
    """

    def __init__(self, wrapped, model_name: str = ""):
        self._wrapped = wrapped
        self._model_name = model_name

    @property
    def embedding_dim(self) -> int:
        return getattr(self._wrapped, "embed_dim", 0)

    @property
    def device(self) -> str:
        d = getattr(self._wrapped, "device", "unknown")
        return str(d)

    def generate_embeddings(self, inputs, batch_size: int = 32) -> np.ndarray:
        if hasattr(self._wrapped, "generate_embeddings"):
            result = self._wrapped.generate_embeddings(inputs, batch_size=batch_size)
            if isinstance(result, np.ndarray):
                return result
            return result.cpu().numpy()

        # Fallback to load_model_and_predict
        if isinstance(inputs, np.ndarray) and inputs.ndim == 4:
            patches_list = inputs
        else:
            patches_list = inputs

        all_embeddings = []
        for i in range(0, len(patches_list), batch_size):
            batch = patches_list[i : i + batch_size]
            embeddings = self._wrapped.load_model_and_predict(batch)
            if isinstance(embeddings, np.ndarray):
                all_embeddings.append(embeddings)
            else:
                all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)


class PathologyProcessor:
    """
    Unified processor for whole slide image (WSI) analysis.

    This class provides a complete pipeline for pathology image processing:
    - WSI loading and visualization
    - Tissue detection (classical and deep learning)
    - Stain normalization (Reinhard, Macenko, Vahadane)
    - Stain separation (H&E deconvolution)
    - Patch extraction from tissue regions
    - Embedding generation with foundation models
    - Embedding aggregation for slide-level representations

    Example:
        >>> processor = PathologyProcessor(model="uni")
        >>> wsi = processor.load_wsi("path/to/slide.svs")
        >>> tissue_mask = processor.detect_tissue(wsi, method="otsu")
        >>> normalized = processor.normalize_stain(wsi, method="macenko")
        >>> patches = processor.extract_patches(wsi, tissue_mask)
        >>> embeddings = processor.generate_embeddings(patches)
        >>> slide_embedding = processor.aggregate_embeddings(embeddings)
    """

    def __init__(
        self,
        model: str = "uni",
        model_path: Optional[str] = None,
        provider: Optional[str] = None,
        **model_kwargs,
    ):
        """
        Initialize PathologyProcessor with specified embedding model.

        Args:
            model: Embedding model alias (e.g. ``"uni"``, ``"h-optimus"``) or a
                provider-specific identifier (e.g. ``"bioptimus/H-optimus-0"``).
                See :func:`honeybee.models.registry.list_models` for presets.
            model_path: Path to model weights (required for ONNX models).
            provider: Override the model provider (``"timm"``, ``"huggingface"``,
                ``"onnx"``, ``"torch"``). Required when *model* is not a preset alias.
            **model_kwargs: Extra keyword arguments forwarded to the provider.
        """
        # Preserve case for HF repo IDs (contain "/"), lowercase for aliases
        self.model_name = model if "/" in model else model.lower()
        self.model_path = model_path
        self._provider = provider
        self._model_kwargs = model_kwargs
        self.embedding_model = None
        self.tissue_detector_dl = None

    def load_wsi(
        self,
        path: Union[str, Path],
        tile_size: int = 512,
        max_patches: int = 500,
        visualize: bool = False,
        verbose: bool = False,
    ):
        """
        Load whole slide image.

        .. deprecated::
            Use ``Slide(path)`` directly from ``honeybee.loaders.Slide``.

        Args:
            path: Path to WSI file (.svs, .tiff, etc.)
            tile_size: Size of tiles for patch extraction
            max_patches: Maximum number of patches to extract
            visualize: Whether to create visualizations
            verbose: Whether to print loading information

        Returns:
            Slide object for WSI manipulation

        Example:
            >>> processor = PathologyProcessor()
            >>> wsi = processor.load_wsi("slide.svs", tile_size=256)
        """
        warnings.warn(
            "PathologyProcessor.load_wsi() is deprecated. "
            "Use Slide(path) from honeybee.loaders.Slide instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..loaders.Slide.slide import Slide

        slide = Slide(
            slide_image_path=str(path),
            tile_size=tile_size,
            max_patches=max_patches,
            visualize=visualize,
            verbose=verbose,
        )
        return slide

    def detect_tissue(
        self,
        wsi,
        method: str = "otsu",
        tissue_detector_path: Optional[str] = None,
        min_tissue_size: int = 1000,
        **kwargs,
    ) -> np.ndarray:
        """
        Detect tissue regions in WSI.

        .. deprecated::
            Use ``slide.detect_tissue(method)`` directly on a Slide object.

        Args:
            wsi: Slide object or image array
            method: Detection method ("otsu", "hsv", "gradient", "otsu_hsv", "deeplearning")
            tissue_detector_path: Path to deep learning model (for "deeplearning" method)
            min_tissue_size: Minimum tissue region size in pixels
            **kwargs: Additional arguments for tissue detector

        Returns:
            Binary tissue mask

        Example:
            >>> mask = processor.detect_tissue(wsi, method="otsu")
            >>> # Deep learning detection
            >>> mask = processor.detect_tissue(wsi, method="deeplearning",
            ...                                tissue_detector_path="model.pt")
        """
        warnings.warn(
            "PathologyProcessor.detect_tissue() is deprecated. "
            "Use slide.detect_tissue(method) directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..loaders.Slide.slide import Slide
        from .wsi import ClassicalTissueDetector

        # Extract image array if Slide object provided
        if isinstance(wsi, Slide):
            image = self._get_overview_image(wsi, min_dim=2048)
        else:
            image = wsi

        if method == "deeplearning":
            from ..models.TissueDetector.tissue_detector import TissueDetector

            # Use deep learning tissue detector (auto-downloads weights if path is None)
            if self.tissue_detector_dl is None:
                self.tissue_detector_dl = TissueDetector(model_path=tissue_detector_path)

            # If we have a Slide object, use its detection method
            if isinstance(wsi, Slide):
                # Reload slide with tissue detector
                wsi_with_detection = Slide(
                    slide_image_path=str(wsi.slide_image_path),
                    tile_size=wsi.tileSize,
                    tissue_detector=self.tissue_detector_dl,
                    visualize=False,
                )

                # Extract tissue mask from predictions
                mask = np.zeros((wsi_with_detection.numTilesInY, wsi_with_detection.numTilesInX))
                for address in wsi_with_detection.iterateTiles():
                    tile_info = wsi_with_detection.tileDictionary[address]
                    if "tissueLevel" in tile_info:
                        mask[address[1], address[0]] = tile_info["tissueLevel"]

                return mask > 0.5  # Binary mask with threshold
            else:
                # For raw images, tile the image and run DL detection on each tile
                import torch
                from PIL import Image

                tile_size = kwargs.get("tile_size", 224)
                h, w = image.shape[:2]
                tiles_y = max(1, h // tile_size)
                tiles_x = max(1, w // tile_size)
                mask = np.zeros((tiles_y, tiles_x), dtype=np.float32)

                for ty in range(tiles_y):
                    for tx in range(tiles_x):
                        y_start = ty * tile_size
                        x_start = tx * tile_size
                        tile = image[y_start : y_start + tile_size, x_start : x_start + tile_size]
                        if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                            continue

                        # Run DL prediction on tile
                        pil_tile = Image.fromarray(tile)
                        input_tensor = self.tissue_detector_dl.transforms(pil_tile).unsqueeze(0)
                        input_tensor = input_tensor.to(self.tissue_detector_dl.device)

                        with torch.no_grad():
                            output = self.tissue_detector_dl.model(input_tensor)
                            probs = torch.softmax(output, dim=1)
                            # Class 2 is tissue in the 3-class model
                            tissue_prob = probs[0, 2].item()
                            mask[ty, tx] = tissue_prob

                return mask > 0.5

        # Use classical detection methods
        detector = ClassicalTissueDetector(method=method, min_tissue_size=min_tissue_size, **kwargs)
        return detector.detect(image)

    def normalize_stain(
        self,
        wsi,
        method: str = "reinhard",
        target: Optional[np.ndarray] = None,
        use_target_params: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize stain appearance.

        Args:
            wsi: Slide object or image array
            method: Normalization method ("reinhard", "macenko", "vahadane")
            target: Target image for normalization (optional)
            use_target_params: Whether to use predefined target parameters
            **kwargs: Additional arguments for normalizer

        Returns:
            Normalized image

        Example:
            >>> normalized = processor.normalize_stain(image, method="macenko")
            >>> # With custom target
            >>> normalized = processor.normalize_stain(image, method="reinhard", target=target_img)
        """
        from ..loaders.Slide.slide import Slide
        from .wsi import (
            STAIN_NORM_TARGETS,
            MacenkoNormalizer,
            ReinhardNormalizer,
            VahadaneNormalizer,
        )

        # Extract image array if Slide object provided
        if isinstance(wsi, Slide):
            image = np.asarray(wsi.slide)
        else:
            image = wsi

        # Initialize normalizer based on method
        if method.lower() == "reinhard":
            normalizer = ReinhardNormalizer()
            if use_target_params and target is None:
                # Use predefined TCGA target
                normalizer.set_target_params(STAIN_NORM_TARGETS["tcga_avg"])
            elif target is not None:
                normalizer.fit(target)
            else:
                raise ValueError("Either target image or use_target_params=True required")

        elif method.lower() == "macenko":
            normalizer = MacenkoNormalizer(**kwargs)
            if use_target_params and target is None:
                normalizer.set_target_params(STAIN_NORM_TARGETS["tcga_avg"])
            elif target is not None:
                normalizer.fit(target)
            else:
                raise ValueError("Either target image or use_target_params=True required")

        elif method.lower() == "vahadane":
            normalizer = VahadaneNormalizer(**kwargs)
            if use_target_params and target is None:
                normalizer.set_target_params(STAIN_NORM_TARGETS["tcga_avg"])
            elif target is not None:
                normalizer.fit(target)
            else:
                raise ValueError("Either target image or use_target_params=True required")
        else:
            raise ValueError(
                f"Unknown normalization method: {method}. Supported: reinhard, macenko, vahadane"
            )

        return normalizer.transform(image)

    def separate_stains(self, wsi, method: str = "hed") -> Dict[str, np.ndarray]:
        """
        Separate H&E stain components.

        Args:
            wsi: Slide object or image array
            method: Separation method ("hed", "macenko", "custom")

        Returns:
            Dictionary with keys:
                - 'hematoxylin': Hematoxylin channel
                - 'eosin': Eosin channel
                - 'dab': DAB/background channel
                - 'rgb_h': RGB visualization of hematoxylin
                - 'rgb_e': RGB visualization of eosin

        Example:
            >>> stains = processor.separate_stains(image)
            >>> h_channel = stains['hematoxylin']
            >>> e_channel = stains['eosin']
        """
        from ..loaders.Slide.slide import Slide
        from .wsi import StainSeparator

        # Extract image array if Slide object provided
        if isinstance(wsi, Slide):
            image = np.asarray(wsi.slide)
        else:
            image = wsi

        separator = StainSeparator(method=method)
        result = separator.separate(image)

        # Rename 'background' to 'dab' for consistency with website docs
        return {
            "hematoxylin": result["hematoxylin"],
            "eosin": result["eosin"],
            "dab": result.get("background", np.zeros_like(result["hematoxylin"])),
            "rgb_h": result.get("rgb_h"),
            "rgb_e": result.get("rgb_e"),
            "rgb_d": result.get("rgb_d"),
            "concentrations": result.get("concentrations"),
        }

    def extract_patches(
        self,
        wsi,
        tissue_mask: Optional[np.ndarray] = None,
        patch_size: int = 256,
        overlap: float = 0.0,
        min_tissue_percentage: float = 0.5,
        target_patch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Extract patches from tissue regions.

        .. deprecated::
            Use ``PatchExtractor`` from ``honeybee.processors.wsi`` instead.

        Args:
            wsi: Slide object
            tissue_mask: Binary tissue mask (if None, uses wsi's tissue detection)
            patch_size: Size of patches to extract at WSI resolution
            overlap: Overlap between patches (0.0 to 1.0)
            min_tissue_percentage: Minimum tissue content for patch inclusion
            target_patch_size: Resize patches to this size (default: same as patch_size)

        Returns:
            Array of patches with shape (N, H, W, 3)

        Example:
            >>> patches = processor.extract_patches(wsi, tissue_mask, patch_size=256)
        """
        warnings.warn(
            "PathologyProcessor.extract_patches() is deprecated. "
            "Use PatchExtractor from honeybee.processors.wsi instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if target_patch_size is None:
            target_patch_size = patch_size

        # Use Slide's built-in patch extraction if available and no custom tissue_mask
        if hasattr(wsi, "load_patches_concurrently") and tissue_mask is None and overlap == 0.0:
            try:
                patches = wsi.load_patches_concurrently(target_patch_size=target_patch_size)
                return patches
            except Exception as e:
                warnings.warn(f"Concurrent patch loading failed: {e}. Using fallback method.")

        # Fallback: manual patch extraction
        patches = []
        threshold = min_tissue_percentage

        # Build set of addresses to iterate, respecting overlap/stride
        all_addresses = list(wsi.iterateTiles())

        # If overlap > 0, filter addresses by stride
        if overlap > 0.0 and hasattr(wsi, "tileSize"):
            tile_size_actual = wsi.tileSize
            stride_tiles = max(1, int(tile_size_actual * (1 - overlap) / tile_size_actual))
            all_addresses = [
                (x, y)
                for (x, y) in all_addresses
                if x % stride_tiles == 0 and y % stride_tiles == 0
            ]

        for address in all_addresses:
            tile_info = wsi.tileDictionary[address]

            # Check tissue content
            tissue_level = tile_info.get("tissueLevel", None)

            if tissue_level is not None:
                # Use existing tissueLevel from deep learning detection
                if tissue_level >= threshold:
                    pass  # will extract below
                else:
                    continue
            elif tissue_mask is not None:
                # Use provided tissue_mask to compute tissue percentage for this tile
                mask_h, mask_w = tissue_mask.shape[:2]
                grid_y = getattr(wsi, "numTilesInY", 1)
                grid_x = getattr(wsi, "numTilesInX", 1)
                ax, ay = address

                # Map tile address to mask region
                region_x_start = int(ax * mask_w / grid_x)
                region_x_end = int((ax + 1) * mask_w / grid_x)
                region_y_start = int(ay * mask_h / grid_y)
                region_y_end = int((ay + 1) * mask_h / grid_y)

                # Clamp to valid range
                region_x_start = max(0, min(region_x_start, mask_w))
                region_x_end = max(0, min(region_x_end, mask_w))
                region_y_start = max(0, min(region_y_start, mask_h))
                region_y_end = max(0, min(region_y_end, mask_h))

                region = tissue_mask[region_y_start:region_y_end, region_x_start:region_x_end]
                if region.size > 0:
                    tissue_pct = np.mean(region > 0)
                else:
                    tissue_pct = 0.0

                if tissue_pct < threshold:
                    continue
            else:
                # No tissueLevel and no mask — include all patches
                pass

            # Extract patch
            patch = wsi.getTile(address, writeToNumpy=True)
            if patch is not None and patch.shape[2] >= 3:
                patch = patch[:, :, :3]  # RGB only

                # Resize if needed
                if patch.shape[0] != target_patch_size or patch.shape[1] != target_patch_size:
                    from skimage.transform import resize

                    patch = resize(
                        patch, (target_patch_size, target_patch_size, 3), preserve_range=True
                    ).astype(np.uint8)

                patches.append(patch)

        return (
            np.array(patches) if patches else np.zeros((0, target_patch_size, target_patch_size, 3))
        )

    def generate_embeddings(
        self, patches, batch_size: int = 32, progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings from patches using the selected foundation model.

        Args:
            patches: Array of patches with shape (N, H, W, 3) or a Patches object
            batch_size: Batch size for processing
            progress: Show a tqdm progress bar over batches

        Returns:
            Embeddings array with shape (N, embedding_dim)

        Example:
            >>> embeddings = processor.generate_embeddings(patches)
            >>> embeddings = processor.generate_embeddings(patches, progress=True)
        """
        # Accept Patches container objects
        try:
            from .wsi.patches import Patches

            if isinstance(patches, Patches):
                patches = patches.images
        except ImportError:
            pass

        # Wrap legacy model objects that were assigned directly
        from ..models.registry.protocol import EmbeddingModel as _EmbProto

        if self.embedding_model is not None and not isinstance(
            self.embedding_model, _EmbProto
        ):
            self.embedding_model = _LegacyEmbedder(self.embedding_model, self.model_name)

        # Lazy-load via the registry
        if self.embedding_model is None:
            from ..models.registry import load_model

            self.embedding_model = load_model(
                model=self.model_name,
                provider=self._provider,
                model_path=self.model_path,
                **self._model_kwargs,
            )

        if progress:
            import math

            from tqdm.auto import tqdm

            from ..models.registry.providers import _TimmImageModel

            patches_list = _TimmImageModel._to_list(patches)
            n_batches = math.ceil(len(patches_list) / batch_size)
            all_embeddings = []
            for i in tqdm(
                range(0, len(patches_list), batch_size),
                total=n_batches,
                desc=f"Embedding ({self.model_name})",
                unit="batch",
            ):
                batch = patches_list[i : i + batch_size]
                emb = self.embedding_model.generate_embeddings(batch, batch_size=len(batch))
                all_embeddings.append(emb)
            return np.vstack(all_embeddings)

        return self.embedding_model.generate_embeddings(patches, batch_size=batch_size)

    def aggregate_embeddings(self, embeddings: np.ndarray, method: str = "mean") -> np.ndarray:
        """
        Aggregate patch embeddings to slide-level representation.

        Args:
            embeddings: Patch embeddings with shape (N, embedding_dim)
            method: Aggregation method ("mean", "max", "median", "std", "concat")

        Returns:
            Aggregated embedding with shape (embedding_dim,) or larger for concat

        Example:
            >>> slide_embedding = processor.aggregate_embeddings(embeddings, method="mean")
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings to aggregate")

        if method == "mean":
            return np.mean(embeddings, axis=0)
        elif method == "max":
            return np.max(embeddings, axis=0)
        elif method == "median":
            return np.median(embeddings, axis=0)
        elif method == "std":
            return np.std(embeddings, axis=0)
        elif method == "concat":
            # Concatenate mean and std
            mean_emb = np.mean(embeddings, axis=0)
            std_emb = np.std(embeddings, axis=0)
            return np.concatenate([mean_emb, std_emb])
        else:
            raise ValueError(
                f"Unknown aggregation method: {method}. Supported: mean, max, median, std, concat"
            )

    def process_slide(
        self,
        slide_path: Union[str, Path],
        normalize_stain: bool = True,
        normalization_method: str = "macenko",
        patch_size: int = 256,
        min_tissue_percentage: float = 0.5,
        aggregation_method: str = "mean",
        **kwargs,
    ) -> Dict:
        """
        Complete end-to-end processing pipeline for a single slide.

        Args:
            slide_path: Path to WSI file
            normalize_stain: Whether to apply stain normalization
            normalization_method: Stain normalization method
            patch_size: Size of patches to extract
            min_tissue_percentage: Minimum tissue content for patches
            aggregation_method: Method for aggregating patch embeddings
            **kwargs: Additional arguments for pipeline components

        Returns:
            Dictionary containing:
                - 'slide': Slide object
                - 'tissue_mask': Tissue detection mask
                - 'patches': Extracted patches
                - 'embeddings': Patch embeddings
                - 'slide_embedding': Aggregated slide-level embedding

        Example:
            >>> result = processor.process_slide("slide.svs")
            >>> slide_embedding = result['slide_embedding']
        """
        # Use new classes internally
        try:
            from ..loaders.Slide.slide import Slide
            from .wsi.patch_extractor import PatchExtractor

            # Load WSI using new Slide class
            wsi = Slide(str(slide_path))

            # Detect tissue
            wsi.detect_tissue(method="otsu_hsv")
            tissue_mask = wsi.tissue_mask

            # Extract patches using new PatchExtractor
            stride = patch_size
            extractor = PatchExtractor(
                patch_size=patch_size,
                stride=stride,
                min_tissue_ratio=min_tissue_percentage,
            )
            patches_obj = extractor.extract(wsi)
            patches = patches_obj.images if len(patches_obj) > 0 else np.array([])

            # Normalize patches if requested
            if normalize_stain and len(patches_obj) > 0:
                try:
                    patches_obj = patches_obj.normalize(method=normalization_method)
                    patches = patches_obj.images
                except Exception as e:
                    warnings.warn(f"Batch normalization failed: {e}")

        except ImportError:
            # Fallback to legacy approach if new classes unavailable
            wsi = self.load_wsi(slide_path, tile_size=patch_size, **kwargs)

            if hasattr(wsi, "tissue_detector") and wsi.tissue_detector is not None:
                tissue_mask = None
            else:
                tissue_mask = self.detect_tissue(wsi, method="otsu_hsv")

            patches = self.extract_patches(
                wsi,
                tissue_mask=tissue_mask,
                patch_size=patch_size,
                min_tissue_percentage=min_tissue_percentage,
            )

            if normalize_stain and len(patches) > 0:
                normalized_patches = []
                for patch in patches:
                    try:
                        normalized_patch = self.normalize_stain(
                            patch, method=normalization_method, use_target_params=True
                        )
                        normalized_patches.append(normalized_patch)
                    except Exception as e:
                        warnings.warn(f"Normalization failed for patch: {e}")
                        normalized_patches.append(patch)
                patches = np.array(normalized_patches)

        # Generate embeddings
        if len(patches) > 0:
            embeddings = self.generate_embeddings(patches)
            slide_embedding = self.aggregate_embeddings(embeddings, method=aggregation_method)
        else:
            embeddings = np.array([])
            slide_embedding = np.array([])

        return {
            "slide": wsi,
            "tissue_mask": tissue_mask,
            "patches": patches,
            "embeddings": embeddings,
            "slide_embedding": slide_embedding,
            "num_patches": len(patches),
        }

    def _get_overview_image(self, wsi, min_dim: int = 1024) -> np.ndarray:
        """Read WSI at a low-res pyramid level for fast overview operations.

        .. deprecated::
            Use ``slide.get_thumbnail()`` directly.
        """
        # Use wsi.slide if available (mock objects or legacy Slide instances)
        if hasattr(wsi, "slide") and not callable(wsi.slide):
            return np.asarray(wsi.slide)
        return np.asarray(wsi)

    # ------------------------------------------------------------------ #
    # Group A: WSI Information & Visualization
    # ------------------------------------------------------------------ #

    def get_slide_info(self, wsi) -> Dict:
        """
        Return metadata dictionary for a loaded WSI.

        .. deprecated::
            Use ``slide.info`` property directly.

        Args:
            wsi: Slide object returned by load_wsi()

        Returns:
            Dict with keys: dimensions, magnification, scanner, num_levels,
            tile_grid, file_size_bytes
        """
        warnings.warn(
            "PathologyProcessor.get_slide_info() is deprecated. "
            "Use slide.info property directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        info: Dict = {}

        # Dimensions at selected level
        info["dimensions"] = {
            "width": int(getattr(wsi.slide, "width", 0)),
            "height": int(getattr(wsi.slide, "height", 0)),
        }

        # Pyramid levels
        resolutions = getattr(wsi.img, "resolutions", None)
        if resolutions is not None:
            info["num_levels"] = resolutions.get("level_count", 1)
            info["level_dimensions"] = resolutions.get("level_dimensions", [])
        else:
            info["num_levels"] = 1
            info["level_dimensions"] = []

        # Scanner / magnification from metadata
        metadata = getattr(wsi.img, "metadata", None) or {}
        info["magnification"] = metadata.get("objectivePower", metadata.get("magnification", None))
        info["scanner"] = metadata.get("scanner", metadata.get("vendor", None))

        # Tile grid
        info["tile_grid"] = {
            "tiles_x": getattr(wsi, "numTilesInX", 0),
            "tiles_y": getattr(wsi, "numTilesInY", 0),
        }

        # File size
        slide_path = getattr(wsi, "slide_image_path", None) or getattr(wsi, "slideFilePath", None)
        if slide_path and os.path.exists(slide_path):
            info["file_size_bytes"] = os.path.getsize(slide_path)
        else:
            info["file_size_bytes"] = None

        return info

    def get_thumbnail(self, wsi, size: tuple = (512, 512)) -> np.ndarray:
        """
        Get a downsampled WSI thumbnail as an RGB numpy array.

        .. deprecated::
            Use ``slide.get_thumbnail(size)`` directly.

        Args:
            wsi: Slide object
            size: Target (width, height) tuple

        Returns:
            RGB numpy array of shape (height, width, 3)
        """
        warnings.warn(
            "PathologyProcessor.get_thumbnail() is deprecated. "
            "Use slide.get_thumbnail(size) directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        from skimage.transform import resize as sk_resize

        image = self._get_overview_image(wsi, min_dim=max(size[0], size[1]))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] > 3:
            image = image[:, :, :3]

        target_h, target_w = size[1], size[0]
        thumbnail = sk_resize(image, (target_h, target_w, 3), preserve_range=True).astype(np.uint8)
        return thumbnail

    def visualize_tissue_mask(
        self,
        wsi,
        tissue_mask: np.ndarray,
        output_path: Optional[str] = None,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlay a tissue mask on a WSI thumbnail.

        .. deprecated::
            Use ``slide.plot_tissue_mask()`` directly.

        Args:
            wsi: Slide object
            tissue_mask: Binary tissue mask
            output_path: Optional file path to save the composite image
            alpha: Transparency of the overlay (0-1)

        Returns:
            Composite RGB image as numpy array
        """
        warnings.warn(
            "PathologyProcessor.visualize_tissue_mask() is deprecated. "
            "Use slide.plot_tissue_mask() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        from skimage.transform import resize as sk_resize

        thumbnail = self.get_thumbnail(wsi)
        h, w = thumbnail.shape[:2]

        # Nearest-neighbor resize preserves binary mask edges
        mask_resized = sk_resize(
            tissue_mask.astype(np.float64), (h, w), order=0, preserve_range=True
        )
        mask_bool = mask_resized > 0.5

        # Create colored overlay: green on tissue, red on background
        overlay = thumbnail.copy().astype(np.float64)
        # Tissue regions: green tint
        overlay[mask_bool, 1] = np.minimum(overlay[mask_bool, 1] + 80, 255)
        # Background regions: red tint
        overlay[~mask_bool, 0] = np.minimum(overlay[~mask_bool, 0] + 100, 255)
        overlay[~mask_bool, 1] = overlay[~mask_bool, 1] * 0.6
        overlay[~mask_bool, 2] = overlay[~mask_bool, 2] * 0.6

        composite = ((1 - alpha) * thumbnail.astype(np.float64) + alpha * overlay).astype(np.uint8)

        if output_path is not None:
            from PIL import Image

            Image.fromarray(composite).save(output_path)

        return composite

    # ------------------------------------------------------------------ #
    # Group B: Patch Operations
    # ------------------------------------------------------------------ #

    def normalize_patches(
        self, patches: np.ndarray, method: str = "macenko", **kwargs
    ) -> np.ndarray:
        """
        Batch-normalize an array of already-extracted patches.

        Args:
            patches: Array of shape (N, H, W, 3)
            method: Normalization method ("reinhard", "macenko", "vahadane")
            **kwargs: Additional arguments for the normalizer

        Returns:
            Normalized patches array of same shape
        """
        normalized = []
        for patch in patches:
            try:
                norm_patch = self.normalize_stain(
                    patch, method=method, use_target_params=True, **kwargs
                )
                normalized.append(norm_patch)
            except Exception:
                normalized.append(patch)
        return np.array(normalized) if normalized else patches

    def save_patches(
        self,
        patches: np.ndarray,
        output_dir: str,
        prefix: str = "patch",
        format: str = "png",
    ) -> List[str]:
        """
        Save patches as individual image files.

        Args:
            patches: Array of shape (N, H, W, 3)
            output_dir: Directory to save patches
            prefix: Filename prefix
            format: Image format ("png", "jpg", etc.)

        Returns:
            List of saved file paths
        """
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        for i, patch in enumerate(patches):
            fname = f"{prefix}_{i:05d}.{format}"
            fpath = os.path.join(output_dir, fname)
            Image.fromarray(patch.astype(np.uint8)).save(fpath)
            saved_paths.append(fpath)
        return saved_paths

    def compute_patch_quality(self, patches: np.ndarray) -> np.ndarray:
        """
        Score each patch (0-1) for quality based on blur, tissue content, and pen marks.

        Args:
            patches: Array of shape (N, H, W, 3)

        Returns:
            Array of shape (N,) with quality scores
        """
        import cv2

        scores = np.zeros(len(patches), dtype=np.float32)

        for i, patch in enumerate(patches):
            patch_uint8 = patch.astype(np.uint8) if patch.dtype != np.uint8 else patch

            # 1. Blur detection via Laplacian variance (higher = sharper)
            gray = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize: typical sharp tissue ~100-2000, blurry <50
            blur_score = min(1.0, laplacian_var / 500.0)

            # 2. Tissue percentage (non-white pixels)
            white_thresh = 220
            non_white = np.mean(np.all(patch_uint8 < white_thresh, axis=-1))
            tissue_score = non_white

            # 3. Pen mark detection (look for saturated non-tissue colors)
            hsv = cv2.cvtColor(patch_uint8, cv2.COLOR_RGB2HSV)
            high_sat = hsv[:, :, 1] > 100
            hue = hsv[:, :, 0]
            # Pen hue ranges in OpenCV (0-180): red <10 or >170, green 35-85, blue 85-130
            pen_hue = (hue < 10) | (hue > 170) | ((hue > 35) & (hue < 130))
            pen_ratio = np.mean(high_sat & pen_hue)

            # Combined score: blur + tissue, then hard-penalize if pen marks detected
            score = 0.5 * blur_score + 0.5 * tissue_score
            if pen_ratio > 0.01:
                score *= max(0.0, 1.0 - pen_ratio * 5)  # 20% pen → score × 0
            scores[i] = score

        return scores

    def get_patch_coordinates(
        self,
        wsi,
        tissue_mask: Optional[np.ndarray] = None,
        patch_size: Optional[int] = None,
        overlap: float = 0.0,
        min_tissue_percentage: float = 0.5,
    ) -> np.ndarray:
        """
        Return patch coordinates in WSI space for patches that would be extracted.

        .. deprecated::
            Use ``PatchExtractor.get_coordinates(slide)`` instead.

        Uses the same filtering logic as extract_patches so coordinates correspond
        1:1 with extracted patches.

        Args:
            wsi: Slide object
            tissue_mask: Binary tissue mask (same as passed to extract_patches)
            patch_size: Output patch size. Used only for tile filtering consistency
                with extract_patches; reported width/height are always the WSI-space
                tile dimensions (not the resized output pixel size).
            overlap: Overlap between patches (0.0 to 1.0)
            min_tissue_percentage: Minimum tissue content for patch inclusion

        Returns:
            Array of shape (N, 4) with columns [x, y, width, height] in WSI space
        """
        warnings.warn(
            "PathologyProcessor.get_patch_coordinates() is deprecated. "
            "Use PatchExtractor.get_coordinates(slide) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        threshold = min_tissue_percentage

        all_addresses = list(wsi.iterateTiles())

        if overlap > 0.0 and hasattr(wsi, "tileSize"):
            tile_size_actual = wsi.tileSize
            stride_tiles = max(1, int(tile_size_actual * (1 - overlap) / tile_size_actual))
            all_addresses = [
                (x, y)
                for (x, y) in all_addresses
                if x % stride_tiles == 0 and y % stride_tiles == 0
            ]

        coords = []
        for address in all_addresses:
            tile_info = wsi.tileDictionary[address]

            tissue_level = tile_info.get("tissueLevel", None)

            if tissue_level is not None:
                if tissue_level < threshold:
                    continue
            elif tissue_mask is not None:
                mask_h, mask_w = tissue_mask.shape[:2]
                grid_y = getattr(wsi, "numTilesInY", 1)
                grid_x = getattr(wsi, "numTilesInX", 1)
                ax, ay = address

                region_x_start = max(0, min(int(ax * mask_w / grid_x), mask_w))
                region_x_end = max(0, min(int((ax + 1) * mask_w / grid_x), mask_w))
                region_y_start = max(0, min(int(ay * mask_h / grid_y), mask_h))
                region_y_end = max(0, min(int((ay + 1) * mask_h / grid_y), mask_h))

                region = tissue_mask[region_y_start:region_y_end, region_x_start:region_x_end]
                tissue_pct = np.mean(region > 0) if region.size > 0 else 0.0

                if tissue_pct < threshold:
                    continue

            w = tile_info.get("width", getattr(wsi, "tileSize", 0))
            h = tile_info.get("height", getattr(wsi, "tileSize", 0))
            coords.append(
                [
                    tile_info.get("x", address[0]),
                    tile_info.get("y", address[1]),
                    w,
                    h,
                ]
            )
        return np.array(coords) if coords else np.zeros((0, 4))

    # ------------------------------------------------------------------ #
    # Group C: Batch & Pipeline
    # ------------------------------------------------------------------ #

    def process_batch(self, slide_paths: List[Union[str, Path]], **kwargs) -> List[Dict]:
        """
        Process multiple slides, collecting results and errors.

        Args:
            slide_paths: List of paths to WSI files
            **kwargs: Arguments forwarded to process_slide()

        Returns:
            List of result dicts. Failed slides have 'error' key.
        """
        results = []
        for path in slide_paths:
            try:
                result = self.process_slide(path, **kwargs)
                result["path"] = str(path)
                results.append(result)
            except Exception as e:
                results.append({"path": str(path), "error": str(e)})
        return results

    # ------------------------------------------------------------------ #
    # Group D: Utilities
    # ------------------------------------------------------------------ #

    def save_embeddings(
        self, embeddings: np.ndarray, path: str, metadata: Optional[Dict] = None
    ) -> None:
        """
        Save embeddings to a .npy file with optional metadata sidecar .json.

        Args:
            embeddings: Numpy array of embeddings
            path: File path (should end with .npy)
            metadata: Optional dict to save alongside as .json
        """
        np.save(path, embeddings)
        if metadata is not None:
            meta_path = str(path).rsplit(".", 1)[0] + ".json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

    def load_embeddings(self, path: str) -> np.ndarray:
        """
        Load embeddings from a .npy file.

        Args:
            path: Path to .npy file

        Returns:
            Numpy array of embeddings
        """
        return np.load(path)

    def get_model_info(self) -> Dict:
        """
        Return information about the configured embedding model.

        Returns:
            Dict with model_name, embedding_dim, expected_input_size, device, is_loaded
        """
        from ..models.registry import _PRESET_REGISTRY

        config = _PRESET_REGISTRY.get(self.model_name)

        device = "unknown"
        if self.embedding_model is not None:
            device_attr = getattr(self.embedding_model, "device", None)
            if device_attr is not None:
                device = str(device_attr)

        return {
            "model_name": self.model_name,
            "embedding_dim": (
                config.embedding_dim
                if config
                else getattr(self.embedding_model, "embedding_dim", None)
            ),
            "expected_input_size": config.input_size if config else 224,
            "device": device,
            "is_loaded": self.embedding_model is not None,
        }

    def compare_normalizations(
        self, image: np.ndarray, methods: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply multiple normalization methods to the same image for comparison.

        Args:
            image: Input RGB image array
            methods: List of method names (default: ["reinhard", "macenko", "vahadane"])

        Returns:
            Dict mapping method name to normalized image
        """
        if methods is None:
            methods = ["reinhard", "macenko", "vahadane"]

        results = {}
        for method in methods:
            try:
                results[method] = self.normalize_stain(image, method=method, use_target_params=True)
            except Exception as e:
                warnings.warn(f"Normalization with {method} failed: {e}")
                results[method] = image.copy()
        return results

    def get_tissue_stats(self, wsi_or_image, method: str = "otsu_hsv") -> Dict:
        """
        Get tissue statistics for a WSI or image array.

        Args:
            wsi_or_image: Slide object or RGB numpy array
            method: Classical detection method to use

        Returns:
            Dict with tissue_ratio, tissue_pixels, total_pixels, num_regions,
            largest_region_area, mean_region_area
        """
        from .wsi import ClassicalTissueDetector

        # Handle Slide objects
        try:
            from ..loaders.Slide.slide import Slide

            if isinstance(wsi_or_image, Slide):
                image = np.asarray(wsi_or_image.slide)
            else:
                image = wsi_or_image
        except ImportError:
            image = wsi_or_image

        detector = ClassicalTissueDetector(method=method)
        return detector.get_tissue_stats(image)

    # ------------------------------------------------------------------ #
    # Group E: Visualization
    # ------------------------------------------------------------------ #

    def plot_feature_map(
        self,
        patches,
        embeddings: np.ndarray,
        slide,
        thumbnail_size: tuple = (4096, 4096),
        umap_kwargs: Optional[Dict] = None,
        figsize: tuple = (20, 10),
        ax=None,
    ):
        """UMAP 3D -> RGB deep feature map overlaid on WSI thumbnail.

        Parameters
        ----------
        patches : Patches
            Patches container (for coordinates).
        embeddings : np.ndarray
            Shape ``(N, D)`` embedding array.
        slide : Slide
            Slide object for thumbnail.
        thumbnail_size : tuple
            Thumbnail resolution.
        umap_kwargs : dict, optional
            Overrides for UMAP constructor.
        figsize : tuple
            Figure size.
        ax : array of Axes, optional
            Pre-created 1x2 axes.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from umap import UMAP

        from .wsi._vis_utils import _composite_overlay, _rasterize_patches
        from .wsi.patches import Patches as PatchesCls

        if isinstance(patches, PatchesCls):
            coordinates = patches.coordinates
        else:
            coordinates = np.asarray(patches)

        # UMAP 3D projection
        default_umap = dict(
            n_components=3, metric="manhattan", n_neighbors=10,
            spread=0.5, random_state=42,
        )
        if umap_kwargs:
            default_umap.update(umap_kwargs)
        reducer = UMAP(**default_umap)
        umap_3d = reducer.fit_transform(embeddings)

        # Min-max normalize to [0, 1] for RGB
        umap_rgb = umap_3d.copy()
        for d in range(3):
            dmin, dmax = umap_rgb[:, d].min(), umap_rgb[:, d].max()
            if dmax > dmin:
                umap_rgb[:, d] = (umap_rgb[:, d] - dmin) / (dmax - dmin)
            else:
                umap_rgb[:, d] = 0.5

        # Rasterize onto thumbnail
        thumbnail = slide.get_thumbnail(size=thumbnail_size)
        thumb_h, thumb_w = thumbnail.shape[:2]

        overlay = _rasterize_patches(
            coordinates, umap_rgb, slide.dimensions,
            (thumb_h, thumb_w), alpha=1.0,
        )
        composite = _composite_overlay(thumbnail, overlay, tissue_blend=0.15)

        # Plot
        created_fig = ax is None
        if ax is not None:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].imshow(thumbnail)
        axes[0].set_title("WSI Thumbnail")
        axes[0].axis("off")

        axes[1].imshow(composite)
        axes[1].set_title(f"UMAP Deep Feature Map ({len(embeddings)} patches, 3D → RGB)")
        axes[1].axis("off")

        fig.suptitle("Deep Feature Visualization — Similar tissue ≈ similar color", fontsize=14)
        fig.tight_layout()
        if created_fig:
            plt.close(fig)
        return fig
