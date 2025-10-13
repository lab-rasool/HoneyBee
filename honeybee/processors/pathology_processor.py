"""
PathologyProcessor - Unified interface for whole slide image processing

This module provides a high-level API for pathology image analysis, combining
tissue detection, stain normalization, patch extraction, and embedding generation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Lazy imports - loaded when needed to avoid requiring all dependencies at import time
# These will be imported when the corresponding methods are called


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

    def __init__(self, model: str = "uni", model_path: Optional[str] = None):
        """
        Initialize PathologyProcessor with specified embedding model.

        Args:
            model: Embedding model to use ("uni", "uni2", "virchow2", "remedis")
            model_path: Path to model weights (required for UNI, optional for others)
        """
        self.model_name = model.lower()
        self.model_path = model_path
        self.embedding_model = None
        self.tissue_detector_dl = None

        # Lazy loading - embedding model will be loaded when needed
        if self.model_name not in ["uni", "uni2", "virchow2", "remedis"]:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Supported models: uni, uni2, virchow2, remedis"
            )

    def load_wsi(
        self,
        path: Union[str, Path],
        tile_size: int = 512,
        max_patches: int = 500,
        visualize: bool = False,
        verbose: bool = False
    ):
        """
        Load whole slide image.

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
        from ..loaders.Slide.slide import Slide

        slide = Slide(
            slide_image_path=str(path),
            tile_size=tile_size,
            max_patches=max_patches,
            visualize=visualize,
            verbose=verbose
        )
        return slide

    def detect_tissue(
        self,
        wsi,
        method: str = "otsu",
        tissue_detector_path: Optional[str] = None,
        min_tissue_size: int = 1000,
        **kwargs
    ) -> np.ndarray:
        """
        Detect tissue regions in WSI.

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
        from ..loaders.Slide.slide import Slide
        from .wsi import ClassicalTissueDetector

        # Extract image array if Slide object provided
        if isinstance(wsi, Slide):
            image = np.asarray(wsi.slide)
        else:
            image = wsi

        if method == "deeplearning":
            from ..models.TissueDetector.tissue_detector import TissueDetector

            # Use deep learning tissue detector
            if self.tissue_detector_dl is None:
                if tissue_detector_path is None:
                    raise ValueError(
                        "tissue_detector_path required for deeplearning method"
                    )
                self.tissue_detector_dl = TissueDetector(model_path=tissue_detector_path)

            # If we have a Slide object, use its detection method
            if isinstance(wsi, Slide):
                # Reload slide with tissue detector
                wsi_with_detection = Slide(
                    slide_image_path=str(wsi.slide_image_path),
                    tile_size=wsi.tileSize,
                    tissue_detector=self.tissue_detector_dl,
                    visualize=False
                )

                # Extract tissue mask from predictions
                mask = np.zeros((wsi_with_detection.numTilesInY, wsi_with_detection.numTilesInX))
                for address in wsi_with_detection.iterateTiles():
                    tile_info = wsi_with_detection.tileDictionary[address]
                    if 'tissueLevel' in tile_info:
                        mask[address[1], address[0]] = tile_info['tissueLevel']

                return mask > 0.5  # Binary mask with threshold
            else:
                # For raw images, we need to tile and detect
                warnings.warn(
                    "Deep learning detection on raw arrays not fully implemented. "
                    "Using classical detection as fallback."
                )
                method = "otsu_hsv"

        # Use classical detection methods
        detector = ClassicalTissueDetector(
            method=method,
            min_tissue_size=min_tissue_size,
            **kwargs
        )
        return detector.detect(image)

    def normalize_stain(
        self,
        wsi,
        method: str = "reinhard",
        target: Optional[np.ndarray] = None,
        use_target_params: bool = True,
        **kwargs
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
            ReinhardNormalizer,
            MacenkoNormalizer,
            VahadaneNormalizer,
            STAIN_NORM_TARGETS
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
                f"Unknown normalization method: {method}. "
                f"Supported: reinhard, macenko, vahadane"
            )

        return normalizer.transform(image)

    def separate_stains(
        self,
        wsi,
        method: str = "hed"
    ) -> Dict[str, np.ndarray]:
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
            'hematoxylin': result['hematoxylin'],
            'eosin': result['eosin'],
            'dab': result.get('background', np.zeros_like(result['hematoxylin'])),
            'rgb_h': result.get('rgb_h'),
            'rgb_e': result.get('rgb_e'),
            'rgb_d': result.get('rgb_d'),
            'concentrations': result.get('concentrations')
        }

    def extract_patches(
        self,
        wsi,
        tissue_mask: Optional[np.ndarray] = None,
        patch_size: int = 256,
        overlap: float = 0.0,
        min_tissue_percentage: float = 0.5,
        target_patch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract patches from tissue regions.

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
        if target_patch_size is None:
            target_patch_size = patch_size

        # Use Slide's built-in patch extraction if available
        if hasattr(wsi, 'load_patches_concurrently'):
            try:
                # Extract patches with high tissue content
                patches = wsi.load_patches_concurrently(target_patch_size=target_patch_size)
                return patches
            except Exception as e:
                warnings.warn(f"Concurrent patch loading failed: {e}. Using fallback method.")

        # Fallback: manual patch extraction
        patches = []
        threshold = min_tissue_percentage

        for address in wsi.iterateTiles():
            tile_info = wsi.tileDictionary[address]

            # Check tissue content
            tissue_level = tile_info.get('tissueLevel', 0)
            if tissue_level >= threshold:
                # Extract patch
                patch = wsi.getTile(address, writeToNumpy=True)
                if patch is not None and patch.shape[2] >= 3:
                    patch = patch[:, :, :3]  # RGB only

                    # Resize if needed
                    if patch.shape[0] != target_patch_size or patch.shape[1] != target_patch_size:
                        from skimage.transform import resize
                        patch = resize(
                            patch,
                            (target_patch_size, target_patch_size, 3),
                            preserve_range=True
                        ).astype(np.uint8)

                    patches.append(patch)

        return np.array(patches) if patches else np.zeros((0, target_patch_size, target_patch_size, 3))

    def generate_embeddings(
        self,
        patches: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings from patches using the selected foundation model.

        Args:
            patches: Array of patches with shape (N, H, W, 3)
            batch_size: Batch size for processing

        Returns:
            Embeddings array with shape (N, embedding_dim)

        Example:
            >>> embeddings = processor.generate_embeddings(patches)
        """
        if self.embedding_model is None:
            if self.model_path is None:
                raise ValueError(
                    f"Model path required for {self.model_name}. "
                    f"Reinitialize with model_path parameter."
                )
            # Lazy load model
            if self.model_name == "uni":
                from ..models.UNI.uni import UNI
                self.embedding_model = UNI(model_path=self.model_path)
            elif self.model_name == "uni2":
                from ..models.UNI2.uni2 import UNI2
                self.embedding_model = UNI2(model_path=self.model_path)
            elif self.model_name == "virchow2":
                from ..models.Virchow2.virchow2 import Virchow2
                self.embedding_model = Virchow2(model_path=self.model_path)
            elif self.model_name == "remedis":
                from ..models.REMEDIS.remedis import REMEDIS
                self.embedding_model = REMEDIS(model_path=self.model_path)

        # Generate embeddings based on model type
        if self.model_name in ["uni", "remedis"]:
            # UNI and REMEDIS use load_model_and_predict
            all_embeddings = []
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size]
                embeddings = self.embedding_model.load_model_and_predict(batch)
                all_embeddings.append(embeddings.cpu().numpy())
            return np.vstack(all_embeddings)

        elif self.model_name in ["uni2", "virchow2"]:
            # UNI2 and Virchow2 have generate_embeddings method
            if hasattr(self.embedding_model, 'generate_embeddings'):
                return self.embedding_model.generate_embeddings(patches, batch_size=batch_size)
            else:
                # Fallback to load_model_and_predict
                all_embeddings = []
                for i in range(0, len(patches), batch_size):
                    batch = patches[i:i+batch_size]
                    embeddings = self.embedding_model.load_model_and_predict(batch)
                    all_embeddings.append(embeddings.cpu().numpy())
                return np.vstack(all_embeddings)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def aggregate_embeddings(
        self,
        embeddings: np.ndarray,
        method: str = "mean"
    ) -> np.ndarray:
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
                f"Unknown aggregation method: {method}. "
                f"Supported: mean, max, median, std, concat"
            )

    def process_slide(
        self,
        slide_path: Union[str, Path],
        normalize_stain: bool = True,
        normalization_method: str = "macenko",
        patch_size: int = 256,
        min_tissue_percentage: float = 0.5,
        aggregation_method: str = "mean",
        **kwargs
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
        # Load WSI
        wsi = self.load_wsi(slide_path, tile_size=patch_size, **kwargs)

        # Detect tissue (using deep learning if slide has tissue detector)
        if hasattr(wsi, 'tissue_detector') and wsi.tissue_detector is not None:
            tissue_mask = None  # Will use slide's internal detection
        else:
            tissue_mask = self.detect_tissue(wsi, method="otsu_hsv")

        # Extract patches
        patches = self.extract_patches(
            wsi,
            tissue_mask=tissue_mask,
            patch_size=patch_size,
            min_tissue_percentage=min_tissue_percentage
        )

        # Normalize patches if requested
        if normalize_stain and len(patches) > 0:
            normalized_patches = []
            for patch in patches:
                try:
                    normalized_patch = self.normalize_stain(
                        patch,
                        method=normalization_method,
                        use_target_params=True
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
            'slide': wsi,
            'tissue_mask': tissue_mask,
            'patches': patches,
            'embeddings': embeddings,
            'slide_embedding': slide_embedding,
            'num_patches': len(patches)
        }
