"""
Pathology Processor for HoneyBee

Complete implementation for processing whole slide images (WSI) with GPU acceleration,
tissue detection, stain normalization, and embedding generation capabilities.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

# External dependencies
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# CuCIM for GPU-accelerated WSI processing
try:
    from cucim import CuImage
    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False
    warnings.warn("CuCIM not available. GPU acceleration will be limited.")

# Import HoneyBee components
import sys
import os
# Add parent directories to path to avoid circular imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from honeybee.loaders.Slide.slide import Slide, WholeSlideImageDataset
from honeybee.models.TissueDetector.tissue_detector import TissueDetector
try:
    from honeybee.models.UNI.uni import UNI
except ImportError:
    UNI = None
    warnings.warn("UNI model not available. Install required dependencies.")
try:
    from honeybee.models.UNI2.uni2 import UNI2
except ImportError:
    UNI2 = None
try:
    from honeybee.models.Virchow2.virchow2 import Virchow2
except ImportError:
    Virchow2 = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_WSI_FORMATS = [".svs", ".tiff", ".tif", ".scn", ".vms", ".vmu", ".ndpi", ".mrxs"]
DEFAULT_TILE_SIZE = 512
DEFAULT_TARGET_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_PATCHES = 500

# Stain normalization targets
STAIN_NORM_TARGETS = {
    "tcga_avg": {
        "mean_lab": np.array([66.98, 128.77, 113.74]),
        "std_lab": np.array([15.89, 10.22, 9.41]),
        "stain_matrix": np.array([
            [0.5626, 0.8269, 0.0000],
            [0.7201, -0.4738, 0.5063],
            [0.4062, -0.3028, -0.8616]
        ])
    }
}


class PathologyProcessor:
    """
    Main processor for pathology data in HoneyBee framework.
    
    Provides comprehensive WSI processing including:
    - GPU-accelerated loading via CuCIM
    - Tissue detection and segmentation
    - Stain normalization (Reinhard, Macenko, Vahadane)
    - Multi-scale patch extraction
    - Embedding generation with multiple models
    """
    
    def __init__(
        self,
        tile_size: int = DEFAULT_TILE_SIZE,
        target_size: int = DEFAULT_TARGET_SIZE,
        max_patches: int = DEFAULT_MAX_PATCHES,
        tissue_detector_path: Optional[str] = None,
        embedding_model: str = "uni",
        device: str = "cuda",
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = 4,
        stain_normalization: Optional[str] = None,
        stain_norm_target: str = "tcga_avg",
        verbose: bool = True
    ):
        """
        Initialize PathologyProcessor.
        
        Args:
            tile_size: Size of tiles to extract from WSI
            target_size: Target size for model input
            max_patches: Maximum number of patches to extract per slide
            tissue_detector_path: Path to tissue detector model weights
            embedding_model: Model to use for embeddings ("uni", "uni2", "virchow2")
            device: Device for computation ("cuda" or "cpu")
            batch_size: Batch size for processing
            num_workers: Number of workers for data loading
            stain_normalization: Stain normalization method ("reinhard", "macenko", "vahadane", None)
            stain_norm_target: Target for stain normalization
            verbose: Whether to print progress messages
        """
        self.tile_size = tile_size
        self.target_size = target_size
        self.max_patches = max_patches
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stain_normalization = stain_normalization
        self.stain_norm_target = stain_norm_target
        self.verbose = verbose
        
        # Initialize tissue detector
        self.tissue_detector = None
        if tissue_detector_path and os.path.exists(tissue_detector_path):
            self.tissue_detector = TissueDetector(tissue_detector_path, device=self.device)
            if self.verbose:
                logger.info(f"Loaded tissue detector from {tissue_detector_path}")
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embedding_model = self._initialize_embedding_model(embedding_model)
        
        # Initialize stain normalizer
        self.stain_normalizer = None
        if stain_normalization:
            self.stain_normalizer = self._initialize_stain_normalizer(stain_normalization)
    
    def _initialize_embedding_model(self, model_name: str):
        """Initialize the specified embedding model."""
        if model_name.lower() == "uni":
            model = UNI(device=self.device)
        elif model_name.lower() == "uni2" and UNI2 is not None:
            model = UNI2(device=self.device)
        elif model_name.lower() == "virchow2" and Virchow2 is not None:
            model = Virchow2(device=self.device)
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        
        if self.verbose:
            logger.info(f"Initialized {model_name} embedding model")
        
        return model
    
    def _initialize_stain_normalizer(self, method: str):
        """Initialize stain normalization method."""
        if method.lower() == "reinhard":
            from honeybee.preprocessing.stain_normalization import ReinhardNormalizer
            normalizer = ReinhardNormalizer()
        elif method.lower() == "macenko":
            from honeybee.preprocessing.stain_normalization import MacenkoNormalizer
            normalizer = MacenkoNormalizer()
        elif method.lower() == "vahadane":
            from honeybee.preprocessing.stain_normalization import VahadaneNormalizer
            normalizer = VahadaneNormalizer()
        else:
            raise ValueError(f"Unsupported stain normalization method: {method}")
        
        # Fit to target if available
        if self.stain_norm_target in STAIN_NORM_TARGETS:
            # Create synthetic target image or use predefined parameters
            normalizer.set_target_params(STAIN_NORM_TARGETS[self.stain_norm_target])
        
        if self.verbose:
            logger.info(f"Initialized {method} stain normalizer")
        
        return normalizer
    
    def process_slide(
        self,
        slide_path: Union[str, Path],
        output_embeddings: bool = True,
        output_patches: bool = False,
        output_coords: bool = True,
        visualize: bool = False,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single whole slide image.
        
        Args:
            slide_path: Path to the slide file
            output_embeddings: Whether to generate embeddings
            output_patches: Whether to return extracted patches
            output_coords: Whether to return patch coordinates
            visualize: Whether to visualize tissue detection
            save_path: Path to save results
            
        Returns:
            Dictionary containing:
                - embeddings: Patch embeddings (if requested)
                - patches: Extracted patches (if requested)
                - coords: Patch coordinates (if requested)
                - metadata: Slide metadata
        """
        slide_path = Path(slide_path)
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide not found: {slide_path}")
        
        if self.verbose:
            logger.info(f"Processing slide: {slide_path.name}")
        
        # Load slide
        slide = Slide(
            str(slide_path),
            tile_size=self.tile_size,
            max_patches=self.max_patches,
            tissue_detector=self.tissue_detector,
            visualize=visualize,
            verbose=self.verbose
        )
        
        # Extract patches
        patches = slide.load_patches_concurrently(target_patch_size=self.target_size)
        
        if self.verbose:
            logger.info(f"Extracted {len(patches)} patches from {slide_path.name}")
        
        # Apply stain normalization if enabled
        if self.stain_normalizer:
            patches = self._normalize_patches(patches)
        
        results = {
            "metadata": {
                "slide_path": str(slide_path),
                "slide_name": slide_path.name,
                "num_patches": len(patches),
                "tile_size": self.tile_size,
                "target_size": self.target_size,
                "embedding_model": self.embedding_model_name,
                "stain_normalization": self.stain_normalization
            }
        }
        
        # Generate embeddings if requested
        if output_embeddings:
            embeddings = self.generate_embeddings(patches)
            results["embeddings"] = embeddings
        
        # Include patches if requested
        if output_patches:
            results["patches"] = patches
        
        # Include coordinates if requested
        if output_coords:
            coords = slide.get_patch_coords()
            results["coords"] = coords
        
        # Save results if path provided
        if save_path:
            self._save_results(results, save_path)
        
        return results
    
    def _normalize_patches(self, patches: np.ndarray) -> np.ndarray:
        """Apply stain normalization to patches."""
        if self.verbose:
            logger.info("Applying stain normalization...")
        
        normalized_patches = np.zeros_like(patches)
        
        for i in tqdm(range(len(patches)), desc="Normalizing patches", disable=not self.verbose):
            normalized_patches[i] = self.stain_normalizer.transform(patches[i])
        
        return normalized_patches
    
    def generate_embeddings(
        self,
        patches: Union[np.ndarray, List[np.ndarray]],
        aggregate: str = "mean"
    ) -> np.ndarray:
        """
        Generate embeddings for patches using the configured model.
        
        Args:
            patches: Array of patches (N, H, W, C) or list of patches
            aggregate: How to aggregate patch embeddings ("mean", "max", "concat", None)
            
        Returns:
            Embeddings array
        """
        if isinstance(patches, list):
            patches = np.array(patches)
        
        # Create dataset and dataloader
        dataset = PatchDataset(patches, transform=self.embedding_model.transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Generate embeddings
        embeddings_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings", disable=not self.verbose):
                batch = batch.to(self.device)
                
                # Get embeddings from model
                if hasattr(self.embedding_model, 'generate_embeddings'):
                    emb = self.embedding_model.generate_embeddings(batch)
                else:
                    emb = self.embedding_model(batch)
                
                embeddings_list.append(emb.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings_list, axis=0)
        
        # Aggregate if requested
        if aggregate == "mean":
            embeddings = np.mean(embeddings, axis=0, keepdims=True)
        elif aggregate == "max":
            embeddings = np.max(embeddings, axis=0, keepdims=True)
        elif aggregate == "concat":
            embeddings = embeddings.reshape(1, -1)
        
        return embeddings
    
    def process_batch(
        self,
        slide_paths: List[Union[str, Path]],
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple slides in batch.
        
        Args:
            slide_paths: List of paths to slides
            output_dir: Directory to save results
            **kwargs: Additional arguments passed to process_slide
            
        Returns:
            Dictionary mapping slide names to results
        """
        results = {}
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for slide_path in tqdm(slide_paths, desc="Processing slides"):
            slide_name = Path(slide_path).name
            
            try:
                save_path = None
                if output_dir:
                    save_path = os.path.join(output_dir, f"{Path(slide_path).stem}_results.npz")
                
                result = self.process_slide(slide_path, save_path=save_path, **kwargs)
                results[slide_name] = result
                
            except Exception as e:
                logger.error(f"Error processing {slide_name}: {str(e)}")
                results[slide_name] = {"error": str(e)}
        
        return results
    
    def _save_results(self, results: Dict[str, Any], save_path: str):
        """Save processing results."""
        # Convert to saveable format
        save_dict = {}
        
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                save_dict[key] = value
            elif key == "metadata":
                # Save metadata as JSON string
                import json
                save_dict["metadata_json"] = json.dumps(value)
        
        # Save as compressed numpy archive
        np.savez_compressed(save_path, **save_dict)
        
        if self.verbose:
            logger.info(f"Saved results to {save_path}")


class PatchDataset(Dataset):
    """Dataset for patch processing."""
    
    def __init__(self, patches: np.ndarray, transform=None):
        self.patches = patches
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        # Convert to PIL Image
        if patch.dtype != np.uint8:
            patch = (patch * 255).astype(np.uint8)
        
        patch = Image.fromarray(patch)
        
        # Apply transforms
        if self.transform:
            patch = self.transform(patch)
        
        return patch


# Convenience function for quick processing
def process_pathology(
    slide_path: Union[str, Path],
    embedding_model: str = "uni",
    stain_normalization: Optional[str] = None,
    device: str = "cuda",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to process a pathology slide.
    
    Args:
        slide_path: Path to the slide
        embedding_model: Model to use for embeddings
        stain_normalization: Stain normalization method
        device: Device for computation
        **kwargs: Additional arguments for PathologyProcessor
        
    Returns:
        Processing results
    """
    processor = PathologyProcessor(
        embedding_model=embedding_model,
        stain_normalization=stain_normalization,
        device=device,
        **kwargs
    )
    
    return processor.process_slide(slide_path, output_embeddings=True)