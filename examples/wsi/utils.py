"""
Utility functions for WSI processing examples.

This module provides common utility functions used across all WSI processing examples,
including timing decorators, configuration helpers, and file I/O utilities.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from functools import wraps
import torch
import logging
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV not available, some features may be limited")


def create_synthetic_wsi_patch(
    size: Tuple[int, int] = (512, 512),
    tissue_type: str = "normal",
    add_noise: bool = True
) -> np.ndarray:
    """
    Create a synthetic WSI patch for testing purposes.
    
    Args:
        size: Patch size (height, width)
        tissue_type: Type of tissue to simulate ("normal", "tumor", "stroma")
        add_noise: Whether to add realistic noise
        
    Returns:
        RGB image as numpy array
    """
    h, w = size
    
    # Base colors for different tissue types (RGB)
    tissue_colors = {
        "normal": {
            "nuclei": np.array([100, 50, 150]),      # Purple nuclei
            "cytoplasm": np.array([240, 220, 230]),  # Pink cytoplasm
            "background": np.array([250, 240, 245])   # Light background
        },
        "tumor": {
            "nuclei": np.array([80, 40, 120]),       # Darker purple
            "cytoplasm": np.array([220, 180, 200]),  # Darker pink
            "background": np.array([240, 220, 230])
        },
        "stroma": {
            "nuclei": np.array([120, 80, 140]),
            "cytoplasm": np.array([250, 200, 220]),  # More eosinophilic
            "background": np.array([255, 245, 250])
        }
    }
    
    colors = tissue_colors.get(tissue_type, tissue_colors["normal"])
    
    # Create base image
    img = np.ones((h, w, 3), dtype=np.uint8) * colors["background"]
    
    # Add nuclei
    num_nuclei = np.random.randint(20, 50)
    for _ in range(num_nuclei):
        center = (np.random.randint(20, w-20), np.random.randint(20, h-20))
        radius = np.random.randint(8, 15)
        color = colors["nuclei"] + np.random.randint(-20, 20, 3)
        color = np.clip(color, 0, 255).astype(np.uint8)
        
        if cv2 is not None:
            cv2.circle(img, center, radius, color.tolist(), -1)
        else:
            # Simple fallback without cv2
            y, x = np.ogrid[:h, :w]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            img[mask] = color
    
    # Add cytoplasm regions
    num_cells = np.random.randint(10, 20)
    for _ in range(num_cells):
        center = (np.random.randint(30, w-30), np.random.randint(30, h-30))
        axes = (np.random.randint(20, 40), np.random.randint(20, 40))
        angle = np.random.randint(0, 180)
        color = colors["cytoplasm"] + np.random.randint(-20, 20, 3)
        color = np.clip(color, 0, 255).astype(np.uint8)
        
        if cv2 is not None:
            cv2.ellipse(img, center, axes, angle, 0, 360, color.tolist(), -1)
        else:
            # Simple ellipse approximation without cv2
            y, x = np.ogrid[:h, :w]
            # Simplified circular shape as fallback
            mask = (x - center[0])**2 / axes[0]**2 + (y - center[1])**2 / axes[1]**2 <= 1
            img[mask] = color
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, 5, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return img


def visualize_patches(
    patches: List[np.ndarray],
    titles: Optional[List[str]] = None,
    max_display: int = 9,
    figsize: Tuple[int, int] = (12, 12)
) -> None:
    """
    Visualize a grid of patches.
    
    Args:
        patches: List of image patches
        titles: Optional titles for each patch
        max_display: Maximum number of patches to display
        figsize: Figure size
    """
    n_patches = min(len(patches), max_display)
    n_cols = int(np.ceil(np.sqrt(n_patches)))
    n_rows = int(np.ceil(n_patches / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(n_patches):
        axes[i].imshow(patches[i])
        axes[i].axis('off')
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
    
    # Hide unused subplots
    for i in range(n_patches, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_tissue_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    mask_color: Tuple[int, int, int] = (0, 255, 0)
) -> None:
    """
    Plot an image with tissue mask overlay.
    
    Args:
        image: Original image
        mask: Binary tissue mask
        alpha: Transparency of overlay
        mask_color: Color for mask overlay (RGB)
    """
    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[mask > 0] = mask_color
    
    # Blend with original image
    result = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Tissue Mask')
    axes[1].axis('off')
    
    axes[2].imshow(result)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_stain_normalizations(
    original: np.ndarray,
    normalized_images: dict,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Compare different stain normalization results.
    
    Args:
        original: Original image
        normalized_images: Dictionary of method_name -> normalized_image
        figsize: Figure size
    """
    n_images = len(normalized_images) + 1
    n_cols = min(3, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Show original
    axes[0].imshow(original)
    axes[0].set_title('Original', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    # Show normalized versions
    for i, (method, img) in enumerate(normalized_images.items(), 1):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(method, fontsize=14, weight='bold')
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_stain_separation(
    original: np.ndarray,
    h_channel: np.ndarray,
    e_channel: np.ndarray,
    d_channel: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Visualize stain separation results.
    
    Args:
        original: Original RGB image
        h_channel: Hematoxylin channel
        e_channel: Eosin channel
        d_channel: Optional DAB channel
        figsize: Figure size
    """
    n_images = 4 if d_channel is not None else 3
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    axes[0].imshow(original)
    axes[0].set_title('Original', fontsize=14, weight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(h_channel)
    axes[1].set_title('Hematoxylin', fontsize=14, weight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(e_channel)
    axes[2].set_title('Eosin', fontsize=14, weight='bold')
    axes[2].axis('off')
    
    if d_channel is not None:
        axes[3].imshow(d_channel)
        axes[3].set_title('DAB', fontsize=14, weight='bold')
        axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "t-SNE of Patch Embeddings",
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot t-SNE visualization of embeddings.
    
    Args:
        embeddings: Embedding matrix (n_samples, n_features)
        labels: Optional labels for coloring
        title: Plot title
        figsize: Figure size
    """
    from sklearn.manifold import TSNE
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    
    if labels is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.show()


def save_results(
    results: dict,
    output_dir: Union[str, Path],
    prefix: str = "results"
) -> None:
    """
    Save processing results to disk.
    
    Args:
        results: Dictionary of results to save
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np_results = {k: v for k, v in results.items() if isinstance(v, np.ndarray)}
    if np_results:
        np.savez_compressed(output_dir / f"{prefix}_arrays.npz", **np_results)
    
    # Save images
    for key, value in results.items():
        if key.endswith('_image') and isinstance(value, np.ndarray):
            cv2.imwrite(str(output_dir / f"{prefix}_{key}.png"), 
                       cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
    
    print(f"Results saved to {output_dir}")


def benchmark_processing_time(func):
    """Decorator to benchmark function execution time."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


def timer_decorator(func):
    """
    Decorator to measure execution time of functions.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds")
        return result
    return wrapper


def get_sample_wsi_path() -> str:
    """
    Get the path to the sample WSI file.
    
    Returns:
        Path to sample.svs
    """
    return "/mnt/f/Projects/HoneyBee/examples/samples/sample.svs"


def get_tissue_detector_path() -> str:
    """
    Get the path to the tissue detector model.
    
    Returns:
        Path to HnE.pt model
    """
    return "/mnt/d/Models/TissueDetector/HnE.pt"


def create_output_dir(base_dir: str, example_name: str) -> str:
    """
    Create output directory for example results.
    
    Args:
        base_dir: Base directory for outputs
        example_name: Name of the example
        
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{example_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory path
    """
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device instance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")
    return device


def print_memory_usage() -> None:
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def save_numpy_compressed(array: np.ndarray, filepath: str) -> None:
    """
    Save numpy array in compressed format.
    
    Args:
        array: Numpy array to save
        filepath: Output file path (without extension)
    """
    np.savez_compressed(filepath, data=array)
    print(f"Array saved to: {filepath}.npz")


def load_numpy_compressed(filepath: str) -> np.ndarray:
    """
    Load numpy array from compressed format.
    
    Args:
        filepath: Input file path (with or without .npz extension)
        
    Returns:
        Loaded numpy array
    """
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    data = np.load(filepath)
    return data['data']


def calculate_patch_grid(slide_width: int, slide_height: int, 
                        patch_size: int, overlap: int = 0) -> List[Tuple[int, int]]:
    """
    Calculate grid positions for patch extraction.
    
    Args:
        slide_width: Width of the slide
        slide_height: Height of the slide
        patch_size: Size of each patch
        overlap: Overlap between patches
        
    Returns:
        List of (x, y) coordinates for patch extraction
    """
    stride = patch_size - overlap
    positions = []
    
    for y in range(0, slide_height - patch_size + 1, stride):
        for x in range(0, slide_width - patch_size + 1, stride):
            positions.append((x, y))
    
    return positions


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def print_example_header(title: str, description: str) -> None:
    """
    Print formatted header for examples.
    
    Args:
        title: Example title
        description: Example description
    """
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)
    print(f"{description}")
    print("-" * 80)
    print()


def print_section_header(section: str) -> None:
    """
    Print formatted section header.
    
    Args:
        section: Section name
    """
    print()
    print(f"--- {section} ---")
    print()


def create_results_summary(results: Dict[str, Any], output_dir: str) -> None:
    """
    Create a summary of results in text format.
    
    Args:
        results: Dictionary of results
        output_dir: Output directory
    """
    summary_path = os.path.join(output_dir, "results_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}:\n")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"  {value}\n")
            f.write("\n")
    
    print(f"Results summary saved to: {summary_path}")


def validate_paths(*paths: str) -> bool:
    """
    Validate that all provided paths exist.
    
    Args:
        *paths: Variable number of paths to check
        
    Returns:
        True if all paths exist, False otherwise
    """
    all_exist = True
    for path in paths:
        if not os.path.exists(path):
            print(f"WARNING: Path does not exist: {path}")
            all_exist = False
        else:
            print(f"Found: {path}")
    return all_exist


class ProgressTracker:
    """Simple progress tracker for batch operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            avg_time = elapsed / self.current
            remaining = avg_time * (self.total - self.current)
            
            print(f"\r{self.description}: {self.current}/{self.total} "
                  f"({self.current/self.total*100:.1f}%) "
                  f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s", 
                  end='', flush=True)
    
    def finish(self) -> None:
        """Mark progress as complete."""
        elapsed = time.time() - self.start_time
        print(f"\n{self.description} completed in {elapsed:.1f} seconds")


def get_sample_wsi_path() -> str:
    """
    Get the path to a sample WSI for testing.
    
    Returns:
        Path to sample WSI file
    """
    # Use the sample WSI file in the examples directory
    sample_path = "/mnt/f/Projects/HoneyBee/examples/samples/sample.svs"
    if not os.path.exists(sample_path):
        print(f"WARNING: Sample WSI not found at {sample_path}")
        print("Please update get_sample_wsi_path() in utils.py with a valid WSI path")
    return sample_path


def get_tissue_detector_path() -> str:
    """
    Get the path to the tissue detector model.
    
    Returns:
        Path to tissue detector model file
    """
    # Use the path from the reference code
    model_path = "/mnt/f/Projects/Multimodal-Transformer/models/deep-tissue-detector_densenet_state-dict.pt"
    if not os.path.exists(model_path):
        print(f"WARNING: Tissue detector model not found at {model_path}")
        print("Please update get_tissue_detector_path() in utils.py with a valid model path")
    return model_path


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for WSI processing.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "wsi_path": get_sample_wsi_path(),
        "tissue_detector_path": get_tissue_detector_path(),
        "tile_size": 512,
        "target_size": 224,
        "max_patches": 100,
        "batch_size": 32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "output_dir": "/mnt/f/Projects/HoneyBee/examples/wsi/tmp",
        "visualize": True,
        "save_results": True
    }