"""
Visualization functions for WSI processing examples.

This module provides visualization utilities for displaying WSI processing results,
including patch grids, tissue masks, stain normalization comparisons, and embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional, Dict, Union
import seaborn as sns
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV not available, some features may be limited")


def setup_plot_style():
    """Set up consistent plotting style for all visualizations."""
    plt.style.use("default")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.titlesize"] = 14


def plot_wsi_overview(
    image: np.ndarray, metadata: Dict, 
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot WSI overview with metadata information.

    Args:
        image: WSI thumbnail or low-resolution image
        metadata: WSI metadata dictionary
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[2, 1])

    # Main image
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(image)
    ax_img.set_title("WSI Overview", fontsize=14, weight="bold")
    ax_img.axis("off")

    # Metadata
    ax_meta = fig.add_subplot(gs[1, :])
    ax_meta.axis("off")

    # Format metadata text
    meta_text = "Metadata:\n"
    meta_text += f"Dimensions: {metadata.get('dimensions', 'N/A')}\n"
    meta_text += f"Levels: {metadata.get('level_count', 'N/A')}\n"
    meta_text += f"Objective Power: {metadata.get('objective_power', 'N/A')}\n"
    meta_text += f"Microns per pixel: {metadata.get('mpp', 'N/A')}\n"
    meta_text += f"Vendor: {metadata.get('vendor', 'N/A')}"

    ax_meta.text(
        0.1,
        0.5,
        meta_text,
        transform=ax_meta.transAxes,
        fontsize=10,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
    )

    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_multi_resolution_pyramid(
    images: List[np.ndarray],
    resolutions: List[Tuple[int, int]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plot multi-resolution pyramid levels.

    Args:
        images: List of images at different resolutions
        resolutions: List of (width, height) for each level
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    n_levels = len(images)
    fig, axes = plt.subplots(1, n_levels, figsize=figsize)

    if n_levels == 1:
        axes = [axes]

    for i, (img, res) in enumerate(zip(images, resolutions)):
        axes[i].imshow(img)
        axes[i].set_title(f"Level {i}\n{res[0]}x{res[1]}", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Multi-Resolution Pyramid", fontsize=14, weight="bold")
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_tissue_detection_comparison(
    original: np.ndarray,
    classical_mask: np.ndarray,
    deep_mask: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Compare tissue detection methods.

    Args:
        original: Original image
        classical_mask: Classical detection mask
        deep_mask: Deep learning detection mask (optional)
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    n_plots = 3 if deep_mask is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Original image
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=12, weight="bold")
    axes[0].axis("off")

    # Classical detection
    axes[1].imshow(classical_mask, cmap="gray")
    axes[1].set_title("Classical Detection (Otsu)", fontsize=12, weight="bold")
    axes[1].axis("off")

    # Deep learning detection
    if deep_mask is not None:
        axes[2].imshow(deep_mask, cmap="gray")
        axes[2].set_title("Deep Learning Detection", fontsize=12, weight="bold")
        axes[2].axis("off")

    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_patch_grid_on_wsi(
    image: np.ndarray,
    patch_coords: List[Tuple[int, int]],
    patch_size: int,
    highlight_indices: Optional[List[int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot patch extraction grid on WSI.

    Args:
        image: WSI image
        patch_coords: List of (x, y) coordinates for patches
        patch_size: Size of each patch
        highlight_indices: Optional indices to highlight specific patches
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(image)

    # Draw all patches
    for i, (x, y) in enumerate(patch_coords):
        color = "red" if highlight_indices and i in highlight_indices else "green"
        linewidth = 3 if highlight_indices and i in highlight_indices else 1
        rect = Rectangle(
            (x, y),
            patch_size,
            patch_size,
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_title(
        f"Patch Extraction Grid ({len(patch_coords)} patches)",
        fontsize=14,
        weight="bold",
    )
    ax.axis("off")

    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_stain_normalization_comparison(
    source: np.ndarray,
    target: np.ndarray,
    normalized_images: Dict[str, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (18, 12),
) -> plt.Figure:
    """
    Compare different stain normalization methods.

    Args:
        source: Source image to normalize
        target: Target image for normalization
        normalized_images: Dictionary of method_name -> normalized_image
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    n_methods = len(normalized_images)
    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = GridSpec(3, max(3, n_methods), figure=fig)

    # Source and target in top row
    ax_source = fig.add_subplot(gs[0, 0])
    ax_source.imshow(source)
    ax_source.set_title("Source Image", fontsize=12, weight="bold")
    ax_source.axis("off")

    ax_arrow1 = fig.add_subplot(gs[0, 1])
    ax_arrow1.text(0.5, 0.5, "→", fontsize=50, ha="center", va="center")
    ax_arrow1.axis("off")

    ax_target = fig.add_subplot(gs[0, 2])
    ax_target.imshow(target)
    ax_target.set_title("Target Image", fontsize=12, weight="bold")
    ax_target.axis("off")

    # Normalized results in bottom rows
    for i, (method, img) in enumerate(normalized_images.items()):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(f"{method} Normalized", fontsize=11, weight="bold")
        ax.axis("off")

    plt.suptitle("Stain Normalization Comparison", fontsize=14, weight="bold")
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_stain_separation(
    original: np.ndarray,
    h_channel: np.ndarray,
    e_channel: np.ndarray,
    d_channel: Optional[np.ndarray] = None,
    reconstructed: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (18, 8),
) -> plt.Figure:
    """
    Visualize stain separation results.

    Args:
        original: Original RGB image
        h_channel: Hematoxylin channel
        e_channel: Eosin channel
        d_channel: Optional DAB channel
        reconstructed: Optional reconstructed image
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    n_images = 3
    if d_channel is not None:
        n_images += 1
    if reconstructed is not None:
        n_images += 1

    fig, axes = plt.subplots(2, max(3, (n_images + 1) // 2), figsize=figsize)
    axes = axes.flatten()

    # Original
    axes[0].imshow(original)
    axes[0].set_title("Original H&E", fontsize=12, weight="bold")
    axes[0].axis("off")

    # Hematoxylin
    axes[1].imshow(h_channel)
    axes[1].set_title("Hematoxylin", fontsize=12, weight="bold")
    axes[1].axis("off")

    # Eosin
    axes[2].imshow(e_channel)
    axes[2].set_title("Eosin", fontsize=12, weight="bold")
    axes[2].axis("off")

    idx = 3

    # DAB if present
    if d_channel is not None:
        axes[idx].imshow(d_channel)
        axes[idx].set_title("DAB", fontsize=12, weight="bold")
        axes[idx].axis("off")
        idx += 1

    # Reconstructed if present
    if reconstructed is not None:
        axes[idx].imshow(reconstructed)
        axes[idx].set_title("Reconstructed", fontsize=12, weight="bold")
        axes[idx].axis("off")
        idx += 1

    # Hide unused axes
    for i in range(idx, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Stain Separation Analysis", fontsize=14, weight="bold")
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_patch_samples(
    patches: List[np.ndarray],
    labels: Optional[List[str]] = None,
    n_samples: int = 16,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 12),
) -> plt.Figure:
    """
    Plot sample patches in a grid.

    Args:
        patches: List of patch images
        labels: Optional labels for each patch
        n_samples: Number of samples to display
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    n_samples = min(n_samples, len(patches))
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    for idx in range(n_samples):
        row = idx // n_cols
        col = idx % n_cols

        axes[row, col].imshow(patches[idx])
        if labels and idx < len(labels):
            axes[row, col].set_title(labels[idx], fontsize=9)
        axes[row, col].axis("off")

    # Hide empty subplots
    for idx in range(n_samples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.suptitle("Sample Patches", fontsize=14, weight="bold")
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_embedding_distributions(
    embeddings: Dict[str, np.ndarray], 
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot distribution of embeddings from different models.

    Args:
        embeddings: Dictionary of model_name -> embedding_matrix
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    n_models = len(embeddings)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, emb) in enumerate(embeddings.items()):
        # Flatten embeddings and plot distribution
        emb_flat = emb.flatten()
        axes[idx].hist(emb_flat, bins=50, alpha=0.7, edgecolor="black")
        axes[idx].set_title(f"{model_name} Distribution", fontsize=11)
        axes[idx].set_xlabel("Embedding Value")
        axes[idx].set_ylabel("Frequency")

        # Add statistics
        mean_val = np.mean(emb_flat)
        std_val = np.std(emb_flat)
        axes[idx].axvline(
            mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.3f}"
        )
        axes[idx].axvline(
            mean_val + std_val,
            color="orange",
            linestyle="--",
            label=f"Std: {std_val:.3f}",
        )
        axes[idx].axvline(mean_val - std_val, color="orange", linestyle="--")
        axes[idx].legend()

    plt.suptitle("Embedding Distributions", fontsize=14, weight="bold")
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_attention_heatmap(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot attention heatmap overlay on image.

    Args:
        image: Original image
        attention_map: Attention weights
        alpha: Transparency for overlay
        colormap: Colormap for attention
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, weight="bold")
    axes[0].axis("off")

    # Attention map
    im = axes[1].imshow(attention_map, cmap=colormap)
    axes[1].set_title("Attention Map", fontsize=12, weight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attention_map, cmap=colormap, alpha=alpha)
    axes[2].set_title("Attention Overlay", fontsize=12, weight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_processing_timeline(
    steps: List[str], times: List[float], 
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot processing timeline showing time taken for each step.

    Args:
        steps: List of processing step names
        times: List of times in seconds
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create horizontal bar chart
    y_pos = np.arange(len(steps))
    bars = ax.barh(y_pos, times)

    # Color bars based on time
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels
    for i, (step, time) in enumerate(zip(steps, times)):
        ax.text(time + 0.1, i, f"{time:.2f}s", va="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(steps)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Processing Timeline", fontsize=14, weight="bold")

    # Add total time
    total_time = sum(times)
    ax.text(
        0.95,
        0.05,
        f"Total: {total_time:.2f}s",
        transform=ax.transAxes,
        ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def create_comparison_figure(
    results: Dict[str, Dict],
    metric: str = "accuracy",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Create comparison figure for different methods.

    Args:
        results: Dictionary of method_name -> metrics_dict
        metric: Metric to compare
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure object
    """
    methods = list(results.keys())
    values = [results[method].get(metric, 0) for method in methods]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    bars = ax.bar(methods, values)

    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel(metric.capitalize())
    ax.set_title(
        f"Method Comparison: {metric.capitalize()}", fontsize=14, weight="bold"
    )
    ax.set_ylim(0, max(values) * 1.1)

    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30
) -> plt.Figure:
    """
    Plot embeddings using t-SNE visualization.
    
    Args:
        embeddings: Array of embeddings (N, D)
        labels: Optional labels for coloring
        save_path: Optional path to save the figure
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
        
    Returns:
        Matplotlib figure object
    """
    from sklearn.manifold import TSNE
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if labels is not None:
        # Color by labels
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[colors[i]], label=label, alpha=0.7, s=50)
        ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)
    
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title('Embedding Visualization (t-SNE)', fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_pyramid_levels(
    img_obj,
    max_levels: int = 4,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Plot WSI pyramid levels.
    
    Args:
        img_obj: CuCIM image object
        max_levels: Maximum number of levels to display
        save_path: Optional path to save the figure
        figsize: Figure size (auto-calculated if None)
        
    Returns:
        Matplotlib figure object
    """
    resolutions = img_obj.resolutions
    num_levels = min(resolutions['level_count'], max_levels)
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (4 * num_levels, 4)
    
    fig, axes = plt.subplots(1, num_levels, figsize=figsize)
    if num_levels == 1:
        axes = [axes]
    
    for level in range(num_levels):
        # Read thumbnail at this level
        dims = resolutions['level_dimensions'][level]
        thumbnail = img_obj.read_region(
            location=[0, 0],
            size=[min(dims[0], 1000), min(dims[1], 1000)],
            level=level
        )
        
        axes[level].imshow(np.asarray(thumbnail))
        axes[level].set_title(f"Level {level}\n{dims[0]}×{dims[1]}", fontsize=10)
        axes[level].axis('off')
    
    plt.suptitle("WSI Pyramid Levels", fontsize=14, weight='bold')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        save_path: Optional path to save the figure
        figsize: Figure size
        cmap: Colormap
        
    Returns:
        Matplotlib figure object
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    if class_names is not None:
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title("Confusion Matrix", fontsize=14, weight='bold')
    fig.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    
    return fig


def save_figure(fig: plt.Figure, filepath: Union[str, Path], dpi: int = 300) -> None:
    """
    Save figure to file.

    Args:
        fig: Matplotlib figure
        filepath: Output file path
        dpi: Resolution in dots per inch
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved to: {filepath}")


# Initialize plot style when module is imported
setup_plot_style()
