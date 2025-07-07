"""
Utility Functions for Radiology Processing

Helper functions for visualization, I/O, metrics, and results export.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.cm as cm
from typing import Union, Tuple, Optional, Dict, List, Any
import cv2
import pandas as pd
from pathlib import Path
import json
import logging
from scipy import ndimage
import seaborn as sns

logger = logging.getLogger(__name__)


def visualize_slices(volume: np.ndarray,
                    slices: Optional[List[int]] = None,
                    masks: Optional[Dict[str, np.ndarray]] = None,
                    window: Optional[Tuple[float, float]] = None,
                    cmap: str = 'gray',
                    figsize: Tuple[int, int] = (15, 5),
                    save_path: Optional[str] = None) -> None:
    """
    Visualize multiple slices from a volume with optional masks
    
    Args:
        volume: 3D medical image volume
        slices: List of slice indices to show (if None, show evenly spaced)
        masks: Dictionary of mask names to mask arrays
        window: Window level (center, width) for display
        cmap: Colormap for image display
        figsize: Figure size
        save_path: Path to save figure
    """
    if len(volume.shape) != 3:
        raise ValueError("Volume must be 3D")
    
    # Select slices to display
    if slices is None:
        n_slices = min(5, volume.shape[0])
        slices = np.linspace(0, volume.shape[0]-1, n_slices, dtype=int)
    
    n_slices = len(slices)
    n_masks = len(masks) if masks else 0
    n_rows = 1 + n_masks
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_slices, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_slices == 1:
        axes = axes.reshape(-1, 1)
    
    # Apply windowing if specified
    if window is not None:
        center, width = window
        vmin = center - width / 2
        vmax = center + width / 2
    else:
        vmin, vmax = None, None
    
    # Plot slices
    for i, slice_idx in enumerate(slices):
        # Original image
        axes[0, i].imshow(volume[slice_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'Slice {slice_idx}')
        axes[0, i].axis('off')
        
        # Masks
        if masks:
            for j, (mask_name, mask) in enumerate(masks.items()):
                axes[j+1, i].imshow(volume[slice_idx], cmap='gray', vmin=vmin, vmax=vmax, alpha=0.7)
                
                if len(mask.shape) == 3:
                    mask_slice = mask[slice_idx]
                else:
                    mask_slice = mask
                
                # Overlay mask
                masked = np.ma.masked_where(mask_slice == 0, mask_slice)
                axes[j+1, i].imshow(masked, cmap='jet', alpha=0.5)
                
                if i == 0:
                    axes[j+1, i].set_ylabel(mask_name)
                axes[j+1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()


def save_processed_image(image: np.ndarray,
                        metadata: Dict[str, Any],
                        output_path: Union[str, Path],
                        format: str = 'nifti') -> None:
    """
    Save processed image with metadata
    
    Args:
        image: Processed image array
        metadata: Metadata dictionary
        output_path: Output file path
        format: Output format ('nifti', 'numpy', 'dicom')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'nifti':
        import nibabel as nib
        
        # Create affine matrix from metadata
        affine = np.eye(4)
        if 'pixel_spacing' in metadata:
            spacing = metadata['pixel_spacing']
            affine[0, 0] = spacing[0]
            affine[1, 1] = spacing[1]
            affine[2, 2] = spacing[2] if len(spacing) > 2 else 1.0
        
        nii = nib.Nifti1Image(image, affine)
        nib.save(nii, str(output_path.with_suffix('.nii.gz')))
        
        # Save metadata as JSON
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    elif format == 'numpy':
        # Save as numpy array
        np.save(str(output_path.with_suffix('.npy')), image)
        
        # Save metadata
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    elif format == 'dicom':
        # DICOM saving would require pydicom and proper DICOM structure
        logger.warning("DICOM export not implemented, saving as NIfTI instead")
        save_processed_image(image, metadata, output_path, format='nifti')
    
    logger.info(f"Saved processed image to {output_path}")


def create_montage(images: List[np.ndarray],
                  grid_shape: Optional[Tuple[int, int]] = None,
                  spacing: int = 5,
                  background: float = 0) -> np.ndarray:
    """
    Create a montage of multiple 2D images
    
    Args:
        images: List of 2D images
        grid_shape: (rows, cols) for montage layout
        spacing: Spacing between images
        background: Background value
    """
    n_images = len(images)
    
    # Determine grid shape
    if grid_shape is None:
        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))
    else:
        n_rows, n_cols = grid_shape
    
    # Get max dimensions
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    
    # Create montage
    montage_height = n_rows * max_height + (n_rows - 1) * spacing
    montage_width = n_cols * max_width + (n_cols - 1) * spacing
    
    montage = np.full((montage_height, montage_width), background, dtype=images[0].dtype)
    
    # Place images
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        
        y_start = row * (max_height + spacing)
        x_start = col * (max_width + spacing)
        
        # Center image in cell
        y_offset = (max_height - img.shape[0]) // 2
        x_offset = (max_width - img.shape[1]) // 2
        
        montage[y_start + y_offset:y_start + y_offset + img.shape[0],
                x_start + x_offset:x_start + x_offset + img.shape[1]] = img
    
    return montage


def calculate_metrics(prediction: np.ndarray,
                     ground_truth: np.ndarray,
                     metrics: List[str] = ['dice', 'jaccard', 'sensitivity', 'specificity']) -> Dict[str, float]:
    """
    Calculate segmentation metrics
    
    Args:
        prediction: Binary prediction mask
        ground_truth: Binary ground truth mask
        metrics: List of metrics to calculate
    """
    results = {}
    
    # Ensure binary masks
    pred_binary = prediction > 0
    gt_binary = ground_truth > 0
    
    # Calculate basic components
    tp = np.sum(pred_binary & gt_binary)
    fp = np.sum(pred_binary & ~gt_binary)
    fn = np.sum(~pred_binary & gt_binary)
    tn = np.sum(~pred_binary & ~gt_binary)
    
    # Calculate metrics
    if 'dice' in metrics:
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        results['dice'] = float(dice)
    
    if 'jaccard' in metrics:
        jaccard = tp / (tp + fp + fn + 1e-8)
        results['jaccard'] = float(jaccard)
    
    if 'sensitivity' in metrics:
        sensitivity = tp / (tp + fn + 1e-8)
        results['sensitivity'] = float(sensitivity)
    
    if 'specificity' in metrics:
        specificity = tn / (tn + fp + 1e-8)
        results['specificity'] = float(specificity)
    
    if 'precision' in metrics:
        precision = tp / (tp + fp + 1e-8)
        results['precision'] = float(precision)
    
    if 'accuracy' in metrics:
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        results['accuracy'] = float(accuracy)
    
    if 'hausdorff' in metrics:
        # Hausdorff distance
        from scipy.ndimage import distance_transform_edt
        
        if np.any(pred_binary) and np.any(gt_binary):
            pred_dist = distance_transform_edt(~pred_binary)
            gt_dist = distance_transform_edt(~gt_binary)
            
            hausdorff = max(np.max(pred_dist[gt_binary]), 
                          np.max(gt_dist[pred_binary]))
            results['hausdorff'] = float(hausdorff)
        else:
            results['hausdorff'] = float('inf')
    
    return results


def export_results(results: Dict[str, Any],
                  output_path: Union[str, Path],
                  format: str = 'json') -> None:
    """
    Export processing results
    
    Args:
        results: Dictionary of results
        output_path: Output file path
        format: Export format ('json', 'csv', 'excel')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format == 'csv':
        # Flatten nested dictionaries
        flat_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_results[f"{key}_{sub_key}"] = sub_value
            else:
                flat_results[key] = value
        
        df = pd.DataFrame([flat_results])
        df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    elif format == 'excel':
        # Create Excel with multiple sheets for different result types
        with pd.ExcelWriter(output_path.with_suffix('.xlsx')) as writer:
            for key, value in results.items():
                if isinstance(value, dict):
                    df = pd.DataFrame([value])
                    df.to_excel(writer, sheet_name=key, index=False)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    df = pd.DataFrame(value)
                    df.to_excel(writer, sheet_name=key, index=False)
    
    logger.info(f"Exported results to {output_path}")


def plot_intensity_histogram(image: np.ndarray,
                           mask: Optional[np.ndarray] = None,
                           bins: int = 256,
                           title: str = "Intensity Histogram",
                           save_path: Optional[str] = None) -> None:
    """Plot intensity histogram of image"""
    plt.figure(figsize=(10, 6))
    
    if mask is not None:
        values = image[mask > 0]
        label = "Masked region"
    else:
        values = image.flatten()
        label = "Whole image"
    
    plt.hist(values, bins=bins, density=True, alpha=0.7, label=label)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


def visualize_3d_surface(mask: np.ndarray,
                        spacing: Tuple[float, float, float] = (1, 1, 1),
                        color: str = 'cyan',
                        opacity: float = 0.5,
                        save_path: Optional[str] = None) -> None:
    """
    Visualize 3D surface rendering of a mask
    
    Note: Requires plotly for 3D visualization
    """
    try:
        import plotly.graph_objects as go
        from skimage import measure
        
        # Generate surface mesh
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
        
        # Create mesh
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=opacity,
                name='Surface'
            )
        ])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title='3D Surface Rendering'
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved 3D visualization to {save_path}")
        else:
            fig.show()
            
    except ImportError:
        logger.error("Plotly not installed. Install with: pip install plotly")


def create_overlay(image: np.ndarray,
                  mask: np.ndarray,
                  alpha: float = 0.5,
                  color_map: str = 'jet') -> np.ndarray:
    """
    Create overlay of mask on image
    
    Args:
        image: Base image (2D or 3D)
        mask: Binary or label mask
        alpha: Transparency of overlay
        color_map: Colormap for mask
    """
    # Normalize image to 0-1
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Convert to RGB
    if len(img_norm.shape) == 2:
        img_rgb = np.stack([img_norm] * 3, axis=-1)
    else:
        # For 3D, process middle slice
        middle = img_norm.shape[0] // 2
        img_slice = img_norm[middle]
        img_rgb = np.stack([img_slice] * 3, axis=-1)
        
        if len(mask.shape) == 3:
            mask = mask[middle]
    
    # Create colored mask
    if mask.max() > 1:  # Label mask
        cmap = cm.get_cmap(color_map)
        mask_colored = cmap(mask / mask.max())[:, :, :3]
    else:  # Binary mask
        mask_colored = np.zeros((*mask.shape, 3))
        mask_colored[mask > 0] = [1, 0, 0]  # Red
    
    # Blend
    overlay = (1 - alpha) * img_rgb + alpha * mask_colored * (mask > 0)[:, :, np.newaxis]
    
    return np.clip(overlay, 0, 1)


def plot_segmentation_comparison(original: np.ndarray,
                               predictions: Dict[str, np.ndarray],
                               ground_truth: Optional[np.ndarray] = None,
                               slice_idx: Optional[int] = None,
                               figsize: Tuple[int, int] = (15, 5)) -> None:
    """Compare multiple segmentation results"""
    n_methods = len(predictions) + (2 if ground_truth is not None else 1)
    
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]
    
    # Handle 3D volumes
    if len(original.shape) == 3:
        if slice_idx is None:
            slice_idx = original.shape[0] // 2
        original = original[slice_idx]
        predictions = {k: v[slice_idx] if len(v.shape) == 3 else v 
                      for k, v in predictions.items()}
        if ground_truth is not None and len(ground_truth.shape) == 3:
            ground_truth = ground_truth[slice_idx]
    
    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Predictions
    idx = 1
    for method_name, pred_mask in predictions.items():
        overlay = create_overlay(original, pred_mask)
        axes[idx].imshow(overlay)
        axes[idx].set_title(method_name)
        axes[idx].axis('off')
        idx += 1
    
    # Ground truth
    if ground_truth is not None:
        overlay = create_overlay(original, ground_truth, color_map='spring')
        axes[idx].imshow(overlay)
        axes[idx].set_title('Ground Truth')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def annotate_nodules(image: np.ndarray,
                    nodules: List[Dict],
                    slice_idx: Optional[int] = None,
                    save_path: Optional[str] = None) -> None:
    """Annotate detected nodules on image"""
    # Handle 3D
    if len(image.shape) == 3:
        if slice_idx is None:
            slice_idx = image.shape[0] // 2
        image = image[slice_idx]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    
    # Draw nodules
    for i, nodule in enumerate(nodules):
        pos = nodule['position']
        
        # Check if nodule is on this slice (for 3D)
        if len(pos) == 3:
            if abs(pos[0] - slice_idx) > nodule['radius']:
                continue
            y, x = pos[1], pos[2]
        else:
            y, x = pos[0], pos[1]
        
        # Draw circle
        circle = Circle((x, y), nodule['radius'], 
                       fill=False, color='red', linewidth=2)
        ax.add_patch(circle)
        
        # Add label
        ax.text(x + nodule['radius'], y, f"N{i+1}\n{nodule['diameter']:.1f}mm",
                color='yellow', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    ax.set_title(f'Detected Nodules (Slice {slice_idx})')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    # Create sample data
    sample_volume = np.random.randn(10, 256, 256)
    sample_mask = np.zeros_like(sample_volume)
    sample_mask[4:7, 100:150, 100:150] = 1
    
    # Visualize slices
    visualize_slices(sample_volume, masks={'Segmentation': sample_mask})
    
    # Create montage
    slices = [sample_volume[i] for i in range(0, 10, 2)]
    montage = create_montage(slices)
    plt.figure(figsize=(10, 10))
    plt.imshow(montage, cmap='gray')
    plt.title('Slice Montage')
    plt.axis('off')
    plt.show()