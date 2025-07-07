"""
RadImageNet Visualization Showcase

Quick demo script to generate a variety of feature map visualizations
similar to the style shown in the user's example image.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import logging

from radiology.data_management import load_dicom_series
from radiology.preprocessing import preprocess_ct
from radiology.ai_integration import RadImageNetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_feature_map_grid(processor, image, save_path='feature_maps_showcase.png'):
    """Create a grid of feature maps similar to the user's example"""
    
    # Extract features
    features = processor.extract_features(image)
    
    if not features:
        logger.warning("No features extracted")
        return
    
    # Create figure
    fig = plt.figure(figsize=(16, 16))
    
    # Get all layer names
    layer_names = list(features.keys())
    
    # Select 4 layers to visualize
    if len(layer_names) >= 4:
        selected_layers = [
            layer_names[0],  # Early layer
            layer_names[len(layer_names)//3],  # Early-middle
            layer_names[2*len(layer_names)//3],  # Late-middle
            layer_names[-1]  # Final layer
        ]
    else:
        selected_layers = layer_names[:4]
    
    # Create 2x2 grid
    for idx, layer_name in enumerate(selected_layers):
        ax = plt.subplot(2, 2, idx + 1)
        
        feature_map = features[layer_name]
        
        # Handle different shapes
        if len(feature_map.shape) == 4:
            # Average across batch and channels
            feature_avg = feature_map[0].mean(axis=0)
        elif len(feature_map.shape) == 3:
            # Average across channels
            feature_avg = feature_map.mean(axis=0)
        else:
            feature_avg = feature_map
        
        # Display with colormap
        im = ax.imshow(feature_avg, cmap='viridis', interpolation='nearest')
        
        # Add title
        short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
        ax.set_title(f'{short_name} (avg across channels)', fontsize=14)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('RadImageNet Feature Maps at Different Depths', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Feature map grid saved to: {save_path}")


def create_channel_visualization(processor, image, layer_idx=1, save_path='channels_showcase.png'):
    """Visualize individual channels from a specific layer"""
    
    # Extract features
    features = processor.extract_features(image)
    
    if not features:
        logger.warning("No features extracted")
        return
    
    # Get specific layer
    layer_names = list(features.keys())
    if layer_idx >= len(layer_names):
        layer_idx = len(layer_names) - 1
    
    layer_name = layer_names[layer_idx]
    feature_map = features[layer_name]
    
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0]  # Remove batch dimension
    
    # Select channels to display
    n_channels = feature_map.shape[0]
    n_display = min(16, n_channels)
    
    # Create figure
    fig = plt.figure(figsize=(16, 16))
    grid_size = int(np.sqrt(n_display))
    
    for i in range(n_display):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        
        # Get channel
        channel = feature_map[i]
        
        # Use different colormaps for variety
        cmaps = ['viridis', 'plasma', 'inferno', 'magma']
        cmap = cmaps[i % len(cmaps)]
        
        im = ax.imshow(channel, cmap=cmap, interpolation='nearest')
        ax.set_title(f'Channel {i}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle(f'Individual Channels from {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Channel visualization saved to: {save_path}")


def create_activation_heatmap(processor, image, save_path='activation_heatmap.png'):
    """Create activation heatmaps across multiple layers"""
    
    # Extract features
    features = processor.extract_features(image)
    
    if not features:
        logger.warning("No features extracted")
        return
    
    # Create figure
    fig = plt.figure(figsize=(20, 8))
    
    # Get all layers
    layer_names = list(features.keys())
    n_layers = min(6, len(layer_names))
    selected_indices = np.linspace(0, len(layer_names)-1, n_layers, dtype=int)
    
    for idx, layer_idx in enumerate(selected_indices):
        ax = plt.subplot(2, 3, idx + 1)
        
        layer_name = layer_names[layer_idx]
        feature_map = features[layer_name]
        
        if len(feature_map.shape) == 4:
            feature_map = feature_map[0]
        
        # Compute activation statistics
        if len(feature_map.shape) == 3:
            # Max activation across channels
            max_activation = feature_map.max(axis=0)
            mean_activation = feature_map.mean(axis=0)
        else:
            max_activation = feature_map
            mean_activation = feature_map
        
        # Create heatmap
        im = ax.imshow(mean_activation, cmap='hot', interpolation='bilinear')
        
        # Add contours for max activations
        if max_activation.shape == mean_activation.shape:
            contours = ax.contour(max_activation, levels=5, colors='cyan', alpha=0.5, linewidths=1)
        
        short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
        ax.set_title(f'Layer {layer_idx}: {short_name}', fontsize=12)
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Activation Heatmaps Across Network Depth', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Activation heatmap saved to: {save_path}")


def create_feature_evolution_plot(processor, image, save_path='feature_evolution.png'):
    """Visualize how features evolve through the network"""
    
    # Extract features
    features = processor.extract_features(image)
    
    if not features:
        logger.warning("No features extracted")
        return
    
    # Collect statistics
    stats = {
        'layer_names': [],
        'mean_activation': [],
        'std_activation': [],
        'sparsity': [],
        'max_activation': []
    }
    
    for layer_name, feature_map in features.items():
        if len(feature_map.shape) >= 2:
            stats['layer_names'].append(layer_name.split('.')[-1])
            stats['mean_activation'].append(float(np.mean(feature_map)))
            stats['std_activation'].append(float(np.std(feature_map)))
            stats['sparsity'].append(float(np.mean(feature_map == 0)))
            stats['max_activation'].append(float(np.max(feature_map)))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = range(len(stats['layer_names']))
    
    # Mean activation
    ax = axes[0, 0]
    ax.plot(x, stats['mean_activation'], 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Mean Activation Evolution')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(stats['layer_names'][::2], rotation=45, ha='right')
    
    # Standard deviation
    ax = axes[0, 1]
    ax.plot(x, stats['std_activation'], 'g-s', linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Std Activation')
    ax.set_title('Activation Variance Evolution')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(stats['layer_names'][::2], rotation=45, ha='right')
    
    # Sparsity
    ax = axes[1, 0]
    ax.plot(x, stats['sparsity'], 'r-^', linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Sparsity (fraction of zeros)')
    ax.set_title('Feature Sparsity Evolution')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(stats['layer_names'][::2], rotation=45, ha='right')
    
    # Max activation
    ax = axes[1, 1]
    ax.plot(x, stats['max_activation'], 'm-d', linewidth=2, markersize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Max Activation')
    ax.set_title('Maximum Activation Evolution')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(stats['layer_names'][::2], rotation=45, ha='right')
    
    plt.suptitle('Feature Statistics Evolution Through Network', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Feature evolution plot saved to: {save_path}")


def main():
    """Generate showcase visualizations"""
    
    logger.info("RadImageNet Visualization Showcase")
    logger.info("==================================")
    
    # Create output directory
    output_dir = Path("results_showcase")
    output_dir.mkdir(exist_ok=True)
    
    # Load sample data
    ct_dir = Path("../samples/CT")
    if not ct_dir.exists():
        logger.error(f"CT samples not found at {ct_dir}")
        logger.info("Creating synthetic data for demonstration...")
        
        # Create synthetic CT-like data
        ct_volume = np.zeros((20, 256, 256))
        for i in range(20):
            # Add circular structures
            y, x = np.ogrid[:256, :256]
            center1 = (128 + i*3, 100)
            center2 = (100, 128 + i*2)
            mask1 = (x - center1[0])**2 + (y - center1[1])**2 <= (40 + i)**2
            mask2 = (x - center2[0])**2 + (y - center2[1])**2 <= (30 + i)**2
            ct_volume[i][mask1] = 200
            ct_volume[i][mask2] = 150
            # Add noise
            ct_volume[i] += np.random.randn(256, 256) * 20
    else:
        ct_volume, metadata = load_dicom_series(ct_dir)
        logger.info(f"Loaded CT volume: {ct_volume.shape}")
    
    # Select middle slice
    middle_slice = ct_volume.shape[0] // 2
    ct_slice = preprocess_ct(ct_volume[middle_slice:middle_slice+1], window='lung')[0]
    
    # Initialize processor
    processor = RadImageNetProcessor('densenet121', pretrained=True)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # 1. Feature map grid (like the user's example)
    create_feature_map_grid(
        processor, ct_slice, 
        save_path=output_dir / 'feature_maps_grid.png'
    )
    
    # 2. Individual channels
    create_channel_visualization(
        processor, ct_slice,
        layer_idx=1,
        save_path=output_dir / 'channels_visualization.png'
    )
    
    # 3. Activation heatmaps
    create_activation_heatmap(
        processor, ct_slice,
        save_path=output_dir / 'activation_heatmaps.png'
    )
    
    # 4. Feature evolution
    create_feature_evolution_plot(
        processor, ct_slice,
        save_path=output_dir / 'feature_evolution.png'
    )
    
    logger.info(f"\nAll visualizations saved to: {output_dir}/")
    logger.info("\nShowcase complete!")
    logger.info("\nTo generate more comprehensive visualizations, run:")
    logger.info("  python example_radimagenet_visualizations.py")
    logger.info("  python example_radimagenet_gradcam.py")
    logger.info("  python example_radimagenet_feature_analysis.py")


if __name__ == "__main__":
    main()