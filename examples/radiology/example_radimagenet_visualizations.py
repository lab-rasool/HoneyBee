"""
Enhanced RadImageNet Visualizations

Creates comprehensive visualizations of RadImageNet model features, activations,
and processing stages for deeper insights into medical image analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import cv2
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Import radiology modules
from radiology.data_management import load_dicom_series
from radiology.preprocessing import preprocess_ct, WindowLevelAdjuster
from radiology.ai_integration import (
    RadImageNetProcessor, 
    create_embedding_model,
    process_2d_slices
)
from radiology.utils import visualize_slices

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureVisualizer:
    """Enhanced feature visualization for RadImageNet models"""
    
    def __init__(self, model_name='densenet121'):
        self.processor = RadImageNetProcessor(model_name=model_name, pretrained=True)
        self.model = self.processor.model  # Access PyTorch model
        self.device = self.processor.device
        self.activations = {}
        self.gradients = {}
        self.hook_handles = []
        
    def register_hooks(self, layers=None):
        """Register forward and backward hooks for visualization"""
        if layers is None:
            # Default to key layers for DenseNet
            if 'densenet' in self.processor.model_name:
                layers = ['features.denseblock1', 'features.denseblock2', 
                         'features.denseblock3', 'features.denseblock4']
            else:
                layers = []
        
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks
        for name, module in self.model.named_modules():
            if any(layer in name for layer in layers):
                handle_fwd = module.register_forward_hook(get_activation(name))
                handle_bwd = module.register_backward_hook(get_gradient(name))
                self.hook_handles.extend([handle_fwd, handle_bwd])
                
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
    def get_feature_maps(self, image):
        """Extract feature maps from all layers"""
        self.activations = {}
        
        # Prepare input
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            image_tensor = image
            
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image_tensor)
            
        return self.activations
    
    def visualize_feature_progression(self, image, save_path='feature_progression.png'):
        """Visualize how features evolve through network depth"""
        features = self.get_feature_maps(image)
        
        # Sort by depth
        sorted_layers = sorted(features.keys(), key=lambda x: len(x.split('.')))
        
        # Create visualization
        n_layers = min(8, len(sorted_layers))  # Limit to 8 layers
        selected_layers = np.linspace(0, len(sorted_layers)-1, n_layers, dtype=int)
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, n_layers, height_ratios=[1, 1, 0.5])
        
        for idx, layer_idx in enumerate(selected_layers):
            layer_name = sorted_layers[layer_idx]
            feature_map = features[layer_name]
            
            if len(feature_map.shape) == 4:  # B x C x H x W
                feature_map = feature_map[0]  # Remove batch dimension
            
            # Average feature map
            ax1 = plt.subplot(gs[0, idx])
            avg_feature = feature_map.mean(dim=0).cpu().numpy()
            im1 = ax1.imshow(avg_feature, cmap='viridis')
            ax1.set_title(f'Layer {layer_idx+1}\n(Average)', fontsize=10)
            ax1.axis('off')
            
            # Maximum activation
            ax2 = plt.subplot(gs[1, idx])
            max_feature = feature_map.max(dim=0)[0].cpu().numpy()
            im2 = ax2.imshow(max_feature, cmap='hot')
            ax2.set_title('Max Activation', fontsize=10)
            ax2.axis('off')
            
            # Channel statistics
            ax3 = plt.subplot(gs[2, idx])
            channel_means = feature_map.mean(dim=(1,2)).cpu().numpy()
            ax3.hist(channel_means, bins=20, alpha=0.7, color='blue')
            ax3.set_xlabel('Mean Activation', fontsize=8)
            ax3.set_ylabel('Channels', fontsize=8)
            ax3.tick_params(labelsize=6)
            
        plt.suptitle('Feature Evolution Through Network Depth', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature progression saved to: {save_path}")
        
    def create_activation_grid(self, feature_maps, layer_name, n_features=64, 
                              save_path='activation_grid.png'):
        """Create a grid visualization of individual feature maps"""
        if layer_name not in feature_maps:
            logger.error(f"Layer {layer_name} not found")
            return
            
        features = feature_maps[layer_name]
        if len(features.shape) == 4:
            features = features[0]  # Remove batch dimension
            
        n_channels = features.shape[0]
        n_show = min(n_features, n_channels)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_show)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        axes = axes.ravel()
        
        # Normalize features for better visualization
        features_np = features.cpu().numpy()
        
        for idx in range(n_show):
            ax = axes[idx]
            feature = features_np[idx]
            
            # Apply different colormaps for variety
            cmap_options = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
            cmap = cmap_options[idx % len(cmap_options)]
            
            im = ax.imshow(feature, cmap=cmap)
            ax.set_title(f'Channel {idx}', fontsize=8)
            ax.axis('off')
            
            # Add statistics as text
            stats_text = f'μ={feature.mean():.2f}\nσ={feature.std():.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=6, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for idx in range(n_show, len(axes)):
            axes[idx].axis('off')
            
        plt.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Activation grid saved to: {save_path}")
        
    def visualize_receptive_fields(self, image, save_path='receptive_fields.png'):
        """Visualize effective receptive fields at different layers"""
        features = self.get_feature_maps(image)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Original image
        ax = axes[0]
        if isinstance(image, torch.Tensor):
            img_np = image.squeeze().cpu().numpy()
        else:
            img_np = image
        ax.imshow(img_np, cmap='gray')
        ax.set_title('Original Image')
        ax.axis('off')
        
        # Select layers at different depths
        layer_names = list(features.keys())
        selected_indices = np.linspace(0, len(layer_names)-1, 5, dtype=int)
        
        for idx, layer_idx in enumerate(selected_indices):
            ax = axes[idx + 1]
            layer_name = layer_names[layer_idx]
            feature_map = features[layer_name]
            
            if len(feature_map.shape) == 4:
                feature_map = feature_map[0]
            
            # Find maximum activation location
            avg_feature = feature_map.mean(dim=0)
            max_loc = torch.where(avg_feature == avg_feature.max())
            y, x = max_loc[0][0].item(), max_loc[1][0].item()
            
            # Estimate receptive field size (simplified)
            rf_size = 2 ** (layer_idx // 4 + 1) * 7  # Rough estimate
            
            # Overlay on original image
            ax.imshow(img_np, cmap='gray', alpha=0.7)
            
            # Scale coordinates
            scale_y = img_np.shape[0] / avg_feature.shape[0]
            scale_x = img_np.shape[1] / avg_feature.shape[1]
            
            # Draw receptive field
            rect = Rectangle((x * scale_x - rf_size//2, y * scale_y - rf_size//2),
                           rf_size, rf_size, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            
            ax.set_title(f'Layer {layer_idx}: RF ≈ {rf_size}×{rf_size}')
            ax.axis('off')
            
        plt.suptitle('Receptive Field Visualization', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Receptive fields saved to: {save_path}")
        
    def create_channel_correlation_matrix(self, feature_maps, layer_name,
                                        save_path='channel_correlation.png'):
        """Visualize correlation between different channels"""
        if layer_name not in feature_maps:
            return
            
        features = feature_maps[layer_name]
        if len(features.shape) == 4:
            features = features[0]  # Remove batch dimension
            
        # Flatten spatial dimensions
        n_channels = features.shape[0]
        features_flat = features.view(n_channels, -1)
        
        # Compute correlation matrix
        correlation = torch.corrcoef(features_flat).cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Use mask for upper triangle
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        
        sns.heatmap(correlation, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title(f'Channel Correlation Matrix - {layer_name}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Channel correlation saved to: {save_path}")
        
    def visualize_activation_distributions(self, image, save_path='activation_dist.png'):
        """Visualize activation value distributions across layers"""
        features = self.get_feature_maps(image)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        # 1. Activation magnitude by layer
        ax = axes[0]
        layer_names = []
        mean_activations = []
        std_activations = []
        
        for name, feat in features.items():
            layer_names.append(name.split('.')[-1])
            mean_act = feat.mean().item()
            std_act = feat.std().item()
            mean_activations.append(mean_act)
            std_activations.append(std_act)
            
        x_pos = np.arange(len(layer_names))
        ax.bar(x_pos, mean_activations, yerr=std_activations, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Activation')
        ax.set_title('Activation Magnitude by Layer')
        
        # 2. Sparsity analysis
        ax = axes[1]
        sparsity_levels = []
        for feat in features.values():
            sparsity = (feat == 0).float().mean().item()
            sparsity_levels.append(sparsity)
            
        ax.plot(sparsity_levels, 'o-', color='red')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Sparsity (% zeros)')
        ax.set_title('Activation Sparsity Through Network')
        ax.grid(True, alpha=0.3)
        
        # 3. Distribution violin plot
        ax = axes[2]
        # Sample from each layer
        data_for_violin = []
        labels_for_violin = []
        
        for idx, (name, feat) in enumerate(list(features.items())[:8]):
            # Sample random activations
            flat_feat = feat.flatten()
            n_samples = min(1000, flat_feat.shape[0])
            samples = flat_feat[torch.randperm(flat_feat.shape[0])[:n_samples]].cpu().numpy()
            data_for_violin.extend(samples)
            labels_for_violin.extend([f'L{idx}'] * n_samples)
            
        # Create violin plot
        import pandas as pd
        df = pd.DataFrame({'Activation': data_for_violin, 'Layer': labels_for_violin})
        
        positions = range(len(df['Layer'].unique()))
        parts = ax.violinplot([df[df['Layer'] == f'L{i}']['Activation'].values 
                              for i in range(len(positions))],
                             positions=positions, showmeans=True)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Activation Value')
        ax.set_title('Activation Value Distributions')
        
        # 4. Activation patterns heatmap
        ax = axes[3]
        # Create pattern matrix (simplified)
        pattern_matrix = np.zeros((len(features), 10))
        
        for idx, feat in enumerate(features.values()):
            # Get statistics for pattern
            pattern_matrix[idx, 0] = feat.mean().item()
            pattern_matrix[idx, 1] = feat.std().item()
            pattern_matrix[idx, 2] = feat.max().item()
            pattern_matrix[idx, 3] = feat.min().item()
            pattern_matrix[idx, 4] = (feat > 0).float().mean().item()
            pattern_matrix[idx, 5] = (feat < 0).float().mean().item()
            pattern_matrix[idx, 6] = feat.abs().mean().item()
            pattern_matrix[idx, 7] = torch.quantile(feat.flatten(), 0.25).item()
            pattern_matrix[idx, 8] = torch.quantile(feat.flatten(), 0.50).item()
            pattern_matrix[idx, 9] = torch.quantile(feat.flatten(), 0.75).item()
            
        stats_labels = ['Mean', 'Std', 'Max', 'Min', '% Pos', '% Neg', 
                       'Abs Mean', 'Q1', 'Q2', 'Q3']
        
        im = ax.imshow(pattern_matrix.T, aspect='auto', cmap='RdBu_r')
        ax.set_yticks(range(len(stats_labels)))
        ax.set_yticklabels(stats_labels)
        ax.set_xlabel('Layer Index')
        ax.set_title('Activation Statistics Heatmap')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle('Comprehensive Activation Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Activation distributions saved to: {save_path}")


def create_3d_feature_visualization(feature_map, save_path='feature_3d.png'):
    """Create 3D visualization of feature maps"""
    if len(feature_map.shape) == 4:
        feature_map = feature_map[0]  # Remove batch dimension
        
    # Select top N channels by variance
    variances = feature_map.var(dim=(1, 2))
    top_channels = variances.argsort(descending=True)[:3]
    
    fig = plt.figure(figsize=(15, 5))
    
    for idx, channel_idx in enumerate(top_channels):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        feature = feature_map[channel_idx].cpu().numpy()
        
        # Create mesh
        x = np.arange(feature.shape[1])
        y = np.arange(feature.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, feature, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.8)
        
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_zlabel('Activation')
        ax.set_title(f'Channel {channel_idx.item()}')
        
        # Add contour projections
        ax.contour(X, Y, feature, zdir='z', offset=feature.min(), 
                  cmap='viridis', alpha=0.5)
        
    plt.suptitle('3D Feature Map Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"3D visualization saved to: {save_path}")


def create_feature_similarity_matrix(features1, features2, save_path='similarity.png'):
    """Visualize similarity between features from different images/layers"""
    # Flatten features
    if len(features1.shape) > 2:
        features1 = features1.view(features1.shape[0], -1)
    if len(features2.shape) > 2:
        features2 = features2.view(features2.shape[0], -1)
        
    # Compute cosine similarity
    features1_norm = F.normalize(features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)
    
    similarity = torch.mm(features1_norm, features2_norm.t()).cpu().numpy()
    
    # Visualize
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Cosine Similarity'})
    plt.xlabel('Features 2')
    plt.ylabel('Features 1')
    plt.title('Feature Similarity Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Similarity matrix saved to: {save_path}")


def create_comprehensive_visualization_report(ct_volume):
    """Generate comprehensive visualization report"""
    logger.info("Generating Comprehensive Visualization Report")
    logger.info("=" * 50)
    
    # Create output directory
    output_dir = Path("results_radimagenet_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    visualizer = FeatureVisualizer('densenet121')
    visualizer.register_hooks()
    
    # Select representative slices
    middle_slice = ct_volume.shape[0] // 2
    slices = [middle_slice - 10, middle_slice, middle_slice + 10]
    
    for slice_idx in slices:
        logger.info(f"\nProcessing slice {slice_idx}...")
        
        # Preprocess slice
        ct_slice = preprocess_ct(ct_volume[slice_idx:slice_idx+1], window='lung')[0]
        
        # Get feature maps
        feature_maps = visualizer.get_feature_maps(ct_slice)
        
        # 1. Feature progression
        visualizer.visualize_feature_progression(
            ct_slice, 
            save_path=output_dir / f'feature_progression_slice{slice_idx}.png'
        )
        
        # 2. Activation grids for different layers
        for layer_name in list(feature_maps.keys())[:4]:
            visualizer.create_activation_grid(
                feature_maps, layer_name,
                save_path=output_dir / f'activation_grid_{layer_name}_slice{slice_idx}.png'
            )
        
        # 3. Receptive fields
        visualizer.visualize_receptive_fields(
            ct_slice,
            save_path=output_dir / f'receptive_fields_slice{slice_idx}.png'
        )
        
        # 4. Channel correlations
        middle_layer = list(feature_maps.keys())[len(feature_maps)//2]
        visualizer.create_channel_correlation_matrix(
            feature_maps, middle_layer,
            save_path=output_dir / f'channel_correlation_slice{slice_idx}.png'
        )
        
        # 5. Activation distributions
        visualizer.visualize_activation_distributions(
            ct_slice,
            save_path=output_dir / f'activation_distributions_slice{slice_idx}.png'
        )
        
        # 6. 3D visualizations for selected layers
        for layer_name in list(feature_maps.keys())[::len(feature_maps)//3]:
            if layer_name in feature_maps:
                create_3d_feature_visualization(
                    feature_maps[layer_name],
                    save_path=output_dir / f'feature_3d_{layer_name}_slice{slice_idx}.png'
                )
    
    # Clean up hooks
    visualizer.remove_hooks()
    
    # Create comparison visualizations
    logger.info("\nCreating comparison visualizations...")
    
    # Compare features between slices
    features_comparison = []
    for slice_idx in slices:
        ct_slice = preprocess_ct(ct_volume[slice_idx:slice_idx+1], window='lung')[0]
        embedding = visualizer.processor.generate_embeddings(ct_slice, mode='2d')
        features_comparison.append(embedding)
    
    # Feature evolution across slices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (slice_idx, features) in enumerate(zip(slices, features_comparison)):
        ax = axes[idx]
        # Reshape to 2D for visualization
        feat_2d = features.reshape(32, -1)[:, :32]  # Take subset
        im = ax.imshow(feat_2d, cmap='viridis', aspect='auto')
        ax.set_title(f'Slice {slice_idx} Features')
        ax.set_xlabel('Feature Dimension')
        ax.set_ylabel('Feature Index')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Feature Comparison Across Slices', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_comparison_slices.png', dpi=150, bbox_inches='tight')
    
    logger.info(f"\nAll visualizations saved to: {output_dir}")
    logger.info("Visualization report complete!")


def main():
    """Main function to run all visualizations"""
    
    # Load sample CT data
    ct_dir = Path("../samples/CT")
    if not ct_dir.exists():
        logger.error(f"CT samples not found at {ct_dir}")
        logger.info("Creating synthetic data for demonstration...")
        
        # Create synthetic CT-like data
        ct_volume = np.random.randn(30, 512, 512) * 30 + 100
        # Add some structure
        for i in range(30):
            center = (256, 256)
            y, x = np.ogrid[:512, :512]
            mask = (x - center[0])**2 + (y - center[1])**2 <= (100 + i*3)**2
            ct_volume[i][mask] += 50
    else:
        ct_volume, metadata = load_dicom_series(ct_dir)
        logger.info(f"Loaded CT volume: {ct_volume.shape}")
    
    # Generate comprehensive visualizations
    create_comprehensive_visualization_report(ct_volume)
    
    logger.info("\n" + "=" * 50)
    logger.info("Enhanced visualization demo complete!")
    logger.info("Check results_radimagenet_visualizations/ for all outputs")


if __name__ == "__main__":
    main()