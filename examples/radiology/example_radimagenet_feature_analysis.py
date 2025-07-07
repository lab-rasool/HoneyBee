"""
Deep Feature Analysis and Evolution Visualization for RadImageNet

Comprehensive analysis of feature evolution, channel relationships, and
network behavior across different depths and medical image characteristics.
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
from matplotlib.patches import Circle, Rectangle
import seaborn as sns
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import cv2
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# Import radiology modules
from radiology.data_management import load_dicom_series
from radiology.preprocessing import preprocess_ct
from radiology.ai_integration import RadImageNetProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepFeatureAnalyzer:
    """Advanced analysis of deep features from RadImageNet models"""

    def __init__(self, model_name="densenet121"):
        self.processor = RadImageNetProcessor(model_name=model_name, pretrained=True)
        self.model = self.processor.model
        self.device = self.processor.device
        self.feature_maps = {}
        self.hook_handles = []

    def register_all_hooks(self):
        """Register hooks on all convolutional layers"""

        def get_features(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU)):
                handle = module.register_forward_hook(get_features(name))
                self.hook_handles.append(handle)

    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.feature_maps = {}

    def extract_all_features(self, image):
        """Extract features from all layers"""
        self.feature_maps = {}

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            image_tensor = image

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            _ = self.model(image_tensor)

        return self.feature_maps

    def analyze_feature_evolution(self, features, save_path="feature_evolution.png"):
        """Analyze how features evolve through the network"""
        # Sort layers by depth
        sorted_layers = sorted(features.keys(), key=lambda x: len(x.split(".")))

        # Collect statistics
        stats_data = {
            "layer": [],
            "depth": [],
            "mean_activation": [],
            "std_activation": [],
            "sparsity": [],
            "channel_diversity": [],
            "spatial_entropy": [],
            "feature_complexity": [],
        }

        for depth, layer_name in enumerate(sorted_layers):
            feat = features[layer_name]
            if len(feat.shape) == 4:  # B x C x H x W
                feat = feat[0]  # Remove batch

                stats_data["layer"].append(layer_name.split(".")[-1])
                stats_data["depth"].append(depth)
                stats_data["mean_activation"].append(feat.mean().item())
                stats_data["std_activation"].append(feat.std().item())
                stats_data["sparsity"].append((feat == 0).float().mean().item())

                # Channel diversity (variance across channels)
                channel_means = feat.mean(dim=(1, 2))
                stats_data["channel_diversity"].append(channel_means.std().item())

                # Spatial entropy (information content)
                spatial_entropy = 0
                for c in range(feat.shape[0]):
                    channel = feat[c].cpu().numpy()
                    if channel.std() > 0:
                        hist, _ = np.histogram(channel, bins=50)
                        hist = hist / hist.sum()
                        hist = hist[hist > 0]
                        spatial_entropy += -np.sum(hist * np.log(hist))
                stats_data["spatial_entropy"].append(spatial_entropy / feat.shape[0])

                # Feature complexity (gradient magnitude)
                grad_x = torch.abs(feat[:, :, 1:] - feat[:, :, :-1]).mean()
                grad_y = torch.abs(feat[:, 1:, :] - feat[:, :-1, :]).mean()
                stats_data["feature_complexity"].append((grad_x + grad_y).item())

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # 1. Feature statistics evolution
        ax1 = fig.add_subplot(gs[0, :])
        x = range(len(stats_data["depth"]))

        ax1_twin = ax1.twinx()

        # Primary axis - activations
        l1 = ax1.plot(x, stats_data["mean_activation"], "b-", label="Mean Activation")
        l2 = ax1.plot(x, stats_data["std_activation"], "g-", label="Std Activation")
        ax1.set_xlabel("Layer Depth")
        ax1.set_ylabel("Activation Value", color="b")
        ax1.tick_params(axis="y", labelcolor="b")

        # Secondary axis - sparsity
        l3 = ax1_twin.plot(x, stats_data["sparsity"], "r-", label="Sparsity")
        ax1_twin.set_ylabel("Sparsity", color="r")
        ax1_twin.tick_params(axis="y", labelcolor="r")

        # Legend
        lns = l1 + l2 + l3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="upper left")
        ax1.set_title("Feature Statistics Evolution Through Network")
        ax1.grid(True, alpha=0.3)

        # 2. Channel diversity heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        diversity_matrix = np.array(stats_data["channel_diversity"]).reshape(-1, 1)
        im = ax2.imshow(diversity_matrix.T, aspect="auto", cmap="viridis")
        ax2.set_xlabel("Layer Depth")
        ax2.set_title("Channel Diversity Evolution")
        ax2.set_yticks([])
        plt.colorbar(im, ax=ax2)

        # 3. Spatial entropy evolution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(stats_data["spatial_entropy"], "purple", linewidth=2)
        ax3.fill_between(
            range(len(stats_data["spatial_entropy"])),
            stats_data["spatial_entropy"],
            alpha=0.3,
            color="purple",
        )
        ax3.set_xlabel("Layer Depth")
        ax3.set_ylabel("Spatial Entropy")
        ax3.set_title("Information Content Evolution")
        ax3.grid(True, alpha=0.3)

        # 4. Feature complexity
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(stats_data["feature_complexity"], "orange", linewidth=2, marker="o")
        ax4.set_xlabel("Layer Depth")
        ax4.set_ylabel("Feature Complexity")
        ax4.set_title("Gradient-based Complexity")
        ax4.grid(True, alpha=0.3)

        # 5. Feature map samples at different depths
        sample_indices = np.linspace(0, len(sorted_layers) - 1, 5, dtype=int)
        for idx, sample_idx in enumerate(sample_indices):
            ax = fig.add_subplot(gs[2, idx % 3])

            layer_name = sorted_layers[sample_idx]
            feat = features[layer_name]
            if len(feat.shape) == 4:
                feat = feat[0]

            # Show average feature map
            avg_feat = feat.mean(dim=0).cpu().numpy()
            im = ax.imshow(avg_feat, cmap="viridis")
            ax.set_title(f"Layer {sample_idx}: {layer_name.split('.')[-1]}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle("Deep Feature Evolution Analysis", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature evolution analysis saved to: {save_path}")

        return stats_data

    def analyze_channel_relationships(
        self, features, layer_name, save_path="channel_analysis.png"
    ):
        """Deep analysis of channel relationships and clustering"""
        if layer_name not in features:
            logger.error(f"Layer {layer_name} not found")
            return

        feat = features[layer_name]
        if len(feat.shape) == 4:
            feat = feat[0]  # Remove batch

        n_channels = feat.shape[0]

        # Flatten spatial dimensions for analysis
        feat_flat = feat.view(n_channels, -1).cpu().numpy()

        # Create visualization
        fig = plt.figure(figsize=(20, 18))
        gs = gridspec.GridSpec(4, 3, figure=fig, height_ratios=[1, 1, 0.8, 0.8])

        # 1. Channel correlation network
        ax1 = fig.add_subplot(gs[0, :2])

        # Compute correlation
        corr_matrix = np.corrcoef(feat_flat)

        # Create network graph for strong correlations
        threshold = 0.7
        G = nx.Graph()

        for i in range(n_channels):
            G.add_node(i)

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                if abs(corr_matrix[i, j]) > threshold:
                    G.add_edge(i, j, weight=abs(corr_matrix[i, j]))

        # Layout and draw
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Node colors by activation strength
        node_colors = feat.mean(dim=(1, 2)).cpu().numpy()

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, cmap="viridis", node_size=300, ax=ax1
        )
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax1)

        ax1.set_title(f"Channel Correlation Network (threshold={threshold})")
        ax1.axis("off")

        # 2. Channel clustering
        ax2 = fig.add_subplot(gs[0, 2])

        # Perform clustering
        n_clusters = min(8, n_channels // 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(feat_flat)

        # Visualize clusters
        cluster_matrix = cluster_labels.reshape(1, -1)
        ax2.imshow(cluster_matrix, aspect="auto", cmap="tab10")
        ax2.set_xlabel("Channel Index")
        ax2.set_title(f"Channel Clusters (k={n_clusters})")
        ax2.set_yticks([])

        # 3. PCA of channels
        ax3 = fig.add_subplot(gs[1, 0])

        pca = PCA(n_components=2)
        channels_pca = pca.fit_transform(feat_flat)

        scatter = ax3.scatter(
            channels_pca[:, 0],
            channels_pca[:, 1],
            c=cluster_labels,
            cmap="tab10",
            s=100,
        )
        ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax3.set_title("Channel PCA Projection")
        plt.colorbar(scatter, ax=ax3, label="Cluster")

        # 4. Channel activation patterns
        ax4 = fig.add_subplot(gs[1, 1])

        # Sort channels by mean activation
        channel_means = feat.mean(dim=(1, 2)).cpu().numpy()
        sorted_indices = np.argsort(channel_means)[::-1]

        # Show top channels
        n_show = min(16, n_channels)
        top_channels = sorted_indices[:n_show]

        pattern_matrix = np.zeros((n_show, 4))
        for idx, ch_idx in enumerate(top_channels):
            ch_feat = feat[ch_idx].cpu().numpy()
            pattern_matrix[idx, 0] = ch_feat.mean()
            pattern_matrix[idx, 1] = ch_feat.std()
            pattern_matrix[idx, 2] = (ch_feat > 0).mean()
            pattern_matrix[idx, 3] = stats.skew(ch_feat.flatten())

        im = ax4.imshow(pattern_matrix.T, aspect="auto", cmap="RdBu_r")
        ax4.set_xlabel("Channel Rank")
        ax4.set_ylabel("Statistic")
        ax4.set_yticks(range(4))
        ax4.set_yticklabels(["Mean", "Std", "Sparsity", "Skewness"])
        ax4.set_title("Top Channel Statistics")
        plt.colorbar(im, ax=ax4)

        # 5. Channel response patterns
        ax5 = fig.add_subplot(gs[1, 2])

        # Compute channel response diversity
        response_diversity = []
        for ch_idx in range(n_channels):
            ch_feat = feat[ch_idx].cpu().numpy()
            # Compute local variance
            kernel = np.ones((3, 3)) / 9
            local_mean = signal.convolve2d(ch_feat, kernel, mode="same")
            local_var = (
                signal.convolve2d(ch_feat**2, kernel, mode="same") - local_mean**2
            )
            response_diversity.append(local_var.mean())

        ax5.hist(response_diversity, bins=30, alpha=0.7, color="green")
        ax5.set_xlabel("Response Diversity")
        ax5.set_ylabel("Number of Channels")
        ax5.set_title("Channel Response Diversity Distribution")
        ax5.grid(True, alpha=0.3)

        # 6. Sample channel visualizations
        for idx in range(6):
            row = 2 + idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])

            if idx < len(top_channels):
                ch_idx = top_channels[idx]
                ch_feat = feat[ch_idx].cpu().numpy()

                im = ax.imshow(ch_feat, cmap="viridis")
                ax.set_title(f"Channel {ch_idx} (rank {idx + 1})")
                ax.axis("off")

                # Add statistics
                stats_text = f"μ={ch_feat.mean():.2f}\nσ={ch_feat.std():.2f}"
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.suptitle(f"Channel Relationship Analysis - {layer_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Channel analysis saved to: {save_path}")

        return cluster_labels, channels_pca

    def analyze_receptive_field_evolution(self, image, save_path="rf_evolution.png"):
        """Analyze effective receptive field evolution"""
        features = self.extract_all_features(image)

        # Sort layers by depth
        sorted_layers = sorted(features.keys(), key=lambda x: len(x.split(".")))

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig)

        # Original image
        ax = fig.add_subplot(gs[0, 0])
        if isinstance(image, torch.Tensor):
            img_np = image.squeeze().cpu().numpy()
        else:
            img_np = image
        ax.imshow(img_np, cmap="gray")
        ax.set_title("Original Image")
        ax.axis("off")

        # Sample layers
        layer_indices = np.linspace(0, len(sorted_layers) - 1, 11, dtype=int)

        for idx, layer_idx in enumerate(layer_indices):
            row = (idx + 1) // 4
            col = (idx + 1) % 4
            ax = fig.add_subplot(gs[row, col])

            layer_name = sorted_layers[layer_idx]
            feat = features[layer_name]

            if len(feat.shape) == 4:
                feat = feat[0]

                # Compute effective receptive field
                # Find maximum activation
                avg_feat = feat.mean(dim=0)
                max_val = avg_feat.max()
                max_loc = (avg_feat == max_val).nonzero()[0]

                if len(max_loc) > 0:
                    y, x = max_loc[0].item(), max_loc[1].item()

                    # Estimate RF size based on layer depth
                    rf_size = min(img_np.shape[0], 2 ** (layer_idx // 3 + 2) * 3)

                    # Create attention map
                    attention = torch.zeros_like(avg_feat)
                    y_start = max(0, y - rf_size // 2)
                    y_end = min(avg_feat.shape[0], y + rf_size // 2)
                    x_start = max(0, x - rf_size // 2)
                    x_end = min(avg_feat.shape[1], x + rf_size // 2)

                    attention[y_start:y_end, x_start:x_end] = 1

                    # Resize to original image size
                    attention_resized = (
                        F.interpolate(
                            attention.unsqueeze(0).unsqueeze(0).float(),
                            size=img_np.shape,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

                    # Overlay
                    ax.imshow(img_np, cmap="gray", alpha=0.7)
                    ax.imshow(attention_resized, cmap="Reds", alpha=0.3)

                    # Add circle at center
                    scale_y = img_np.shape[0] / avg_feat.shape[0]
                    scale_x = img_np.shape[1] / avg_feat.shape[1]
                    circle = Circle(
                        (x * scale_x, y * scale_y),
                        rf_size / 2,
                        fill=False,
                        color="red",
                        linewidth=2,
                    )
                    ax.add_patch(circle)

                    ax.set_title(f"Layer {layer_idx}: RF≈{rf_size}")
                else:
                    ax.imshow(img_np, cmap="gray")
                    ax.set_title(f"Layer {layer_idx}")
            else:
                ax.imshow(img_np, cmap="gray")
                ax.set_title(f"Layer {layer_idx}")

            ax.axis("off")

        plt.suptitle("Receptive Field Evolution Through Network", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Receptive field evolution saved to: {save_path}")

    def create_filter_visualization(self, save_path="filter_viz.png"):
        """Visualize learned filters from early layers"""
        # Find first few conv layers
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and len(conv_layers) < 4:
                conv_layers.append((name, module))

        n_layers = len(conv_layers)
        
        # Create figure with subplots for each layer
        fig, axes = plt.subplots(n_layers, 1, figsize=(20, 4 * n_layers))
        if n_layers == 1:
            axes = [axes]

        for layer_idx, (layer_name, conv_layer) in enumerate(conv_layers):
            filters = conv_layer.weight.data.cpu()
            n_filters = filters.shape[0]
            n_show = min(16, n_filters)  # Show fewer filters per layer

            # Create a grid of filters for this layer
            grid_size = int(np.ceil(np.sqrt(n_show)))
            filter_grid = np.zeros((grid_size * filters.shape[2], grid_size * filters.shape[3]))
            
            for filter_idx in range(n_show):
                row = filter_idx // grid_size
                col = filter_idx % grid_size
                
                # Get filter
                filt = filters[filter_idx]

                # Handle different numbers of input channels
                if filt.shape[0] == 1:
                    # Single channel
                    filt_img = filt[0].numpy()
                elif filt.shape[0] == 3:
                    # RGB - take mean
                    filt_img = filt.mean(dim=0).numpy()
                else:
                    # Many channels - take first
                    filt_img = filt[0].numpy()

                # Normalize for visualization
                filt_img = (filt_img - filt_img.min()) / (
                    filt_img.max() - filt_img.min() + 1e-8
                )
                
                # Place in grid
                h, w = filt_img.shape
                filter_grid[row*h:(row+1)*h, col*w:(col+1)*w] = filt_img

            # Display the grid for this layer
            axes[layer_idx].imshow(filter_grid, cmap="gray")
            axes[layer_idx].set_title(f"Layer: {layer_name} (showing {n_show}/{n_filters} filters)")
            axes[layer_idx].axis("off")

        plt.suptitle("Learned Filters Visualization", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Filter visualization saved to: {save_path}")


def create_temporal_feature_evolution(ct_volume, save_path="temporal_evolution.png"):
    """Analyze feature evolution across multiple slices (temporal/spatial)"""
    analyzer = DeepFeatureAnalyzer("densenet121")
    analyzer.register_all_hooks()

    # Select slices across volume
    n_slices = min(10, ct_volume.shape[0])
    slice_indices = np.linspace(0, ct_volume.shape[0] - 1, n_slices, dtype=int)

    # Collect features for each slice
    all_features = []
    slice_positions = []

    logger.info("Extracting features across slices...")
    for slice_idx in slice_indices:
        ct_slice = preprocess_ct(ct_volume[slice_idx : slice_idx + 1], window="lung")[0]
        features = analyzer.extract_all_features(ct_slice)

        # Get embedding from last layer
        last_layer = list(features.keys())[-1]
        if last_layer in features:
            feat = features[last_layer]
            if len(feat.shape) == 4:
                # Global average pooling
                embedding = feat.mean(dim=[2, 3]).squeeze().cpu().numpy()
                all_features.append(embedding)
                slice_positions.append(slice_idx / ct_volume.shape[0])

    all_features = np.array(all_features)

    # Create visualization
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # 1. Feature evolution heatmap
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(all_features.T[:100], aspect="auto", cmap="viridis")
    ax1.set_xlabel("Slice Position")
    ax1.set_ylabel("Feature Dimension")
    ax1.set_title("Feature Evolution Across CT Volume")
    plt.colorbar(im, ax=ax1)

    # 2. Feature trajectory in 2D
    ax2 = fig.add_subplot(gs[1, 0])

    # PCA projection
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(all_features)

    # Plot trajectory
    ax2.plot(features_pca[:, 0], features_pca[:, 1], "o-", markersize=8)

    # Color by position
    scatter = ax2.scatter(
        features_pca[:, 0], features_pca[:, 1], c=slice_positions, cmap="viridis", s=100
    )
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax2.set_title("Feature Trajectory in PCA Space")
    plt.colorbar(scatter, ax=ax2, label="Slice Position")

    # 3. Feature distance matrix
    ax3 = fig.add_subplot(gs[1, 1])

    # Compute pairwise distances
    dist_matrix = squareform(pdist(all_features, "euclidean"))

    im = ax3.imshow(dist_matrix, cmap="viridis")
    ax3.set_xlabel("Slice Index")
    ax3.set_ylabel("Slice Index")
    ax3.set_title("Feature Distance Matrix")
    plt.colorbar(im, ax=ax3)

    # 4. Feature variability
    ax4 = fig.add_subplot(gs[1, 2])

    feature_std = all_features.std(axis=0)
    top_variable = np.argsort(feature_std)[::-1][:50]

    ax4.bar(range(len(top_variable)), feature_std[top_variable])
    ax4.set_xlabel("Feature Index (sorted)")
    ax4.set_ylabel("Standard Deviation")
    ax4.set_title("Top 50 Most Variable Features")
    ax4.grid(True, alpha=0.3)

    # 5. Slice similarity network
    ax5 = fig.add_subplot(gs[2, :])

    # Create similarity graph
    G = nx.Graph()
    threshold = np.percentile(dist_matrix, 30)  # Connect similar slices

    for i in range(n_slices):
        G.add_node(i, pos=slice_positions[i])

    for i in range(n_slices):
        for j in range(i + 1, n_slices):
            if dist_matrix[i, j] < threshold:
                G.add_edge(i, j, weight=1 / dist_matrix[i, j])

    # Layout
    pos = {}
    for i in range(n_slices):
        angle = 2 * np.pi * i / n_slices
        pos[i] = (np.cos(angle), np.sin(angle))

    # Draw
    node_colors = slice_positions
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, cmap="viridis", node_size=500, ax=ax5
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax5)
    nx.draw_networkx_labels(
        G, pos, {i: f"{i}" for i in range(n_slices)}, font_size=10, ax=ax5
    )

    ax5.set_title("Slice Similarity Network (based on features)")
    ax5.axis("off")

    plt.suptitle("Temporal/Spatial Feature Evolution Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Temporal evolution saved to: {save_path}")

    analyzer.remove_hooks()

    return all_features, slice_positions


def create_comprehensive_feature_report(ct_volume):
    """Generate comprehensive feature analysis report"""
    logger.info("Generating Comprehensive Feature Analysis Report")
    logger.info("=" * 50)

    # Create output directory
    output_dir = Path("results_feature_analysis")
    output_dir.mkdir(exist_ok=True)

    # Initialize analyzer
    analyzer = DeepFeatureAnalyzer("densenet121")
    analyzer.register_all_hooks()

    # Select representative slice
    middle_slice = ct_volume.shape[0] // 2
    ct_slice = preprocess_ct(ct_volume[middle_slice : middle_slice + 1], window="lung")[
        0
    ]

    # 1. Extract all features
    logger.info("Extracting features from all layers...")
    all_features = analyzer.extract_all_features(ct_slice)

    # 2. Feature evolution analysis
    logger.info("Analyzing feature evolution...")
    stats_data = analyzer.analyze_feature_evolution(
        all_features, save_path=output_dir / "feature_evolution_analysis.png"
    )

    # 3. Channel relationship analysis
    logger.info("Analyzing channel relationships...")
    # Select a middle layer
    layer_names = list(all_features.keys())
    middle_layer = layer_names[len(layer_names) // 2]

    cluster_labels, channels_pca = analyzer.analyze_channel_relationships(
        all_features,
        middle_layer,
        save_path=output_dir / f"channel_analysis_{middle_layer}.png",
    )

    # 4. Receptive field evolution
    logger.info("Analyzing receptive field evolution...")
    analyzer.analyze_receptive_field_evolution(
        ct_slice, save_path=output_dir / "receptive_field_evolution.png"
    )

    # 5. Filter visualization
    logger.info("Visualizing learned filters...")
    analyzer.create_filter_visualization(
        save_path=output_dir / "filter_visualization.png"
    )

    # Clean up hooks
    analyzer.remove_hooks()

    # 6. Temporal feature evolution
    logger.info("Analyzing temporal feature evolution...")
    temporal_features, slice_positions = create_temporal_feature_evolution(
        ct_volume, save_path=output_dir / "temporal_feature_evolution.png"
    )

    logger.info(f"\nAll feature analyses saved to: {output_dir}")
    logger.info("Feature analysis report complete!")

    return stats_data, all_features


def main():
    """Main function to run feature analysis"""

    # Load sample CT data
    ct_dir = Path("../samples/CT")
    if not ct_dir.exists():
        logger.error(f"CT samples not found at {ct_dir}")
        logger.info("Creating synthetic data for demonstration...")

        # Create synthetic CT-like data with structures
        ct_volume = np.zeros((30, 256, 256))
        for i in range(30):
            # Add circular structure
            center = (128 + i * 2, 128)
            y, x = np.ogrid[:256, :256]
            mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= (50 + i) ** 2
            ct_volume[i][mask] = 150

            # Add some noise
            ct_volume[i] += np.random.randn(256, 256) * 10
    else:
        ct_volume, metadata = load_dicom_series(ct_dir)
        logger.info(f"Loaded CT volume: {ct_volume.shape}")

    # Generate comprehensive feature analysis
    stats_data, features = create_comprehensive_feature_report(ct_volume)

    logger.info("\n" + "=" * 50)
    logger.info("Feature analysis demo complete!")
    logger.info("Check results_feature_analysis/ for all outputs")


if __name__ == "__main__":
    main()
