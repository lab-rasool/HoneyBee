#!/usr/bin/env python3
"""
Create advanced journal-quality t-SNE visualizations with multiple style options.
Provides various publication-ready formats and styles.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import seaborn as sns
import os
import json
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap

# Set matplotlib parameters for journal quality
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['axes.linewidth'] = 1.5

def compute_confidence_ellipse(x, y, n_std=2.0):
    """
    Create a confidence ellipse for the data points.
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    if len(x) < 3:
        return None
        
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', linewidth=2)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    
    ellipse.set_transform(transf)
    
    return ellipse

def create_nature_style_tsne():
    """Create Nature journal style t-SNE plots."""
    
    # Load data
    embeddings_dir = 'embeddings'
    figures_dir = 'figures'
    
    X_norm_tsne = np.load(os.path.join(embeddings_dir, 'tsne_normalized.npy'))
    X_no_norm_tsne = np.load(os.path.join(embeddings_dir, 'tsne_non_normalized.npy'))
    y = np.load(os.path.join(embeddings_dir, 'labels.npy'))
    
    with open(os.path.join(embeddings_dir, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    
    labels = [''] * len(y)
    for cancer_type, label_idx in label_mapping.items():
        mask = y == label_idx
        labels = np.array(labels)
        labels[mask] = cancer_type
    
    # Nature style colors
    colors = ['#0173B2', '#DE8F05']  # Blue and orange from Nature palette
    
    # Create figure
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    
    for idx, (X_tsne, title) in enumerate([
        (X_norm_tsne, 'Stain normalized'),
        (X_no_norm_tsne, 'Non-normalized')
    ]):
        ax = fig.add_subplot(gs[0, idx])
        
        # Plot each cancer type
        for i, label in enumerate(np.unique(y)):
            idx_mask = y == label
            cancer_type = labels[np.where(y == label)[0][0]]
            xy = X_tsne[idx_mask]
            
            # Plot points
            scatter = ax.scatter(xy[:, 0], xy[:, 1], c=colors[i], label=cancer_type,
                               alpha=0.6, s=40, edgecolors='none', rasterized=True)
            
            # Add confidence ellipse
            if len(xy) > 3:
                ellipse = compute_confidence_ellipse(xy[:, 0], xy[:, 1], n_std=2)
                if ellipse:
                    ellipse.set_edgecolor(colors[i])
                    ellipse.set_facecolor('none')
                    ellipse.set_linewidth(2)
                    ellipse.set_linestyle('--')
                    ellipse.set_alpha(0.8)
                    ax.add_artist(ellipse)
        
        # Styling
        ax.set_xlabel('t-SNE 1', fontsize=16, fontweight='normal')
        ax.set_ylabel('t-SNE 2', fontsize=16, fontweight='normal')
        ax.set_title(title, fontsize=18, fontweight='normal', pad=10)
        
        # Clean spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        
        # Tick styling
        ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.5)
        ax.xaxis.set_tick_params(top=False)
        ax.yaxis.set_tick_params(right=False)
        
        # Legend
        legend = ax.legend(frameon=True, fontsize=14, loc='best',
                          handletextpad=0.5, columnspacing=1, borderpad=0.5)
        legend.get_frame().set_linewidth(1.5)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # Set equal aspect
        ax.set_aspect('equal', 'box')
    
    # Save
    output_path = os.path.join(figures_dir, 'tsne_nature_style.pdf')
    plt.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Saved Nature-style figure to: {output_path}")
    plt.close()

def create_science_style_tsne():
    """Create Science journal style t-SNE plots with density contours."""
    
    # Load data
    embeddings_dir = 'embeddings'
    figures_dir = 'figures'
    
    X_norm_tsne = np.load(os.path.join(embeddings_dir, 'tsne_normalized.npy'))
    X_no_norm_tsne = np.load(os.path.join(embeddings_dir, 'tsne_non_normalized.npy'))
    y = np.load(os.path.join(embeddings_dir, 'labels.npy'))
    
    with open(os.path.join(embeddings_dir, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    
    labels = [''] * len(y)
    for cancer_type, label_idx in label_mapping.items():
        mask = y == label_idx
        labels = np.array(labels)
        labels[mask] = cancer_type
    
    # Science style - monochromatic with patterns
    base_colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    for ax_idx, (X_tsne, title, ax) in enumerate([
        (X_norm_tsne, 'Stain normalized', axes[0]),
        (X_no_norm_tsne, 'Non-normalized', axes[1])
    ]):
        # Create meshgrid for density
        x_min, x_max = X_tsne[:, 0].min() - 10, X_tsne[:, 0].max() + 10
        y_min, y_max = X_tsne[:, 1].min() - 10, X_tsne[:, 1].max() + 10
        xx, yy = np.mgrid[x_min:x_max:250j, y_min:y_max:250j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        for i, label in enumerate(np.unique(y)):
            idx = y == label
            cancer_type = labels[np.where(y == label)[0][0]]
            xy = X_tsne[idx]
            
            if len(xy) > 5:
                # Calculate density
                kernel = gaussian_kde(xy.T)
                kernel.set_bandwidth(kernel.factor * 0.7)
                f = np.reshape(kernel(positions).T, xx.shape)
                
                # Create custom colormap
                colors_i = plt.cm.colors.LinearSegmentedColormap.from_list(
                    "", ["white", base_colors[i]], N=256)
                
                # Plot density contours
                levels = np.percentile(f[f > 0], [50, 70, 85, 95])
                contour = ax.contour(xx, yy, f, levels=levels, colors=base_colors[i], 
                                   linewidths=2, alpha=0.8)
                
                # Add contour labels
                ax.clabel(contour, inline=True, fontsize=10, fmt='%1.0f%%')
            
            # Plot scatter
            ax.scatter(xy[:, 0], xy[:, 1], c=base_colors[i], label=cancer_type,
                      alpha=0.4, s=30, edgecolors='black', linewidth=0.5)
        
        # Styling
        ax.set_xlabel('t-SNE 1', fontsize=14)
        ax.set_ylabel('t-SNE 2', fontsize=14)
        ax.set_title(title, fontsize=16, pad=8)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Ticks
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Legend
        ax.legend(frameon=True, fontsize=12, loc='upper right',
                 framealpha=0.95, edgecolor='black')
        
        ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(figures_dir, 'tsne_science_style.pdf')
    plt.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Saved Science-style figure to: {output_path}")
    plt.close()

def create_cell_style_tsne():
    """Create Cell journal style t-SNE plots with hexbin density."""
    
    # Load data
    embeddings_dir = 'embeddings'
    figures_dir = 'figures'
    
    X_norm_tsne = np.load(os.path.join(embeddings_dir, 'tsne_normalized.npy'))
    X_no_norm_tsne = np.load(os.path.join(embeddings_dir, 'tsne_non_normalized.npy'))
    y = np.load(os.path.join(embeddings_dir, 'labels.npy'))
    
    with open(os.path.join(embeddings_dir, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    
    labels = [''] * len(y)
    for cancer_type, label_idx in label_mapping.items():
        mask = y == label_idx
        labels = np.array(labels)
        labels[mask] = cancer_type
    
    # Cell style colors
    colors = ['#E64B35', '#4DBBD5']  # Red and cyan from Cell palette
    
    fig = plt.figure(figsize=(14, 6))
    
    for idx, (X_tsne, title) in enumerate([
        (X_norm_tsne, 'Stain normalized'),
        (X_no_norm_tsne, 'Non-normalized')
    ]):
        ax = plt.subplot(1, 2, idx + 1)
        
        # Background hexbin for overall density
        ax.hexbin(X_tsne[:, 0], X_tsne[:, 1], gridsize=30, 
                 cmap='Greys', alpha=0.2, mincnt=1)
        
        # Plot each class
        for i, label in enumerate(np.unique(y)):
            idx_mask = y == label
            cancer_type = labels[np.where(y == label)[0][0]]
            xy = X_tsne[idx_mask]
            
            # Main scatter plot
            ax.scatter(xy[:, 0], xy[:, 1], c=colors[i], label=cancer_type,
                      alpha=0.7, s=50, edgecolors='white', linewidth=1)
            
            # Add centroid
            centroid = np.mean(xy, axis=0)
            ax.scatter(centroid[0], centroid[1], c='black', marker='x', 
                      s=200, linewidth=3)
            ax.scatter(centroid[0], centroid[1], c=colors[i], marker='x', 
                      s=150, linewidth=2)
        
        # Styling
        ax.set_xlabel('t-SNE dimension 1', fontsize=14, fontweight='bold')
        ax.set_ylabel('t-SNE dimension 2', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        
        # Frame
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')
        
        # Ticks
        ax.tick_params(axis='both', which='major', labelsize=12, 
                      width=2, length=6)
        
        # Legend with title
        legend = ax.legend(title='Cancer type', frameon=True, fontsize=12,
                          title_fontsize=13, loc='best', framealpha=0.95)
        legend.get_frame().set_linewidth(2)
        
        ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(figures_dir, 'tsne_cell_style.pdf')
    plt.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Saved Cell-style figure to: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("Creating journal-quality t-SNE visualizations...")
    create_nature_style_tsne()
    create_science_style_tsne()
    create_cell_style_tsne()
    print("All visualizations created successfully!")