#!/usr/bin/env python3
"""
Create journal-quality t-SNE density plots for staining normalization comparison.
Improvements:
- PDF output format
- Larger, more visible points
- Larger text for publication
- Better color schemes
- Enhanced visual clarity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
import os
import json

# Set matplotlib parameters for journal quality
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts for editing in Adobe Illustrator
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 14

def create_journal_quality_tsne_density():
    """Create publication-ready t-SNE density plots."""
    
    # Load the pre-computed t-SNE embeddings
    embeddings_dir = 'embeddings'
    figures_dir = 'figures'
    
    # Load data
    X_norm_tsne = np.load(os.path.join(embeddings_dir, 'tsne_normalized.npy'))
    X_no_norm_tsne = np.load(os.path.join(embeddings_dir, 'tsne_non_normalized.npy'))
    y = np.load(os.path.join(embeddings_dir, 'labels.npy'))
    
    # Load label mapping
    with open(os.path.join(embeddings_dir, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    
    # Get cancer type labels
    labels = [''] * len(y)
    for cancer_type, label_idx in label_mapping.items():
        mask = y == label_idx
        labels = np.array(labels)
        labels[mask] = cancer_type
    
    # Create figure with better size for journal publication
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define better colors for publication
    colors = ['#2E86AB', '#A23B72']  # Professional blue and burgundy
    
    # Function to create density plot
    def plot_density(ax, X_tsne, y, labels, title, colors):
        # Set axis limits with padding
        x_min, x_max = X_tsne[:, 0].min() - 10, X_tsne[:, 0].max() + 10
        y_min, y_max = X_tsne[:, 1].min() - 10, X_tsne[:, 1].max() + 10
        
        # Create higher resolution grid for smoother contours
        xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        # Plot each cancer type
        for i, label in enumerate(np.unique(y)):
            idx = y == label
            cancer_type = labels[np.where(y == label)[0][0]]
            
            # Get points for this class
            xy = X_tsne[idx]
            
            if len(xy) > 5:  # Need enough points for KDE
                # Calculate density with optimized bandwidth
                kernel = gaussian_kde(xy.T)
                kernel.set_bandwidth(kernel.factor * 0.8)  # Slightly tighter bandwidth
                
                # Evaluate kernel
                f = np.reshape(kernel(positions).T, xx.shape)
                
                # Plot filled contours with better levels
                levels = np.linspace(f.min(), f.max(), 15)
                contours = ax.contourf(xx, yy, f, levels=levels, alpha=0.5, 
                                      cmap='Blues' if i == 0 else 'Reds')
                
                # Add contour lines for clarity
                ax.contour(xx, yy, f, levels=5, colors=colors[i], alpha=0.8, linewidths=1.5)
            
            # Plot scatter points with better visibility
            ax.scatter(xy[:, 0], xy[:, 1], c=colors[i], label=cancer_type, 
                      alpha=0.8, s=60, edgecolors='white', linewidth=0.5)
        
        # Customize axis
        ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('t-SNE 1', fontsize=16)
        ax.set_ylabel('t-SNE 2', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Legend with better positioning and style
        legend = ax.legend(fontsize=14, frameon=True, fancybox=True, 
                          shadow=True, loc='best', markerscale=1.5)
        legend.get_frame().set_alpha(0.9)
        
        # Set aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return ax
    
    # Create normalized plot
    plot_density(ax1, X_norm_tsne, y, labels, 
                'With Stain Normalization', colors)
    
    # Create non-normalized plot
    plot_density(ax2, X_no_norm_tsne, y, labels, 
                'Without Stain Normalization', colors)
    
    # Add overall title
    fig.suptitle('t-SNE Density Visualization of Pathology Embeddings', 
                 fontsize=20, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as PDF with high quality
    output_path = os.path.join(figures_dir, 'tsne_density_journal_quality.pdf')
    plt.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved journal-quality figure to: {output_path}")
    
    # Also save as high-resolution PNG for preview
    output_path_png = os.path.join(figures_dir, 'tsne_density_journal_quality.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved preview PNG to: {output_path_png}")
    
    plt.close()
    
    # Create individual plots for each condition (often required by journals)
    for condition, X_tsne, suffix in [
        ('With Stain Normalization', X_norm_tsne, 'normalized'),
        ('Without Stain Normalization', X_no_norm_tsne, 'non_normalized')
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_density(ax, X_tsne, y, labels, condition, colors)
        
        output_path = os.path.join(figures_dir, f'tsne_density_{suffix}_journal_quality.pdf')
        plt.savefig(output_path, format='pdf', dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved individual figure to: {output_path}")
        plt.close()

if __name__ == '__main__':
    create_journal_quality_tsne_density()