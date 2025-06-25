#!/usr/bin/env python3
"""
Create a side-by-side comparison of confusion matrices for all fusion methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Load confusion matrix data
methods = ["concat", "mean_pool", "kp"]
method_names = ["Concatenation", "Mean Pooling", "Kronecker Product"]
cm_data = []

for method in methods:
    with open(f"classification_results_multimodal/figures/confusion_matrix_{method}_data.json", 'r') as f:
        cm_data.append(json.load(f))

# Create figure with subplots
fig = plt.figure(figsize=(24, 8))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

# Get common labels (should be same for all)
labels = cm_data[0]["labels"]

# Create colormap limits based on all matrices
vmax = 0
for data in cm_data:
    cm = np.array(data["confusion_matrix"])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    vmax = max(vmax, np.nanmax(cm_norm))

# Plot each confusion matrix
for idx, (data, method_name) in enumerate(zip(cm_data, method_names)):
    ax = fig.add_subplot(gs[0, idx])
    
    # Get confusion matrix and normalize
    cm = np.array(data["confusion_matrix"])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create mask for zero values
    mask = cm == 0
    
    # Plot heatmap without annotations (too crowded for side-by-side)
    sns.heatmap(cm_normalized,
                mask=mask,
                cmap='Blues',
                vmin=0,
                vmax=vmax,
                cbar_kws={'label': 'Proportion'} if idx == 2 else None,
                cbar=idx == 2,  # Only show colorbar for last plot
                xticklabels=labels,
                yticklabels=labels if idx == 0 else False,  # Only show y-labels for first plot
                square=True,
                linewidths=0.1,
                linecolor='gray',
                ax=ax)
    
    # Set title with accuracy
    ax.set_title(f'{method_name}\nAccuracy: {data["accuracy"]:.4f}', fontsize=14, pad=10)
    
    # Set axis labels
    if idx == 1:
        ax.set_xlabel('Predicted Label', fontsize=12)
    if idx == 0:
        ax.set_ylabel('True Label', fontsize=12)
    
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    if idx == 0:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

# Add main title
fig.suptitle('Confusion Matrices Comparison - Multimodal Fusion Methods', fontsize=18, y=1.02)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = "classification_results_multimodal/figures/confusion_matrices_comparison"
for fmt in ['png', 'pdf']:
    filepath = f"{output_path}.{fmt}"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")

plt.close()

# Create a summary statistics comparison
print("\n" + "="*60)
print("CONFUSION MATRIX STATISTICS SUMMARY")
print("="*60)

for data, method_name in zip(cm_data, method_names):
    cm = np.array(data["confusion_matrix"])
    
    # Calculate per-class metrics
    precision = np.diag(cm) / cm.sum(axis=0)
    recall = np.diag(cm) / cm.sum(axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Handle NaN values
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    
    print(f"\n{method_name.upper()}:")
    print(f"  Overall Accuracy: {data['accuracy']:.4f}")
    print(f"  Mean Precision: {np.mean(precision):.4f} (±{np.std(precision):.4f})")
    print(f"  Mean Recall: {np.mean(recall):.4f} (±{np.std(recall):.4f})")
    print(f"  Mean F1-Score: {np.mean(f1):.4f} (±{np.std(f1):.4f})")
    
    # Find best and worst performing classes
    best_idx = np.argmax(f1)
    worst_idx = np.argmin(f1[f1 > 0])  # Exclude zeros
    
    print(f"  Best performing: {labels[best_idx]} (F1={f1[best_idx]:.3f})")
    print(f"  Worst performing: {labels[worst_idx]} (F1={f1[worst_idx]:.3f})")

print("\n" + "="*60)