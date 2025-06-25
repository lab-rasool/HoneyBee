#!/usr/bin/env python3
"""
Generate confusion matrices for multimodal fusion methods.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import os
from utils.data_loader import load_embeddings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Constants
OUTPUT_DIR = "classification_results_multimodal"
FUSION_METHODS = ["concat", "mean_pool", "kp"]
TEST_SIZE = 0.2
N_ESTIMATORS = 100
RANDOM_SEED = 42

# Set random seed
np.random.seed(RANDOM_SEED)


def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """Plot and save confusion matrix with better visualization."""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Create heatmap with annotations showing both count and percentage
    # Use a mask for zero values to make them white
    mask = cm == 0
    
    # Create custom annotation text
    annot_text = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                annot_text[i, j] = ""
            else:
                annot_text[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"
    
    # Plot heatmap
    sns.heatmap(cm_normalized, 
                annot=annot_text, 
                fmt='', 
                cmap='Blues',
                mask=mask,
                cbar_kws={'label': 'Proportion'},
                xticklabels=labels, 
                yticklabels=labels,
                vmin=0,
                vmax=1,
                linewidths=0.5,
                linecolor='gray',
                square=True)
    
    # Customize plot
    plt.title(title, fontsize=20, pad=20)
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    for fmt in ['png', 'pdf', 'svg']:
        filepath = f"{output_path}.{fmt}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    plt.close()
    
    return cm


def generate_fusion_confusion_matrix(fusion_method):
    """Generate confusion matrix for a specific fusion method."""
    print(f"\nGenerating confusion matrix for {fusion_method.upper()} fusion...")
    
    # Load embeddings with specific fusion method
    embeddings = load_embeddings(fusion_method=fusion_method)
    
    if "multimodal" not in embeddings:
        print(f"No multimodal embeddings found for {fusion_method}")
        return None
    
    # Get multimodal data
    X = embeddings["multimodal"]["X"]
    y = embeddings["multimodal"]["y"]
    
    # Handle NaN values
    if np.isnan(X).any():
        X = np.nan_to_num(X)
    
    # Get unique labels
    unique_labels = np.unique(y)
    print(f"Number of classes: {len(unique_labels)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    # Train classifier
    print("Training classifier...")
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix
    output_path = os.path.join(OUTPUT_DIR, "figures", f"confusion_matrix_{fusion_method}")
    cm = plot_confusion_matrix(
        y_test, 
        y_pred, 
        unique_labels,
        f"Confusion Matrix - Multimodal {fusion_method.upper()} Fusion\nAccuracy: {accuracy:.4f}",
        output_path
    )
    
    # Save raw confusion matrix data
    cm_data = {
        "method": fusion_method,
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "labels": unique_labels.tolist()
    }
    
    with open(f"{output_path}_data.json", 'w') as f:
        json.dump(cm_data, f, indent=2)
    
    return cm_data


def create_comparison_heatmap(all_results):
    """Create a comparison heatmap showing accuracy by cancer type for each method."""
    # Prepare data for heatmap
    methods = []
    cancer_types = set()
    
    # Get all cancer types
    for result in all_results:
        cancer_types.update(result["labels"])
    
    cancer_types = sorted(cancer_types)
    
    # Create accuracy matrix for each cancer type
    accuracy_matrix = []
    
    for result in all_results:
        method_accuracies = []
        cm = np.array(result["confusion_matrix"])
        labels = result["labels"]
        
        for cancer in cancer_types:
            if cancer in labels:
                idx = labels.index(cancer)
                # Calculate recall (sensitivity) for this cancer type
                if cm[idx].sum() > 0:
                    accuracy = cm[idx, idx] / cm[idx].sum()
                else:
                    accuracy = 0
            else:
                accuracy = np.nan
            method_accuracies.append(accuracy)
        
        accuracy_matrix.append(method_accuracies)
        methods.append(result["method"].upper())
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Convert to numpy array
    accuracy_matrix = np.array(accuracy_matrix)
    
    # Create heatmap
    sns.heatmap(accuracy_matrix,
                xticklabels=cancer_types,
                yticklabels=methods,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Recall (Sensitivity)'},
                linewidths=0.5,
                linecolor='gray')
    
    plt.title('Cancer Type Detection Performance by Fusion Method', fontsize=18, pad=20)
    plt.xlabel('Cancer Type', fontsize=14)
    plt.ylabel('Fusion Method', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "figures", "fusion_methods_cancer_comparison")
    for fmt in ['png', 'pdf', 'svg']:
        filepath = f"{output_path}.{fmt}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.close()
    print(f"\nSaved comparison heatmap: {output_path}")


def main():
    """Generate confusion matrices for all fusion methods."""
    print("Generating confusion matrices for multimodal fusion methods...")
    
    # Ensure output directory exists
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    
    # Generate confusion matrix for each fusion method
    all_results = []
    
    for method in FUSION_METHODS:
        result = generate_fusion_confusion_matrix(method)
        if result:
            all_results.append(result)
    
    # Create comparison heatmap
    if all_results:
        create_comparison_heatmap(all_results)
    
    print("\nAll confusion matrices generated successfully!")
    print(f"Results saved in: {OUTPUT_DIR}/figures/")


if __name__ == "__main__":
    main()