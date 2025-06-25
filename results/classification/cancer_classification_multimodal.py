#!/usr/bin/env python3
"""
Cancer classification using embeddings from multiple modalities with different fusion methods.
This script loads embeddings and runs classification experiments using three fusion methods:
- Concatenation
- Mean Pooling
- Kronecker Product (KP)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to avoid tkinter warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
import os
import random
from tqdm.auto import tqdm
import warnings
import json
from datetime import datetime

# Import data loader
from utils.data_loader import load_embeddings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define constants
N_RUNS = 10
TEST_SIZE = 0.2
N_ESTIMATORS = 100
OUTPUT_DIR = "classification_results_multimodal"
FUSION_METHODS = ["concat", "mean_pool", "kp"]


def create_output_directories():
    """Create output directory structure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    subdirs = ["models", "figures", "results", "reports"]
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
    
    return OUTPUT_DIR


def run_classification_experiment(X, y, modality_name):
    """Run classification experiments with multiple random seeds."""
    # Check if X has more than 2 dimensions and flatten if necessary
    original_shape = X.shape
    if len(X.shape) > 2:
        print(f"Reshaping {modality_name} embeddings from {original_shape} to 2D")
        X = X.reshape(X.shape[0], -1)
        print(f"New shape: {X.shape}")
    
    # Handle NaN values
    if np.isnan(X).any():
        print(f"Replacing NaN values in {modality_name} embeddings")
        X = np.nan_to_num(X)
    
    # Run multiple experiments
    accuracies = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    random_seeds = np.random.randint(0, 10000, size=N_RUNS)
    
    # Initialize variables to store best model and predictions
    best_accuracy = 0
    best_y_test = None
    best_y_pred = None
    best_clf = None
    
    for seed in tqdm(random_seeds, desc=f"Running {modality_name} Classification"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Store results of best run
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_y_test = y_test
            best_y_pred = y_pred
            best_clf = clf
        
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Calculate statistics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)
    mean_recall = np.mean(recall_scores)
    std_recall = np.std(recall_scores)
    
    return {
        "modality": modality_name,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "best_y_test": best_y_test,
        "best_y_pred": best_y_pred,
        "best_clf": best_clf,
        "accuracies": accuracies
    }


def plot_fusion_comparison(fusion_results, output_dir):
    """Create comparison plot for different fusion methods."""
    # Prepare data
    methods = list(fusion_results.keys())
    accuracies = [fusion_results[m]['mean_accuracy'] for m in methods]
    stds = [fusion_results[m]['std_accuracy'] for m in methods]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    
    bars = ax.bar(x, accuracies, yerr=stds, capsize=10, alpha=0.8)
    
    # Color bars differently
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels on bars
    for i, (acc, std) in enumerate(zip(accuracies, stds)):
        ax.text(i, acc + std + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Fusion Method', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Multimodal Fusion Methods Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save figure
    for fmt in ['png', 'pdf', 'svg']:
        filepath = os.path.join(output_dir, 'figures', f'fusion_methods_comparison.{fmt}')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_modality_presence_analysis(embeddings, output_dir):
    """Analyze and plot which modalities are present for each patient."""
    # Get all modalities except multimodal
    modalities = [m for m in embeddings.keys() if m != "multimodal"]
    
    # Create a matrix of patient x modality presence
    all_patients = set()
    for modality in modalities:
        all_patients.update(embeddings[modality]["patient_ids"])
    
    all_patients = sorted(all_patients)
    
    # Create presence matrix
    presence_matrix = np.zeros((len(all_patients), len(modalities)))
    patient_to_idx = {p: i for i, p in enumerate(all_patients)}
    
    for j, modality in enumerate(modalities):
        for patient_id in embeddings[modality]["patient_ids"]:
            if patient_id in patient_to_idx:
                presence_matrix[patient_to_idx[patient_id], j] = 1
    
    # Count combinations
    from collections import Counter
    combinations = []
    for row in presence_matrix:
        combo = tuple(modalities[i] for i in range(len(modalities)) if row[i] == 1)
        if len(combo) >= 2:  # Only count if at least 2 modalities
            combinations.append(",".join(combo))
    
    combo_counts = Counter(combinations)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Modality coverage
    modality_counts = presence_matrix.sum(axis=0)
    ax1.bar(range(len(modalities)), modality_counts)
    ax1.set_xticks(range(len(modalities)))
    ax1.set_xticklabels(modalities, rotation=45)
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Modality Coverage')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Top combinations
    top_combos = combo_counts.most_common(10)
    if top_combos:
        combos, counts = zip(*top_combos)
        y_pos = np.arange(len(combos))
        ax2.barh(y_pos, counts)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(combos)
        ax2.set_xlabel('Number of Patients')
        ax2.set_title('Top 10 Modality Combinations')
        ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save figure
    for fmt in ['png', 'pdf', 'svg']:
        filepath = os.path.join(output_dir, 'figures', f'modality_presence_analysis.{fmt}')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return combo_counts


def generate_comprehensive_report(all_results, combo_counts, output_dir):
    """Generate a comprehensive report comparing all methods."""
    report_lines = []
    report_lines.append("# Multimodal Cancer Classification Results")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Fusion methods comparison
    report_lines.append("## Multimodal Fusion Methods Comparison")
    report_lines.append("")
    
    fusion_results = {k: v for k, v in all_results.items() if k.startswith("multimodal_")}
    if fusion_results:
        best_fusion = max(fusion_results.keys(), key=lambda k: fusion_results[k]['mean_accuracy'])
        report_lines.append(f"Best fusion method: {best_fusion.replace('multimodal_', '').upper()}")
        report_lines.append(f"Best accuracy: {fusion_results[best_fusion]['mean_accuracy']:.4f} "
                           f"(±{fusion_results[best_fusion]['std_accuracy']:.4f})")
        report_lines.append("")
        
        report_lines.append("### Detailed Fusion Results")
        for method in FUSION_METHODS:
            key = f"multimodal_{method}"
            if key in fusion_results:
                result = fusion_results[key]
                report_lines.append(f"\n#### {method.upper()}")
                report_lines.append(f"- Accuracy: {result['mean_accuracy']:.4f} (±{result['std_accuracy']:.4f})")
                report_lines.append(f"- F1-Score: {result['mean_f1']:.4f} (±{result['std_f1']:.4f})")
                report_lines.append(f"- Precision: {result['mean_precision']:.4f} (±{result['std_precision']:.4f})")
                report_lines.append(f"- Recall: {result['mean_recall']:.4f} (±{result['std_recall']:.4f})")
    
    # Individual modalities
    report_lines.append("\n## Individual Modality Results")
    report_lines.append("")
    
    individual_results = {k: v for k, v in all_results.items() if not k.startswith("multimodal")}
    for modality, result in individual_results.items():
        report_lines.append(f"\n### {modality.upper()}")
        report_lines.append(f"- Accuracy: {result['mean_accuracy']:.4f} (±{result['std_accuracy']:.4f})")
        report_lines.append(f"- F1-Score: {result['mean_f1']:.4f} (±{result['std_f1']:.4f})")
    
    # Modality combination statistics
    report_lines.append("\n## Modality Combination Statistics")
    report_lines.append("")
    report_lines.append("Top 10 most common modality combinations:")
    for combo, count in combo_counts.most_common(10):
        report_lines.append(f"- {combo}: {count} patients")
    
    # Save report
    report_path = os.path.join(output_dir, 'reports', 'multimodal_fusion_comparison.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to: {report_path}")
    
    return report_path


def main():
    """Main function to run all classification experiments."""
    print(f"Starting multimodal cancer classification experiments - {datetime.now()}")
    
    # Create output directories
    output_dir = create_output_directories()
    
    # Store all results
    all_results = {}
    
    # Test each fusion method
    fusion_results = {}
    
    for fusion_method in FUSION_METHODS:
        print(f"\n{'='*60}")
        print(f"Testing fusion method: {fusion_method.upper()}")
        print('='*60)
        
        # Load embeddings with specific fusion method
        embeddings = load_embeddings(fusion_method=fusion_method)
        
        if not embeddings:
            print(f"No embeddings found for {fusion_method}!")
            continue
        
        # Run classification for all modalities
        for modality, data in embeddings.items():
            # Use different naming for multimodal results
            if modality == "multimodal":
                result_key = f"multimodal_{fusion_method}"
            else:
                result_key = modality
            
            # Skip individual modalities if we already processed them
            if modality != "multimodal" and modality in all_results:
                continue
            
            print(f"\nRunning classification for {modality} embeddings...")
            print(f"Data shape: {data['X'].shape}")
            print(f"Number of classes: {len(np.unique(data['y']))}")
            
            # Run experiment
            result = run_classification_experiment(data["X"], data["y"], modality)
            all_results[result_key] = result
            
            if modality == "multimodal":
                fusion_results[fusion_method] = result
            
            # Print classification report
            if result["best_y_test"] is not None:
                print(f"\nClassification Report for {modality} ({fusion_method if modality == 'multimodal' else 'N/A'}):")
                print(classification_report(result["best_y_test"], result["best_y_pred"]))
    
    # Analyze modality presence
    print("\nAnalyzing modality presence across patients...")
    embeddings_for_analysis = load_embeddings()  # Load once more for analysis
    combo_counts = plot_modality_presence_analysis(embeddings_for_analysis, output_dir)
    
    # Create comparison plots
    if fusion_results:
        plot_fusion_comparison(fusion_results, output_dir)
    
    # Generate comprehensive report
    report_path = generate_comprehensive_report(all_results, combo_counts, output_dir)
    
    # Save all results as JSON
    results_to_save = {
        key: {
            "mean_accuracy": float(result["mean_accuracy"]),
            "std_accuracy": float(result["std_accuracy"]),
            "mean_f1": float(result["mean_f1"]),
            "std_f1": float(result["std_f1"]),
            "mean_precision": float(result["mean_precision"]),
            "std_precision": float(result["std_precision"]),
            "mean_recall": float(result["mean_recall"]),
            "std_recall": float(result["std_recall"]),
            "accuracies": [float(acc) for acc in result["accuracies"]]
        }
        for key, result in all_results.items()
    }
    
    with open(os.path.join(output_dir, 'results', 'all_fusion_results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("MULTIMODAL FUSION COMPARISON SUMMARY")
    print("="*60)
    
    print("\nFusion Methods:")
    for method in FUSION_METHODS:
        key = f"multimodal_{method}"
        if key in all_results:
            result = all_results[key]
            print(f"\n{method.upper()}:")
            print(f"  Accuracy: {result['mean_accuracy']:.4f} (±{result['std_accuracy']:.4f})")
            print(f"  F1-Score: {result['mean_f1']:.4f} (±{result['std_f1']:.4f})")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"Analysis completed - {datetime.now()}")


if __name__ == "__main__":
    main()