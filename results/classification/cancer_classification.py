#!/usr/bin/env python3
"""
Cancer classification using embeddings from multiple modalities.
This script loads embeddings and runs classification experiments.
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
OUTPUT_DIR = "classification_results"


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


def plot_confusion_matrix(y_true, y_pred, labels, modality, output_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {modality}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure
    for fmt in ['png', 'pdf', 'svg']:
        filepath = os.path.join(output_dir, 'figures', f'{modality}_confusion_matrix.{fmt}')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return cm

def plot_performance_comparison(results, output_dir):
    """Create bar plot comparing performance across modalities."""
    # Prepare data for plotting
    modalities = list(results.keys())
    metrics = {
        'Accuracy': [results[m]['mean_accuracy'] for m in modalities],
        'F1-Score': [results[m]['mean_f1'] for m in modalities],
        'Precision': [results[m]['mean_precision'] for m in modalities],
        'Recall': [results[m]['mean_recall'] for m in modalities]
    }
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(modalities))
    width = 0.2
    multiplier = 0
    
    for metric, values in metrics.items():
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=metric)
        
        # Add error bars for accuracy
        if metric == 'Accuracy':
            errors = [results[m]['std_accuracy'] for m in modalities]
            ax.errorbar(x + offset, values, yerr=errors, fmt='none', color='black', capsize=5)
        
        multiplier += 1
    
    ax.set_xlabel('Modality')
    ax.set_ylabel('Score')
    ax.set_title('Classification Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(modalities)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save figure
    for fmt in ['png', 'pdf', 'svg']:
        filepath = os.path.join(output_dir, 'figures', f'performance_comparison.{fmt}')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    plt.close()


def generate_classification_report_text(results, output_dir):
    """Generate a comprehensive text report of classification results."""
    report_lines = []
    report_lines.append("# Cancer Classification Results Report")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overall summary
    report_lines.append("## Overall Performance Summary")
    report_lines.append("")
    
    best_modality = max(results.keys(), key=lambda k: results[k]['mean_accuracy'])
    report_lines.append(f"Best performing modality: {best_modality}")
    report_lines.append(f"Best accuracy: {results[best_modality]['mean_accuracy']:.4f} "
                       f"(±{results[best_modality]['std_accuracy']:.4f})")
    report_lines.append("")
    
    # Detailed results by modality
    report_lines.append("## Detailed Results by Modality")
    report_lines.append("")
    
    for modality, result in results.items():
        report_lines.append(f"### {modality.upper()}")
        report_lines.append(f"- Accuracy: {result['mean_accuracy']:.4f} (±{result['std_accuracy']:.4f})")
        report_lines.append(f"- F1-Score: {result['mean_f1']:.4f} (±{result['std_f1']:.4f})")
        report_lines.append(f"- Precision: {result['mean_precision']:.4f} (±{result['std_precision']:.4f})")
        report_lines.append(f"- Recall: {result['mean_recall']:.4f} (±{result['std_recall']:.4f})")
        report_lines.append("")
    
    # Save report
    report_path = os.path.join(output_dir, 'reports', 'classification_summary.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to: {report_path}")
    
    return report_path


def main():
    """Main function to run all classification experiments."""
    print(f"Starting cancer classification experiments - {datetime.now()}")
    
    # Create output directories
    output_dir = create_output_directories()
    
    # Load embeddings
    embeddings = load_embeddings()
    
    if not embeddings:
        print("No embeddings found! Please ensure embeddings are available.")
        return
    
    # Run classification experiments for all modalities
    results = {}
    confusion_matrices = {}
    
    for modality, data in embeddings.items():
        print(f"\nRunning classification for {modality} embeddings...")
        print(f"Data shape: {data['X'].shape}")
        print(f"Number of classes: {len(np.unique(data['y']))}")
        
        # Run experiment
        result = run_classification_experiment(data["X"], data["y"], modality)
        results[modality] = result
        
        # Create confusion matrix
        if result["best_y_test"] is not None:
            unique_labels = np.unique(data["y"])
            cm = plot_confusion_matrix(
                result["best_y_test"], 
                result["best_y_pred"], 
                unique_labels,
                modality,
                output_dir
            )
            confusion_matrices[modality] = {
                "matrix": cm.tolist(),
                "labels": unique_labels.tolist()
            }
            
            # Print classification report
            print(f"\nClassification Report for {modality}:")
            print(classification_report(result["best_y_test"], result["best_y_pred"]))
            
    # Save results
    results_to_save = {
        modality: {
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
        for modality, result in results.items()
    }
    
    # Save classification results
    with open(os.path.join(output_dir, 'results', 'classification_results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Save confusion matrices
    with open(os.path.join(output_dir, 'results', 'confusion_matrices.json'), 'w') as f:
        json.dump(confusion_matrices, f, indent=2)
    
    # Create performance comparison plot
    plot_performance_comparison(results, output_dir)
    
    # Generate report
    report_path = generate_classification_report_text(results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    
    for modality, result in results.items():
        print(f"\n{modality.upper()}:")
        print(f"  Accuracy: {result['mean_accuracy']:.4f} (±{result['std_accuracy']:.4f})")
        print(f"  F1-Score: {result['mean_f1']:.4f} (±{result['std_f1']:.4f})")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"Analysis completed - {datetime.now()}")


if __name__ == "__main__":
    main()