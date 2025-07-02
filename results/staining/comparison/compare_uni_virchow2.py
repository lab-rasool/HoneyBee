#!/usr/bin/env python3
"""
Compare results between UNI and Virchow2 stain normalization experiments
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths to handle pickle loading
sys.path.append('/mnt/f/Projects/HoneyBee/results/staining')

# Define the SimpleClassifier class to handle unpickling
class SimpleClassifier(nn.Module):
    """Simple neural network classifier"""
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def load_results(results_dir):
    """Load results from a directory"""
    results_file = os.path.join(results_dir, 'results.pkl')
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    return results


def create_comparison_plots(uni_results, virchow2_results, output_dir):
    """Create comparison plots between UNI and Virchow2"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract accuracy data
    models = ['logistic_regression', 'random_forest', 'neural_network']
    
    # Create comparison dataframe
    data = []
    for model in models:
        # UNI results
        if model in uni_results['results_with_normalization']:
            data.append({
                'Model': model.replace('_', ' ').title(),
                'Foundation Model': 'UNI',
                'Normalization': 'With',
                'Accuracy': uni_results['results_with_normalization'][model]['accuracy']
            })
            data.append({
                'Model': model.replace('_', ' ').title(),
                'Foundation Model': 'UNI',
                'Normalization': 'Without',
                'Accuracy': uni_results['results_without_normalization'][model]['accuracy']
            })
        
        # Virchow2 results
        if model in virchow2_results['results_with_normalization']:
            data.append({
                'Model': model.replace('_', ' ').title(),
                'Foundation Model': 'Virchow2',
                'Normalization': 'With',
                'Accuracy': virchow2_results['results_with_normalization'][model]['accuracy']
            })
            data.append({
                'Model': model.replace('_', ' ').title(),
                'Foundation Model': 'Virchow2',
                'Normalization': 'Without',
                'Accuracy': virchow2_results['results_without_normalization'][model]['accuracy']
            })
    
    df = pd.DataFrame(data)
    
    # Plot 1: Grouped bar plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Create grouped bar plot
    x = np.arange(len(models))
    width = 0.2
    
    # Plot bars
    for i, (fm, norm) in enumerate([('UNI', 'With'), ('UNI', 'Without'), 
                                     ('Virchow2', 'With'), ('Virchow2', 'Without')]):
        subset = df[(df['Foundation Model'] == fm) & (df['Normalization'] == norm)]
        if not subset.empty:
            accuracies = [subset[subset['Model'] == m.replace('_', ' ').title()]['Accuracy'].values[0] 
                         if len(subset[subset['Model'] == m.replace('_', ' ').title()]) > 0 else 0
                         for m in models]
            
            offset = (i - 1.5) * width
            color = plt.cm.Paired(i * 3)
            label = f'{fm} - {norm} Norm'
            ax.bar(x + offset, accuracies, width, label=label, alpha=0.8, color=color)
    
    ax.set_xlabel('Classification Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Comparison of UNI vs Virchow2 with Stain Normalization', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for i, rect_group in enumerate(ax.patches):
        height = rect_group.get_height()
        if height > 0:
            ax.text(rect_group.get_x() + rect_group.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uni_vs_virchow2_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Improvement comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    improvements = []
    labels = []
    colors = []
    
    for model in models:
        # UNI improvement
        if model in uni_results['results_with_normalization']:
            uni_with = uni_results['results_with_normalization'][model]['accuracy']
            uni_without = uni_results['results_without_normalization'][model]['accuracy']
            uni_improvement = uni_with - uni_without
            improvements.append(uni_improvement)
            labels.append(f'UNI - {model.replace("_", " ").title()}')
            colors.append('#FF6B6B' if uni_improvement < 0 else '#4ECDC4')
        
        # Virchow2 improvement
        if model in virchow2_results['results_with_normalization']:
            v2_with = virchow2_results['results_with_normalization'][model]['accuracy']
            v2_without = virchow2_results['results_without_normalization'][model]['accuracy']
            v2_improvement = v2_with - v2_without
            improvements.append(v2_improvement)
            labels.append(f'Virchow2 - {model.replace("_", " ").title()}')
            colors.append('#FFB6B6' if v2_improvement < 0 else '#4ED4DC')
    
    # Create bar plot
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, improvements, color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Accuracy Improvement from Stain Normalization')
    ax.set_title('Impact of Stain Normalization: UNI vs Virchow2')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (improvement, label) in enumerate(zip(improvements, labels)):
        if improvement != 0:
            ax.text(improvement, i, f'{improvement:+.3f}', 
                   ha='left' if improvement > 0 else 'right',
                   va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalization_impact_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Summary statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Average accuracy by foundation model
    avg_accuracies = df.groupby(['Foundation Model', 'Normalization'])['Accuracy'].mean().unstack()
    avg_accuracies.plot(kind='bar', ax=ax1, alpha=0.8)
    ax1.set_title('Average Accuracy by Foundation Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Foundation Model')
    ax1.legend(title='Normalization')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Best model for each foundation model
    best_models = df.groupby('Foundation Model')['Accuracy'].max()
    best_models.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
    ax2.set_title('Best Accuracy by Foundation Model')
    ax2.set_ylabel('Best Accuracy')
    ax2.set_xlabel('Foundation Model')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(best_models):
        ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Heatmap of all results
    pivot_data = df.pivot_table(values='Accuracy', 
                                index=['Foundation Model', 'Model'], 
                                columns='Normalization')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0.5, ax=ax3, cbar_kws={'label': 'Accuracy'})
    ax3.set_title('Accuracy Heatmap')
    
    # Summary text
    ax4.axis('off')
    summary_text = "Summary Statistics:\n\n"
    
    # Calculate overall statistics
    uni_avg = df[df['Foundation Model'] == 'UNI']['Accuracy'].mean()
    virchow2_avg = df[df['Foundation Model'] == 'Virchow2']['Accuracy'].mean()
    
    uni_with_norm = df[(df['Foundation Model'] == 'UNI') & 
                       (df['Normalization'] == 'With')]['Accuracy'].mean()
    uni_without_norm = df[(df['Foundation Model'] == 'UNI') & 
                          (df['Normalization'] == 'Without')]['Accuracy'].mean()
    
    virchow2_with_norm = df[(df['Foundation Model'] == 'Virchow2') & 
                            (df['Normalization'] == 'With')]['Accuracy'].mean()
    virchow2_without_norm = df[(df['Foundation Model'] == 'Virchow2') & 
                               (df['Normalization'] == 'Without')]['Accuracy'].mean()
    
    summary_text += f"UNI Average Accuracy: {uni_avg:.3f}\n"
    summary_text += f"Virchow2 Average Accuracy: {virchow2_avg:.3f}\n\n"
    
    summary_text += f"UNI with Normalization: {uni_with_norm:.3f}\n"
    summary_text += f"UNI without Normalization: {uni_without_norm:.3f}\n"
    summary_text += f"UNI Normalization Impact: {uni_with_norm - uni_without_norm:+.3f}\n\n"
    
    summary_text += f"Virchow2 with Normalization: {virchow2_with_norm:.3f}\n"
    summary_text += f"Virchow2 without Normalization: {virchow2_without_norm:.3f}\n"
    summary_text += f"Virchow2 Normalization Impact: {virchow2_with_norm - virchow2_without_norm:+.3f}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    ax4.set_title('Summary Statistics')
    
    plt.suptitle('UNI vs Virchow2: Comprehensive Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    df.to_csv(os.path.join(output_dir, 'comparison_data.csv'), index=False)
    
    print(f"Comparison plots saved to {output_dir}")
    return df


def main():
    # Directories
    uni_dir = '/mnt/f/Projects/HoneyBee/results/staining'
    virchow2_dir = '/mnt/f/Projects/HoneyBee/results/staining_virchow2'
    output_dir = '/mnt/f/Projects/HoneyBee/results/staining_comparison'
    
    # Load results
    print("Loading UNI results...")
    uni_results = load_results(uni_dir)
    
    print("Loading Virchow2 results...")
    virchow2_results = load_results(virchow2_dir)
    
    if uni_results is None:
        print("UNI results not found. Please run stain_normalization_comparison.py first.")
        return
    
    if virchow2_results is None:
        print("Virchow2 results not found. Please run stain_normalization_comparison_virchow2.py first.")
        return
    
    # Create comparison plots
    print("Creating comparison plots...")
    df = create_comparison_plots(uni_results, virchow2_results, output_dir)
    
    print("\nComparison complete!")
    print(f"\nResults summary:")
    print(df.groupby(['Foundation Model', 'Normalization'])['Accuracy'].agg(['mean', 'std']))


if __name__ == "__main__":
    main()