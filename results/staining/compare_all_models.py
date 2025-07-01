#!/usr/bin/env python3
"""
Compare results between UNI, UNI2-h, and Virchow2 stain normalization experiments
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


def create_comparison_plots(results_dict, output_dir):
    """Create comparison plots between all models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract accuracy data
    models = ['logistic_regression', 'random_forest', 'neural_network']
    
    # Create comparison dataframe
    data = []
    for foundation_model, results in results_dict.items():
        if results is None:
            continue
            
        for model in models:
            # With normalization
            if model in results['results_with_normalization']:
                data.append({
                    'Model': model.replace('_', ' ').title(),
                    'Foundation Model': foundation_model,
                    'Normalization': 'With',
                    'Accuracy': results['results_with_normalization'][model]['accuracy']
                })
            # Without normalization
            if model in results['results_without_normalization']:
                data.append({
                    'Model': model.replace('_', ' ').title(),
                    'Foundation Model': foundation_model,
                    'Normalization': 'Without',
                    'Accuracy': results['results_without_normalization'][model]['accuracy']
                })
    
    df = pd.DataFrame(data)
    
    # Plot 1: Comprehensive comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Grouped bar plot
    ax = axes[0, 0]
    pivot_df = df.pivot_table(values='Accuracy', 
                               index=['Foundation Model', 'Model'], 
                               columns='Normalization').reset_index()
    
    x = np.arange(len(pivot_df))
    width = 0.35
    
    ax.bar(x - width/2, pivot_df['With'], width, label='With Normalization', alpha=0.8)
    ax.bar(x + width/2, pivot_df['Without'], width, label='Without Normalization', alpha=0.8)
    
    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across All Models')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['Foundation Model']}\n{row['Model']}" 
                        for _, row in pivot_df.iterrows()], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Heatmap
    ax = axes[0, 1]
    heatmap_data = df.pivot_table(values='Accuracy', 
                                   index=['Foundation Model', 'Model'], 
                                   columns='Normalization')
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0.7, ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_title('Accuracy Heatmap')
    
    # Subplot 3: Impact of normalization
    ax = axes[1, 0]
    
    impact_data = []
    for foundation_model in df['Foundation Model'].unique():
        for model in df['Model'].unique():
            subset = df[(df['Foundation Model'] == foundation_model) & (df['Model'] == model)]
            if len(subset) == 2:
                with_norm = subset[subset['Normalization'] == 'With']['Accuracy'].values[0]
                without_norm = subset[subset['Normalization'] == 'Without']['Accuracy'].values[0]
                impact = with_norm - without_norm
                impact_data.append({
                    'Foundation Model': foundation_model,
                    'Model': model,
                    'Impact': impact
                })
    
    impact_df = pd.DataFrame(impact_data)
    pivot_impact = impact_df.pivot(index='Model', columns='Foundation Model', values='Impact')
    
    pivot_impact.plot(kind='bar', ax=ax, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Accuracy Change')
    ax.set_title('Impact of Stain Normalization by Foundation Model')
    ax.set_xlabel('Classification Model')
    ax.legend(title='Foundation Model')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Average performance
    ax = axes[1, 1]
    avg_performance = df.groupby(['Foundation Model', 'Normalization'])['Accuracy'].mean().unstack()
    avg_performance.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Average Performance by Foundation Model')
    ax.set_xlabel('Foundation Model')
    ax.legend(title='Normalization')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    plt.suptitle('Comprehensive Comparison: UNI vs UNI2-h vs Virchow2', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_models_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Foundation model ranking
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Best accuracy for each foundation model
    best_acc = df.groupby('Foundation Model')['Accuracy'].max()
    best_acc.sort_values(ascending=True).plot(kind='barh', ax=ax1, color='skyblue', alpha=0.8)
    ax1.set_xlabel('Best Accuracy Achieved')
    ax1.set_title('Maximum Accuracy by Foundation Model')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (model, acc) in enumerate(best_acc.sort_values(ascending=True).items()):
        ax1.text(acc, i, f' {acc:.3f}', va='center')
    
    # Average improvement from normalization
    avg_impact = []
    for fm in impact_df['Foundation Model'].unique():
        impacts = impact_df[impact_df['Foundation Model'] == fm]['Impact'].values
        avg_impact.append({
            'Foundation Model': fm,
            'Average Impact': np.mean(impacts),
            'Std Impact': np.std(impacts)
        })
    
    avg_impact_df = pd.DataFrame(avg_impact).sort_values('Average Impact')
    
    ax2.barh(avg_impact_df['Foundation Model'], avg_impact_df['Average Impact'], 
             xerr=avg_impact_df['Std Impact'], alpha=0.8, color='lightcoral', capsize=5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Average Accuracy Improvement from Normalization')
    ax2.set_title('Normalization Sensitivity by Foundation Model')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Foundation Model Performance Characteristics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'foundation_model_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save summary statistics
    summary_stats = pd.DataFrame({
        'Foundation Model': [],
        'Avg Accuracy': [],
        'Best Accuracy': [],
        'Avg With Norm': [],
        'Avg Without Norm': [],
        'Normalization Impact': [],
        'Consistency (Std)': []
    })
    
    for fm in df['Foundation Model'].unique():
        subset = df[df['Foundation Model'] == fm]
        with_norm = subset[subset['Normalization'] == 'With']['Accuracy'].values
        without_norm = subset[subset['Normalization'] == 'Without']['Accuracy'].values
        
        stats = {
            'Foundation Model': fm,
            'Avg Accuracy': subset['Accuracy'].mean(),
            'Best Accuracy': subset['Accuracy'].max(),
            'Avg With Norm': with_norm.mean() if len(with_norm) > 0 else np.nan,
            'Avg Without Norm': without_norm.mean() if len(without_norm) > 0 else np.nan,
            'Normalization Impact': (with_norm.mean() - without_norm.mean()) if len(with_norm) > 0 and len(without_norm) > 0 else np.nan,
            'Consistency (Std)': subset['Accuracy'].std()
        }
        summary_stats = pd.concat([summary_stats, pd.DataFrame([stats])], ignore_index=True)
    
    summary_stats.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False)
    
    # Save detailed results
    df.to_csv(os.path.join(output_dir, 'all_results_detailed.csv'), index=False)
    
    print("\nSummary Statistics:")
    print(summary_stats.to_string(index=False))
    
    return df, summary_stats


def main():
    # Directories
    uni_dir = '/mnt/f/Projects/HoneyBee/results/staining'
    uni2_dir = '/mnt/f/Projects/HoneyBee/results/staining_uni2'
    virchow2_dir = '/mnt/f/Projects/HoneyBee/results/staining_virchow2'
    output_dir = '/mnt/f/Projects/HoneyBee/results/staining_all_models_comparison'
    
    # Load all results
    print("Loading results...")
    results_dict = {
        'UNI': load_results(uni_dir),
        'UNI2-h': load_results(uni2_dir),
        'Virchow2': load_results(virchow2_dir)
    }
    
    # Check which results are available
    available_models = [k for k, v in results_dict.items() if v is not None]
    print(f"\nAvailable models: {', '.join(available_models)}")
    
    if len(available_models) < 2:
        print("Need at least 2 models for comparison. Please run the missing experiments.")
        return
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    df, summary_stats = create_comparison_plots(results_dict, output_dir)
    
    print(f"\nComparison complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()