#!/usr/bin/env python3
"""
Retrieval analysis for multimodal embeddings with different fusion methods.
"""

import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, '/mnt/f/Projects/HoneyBee/results/classification')
from utils.data_loader import load_embeddings as load_classification_embeddings
sys.path.pop(0)

# Import retrieval functions from the main script
from run_all_retrieval_analyses import (
    compute_precision_at_k, compute_stable_precision_at_k,
    compute_ami_scores, analyze_failures,
    K_VALUES, REPORT_K_VALUES, N_RUNS, OUTPUT_DIR
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set random seed
np.random.seed(42)

FUSION_METHODS = ["concat", "mean_pool", "kp"]
FUSION_OUTPUT_DIR = "unified_retrieval_results/multimodal"


def run_multimodal_analysis(fusion_method):
    """Run retrieval analysis for a specific fusion method."""
    print(f"\n{'='*60}")
    print(f"Processing multimodal ({fusion_method}) embeddings")
    print('='*60)
    
    # Load embeddings with specific fusion method
    embeddings_dict = load_classification_embeddings(fusion_method=fusion_method)
    
    if "multimodal" not in embeddings_dict:
        print(f"✗ No multimodal embeddings found for {fusion_method}")
        return None
    
    data = embeddings_dict["multimodal"]
    X = data['X']
    y = data['y']
    
    # Handle NaN values
    if np.isnan(X).any():
        print("  Warning: Found NaN values, replacing with 0")
        X = np.nan_to_num(X)
    
    print(f"✓ Loaded embeddings: {X.shape}")
    print(f"  Number of cancer types: {len(np.unique(y))}")
    
    results = {
        'data_shape': X.shape,
        'n_classes': len(np.unique(y))
    }
    
    try:
        # 1. Basic retrieval analysis
        print("\n1. Running basic retrieval analysis...")
        mean_precisions, std_precisions = compute_precision_at_k(X, y, K_VALUES, N_RUNS)
        results['basic_retrieval'] = {
            'mean': mean_precisions,
            'std': std_precisions
        }
        print(f"   Precision@10: {mean_precisions[10]:.4f} ± {std_precisions[10]:.4f}")
        
        # 2. AMI analysis (use smaller sample for speed)
        print("\n2. Running AMI analysis...")
        sample_size = min(1000, len(y))
        sample_idx = np.random.choice(len(y), sample_size, replace=False)
        ami_results = compute_ami_scores(X[sample_idx], y[sample_idx])
        results['ami'] = ami_results
        print(f"   Clustering AMI: {ami_results['clustering_ami']:.4f}")
        print(f"   Retrieval AMI@10: {ami_results['retrieval_ami'][10]:.4f}")
        
        # 3. Failure analysis
        print("\n3. Running failure analysis...")
        failure_stats, failure_df = analyze_failures(X, y, k=10)
        results['failure_analysis'] = failure_stats
        print(f"   Failure rate: {failure_stats['failure_rate']:.4f}")
        
        # 4. Create visualizations
        print("\n4. Creating visualizations...")
        # Visualizations will be created in the comparison plots
        
    except Exception as e:
        print(f"✗ Error processing multimodal ({fusion_method}): {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    return results


def create_fusion_comparison_plots(all_results):
    """Create comparison plots for different fusion methods."""
    print("\nCreating fusion method comparison plots...")
    
    # 1. Precision@k comparison
    plt.figure(figsize=(12, 8))
    
    colors = {'concat': '#1f77b4', 'mean_pool': '#ff7f0e', 'kp': '#2ca02c'}
    
    for method, results in all_results.items():
        if results and 'basic_retrieval' in results:
            k_values = sorted(results['basic_retrieval']['mean'].keys())
            mean_values = [results['basic_retrieval']['mean'][k] for k in k_values]
            plt.plot(k_values, mean_values, linewidth=2, 
                    label=method.upper(), color=colors.get(method, 'gray'))
    
    plt.xlabel('k', fontsize=12)
    plt.ylabel('Precision@k', fontsize=12)
    plt.title('Multimodal Fusion Methods - Retrieval Performance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    output_path = os.path.join(FUSION_OUTPUT_DIR, 'figures', 'fusion_methods_precision_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot comparison for key metrics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    methods = []
    p_at_10 = []
    ami_scores = []
    failure_rates = []
    
    for method, results in all_results.items():
        if results:
            methods.append(method.upper())
            p_at_10.append(results['basic_retrieval']['mean'][10] if 'basic_retrieval' in results else 0)
            ami_scores.append(results['ami']['clustering_ami'] if 'ami' in results else 0)
            failure_rates.append(results['failure_analysis']['failure_rate'] if 'failure_analysis' in results else 0)
    
    x = np.arange(len(methods))
    
    # Precision@10
    bars1 = ax1.bar(x, p_at_10, color=[colors.get(m.lower(), 'gray') for m in methods])
    ax1.set_ylabel('Precision@10', fontsize=12)
    ax1.set_title('Retrieval Precision@10', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for i, v in enumerate(p_at_10):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # AMI scores
    bars2 = ax2.bar(x, ami_scores, color=[colors.get(m.lower(), 'gray') for m in methods])
    ax2.set_ylabel('AMI Score', fontsize=12)
    ax2.set_title('Clustering AMI', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(ami_scores):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Failure rates
    bars3 = ax3.bar(x, failure_rates, color=[colors.get(m.lower(), 'gray') for m in methods])
    ax3.set_ylabel('Failure Rate', fontsize=12)
    ax3.set_title('Retrieval Failure Rate', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.0)
    
    # Add value labels
    for i, v in enumerate(failure_rates):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Multimodal Fusion Methods - Retrieval Analysis Comparison', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(FUSION_OUTPUT_DIR, 'figures', 'fusion_methods_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plots to {FUSION_OUTPUT_DIR}/figures/")


def generate_fusion_report(all_results):
    """Generate a comprehensive report for multimodal fusion retrieval results."""
    report_path = os.path.join(FUSION_OUTPUT_DIR, 'reports', 'multimodal_fusion_retrieval_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("MULTIMODAL FUSION RETRIEVAL ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("SUMMARY RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Fusion Method':<15} {'Samples':<10} {'Classes':<10} "
               f"{'P@10':<15} {'AMI':<12} {'Failure%':<12}\n")
        f.write("-"*80 + "\n")
        
        for method in FUSION_METHODS:
            if method in all_results and all_results[method]:
                results = all_results[method]
                p_at_10 = results['basic_retrieval']['mean'][10]
                p_at_10_std = results['basic_retrieval']['std'][10]
                ami = results['ami']['clustering_ami'] if 'ami' in results else 0
                failure_rate = results['failure_analysis']['failure_rate'] * 100 if 'failure_analysis' in results else 0
                
                f.write(f"{method.upper():<15} "
                       f"{results['data_shape'][0]:<10} "
                       f"{results['n_classes']:<10} "
                       f"{p_at_10:.3f}±{p_at_10_std:.3f}  "
                       f"{ami:<12.3f} "
                       f"{failure_rate:<12.1f}\n")
        
        # Detailed results
        f.write("\n\nDETAILED RESULTS BY FUSION METHOD\n")
        f.write("="*80 + "\n")
        
        for method in FUSION_METHODS:
            if method not in all_results or not all_results[method]:
                continue
                
            results = all_results[method]
            f.write(f"\n{method.upper()}\n")
            f.write("-"*40 + "\n")
            
            # Data characteristics
            f.write(f"Data shape: {results['data_shape']}\n")
            f.write(f"Number of classes: {results['n_classes']}\n")
            
            # Retrieval performance
            if 'basic_retrieval' in results:
                f.write(f"\nRetrieval Performance:\n")
                for k in REPORT_K_VALUES:
                    mean = results['basic_retrieval']['mean'][k]
                    std = results['basic_retrieval']['std'][k]
                    f.write(f"  Precision@{k}: {mean:.4f} ± {std:.4f}\n")
            
            # AMI results
            if 'ami' in results:
                f.write(f"\nClustering Quality:\n")
                f.write(f"  Clustering AMI: {results['ami']['clustering_ami']:.4f}\n")
                f.write(f"  Clustering NMI: {results['ami']['clustering_nmi']:.4f}\n")
                f.write(f"  Retrieval-based AMI:\n")
                for k in REPORT_K_VALUES:
                    ami = results['ami']['retrieval_ami'][k]
                    acc = results['ami']['retrieval_acc'][k]
                    f.write(f"    k={k}: AMI={ami:.4f}, Acc={acc:.4f}\n")
            
            # Failure analysis
            if 'failure_analysis' in results:
                f.write(f"\nFailure Analysis:\n")
                f.write(f"  Total failures: {results['failure_analysis']['total_failures']}\n")
                f.write(f"  Failure rate: {results['failure_analysis']['failure_rate']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
    
    print(f"✓ Report saved to: {report_path}")


def main():
    """Run multimodal retrieval analysis for all fusion methods."""
    print("Starting multimodal retrieval analysis...")
    
    # Create output directories
    os.makedirs(FUSION_OUTPUT_DIR, exist_ok=True)
    for subdir in ['figures', 'data', 'reports']:
        os.makedirs(os.path.join(FUSION_OUTPUT_DIR, subdir), exist_ok=True)
    
    # Run analysis for each fusion method
    all_results = {}
    
    for fusion_method in FUSION_METHODS:
        results = run_multimodal_analysis(fusion_method)
        if results:
            all_results[fusion_method] = results
    
    # Create comparison visualizations
    if all_results:
        create_fusion_comparison_plots(all_results)
        generate_fusion_report(all_results)
    
    print("\n✓ Multimodal retrieval analysis complete!")
    print(f"Results saved to: {FUSION_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()