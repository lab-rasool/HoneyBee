#!/usr/bin/env python3
"""
Unified retrieval analysis script that runs all retrieval benchmarks and analyses.
This combines retrieval benchmarking, AMI analysis, and failure analysis into a single workflow.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from scipy import stats
from tqdm.auto import tqdm
import os
import json
import warnings
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# Import data loader
from utils.data_loader import load_saved_embeddings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Define constants
OUTPUT_DIR = "unified_retrieval_results"
N_RUNS = 3  # Number of runs for basic retrieval
N_FOLDS = 5  # Number of folds for stable analysis
N_BOOTSTRAP = 1000  # Number of bootstrap samples
SAMPLE_SIZE = 1000  # Sample size for fast AMI analysis
K_VALUES = list(range(1, 51))  # k values for retrieval
REPORT_K_VALUES = [5, 10, 20]  # k values to report in detail


def create_output_directories():
    """Create unified output directory structure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    subdirs = ["figures", "data", "reports"]
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
    
    print(f"✓ Created output directory: {OUTPUT_DIR}/")


def compute_precision_at_k(embeddings, labels, k_values=K_VALUES, n_runs=N_RUNS):
    """Compute precision@k for retrieval task with multiple runs."""
    n_samples = len(labels)
    all_precisions = []
    
    # Ensure labels is a numpy array
    labels = np.array(labels)
    
    for run in range(n_runs):
        # Shuffle data for this run
        shuffle_idx = np.random.permutation(n_samples)
        embeddings_shuffled = embeddings[shuffle_idx]
        labels_shuffled = labels[shuffle_idx]
        
        # Normalize embeddings
        embeddings_norm = normalize(embeddings_shuffled, norm='l2')
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings_norm)
        
        # Set diagonal to -inf to exclude self-similarity
        np.fill_diagonal(similarity_matrix, -np.inf)
        
        # Compute precision at different k values
        precisions = {k: [] for k in k_values}
        
        for i in range(n_samples):
            # Get k nearest neighbors
            neighbors = np.argsort(similarity_matrix[i])[-max(k_values):][::-1]
            
            for k in k_values:
                # Check if neighbors have same label
                k_neighbors = neighbors[:k]
                same_label = labels_shuffled[k_neighbors] == labels_shuffled[i]
                precision = np.mean(same_label)
                precisions[k].append(precision)
        
        # Average precision for this run
        run_precisions = {k: np.mean(precisions[k]) for k in k_values}
        all_precisions.append(run_precisions)
    
    # Compute mean and std across runs
    mean_precisions = {}
    std_precisions = {}
    
    for k in k_values:
        k_values_across_runs = [run[k] for run in all_precisions]
        mean_precisions[k] = np.mean(k_values_across_runs)
        std_precisions[k] = np.std(k_values_across_runs)
    
    return mean_precisions, std_precisions


def compute_stable_precision_at_k(embeddings, labels, k_values=K_VALUES, n_folds=N_FOLDS, n_bootstrap=N_BOOTSTRAP):
    """Compute precision@k using stratified CV and bootstrap for stability."""
    n_samples = len(labels)
    labels = np.array(labels)
    embeddings_norm = normalize(embeddings, norm='l2')
    
    # Store per-sample precisions for bootstrap
    sample_precisions = {k: [] for k in k_values}
    
    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        # Get test set
        test_embeddings = embeddings_norm[test_idx]
        test_labels = labels[test_idx]
        
        # Compute similarity for test samples against all samples
        similarity_matrix = cosine_similarity(test_embeddings, embeddings_norm)
        
        # For each test sample
        for i, test_sample_idx in enumerate(test_idx):
            # Exclude self from neighbors
            similarities = similarity_matrix[i].copy()
            similarities[test_sample_idx] = -np.inf
            
            # Get k nearest neighbors
            neighbors = np.argsort(similarities)[-max(k_values):][::-1]
            
            for k in k_values:
                k_neighbors = neighbors[:k]
                same_label = labels[k_neighbors] == test_labels[i]
                precision = np.mean(same_label)
                sample_precisions[k].append(precision)
    
    # Convert to arrays for bootstrap
    sample_precisions = {k: np.array(v) for k, v in sample_precisions.items()}
    
    # Bootstrap for confidence intervals
    bootstrap_means = {k: [] for k in k_values}
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resample_idx = resample(np.arange(len(sample_precisions[1])), replace=True)
        
        for k in k_values:
            bootstrap_mean = np.mean(sample_precisions[k][resample_idx])
            bootstrap_means[k].append(bootstrap_mean)
    
    # Compute statistics
    results = {}
    for k in k_values:
        bootstrap_dist = np.array(bootstrap_means[k])
        results[k] = {
            'mean': np.mean(sample_precisions[k]),
            'std': np.std(sample_precisions[k]),
            'ci_lower': np.percentile(bootstrap_dist, 2.5),
            'ci_upper': np.percentile(bootstrap_dist, 97.5)
        }
    
    return results


def compute_ami_scores(embeddings, labels, sample_size=SAMPLE_SIZE):
    """Compute AMI scores for clustering quality assessment."""
    n_samples = len(labels)
    labels = np.array(labels)
    
    # Sample data if too large
    if n_samples > sample_size:
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        embeddings_sampled = embeddings[sample_idx]
        labels_sampled = labels[sample_idx]
    else:
        embeddings_sampled = embeddings
        labels_sampled = labels
    
    # Normalize embeddings
    embeddings_norm = normalize(embeddings_sampled, norm='l2')
    
    # Clustering AMI
    n_clusters = len(np.unique(labels_sampled))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_norm)
    clustering_ami = adjusted_mutual_info_score(labels_sampled, cluster_labels)
    clustering_nmi = normalized_mutual_info_score(labels_sampled, cluster_labels)
    
    # Retrieval-based AMI
    similarity_matrix = cosine_similarity(embeddings_norm)
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    retrieval_ami = {}
    retrieval_acc = {}
    
    for k in REPORT_K_VALUES:
        pseudo_labels = []
        
        for i in range(len(labels_sampled)):
            neighbors = np.argsort(similarity_matrix[i])[-k:][::-1]
            neighbor_labels = labels_sampled[neighbors]
            # Majority vote
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            pseudo_labels.append(most_common)
        
        pseudo_labels = np.array(pseudo_labels)
        retrieval_ami[k] = adjusted_mutual_info_score(labels_sampled, pseudo_labels)
        retrieval_acc[k] = np.mean(pseudo_labels == labels_sampled)
    
    return {
        'clustering_ami': clustering_ami,
        'clustering_nmi': clustering_nmi,
        'retrieval_ami': retrieval_ami,
        'retrieval_acc': retrieval_acc
    }


def analyze_failures(embeddings, labels, k=10):
    """Analyze retrieval failures and confusion patterns."""
    n_samples = len(labels)
    labels = np.array(labels)
    embeddings_norm = normalize(embeddings, norm='l2')
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings_norm)
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    # Track failures
    failures = []
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for i in range(n_samples):
        neighbors = np.argsort(similarity_matrix[i])[-k:][::-1]
        neighbor_labels = labels[neighbors]
        
        # Check each neighbor
        for j, neighbor_idx in enumerate(neighbors):
            if labels[neighbor_idx] != labels[i]:
                failures.append({
                    'query_idx': i,
                    'query_label': labels[i],
                    'retrieved_idx': neighbor_idx,
                    'retrieved_label': labels[neighbor_idx],
                    'rank': j + 1,
                    'similarity': similarity_matrix[i, neighbor_idx]
                })
                confusion_matrix[labels[i]][labels[neighbor_idx]] += 1
    
    # Convert to DataFrame for analysis
    failures_df = pd.DataFrame(failures)
    
    # Compute failure statistics
    failure_stats = {
        'total_failures': len(failures),
        'failure_rate': len(failures) / (n_samples * k),
        'failures_by_query_label': failures_df.groupby('query_label').size().to_dict() if len(failures) > 0 else {},
        'failures_by_retrieved_label': failures_df.groupby('retrieved_label').size().to_dict() if len(failures) > 0 else {},
        'confusion_matrix': dict(confusion_matrix)
    }
    
    return failure_stats, failures_df


def create_visualizations(results, modality_name):
    """Create comprehensive visualizations for a modality."""
    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    
    # 1. Precision@k curve
    plt.figure(figsize=(10, 6))
    k_values = sorted(results['basic_retrieval']['mean'].keys())
    mean_values = [results['basic_retrieval']['mean'][k] for k in k_values]
    std_values = [results['basic_retrieval']['std'][k] for k in k_values]
    
    plt.plot(k_values, mean_values, 'b-', linewidth=2, label=f'{modality_name}')
    plt.fill_between(k_values, 
                     np.array(mean_values) - np.array(std_values),
                     np.array(mean_values) + np.array(std_values),
                     alpha=0.3)
    
    plt.xlabel('k')
    plt.ylabel('Precision@k')
    plt.title(f'Retrieval Precision@k - {modality_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{modality_name.lower()}_precision_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Stability comparison (if stable results available)
    if 'stable_retrieval' in results:
        plt.figure(figsize=(12, 6))
        
        # Compare basic vs stable for report k values
        x = np.arange(len(REPORT_K_VALUES))
        width = 0.35
        
        basic_means = [results['basic_retrieval']['mean'][k] for k in REPORT_K_VALUES]
        basic_stds = [results['basic_retrieval']['std'][k] for k in REPORT_K_VALUES]
        
        stable_means = [results['stable_retrieval'][k]['mean'] for k in REPORT_K_VALUES]
        stable_cis = [(results['stable_retrieval'][k]['ci_upper'] - results['stable_retrieval'][k]['ci_lower'])/2 
                      for k in REPORT_K_VALUES]
        
        plt.bar(x - width/2, basic_means, width, yerr=basic_stds, 
                label='Basic (±std)', capsize=5)
        plt.bar(x + width/2, stable_means, width, yerr=stable_cis,
                label='Stable (95% CI)', capsize=5)
        
        plt.xlabel('k')
        plt.ylabel('Precision@k')
        plt.title(f'Basic vs Stable Retrieval - {modality_name}')
        plt.xticks(x, REPORT_K_VALUES)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{modality_name.lower()}_stability_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


def run_all_analyses():
    """Run all retrieval analyses for all modalities."""
    create_output_directories()
    
    # Define modalities to analyze
    modalities = ['clinical', 'pathology', 'radiology', 'molecular', 'wsi']
    
    # Store all results
    all_results = {}
    
    # Process each modality
    for modality in modalities:
        print(f"\n{'='*60}")
        print(f"Processing {modality} modality")
        print('='*60)
        
        try:
            # Load embeddings
            print(f"Loading {modality} embeddings...")
            data = load_saved_embeddings(modality)
            
            if data is None:
                print(f"✗ Failed to load {modality} embeddings")
                continue
            
            X, y, patient_ids = data['X'], data['y'], data['patient_ids']
            
            # Handle NaN values
            if np.isnan(X).any():
                print("  Warning: Found NaN values, replacing with 0")
                X = np.nan_to_num(X)
            
            print(f"✓ Loaded embeddings: {X.shape}")
            print(f"  Number of cancer types: {len(np.unique(y))}")
            
            # Initialize results dictionary
            modality_results = {
                'data_shape': X.shape,
                'n_classes': len(np.unique(y)),
                'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
            }
            
            # 1. Basic retrieval analysis
            print(f"\n1. Running basic retrieval analysis...")
            mean_prec, std_prec = compute_precision_at_k(X, y, n_runs=N_RUNS)
            modality_results['basic_retrieval'] = {
                'mean': mean_prec,
                'std': std_prec
            }
            print(f"   Precision@10: {mean_prec[10]:.4f} ± {std_prec[10]:.4f}")
            
            # 2. Stable retrieval analysis (only for smaller datasets)
            if len(y) < 5000:  # Skip for very large datasets
                print(f"\n2. Running stable retrieval analysis...")
                stable_results = compute_stable_precision_at_k(X, y, n_folds=N_FOLDS)
                modality_results['stable_retrieval'] = stable_results
                print(f"   Precision@10: {stable_results[10]['mean']:.4f} " + 
                      f"(95% CI: [{stable_results[10]['ci_lower']:.4f}, {stable_results[10]['ci_upper']:.4f}])")
            else:
                print(f"\n2. Skipping stable analysis (dataset too large: {len(y)} samples)")
            
            # 3. AMI analysis
            print(f"\n3. Running AMI analysis...")
            ami_results = compute_ami_scores(X, y, sample_size=SAMPLE_SIZE)
            modality_results['ami'] = ami_results
            print(f"   Clustering AMI: {ami_results['clustering_ami']:.4f}")
            print(f"   Retrieval AMI@10: {ami_results['retrieval_ami'][10]:.4f}")
            
            # 4. Failure analysis
            print(f"\n4. Running failure analysis...")
            failure_stats, failures_df = analyze_failures(X, y, k=10)
            modality_results['failure_analysis'] = failure_stats
            print(f"   Failure rate: {failure_stats['failure_rate']:.4f}")
            
            # 5. Create visualizations
            print(f"\n5. Creating visualizations...")
            create_visualizations(modality_results, modality)
            
            # Store results
            all_results[modality] = modality_results
            
        except Exception as e:
            print(f"✗ Error processing {modality}: {str(e)}")
            continue
    
    # Create comparison visualizations
    print(f"\n{'='*60}")
    print("Creating comparison visualizations")
    print('='*60)
    
    create_comparison_plots(all_results)
    
    # Save all results
    print(f"\nSaving results...")
    
    # Save JSON results
    results_file = os.path.join(OUTPUT_DIR, "data", "all_retrieval_results.json")
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types
        json_results = {}
        for modality, results in all_results.items():
            json_results[modality] = {
                'data_shape': results['data_shape'],
                'n_classes': results['n_classes'],
                'basic_retrieval': {
                    'mean': {int(k): float(v) for k, v in results['basic_retrieval']['mean'].items()},
                    'std': {int(k): float(v) for k, v in results['basic_retrieval']['std'].items()}
                },
                'ami': {
                    'clustering_ami': float(results['ami']['clustering_ami']),
                    'clustering_nmi': float(results['ami']['clustering_nmi']),
                    'retrieval_ami': {int(k): float(v) for k, v in results['ami']['retrieval_ami'].items()},
                    'retrieval_acc': {int(k): float(v) for k, v in results['ami']['retrieval_acc'].items()}
                },
                'failure_analysis': results['failure_analysis']
            }
            
            # Add stable results if available
            if 'stable_retrieval' in results:
                json_results[modality]['stable_retrieval'] = {
                    int(k): {
                        'mean': float(v['mean']),
                        'std': float(v['std']),
                        'ci_lower': float(v['ci_lower']),
                        'ci_upper': float(v['ci_upper'])
                    } for k, v in results['stable_retrieval'].items()
                }
        
        json.dump(json_results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(all_results)
    
    print(f"\n✓ All analyses complete! Results saved to: {OUTPUT_DIR}/")


def create_comparison_plots(all_results):
    """Create plots comparing all modalities."""
    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    
    # 1. Precision@k comparison
    plt.figure(figsize=(12, 8))
    
    for modality, results in all_results.items():
        if 'basic_retrieval' not in results:
            continue
            
        k_values = sorted(results['basic_retrieval']['mean'].keys())
        mean_values = [results['basic_retrieval']['mean'][k] for k in k_values]
        
        plt.plot(k_values, mean_values, linewidth=2, label=modality.capitalize())
    
    plt.xlabel('k')
    plt.ylabel('Precision@k')
    plt.title('Retrieval Precision@k - All Modalities')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'all_modalities_precision_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. AMI comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    modalities = []
    clustering_amis = []
    retrieval_amis = []
    
    for modality, results in all_results.items():
        if 'ami' not in results:
            continue
            
        modalities.append(modality.capitalize())
        clustering_amis.append(results['ami']['clustering_ami'])
        retrieval_amis.append(results['ami']['retrieval_ami'][10])
    
    x = np.arange(len(modalities))
    
    # Clustering AMI
    ax1.bar(x, clustering_amis, color='skyblue')
    ax1.set_xlabel('Modality')
    ax1.set_ylabel('AMI Score')
    ax1.set_title('Clustering AMI by Modality')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modalities)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(clustering_amis):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Retrieval AMI@10
    ax2.bar(x, retrieval_amis, color='lightcoral')
    ax2.set_xlabel('Modality')
    ax2.set_ylabel('AMI Score')
    ax2.set_title('Retrieval AMI@10 by Modality')
    ax2.set_xticks(x)
    ax2.set_xticklabels(modalities)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(retrieval_amis):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'ami_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance summary heatmap
    metrics = ['Precision@5', 'Precision@10', 'Precision@20', 'Clustering AMI', 'Retrieval AMI@10']
    modality_names = []
    metric_values = []
    
    for modality, results in all_results.items():
        if 'basic_retrieval' not in results or 'ami' not in results:
            continue
            
        modality_names.append(modality.capitalize())
        values = [
            results['basic_retrieval']['mean'][5],
            results['basic_retrieval']['mean'][10],
            results['basic_retrieval']['mean'][20],
            results['ami']['clustering_ami'],
            results['ami']['retrieval_ami'][10]
        ]
        metric_values.append(values)
    
    if metric_values:
        metric_values = np.array(metric_values).T
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(metric_values, 
                    xticklabels=modality_names,
                    yticklabels=metrics,
                    annot=True, 
                    fmt='.3f',
                    cmap='YlOrRd',
                    cbar_kws={'label': 'Score'})
        plt.title('Retrieval Performance Summary')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'performance_summary_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()


def generate_summary_report(all_results):
    """Generate a comprehensive text summary report."""
    report_path = os.path.join(OUTPUT_DIR, "reports", "retrieval_summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("UNIFIED RETRIEVAL ANALYSIS SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary table
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Modality':<15} {'Samples':<10} {'Classes':<10} {'P@10':<12} {'AMI':<12} {'Failure%':<12}\n")
        f.write("-"*80 + "\n")
        
        for modality, results in sorted(all_results.items()):
            if 'basic_retrieval' not in results:
                continue
                
            p_at_10 = results['basic_retrieval']['mean'][10]
            p_at_10_std = results['basic_retrieval']['std'][10]
            ami = results['ami']['clustering_ami'] if 'ami' in results else 0
            failure_rate = results['failure_analysis']['failure_rate'] * 100 if 'failure_analysis' in results else 0
            
            f.write(f"{modality.capitalize():<15} "
                   f"{results['data_shape'][0]:<10} "
                   f"{results['n_classes']:<10} "
                   f"{p_at_10:.3f}±{p_at_10_std:.3f}  "
                   f"{ami:<12.3f} "
                   f"{failure_rate:<12.1f}\n")
        
        f.write("\n\nDETAILED RESULTS BY MODALITY\n")
        f.write("="*80 + "\n")
        
        for modality, results in sorted(all_results.items()):
            f.write(f"\n{modality.upper()}\n")
            f.write("-"*40 + "\n")
            
            # Data characteristics
            f.write(f"Data shape: {results['data_shape']}\n")
            f.write(f"Number of classes: {results['n_classes']}\n")
            
            # Basic retrieval results
            if 'basic_retrieval' in results:
                f.write(f"\nRetrieval Performance:\n")
                for k in REPORT_K_VALUES:
                    mean = results['basic_retrieval']['mean'][k]
                    std = results['basic_retrieval']['std'][k]
                    f.write(f"  Precision@{k}: {mean:.4f} ± {std:.4f}\n")
            
            # Stable retrieval results
            if 'stable_retrieval' in results:
                f.write(f"\nStable Retrieval (95% CI):\n")
                for k in REPORT_K_VALUES:
                    mean = results['stable_retrieval'][k]['mean']
                    ci_lower = results['stable_retrieval'][k]['ci_lower']
                    ci_upper = results['stable_retrieval'][k]['ci_upper']
                    f.write(f"  Precision@{k}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]\n")
            
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
                
                # Top confused pairs
                if results['failure_analysis']['confusion_matrix']:
                    f.write(f"  Most confused cancer type pairs:\n")
                    confusion_pairs = []
                    for query_label, retrieved_dict in results['failure_analysis']['confusion_matrix'].items():
                        for retrieved_label, count in retrieved_dict.items():
                            confusion_pairs.append((count, query_label, retrieved_label))
                    
                    confusion_pairs.sort(reverse=True)
                    for count, query, retrieved in confusion_pairs[:5]:
                        f.write(f"    {query} → {retrieved}: {count} times\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")


if __name__ == "__main__":
    print("Starting unified retrieval analysis...")
    print(f"Output directory: {OUTPUT_DIR}/")
    run_all_analyses()