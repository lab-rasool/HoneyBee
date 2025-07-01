import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


class LLMRetrievalAnalysis:
    """Perform retrieval analysis on LLM-generated embeddings."""
    
    def __init__(self, embeddings_dir: str):
        """Initialize with embeddings directory."""
        self.embeddings_dir = embeddings_dir
        self.results = defaultdict(dict)
        
    def load_embeddings(self, text_type: str, model_key: str):
        """Load embeddings for a specific text type and model."""
        filename = f"{text_type}_{model_key}_embeddings.pkl"
        filepath = os.path.join(self.embeddings_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            return None, None
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        return data['embeddings'], data['project_ids']
    
    def compute_precision_at_k(self, embeddings, labels, k_values=None, n_runs=3):
        """Compute precision@k for various k values."""
        if k_values is None:
            k_values = [1, 5, 10, 20, 30, 40, 50]
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Initialize results storage
        precisions = defaultdict(list)
        
        for run in range(n_runs):
            # Random seed for consistency
            np.random.seed(run)
            
            # Shuffle indices for this run
            indices = np.arange(len(embeddings))
            np.random.shuffle(indices)
            
            # Compute precision@k for each k
            for k in k_values:
                precision_scores = []
                
                for i in range(len(embeddings)):
                    # Get query label
                    query_label = encoded_labels[i]
                    
                    # Get similarities to all other samples
                    similarities = sim_matrix[i].copy()
                    similarities[i] = -1  # Exclude self
                    
                    # Get top k most similar indices
                    top_k_indices = np.argsort(similarities)[-k:][::-1]
                    
                    # Count how many have the same label
                    correct = sum(encoded_labels[idx] == query_label for idx in top_k_indices)
                    
                    precision_scores.append(correct / k)
                
                precisions[k].append(np.mean(precision_scores))
        
        # Compute mean and std for each k
        results = {}
        for k in k_values:
            results[f'precision@{k}'] = {
                'mean': np.mean(precisions[k]),
                'std': np.std(precisions[k])
            }
        
        return results, precisions
    
    def compute_ami_scores(self, embeddings, labels, n_clusters=None):
        """Compute AMI scores for clustering quality."""
        from sklearn.cluster import KMeans
        
        # Encode labels
        le = LabelEncoder()
        true_labels = le.fit_transform(labels)
        
        if n_clusters is None:
            n_clusters = len(np.unique(true_labels))
        
        ami_scores = []
        
        # Run multiple times for stability
        for seed in range(5):
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            predicted_labels = kmeans.fit_predict(embeddings)
            
            ami = adjusted_mutual_info_score(true_labels, predicted_labels)
            ami_scores.append(ami)
        
        return {
            'mean': np.mean(ami_scores),
            'std': np.std(ami_scores)
        }
    
    def compute_retrieval_ami(self, embeddings, labels):
        """Compute AMI based on retrieval results."""
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # For each sample, assign it to the most common class among its k nearest neighbors
        k = 10  # Use top 10 neighbors
        predicted_labels = []
        
        for i in range(len(embeddings)):
            similarities = sim_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            
            # Get top k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            # Get labels of top k
            neighbor_labels = encoded_labels[top_k_indices]
            
            # Assign to most common label
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predicted_labels.append(predicted_label)
        
        ami = adjusted_mutual_info_score(encoded_labels, predicted_labels)
        
        return ami
    
    def failure_analysis(self, embeddings, labels, k=10):
        """Analyze retrieval failures and confusion patterns."""
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Track confusion patterns
        confusion_counts = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(embeddings)):
            query_label = encoded_labels[i]
            query_label_name = labels[i]
            
            similarities = sim_matrix[i].copy()
            similarities[i] = -1
            
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            
            for idx in top_k_indices:
                retrieved_label = encoded_labels[idx]
                retrieved_label_name = labels[idx]
                
                if query_label != retrieved_label:
                    confusion_counts[query_label_name][retrieved_label_name] += 1
        
        return confusion_counts
    
    def plot_precision_curves(self, all_results, save_path):
        """Plot precision@k curves for all models."""
        plt.figure(figsize=(12, 8))
        
        k_values = [1, 5, 10, 20, 30, 40, 50]
        
        # Colors for different models
        colors = {
            'gatortron': 'blue',
            'qwen': 'green',
            'medgemma': 'red',
            'llama': 'orange'
        }
        
        # Line styles for text types
        line_styles = {
            'clinical': '-',
            'pathology': '--'
        }
        
        for model_name, results in all_results.items():
            text_type, model = model_name.split('_')
            
            if 'precisions' in results:
                means = [results[f'precision@{k}']['mean'] for k in k_values]
                stds = [results[f'precision@{k}']['std'] for k in k_values]
                
                color = colors.get(model, 'gray')
                line_style = line_styles.get(text_type, '-')
                
                plt.plot(k_values, means, 
                        color=color, 
                        linestyle=line_style,
                        marker='o',
                        label=f'{model} ({text_type})')
                
                # Add error bars
                plt.fill_between(k_values, 
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=color, alpha=0.2)
        
        plt.xlabel('k', fontsize=12)
        plt.ylabel('Precision@k', fontsize=12)
        plt.title('Retrieval Performance: Precision@k', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ami_comparison(self, all_results, save_path):
        """Plot AMI scores comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract AMI scores
        models = []
        clustering_ami = []
        retrieval_ami = []
        
        for model_name, results in all_results.items():
            if 'ami_clustering' in results:
                models.append(model_name)
                clustering_ami.append(results['ami_clustering']['mean'])
                retrieval_ami.append(results['ami_retrieval'])
        
        x = np.arange(len(models))
        width = 0.35
        
        # Clustering AMI
        bars1 = ax1.bar(x - width/2, clustering_ami, width, label='Clustering AMI')
        bars2 = ax1.bar(x + width/2, retrieval_ami, width, label='Retrieval AMI')
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('AMI Score')
        ax1.set_title('AMI Scores Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Scatter plot
        ax2.scatter(clustering_ami, retrieval_ami)
        for i, model in enumerate(models):
            ax2.annotate(model, (clustering_ami[i], retrieval_ami[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Clustering AMI')
        ax2.set_ylabel('Retrieval AMI')
        ax2.set_title('Clustering vs Retrieval AMI')
        ax2.grid(True, alpha=0.3)
        
        # Add diagonal line
        min_val = min(min(clustering_ami), min(retrieval_ami))
        max_val = max(max(clustering_ami), max(retrieval_ami))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_analyses(self):
        """Run retrieval analysis for all model and text type combinations."""
        output_dir = os.path.join(self.embeddings_dir, '..', 'retrieval_results_all_models')
        os.makedirs(output_dir, exist_ok=True)
        
        # Include all 4 models
        models = ['gatortron', 'qwen', 'medgemma', 'llama']
        text_types = ['clinical', 'pathology']
        
        all_results = {}
        
        for text_type in text_types:
            for model in models:
                print(f"\n=== Running retrieval analysis for {text_type} - {model} ===")
                
                # Load embeddings
                embeddings, project_ids = self.load_embeddings(text_type, model)
                
                if embeddings is None:
                    continue
                
                print(f"Dataset shape: {embeddings.shape}")
                
                # Compute precision@k
                print("Computing precision@k...")
                precision_results, precisions_raw = self.compute_precision_at_k(embeddings, project_ids)
                
                # Compute AMI scores
                print("Computing AMI scores...")
                ami_clustering = self.compute_ami_scores(embeddings, project_ids)
                ami_retrieval = self.compute_retrieval_ami(embeddings, project_ids)
                
                # Failure analysis
                print("Performing failure analysis...")
                confusion_patterns = self.failure_analysis(embeddings, project_ids)
                
                # Store results
                key = f"{text_type}_{model}"
                results = {
                    **precision_results,
                    'precisions': precision_results,
                    'ami_clustering': ami_clustering,
                    'ami_retrieval': ami_retrieval,
                    'confusion_patterns': dict(confusion_patterns)
                }
                
                all_results[key] = results
                self.results[text_type][model] = results
                
                # Print summary
                print(f"Precision@1:  {precision_results['precision@1']['mean']:.4f} ± {precision_results['precision@1']['std']:.4f}")
                print(f"Precision@10: {precision_results['precision@10']['mean']:.4f} ± {precision_results['precision@10']['std']:.4f}")
                print(f"AMI (clustering): {ami_clustering['mean']:.4f} ± {ami_clustering['std']:.4f}")
                print(f"AMI (retrieval):  {ami_retrieval:.4f}")
        
        # Plot comparisons
        precision_plot_path = os.path.join(output_dir, 'precision_curves_all_models.png')
        self.plot_precision_curves(all_results, precision_plot_path)
        
        ami_plot_path = os.path.join(output_dir, 'ami_comparison_all_models.png')
        self.plot_ami_comparison(all_results, ami_plot_path)
        
        # Save results as JSON
        results_json = {}
        for key, result in all_results.items():
            results_json[key] = {k: v for k, v in result.items() if k != 'confusion_patterns'}
            results_json[key]['top_confusions'] = self._get_top_confusions(result['confusion_patterns'])
        
        json_path = os.path.join(output_dir, 'retrieval_results_all_models.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Create summary report
        self.create_summary_report(all_results, output_dir)
        
        return all_results
    
    def _get_top_confusions(self, confusion_patterns, top_n=5):
        """Get top confusion pairs."""
        confusion_list = []
        for source, targets in confusion_patterns.items():
            for target, count in targets.items():
                confusion_list.append({
                    'source': source,
                    'target': target,
                    'count': count
                })
        
        # Sort by count
        confusion_list.sort(key=lambda x: x['count'], reverse=True)
        
        return confusion_list[:top_n]
    
    def create_summary_report(self, results, output_dir):
        """Create a text summary report."""
        report_path = os.path.join(output_dir, 'retrieval_summary_all_models.txt')
        
        with open(report_path, 'w') as f:
            f.write("RETRIEVAL ANALYSIS SUMMARY - ALL MODELS\n")
            f.write("=" * 60 + "\n\n")
            
            # Best model by precision@10
            best_precision = 0
            best_model = None
            
            for model_name, result in results.items():
                p10 = result.get('precision@10', {}).get('mean', 0)
                if p10 > best_precision:
                    best_precision = p10
                    best_model = model_name
            
            f.write(f"Best Model (Precision@10): {best_model} ({best_precision:.4f})\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 60 + "\n\n")
            
            for model_name, result in sorted(results.items()):
                f.write(f"{model_name.upper()}:\n")
                
                # Precision metrics
                for k in [1, 5, 10, 20, 50]:
                    key = f'precision@{k}'
                    if key in result:
                        mean = result[key]['mean']
                        std = result[key]['std']
                        f.write(f"  Precision@{k:2d}: {mean:.4f} ± {std:.4f}\n")
                
                # AMI metrics
                if 'ami_clustering' in result:
                    f.write(f"  AMI (clustering): {result['ami_clustering']['mean']:.4f} ± {result['ami_clustering']['std']:.4f}\n")
                    f.write(f"  AMI (retrieval):  {result['ami_retrieval']:.4f}\n")
                
                f.write("\n")
            
            # Model comparison
            f.write("MODEL COMPARISON:\n")
            f.write("-" * 60 + "\n\n")
            
            # Compare encoder-only vs decoder-only
            f.write("ENCODER-ONLY vs DECODER-ONLY (Precision@10):\n")
            f.write("-" * 30 + "\n")
            
            for text_type in ['clinical', 'pathology']:
                f.write(f"\n{text_type.upper()}:\n")
                
                # Encoder-only models
                f.write("  Encoder-only models:\n")
                for model in ['gatortron', 'qwen']:
                    key = f'{text_type}_{model}'
                    if key in results and 'precision@10' in results[key]:
                        p10 = results[key]['precision@10']['mean']
                        f.write(f"    {model}: {p10:.4f}\n")
                
                # Decoder-only models
                f.write("  Decoder-only models:\n")
                for model in ['medgemma', 'llama']:
                    key = f'{text_type}_{model}'
                    if key in results and 'precision@10' in results[key]:
                        p10 = results[key]['precision@10']['mean']
                        f.write(f"    {model}: {p10:.4f}\n")
            
            # Medical vs General models
            f.write("\n\nMEDICAL vs GENERAL PURPOSE (Precision@10):\n")
            f.write("-" * 30 + "\n")
            
            for text_type in ['clinical', 'pathology']:
                f.write(f"\n{text_type.upper()}:\n")
                
                # Medical models
                f.write("  Medical-focused models:\n")
                for model in ['gatortron', 'medgemma']:
                    key = f'{text_type}_{model}'
                    if key in results and 'precision@10' in results[key]:
                        p10 = results[key]['precision@10']['mean']
                        f.write(f"    {model}: {p10:.4f}\n")
                
                # General models
                f.write("  General-purpose models:\n")
                for model in ['qwen', 'llama']:
                    key = f'{text_type}_{model}'
                    if key in results and 'precision@10' in results[key]:
                        p10 = results[key]['precision@10']['mean']
                        f.write(f"    {model}: {p10:.4f}\n")
        
        print(f"\nSummary report saved to: {report_path}")


def main():
    """Main execution function."""
    embeddings_dir = "/mnt/f/Projects/HoneyBee/results/llm/embeddings"
    
    # Run retrieval analysis
    analyzer = LLMRetrievalAnalysis(embeddings_dir)
    results = analyzer.run_all_analyses()
    
    print("\nRetrieval analysis complete!")
    print(f"Results saved to: {os.path.join(embeddings_dir, '..', 'retrieval_results_all_models')}")


if __name__ == "__main__":
    main()