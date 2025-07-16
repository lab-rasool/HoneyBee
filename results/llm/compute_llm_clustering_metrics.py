"""
Compute AMI and NMI scores for pre-trained and fine-tuned LLM embeddings.
Follows the same approach as the retrieval analysis scripts.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import json
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed
np.random.seed(42)

# Constants
SAMPLE_SIZE = 1000  # Sample size for fast AMI analysis


def load_embeddings(filepath):
    """Load embeddings from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_ami_nmi_scores(embeddings, labels, sample_size=SAMPLE_SIZE):
    """Compute AMI and NMI scores for clustering quality assessment."""
    n_samples = len(labels)
    labels = np.array(labels)
    
    # Remove any None or NaN labels
    valid_mask = pd.Series(labels).notna()
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]
    
    # Sample data if too large
    if n_samples > sample_size:
        sample_idx = np.random.choice(len(labels), sample_size, replace=False)
        embeddings_sampled = embeddings[sample_idx]
        labels_sampled = labels[sample_idx]
    else:
        embeddings_sampled = embeddings
        labels_sampled = labels
    
    # Normalize embeddings
    embeddings_norm = normalize(embeddings_sampled, norm='l2')
    
    # Determine number of clusters
    n_clusters = len(np.unique(labels_sampled))
    
    # Perform clustering
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_norm)
    
    # Compute metrics
    ami = adjusted_mutual_info_score(labels_sampled, cluster_labels)
    nmi = normalized_mutual_info_score(labels_sampled, cluster_labels)
    
    return {
        'ami': ami,
        'nmi': nmi,
        'n_samples': len(labels_sampled),
        'n_clusters': n_clusters
    }


def main():
    """Main execution"""
    print("=== Computing AMI and NMI Scores for LLM Embeddings ===\n")
    
    # Directories
    pretrained_dir = "/mnt/f/Projects/HoneyBee/results/llm/embeddings"
    finetuned_dir = "/mnt/f/Projects/HoneyBee/results/llm/finetuned_embeddings"
    output_dir = "/mnt/f/Projects/HoneyBee/results/llm/clustering_metrics"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Models and text types
    models = ['gatortron', 'qwen', 'medgemma', 'llama']
    text_types = ['clinical', 'pathology']
    
    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'sample_size': SAMPLE_SIZE,
        'metrics': {}
    }
    
    # Summary for display
    summary_data = []
    
    # Process each model and text type
    for text_type in text_types:
        print(f"\n{text_type.upper()} TEXT:")
        print("-" * 60)
        
        for model in models:
            print(f"\n  {model.upper()}:")
            
            # Pre-trained embeddings
            pretrained_path = os.path.join(pretrained_dir, f"{text_type}_{model}_embeddings.pkl")
            pretrained_data = load_embeddings(pretrained_path)
            
            if pretrained_data:
                embeddings = pretrained_data['embeddings']
                labels = pretrained_data['project_ids']
                
                scores = compute_ami_nmi_scores(embeddings, labels)
                
                key = f"{text_type}_{model}_pretrained"
                results['metrics'][key] = scores
                
                print(f"    Pre-trained  - AMI: {scores['ami']:.4f}, NMI: {scores['nmi']:.4f} "
                      f"(n={scores['n_samples']}, clusters={scores['n_clusters']})")
                
                summary_data.append({
                    'text_type': text_type,
                    'model': model,
                    'embeddings': 'pretrained',
                    'ami': scores['ami'],
                    'nmi': scores['nmi'],
                    'n_samples': scores['n_samples'],
                    'n_clusters': scores['n_clusters']
                })
            
            # Fine-tuned embeddings
            finetuned_path = os.path.join(finetuned_dir, f"{text_type}_{model}_finetuned_embeddings.pkl")
            if os.path.exists(finetuned_path):
                finetuned_data = load_embeddings(finetuned_path)
                
                if finetuned_data:
                    embeddings = finetuned_data['embeddings']
                    labels = finetuned_data['project_ids']
                    
                    scores = compute_ami_nmi_scores(embeddings, labels)
                    
                    key = f"{text_type}_{model}_finetuned"
                    results['metrics'][key] = scores
                    
                    print(f"    Fine-tuned   - AMI: {scores['ami']:.4f}, NMI: {scores['nmi']:.4f} "
                          f"(n={scores['n_samples']}, clusters={scores['n_clusters']})")
                    
                    summary_data.append({
                        'text_type': text_type,
                        'model': model,
                        'embeddings': 'finetuned',
                        'ami': scores['ami'],
                        'nmi': scores['nmi'],
                        'n_samples': scores['n_samples'],
                        'n_clusters': scores['n_clusters']
                    })
                    
                    # Show improvement
                    if f"{text_type}_{model}_pretrained" in results['metrics']:
                        pretrained_scores = results['metrics'][f"{text_type}_{model}_pretrained"]
                        ami_diff = scores['ami'] - pretrained_scores['ami']
                        nmi_diff = scores['nmi'] - pretrained_scores['nmi']
                        print(f"    Improvement  - AMI: {ami_diff:+.4f}, NMI: {nmi_diff:+.4f}")
    
    # Save results
    # JSON format
    json_path = os.path.join(output_dir, 'ami_nmi_scores.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved detailed results to: {json_path}")
    
    # CSV summary
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'ami_nmi_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary to: {csv_path}")
    
    # Generate report
    report_path = os.path.join(output_dir, 'clustering_metrics_report.txt')
    with open(report_path, 'w') as f:
        f.write("LLM EMBEDDINGS CLUSTERING METRICS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample size: {SAMPLE_SIZE}\n\n")
        
        # Summary table
        f.write("SUMMARY RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Text Type':<12} {'Model':<12} {'Embeddings':<12} "
                f"{'AMI':<10} {'NMI':<10} {'Samples':<10} {'Clusters':<10}\n")
        f.write("-"*80 + "\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"{row['text_type']:<12} {row['model']:<12} {row['embeddings']:<12} "
                    f"{row['ami']:<10.4f} {row['nmi']:<10.4f} "
                    f"{row['n_samples']:<10} {row['n_clusters']:<10}\n")
        
        # Improvements section
        f.write("\n\nIMPROVEMENTS (Fine-tuned vs Pre-trained)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Text Type':<12} {'Model':<12} {'AMI Change':<15} {'NMI Change':<15}\n")
        f.write("-"*80 + "\n")
        
        for text_type in text_types:
            for model in models:
                pretrained = summary_df[(summary_df['text_type'] == text_type) & 
                                      (summary_df['model'] == model) & 
                                      (summary_df['embeddings'] == 'pretrained')]
                finetuned = summary_df[(summary_df['text_type'] == text_type) & 
                                     (summary_df['model'] == model) & 
                                     (summary_df['embeddings'] == 'finetuned')]
                
                if len(pretrained) > 0 and len(finetuned) > 0:
                    ami_change = finetuned.iloc[0]['ami'] - pretrained.iloc[0]['ami']
                    nmi_change = finetuned.iloc[0]['nmi'] - pretrained.iloc[0]['nmi']
                    f.write(f"{text_type:<12} {model:<12} {ami_change:<+15.4f} {nmi_change:<+15.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved report to: {report_path}")
    print("\n=== Computation Complete ===")


if __name__ == "__main__":
    main()