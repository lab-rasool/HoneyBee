import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class LLMClassificationAnalysis:
    """Perform classification analysis on LLM-generated embeddings."""
    
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
    
    def prepare_data(self, embeddings, labels):
        """Prepare data for classification."""
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        # Remove rare classes (less than 10 samples)
        unique_labels, counts = np.unique(encoded_labels, return_counts=True)
        valid_classes = unique_labels[counts >= 10]
        mask = np.isin(encoded_labels, valid_classes)
        
        embeddings_filtered = embeddings[mask]
        labels_filtered = encoded_labels[mask]
        project_ids_filtered = [labels[i] for i in range(len(labels)) if mask[i]]
        
        # Re-encode to ensure continuous label indices
        le_filtered = LabelEncoder()
        labels_filtered = le_filtered.fit_transform(project_ids_filtered)
        
        return embeddings_filtered, labels_filtered, le_filtered
    
    def run_classification(self, embeddings, labels, n_runs=10):
        """Run classification with multiple random seeds."""
        accuracies = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        all_confusion_matrices = []
        
        for seed in range(n_runs):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, labels, test_size=0.2, random_state=seed, stratify=labels
            )
            
            # Train classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred = clf.predict(X_test)
            
            # Calculate metrics
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
            recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
            all_confusion_matrices.append(confusion_matrix(y_test, y_pred))
        
        # Average confusion matrix
        avg_cm = np.mean(all_confusion_matrices, axis=0)
        
        return {
            'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies)},
            'f1_score': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)},
            'precision': {'mean': np.mean(precision_scores), 'std': np.std(precision_scores)},
            'recall': {'mean': np.mean(recall_scores), 'std': np.std(recall_scores)},
            'confusion_matrix': avg_cm,
            'all_accuracies': accuracies
        }
    
    def plot_confusion_matrix(self, cm, labels, title, save_path):
        """Plot confusion matrix."""
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_comparison(self, results, save_path):
        """Plot comparison of metrics across models."""
        # Extract metrics
        models = list(results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            means = [results[model][metric]['mean'] for model in models]
            stds = [results[model][metric]['std'] for model in models]
            
            x = np.arange(len(models))
            
            bars = axes[idx].bar(x, means, yerr=stds, capsize=5)
            
            # Color bars by model type
            colors = []
            for model in models:
                if 'gatortron' in model:
                    colors.append('blue')
                elif 'qwen' in model:
                    colors.append('green')
                elif 'medgemma' in model:
                    colors.append('red')
                elif 'llama' in model:
                    colors.append('orange')
                else:
                    colors.append('gray')
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(models, rotation=45, ha='right')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_ylim(0, 1.05)
            axes[idx].grid(axis='y', alpha=0.3)
            
        plt.suptitle('Classification Performance Metrics Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_analyses(self):
        """Run classification analysis for all model and text type combinations."""
        output_dir = os.path.join(self.embeddings_dir, '..', 'classification_results_all_models')
        os.makedirs(output_dir, exist_ok=True)
        
        # Include all 4 models
        models = ['gatortron', 'qwen', 'medgemma', 'llama']
        text_types = ['clinical', 'pathology']
        
        all_results = {}
        
        for text_type in text_types:
            for model in models:
                print(f"\n=== Running classification for {text_type} - {model} ===")
                
                # Load embeddings
                embeddings, project_ids = self.load_embeddings(text_type, model)
                
                if embeddings is None:
                    continue
                
                # Prepare data
                embeddings_filtered, labels_filtered, label_encoder = self.prepare_data(
                    embeddings, project_ids
                )
                
                print(f"Dataset shape: {embeddings_filtered.shape}")
                print(f"Number of classes: {len(np.unique(labels_filtered))}")
                
                # Run classification
                results = self.run_classification(embeddings_filtered, labels_filtered)
                
                # Store results
                key = f"{text_type}_{model}"
                all_results[key] = results
                self.results[text_type][model] = results
                
                # Print results
                print(f"Accuracy: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
                print(f"F1 Score: {results['f1_score']['mean']:.4f} ± {results['f1_score']['std']:.4f}")
                
                # Plot confusion matrix
                cm_path = os.path.join(output_dir, f'confusion_matrix_{text_type}_{model}.png')
                self.plot_confusion_matrix(
                    results['confusion_matrix'],
                    label_encoder.classes_,
                    f'Confusion Matrix - {text_type.capitalize()} {model.upper()}',
                    cm_path
                )
        
        # Plot overall comparison
        comparison_path = os.path.join(output_dir, 'metrics_comparison_all_models.png')
        self.plot_metrics_comparison(all_results, comparison_path)
        
        # Save results as JSON
        results_json = {}
        for key, result in all_results.items():
            results_json[key] = {
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score'],
                'precision': result['precision'],
                'recall': result['recall'],
                'confusion_matrix': result['confusion_matrix'].tolist()
            }
        
        json_path = os.path.join(output_dir, 'classification_results_all_models.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Create summary report
        self.create_summary_report(all_results, output_dir)
        
        return all_results
    
    def create_summary_report(self, results, output_dir):
        """Create a text summary report."""
        report_path = os.path.join(output_dir, 'classification_summary_all_models.txt')
        
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION ANALYSIS SUMMARY - ALL MODELS\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall best model
            best_acc = 0
            best_model = None
            
            for model_name, result in results.items():
                acc = result['accuracy']['mean']
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name
            
            f.write(f"Best Model: {best_model} (Accuracy: {best_acc:.4f})\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 60 + "\n\n")
            
            for model_name, result in sorted(results.items()):
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  Accuracy:  {result['accuracy']['mean']:.4f} ± {result['accuracy']['std']:.4f}\n")
                f.write(f"  F1 Score:  {result['f1_score']['mean']:.4f} ± {result['f1_score']['std']:.4f}\n")
                f.write(f"  Precision: {result['precision']['mean']:.4f} ± {result['precision']['std']:.4f}\n")
                f.write(f"  Recall:    {result['recall']['mean']:.4f} ± {result['recall']['std']:.4f}\n\n")
            
            # Model comparison
            f.write("MODEL COMPARISON:\n")
            f.write("-" * 60 + "\n\n")
            
            # Compare encoder-only vs decoder-only
            f.write("ENCODER-ONLY vs DECODER-ONLY:\n")
            f.write("-" * 30 + "\n")
            
            for text_type in ['clinical', 'pathology']:
                f.write(f"\n{text_type.upper()}:\n")
                
                # Encoder-only models
                f.write("  Encoder-only models:\n")
                for model in ['gatortron', 'qwen']:
                    key = f'{text_type}_{model}'
                    if key in results:
                        acc = results[key]['accuracy']['mean']
                        f.write(f"    {model}: {acc:.4f}\n")
                
                # Decoder-only models
                f.write("  Decoder-only models:\n")
                for model in ['medgemma', 'llama']:
                    key = f'{text_type}_{model}'
                    if key in results:
                        acc = results[key]['accuracy']['mean']
                        f.write(f"    {model}: {acc:.4f}\n")
            
            # Medical vs General models
            f.write("\n\nMEDICAL vs GENERAL PURPOSE:\n")
            f.write("-" * 30 + "\n")
            
            for text_type in ['clinical', 'pathology']:
                f.write(f"\n{text_type.upper()}:\n")
                
                # Medical models
                f.write("  Medical-focused models:\n")
                for model in ['gatortron', 'medgemma']:
                    key = f'{text_type}_{model}'
                    if key in results:
                        acc = results[key]['accuracy']['mean']
                        f.write(f"    {model}: {acc:.4f}\n")
                
                # General models
                f.write("  General-purpose models:\n")
                for model in ['qwen', 'llama']:
                    key = f'{text_type}_{model}'
                    if key in results:
                        acc = results[key]['accuracy']['mean']
                        f.write(f"    {model}: {acc:.4f}\n")
        
        print(f"\nSummary report saved to: {report_path}")


def main():
    """Main execution function."""
    embeddings_dir = "/mnt/f/Projects/HoneyBee/results/llm/embeddings"
    
    # Run classification analysis
    analyzer = LLMClassificationAnalysis(embeddings_dir)
    results = analyzer.run_all_analyses()
    
    print("\nClassification analysis complete!")
    print(f"Results saved to: {os.path.join(embeddings_dir, '..', 'classification_results_all_models')}")


if __name__ == "__main__":
    main()