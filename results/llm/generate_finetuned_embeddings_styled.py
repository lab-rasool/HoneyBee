#!/usr/bin/env python3
"""
Generate embeddings using fine-tuned pathology classification models with matching t-SNE styling.
Extracts representations from the penultimate layer of the trained classifiers.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style to match original
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
MODELS = ['gatortron', 'qwen', 'llama', 'medgemma']
EMBEDDINGS_DIR = './embeddings'
FINETUNED_MODELS_DIR = './pathology_finetuning_results/trained_models'
OUTPUT_DIR = './finetuned_embeddings'
VISUALIZATION_DIR = './finetuned_tsne_visualizations'


class SimpleClassifier(nn.Module):
    """Improved MLP classifier for embeddings (must match training architecture)."""
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)
    
    def get_embeddings(self, x):
        """Extract embeddings from penultimate layer."""
        self.eval()
        with torch.no_grad():
            # Process through layers to get penultimate activations
            for i, layer in enumerate(self.classifier):
                x = layer(x)
                # Stop before the last linear layer (which is at index 12)
                if i == 11:  # After the last dropout, before final linear
                    break
        return x


class FinetunedTSNEVisualizer:
    """Class to handle t-SNE visualization for fine-tuned embeddings with original styling."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories matching original structure
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        os.makedirs(os.path.join(VISUALIZATION_DIR, 'individual'), exist_ok=True)
        os.makedirs(os.path.join(VISUALIZATION_DIR, 'comparisons'), exist_ok=True)
        
        self.embeddings_data = {}
        self.finetuned_embeddings_data = {}
    
    def get_tcga_color_scheme(self):
        """Return TCGA-specific color scheme - exact copy from original."""
        return {
            # Brain cancers - blues
            'TCGA-GBM': '#4A90E2',  # Glioblastoma
            'TCGA-LGG': '#7BB3F0',  # Lower grade glioma
            
            # Kidney cancers - greens  
            'TCGA-KIRC': '#27AE60',  # Clear cell
            'TCGA-KIRP': '#52C77F',  # Papillary
            'TCGA-KICH': '#7ED99F',  # Chromophobe
            
            # Lung cancers - oranges
            'TCGA-LUAD': '#F39C12',  # Adenocarcinoma
            'TCGA-LUSC': '#FDB94E',  # Squamous cell
            
            # GI cancers - reds/pinks
            'TCGA-COAD': '#E74C3C',  # Colon
            'TCGA-READ': '#F17A72',  # Rectal
            'TCGA-STAD': '#FF9999',  # Stomach
            'TCGA-ESCA': '#FFB3B3',  # Esophageal
            
            # Liver/pancreas - purples
            'TCGA-LIHC': '#9B59B6',  # Liver
            'TCGA-PAAD': '#B983CC',  # Pancreatic
            'TCGA-CHOL': '#D7BDE2',  # Cholangiocarcinoma
            
            # Gynecological - magentas
            'TCGA-BRCA': '#E91E63',  # Breast
            'TCGA-OV': '#F06292',   # Ovarian
            'TCGA-UCEC': '#F8BBD0',  # Endometrial
            'TCGA-CESC': '#FCE4EC',  # Cervical
            'TCGA-UCS': '#FFCDD2',   # Uterine carcinosarcoma
            
            # Urological - teals
            'TCGA-BLCA': '#00BCD4',  # Bladder
            'TCGA-PRAD': '#4DD0E1',  # Prostate
            'TCGA-TGCT': '#80DEEA',  # Testicular
            
            # Blood cancers - light blues
            'TCGA-LAML': '#03A9F4',  # Acute myeloid leukemia
            'TCGA-DLBC': '#64B5F6',  # Lymphoma
            
            # Skin/melanoma - browns
            'TCGA-SKCM': '#795548',  # Melanoma
            'TCGA-UVM': '#A1887F',   # Uveal melanoma
            
            # Others - distinct colors
            'TCGA-ACC': '#FF6B6B',   # Adrenocortical
            'TCGA-HNSC': '#4ECDC4',  # Head and neck
            'TCGA-MESO': '#95E1D3',  # Mesothelioma
            'TCGA-PCPG': '#F38181',  # Pheochromocytoma
            'TCGA-SARC': '#AA96DA',  # Sarcoma
            'TCGA-THCA': '#C7CEEA',  # Thyroid
            'TCGA-THYM': '#B2EBF2',  # Thymoma
        }
    
    def get_tcga_descriptions(self):
        """Return TCGA cancer type descriptions - exact copy from original."""
        return {
            'TCGA-ACC': 'Adrenocortical carcinoma',
            'TCGA-BLCA': 'Bladder urothelial carcinoma',
            'TCGA-BRCA': 'Breast invasive carcinoma',
            'TCGA-CESC': 'Cervical squamous cell carcinoma',
            'TCGA-CHOL': 'Cholangiocarcinoma',
            'TCGA-COAD': 'Colon adenocarcinoma',
            'TCGA-DLBC': 'Diffuse large B-cell lymphoma',
            'TCGA-ESCA': 'Esophageal carcinoma',
            'TCGA-GBM': 'Glioblastoma multiforme',
            'TCGA-HNSC': 'Head and neck squamous cell carcinoma',
            'TCGA-KICH': 'Kidney chromophobe',
            'TCGA-KIRC': 'Kidney renal clear cell carcinoma',
            'TCGA-KIRP': 'Kidney renal papillary cell carcinoma',
            'TCGA-LGG': 'Brain lower grade glioma',
            'TCGA-LIHC': 'Liver hepatocellular carcinoma',
            'TCGA-LUAD': 'Lung adenocarcinoma',
            'TCGA-LUSC': 'Lung squamous cell carcinoma',
            'TCGA-MESO': 'Mesothelioma',
            'TCGA-OV': 'Ovarian serous cystadenocarcinoma',
            'TCGA-PAAD': 'Pancreatic adenocarcinoma',
            'TCGA-PCPG': 'Pheochromocytoma and paraganglioma',
            'TCGA-PRAD': 'Prostate adenocarcinoma',
            'TCGA-READ': 'Rectum adenocarcinoma',
            'TCGA-SARC': 'Sarcoma',
            'TCGA-SKCM': 'Skin cutaneous melanoma',
            'TCGA-STAD': 'Stomach adenocarcinoma',
            'TCGA-TGCT': 'Testicular germ cell tumors',
            'TCGA-THCA': 'Thyroid carcinoma',
            'TCGA-THYM': 'Thymoma',
            'TCGA-UCEC': 'Uterine corpus endometrial carcinoma',
            'TCGA-UCS': 'Uterine carcinosarcoma',
            'TCGA-UVM': 'Uveal melanoma'
        }
    
    def create_tsne_plot_with_separate_legend(self, embeddings, labels, title, filename_base, figsize=(10, 8)):
        """Create a t-SNE plot with a separate legend file - matching original styling."""
        # Convert labels to strings to handle mixed types
        labels = pd.Series(labels).fillna('Unknown').astype(str).values
        
        # Get unique labels and create color mapping
        unique_labels = [l for l in np.unique(labels) if l != 'Unknown']
        unique_labels.sort()
        
        # Add 'Unknown' at the end if it exists
        if 'Unknown' in labels:
            unique_labels.append('Unknown')
        
        n_colors = len(unique_labels)
        
        # Get color schemes
        tcga_color_groups = self.get_tcga_color_scheme()
        tcga_descriptions = self.get_tcga_descriptions()
        
        # Markers for visual distinction
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd']
        
        color_dict = {}
        marker_dict = {}
        
        for i, label in enumerate(unique_labels):
            if label == 'Unknown':
                color_dict[label] = 'lightgray'
                marker_dict[label] = 'o'
            elif label in tcga_color_groups:
                color_dict[label] = tcga_color_groups[label]
                marker_dict[label] = markers[i % len(markers)]
            else:
                # Fallback color
                color_dict[label] = plt.cm.Set3(i % 12)
                marker_dict[label] = markers[i % len(markers)]
        
        # Create main plot without legend
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each category separately
        for label in unique_labels:
            mask = labels == label
            if np.any(mask):
                color = color_dict[label]
                marker = marker_dict[label]
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                          c=[color], marker=marker, alpha=0.6, s=50, 
                          edgecolors='black', linewidth=0.5, label=label)
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        # ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{filename_base}_plot.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}_plot.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        # Create separate legend file
        fig_height = max(6, n_colors * 0.3)
        fig_width = 12
        
        fig_legend, ax_legend = plt.subplots(figsize=(fig_width, fig_height))
        ax_legend.axis('off')
        
        # Create legend entries
        legend_entries = []
        for label in unique_labels:
            color = color_dict[label]
            marker = marker_dict[label]
            
            # Get description
            description = tcga_descriptions.get(label, label)
            
            # Create legend entry with marker and description
            legend_entries.append(
                ax_legend.scatter([], [], c=[color], marker=marker, s=100, 
                                edgecolors='black', linewidth=0.5, 
                                label=f"{label} - {description}")
            )
        
        # Add legend to figure
        ax_legend.legend(handles=legend_entries, loc='center', ncol=2, 
                        frameon=True, fancybox=True, shadow=True,
                        fontsize=10, title='TCGA Cancer Types', title_fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{filename_base}_legend.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}_legend.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename_base}_plot.pdf/svg and {filename_base}_legend.pdf/svg")
    
    def compute_tsne(self, embeddings, perplexity=30):
        """Compute t-SNE on embeddings - matching original."""
        embeddings_sampled = embeddings
        indices = np.arange(embeddings.shape[0])
        sampled = False
        
        # Standardize
        print("Standardizing embeddings...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_sampled)
        
        # Compute t-SNE
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, 
                    random_state=42, verbose=0)
        embeddings_2d = tsne.fit_transform(embeddings_scaled)
        
        return embeddings_2d, indices if sampled else None
    
    def create_model_comparison_plot(self, all_tsne_results):
        """Create a 2x2 comparison plot for all models - matching original styling."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.ravel()
        
        models = ['gatortron', 'qwen', 'medgemma', 'llama']
        model_titles = {
            'gatortron': 'GatorTron (Medical Encoder)',
            'qwen': 'Qwen (General Encoder)',
            'medgemma': 'Med-Gemma (Medical Decoder)', 
            'llama': 'Llama-3.2-1B (General Decoder)'
        }
        
        tcga_color_groups = self.get_tcga_color_scheme()
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd']
        
        for idx, model in enumerate(models):
            key = f"pathology_{model}_finetuned"
            if key not in all_tsne_results:
                continue
                
            ax = axes[idx]
            embeddings_2d, labels, indices = all_tsne_results[key]
            
            # Convert labels
            labels = pd.Series(labels).fillna('Unknown').astype(str).values
            unique_labels = sorted([l for l in np.unique(labels) if l != 'Unknown'])
            if 'Unknown' in labels:
                unique_labels.append('Unknown')
            
            # Plot each cancer type
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if np.any(mask):
                    color = tcga_color_groups.get(label, plt.cm.Set3(i % 12))
                    marker = markers[i % len(markers)]
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                             c=[color], marker=marker, alpha=0.6, s=30, 
                             edgecolors='black', linewidth=0.3)
            
            ax.set_title(f'{model_titles[model]}\n(Fine-tuned)', fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1', fontsize=10)
            ax.set_ylabel('t-SNE 2', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # plt.suptitle('Pathology Text Embeddings - Fine-tuned Model Comparison', 
        #             fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename_base = os.path.join(VISUALIZATION_DIR, 'comparisons', 
                                    'tsne_comparison_pathology_finetuned')
        plt.savefig(f"{filename_base}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: {filename_base}.pdf/svg")
    
    def create_original_vs_finetuned_comparison(self, original_tsne_results, finetuned_tsne_results):
        """Create a comparison plot showing original vs finetuned embeddings with integrated legend."""
        import matplotlib.gridspec as gridspec
        import textwrap
        
        # Create figure with GridSpec for custom layout - 4 rows x 3 columns
        fig = plt.figure(figsize=(14, 16))
        gs = gridspec.GridSpec(4, 3, width_ratios=[1, 1, 0.6], 
                              height_ratios=[1, 1, 1, 1], hspace=0.25, wspace=0.2,
                              left=0.05, right=0.95, top=0.98, bottom=0.02)
        
        models = ['gatortron', 'qwen', 'medgemma', 'llama']
        model_titles = {
            'gatortron': 'GatorTron',
            'qwen': 'Qwen',
            'medgemma': 'Med-Gemma', 
            'llama': 'Llama-3.2-1B'
        }
        
        tcga_color_groups = self.get_tcga_color_scheme()
        tcga_descriptions = self.get_tcga_descriptions()
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd']
        
        # Collect all unique labels for consistent legend
        all_labels = set()
        for model in models:
            orig_key = f"pathology_{model}"
            finetuned_key = f"pathology_{model}_finetuned"
            if orig_key in original_tsne_results:
                _, labels, _ = original_tsne_results[orig_key]
                labels = pd.Series(labels).fillna('Unknown').astype(str).values
                all_labels.update(labels)
        
        unique_labels = sorted([l for l in all_labels if l != 'Unknown'])
        if 'Unknown' in all_labels:
            unique_labels.append('Unknown')
        
        # Create color and marker mappings
        color_dict = {}
        marker_dict = {}
        for i, label in enumerate(unique_labels):
            if label == 'Unknown':
                color_dict[label] = 'lightgray'
                marker_dict[label] = 'o'
            elif label in tcga_color_groups:
                color_dict[label] = tcga_color_groups[label]
                marker_dict[label] = markers[i % len(markers)]
            else:
                color_dict[label] = plt.cm.Set3(i % 12)
                marker_dict[label] = markers[i % len(markers)]
        
        # Plot embeddings - each row is a model, column 0 is original, column 1 is finetuned
        for row_idx, model in enumerate(models):
            # Plot original embeddings (left column)
            key = f"pathology_{model}"
            if key in original_tsne_results:
                ax = fig.add_subplot(gs[row_idx, 0])
                embeddings_2d, labels, indices = original_tsne_results[key]
                
                # Convert labels
                labels = pd.Series(labels).fillna('Unknown').astype(str).values
                
                # Plot each cancer type
                for label in unique_labels:
                    mask = labels == label
                    if np.any(mask):
                        color = color_dict[label]
                        marker = marker_dict[label]
                        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                                 c=[color], marker=marker, alpha=0.6, s=50, 
                                 edgecolors='black', linewidth=0.4)
                
                ax.set_title(f'{model_titles[model]} (Original)', fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel('t-SNE 1', fontsize=12)
                ax.set_ylabel('t-SNE 2', fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            # Plot finetuned embeddings (right column)
            key_finetuned = f"pathology_{model}_finetuned"
            if key_finetuned in finetuned_tsne_results:
                ax = fig.add_subplot(gs[row_idx, 1])
                embeddings_2d, labels, indices = finetuned_tsne_results[key_finetuned]
                
                # Convert labels
                labels = pd.Series(labels).fillna('Unknown').astype(str).values
                
                # Plot each cancer type
                for label in unique_labels:
                    mask = labels == label
                    if np.any(mask):
                        color = color_dict[label]
                        marker = marker_dict[label]
                        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                                 c=[color], marker=marker, alpha=0.6, s=50, 
                                 edgecolors='black', linewidth=0.4)
                
                ax.set_title(f'{model_titles[model]} (Fine-tuned)', fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel('t-SNE 1', fontsize=12)
                ax.set_ylabel('t-SNE 2', fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        # Create legend in the 3rd column (spanning all rows)
        ax_legend = fig.add_subplot(gs[:, 2])
        ax_legend.axis('off')
        
        # Create legend entries
        legend_entries = []
        for label in unique_labels:
            color = color_dict[label]
            marker = marker_dict[label]
            
            # Get description
            description = tcga_descriptions.get(label, label)
            
            # Wrap long descriptions
            wrapped_desc = textwrap.fill(description, width=30)
            
            # Create legend entry with marker and description
            legend_entries.append(
                ax_legend.scatter([], [], c=[color], marker=marker, s=100, 
                                edgecolors='black', linewidth=0.1, alpha=0.8,
                                label=f"{label}:\n{wrapped_desc}")
            )
        
        # Add legend spanning full height with proper boundaries
        legend = ax_legend.legend(handles=legend_entries, loc='center', ncol=1, 
                                 frameon=True, fancybox=True, shadow=True,
                                 fontsize=12, title='TCGA Cancer Types', title_fontsize=12,
                                 labelspacing=0.75, handletextpad=0.4,
                                 bbox_to_anchor=(0.5, 0.5), bbox_transform=ax_legend.transAxes,
                                 borderaxespad=0)
        
        # Save the figure
        filename_base = os.path.join(VISUALIZATION_DIR, 'comparisons', 
                                    'tsne_original_vs_finetuned_pathology')
        plt.savefig(f"{filename_base}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"Saved original vs finetuned comparison: {filename_base}.pdf/svg")
    
    def load_original_embeddings(self, model_name):
        """Load original embeddings and metadata."""
        embeddings_path = os.path.join(EMBEDDINGS_DIR, f'pathology_{model_name}_embeddings.pkl')
        
        if not os.path.exists(embeddings_path):
            print(f"Warning: Original embeddings not found for {model_name}")
            return None
        
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def load_finetuned_model(self, model_name):
        """Load fine-tuned model."""
        model_path = os.path.join(FINETUNED_MODELS_DIR, f'{model_name}_classifier.pt')
        
        if not os.path.exists(model_path):
            print(f"Warning: Fine-tuned model not found for {model_name}")
            return None
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model
        model = SimpleClassifier(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_classes=checkpoint['num_classes']
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Store normalization parameters
        model.train_mean = checkpoint.get('train_mean')
        model.train_std = checkpoint.get('train_std')
        
        if model.train_mean is not None:
            model.train_mean = model.train_mean.to(self.device)
            model.train_std = model.train_std.to(self.device)
        
        return model, checkpoint['label_encoder']
    
    def generate_finetuned_embeddings(self, model, original_embeddings):
        """Generate embeddings using fine-tuned model."""
        model.eval()
        
        # Convert to tensor
        embeddings_tensor = torch.FloatTensor(original_embeddings).to(self.device)
        
        # Apply normalization if available
        if hasattr(model, 'train_mean') and model.train_mean is not None:
            embeddings_tensor = (embeddings_tensor - model.train_mean) / model.train_std
        
        # Process in batches
        batch_size = 256
        all_embeddings = []
        
        for i in range(0, len(embeddings_tensor), batch_size):
            batch = embeddings_tensor[i:i+batch_size]
            batch_embeddings = model.get_embeddings(batch)
            all_embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        finetuned_embeddings = np.vstack(all_embeddings)
        
        return finetuned_embeddings
    
    def run_visualization(self):
        """Run t-SNE visualization for all fine-tuned models."""
        print("\n=== Running Fine-tuned t-SNE Visualization ===")
        
        all_tsne_results = {}
        original_tsne_results = {}
        
        for model in MODELS:
            key = f"pathology_{model}_finetuned"
            print(f"\nProcessing {model}...")
            
            # Load original embeddings
            original_data = self.load_original_embeddings(model)
            if original_data is None:
                continue
            
            # Check if fine-tuned embeddings already exist
            finetuned_path = os.path.join(OUTPUT_DIR, f'pathology_{model}_finetuned_embeddings.pkl')
            
            if os.path.exists(finetuned_path):
                # Load existing fine-tuned embeddings
                print(f"Loading existing fine-tuned embeddings for {model}...")
                with open(finetuned_path, 'rb') as f:
                    finetuned_data = pickle.load(f)
                embeddings = finetuned_data['embeddings']
            else:
                # Generate new fine-tuned embeddings
                print(f"Generating fine-tuned embeddings for {model}...")
                model_nn, label_encoder = self.load_finetuned_model(model)
                if model_nn is None:
                    continue
                
                embeddings = self.generate_finetuned_embeddings(model_nn, original_data['embeddings'])
                
                # Save fine-tuned embeddings
                output_data = {
                    'embeddings': embeddings,
                    'project_ids': original_data['project_ids'],
                    'patient_ids': original_data.get('patient_ids'),
                    'label_encoder': label_encoder,
                    'original_shape': original_data['embeddings'].shape,
                    'finetuned_shape': embeddings.shape
                }
                
                with open(finetuned_path, 'wb') as f:
                    pickle.dump(output_data, f)
                print(f"Saved fine-tuned embeddings to {finetuned_path}")
            
            # Get labels
            project_ids = original_data['project_ids']
            
            # Compute t-SNE for finetuned embeddings
            embeddings_2d, indices = self.compute_tsne(embeddings)
            
            # Handle sampling for labels
            if indices is not None:
                project_ids_plot = [project_ids[i] for i in indices]
            else:
                project_ids_plot = project_ids
            
            # Store for comparison plot
            all_tsne_results[key] = (embeddings_2d, project_ids_plot, indices)
            
            # Create individual plot with separate legend
            title = f't-SNE: {model.upper()} - Pathology Text (Fine-tuned)'
            filename_base = os.path.join(VISUALIZATION_DIR, 'individual', f'tsne_pathology_{model}_finetuned')
            
            self.create_tsne_plot_with_separate_legend(
                embeddings_2d, project_ids_plot, title, filename_base
            )
            
            # Compute t-SNE for original embeddings
            print(f"Computing t-SNE for original {model} embeddings...")
            original_embeddings_2d, original_indices = self.compute_tsne(original_data['embeddings'])
            
            # Handle sampling for original labels
            if original_indices is not None:
                original_project_ids_plot = [project_ids[i] for i in original_indices]
            else:
                original_project_ids_plot = project_ids
            
            # Store original t-SNE results
            original_key = f"pathology_{model}"
            original_tsne_results[original_key] = (original_embeddings_2d, original_project_ids_plot, original_indices)
        
        # Create comparison plot
        if all_tsne_results:
            self.create_model_comparison_plot(all_tsne_results)
            
        # Create original vs finetuned comparison
        if original_tsne_results and all_tsne_results:
            print("\nCreating original vs fine-tuned comparison...")
            self.create_original_vs_finetuned_comparison(original_tsne_results, all_tsne_results)
        
        print("\n=== Fine-tuned Visualization Complete ===")


def main():
    """Main execution"""
    visualizer = FinetunedTSNEVisualizer()
    visualizer.run_visualization()


if __name__ == "__main__":
    main()