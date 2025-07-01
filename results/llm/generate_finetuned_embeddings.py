#!/usr/bin/env python3
"""
Generate embeddings using fine-tuned pathology classification models.
Extracts representations from the penultimate layer of the trained classifiers.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style
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


class FinetunedEmbeddingsGenerator:
    """Generate embeddings using fine-tuned models."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        os.makedirs(os.path.join(VISUALIZATION_DIR, 'individual'), exist_ok=True)
        os.makedirs(os.path.join(VISUALIZATION_DIR, 'comparison'), exist_ok=True)
        
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
    
    def get_tcga_color_scheme(self):
        """Return TCGA-specific color scheme."""
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
    
    def create_tsne_visualization(self, embeddings, labels, title, filename_base):
        """Create t-SNE visualization of embeddings."""
        print(f"  Creating t-SNE visualization for {title}...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_scaled)
        
        # Get color scheme
        tcga_colors = self.get_tcga_color_scheme()
        
        # Convert labels to strings
        labels = pd.Series(labels).fillna('Unknown').astype(str).values
        unique_labels = sorted(np.unique(labels))
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        for label in unique_labels:
            mask = labels == label
            color = tcga_colors.get(label, plt.cm.Set3(hash(label) % 12))
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[color], label=label, alpha=0.6, s=50,
                       edgecolors='black', linewidth=0.5)
        
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Create legend outside plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  ncol=1, fontsize=8, frameon=True)
        
        plt.tight_layout()
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comparison_plot(self, all_embeddings):
        """Create comparison plot of original vs fine-tuned embeddings."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        tcga_colors = self.get_tcga_color_scheme()
        
        for idx, model in enumerate(MODELS):
            if model not in all_embeddings:
                continue
            
            # Original embeddings
            ax_orig = axes[0, idx]
            orig_data = all_embeddings[model]['original']
            self._plot_tsne_subplot(ax_orig, orig_data['tsne'], orig_data['labels'],
                                   f'{model.upper()}\n(Original)', tcga_colors)
            
            # Fine-tuned embeddings
            ax_fine = axes[1, idx]
            fine_data = all_embeddings[model]['finetuned']
            self._plot_tsne_subplot(ax_fine, fine_data['tsne'], fine_data['labels'],
                                   f'{model.upper()}\n(Fine-tuned)', tcga_colors)
        
        plt.suptitle('Pathology Embeddings: Original vs Fine-tuned Models', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = os.path.join(VISUALIZATION_DIR, 'comparison', 
                               'embeddings_comparison_all_models')
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: {filename}.png/pdf")
    
    def _plot_tsne_subplot(self, ax, embeddings_2d, labels, title, color_scheme):
        """Helper to plot t-SNE in subplot."""
        labels = pd.Series(labels).fillna('Unknown').astype(str).values
        unique_labels = sorted(np.unique(labels))
        
        for label in unique_labels:
            mask = labels == label
            color = color_scheme.get(label, plt.cm.Set3(hash(label) % 12))
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[color], alpha=0.6, s=30,
                      edgecolors='black', linewidth=0.3)
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=8)
        ax.set_ylabel('t-SNE 2', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def process_model(self, model_name):
        """Process a single model."""
        print(f"\n{'='*60}")
        print(f"Processing {model_name.upper()}")
        print(f"{'='*60}")
        
        # Load original embeddings
        original_data = self.load_original_embeddings(model_name)
        if original_data is None:
            return None
        
        print(f"Loaded original embeddings: shape {original_data['embeddings'].shape}")
        
        # Load fine-tuned model
        model, label_encoder = self.load_finetuned_model(model_name)
        if model is None:
            return None
        
        # Generate fine-tuned embeddings
        print("Generating fine-tuned embeddings...")
        finetuned_embeddings = self.generate_finetuned_embeddings(
            model, original_data['embeddings']
        )
        
        print(f"Generated fine-tuned embeddings: shape {finetuned_embeddings.shape}")
        
        # Save fine-tuned embeddings
        output_data = {
            'embeddings': finetuned_embeddings,
            'project_ids': original_data['project_ids'],
            'patient_ids': original_data.get('patient_ids'),
            'label_encoder': label_encoder,
            'original_shape': original_data['embeddings'].shape,
            'finetuned_shape': finetuned_embeddings.shape
        }
        
        output_path = os.path.join(OUTPUT_DIR, f'pathology_{model_name}_finetuned_embeddings.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Saved fine-tuned embeddings to {output_path}")
        
        # Create individual visualizations
        print("Creating visualizations...")
        
        # Original embeddings t-SNE
        orig_tsne = self._compute_tsne(original_data['embeddings'])
        orig_filename = os.path.join(VISUALIZATION_DIR, 'individual', 
                                    f'tsne_pathology_{model_name}_original')
        self.create_tsne_visualization(
            original_data['embeddings'], 
            original_data['project_ids'],
            f'{model_name.upper()} - Original Pathology Embeddings',
            orig_filename
        )
        
        # Fine-tuned embeddings t-SNE
        fine_tsne = self._compute_tsne(finetuned_embeddings)
        fine_filename = os.path.join(VISUALIZATION_DIR, 'individual',
                                    f'tsne_pathology_{model_name}_finetuned')
        self.create_tsne_visualization(
            finetuned_embeddings,
            original_data['project_ids'],
            f'{model_name.upper()} - Fine-tuned Pathology Embeddings',
            fine_filename
        )
        
        return {
            'original': {
                'embeddings': original_data['embeddings'],
                'tsne': orig_tsne,
                'labels': original_data['project_ids']
            },
            'finetuned': {
                'embeddings': finetuned_embeddings,
                'tsne': fine_tsne,
                'labels': original_data['project_ids']
            }
        }
    
    def _compute_tsne(self, embeddings):
        """Compute t-SNE embeddings."""
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        return tsne.fit_transform(embeddings_scaled)
    
    def run(self):
        """Run the complete pipeline."""
        print("Starting Fine-tuned Embeddings Generation Pipeline")
        print("=" * 60)
        
        all_embeddings = {}
        
        # Process each model
        for model in MODELS:
            try:
                result = self.process_model(model)
                if result is not None:
                    all_embeddings[model] = result
            except Exception as e:
                print(f"Error processing {model}: {e}")
        
        # Create comparison plot
        if all_embeddings:
            print("\nCreating comparison visualizations...")
            self.create_comparison_plot(all_embeddings)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Fine-tuned embeddings saved to: {OUTPUT_DIR}/")
        print(f"Visualizations saved to: {VISUALIZATION_DIR}/")
        print("\nGenerated files:")
        print(f"  - {OUTPUT_DIR}/pathology_*_finetuned_embeddings.pkl")
        print(f"  - {VISUALIZATION_DIR}/individual/tsne_*.png/pdf")
        print(f"  - {VISUALIZATION_DIR}/comparison/embeddings_comparison_all_models.png/pdf")


def main():
    """Main execution function."""
    generator = FinetunedEmbeddingsGenerator()
    generator.run()


if __name__ == "__main__":
    main()