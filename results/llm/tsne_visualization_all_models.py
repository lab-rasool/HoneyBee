"""
t-SNE visualization script for all LLM embeddings (GatorTron, Qwen, Med-Gemma, Llama).
Generates PDF files with separate legends for better readability.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class LLMTSNEVisualizer:
    """Class to handle t-SNE visualization for all LLM embeddings"""
    
    def __init__(self, embeddings_dir, output_dir):
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.embeddings_data = {}
        self.metadata = None
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'individual'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
        
    def load_embeddings(self):
        """Load all LLM embeddings and metadata"""
        print("Loading LLM embeddings...")
        
        models = ['gatortron', 'qwen', 'medgemma', 'llama']
        text_types = ['clinical', 'pathology']
        
        for text_type in text_types:
            for model in models:
                key = f"{text_type}_{model}"
                filename = f"{text_type}_{model}_embeddings.pkl"
                filepath = os.path.join(self.embeddings_dir, filename)
                
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    self.embeddings_data[key] = data
                    print(f"  Loaded {key}: shape {data['embeddings'].shape}")
                    
                    # Store metadata from first loaded file
                    if self.metadata is None:
                        self.metadata = {
                            'patient_ids': data['patient_ids'],
                            'project_ids': data['project_ids']
                        }
                except Exception as e:
                    print(f"  Warning: Could not load {key}: {e}")
    
    def get_tcga_color_scheme(self):
        """Return TCGA-specific color scheme"""
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
        """Return TCGA cancer type descriptions"""
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
        """Create a t-SNE plot with a separate legend file"""
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
        ax.set_title(title, fontsize=14, fontweight='bold')
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
        """Compute t-SNE on embeddings"""
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
    
    def create_model_comparison_plot(self, all_tsne_results, text_type):
        """Create a 2x2 comparison plot for all models"""
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
            key = f"{text_type}_{model}"
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
            
            ax.set_title(model_titles[model], fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1', fontsize=10)
            ax.set_ylabel('t-SNE 2', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.suptitle(f'{text_type.capitalize()} Text Embeddings - Model Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename_base = os.path.join(self.output_dir, 'comparisons', 
                                    f'tsne_comparison_{text_type}')
        plt.savefig(f"{filename_base}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{filename_base}.svg", format='svg', bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: {filename_base}.pdf/svg")
    
    def run_visualization(self):
        """Run t-SNE visualization for all models"""
        print("\n=== Running t-SNE Visualization ===")
        
        # Load embeddings
        self.load_embeddings()
        
        if not self.embeddings_data:
            print("No embeddings found!")
            return
        
        text_types = ['clinical', 'pathology']
        models = ['gatortron', 'qwen', 'medgemma', 'llama']
        
        for text_type in text_types:
            print(f"\n--- Processing {text_type} embeddings ---")
            
            all_tsne_results = {}
            
            for model in models:
                key = f"{text_type}_{model}"
                if key not in self.embeddings_data:
                    print(f"Skipping {key} - not found")
                    continue
                
                print(f"\nProcessing {model}...")
                
                # Get embeddings and labels
                embeddings = self.embeddings_data[key]['embeddings']
                project_ids = self.embeddings_data[key]['project_ids']
                
                # Compute t-SNE
                embeddings_2d, indices = self.compute_tsne(embeddings)
                
                # Handle sampling for labels
                if indices is not None:
                    project_ids_plot = [project_ids[i] for i in indices]
                else:
                    project_ids_plot = project_ids
                
                # Store for comparison plot
                all_tsne_results[key] = (embeddings_2d, project_ids_plot, indices)
                
                # Create individual plot
                title = f't-SNE: {model.upper()} - {text_type.capitalize()} Text'
                filename_base = os.path.join(self.output_dir, 'individual', f'tsne_{text_type}_{model}')
                
                self.create_tsne_plot_with_separate_legend(
                    embeddings_2d, project_ids_plot, title, filename_base
                )
            
            # Create comparison plot
            if all_tsne_results:
                self.create_model_comparison_plot(all_tsne_results, text_type)
        
        print("\n=== Visualization Complete ===")


def main():
    """Main execution"""
    embeddings_dir = "/mnt/f/Projects/HoneyBee/results/llm/embeddings"
    output_dir = "/mnt/f/Projects/HoneyBee/results/llm/tsne_visualizations"
    
    visualizer = LLMTSNEVisualizer(embeddings_dir, output_dir)
    visualizer.run_visualization()


if __name__ == "__main__":
    main()