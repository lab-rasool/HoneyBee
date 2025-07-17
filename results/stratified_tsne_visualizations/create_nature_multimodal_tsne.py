#!/usr/bin/env python3
"""
Generate publication-quality t-SNE visualizations for Nature publication.
Creates a 4-panel figure with three multimodal approaches and a shared legend.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import pickle
import warnings
warnings.filterwarnings('ignore')

# Nature publication standards
NATURE_CONFIG = {
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'figure.figsize': (7.2, 7.2),  # Square format for 4 panels
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'font.family': 'Arial',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 30,  # Smaller markers for t-SNE
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Consistent color scheme for sex
SEX_COLORS = {
    'male': '#4A90E2',      # Blue
    'female': '#E63946',    # Red
    'unknown': '#95A5A6',   # Gray
}

def setup_nature_style():
    """Configure matplotlib for Nature standards."""
    for key, value in NATURE_CONFIG.items():
        matplotlib.rcParams[key] = value

class NatureTSNEVisualizer:
    """Generate Nature-compliant t-SNE visualizations for multimodal embeddings"""
    
    def __init__(self):
        self.patient_data = None
        self.embeddings_data = {}
        self.patient_modality_map = {}
        
    def load_data(self):
        """Load patient metadata and embeddings"""
        print("Loading patient metadata...")
        self.patient_data = pd.read_csv('/mnt/f/Projects/HoneyBee/results/shared_data/embeddings/patient_data_with_embeddings.csv')
        
        # Standardize sex values
        self.patient_data['gender_clean'] = self.patient_data['gender'].str.lower().fillna('unknown')
        self.patient_data.loc[~self.patient_data['gender_clean'].isin(['male', 'female']), 'gender_clean'] = 'unknown'
        
        print("Loading embeddings...")
        self._load_embeddings()
        
        print("Building patient-modality mapping...")
        self._build_patient_modality_map()
        
    def _load_embeddings(self):
        """Load embeddings from pickle and numpy files"""
        for modality in ['clinical', 'molecular', 'pathology', 'wsi']:
            pkl_path = f'/mnt/f/Projects/HoneyBee/results/shared_data/embeddings/{modality}_embeddings.pkl'
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                self.embeddings_data[modality] = data
                print(f"  {modality.upper()} embeddings shape: {data['X'].shape}")
            except Exception as e:
                print(f"  Error loading {modality}: {e}")
        
        # Load radiology from npy file
        try:
            radiology_emb = np.load('/mnt/f/Projects/HoneyBee/results/shared_data/embeddings/radiology_embeddings.npy')
            self.embeddings_data['radiology'] = {
                'X': radiology_emb,
                'patient_ids': self.patient_data['case_id'].iloc[:len(radiology_emb)].values
            }
            print(f"  Radiology embeddings shape: {radiology_emb.shape}")
        except Exception as e:
            print(f"  Error loading radiology: {e}")
    
    def _build_patient_modality_map(self):
        """Build mapping of patients to their available modalities"""
        for modality_name, modality_data in self.embeddings_data.items():
            patient_ids = modality_data['patient_ids']
            
            if modality_name == 'molecular':
                case_id_to_idx = {pid: i for i, pid in enumerate(self.patient_data['case_id'])}
                case_sub_to_idx = {pid: i for i, pid in enumerate(self.patient_data['case_submitter_id'])}
                
                for i, pid in enumerate(patient_ids):
                    patient_idx = case_id_to_idx.get(pid) or case_sub_to_idx.get(pid)
                    if patient_idx is not None:
                        case_id = self.patient_data.iloc[patient_idx]['case_id']
                        if case_id not in self.patient_modality_map:
                            self.patient_modality_map[case_id] = []
                        self.patient_modality_map[case_id].append((modality_name, i))
            else:
                n_samples = len(patient_ids)
                for i in range(min(n_samples, len(self.patient_data))):
                    case_id = self.patient_data.iloc[i]['case_id']
                    if case_id not in self.patient_modality_map:
                        self.patient_modality_map[case_id] = []
                    self.patient_modality_map[case_id].append((modality_name, i))
    
    def create_multimodal_embeddings_batch(self, patient_list):
        """Create multimodal embeddings efficiently using batch processing"""
        print(f"\nCreating multimodal embeddings for {len(patient_list)} patients...")
        
        n_patients = len(patient_list)
        
        # Determine maximum dimensions
        max_concat_dim = sum(emb['X'].shape[1] for emb in self.embeddings_data.values())
        max_mean_dim = 1024
        max_kron_dim = 100
        
        # Initialize result arrays
        concat_embeddings = np.zeros((n_patients, max_concat_dim))
        mean_embeddings = np.zeros((n_patients, max_mean_dim))
        kron_embeddings = np.zeros((n_patients, max_kron_dim))
        
        # Track valid embeddings
        valid_concat = []
        valid_mean = []
        valid_kron = []
        
        # Process patients
        for idx, patient_id in enumerate(patient_list):
            if idx % 1000 == 0:
                print(f"  Processing patient {idx}/{n_patients}...")
            
            available_mods = self.patient_modality_map.get(patient_id, [])
            
            if len(available_mods) < 2:
                continue
            
            # Get embeddings for this patient
            patient_embeddings = {}
            for mod_name, mod_idx in available_mods:
                emb = self.embeddings_data[mod_name]['X'][mod_idx]
                if not np.all(np.isnan(emb)):
                    patient_embeddings[mod_name] = emb
            
            if len(patient_embeddings) < 2:
                continue
            
            # 1. Concatenation
            concat_start = 0
            for mod_name in sorted(patient_embeddings.keys()):
                emb = patient_embeddings[mod_name]
                emb_len = len(emb)
                concat_embeddings[idx, concat_start:concat_start+emb_len] = emb
                concat_start += emb_len
            valid_concat.append(idx)
            
            # 2. Mean pooling
            mean_parts = []
            for mod_name in sorted(patient_embeddings.keys()):
                emb = patient_embeddings[mod_name]
                if len(emb) < max_mean_dim:
                    emb = np.pad(emb, (0, max_mean_dim - len(emb)), mode='constant')
                else:
                    emb = emb[:max_mean_dim]
                mean_parts.append(emb)
            mean_embeddings[idx] = np.mean(mean_parts, axis=0)
            valid_mean.append(idx)
            
            # 3. Kronecker product
            sorted_mods = sorted(patient_embeddings.keys())
            if len(sorted_mods) >= 2:
                emb1 = patient_embeddings[sorted_mods[0]][:10]
                emb2 = patient_embeddings[sorted_mods[1]][:10]
                kron = np.kron(emb1, emb2)
                kron_embeddings[idx, :len(kron)] = kron
                valid_kron.append(idx)
        
        return {
            'concat': (concat_embeddings[valid_concat], valid_concat),
            'mean_pool': (mean_embeddings[valid_mean], valid_mean),
            'kronecker': (kron_embeddings[valid_kron], valid_kron),
            'patient_list': patient_list
        }
    
    def create_nature_figure(self, output_path):
        """Create Nature-compliant 4-panel figure with t-SNE visualizations"""
        # Get patients with 2+ modalities
        multimodal_patients = [pid for pid, mods in self.patient_modality_map.items() if len(mods) >= 2]
        print(f"\nTotal patients with 2+ modalities: {len(multimodal_patients)}")
        
        # Create multimodal embeddings
        multimodal_results = self.create_multimodal_embeddings_batch(multimodal_patients)
        patient_list = multimodal_results['patient_list']
        
        # Create figure with GridSpec (2x2 grid)
        fig = plt.figure(figsize=NATURE_CONFIG['figure.figsize'])
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # Process each fusion method
        fusion_methods = ['concat', 'mean_pool', 'kronecker']
        fusion_titles = ['Concatenation', 'Mean pooling', 'Kronecker product']
        
        for idx, (fusion_method, title) in enumerate(zip(fusion_methods, fusion_titles)):
            # Determine subplot position
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])
            
            # Get embeddings
            fusion_embeddings, valid_indices = multimodal_results[fusion_method]
            
            if len(fusion_embeddings) == 0:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                continue
            
            # Get patient data for valid embeddings
            valid_patient_ids = [patient_list[i] for i in valid_indices]
            fusion_patient_data = self.patient_data[self.patient_data['case_id'].isin(valid_patient_ids)].iloc[:len(fusion_embeddings)]
            
            # Compute t-SNE
            print(f"\nComputing t-SNE for {fusion_method}...")
            perplexity = min(30, len(fusion_embeddings)//2)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
            embeddings_2d = tsne.fit_transform(fusion_embeddings)
            
            # Plot by sex
            for sex in ['male', 'female', 'unknown']:
                mask = fusion_patient_data['gender_clean'] == sex
                if np.any(mask):
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                              c=SEX_COLORS[sex], s=20, alpha=0.6, 
                              edgecolors='black', linewidth=0.5)
            
            # Styling
            ax.set_xlabel('t-SNE 1', fontsize=9)
            ax.set_ylabel('t-SNE 2', fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Add panel label
            panel_label = chr(65 + idx)  # A, B, C
            ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')
        
        # Use 4th subplot for legend
        ax_legend = fig.add_subplot(gs[1, 1])
        ax_legend.axis('off')
        
        # Create legend elements
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=SEX_COLORS['male'], 
                   markersize=8, label='Male', markeredgecolor='black', markeredgewidth=0.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=SEX_COLORS['female'], 
                   markersize=8, label='Female', markeredgecolor='black', markeredgewidth=0.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=SEX_COLORS['unknown'], 
                   markersize=8, label='Unknown', markeredgecolor='black', markeredgewidth=0.5),
        ]
        
        # Add legend in center
        legend = ax_legend.legend(handles=legend_elements, loc='center', 
                                 frameon=False, fontsize=9, ncol=1,
                                 title='Sex', title_fontsize=10)
        
        # Save figure
        plt.tight_layout()
        
        for fmt in ['pdf', 'png', 'svg', 'eps']:
            filepath = f"{output_path}.{fmt}"
            plt.savefig(filepath, format=fmt, bbox_inches='tight', 
                        transparent=True, pad_inches=0.1)
            print(f"Saved: {filepath}")
        
        plt.close()

def main():
    """Generate Nature-compliant multimodal t-SNE visualizations"""
    print("=" * 80)
    print("GENERATING MULTIMODAL t-SNE VISUALIZATIONS FOR NATURE PUBLICATION")
    print("=" * 80)
    
    # Setup
    setup_nature_style()
    
    # Create visualizer
    visualizer = NatureTSNEVisualizer()
    
    # Load data
    visualizer.load_data()
    
    # Create figure
    output_path = '/mnt/f/Projects/HoneyBee/results/stratified_tsne_visualizations/nature_multimodal_tsne_by_sex'
    visualizer.create_nature_figure(output_path)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved as: {output_path}.*")
    print("Formats: PDF (primary), PNG, SVG, EPS")
    print("\nFigure contains:")
    print("  - Panel A: Concatenation fusion")
    print("  - Panel B: Mean pooling fusion")
    print("  - Panel C: Kronecker product fusion")
    print("  - Panel D: Legend and method descriptions")

if __name__ == "__main__":
    main()