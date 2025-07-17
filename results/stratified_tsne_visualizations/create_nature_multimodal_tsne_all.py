#!/usr/bin/env python3
"""
Generate publication-quality t-SNE visualizations for Nature publication.
Creates 4-panel figures with three multimodal approaches and a shared legend.
Generates separate figures for sex, age groups, and cancer types.
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

# Consistent color schemes
SEX_COLORS = {
    'male': '#4A90E2',      # Blue
    'female': '#E63946',    # Red
    'unknown': '#95A5A6',   # Gray
}

AGE_COLORS = {
    '<40': '#3498db',       # Bright blue
    '40-49': '#2ecc71',     # Green
    '50-59': '#f39c12',     # Orange
    '60-69': '#e74c3c',     # Red
    '70+': '#9b59b6',       # Purple
    'Unknown': '#95a5a6',   # Gray
}

# TCGA project colors (grouped by organ system)
TCGA_COLORS = {
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
        self.multimodal_results = None
        
    def load_data(self):
        """Load patient metadata and embeddings"""
        print("Loading patient metadata...")
        self.patient_data = pd.read_csv('/mnt/f/Projects/HoneyBee/results/shared_data/embeddings/patient_data_with_embeddings.csv')
        
        # Standardize sex values
        self.patient_data['gender_clean'] = self.patient_data['gender'].str.lower().fillna('unknown')
        self.patient_data.loc[~self.patient_data['gender_clean'].isin(['male', 'female']), 'gender_clean'] = 'unknown'
        
        # Create age groups
        self.patient_data['age_group'] = self.patient_data['age_at_index'].apply(self._categorize_age)
        
        print("Loading embeddings...")
        self._load_embeddings()
        
        print("Building patient-modality mapping...")
        self._build_patient_modality_map()
        
    def _categorize_age(self, age):
        """Categorize age into groups"""
        if pd.isna(age) or age == 'Unknown':
            return 'Unknown'
        try:
            age_val = float(age)
            if age_val < 40:
                return '<40'
            elif age_val < 50:
                return '40-49'
            elif age_val < 60:
                return '50-59'
            elif age_val < 70:
                return '60-69'
            else:
                return '70+'
        except:
            return 'Unknown'
        
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
    
    def compute_tsne_once(self):
        """Compute t-SNE embeddings once and store them"""
        # Get patients with 2+ modalities
        multimodal_patients = [pid for pid, mods in self.patient_modality_map.items() if len(mods) >= 2]
        print(f"\nTotal patients with 2+ modalities: {len(multimodal_patients)}")
        
        # Create multimodal embeddings
        self.multimodal_results = self.create_multimodal_embeddings_batch(multimodal_patients)
        
        # Compute t-SNE for each fusion method
        self.tsne_embeddings = {}
        fusion_methods = ['concat', 'mean_pool', 'kronecker']
        
        for fusion_method in fusion_methods:
            fusion_embeddings, valid_indices = self.multimodal_results[fusion_method]
            
            if len(fusion_embeddings) == 0:
                continue
            
            print(f"\nComputing t-SNE for {fusion_method}...")
            perplexity = min(30, len(fusion_embeddings)//2)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
            embeddings_2d = tsne.fit_transform(fusion_embeddings)
            
            self.tsne_embeddings[fusion_method] = {
                'embeddings_2d': embeddings_2d,
                'valid_indices': valid_indices
            }
    
    def create_nature_figure(self, color_by='sex', output_path=None):
        """Create Nature-compliant 4-panel figure with t-SNE visualizations"""
        if self.multimodal_results is None:
            self.compute_tsne_once()
        
        # Create figure with GridSpec (2x2 grid)
        fig = plt.figure(figsize=NATURE_CONFIG['figure.figsize'])
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # Process each fusion method
        fusion_methods = ['concat', 'mean_pool', 'kronecker']
        fusion_titles = ['Concatenation', 'Mean pooling', 'Kronecker product']
        
        # Determine color scheme and legend elements based on color_by
        if color_by == 'sex':
            color_dict = SEX_COLORS
            legend_title = 'Sex'
        elif color_by == 'age':
            color_dict = AGE_COLORS
            legend_title = 'Age group'
        elif color_by == 'cancer_type':
            color_dict = TCGA_COLORS
            legend_title = 'Cancer type'
        
        for idx, (fusion_method, title) in enumerate(zip(fusion_methods, fusion_titles)):
            # Determine subplot position
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])
            
            if fusion_method not in self.tsne_embeddings:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                continue
            
            # Get pre-computed t-SNE embeddings
            embeddings_2d = self.tsne_embeddings[fusion_method]['embeddings_2d']
            valid_indices = self.tsne_embeddings[fusion_method]['valid_indices']
            
            # Get patient data for valid embeddings
            patient_list = self.multimodal_results['patient_list']
            valid_patient_ids = [patient_list[i] for i in valid_indices]
            fusion_patient_data = self.patient_data[self.patient_data['case_id'].isin(valid_patient_ids)].iloc[:len(embeddings_2d)]
            
            # Get values for coloring
            if color_by == 'sex':
                color_values = fusion_patient_data['gender_clean'].values
            elif color_by == 'age':
                color_values = fusion_patient_data['age_group'].values
            elif color_by == 'cancer_type':
                color_values = fusion_patient_data['project_id'].values
            
            # Plot each category
            unique_values = pd.Series(color_values).unique()
            
            # For cancer types, use markers to distinguish similar colors
            if color_by == 'cancer_type':
                # Markers for visual distinction
                markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd', 
                          '<', '>', '8', 'P', 'H', '+', 'x', '|', '_']
                
                # Sort values for consistent marker assignment
                sorted_values = sorted([v for v in unique_values if v != 'Unknown'])
                if 'Unknown' in unique_values:
                    sorted_values.append('Unknown')
                
                for i, value in enumerate(sorted_values):
                    mask = color_values == value
                    if np.any(mask):
                        if value in TCGA_COLORS:
                            color = TCGA_COLORS[value]
                            marker = markers[i % len(markers)]
                        else:
                            color = 'lightgray'
                            marker = 'o'
                        
                        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                                  c=color, marker=marker, s=30, alpha=0.6, 
                                  edgecolors='black', linewidth=0.5)
            else:
                # For sex and age, use simple circles
                for value in unique_values:
                    if value in color_dict:
                        mask = color_values == value
                        if np.any(mask):
                            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                                      c=color_dict[value], s=20, alpha=0.6, 
                                      edgecolors='black', linewidth=0.5)
                    else:
                        # Fallback color for unknown values
                        mask = color_values == value
                        if np.any(mask):
                            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                                      c='lightgray', s=20, alpha=0.6, 
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
        
        # Create legend elements based on color_by
        legend_elements = []
        
        if color_by == 'sex':
            for label, color in SEX_COLORS.items():
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                           markersize=8, label=label.capitalize(), 
                           markeredgecolor='black', markeredgewidth=0.5)
                )
        elif color_by == 'age':
            for label in ['<40', '40-49', '50-59', '60-69', '70+', 'Unknown']:
                if label in AGE_COLORS:
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=AGE_COLORS[label], 
                               markersize=8, label=label, 
                               markeredgecolor='black', markeredgewidth=0.5)
                    )
        elif color_by == 'cancer_type':
            # Get unique cancer types from the data
            all_cancer_types = []
            for fusion_method in fusion_methods:
                if fusion_method in self.tsne_embeddings:
                    valid_indices = self.tsne_embeddings[fusion_method]['valid_indices']
                    patient_list = self.multimodal_results['patient_list']
                    valid_patient_ids = [patient_list[i] for i in valid_indices]
                    fusion_patient_data = self.patient_data[self.patient_data['case_id'].isin(valid_patient_ids)]
                    all_cancer_types.extend(fusion_patient_data['project_id'].unique())
            
            # Sort values for consistent marker assignment
            unique_cancer_types = sorted(list(set([v for v in all_cancer_types if v != 'Unknown'])))
            if 'Unknown' in all_cancer_types:
                unique_cancer_types.append('Unknown')
            
            # Markers for visual distinction
            markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd', 
                      '<', '>', '8', 'P', 'H', '+', 'x', '|', '_']
            
            # Create legend with appropriate layout
            n_types = len(unique_cancer_types)
            if n_types <= 10:
                ncol = 1
            elif n_types <= 20:
                ncol = 2
            else:
                ncol = 3
            
            for i, cancer_type in enumerate(unique_cancer_types):
                if cancer_type == 'Unknown':
                    color = 'lightgray'
                    marker = 'o'
                elif cancer_type in TCGA_COLORS:
                    color = TCGA_COLORS[cancer_type]
                    marker = markers[i % len(markers)]
                else:
                    color = 'lightgray'
                    marker = 'o'
                
                legend_elements.append(
                    Line2D([0], [0], marker=marker, color='w', 
                           markerfacecolor=color, 
                           markersize=6, label=cancer_type, 
                           markeredgecolor='black', markeredgewidth=0.5)
                )
        
        # Add legend
        if color_by == 'cancer_type' and len(legend_elements) > 10:
            # For many cancer types, use smaller font and multiple columns
            legend = ax_legend.legend(handles=legend_elements, loc='center', 
                                     frameon=False, fontsize=6, ncol=ncol,
                                     title=legend_title, title_fontsize=8,
                                     columnspacing=1.0, handletextpad=0.5)
        else:
            # For sex and age, use standard layout
            legend = ax_legend.legend(handles=legend_elements, loc='center', 
                                     frameon=False, fontsize=9, ncol=1,
                                     title=legend_title, title_fontsize=10)
        
        # Save figure
        plt.tight_layout()
        
        if output_path:
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
    
    # Create figures for different stratifications
    output_dir = '/mnt/f/Projects/HoneyBee/results/stratified_tsne_visualizations'
    
    # 1. By sex
    print("\nGenerating figure colored by sex...")
    visualizer.create_nature_figure(
        color_by='sex',
        output_path=f'{output_dir}/nature_multimodal_tsne_by_sex'
    )
    
    # 2. By age group
    print("\nGenerating figure colored by age group...")
    visualizer.create_nature_figure(
        color_by='age',
        output_path=f'{output_dir}/nature_multimodal_tsne_by_age'
    )
    
    # 3. By cancer type
    print("\nGenerating figure colored by cancer type...")
    visualizer.create_nature_figure(
        color_by='cancer_type',
        output_path=f'{output_dir}/nature_multimodal_tsne_by_cancer_type'
    )
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved in: {output_dir}")
    print("\nGenerated figures:")
    print("  - nature_multimodal_tsne_by_sex.*")
    print("  - nature_multimodal_tsne_by_age.*")
    print("  - nature_multimodal_tsne_by_cancer_type.*")
    print("\nFormats: PDF (primary), PNG, SVG, EPS")

if __name__ == "__main__":
    main()