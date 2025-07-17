#!/usr/bin/env python3
"""
Generate publication-quality t-SNE visualizations for Nature publication.
Creates figures for individual modalities stratified by sex, age, and cancer type.
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
    'figure.figsize': (10, 7.5),  # Wider format for 2x3 layout
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
    'lines.markersize': 30,
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

class NatureIndividualTSNEVisualizer:
    """Generate Nature-compliant t-SNE visualizations for individual modalities"""
    
    def __init__(self):
        self.patient_data = None
        self.embeddings_data = {}
        
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
        # Load embeddings
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
    
    def get_patient_data_for_modality(self, modality_name, modality_data):
        """Get patient data matched to modality embeddings"""
        embeddings = modality_data['X']
        patient_ids = modality_data['patient_ids']
        
        if modality_name == 'molecular':
            # Special handling for molecular data
            modality_patient_data = []
            valid_indices = []
            
            for i, pid in enumerate(patient_ids):
                mask = (self.patient_data['case_id'] == pid) | (self.patient_data['case_submitter_id'] == pid)
                if mask.any():
                    modality_patient_data.append(self.patient_data[mask].iloc[0])
                    valid_indices.append(i)
            
            if modality_patient_data:
                modality_patient_data = pd.DataFrame(modality_patient_data).reset_index(drop=True)
                embeddings = embeddings[valid_indices]
                return embeddings, modality_patient_data
            else:
                return None, None
        else:
            # For other modalities
            n_samples = len(embeddings)
            modality_patient_data = self.patient_data.iloc[:n_samples].copy()
            return embeddings, modality_patient_data
    
    def create_nature_figure(self, color_by='sex', output_path=None):
        """Create Nature-compliant figure with 5 modality panels + legend"""
        # Order of modalities for display
        modalities = ['clinical', 'pathology', 'radiology', 'molecular', 'wsi']
        modality_titles = ['Clinical', 'Pathology', 'Radiology', 'Molecular', 'WSI']
        
        # Create figure with GridSpec (2x3 grid)
        fig = plt.figure(figsize=NATURE_CONFIG['figure.figsize'])
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
        
        # Determine color scheme based on color_by
        if color_by == 'sex':
            color_dict = SEX_COLORS
            legend_title = 'Sex'
        elif color_by == 'age':
            color_dict = AGE_COLORS
            legend_title = 'Age group'
        elif color_by == 'cancer_type':
            color_dict = TCGA_COLORS
            legend_title = 'Cancer type'
        
        # Track all unique values for legend
        all_unique_values = set()
        
        for idx, (modality, title) in enumerate(zip(modalities, modality_titles)):
            # Determine subplot position (2 rows, 3 columns)
            row = idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            if modality not in self.embeddings_data:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                continue
            
            # Get embeddings and patient data
            embeddings, modality_patient_data = self.get_patient_data_for_modality(
                modality, self.embeddings_data[modality]
            )
            
            if embeddings is None or len(embeddings) < 10:
                ax.text(0.5, 0.5, 'Insufficient data', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                continue
            
            # Handle NaN values
            if np.isnan(embeddings).any():
                embeddings = np.nan_to_num(embeddings, nan=0.0)
            
            # Compute t-SNE
            print(f"\nComputing t-SNE for {modality}...")
            perplexity = min(30, len(embeddings)//2)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Get values for coloring
            if color_by == 'sex':
                color_values = modality_patient_data['gender_clean'].values
            elif color_by == 'age':
                color_values = modality_patient_data['age_group'].values
            elif color_by == 'cancer_type':
                color_values = modality_patient_data['project_id'].values
            
            # Plot each category
            unique_values = pd.Series(color_values).unique()
            all_unique_values.update(unique_values)
            
            # For cancer types, use markers
            if color_by == 'cancer_type':
                markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd', 
                          '<', '>', '8', 'P', 'H', '+', 'x', '|', '_']
                
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
            
            # Add panel label with more space from plot
            panel_label = chr(65 + idx)  # A, B, C, D, E
            ax.text(-0.20, 1.05, panel_label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')
        
        # Use 6th subplot (bottom right) for legend
        ax_legend = fig.add_subplot(gs[1, 2])
        ax_legend.axis('off')
        
        # Create legend elements
        legend_elements = []
        
        if color_by == 'sex':
            for label, color in SEX_COLORS.items():
                if label in all_unique_values or label.capitalize() in all_unique_values:
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                               markersize=8, label=label.capitalize(), 
                               markeredgecolor='black', markeredgewidth=0.5)
                    )
        elif color_by == 'age':
            age_order = ['<40', '40-49', '50-59', '60-69', '70+', 'Unknown']
            for label in age_order:
                if label in all_unique_values and label in AGE_COLORS:
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=AGE_COLORS[label], 
                               markersize=8, label=label, 
                               markeredgecolor='black', markeredgewidth=0.5)
                    )
        elif color_by == 'cancer_type':
            # Sort values
            sorted_values = sorted([v for v in all_unique_values if v != 'Unknown'])
            if 'Unknown' in all_unique_values:
                sorted_values.append('Unknown')
            
            markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd', 
                      '<', '>', '8', 'P', 'H', '+', 'x', '|', '_']
            
            n_types = len(sorted_values)
            if n_types <= 15:
                ncol = 1
            elif n_types <= 30:
                ncol = 2
            else:
                ncol = 3
            
            for i, cancer_type in enumerate(sorted_values):
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
        if color_by == 'cancer_type' and len(legend_elements) > 15:
            legend = ax_legend.legend(handles=legend_elements, loc='center', 
                                     frameon=False, fontsize=6, ncol=ncol,
                                     title=legend_title, title_fontsize=8,
                                     columnspacing=1.0, handletextpad=0.5)
        else:
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
    """Generate Nature-compliant individual modality t-SNE visualizations"""
    print("=" * 80)
    print("GENERATING INDIVIDUAL MODALITY t-SNE VISUALIZATIONS FOR NATURE PUBLICATION")
    print("=" * 80)
    
    # Setup
    setup_nature_style()
    
    # Create visualizer
    visualizer = NatureIndividualTSNEVisualizer()
    
    # Load data
    visualizer.load_data()
    
    # Create figures for different stratifications
    output_dir = '/mnt/f/Projects/HoneyBee/results/stratified_tsne_visualizations'
    
    # 1. By sex
    print("\nGenerating figure colored by sex...")
    visualizer.create_nature_figure(
        color_by='sex',
        output_path=f'{output_dir}/nature_individual_tsne_by_sex'
    )
    
    # 2. By age group
    print("\nGenerating figure colored by age group...")
    visualizer.create_nature_figure(
        color_by='age',
        output_path=f'{output_dir}/nature_individual_tsne_by_age'
    )
    
    # 3. By cancer type
    print("\nGenerating figure colored by cancer type...")
    visualizer.create_nature_figure(
        color_by='cancer_type',
        output_path=f'{output_dir}/nature_individual_tsne_by_cancer_type'
    )
    
    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved in: {output_dir}")
    print("\nGenerated figures:")
    print("  - nature_individual_tsne_by_sex.*")
    print("  - nature_individual_tsne_by_age.*")
    print("  - nature_individual_tsne_by_cancer_type.*")
    print("\nFormats: PDF (primary), PNG, SVG, EPS")

if __name__ == "__main__":
    main()