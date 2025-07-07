"""
Comprehensive t-SNE visualization script for HoneyBee multimodal embeddings.
Generates both individual modality and multimodal fusion visualizations.
Outputs PDF files with separate legends for better readability.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import pickle
import argparse
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TSNEVisualizer:
    """Class to handle t-SNE visualization for multimodal embeddings"""
    
    def __init__(self, output_dir='/mnt/f/Projects/HoneyBee/results/stratified_tsne_visualizations'):
        self.output_dir = output_dir
        self.patient_data = None
        self.embeddings_data = {}
        self.patient_modality_map = {}
        
    def load_data(self):
        """Load patient metadata and embeddings"""
        print("Loading patient metadata...")
        self.patient_data = pd.read_csv('/mnt/f/Projects/HoneyBee/results/shared_data/embeddings/patient_data_with_embeddings.csv')
        
        # Create age groups
        self.patient_data['age_group'] = self.patient_data['age_at_index'].apply(self._categorize_age)
        
        # Load embeddings
        print("Loading embeddings...")
        self._load_embeddings()
        
        # Build patient-modality mapping
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
        # Load pickle files
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
                # Special handling for molecular data
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
                # For other modalities
                n_samples = len(patient_ids)
                for i in range(min(n_samples, len(self.patient_data))):
                    case_id = self.patient_data.iloc[i]['case_id']
                    if case_id not in self.patient_modality_map:
                        self.patient_modality_map[case_id] = []
                    self.patient_modality_map[case_id].append((modality_name, i))
        
        # Print statistics
        modality_counts = {}
        for patient_id, mods in self.patient_modality_map.items():
            n_mods = len(mods)
            if n_mods not in modality_counts:
                modality_counts[n_mods] = 0
            modality_counts[n_mods] += 1
        
        print("\nPatients by number of modalities:")
        for n_mods in sorted(modality_counts.keys()):
            print(f"  {n_mods} modalities: {modality_counts[n_mods]} patients")
    
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
        
        # Determine if this is a cancer type/site visualization or TCGA project
        is_cancer_type = 'cancer type' in title.lower() or 'organ site' in title.lower()
        is_tcga = any(label.startswith('TCGA-') for label in unique_labels if label != 'Unknown')
        
        n_colors = len(unique_labels)
        
        if (is_cancer_type or is_tcga) and n_colors > 10:
            # For TCGA projects and cancer types/sites, use a custom color scheme
            # Group similar cancer types by color family
            tcga_color_groups = {
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
            
            # Markers for visual distinction
            markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', 'd', 
                      '<', '>', '8', 'P', 'H', '+', 'x', '|', '_']
            
            color_dict = {}
            marker_dict = {}
            
            # Fallback colors for non-TCGA labels
            fallback_colors = plt.cm.Set3(np.linspace(0, 1, 12))
            fallback_idx = 0
            
            for i, label in enumerate(unique_labels):
                if label == 'Unknown':
                    color_dict[label] = 'lightgray'
                    marker_dict[label] = 'o'
                elif label in tcga_color_groups:
                    color_dict[label] = tcga_color_groups[label]
                    marker_dict[label] = markers[i % len(markers)]
                else:
                    # Use fallback color for non-TCGA labels
                    color_dict[label] = fallback_colors[fallback_idx % len(fallback_colors)]
                    marker_dict[label] = markers[i % len(markers)]
                    fallback_idx += 1
        else:
            # For other categories or small number of cancer types, use different colors
            if n_colors <= 10:
                colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
            elif n_colors <= 20:
                colors = plt.cm.tab20(np.linspace(0, 1, n_colors))
            else:
                colors = plt.cm.hsv(np.linspace(0, 0.9, n_colors))
            
            color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}
            marker_dict = {label: 'o' for label in unique_labels}  # All use circles
        
        # Create main plot without legend
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each category separately to use different markers
        for label in unique_labels:
            mask = labels == label
            if np.any(mask):
                color = color_dict[label] if label != 'Unknown' else 'lightgray'
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
        plt.close()
        
        # Always create separate legend file
        # Adjust figure size based on number of categories
        if n_colors <= 5:
            fig_height = 2
        elif n_colors <= 10:
            fig_height = 3
        elif n_colors <= 15:
            fig_height = 4
        else:
            fig_height = max(6, n_colors * 0.3)
        
        # Wider figure for TCGA descriptions
        fig_width = 12 if (is_tcga and n_colors > 10) else 8
            
        fig_legend, ax_legend = plt.subplots(figsize=(fig_width, fig_height))
        ax_legend.axis('off')
        
        # Create legend elements
        from matplotlib.lines import Line2D
        legend_elements = []
        
        if (is_cancer_type or is_tcga) and n_colors > 10:
            # For cancer types with markers, create Line2D elements
            # Add cancer type descriptions for TCGA projects
            tcga_descriptions = {
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
                'TCGA-LAML': 'Acute myeloid leukemia',
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
            
            for label in unique_labels:
                if label == 'Unknown':
                    color = 'lightgray'
                    marker = 'o'
                    display_label = label
                else:
                    color = color_dict[label]
                    marker = marker_dict[label]
                    # Add description if it's a TCGA project
                    if label in tcga_descriptions:
                        display_label = f"{label}: {tcga_descriptions[label]}"
                    else:
                        display_label = label
                legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor=color, markeredgecolor='black',
                                            markersize=8, label=display_label, linestyle='None'))
        else:
            # For other categories, use patches
            for label in unique_labels:
                if label == 'Unknown':
                    color = 'lightgray'
                else:
                    color = color_dict[label]
                legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))
        
        # Calculate number of columns based on number of labels
        if n_colors <= 5:
            ncol = 1
        elif n_colors <= 10:
            ncol = 1
        elif n_colors <= 20:
            ncol = 2
        else:
            ncol = 3
        
        # Add legend
        legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=ncol, 
                                 fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.savefig(f"{filename_base}_legend.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_individual_modalities(self):
        """Create t-SNE visualizations for each individual modality"""
        print("\n" + "="*60)
        print("PROCESSING INDIVIDUAL MODALITIES")
        print("="*60)
        
        for modality_name, modality_data in self.embeddings_data.items():
            print(f"\nProcessing {modality_name} embeddings...")
            
            # Get embeddings and patient IDs
            embeddings = modality_data['X']
            patient_ids = modality_data['patient_ids']
            
            # Get patient data for this modality
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
                    print(f"  Matched {len(modality_patient_data)} patients")
                else:
                    print(f"  Warning: Could not match patient IDs for {modality_name}")
                    continue
            else:
                # For other modalities
                n_samples = len(embeddings)
                modality_patient_data = self.patient_data.iloc[:n_samples].copy()
            
            # Handle NaN values
            if np.isnan(embeddings).any():
                print(f"  Warning: Found NaN values, replacing with zeros")
                embeddings = np.nan_to_num(embeddings, nan=0.0)
            
            # Skip if too few samples
            if len(embeddings) < 10:
                print(f"  Skipping {modality_name} - too few samples ({len(embeddings)})")
                continue
            
            # Compute t-SNE
            print(f"  Computing t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//2), n_iter=1000)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # Create modality subdirectory
            modality_dir = os.path.join(self.output_dir, modality_name)
            os.makedirs(modality_dir, exist_ok=True)
            
            # Create visualizations
            print(f"  Creating visualizations...")
            
            # By cancer type
            cancer_types = modality_patient_data['project_id'].values
            self.create_tsne_plot_with_separate_legend(
                embeddings_2d, cancer_types, 
                f't-SNE of {modality_name.capitalize()} Embeddings by Cancer Type',
                os.path.join(modality_dir, f'{modality_name}_tsne_by_cancer_type')
            )
            
            # By sex
            genders = modality_patient_data['gender'].values
            self.create_tsne_plot_with_separate_legend(
                embeddings_2d, genders,
                f't-SNE of {modality_name.capitalize()} Embeddings by Sex',
                os.path.join(modality_dir, f'{modality_name}_tsne_by_sex')
            )
            
            # By age group
            age_groups = modality_patient_data['age_group'].values
            self.create_tsne_plot_with_separate_legend(
                embeddings_2d, age_groups,
                f't-SNE of {modality_name.capitalize()} Embeddings by Age Group',
                os.path.join(modality_dir, f'{modality_name}_tsne_by_age_group')
            )
            
            # By organ site
            if 'tissue_or_organ_of_origin' in modality_patient_data.columns:
                organ_sites = modality_patient_data['tissue_or_organ_of_origin'].values
                self.create_tsne_plot_with_separate_legend(
                    embeddings_2d, organ_sites,
                    f't-SNE of {modality_name.capitalize()} Embeddings by Organ Site',
                    os.path.join(modality_dir, f'{modality_name}_tsne_by_organ_site')
                )
    
    def create_multimodal_embeddings_batch(self, patient_list, max_patients=3000):
        """Create multimodal embeddings efficiently using batch processing"""
        # Sample if too many patients
        if len(patient_list) > max_patients:
            print(f"\nSampling {max_patients} patients for visualization...")
            np.random.seed(42)
            patient_list = np.random.choice(patient_list, max_patients, replace=False).tolist()
        
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
    
    def visualize_multimodal_embeddings(self):
        """Create t-SNE visualizations for multimodal embeddings"""
        print("\n" + "="*60)
        print("CREATING MULTIMODAL EMBEDDINGS")
        print("="*60)
        
        # Get patients with 2+ modalities
        multimodal_patients = [pid for pid, mods in self.patient_modality_map.items() if len(mods) >= 2]
        print(f"\nTotal patients with 2+ modalities: {len(multimodal_patients)}")
        
        # Create multimodal embeddings
        multimodal_results = self.create_multimodal_embeddings_batch(multimodal_patients)
        patient_list = multimodal_results['patient_list']
        
        # Process each fusion method
        for fusion_method in ['concat', 'mean_pool', 'kronecker']:
            fusion_embeddings, valid_indices = multimodal_results[fusion_method]
            
            if len(fusion_embeddings) == 0:
                print(f"\nSkipping {fusion_method} - no valid embeddings")
                continue
            
            print(f"\nProcessing multimodal {fusion_method} embeddings...")
            print(f"  Shape: {fusion_embeddings.shape}")
            
            # Get patient data for valid embeddings
            valid_patient_ids = [patient_list[i] for i in valid_indices]
            fusion_patient_data = self.patient_data[self.patient_data['case_id'].isin(valid_patient_ids)].iloc[:len(fusion_embeddings)]
            
            # Compute t-SNE
            print(f"  Computing t-SNE...")
            perplexity = min(30, len(fusion_embeddings)//2)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
            embeddings_2d = tsne.fit_transform(fusion_embeddings)
            
            # Create fusion method subdirectory
            fusion_dir = os.path.join(self.output_dir, f'multimodal_{fusion_method}')
            os.makedirs(fusion_dir, exist_ok=True)
            
            # Create visualizations
            print(f"  Creating visualizations...")
            
            # By cancer type
            cancer_types = fusion_patient_data['project_id'].values
            self.create_tsne_plot_with_separate_legend(
                embeddings_2d, cancer_types, 
                f't-SNE of Multimodal {fusion_method.upper()} by Cancer Type',
                os.path.join(fusion_dir, f'{fusion_method}_tsne_by_cancer_type')
            )
            
            # By modality combination
            modality_labels = []
            for pid in fusion_patient_data['case_id'].values:
                mods = self.patient_modality_map.get(pid, [])
                mod_names = sorted([m[0] for m in mods])
                modality_labels.append('+'.join(mod_names[:3]))
            
            self.create_tsne_plot_with_separate_legend(
                embeddings_2d, modality_labels,
                f't-SNE of Multimodal {fusion_method.upper()} by Modality Combination',
                os.path.join(fusion_dir, f'{fusion_method}_tsne_by_modality_combo')
            )
            
            # Save modality combination statistics
            combo_counts = pd.Series(modality_labels).value_counts()
            combo_stats = pd.DataFrame({
                'modality_combination': combo_counts.index,
                'count': combo_counts.values,
                'percentage': (combo_counts.values / len(modality_labels) * 100).round(2)
            })
            combo_stats.to_csv(os.path.join(fusion_dir, f'{fusion_method}_modality_combination_stats.csv'), index=False)
        
        print(f"\nSuccessful embeddings:")
        for method in ['concat', 'mean_pool', 'kronecker']:
            if method in multimodal_results:
                _, valid_indices = multimodal_results[method]
                print(f"  {method}: {len(valid_indices)} patients")


def main():
    parser = argparse.ArgumentParser(description='Create t-SNE visualizations for HoneyBee embeddings')
    parser.add_argument('--output-dir', type=str, 
                        default='/mnt/f/Projects/HoneyBee/results/stratified_tsne_visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--individual', action='store_true', help='Create individual modality visualizations')
    parser.add_argument('--multimodal', action='store_true', help='Create multimodal visualizations')
    parser.add_argument('--all', action='store_true', help='Create all visualizations (default)')
    
    args = parser.parse_args()
    
    # Default to all if no specific option is chosen
    if not args.individual and not args.multimodal:
        args.all = True
    
    # Create visualizer
    visualizer = TSNEVisualizer(args.output_dir)
    
    # Load data
    visualizer.load_data()
    
    # Create visualizations
    if args.all or args.individual:
        visualizer.visualize_individual_modalities()
    
    if args.all or args.multimodal:
        visualizer.visualize_multimodal_embeddings()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("="*60)
    print("\nOutput format: PDF with separate legend files for all visualizations")
    print("Legend files: *_legend.pdf")
    print("Plot files: *_plot.pdf")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()