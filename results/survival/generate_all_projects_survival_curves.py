#!/usr/bin/env python3
"""
Generate all_survival_curves figures for all TCGA projects using Nature publication standards.
Creates clean figures similar to BLCA with proper formatting.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from datetime import datetime
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# DeepSurv model class definition (needed for unpickling)
class DeepSurvModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = None):
        super(DeepSurvModel, self).__init__()
        
        if hidden_dims is None:
            # Adaptive architecture based on input dimension
            if input_dim > 100:
                hidden_dims = [256, 128, 64]
            else:
                hidden_dims = [64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Use LayerNorm instead of BatchNorm to avoid batch size issues
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.model(x)

# Nature publication standards
NATURE_CONFIG = {
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'figure.figsize': (7.2, 8),  # Double column width
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
    'lines.markersize': 4,
    'axes.spines.top': False,
    'axes.spines.right': False,
}

# Consistent color scheme
COLORS = {
    'low_risk': '#2E86AB',    # Blue
    'high_risk': '#E63946',   # Red
    'ci_alpha': 0.2,          # Confidence interval transparency
    'censored': 'black',      # Censored markers
}

# List of all TCGA projects
TCGA_PROJECTS = [
    'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD',
    'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC',
    'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC',
    'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ',
    'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM',
    'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM'
]

# We'll determine which modalities are available automatically based on existing models

def setup_nature_style():
    """Configure matplotlib for Nature standards."""
    for key, value in NATURE_CONFIG.items():
        rcParams[key] = value
    sns.set_style("ticks")
    sns.despine()


def load_embeddings(data_path):
    """Load pre-computed embeddings."""
    embeddings = {}
    
    # Load embeddings as numpy arrays (matching original format)
    modalities = ['clinical', 'pathology', 'radiology', 'molecular', 'wsi']
    for modality in modalities:
        # Try numpy format first
        npy_path = os.path.join(data_path, 'embeddings', f'{modality}_embeddings.npy')
        pkl_path = os.path.join(data_path, 'embeddings', f'{modality}_embeddings.pkl')
        
        if os.path.exists(npy_path):
            embeddings[modality] = np.load(npy_path)
            print(f"    Loaded {modality} embeddings: {embeddings[modality].shape}")
        elif os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'X' in data:
                    embeddings[modality] = data['X']
                    print(f"    Loaded {modality} embeddings: {embeddings[modality].shape}")
    
    return embeddings


def create_multimodal_embeddings(embeddings, patient_data, project_id, available_modalities):
    """Create multimodal embeddings matching original approach."""
    multimodal = {}
    
    # Get project patient indices
    project_mask = patient_data['project_id'] == project_id
    project_indices = np.where(project_mask)[0]
    
    # Use only the available modalities for this project
    modalities_to_use = [mod for mod in ['clinical', 'pathology', 'molecular', 'wsi', 'radiology'] 
                         if mod in available_modalities]
    
    # Concatenation fusion
    concat_list = []
    valid_indices = []
    max_concat_size = 0
    
    # First pass: collect embeddings and find max size
    temp_concat_list = []
    for idx in project_indices:
        patient_embeddings = []
        for modality in modalities_to_use:
            if modality in embeddings and idx < len(embeddings[modality]):
                emb = embeddings[modality][idx]
                if not np.all(np.isnan(emb)):
                    patient_embeddings.append(emb)
        
        if len(patient_embeddings) >= 2:  # At least 2 modalities
            concat_emb = np.concatenate(patient_embeddings)
            temp_concat_list.append((idx, concat_emb))
            max_concat_size = max(max_concat_size, len(concat_emb))
    
    # Second pass: pad all embeddings to the same size
    for idx, concat_emb in temp_concat_list:
        if len(concat_emb) < max_concat_size:
            # Pad with zeros to match max size
            padded_emb = np.pad(concat_emb, (0, max_concat_size - len(concat_emb)), mode='constant')
        else:
            padded_emb = concat_emb
        concat_list.append(padded_emb)
        valid_indices.append(idx)
    
    if concat_list:
        multimodal['concat'] = np.array(concat_list)
        multimodal['concat_indices'] = np.array(valid_indices)
    
    # Mean pooling fusion
    mean_list = []
    valid_indices = []
    
    for idx in project_indices:
        patient_embeddings = []
        for modality in modalities_to_use:
            if modality in embeddings and idx < len(embeddings[modality]):
                emb = embeddings[modality][idx]
                if not np.all(np.isnan(emb)):
                    # Resize to common dimension
                    if len(emb) != 1024:
                        emb = np.pad(emb, (0, 1024 - len(emb)), mode='constant')[:1024]
                    patient_embeddings.append(emb)
        
        if patient_embeddings:
            mean_emb = np.mean(patient_embeddings, axis=0)
            mean_list.append(mean_emb)
            valid_indices.append(idx)
    
    if mean_list:
        multimodal['mean_pool'] = np.array(mean_list)
        multimodal['mean_pool_indices'] = np.array(valid_indices)
    
    # Kronecker product fusion
    kronecker_list = []
    valid_indices = []
    
    # Fixed size for Kronecker features
    KRONECKER_SIZE = 100
    
    for idx in project_indices:
        patient_embeddings = []
        for modality in modalities_to_use:
            if modality in embeddings and idx < len(embeddings[modality]):
                emb = embeddings[modality][idx]
                if not np.all(np.isnan(emb)):
                    # Take first few dimensions for Kronecker
                    patient_embeddings.append(emb[:10])  # Use first 10 dims
        
        if len(patient_embeddings) >= 2:
            # Compute pairwise Kronecker products and flatten
            kron_features = []
            for i in range(len(patient_embeddings)):
                for j in range(i+1, len(patient_embeddings)):
                    kron = np.kron(patient_embeddings[i], patient_embeddings[j])
                    kron_features.extend(kron.flatten())
            
            # Ensure fixed size
            kron_array = np.array(kron_features)
            if len(kron_array) >= KRONECKER_SIZE:
                final_kron = kron_array[:KRONECKER_SIZE]
            else:
                # Pad if needed
                final_kron = np.pad(kron_array, (0, KRONECKER_SIZE - len(kron_array)), mode='constant')
            
            kronecker_list.append(final_kron)
            valid_indices.append(idx)
    
    if kronecker_list:
        multimodal['kronecker'] = np.array(kronecker_list)
        multimodal['kronecker_indices'] = np.array(valid_indices)
    
    return multimodal


def load_model(model_path):
    """Load a trained model."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def prepare_data_for_prediction(embeddings, patient_data, modality, project_id):
    """Prepare data for model prediction."""
    # Get project patients
    project_mask = patient_data['project_id'] == project_id
    project_data = patient_data[project_mask].copy()
    
    if modality in ['concat', 'mean_pool', 'kronecker']:
        # For multimodal embeddings
        if modality in embeddings and f'{modality}_indices' in embeddings:
            # Use the pre-selected indices
            indices = embeddings[f'{modality}_indices']
            X = embeddings[modality]
            
            # Get corresponding patient data
            filtered_data = patient_data.iloc[indices].copy()
        else:
            return np.array([]).reshape(0, 0), np.array([]), pd.DataFrame()
    else:
        # For single modality embeddings
        if modality not in embeddings:
            return np.array([]).reshape(0, 0), np.array([]), pd.DataFrame()
        
        # Get project patient indices
        project_indices = project_data.index.tolist()
        
        # Filter for valid embeddings
        X_list = []
        valid_indices = []
        
        for idx in project_indices:
            if idx < len(embeddings[modality]):
                emb = embeddings[modality][idx]
                if not np.all(np.isnan(emb)):
                    X_list.append(emb)
                    valid_indices.append(idx)
        
        if not X_list:
            return np.array([]).reshape(0, 0), np.array([]), pd.DataFrame()
        
        X = np.array(X_list)
        filtered_data = patient_data.iloc[valid_indices].copy()
    
    # Create survival outcome array
    y = np.array([(event, time) for event, time in 
                  zip(filtered_data['event'], filtered_data['survival_time'])],
                 dtype=[('event', bool), ('time', float)])
    
    # Remove patients with missing survival data
    valid_mask = ~np.isnan(y['time'])
    X = X[valid_mask]
    y = y[valid_mask]
    filtered_data = filtered_data[valid_mask]
    
    return X, y, filtered_data


def calculate_risk_scores(model_data, X, model_type):
    """Calculate risk scores using the trained model."""
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Handle dimension mismatch
    expected_features = scaler.n_features_in_
    if X.shape[1] != expected_features:
        print(f"        Dimension mismatch: {X.shape[1]} vs expected {expected_features}")
        # Pad or truncate to match expected dimensions
        if X.shape[1] < expected_features:
            # Pad with zeros
            padding = expected_features - X.shape[1]
            X = np.pad(X, ((0, 0), (0, padding)), mode='constant')
        else:
            # Truncate
            X = X[:, :expected_features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Handle NaN values
    if np.any(np.isnan(X_scaled)):
        imputer = SimpleImputer(strategy='mean')
        X_scaled = imputer.fit_transform(X_scaled)
    
    # Calculate risk scores based on model type
    if model_type == 'cox':
        # Check if PCA was used
        if hasattr(model, '_pca') and model._pca is not None:
            X_transformed = model._pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        # Create DataFrame for lifelines
        df = pd.DataFrame(X_transformed, columns=[f'feature_{i}' for i in range(X_transformed.shape[1])])
        risk_scores = model.predict_log_partial_hazard(df).values
        
    elif model_type == 'deepsurv':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.ascontiguousarray(X_scaled)).to(device)
            risk_scores = model(X_tensor).squeeze().cpu().numpy()
        
        # Handle NaN values
        if np.any(np.isnan(risk_scores)):
            risk_scores = np.nan_to_num(risk_scores, nan=0.0)
            
    elif model_type == 'rsf':
        # For Random Survival Forest
        # scikit-survival RSF predict() returns risk scores directly (1D array)
        risk_scores = model.predict(X_scaled)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return risk_scores


def plot_survival_curve(ax, y, risk_groups, show_legend=False):
    """Plot Kaplan-Meier survival curves without title."""
    kmf = KaplanMeierFitter()
    
    # Extract time and event arrays
    times = y['time']
    events = y['event']
    
    # Plot low risk group
    mask_low = ~risk_groups
    if np.sum(mask_low) > 0:
        kmf.fit(times[mask_low], events[mask_low], label=f'Low risk (n={np.sum(mask_low)})')
        kmf.plot_survival_function(ax=ax, color=COLORS['low_risk'], 
                                  linewidth=2, ci_show=True, 
                                  ci_alpha=COLORS['ci_alpha'],
                                  show_censors=True, censor_styles={'marker': 'x', 'ms': 6})
    
    # Plot high risk group
    mask_high = risk_groups
    if np.sum(mask_high) > 0:
        kmf.fit(times[mask_high], events[mask_high], label=f'High risk (n={np.sum(mask_high)})')
        kmf.plot_survival_function(ax=ax, color=COLORS['high_risk'], 
                                  linewidth=2, ci_show=True, 
                                  ci_alpha=COLORS['ci_alpha'],
                                  show_censors=True, censor_styles={'marker': 'x', 'ms': 6})
    
    # Perform log-rank test
    if np.sum(mask_low) > 0 and np.sum(mask_high) > 0:
        results = logrank_test(times[mask_low], times[mask_high], 
                              events[mask_low], events[mask_high])
        
        # Add p-value with scientific notation categories
        p_value = results.p_value
        if p_value < 0.001:
            p_text = 'P < 0.001'
        elif p_value < 0.01:
            p_text = 'P < 0.01'
        elif p_value < 0.05:
            p_text = 'P < 0.05'
        else:
            p_text = 'P ≥ 0.05'
            
        ax.text(0.98, 0.95, p_text, transform=ax.transAxes, 
                ha='right', va='top', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Styling
    ax.set_xlabel('Time (days)', fontsize=9)
    ax.set_ylabel('Survival probability', fontsize=9)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Remove legend unless specifically requested
    if not show_legend:
        ax.get_legend().remove()


def create_all_panels_figure(results_dict, output_dir, project_id):
    """Create a single large figure with all survival curves."""
    models = ['cox', 'deepsurv', 'rsf']
    
    # Determine which modalities actually have results
    available_modalities = []
    all_modalities = ['clinical', 'pathology', 'wsi', 'radiology', 'molecular', 'kronecker', 'concat', 'mean_pool']
    
    for modality in all_modalities:
        # Check if this modality has at least one model with results
        has_results = any(f'{modality}_{model}' in results_dict for model in models)
        if has_results:
            available_modalities.append(modality)
    
    if not available_modalities:
        print(f"    WARNING: No modalities with results for {project_id}")
        return
    
    print(f"    Available modalities: {', '.join(available_modalities)}")
    
    # Create large figure with more spacing
    n_rows = len(available_modalities)
    fig = plt.figure(figsize=(15, 2.5 * n_rows + 1))
    gs = gridspec.GridSpec(n_rows + 1, 3, height_ratios=[1]*n_rows + [0.2], 
                          hspace=0.6, wspace=0.3)
    
    # Create subplots
    for i, modality in enumerate(available_modalities):
        for j, model in enumerate(models):
            ax = fig.add_subplot(gs[i, j])
            
            # Get results
            key = f'{modality}_{model}'
            if key in results_dict:
                y, risk_groups = results_dict[key]
                plot_survival_curve(ax, y, risk_groups, show_legend=False)
            else:
                ax.text(0.5, 0.5, 'No data', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=8, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            # Add subtitle with modality and model - positioned higher to avoid overlap
            ax.text(0.5, 1.08, f'{modality.upper()} - {model.upper()}', 
                   transform=ax.transAxes, ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    # Add shared legend at bottom
    ax_legend = fig.add_subplot(gs[n_rows, :])
    ax_legend.axis('off')
    
    # Create legend elements
    legend_elements = [
        Line2D([0], [0], color=COLORS['low_risk'], lw=2, label='Low risk group'),
        Line2D([0], [0], color=COLORS['high_risk'], lw=2, label='High risk group'),
        Patch(facecolor=COLORS['low_risk'], alpha=COLORS['ci_alpha'], label='95% Confidence interval'),
        Line2D([0], [0], marker='x', color=COLORS['censored'], linestyle='None', 
               markersize=8, label='Censored', markeredgewidth=1.5),
    ]
    
    ax_legend.legend(handles=legend_elements, loc='center', frameon=False, 
                     fontsize=10, ncol=4, columnspacing=2)
    
    # Save figure
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        filepath = os.path.join(output_dir, f'{project_id}_all_survival_curves.{fmt}')
        plt.savefig(filepath, format=fmt, bbox_inches='tight', 
                    transparent=True, pad_inches=0.1, dpi=300)
    
    plt.close()
    print(f"    Created all_survival_curves figure for {project_id} with {n_rows} modalities")


def process_project(project_id, data_path, model_dir, output_base_dir):
    """Process a single TCGA project."""
    print(f"\n{'='*60}")
    print(f"Processing {project_id}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, project_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # First, detect which modalities have models for this project
    print("  Detecting available modalities...")
    base_modalities = ['clinical', 'pathology', 'wsi', 'radiology', 'molecular']
    models = ['cox', 'deepsurv', 'rsf']
    
    available_modalities = []
    for modality in base_modalities:
        # Check if at least one model exists for this modality
        has_model = any(os.path.exists(os.path.join(model_dir, f'{project_id}_{modality}_{model}_best.pkl'))
                       for model in models)
        if has_model:
            available_modalities.append(modality)
    
    if not available_modalities:
        print(f"  WARNING: No models found for {project_id}")
        return
    
    print(f"  Available base modalities: {', '.join(available_modalities)}")
    
    # Load embeddings
    print("  Loading data...")
    embeddings = load_embeddings(data_path)
    
    # Load all patient data
    patient_data_path = os.path.join(data_path, 'embeddings/patient_data_with_embeddings.csv')
    all_patient_data = pd.read_csv(patient_data_path, low_memory=False)
    
    # Extract survival information
    all_patient_data['survival_time'] = all_patient_data['days_to_death'].fillna(all_patient_data['days_to_last_follow_up'])
    all_patient_data['event'] = ~all_patient_data['days_to_death'].isna()
    
    # Check if project has patients
    project_patients = all_patient_data[all_patient_data['project_id'] == project_id]
    if len(project_patients) == 0:
        print(f"  WARNING: No patients found for {project_id}")
        return
    
    print(f"  Total patients: {len(all_patient_data)}")
    print(f"  {project_id} patients: {len(project_patients)}")
    
    # Create multimodal embeddings using only available modalities
    print("  Creating multimodal embeddings...")
    multimodal_embeddings = create_multimodal_embeddings(embeddings, all_patient_data, project_id, available_modalities)
    embeddings.update(multimodal_embeddings)
    
    # Add multimodal modalities if they were created
    all_modalities = available_modalities.copy()
    if 'kronecker' in multimodal_embeddings:
        all_modalities.append('kronecker')
    if 'concat' in multimodal_embeddings:
        all_modalities.append('concat')
    if 'mean_pool' in multimodal_embeddings:
        all_modalities.append('mean_pool')
    
    # Generate all combinations
    combinations = [(mod, model) for mod in all_modalities for model in models]
    
    # Store results
    results_dict = {}
    
    # Process each combination
    for modality, model_type in combinations:
        # Load model
        model_path = os.path.join(model_dir, f'{project_id}_{modality}_{model_type}_best.pkl')
        if not os.path.exists(model_path):
            continue
        
        try:
            # Load model
            model_data = load_model(model_path)
            
            # Prepare data
            X, y, filtered_data = prepare_data_for_prediction(embeddings, all_patient_data, modality, project_id)
            
            if len(X) == 0:
                continue
            
            # Calculate risk scores
            risk_scores = calculate_risk_scores(model_data, X, model_type)
            
            # Create risk groups (median split for 2 groups)
            median_risk = np.median(risk_scores)
            risk_groups = risk_scores >= median_risk
            
            # Store results
            results_dict[f'{modality}_{model_type}'] = (y, risk_groups)
            
        except Exception as e:
            continue
    
    # Create all_panels figure if we have results
    if results_dict:
        print("\n  Creating all_survival_curves figure...")
        create_all_panels_figure(results_dict, output_dir, project_id)
        print(f"  Figure saved in: {output_dir}")
    else:
        print(f"  WARNING: No successful model predictions for {project_id}")


def main():
    """Main function to generate survival curves for all projects."""
    print("="*80)
    print("GENERATING ALL_SURVIVAL_CURVES FOR ALL TCGA PROJECTS")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    
    # Setup
    setup_nature_style()
    
    # Paths
    data_path = '/mnt/f/Projects/HoneyBee/results/shared_data'
    model_dir = '/mnt/f/Projects/HoneyBee/results/survival/models'
    output_base_dir = '/mnt/f/Projects/HoneyBee/results/survival/nature_figures'
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each project
    successful_projects = []
    failed_projects = []
    
    for project_id in TCGA_PROJECTS:
        try:
            process_project(project_id, data_path, model_dir, output_base_dir)
            successful_projects.append(project_id)
        except Exception as e:
            print(f"\nERROR processing {project_id}: {str(e)}")
            failed_projects.append(project_id)
            continue
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total projects processed: {len(TCGA_PROJECTS)}")
    print(f"Successful: {len(successful_projects)}")
    print(f"Failed: {len(failed_projects)}")
    
    if failed_projects:
        print(f"\nFailed projects: {', '.join(failed_projects)}")
    
    print(f"\nEnd time: {datetime.now()}")
    print(f"\nAll figures saved in: {output_base_dir}")


if __name__ == "__main__":
    main()