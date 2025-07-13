#!/usr/bin/env python3
"""
Generate BLCA survival curves from trained models for Nature publication.
Creates figures without titles/legends in subplots, with a shared legend.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
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
            print(f"Loaded {modality} embeddings: {embeddings[modality].shape}")
        elif os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'X' in data:
                    embeddings[modality] = data['X']
                    print(f"Loaded {modality} embeddings: {embeddings[modality].shape}")
    
    return embeddings


def load_patient_data(data_path):
    """Load patient survival data from CSV."""
    csv_path = os.path.join(data_path, 'embeddings/patient_data_with_embeddings.csv')
    df = pd.read_csv(csv_path)
    
    # Filter for BLCA patients
    blca_df = df[df['project_id'] == 'TCGA-BLCA'].copy()
    
    # Extract survival information
    # Handle days_to_death and days_to_last_follow_up
    blca_df['survival_time'] = blca_df['days_to_death'].fillna(blca_df['days_to_last_follow_up'])
    blca_df['event'] = ~blca_df['days_to_death'].isna()
    
    return blca_df


def create_multimodal_embeddings(embeddings, patient_data):
    """Create multimodal embeddings matching original approach."""
    multimodal = {}
    n_patients = len(patient_data)
    
    # Get BLCA patient indices
    blca_mask = patient_data['project_id'] == 'TCGA-BLCA'
    blca_indices = np.where(blca_mask)[0]
    
    # Concatenation fusion
    concat_list = []
    valid_indices = []
    
    for idx in blca_indices:
        patient_embeddings = []
        for modality in ['clinical', 'pathology', 'molecular', 'wsi']:
            if modality in embeddings and idx < len(embeddings[modality]):
                emb = embeddings[modality][idx]
                if not np.all(np.isnan(emb)):
                    patient_embeddings.append(emb)
        
        if len(patient_embeddings) >= 2:  # At least 2 modalities
            concat_emb = np.concatenate(patient_embeddings)
            concat_list.append(concat_emb)
            valid_indices.append(idx)
    
    if concat_list:
        multimodal['concat'] = np.array(concat_list)
        multimodal['concat_indices'] = np.array(valid_indices)
    
    # Mean pooling fusion
    mean_list = []
    valid_indices = []
    
    for idx in blca_indices:
        patient_embeddings = []
        for modality in ['clinical', 'pathology', 'molecular', 'wsi']:
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
    
    for idx in blca_indices:
        patient_embeddings = []
        for modality in ['clinical', 'pathology', 'molecular', 'wsi']:
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
                    kron_features.extend(kron[:100])  # Limit size
            
            if kron_features:
                kronecker_list.append(np.array(kron_features[:100]))  # Fixed size
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


def prepare_data_for_prediction(embeddings, patient_data, modality):
    """Prepare data for model prediction."""
    # Get BLCA patients
    blca_mask = patient_data['project_id'] == 'TCGA-BLCA'
    blca_data = patient_data[blca_mask].copy()
    
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
        
        # Get BLCA patient indices
        blca_indices = blca_data.index.tolist()
        
        # Filter for valid embeddings
        X_list = []
        valid_indices = []
        
        for idx in blca_indices:
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
        print(f"    Dimension mismatch: {X.shape[1]} vs expected {expected_features}")
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


def create_composite_figure(results_dict, output_dir):
    """Create multiple composite figures organized by model type."""
    # Note: BLCA doesn't have radiology data
    modalities = ['clinical', 'pathology', 'wsi', 'molecular', 'kronecker', 'concat', 'mean_pool']
    models = ['cox', 'deepsurv', 'rsf']
    
    # Create one figure per model type
    for model in models:
        # Create figure with GridSpec (4x2 grid)
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3)
        
        # Create subplots for each modality
        for idx, modality in enumerate(modalities):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])
            
            # Get results
            key = f'{modality}_{model}'
            if key in results_dict:
                y, risk_groups = results_dict[key]
                plot_survival_curve(ax, y, risk_groups, show_legend=False)
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            # Add panel label
            panel_label = chr(65 + idx)  # A, B, C, D, E, F, G
            ax.text(-0.15, 1.05, panel_label, transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')
        
        # Use 8th subplot for legend and panel descriptions
        ax_legend = fig.add_subplot(gs[3, 1])
        
        # Set up the axes to match other subplots
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        
        # Remove spines to make it cleaner
        for spine in ax_legend.spines.values():
            spine.set_visible(False)
        
        # Create legend elements with shorter labels
        legend_elements = [
            Line2D([0], [0], color=COLORS['low_risk'], lw=3, label='Low risk'),
            Line2D([0], [0], color=COLORS['high_risk'], lw=3, label='High risk'),
            Patch(facecolor=COLORS['low_risk'], alpha=COLORS['ci_alpha'], label='95% CI'),
            Line2D([0], [0], marker='x', color=COLORS['censored'], linestyle='None', 
                   markersize=10, label='Censored', markeredgewidth=2),
        ]
        
        # Add legend in the upper left portion with more vertical spacing
        legend = ax_legend.legend(handles=legend_elements, loc='upper left', 
                                 frameon=False, fontsize=9, ncol=1, 
                                 bbox_to_anchor=(0.05, 0.95),
                                 labelspacing=1.5)  # Increased spacing between legend items
        
        # Add panel descriptions in the right portion, left-aligned without box
        panel_text = (
            "Panel Descriptions:\n\n"
            "A: Clinical data\n"
            "B: Pathology report\n"
            "C: Whole slide image\n"
            "D: Molecular data\n"
            "E: Kronecker fusion\n"
            "F: Concatenation fusion\n"
            "G: Mean pooling fusion"
        )
        
        ax_legend.text(0.55, 0.5, panel_text, transform=ax_legend.transAxes,
                      fontsize=9, ha='left', va='center')
        
        # Save figure
        plt.tight_layout()
        
        for fmt in ['pdf', 'png', 'svg', 'eps']:
            filepath = os.path.join(output_dir, f'BLCA_{model}_survival_curves.{fmt}')
            plt.savefig(filepath, format=fmt, bbox_inches='tight', 
                        transparent=True, pad_inches=0.1)
        
        plt.close()
        print(f"Created composite figure for {model} model")
    
    # Also create a single large figure with all 24 panels
    create_all_panels_figure(results_dict, output_dir)


def create_all_panels_figure(results_dict, output_dir):
    """Create a single large figure with all 21 survival curves (7 modalities x 3 models)."""
    # Note: BLCA doesn't have radiology data
    modalities = ['clinical', 'pathology', 'wsi', 'molecular', 'kronecker', 'concat', 'mean_pool']
    models = ['cox', 'deepsurv', 'rsf']
    
    # Create large figure (7 rows x 3 columns) with more spacing
    fig = plt.figure(figsize=(15, 20))
    gs = gridspec.GridSpec(8, 3, height_ratios=[1]*7 + [0.2], hspace=0.6, wspace=0.3)
    
    # Create subplots
    panel_idx = 0
    for i, modality in enumerate(modalities):
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
            
            panel_idx += 1
    
    # Add shared legend at bottom
    ax_legend = fig.add_subplot(gs[7, :])
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
        filepath = os.path.join(output_dir, f'BLCA_all_survival_curves.{fmt}')
        plt.savefig(filepath, format=fmt, bbox_inches='tight', 
                    transparent=True, pad_inches=0.1, dpi=300)
    
    plt.close()
    print("Created figure with all 24 panels")


def create_individual_panels(results_dict, output_dir):
    """Create individual panels for BioRender."""
    biorender_dir = os.path.join(output_dir, 'biorender_panels')
    os.makedirs(biorender_dir, exist_ok=True)
    
    for key, (y, risk_groups) in results_dict.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        plot_survival_curve(ax, y, risk_groups, show_legend=False)
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in ['png', 'svg']:
            filepath = os.path.join(biorender_dir, f'BLCA_{key}.{fmt}')
            plt.savefig(filepath, format=fmt, dpi=600, 
                       bbox_inches='tight', transparent=True)
        
        plt.close()
        print(f"  Created panel: BLCA_{key}")


def main():
    """Main function to generate BLCA survival curves."""
    print("=" * 80)
    print("GENERATING BLCA SURVIVAL CURVES FOR NATURE PUBLICATION")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    # Setup
    setup_nature_style()
    
    # Paths
    data_path = '/mnt/f/Projects/HoneyBee/results/shared_data'
    model_dir = '/mnt/f/Projects/HoneyBee/results/survival/models'
    output_dir = '/mnt/f/Projects/HoneyBee/results/survival/nature_figures/TCGA-BLCA'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    embeddings = load_embeddings(data_path)
    
    # Load all patient data (not just BLCA)
    patient_data_path = os.path.join(data_path, 'embeddings/patient_data_with_embeddings.csv')
    all_patient_data = pd.read_csv(patient_data_path, low_memory=False)
    
    # Extract survival information
    all_patient_data['survival_time'] = all_patient_data['days_to_death'].fillna(all_patient_data['days_to_last_follow_up'])
    all_patient_data['event'] = ~all_patient_data['days_to_death'].isna()
    
    print(f"Total patients: {len(all_patient_data)}")
    print(f"BLCA patients: {len(all_patient_data[all_patient_data['project_id'] == 'TCGA-BLCA'])}")
    
    # Create multimodal embeddings
    print("\nCreating multimodal embeddings...")
    multimodal_embeddings = create_multimodal_embeddings(embeddings, all_patient_data)
    embeddings.update(multimodal_embeddings)
    
    # Define all modalities and models
    # Note: BLCA doesn't have radiology data
    modalities = ['clinical', 'pathology', 'wsi', 'molecular', 'kronecker', 'concat', 'mean_pool']
    models = ['cox', 'deepsurv', 'rsf']
    
    # Generate all combinations
    combinations = [(mod, model) for mod in modalities for model in models]
    print(f"\nWill process {len(combinations)} combinations")
    
    # Store results
    results_dict = {}
    
    # Process each combination
    for modality, model_type in combinations:
        print(f"\nProcessing {modality} - {model_type}...")
        
        # Load model
        model_path = os.path.join(model_dir, f'TCGA-BLCA_{modality}_{model_type}_best.pkl')
        if not os.path.exists(model_path):
            print(f"  Model not found: {model_path}")
            continue
        
        try:
            # Load model
            model_data = load_model(model_path)
            
            # Prepare data
            X, y, filtered_data = prepare_data_for_prediction(embeddings, all_patient_data, modality)
            print(f"  Data shape: {X.shape}, Patients: {len(y)}")
            
            # Calculate risk scores
            risk_scores = calculate_risk_scores(model_data, X, model_type)
            
            # Create risk groups (median split for 2 groups)
            median_risk = np.median(risk_scores)
            risk_groups = risk_scores >= median_risk
            
            # Store results
            results_dict[f'{modality}_{model_type}'] = (y, risk_groups)
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create figures
    print("\nCreating figures...")
    create_composite_figure(results_dict, output_dir)
    create_individual_panels(results_dict, output_dir)
    
    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now()}")
    print(f"\nFigures saved in: {output_dir}")
    print("\nCreated files:")
    print("  - BLCA_cox_survival_curves.*  (Cox model - 7 modalities)")
    print("  - BLCA_deepsurv_survival_curves.*  (DeepSurv model - 7 modalities)")
    print("  - BLCA_rsf_survival_curves.*  (RSF model - 7 modalities)")
    print("  - BLCA_all_survival_curves.*  (All 21 curves: 7 modalities × 3 models)")
    print("  - biorender_panels/  (Individual panels for each model-modality combination)")
    print("\nFormats: PDF, PNG (600 DPI), SVG, EPS")
    print("\nNote: BLCA dataset does not include radiology data")


if __name__ == "__main__":
    main()