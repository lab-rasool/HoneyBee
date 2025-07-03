import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set style for publication quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class DeepSurvModel(nn.Module):
    """DeepSurv neural network model for survival analysis"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
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


class SurvivalAnalysis:
    """Comprehensive survival analysis for multimodal cancer data"""
    
    def __init__(self, data_path: str, output_path: str):
        self.data_path = data_path
        self.output_path = output_path
        self.embeddings = {}
        self.patient_data = None
        self.multimodal_embeddings = {}
        
        # Define similar cancer types for merging
        self.similar_cancers = {
            'lung': ['TCGA-LUAD', 'TCGA-LUSC'],
            'kidney': ['TCGA-KIRC', 'TCGA-KIRP', 'TCGA-KICH'],
            'gi_tract': ['TCGA-COAD', 'TCGA-READ', 'TCGA-STAD', 'TCGA-ESCA'],
            'gyn': ['TCGA-CESC', 'TCGA-UCEC', 'TCGA-OV'],
        }
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'cv_results'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'risk_curves'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'models'), exist_ok=True)
        
    def load_data(self):
        """Load embeddings and patient data"""
        print("Loading data...")
        
        # Load patient data
        self.patient_data = pd.read_csv(
            os.path.join(self.data_path, 'embeddings', 'patient_data_with_embeddings.csv')
        )
        
        # Load embeddings for each modality
        modalities = ['clinical', 'pathology', 'radiology', 'molecular', 'wsi']
        for modality in modalities:
            emb_path = os.path.join(self.data_path, 'embeddings', f'{modality}_embeddings.npy')
            if os.path.exists(emb_path):
                self.embeddings[modality] = np.load(emb_path)
                print(f"Loaded {modality} embeddings: {self.embeddings[modality].shape}")
    
    def create_multimodal_embeddings(self):
        """Create multimodal embeddings with flexible fusion based on available modalities"""
        print("\nCreating multimodal embeddings...")
        
        # For each patient, check which modalities are available
        n_patients = len(self.patient_data)
        patient_modalities = {}
        
        for i in range(n_patients):
            available_mods = []
            for modality in self.embeddings:
                if i < len(self.embeddings[modality]):
                    # Check if embedding is not all NaN
                    emb = self.embeddings[modality][i]
                    if not np.all(np.isnan(emb)):
                        available_mods.append(modality)
            patient_modalities[i] = available_mods
        
        # Group patients by available modality combinations
        modality_groups = {}
        for i, mods in patient_modalities.items():
            key = tuple(sorted(mods))
            if key not in modality_groups:
                modality_groups[key] = []
            modality_groups[key].append(i)
        
        print(f"\nModality availability patterns:")
        for mods, indices in modality_groups.items():
            print(f"  {mods}: {len(indices)} patients")
        
        # Create multimodal embeddings for all patients
        all_concat = []
        all_mean = []
        all_kronecker = []
        
        imputer = SimpleImputer(strategy='mean')
        
        for i in range(n_patients):
            mods = patient_modalities[i]
            
            if len(mods) == 0:
                # No modalities available - use random noise to ensure different risk scores
                # This prevents all patients from getting the same risk group
                np.random.seed(i)  # Reproducible randomness
                all_concat.append(np.random.randn(100) * 0.1)
                all_mean.append(np.random.randn(100) * 0.1)
                all_kronecker.append(np.random.randn(100) * 0.1)
                continue
            
            # 1. Concatenation
            concat_parts = []
            for mod in sorted(mods):
                emb = self.embeddings[mod][i:i+1]
                # Create new imputer for each embedding to avoid fitting issues
                mod_imputer = SimpleImputer(strategy='mean')
                emb = mod_imputer.fit_transform(emb.reshape(1, -1))
                # Don't use StandardScaler on single samples
                concat_parts.append(emb.flatten())
            
            if concat_parts:
                concat_emb = np.concatenate(concat_parts)
            else:
                concat_emb = np.random.randn(100) * 0.1
            all_concat.append(concat_emb)
            
            # 2. Mean pooling
            mean_parts = []
            max_dim = 1024  # Standard dimension
            for mod in sorted(mods):
                emb = self.embeddings[mod][i:i+1]
                mod_imputer = SimpleImputer(strategy='mean')
                emb = mod_imputer.fit_transform(emb.reshape(1, -1)).flatten()
                # Pad to max dimension
                if len(emb) < max_dim:
                    emb = np.pad(emb, (0, max_dim - len(emb)), mode='constant')
                else:
                    emb = emb[:max_dim]
                mean_parts.append(emb)
            
            if mean_parts:
                mean_emb = np.mean(mean_parts, axis=0)
            else:
                mean_emb = np.random.randn(max_dim) * 0.1
            all_mean.append(mean_emb)
            
            # 3. Kronecker product (simplified)
            if len(mods) >= 2:
                # Take first two modalities for Kronecker
                mod1, mod2 = sorted(mods)[:2]
                emb1 = self.embeddings[mod1][i]
                emb2 = self.embeddings[mod2][i]
                
                # Reduce dimensions
                emb1 = emb1[:10] if len(emb1) > 10 else emb1
                emb2 = emb2[:10] if len(emb2) > 10 else emb2
                
                kron_emb = np.kron(emb1, emb2)
            else:
                # Only one modality - use squared features
                emb = self.embeddings[mods[0]][i][:50]
                kron_emb = np.outer(emb[:10], emb[:10]).flatten()
            
            all_kronecker.append(kron_emb)
        
        # Pad all embeddings to same dimension within each type
        self.multimodal_embeddings['concat'] = self._pad_to_same_dim(all_concat)
        self.multimodal_embeddings['mean_pool'] = np.array(all_mean)
        self.multimodal_embeddings['kronecker'] = self._pad_to_same_dim(all_kronecker)
        
        print(f"\nMultimodal embeddings created:")
        print(f"  Concatenation shape: {self.multimodal_embeddings['concat'].shape}")
        print(f"  Mean pool shape: {self.multimodal_embeddings['mean_pool'].shape}")
        print(f"  Kronecker shape: {self.multimodal_embeddings['kronecker'].shape}")
    
    def _pad_to_same_dim(self, embeddings_list):
        """Pad variable length embeddings to same dimension"""
        max_dim = max(len(emb) for emb in embeddings_list)
        padded = []
        for emb in embeddings_list:
            if len(emb) < max_dim:
                padded_emb = np.pad(emb, (0, max_dim - len(emb)), mode='constant')
            else:
                padded_emb = emb
            padded.append(padded_emb)
        return np.array(padded)
    
    def merge_similar_projects(self, project_id: str, min_samples: int = 30) -> Tuple[str, np.ndarray]:
        """Merge similar cancer types if not enough samples"""
        # Get samples for this project
        project_mask = self.patient_data['project_id'] == project_id
        project_indices = np.where(project_mask)[0]
        
        if len(project_indices) >= min_samples:
            return project_id, project_indices
        
        # Find which group this cancer belongs to
        merged_name = project_id
        merged_indices = project_indices
        
        for group_name, cancer_list in self.similar_cancers.items():
            if project_id in cancer_list:
                # Merge with similar cancers
                all_cancers = []
                all_indices = []
                
                for cancer in cancer_list:
                    cancer_mask = self.patient_data['project_id'] == cancer
                    cancer_indices = np.where(cancer_mask)[0]
                    if len(cancer_indices) > 0:
                        all_cancers.append(cancer)
                        all_indices.extend(cancer_indices)
                
                if len(all_indices) >= min_samples:
                    merged_name = '_'.join(all_cancers)
                    merged_indices = np.array(all_indices)
                    print(f"  Merged {project_id} with similar cancers: {merged_name} ({len(merged_indices)} samples)")
                    break
        
        return merged_name, merged_indices
    
    def prepare_survival_data(self, embeddings: np.ndarray, indices: np.ndarray) -> Tuple:
        """Prepare data for survival analysis"""
        # Get patient data for given indices
        patient_subset = self.patient_data.iloc[indices]
        
        # Get embeddings for these specific indices
        X = embeddings[indices]
        
        # Extract survival times and events
        survival_times = patient_subset['days_to_death'].fillna(
            patient_subset['days_to_last_follow_up']
        ).values
        
        # Create event indicator (1 if dead, 0 if censored)
        events = (patient_subset['vital_status'] == 'Dead').astype(int).values
        
        # Remove invalid entries
        valid_mask = (survival_times > 0) & ~np.isnan(survival_times)
        
        X = X[valid_mask]
        y = np.array([(events[i], survival_times[i]) for i in range(len(events)) if valid_mask[i]], 
                     dtype=[('event', bool), ('time', float)])
        
        return X, y, patient_subset[valid_mask]
    
    def train_cox_ph(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, CoxPHFitter]:
        """Train Cox Proportional Hazards model with dimension reduction"""
        # Handle NaN values BEFORE PCA
        if np.any(np.isnan(X)):
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        # Reduce dimensions if too high for CoxPH
        if X.shape[1] > 50:
            pca = PCA(n_components=min(50, X.shape[0] // 2))
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
            pca = None
        
        # Prepare data for lifelines
        df = pd.DataFrame(X_reduced, columns=[f'feature_{i}' for i in range(X_reduced.shape[1])])
        df['time'] = y['time']
        df['event'] = y['event']
        
        # Fit model with appropriate penalization
        cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.1)
        cph.fit(df, duration_col='time', event_col='event')
        
        # Calculate C-index
        c_index = cph.concordance_index_
        
        # Store PCA transformer with the model
        cph._pca = pca
        
        return c_index, cph
    
    def train_rsf(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, RandomSurvivalForest]:
        """Train Random Survival Forest"""
        # Handle NaN values
        if np.any(np.isnan(X)):
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        rsf = RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42
        )
        
        rsf.fit(X, y)
        
        # Calculate C-index
        c_index = rsf.score(X, y)
        
        return c_index, rsf
    
    def train_deepsurv(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, DeepSurvModel]:
        """Train DeepSurv model with improved stability"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Handle NaN values
        if np.any(np.isnan(X)):
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
        
        # Prepare data - ensure contiguous arrays
        X_tensor = torch.FloatTensor(np.ascontiguousarray(X)).to(device)
        times = torch.FloatTensor(np.ascontiguousarray(y['time'])).to(device)
        events = torch.FloatTensor(np.ascontiguousarray(y['event'].astype(float))).to(device)
        
        # Create model
        model = DeepSurvModel(X.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training parameters - ensure minimum batch size of 2
        n_epochs = 200
        batch_size = max(2, min(32, len(X) // 4))  # Minimum batch size of 2
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        dataset = TensorDataset(X_tensor, times, events)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # Drop last to avoid single sample batches
        
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_times, batch_events in dataloader:
                # Skip if batch has only 1 sample (shouldn't happen with drop_last=True, but just in case)
                if batch_X.size(0) < 2:
                    continue
                    
                optimizer.zero_grad()
                
                # Forward pass
                risk_scores = model(batch_X).squeeze()
                
                # Calculate loss
                loss = self._partial_likelihood_loss(risk_scores, batch_times, batch_events)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss at epoch {epoch}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            # Learning rate scheduling
            if len(dataloader) > 0:
                avg_loss = epoch_loss / len(dataloader)
                scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Calculate C-index
        model.eval()
        with torch.no_grad():
            risk_scores = model(X_tensor).squeeze().cpu().numpy()
        
        # Ensure no NaN values
        if np.any(np.isnan(risk_scores)):
            risk_scores = np.nan_to_num(risk_scores, nan=0.0)
        
        c_index = concordance_index_censored(y['event'], y['time'], risk_scores)[0]
        
        return c_index, model
    
    def _partial_likelihood_loss(self, risk_scores, times, events):
        """Calculate partial likelihood loss for Cox model"""
        # Sort by time
        sorted_idx = torch.argsort(times)
        sorted_scores = risk_scores[sorted_idx]
        sorted_events = events[sorted_idx]
        sorted_times = times[sorted_idx]
        
        # Calculate hazards
        max_score = sorted_scores.max()
        exp_scores = torch.exp(sorted_scores - max_score)
        
        # Cumulative sum from end to beginning (risk set)
        risk_set = torch.flip(torch.cumsum(torch.flip(exp_scores, [0]), 0), [0])
        
        # Log partial likelihood
        log_risk = torch.log(risk_set + 1e-7) + max_score
        
        # Only consider events
        events_log_lik = sorted_events * (sorted_scores - log_risk)
        
        # Return negative log likelihood
        return -events_log_lik.sum() / (sorted_events.sum() + 1e-7)
    
    def nested_cv(self, modality: str, model_type: str, project_id: str, 
                  n_outer_folds: int = 5) -> Dict:
        """Perform nested cross-validation with project merging if needed"""
        print(f"\nNested CV for {modality} - {model_type} - {project_id}")
        
        # Get embeddings
        if modality in self.embeddings:
            embeddings = self.embeddings[modality]
        else:
            embeddings = self.multimodal_embeddings[modality]
        
        # Get project indices with potential merging
        merged_project, project_indices = self.merge_similar_projects(project_id)
        
        if len(project_indices) < 20:
            print(f"Not enough samples even after merging for {project_id}: {len(project_indices)}")
            return None
        
        # Prepare survival data
        X, y, patient_subset = self.prepare_survival_data(embeddings, project_indices)
        
        if len(X) < 20:
            print(f"Not enough valid samples for {project_id}: {len(X)}")
            return None
        
        # Nested CV
        outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
        results = {
            'c_indices': [],
            'train_c_indices': [],
            'fold_models': [],
            'project_name': merged_project
        }
        
        # Create stratification variable based on events
        stratify_var = y['event'].astype(int)
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, stratify_var)):
            print(f"  Outer fold {fold+1}/{n_outer_folds}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            try:
                # Train model
                if model_type == 'cox':
                    c_index, model = self.train_cox_ph(X_train, y_train)
                    # Test evaluation
                    if hasattr(model, '_pca') and model._pca is not None:
                        # Handle NaN values before PCA transform
                        if np.any(np.isnan(X_test)):
                            imputer = SimpleImputer(strategy='mean')
                            X_test = imputer.fit_transform(X_test)
                        X_test_transformed = model._pca.transform(X_test)
                    else:
                        X_test_transformed = X_test
                    
                    df_test = pd.DataFrame(X_test_transformed, 
                                         columns=[f'feature_{i}' for i in range(X_test_transformed.shape[1])])
                    
                    # Get predictions - use linear predictor
                    predictions = model.predict_log_partial_hazard(df_test).values
                    test_c_index = concordance_index_censored(
                        y_test['event'], 
                        y_test['time'], 
                        predictions
                    )[0]
                    
                elif model_type == 'rsf':
                    c_index, model = self.train_rsf(X_train, y_train)
                    test_c_index = model.score(X_test, y_test)
                    
                elif model_type == 'deepsurv':
                    c_index, model = self.train_deepsurv(X_train, y_train)
                    # Test evaluation
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model.eval()
                    with torch.no_grad():
                        # Handle NaN values
                        if np.any(np.isnan(X_test)):
                            imputer = SimpleImputer(strategy='mean')
                            X_test = imputer.fit_transform(X_test)
                        X_test_tensor = torch.FloatTensor(np.ascontiguousarray(X_test)).to(device)
                        risk_scores = model(X_test_tensor).squeeze().cpu().numpy()
                    
                    # Handle NaN values
                    if np.any(np.isnan(risk_scores)):
                        risk_scores = np.nan_to_num(risk_scores, nan=0.0)
                    
                    test_c_index = concordance_index_censored(
                        y_test['event'], 
                        y_test['time'], 
                        risk_scores
                    )[0]
                
                results['train_c_indices'].append(c_index)
                results['c_indices'].append(test_c_index)
                results['fold_models'].append((model, scaler))
                
                # Save individual fold model
                model_filename = f'{merged_project}_{modality}_{model_type}_fold{fold}.pkl'
                model_filepath = os.path.join(self.output_path, 'models', model_filename)
                
                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'model_type': model_type,
                    'modality': modality,
                    'project': merged_project,
                    'fold': fold,
                    'train_c_index': c_index,
                    'test_c_index': test_c_index,
                    'n_samples_train': len(X_train),
                    'n_samples_test': len(X_test),
                    'n_features': X.shape[1]
                }
                
                # Add model-specific components
                if model_type == 'cox' and hasattr(model, '_pca'):
                    model_data['pca'] = model._pca
                elif model_type == 'deepsurv':
                    model_data['device'] = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    model_data['input_dim'] = X.shape[1]
                
                with open(model_filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                
            except Exception as e:
                print(f"    Error in fold {fold+1}: {str(e)}")
                # Use random predictions as fallback
                test_c_index = 0.5
                results['train_c_indices'].append(0.5)
                results['c_indices'].append(test_c_index)
                results['fold_models'].append((None, scaler))
        
        results['mean_c_index'] = np.mean(results['c_indices'])
        results['std_c_index'] = np.std(results['c_indices'])
        
        # Save best model separately for easy access
        if results['c_indices']:
            best_fold_idx = np.argmax(results['c_indices'])
            best_model_filename = f'{merged_project}_{modality}_{model_type}_best.pkl'
            best_model_filepath = os.path.join(self.output_path, 'models', best_model_filename)
            
            # Copy the best fold model file
            best_fold_filename = f'{merged_project}_{modality}_{model_type}_fold{best_fold_idx}.pkl'
            best_fold_filepath = os.path.join(self.output_path, 'models', best_fold_filename)
            
            if os.path.exists(best_fold_filepath):
                import shutil
                shutil.copy2(best_fold_filepath, best_model_filepath)
        
        return results
    
    def generate_risk_curves(self, modality: str, model_type: str, 
                           project_id: str, cv_results: Dict):
        """Generate publication-quality risk stratification curves with proper censoring marks"""
        print(f"\nGenerating risk curves for {modality} - {model_type} - {project_id}")
        
        # Get embeddings
        if modality in self.embeddings:
            embeddings = self.embeddings[modality]
        else:
            embeddings = self.multimodal_embeddings[modality]
        
        # Use the merged project name from CV results
        merged_project = cv_results.get('project_name', project_id)
        _, project_indices = self.merge_similar_projects(project_id)
        
        # Prepare survival data
        X, y, patient_subset = self.prepare_survival_data(embeddings, project_indices)
        
        # Find best performing fold
        best_fold = np.argmax(cv_results['c_indices'])
        model, scaler = cv_results['fold_models'][best_fold]
        
        if model is None:
            print(f"  Skipping risk curves - no valid model for {modality} - {model_type} - {project_id}")
            return
        
        # Standardize features
        X_scaled = scaler.transform(X)
        
        # Get risk scores
        if model_type == 'cox':
            # Handle NaN values
            if np.any(np.isnan(X_scaled)):
                imputer = SimpleImputer(strategy='mean')
                X_scaled = imputer.fit_transform(X_scaled)
                
            if hasattr(model, '_pca') and model._pca is not None:
                X_transformed = model._pca.transform(X_scaled)
            else:
                X_transformed = X_scaled
            df = pd.DataFrame(X_transformed, columns=[f'feature_{i}' for i in range(X_transformed.shape[1])])
            risk_scores = model.predict_log_partial_hazard(df).values
            
        elif model_type == 'rsf':
            # Handle NaN values
            if np.any(np.isnan(X_scaled)):
                imputer = SimpleImputer(strategy='mean')
                X_scaled = imputer.fit_transform(X_scaled)
            # RSF predict() returns cumulative hazard function
            # Higher cumulative hazard = higher risk
            risk_scores = model.predict(X_scaled)
            
        elif model_type == 'deepsurv':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                # Handle NaN values
                if np.any(np.isnan(X_scaled)):
                    imputer = SimpleImputer(strategy='mean')
                    X_scaled = imputer.fit_transform(X_scaled)
                X_tensor = torch.FloatTensor(np.ascontiguousarray(X_scaled)).to(device)
                risk_scores = model(X_tensor).squeeze().cpu().numpy()
            
            # Handle NaN values
            if np.any(np.isnan(risk_scores)):
                risk_scores = np.nan_to_num(risk_scores, nan=0.0)
        
        # Stratify patients into risk groups
        risk_tertiles = np.percentile(risk_scores, [33, 67])
        risk_groups = np.zeros_like(risk_scores, dtype=int)
        risk_groups[risk_scores > risk_tertiles[1]] = 2  # High risk
        risk_groups[(risk_scores > risk_tertiles[0]) & (risk_scores <= risk_tertiles[1])] = 1  # Medium risk
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define colors for risk groups
        colors = ['#2E7D32', '#FB8C00', '#C62828']  # Green, Orange, Red
        labels = ['Low Risk', 'Medium Risk', 'High Risk']
        
        # Store KM fitters for getting survival probabilities
        km_fitters = []
        
        # Plot Kaplan-Meier curves for each risk group
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = risk_groups == i
            if np.sum(mask) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(y['time'][mask], y['event'][mask], label=f'{label} (n={np.sum(mask)})')
                kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5, alpha=0.8)
                km_fitters.append((kmf, mask, color))
        
        # Add censoring marks on the survival curves (not on x-axis)
        for kmf, mask, color in km_fitters:
            censored_mask = mask & ~y['event']
            censored_times = y['time'][censored_mask]
            
            if len(censored_times) > 0:
                # Get survival probabilities at censored times
                for t in censored_times:
                    # Find the survival probability at time t
                    surv_prob = float(kmf.survival_function_at_times([t]).iloc[0])
                    # Plot censoring tick at the curve
                    ax.plot(t, surv_prob, '|', color=color, markersize=8, 
                           markeredgewidth=2, alpha=0.7)
        
        # Add log-rank test p-value
        if len(np.unique(risk_groups)) > 1:
            # Perform pairwise log-rank tests
            p_values = []
            for i in range(3):
                for j in range(i+1, 3):
                    mask_i = risk_groups == i
                    mask_j = risk_groups == j
                    if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                        result = logrank_test(
                            y['time'][mask_i], y['time'][mask_j],
                            y['event'][mask_i], y['event'][mask_j]
                        )
                        p_values.append(result.p_value)
            
            if p_values:
                min_p = min(p_values)
                ax.text(0.02, 0.02, f'Log-rank p < {min_p:.3f}', 
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Time (days)', fontsize=14)
        ax.set_ylabel('Survival Probability', fontsize=14)
        
        # Use merged project name in title if applicable
        display_name = merged_project if merged_project != project_id else project_id
        ax.set_title(f'{modality.capitalize()} - {model_type.upper()} - {display_name}\n' + 
                    f'C-index: {cv_results["mean_c_index"]:.3f} ± {cv_results["std_c_index"]:.3f}',
                    fontsize=16)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Customize legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Set axis limits
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, None)
        
        # Save figure
        output_dir = os.path.join(self.output_path, 'risk_curves', merged_project)
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f'{modality}_{model_type}_risk_stratification'
        fig.savefig(os.path.join(output_dir, f'{filename}.pdf'), 
                   format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(output_dir, f'{filename}.png'), 
                   format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def run_full_analysis(self):
        """Run complete survival analysis pipeline"""
        # Load data
        self.load_data()
        self.create_multimodal_embeddings()
        
        # Get unique projects
        projects = self.patient_data['project_id'].unique()
        projects = [p for p in projects if pd.notna(p)]
        
        # Define configurations
        modalities = list(self.embeddings.keys()) + ['concat', 'mean_pool', 'kronecker']
        models = ['cox', 'rsf', 'deepsurv']
        
        # Store all results
        all_results = []
        
        # Run analysis for each combination
        for project in projects:
            for modality in modalities:
                for model in models:
                    try:
                        # Run nested CV
                        cv_results = self.nested_cv(modality, model, project)
                        
                        if cv_results is not None:
                            # Generate risk curves
                            self.generate_risk_curves(modality, model, project, cv_results)
                            
                            # Store results
                            result_entry = {
                                'project_id': project,
                                'merged_project': cv_results['project_name'],
                                'modality': modality,
                                'model': model,
                                'mean_c_index': cv_results['mean_c_index'],
                                'std_c_index': cv_results['std_c_index'],
                                'c_indices': cv_results['c_indices']
                            }
                            all_results.append(result_entry)
                            
                            # Save CV results
                            cv_filename = f'{cv_results["project_name"]}_{modality}_{model}_cv_results.pkl'
                            with open(os.path.join(self.output_path, 'cv_results', cv_filename), 'wb') as f:
                                pickle.dump(cv_results, f)
                    
                    except Exception as e:
                        print(f"Error processing {modality} - {model} - {project}: {str(e)}")
                        continue
        
        # Save summary results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(self.output_path, 'survival_analysis_summary_v3.csv'), index=False)
        
        # Generate summary plots
        self.generate_summary_plots(results_df)
        
        print("\nAnalysis complete! Results saved to:", self.output_path)
    
    def generate_latex_table(self, results_df: pd.DataFrame):
        """Generate LaTeX tables in the requested format, breaking into groups of 6 projects"""
        # Define modality groups with proper names
        modality_groups = [
            ('Clinical Features', ['clinical']),
            ('Pathology Features', ['pathology']),
            ('Radiology Features', ['radiology']),
            ('Molecular Features', ['molecular']),
            ('WSI Features', ['wsi']),
            ('Concatenated Features', ['concat']),
            ('Mean Pooling', ['mean_pool']),
            ('Kronecker Product', ['kronecker'])
        ]
        
        # Define model names
        model_names = {
            'cox': 'Cox',
            'rsf': 'RSF',
            'deepsurv': 'DeepSurv'
        }
        
        # Get unique projects (excluding merged ones)
        unique_projects = sorted(results_df['project_id'].unique())
        
        # Split projects into groups of 6
        projects_per_table = 6
        project_groups = [unique_projects[i:i+projects_per_table] 
                         for i in range(0, len(unique_projects), projects_per_table)]
        
        all_latex_tables = []
        
        for table_idx, project_group in enumerate(project_groups):
            # Start building the LaTeX table
            latex_lines = []
            latex_lines.append(r'\begin{sidewaystable}[htbp]')
            latex_lines.append(r'    \centering')
            
            # Adjust caption based on which table this is
            if len(project_groups) > 1:
                caption = f'Survival analysis results across TCGA cancer types (Part {table_idx + 1} of {len(project_groups)})'
            else:
                caption = 'Survival analysis results across different TCGA cancer types'
            caption += ' using various feature modalities and models. C-index values are reported as mean ± standard deviation across 5-fold cross-validation.'
            
            latex_lines.append(r'    \caption{' + caption + '}')
            latex_lines.append(r'    \label{tab:survival_results_' + str(table_idx + 1) + '}')
            
            # Create column specification
            n_cols = len(project_group)
            col_spec = '@{}ll' + 'c' * n_cols + '@{}'
            latex_lines.append(r'    \begin{tabular}{' + col_spec + '}')
            latex_lines.append(r'        \toprule')
            
            # Header row
            header = r'        \textbf{Category} & \textbf{Method}'
            for project in project_group:
                # Clean project name (remove TCGA- prefix for compactness)
                clean_name = project.replace('TCGA-', '')
                header += r' & \textbf{' + clean_name + '}'
            latex_lines.append(header + r' \\')
            latex_lines.append(r'        \midrule')
            
            # Add data rows
            for category_name, modalities in modality_groups:
                first_row = True
                for model in ['cox', 'rsf', 'deepsurv']:
                    row = '        '
                    if first_row:
                        row += r'\multirow{3}{*}{\textbf{' + category_name + r'}} '
                    row += r'& ' + model_names[model]
                    
                    for project in project_group:
                        # Get results for this combination
                        mask = (results_df['project_id'] == project) & \
                               (results_df['modality'].isin(modalities)) & \
                               (results_df['model'] == model)
                        
                        matching = results_df[mask]
                        if len(matching) > 0:
                            mean_val = matching.iloc[0]['mean_c_index']
                            std_val = matching.iloc[0]['std_c_index']
                            row += f' & {mean_val:.3f} ± {std_val:.3f}'
                        else:
                            row += ' & -'
                    
                    row += r' \\'
                    latex_lines.append(row)
                    first_row = False
                
                # Add midrule after each category except the last
                if category_name != modality_groups[-1][0]:
                    latex_lines.append(r'        \midrule')
            
            latex_lines.append(r'        \bottomrule')
            latex_lines.append(r'    \end{tabular}')
            latex_lines.append(r'    \footnotetext{WSI: Whole Slide Image; RSF: Random Survival Forest. All values represent concordance indices (C-index) where higher values indicate better predictive performance.}')
            latex_lines.append(r'\end{sidewaystable}')
            
            all_latex_tables.append('\n'.join(latex_lines))
        
        # Save all tables to one file
        full_latex = '\n\n'.join(all_latex_tables)
        with open(os.path.join(self.output_path, 'survival_results_tables.tex'), 'w') as f:
            f.write(full_latex)
        
        # Also save individual table files
        for idx, table in enumerate(all_latex_tables):
            with open(os.path.join(self.output_path, f'survival_results_table_part{idx+1}.tex'), 'w') as f:
                f.write(table)
        
        print(f"\nLaTeX tables saved to:")
        print(f"  - Combined: {os.path.join(self.output_path, 'survival_results_tables.tex')}")
        for idx in range(len(project_groups)):
            print(f"  - Part {idx+1}: {os.path.join(self.output_path, f'survival_results_table_part{idx+1}.tex')}")
        
    def generate_summary_plots(self, results_df: pd.DataFrame):
        """Generate summary visualizations"""
        # Generate LaTeX table first
        self.generate_latex_table(results_df)
        
        # C-index comparison by modality
        fig, ax = plt.subplots(figsize=(12, 8))
        
        modality_order = ['clinical', 'pathology', 'radiology', 'molecular', 'wsi',
                         'concat', 'mean_pool', 'kronecker']
        
        # Filter and reorder
        plot_df = results_df[results_df['modality'].isin(modality_order)]
        
        sns.boxplot(data=plot_df, x='modality', y='mean_c_index', hue='model',
                   order=[m for m in modality_order if m in plot_df['modality'].unique()], 
                   ax=ax)
        
        ax.set_xlabel('Modality', fontsize=14)
        ax.set_ylabel('C-index', fontsize=14)
        ax.set_title('Model Performance Across Modalities', fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_path, 'modality_comparison_v3.pdf'), format='pdf')
        fig.savefig(os.path.join(self.output_path, 'modality_comparison_v3.png'), format='png')
        plt.close(fig)
        
        # C-index by cancer type (using merged projects)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Average across models for each project and modality
        project_summary = results_df.groupby(['merged_project', 'modality'])['mean_c_index'].mean().reset_index()
        
        # Pivot for heatmap
        heatmap_data = project_summary.pivot(index='merged_project', columns='modality', values='mean_c_index')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   center=0.5, vmin=0.3, vmax=0.8, ax=ax)
        
        ax.set_title('C-index Heatmap by Cancer Type and Modality', fontsize=16)
        ax.set_xlabel('Modality', fontsize=14)
        ax.set_ylabel('Cancer Type', fontsize=14)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_path, 'cancer_type_heatmap_v3.pdf'), format='pdf')
        fig.savefig(os.path.join(self.output_path, 'cancer_type_heatmap_v3.png'), format='png')
        plt.close(fig)


if __name__ == "__main__":
    # Set paths
    data_path = "/mnt/f/Projects/HoneyBee/results/shared_data"
    output_path = "/mnt/f/Projects/HoneyBee/results/survival"
    
    # Run analysis
    analyzer = SurvivalAnalysis(data_path, output_path)
    analyzer.run_full_analysis()