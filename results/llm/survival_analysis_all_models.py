import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List
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
from collections import defaultdict
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


class StratifiedSurvivalAnalysis:
    """Stratified survival analysis for all LLM embeddings by cancer type"""
    
    def __init__(self, embeddings_dir: str = 'embeddings', output_dir: str = 'survival_stratified_all_models'):
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.results = defaultdict(dict)
        self.patient_ids = None  # Initialize before loading embeddings
        self.patient_id_to_idx = {}  # Mapping from patient ID to embedding index
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'cv_results'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'cancer_specific'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'risk_curves'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'risk_stratification'), exist_ok=True)
        
        # Load embeddings and patient data
        self.embeddings = self._load_embeddings()
        self.patient_data = self._load_patient_data()
        
        # Create embedding index column if we have patient IDs
        if self.patient_ids is not None:
            self.patient_id_to_idx = {pid: i for i, pid in enumerate(self.patient_ids)}
            self.patient_data['emb_idx'] = self.patient_data['patient_id'].map(self.patient_id_to_idx)
            # Remove patients without embeddings
            self.patient_data = self.patient_data[self.patient_data['emb_idx'].notna()].copy()
            self.patient_data['emb_idx'] = self.patient_data['emb_idx'].astype(int)
        else:
            # If no patient IDs mapping, assume order matches
            self.patient_data['emb_idx'] = range(len(self.patient_data))
        
        # Define cancer groups for stratification
        self.cancer_types = sorted(self.patient_data['cancer_type'].unique())
        
        # Minimum samples required for analysis
        self.min_samples = 30
        self.min_events = 10
    
    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load all available embeddings including all 4 models"""
        embeddings = {}
        
        # Include all 4 models
        embedding_files = {
            'clinical_gatortron': 'clinical_gatortron_embeddings.pkl',
            'clinical_qwen': 'clinical_qwen_embeddings.pkl',
            'clinical_medgemma': 'clinical_medgemma_embeddings.pkl',
            'clinical_llama': 'clinical_llama_embeddings.pkl',
            'pathology_gatortron': 'pathology_gatortron_embeddings.pkl',
            'pathology_qwen': 'pathology_qwen_embeddings.pkl',
            'pathology_medgemma': 'pathology_medgemma_embeddings.pkl',
            'pathology_llama': 'pathology_llama_embeddings.pkl'
        }
        
        for name, filename in embedding_files.items():
            filepath = os.path.join(self.embeddings_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    # Handle both dict and array formats
                    if isinstance(data, dict):
                        embeddings[name] = data['embeddings']
                        # Store patient IDs if available
                        if 'patient_ids' in data and not hasattr(self, 'patient_ids'):
                            self.patient_ids = data['patient_ids']
                    else:
                        embeddings[name] = data
                print(f"Loaded {name} embeddings: {embeddings[name].shape}")
            else:
                print(f"Warning: {filepath} not found, skipping {name}")
        
        return embeddings
    
    def _load_patient_data(self) -> pd.DataFrame:
        """Load patient metadata"""
        # Try multiple possible locations
        paths = [
            os.path.join(self.embeddings_dir, 'patient_data.csv'),
            os.path.join(self.embeddings_dir, '..', 'patient_data.csv'),
            os.path.join(os.path.dirname(self.embeddings_dir), 'patient_data.csv')
        ]
        
        # Look for pickle file with metadata from embedding generation
        pickle_path = os.path.join(self.embeddings_dir, 'all_embeddings.pkl')
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
                if 'metadata' in data:
                    metadata = data['metadata']
                    df = pd.DataFrame({
                        'patient_id': metadata['patient_ids'],
                        'cancer_type': metadata['project_ids'],
                        'survival_time': metadata['survival_time'],
                        'vital_status': metadata['survival_status']
                    })
                    # Clean cancer type names
                    df['cancer_type'] = df['cancer_type'].str.replace('TCGA-', '')
                    df['event'] = (df['vital_status'] == 1).astype(int)
                    return df
        
        # Fallback to CSV
        for path in paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Ensure required columns exist
                required_cols = ['patient_id', 'cancer_type', 'survival_time', 'event']
                if all(col in df.columns for col in required_cols):
                    return df
                else:
                    # Try to map columns
                    col_mapping = {
                        'case_submitter_id': 'patient_id',
                        'project_id': 'cancer_type',
                        'days_to_death': 'survival_time',
                        'vital_status': 'event'
                    }
                    df = df.rename(columns=col_mapping)
                    # Clean cancer type
                    if 'cancer_type' in df.columns:
                        df['cancer_type'] = df['cancer_type'].str.replace('TCGA-', '')
                    return df
        
        raise FileNotFoundError("Could not find patient data file")
    
    def fit_survival_models(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           embedding_name: str) -> Dict[str, float]:
        """Fit multiple survival models and return C-indices"""
        
        # Prepare survival data format
        y_train_structured = np.array([(event, time) for event, time in y_train],
                                     dtype=[('event', bool), ('time', float)])
        y_test_structured = np.array([(event, time) for event, time in y_test],
                                    dtype=[('event', bool), ('time', float)])
        
        results = {}
        
        # 1. Cox Proportional Hazards
        try:
            # Create DataFrame for Cox model
            train_df = pd.DataFrame(X_train, columns=[f'x{i}' for i in range(X_train.shape[1])])
            train_df['time'] = y_train[:, 1]
            train_df['event'] = y_train[:, 0]
            
            test_df = pd.DataFrame(X_test, columns=[f'x{i}' for i in range(X_test.shape[1])])
            
            # Fit Cox model
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train_df, duration_col='time', event_col='event', show_progress=False)
            
            # Predict risk scores
            risk_scores = cph.predict_partial_hazard(test_df).values
            
            # Calculate C-index
            c_index = concordance_index_censored(
                y_test_structured['event'],
                y_test_structured['time'],
                risk_scores  # Higher hazard ratio = higher risk
            )[0]
            results['cox'] = c_index
        except Exception as e:
            print(f"Cox model failed: {e}")
            results['cox'] = 0.5
        
        # 2. Random Survival Forest
        try:
            rsf = RandomSurvivalForest(
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42
            )
            rsf.fit(X_train, y_train_structured)
            
            # Predict risk scores
            risk_scores = rsf.predict(X_test)
            
            # Calculate C-index
            c_index = concordance_index_censored(
                y_test_structured['event'],
                y_test_structured['time'],
                risk_scores
            )[0]
            results['rsf'] = c_index
        except Exception as e:
            print(f"RSF model failed: {e}")
            results['rsf'] = 0.5
        
        # 3. DeepSurv (only if GPU available)
        if torch.cuda.is_available():
            try:
                device = torch.device('cuda')
                
                # Create model
                model = DeepSurvModel(X_train.shape[1]).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train).to(device)
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                y_train_tensor = torch.FloatTensor(y_train).to(device)
                
                # Training
                model.train()
                for epoch in range(100):
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor).squeeze()
                    
                    # Partial likelihood loss for Cox model
                    loss = self._cox_loss(outputs, y_train_tensor)
                    
                    loss.backward()
                    optimizer.step()
                
                # Evaluation
                model.eval()
                with torch.no_grad():
                    risk_scores = model(X_test_tensor).squeeze().cpu().numpy()
                
                # Calculate C-index
                c_index = concordance_index_censored(
                    y_test_structured['event'],
                    y_test_structured['time'],
                    risk_scores
                )[0]
                results['deepsurv'] = c_index
            except Exception as e:
                print(f"DeepSurv model failed: {e}")
                results['deepsurv'] = 0.5
        else:
            results['deepsurv'] = None
        
        return results
    
    def _cox_loss(self, risk_scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cox partial likelihood loss"""
        events = y[:, 0]
        times = y[:, 1]
        
        # Sort by time
        sorted_idx = torch.argsort(times, descending=True)
        risk_scores = risk_scores[sorted_idx]
        events = events[sorted_idx]
        
        # Calculate log partial likelihood
        max_risk = torch.max(risk_scores)
        risk_scores = risk_scores - max_risk
        
        log_risk = torch.log(torch.cumsum(torch.exp(risk_scores), dim=0))
        uncensored_likelihood = risk_scores - log_risk
        censored_likelihood = uncensored_likelihood * events
        
        # Negative log likelihood
        loss = -torch.sum(censored_likelihood) / torch.sum(events)
        
        return loss
    
    def generate_risk_stratification_curves(self, emb_name: str, emb_data: np.ndarray, 
                                          cancer_type: str, model_name: str, 
                                          cv_results: Dict[str, List[float]]):
        """Generate risk stratification curves for a specific cancer type"""
        # Get patients for this cancer type
        cancer_patients = self.patient_data[
            self.patient_data['cancer_type'] == cancer_type
        ].copy()
        
        # Get embeddings and survival data
        patient_indices = cancer_patients['emb_idx'].values
        X = emb_data[patient_indices]
        y = cancer_patients[['event', 'survival_time']].values
        
        # Preprocessing (same as in training)
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # PCA if high dimensional
        if X.shape[1] > 100:
            pca = PCA(n_components=min(50, X.shape[0] - 1))
            X = pca.fit_transform(X)
        
        # Get the best model from cross-validation
        best_fold_idx = np.argmax(cv_results[model_name]) if cv_results[model_name] else 0
        
        # Re-train the model on full dataset for risk stratification
        y_structured = np.array([(event, time) for event, time in y],
                               dtype=[('event', bool), ('time', float)])
        
        # Train the selected model
        if model_name == 'cox':
            # Create DataFrame for Cox model
            df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
            df['time'] = y[:, 1]
            df['event'] = y[:, 0]
            
            # Fit Cox model
            model = CoxPHFitter(penalizer=0.1)
            model.fit(df, duration_col='time', event_col='event', show_progress=False)
            
            # Get risk scores
            risk_scores = model.predict_partial_hazard(df.drop(['time', 'event'], axis=1)).values
            
        elif model_name == 'rsf':
            model = RandomSurvivalForest(
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42
            )
            model.fit(X, y_structured)
            risk_scores = model.predict(X)
            
        elif model_name == 'deepsurv' and torch.cuda.is_available():
            device = torch.device('cuda')
            model = DeepSurvModel(X.shape[1]).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(device)
            y_tensor = torch.FloatTensor(y).to(device)
            
            # Training
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_tensor).squeeze()
                loss = self._cox_loss(outputs, y_tensor)
                loss.backward()
                optimizer.step()
            
            # Get risk scores
            model.eval()
            with torch.no_grad():
                risk_scores = model(X_tensor).squeeze().cpu().numpy()
        else:
            return  # Skip if model not available
        
        # Stratify patients into risk groups (tertiles)
        risk_tertiles = np.percentile(risk_scores, [33, 67])
        risk_groups = np.zeros_like(risk_scores, dtype=int)
        risk_groups[risk_scores > risk_tertiles[1]] = 2  # High risk
        risk_groups[(risk_scores > risk_tertiles[0]) & (risk_scores <= risk_tertiles[1])] = 1  # Medium risk
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define colors for risk groups
        colors = ['#2E7D32', '#FB8C00', '#C62828']  # Green, Orange, Red
        labels = ['Low Risk', 'Medium Risk', 'High Risk']
        
        # Store KM fitters for statistical tests
        km_fitters = []
        
        # Plot Kaplan-Meier curves for each risk group
        for i, (color, label) in enumerate(zip(colors, labels)):
            mask = risk_groups == i
            if np.sum(mask) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(y[mask, 1], y[mask, 0], label=f'{label} (n={np.sum(mask)})')
                kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5, alpha=0.8)
                km_fitters.append((kmf, mask, color))
        
        # Add censoring marks
        for kmf, mask, color in km_fitters:
            censored_mask = mask & (y[:, 0] == 0)
            censored_times = y[censored_mask, 1]
            
            if len(censored_times) > 0:
                for t in censored_times:
                    # Get survival probability at time t
                    surv_prob = float(kmf.survival_function_at_times([t]).iloc[0])
                    ax.plot(t, surv_prob, '|', color=color, markersize=8, 
                           markeredgewidth=2, alpha=0.7)
        
        # Add log-rank test p-value
        if len(np.unique(risk_groups)) > 1:
            p_values = []
            for i in range(3):
                for j in range(i+1, 3):
                    mask_i = risk_groups == i
                    mask_j = risk_groups == j
                    if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                        result = logrank_test(
                            y[mask_i, 1], y[mask_j, 1],
                            y[mask_i, 0], y[mask_j, 0]
                        )
                        p_values.append(result.p_value)
            
            if p_values:
                min_p = min(p_values)
                p_text = f'p < {min_p:.3f}' if min_p < 0.001 else f'p = {min_p:.3f}'
                ax.text(0.02, 0.02, f'Log-rank {p_text}', 
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Time (days)', fontsize=14)
        ax.set_ylabel('Survival Probability', fontsize=14)
        
        # Get mean C-index for title
        mean_ci = np.mean(cv_results[model_name]) if cv_results[model_name] else 0.5
        std_ci = np.std(cv_results[model_name]) if cv_results[model_name] else 0.0
        
        ax.set_title(f'{cancer_type} - {emb_name} - {model_name.upper()}\n' + 
                    f'C-index: {mean_ci:.3f} ± {std_ci:.3f}',
                    fontsize=16)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Customize legend
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Set axis limits
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, None)
        
        # Save figure
        output_dir = os.path.join(self.output_dir, 'risk_stratification', cancer_type)
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f'{emb_name}_{model_name}_risk_stratification'
        fig.savefig(os.path.join(output_dir, f'{filename}.pdf'), 
                   format='pdf', bbox_inches='tight', dpi=300)
        fig.savefig(os.path.join(output_dir, f'{filename}.png'), 
                   format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
    
    def run_stratified_analysis(self):
        """Run survival analysis stratified by cancer type"""
        all_results = []
        
        # Analyze each embedding type
        for emb_name, emb_data in self.embeddings.items():
            print(f"\n=== Analyzing {emb_name} ===")
            
            # Analyze each cancer type
            for cancer_type in self.cancer_types:
                # Get patients for this cancer type
                cancer_patients = self.patient_data[
                    self.patient_data['cancer_type'] == cancer_type
                ].copy()
                
                n_patients = len(cancer_patients)
                n_events = cancer_patients['event'].sum()
                
                # Skip if insufficient data
                if n_patients < self.min_samples or n_events < self.min_events:
                    print(f"Skipping {cancer_type}: {n_patients} patients, {n_events} events")
                    continue
                
                print(f"\nAnalyzing {cancer_type}: {n_patients} patients, {n_events} events")
                
                # Get embeddings for these patients
                patient_indices = cancer_patients['emb_idx'].values
                X = emb_data[patient_indices]
                y = cancer_patients[['event', 'survival_time']].values
                
                # Preprocessing
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(X)
                
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                
                # PCA if high dimensional
                if X.shape[1] > 100:
                    pca = PCA(n_components=min(50, X.shape[0] - 1))
                    X = pca.fit_transform(X)
                
                # Cross-validation
                cv_results = defaultdict(list)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, cancer_patients['event'])):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Fit models
                    fold_results = self.fit_survival_models(
                        X_train, y_train, X_test, y_test, emb_name
                    )
                    
                    for model_name, c_index in fold_results.items():
                        if c_index is not None:
                            cv_results[model_name].append(c_index)
                
                # Calculate mean and std
                result_row = {
                    'cancer_type': cancer_type,
                    'embedding': emb_name,
                    'n_patients': n_patients,
                    'n_events': n_events,
                    'event_rate': n_events / n_patients
                }
                
                for model_name in ['cox', 'rsf', 'deepsurv']:
                    if model_name in cv_results and cv_results[model_name]:
                        mean_ci = np.mean(cv_results[model_name])
                        std_ci = np.std(cv_results[model_name])
                        result_row[f'{model_name}_mean_ci'] = mean_ci
                        result_row[f'{model_name}_std_ci'] = std_ci
                        
                        # Generate risk stratification curves
                        try:
                            self.generate_risk_stratification_curves(
                                emb_name, emb_data, cancer_type, model_name, cv_results
                            )
                        except Exception as e:
                            print(f"  Error generating risk curves for {model_name}: {e}")
                    else:
                        result_row[f'{model_name}_mean_ci'] = None
                        result_row[f'{model_name}_std_ci'] = None
                
                all_results.append(result_row)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_df.to_csv(
            os.path.join(self.output_dir, 'stratified_summary_all_models.csv'),
            index=False
        )
        
        # Save detailed results
        with open(os.path.join(self.output_dir, 'stratified_results_all_models.pkl'), 'wb') as f:
            pickle.dump({
                'results_df': results_df,
                'all_results': all_results,
                'cancer_types': self.cancer_types,
                'embeddings': list(self.embeddings.keys())
            }, f)
        
        return results_df
    
    def plot_results(self, results_df: pd.DataFrame):
        """Create visualizations of survival analysis results"""
        
        # 1. Heatmap of C-indices by cancer type and embedding
        self._plot_heatmap(results_df)
        
        # 2. Box plots comparing models
        self._plot_boxplots(results_df)
        
        # 3. Model comparison by embedding type
        self._plot_model_comparison(results_df)
        
        # 4. Clinical vs Pathology comparison
        self._plot_clinical_vs_pathology(results_df)
        
        # 5. Risk stratification summary
        self.plot_risk_stratification_summary(results_df)
    
    def _plot_heatmap(self, results_df: pd.DataFrame):
        """Create heatmap of C-indices"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        models = ['cox', 'rsf', 'deepsurv']
        model_names = ['Cox PH', 'Random Survival Forest', 'DeepSurv']
        
        for idx, (model, model_name) in enumerate(zip(models, model_names)):
            # Pivot data for heatmap
            pivot_data = results_df.pivot(
                index='cancer_type',
                columns='embedding',
                values=f'{model}_mean_ci'
            )
            
            # Create heatmap
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                vmin=0.5,
                vmax=0.9,
                ax=axes[idx],
                cbar_kws={'label': 'C-index'}
            )
            
            axes[idx].set_title(f'{model_name} Performance')
            axes[idx].set_xlabel('Embedding Type')
            axes[idx].set_ylabel('Cancer Type')
        
        plt.suptitle('Survival Prediction Performance Across Cancer Types', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_heatmap_all_models.png'), dpi=300)
        plt.close()
    
    def _plot_boxplots(self, results_df: pd.DataFrame):
        """Create box plots comparing models"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for box plot
        plot_data = []
        
        for _, row in results_df.iterrows():
            for model in ['cox', 'rsf', 'deepsurv']:
                if row[f'{model}_mean_ci'] is not None:
                    plot_data.append({
                        'Model': model.upper(),
                        'C-index': row[f'{model}_mean_ci'],
                        'Embedding': row['embedding']
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create box plot
        sns.boxplot(data=plot_df, x='Embedding', y='C-index', hue='Model', ax=ax)
        
        ax.set_title('Model Performance Comparison Across Embeddings')
        ax.set_ylabel('C-index')
        ax.set_xlabel('Embedding Type')
        plt.xticks(rotation=45, ha='right')
        
        # Add reference line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison_boxplot_all_models.png'), dpi=300)
        plt.close()
    
    def _plot_model_comparison(self, results_df: pd.DataFrame):
        """Compare encoder vs decoder and medical vs general models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Define model categories
        encoder_models = ['gatortron', 'qwen']
        decoder_models = ['medgemma', 'llama']
        medical_models = ['gatortron', 'medgemma']
        general_models = ['qwen', 'llama']
        
        # Encoder vs Decoder comparison
        encoder_data = results_df[results_df['embedding'].str.contains('|'.join(encoder_models))]
        decoder_data = results_df[results_df['embedding'].str.contains('|'.join(decoder_models))]
        
        comparison_data = []
        
        if not encoder_data.empty:
            for model in ['cox', 'rsf', 'deepsurv']:
                col = f'{model}_mean_ci'
                if col in encoder_data.columns:
                    values = encoder_data[col].dropna()
                    if not values.empty:
                        comparison_data.append({
                            'Architecture': 'Encoder-only',
                            'Model': model.upper(),
                            'C-index': values.mean()
                        })
        
        if not decoder_data.empty:
            for model in ['cox', 'rsf', 'deepsurv']:
                col = f'{model}_mean_ci'
                if col in decoder_data.columns:
                    values = decoder_data[col].dropna()
                    if not values.empty:
                        comparison_data.append({
                            'Architecture': 'Decoder-only',
                            'Model': model.upper(),
                            'C-index': values.mean()
                        })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            sns.barplot(data=comp_df, x='Model', y='C-index', hue='Architecture', ax=ax1)
            ax1.set_title('Encoder vs Decoder Architecture')
            ax1.set_ylim(0.5, 0.9)
        
        # Medical vs General comparison
        medical_data = results_df[results_df['embedding'].str.contains('|'.join(medical_models))]
        general_data = results_df[results_df['embedding'].str.contains('|'.join(general_models))]
        
        comparison_data2 = []
        
        if not medical_data.empty:
            for model in ['cox', 'rsf', 'deepsurv']:
                col = f'{model}_mean_ci'
                if col in medical_data.columns:
                    values = medical_data[col].dropna()
                    if not values.empty:
                        comparison_data2.append({
                            'Domain': 'Medical-focused',
                            'Model': model.upper(),
                            'C-index': values.mean()
                        })
        
        if not general_data.empty:
            for model in ['cox', 'rsf', 'deepsurv']:
                col = f'{model}_mean_ci'
                if col in general_data.columns:
                    values = general_data[col].dropna()
                    if not values.empty:
                        comparison_data2.append({
                            'Domain': 'General-purpose',
                            'Model': model.upper(),
                            'C-index': values.mean()
                        })
        
        if comparison_data2:
            comp_df2 = pd.DataFrame(comparison_data2)
            sns.barplot(data=comp_df2, x='Model', y='C-index', hue='Domain', ax=ax2)
            ax2.set_title('Medical vs General Purpose Models')
            ax2.set_ylim(0.5, 0.9)
        
        plt.suptitle('Model Architecture and Domain Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'architecture_comparison_all_models.png'), dpi=300)
        plt.close()
    
    def _plot_clinical_vs_pathology(self, results_df: pd.DataFrame):
        """Compare clinical vs pathology embeddings"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Separate clinical and pathology results
        clinical_results = results_df[results_df['embedding'].str.contains('clinical')]
        pathology_results = results_df[results_df['embedding'].str.contains('pathology')]
        
        # Calculate mean C-indices
        plot_data = []
        
        for model_type in ['clinical', 'pathology']:
            subset = clinical_results if model_type == 'clinical' else pathology_results
            
            for survival_model in ['cox', 'rsf', 'deepsurv']:
                col = f'{survival_model}_mean_ci'
                if col in subset.columns:
                    values = subset[col].dropna()
                    if not values.empty:
                        plot_data.append({
                            'Text Type': model_type.capitalize(),
                            'Model': survival_model.upper(),
                            'C-index': values.mean()
                        })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.barplot(data=plot_df, x='Model', y='C-index', hue='Text Type', ax=ax)
            
            ax.set_title('Clinical vs Pathology Text Performance')
            ax.set_ylabel('Mean C-index')
            ax.set_ylim(0.5, 0.9)
            
            # Add reference line
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'clinical_vs_pathology_all_models.png'), dpi=300)
        plt.close()
    
    def plot_risk_stratification_summary(self, results_df: pd.DataFrame):
        """Create a summary plot of risk stratification performance across cancer types"""
        # Filter for only significant results (where we have enough samples)
        significant_results = results_df[results_df['n_patients'] >= self.min_samples].copy()
        
        if significant_results.empty:
            return
        
        # Create figure with subplots for each model type
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        models = ['cox', 'rsf', 'deepsurv']
        model_names = ['Cox PH', 'Random Survival Forest', 'DeepSurv']
        
        for idx, (model, model_name) in enumerate(zip(models, model_names)):
            ax = axes[idx]
            
            # Get data for this model
            model_col = f'{model}_mean_ci'
            model_data = significant_results[significant_results[model_col].notna()]
            
            if model_data.empty:
                ax.text(0.5, 0.5, 'No data available', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(model_name)
                continue
            
            # Group by cancer type and get best performing embedding
            best_by_cancer = []
            for cancer in model_data['cancer_type'].unique():
                cancer_data = model_data[model_data['cancer_type'] == cancer]
                best_idx = cancer_data[model_col].idxmax()
                best_row = cancer_data.loc[best_idx]
                best_by_cancer.append({
                    'cancer_type': cancer,
                    'best_embedding': best_row['embedding'],
                    'c_index': best_row[model_col],
                    'n_patients': best_row['n_patients']
                })
            
            best_df = pd.DataFrame(best_by_cancer).sort_values('c_index', ascending=True)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(best_df))
            ax.barh(y_pos, best_df['c_index'], color='steelblue', alpha=0.8)
            
            # Add cancer type labels with sample sizes
            labels = [f"{row['cancer_type']} (n={row['n_patients']})" 
                     for _, row in best_df.iterrows()]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            
            # Add best embedding annotations
            for i, (_, row) in enumerate(best_df.iterrows()):
                emb_short = row['best_embedding'].replace('_embeddings', '').replace('_', ' ')
                ax.text(row['c_index'] + 0.005, i, emb_short, 
                       va='center', fontsize=9, alpha=0.7)
            
            # Customize plot
            ax.set_xlabel('C-index', fontsize=12)
            ax.set_title(model_name, fontsize=14)
            ax.set_xlim(0.5, 1.0)
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, axis='x', alpha=0.3)
        
        plt.suptitle('Best Risk Stratification Performance by Cancer Type', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'risk_stratification_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, results_df: pd.DataFrame):
        """Create a comprehensive summary report"""
        report_path = os.path.join(self.output_dir, 'survival_analysis_report_all_models.md')
        
        with open(report_path, 'w') as f:
            f.write("# Survival Analysis Report - All Models\n\n")
            
            # Overall summary
            f.write("## Executive Summary\n\n")
            
            # Best performing embedding
            best_row = results_df.loc[results_df['cox_mean_ci'].idxmax()]
            f.write(f"**Best Overall Performance:**\n")
            f.write(f"- Embedding: {best_row['embedding']}\n")
            f.write(f"- Cancer Type: {best_row['cancer_type']}\n")
            f.write(f"- Cox C-index: {best_row['cox_mean_ci']:.3f}\n\n")
            
            # Model comparison
            f.write("## Model Performance Summary\n\n")
            
            # Average performance by embedding type
            f.write("### Average C-index by Embedding Type\n\n")
            f.write("| Embedding | Cox PH | RSF | DeepSurv |\n")
            f.write("|-----------|--------|-----|----------|\n")
            
            for emb in sorted(results_df['embedding'].unique()):
                emb_data = results_df[results_df['embedding'] == emb]
                cox_mean = emb_data['cox_mean_ci'].mean()
                rsf_mean = emb_data['rsf_mean_ci'].mean()
                deepsurv_mean = emb_data['deepsurv_mean_ci'].mean()
                
                f.write(f"| {emb} | {cox_mean:.3f} | {rsf_mean:.3f} | ")
                if pd.notna(deepsurv_mean):
                    f.write(f"{deepsurv_mean:.3f} |\n")
                else:
                    f.write("N/A |\n")
            
            # Clinical vs Pathology
            f.write("\n### Clinical vs Pathology Performance\n\n")
            
            clinical_cox = results_df[results_df['embedding'].str.contains('clinical')]['cox_mean_ci'].mean()
            pathology_cox = results_df[results_df['embedding'].str.contains('pathology')]['cox_mean_ci'].mean()
            
            f.write(f"- **Clinical embeddings**: {clinical_cox:.3f} average C-index\n")
            f.write(f"- **Pathology embeddings**: {pathology_cox:.3f} average C-index\n")
            f.write(f"- **Difference**: {abs(clinical_cox - pathology_cox):.3f}\n\n")
            
            # Model architecture comparison
            f.write("### Architecture Comparison\n\n")
            
            encoder_models = ['gatortron', 'qwen']
            decoder_models = ['medgemma', 'llama']
            
            encoder_mean = results_df[
                results_df['embedding'].str.contains('|'.join(encoder_models))
            ]['cox_mean_ci'].mean()
            
            decoder_mean = results_df[
                results_df['embedding'].str.contains('|'.join(decoder_models))
            ]['cox_mean_ci'].mean()
            
            f.write(f"- **Encoder-only models**: {encoder_mean:.3f} average C-index\n")
            f.write(f"- **Decoder-only models**: {decoder_mean:.3f} average C-index\n\n")
            
            # Domain comparison
            f.write("### Domain Comparison\n\n")
            
            medical_models = ['gatortron', 'medgemma']
            general_models = ['qwen', 'llama']
            
            medical_mean = results_df[
                results_df['embedding'].str.contains('|'.join(medical_models))
            ]['cox_mean_ci'].mean()
            
            general_mean = results_df[
                results_df['embedding'].str.contains('|'.join(general_models))
            ]['cox_mean_ci'].mean()
            
            f.write(f"- **Medical-focused models**: {medical_mean:.3f} average C-index\n")
            f.write(f"- **General-purpose models**: {general_mean:.3f} average C-index\n\n")
            
            # Cancer-specific results
            f.write("## Cancer-Specific Results\n\n")
            
            for cancer in sorted(results_df['cancer_type'].unique()):
                cancer_data = results_df[results_df['cancer_type'] == cancer]
                f.write(f"### {cancer}\n")
                f.write(f"- Patients: {cancer_data.iloc[0]['n_patients']}\n")
                f.write(f"- Events: {cancer_data.iloc[0]['n_events']}\n")
                f.write(f"- Best embedding: {cancer_data.loc[cancer_data['cox_mean_ci'].idxmax()]['embedding']}")
                f.write(f" (C-index: {cancer_data['cox_mean_ci'].max():.3f})\n\n")
            
            # Technical details
            f.write("## Technical Details\n\n")
            f.write("- **Cross-validation**: 5-fold stratified\n")
            f.write("- **Preprocessing**: StandardScaler + PCA (if >100 dimensions)\n")
            f.write("- **Minimum requirements**: 30 patients, 10 events per cancer type\n")
            f.write("- **Models evaluated**: Cox PH, Random Survival Forest, DeepSurv (GPU only)\n")
            f.write("- **Risk stratification**: Patients divided into tertiles (Low/Medium/High risk)\n")
            f.write("- **Statistical testing**: Log-rank test for group comparisons\n\n")
            
            # Risk stratification note
            f.write("## Risk Stratification Curves\n\n")
            f.write("Risk stratification curves have been generated for each cancer type with sufficient data.\n")
            f.write("These curves show Kaplan-Meier survival probabilities for Low, Medium, and High risk groups\n")
            f.write("as determined by each model's risk scores. Curves are saved in the `risk_stratification/` directory\n")
            f.write("organized by cancer type.\n")
        
        print(f"\nSummary report saved to: {report_path}")


def main():
    """Main execution function"""
    # Set paths
    embeddings_dir = "/mnt/f/Projects/HoneyBee/results/llm/embeddings"
    output_dir = "/mnt/f/Projects/HoneyBee/results/llm/survival_stratified_all_models"
    
    # Run analysis
    analyzer = StratifiedSurvivalAnalysis(embeddings_dir, output_dir)
    
    print("Running stratified survival analysis for all models...")
    results_df = analyzer.run_stratified_analysis()
    
    print("\nCreating visualizations...")
    analyzer.plot_results(results_df)
    
    print("\nGenerating summary report...")
    analyzer.create_summary_report(results_df)
    
    print("\nSurvival analysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()