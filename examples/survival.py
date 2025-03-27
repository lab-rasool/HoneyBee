import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
import warnings
import pickle
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Constants
EMBEDDINGS_DIR = "multimodal_analysis_results/embeddings"
RESULTS_DIR = "survival_analysis_results"


def create_directory_structure():
    """Create organized directory structure for saving results."""
    # Main directories
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Subfolder for different types of figures
    figures_subfolders = [
        "kaplan_meier",
        "stratified_curves",
        "comparison_plots",
        "cross_validation",
    ]

    # Create figure subfolders
    figure_dirs = {}
    for subfolder in figures_subfolders:
        full_path = os.path.join(RESULTS_DIR, "figures", subfolder)
        os.makedirs(full_path, exist_ok=True)
        figure_dirs[subfolder] = full_path

    # Create other necessary directories
    models_dir = os.path.join(RESULTS_DIR, "models")
    data_dir = os.path.join(RESULTS_DIR, "data")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    return {"figures": figure_dirs, "models": models_dir, "data": data_dir}


# Create directories and get paths
DIRS = create_directory_structure()


def load_survival_data():
    """
    Load patient data with embeddings and prepare survival variables.
    """
    print("Loading patient data and embeddings...")

    # Load the patient data with embeddings paths
    patients_df = pd.read_csv(
        os.path.join(EMBEDDINGS_DIR, "patient_data_with_embeddings.csv")
    )

    # Extract survival information
    # First, identify the appropriate column names for survival data
    possible_survival_days_cols = [
        "days_to_death",
        "days_to_last_follow_up",
        "survival_days",
        "days_to_last_followup",
        "overall_survival_days",
    ]
    possible_vital_status_cols = [
        "vital_status",
        "dead",
        "deceased",
        "death",
        "OS_status",
    ]

    # Find the actual column names in our data
    survival_days_col = None
    vital_status_col = None

    for col in possible_survival_days_cols:
        if col in patients_df.columns:
            survival_days_col = col
            break

    for col in possible_vital_status_cols:
        if col in patients_df.columns:
            vital_status_col = col
            break

    if survival_days_col is None or vital_status_col is None:
        print("Warning: Could not find survival information in the data.")
        print(f"Available columns: {patients_df.columns.tolist()}")
        # Create dummy survival data for demonstration purposes
        patients_df["survival_days"] = np.random.randint(
            10, 3650, size=len(patients_df)
        )
        patients_df["vital_status"] = np.random.choice(
            ["Alive", "Dead"], size=len(patients_df), p=[0.7, 0.3]
        )
        survival_days_col = "survival_days"
        vital_status_col = "vital_status"
        print("Created dummy survival data for demonstration.")

    # Standardize vital status to 1=Dead, 0=Alive
    patients_df["event"] = 0
    if vital_status_col == "vital_status":
        patients_df.loc[
            patients_df[vital_status_col].isin(
                ["Dead", "Deceased", "dead", "deceased", "DECEASED", "TRUE", True, 1]
            ),
            "event",
        ] = 1
    else:
        # For other potential column names, we may need different logic
        patients_df.loc[patients_df[vital_status_col] == 1, "event"] = 1

    # Store the original indices before filtering
    patients_df["original_index"] = patients_df.index.values

    # Handle missing or problematic survival days
    patients_df[survival_days_col] = pd.to_numeric(
        patients_df[survival_days_col], errors="coerce"
    )
    patients_df = patients_df.dropna(subset=[survival_days_col])
    patients_df = patients_df[patients_df[survival_days_col] > 0]

    # Print summary of survival data
    print(f"Total patients with survival data: {len(patients_df)}")
    print(f"Number of events (deaths): {patients_df['event'].sum()}")
    print(f"Censoring rate: {1 - patients_df['event'].mean():.2f}")
    print(f"Median follow-up time: {patients_df[survival_days_col].median():.1f} days")

    # Load the embeddings for each modality
    print("Loading embeddings...")

    # Check which embedding files exist and load them
    modalities = ["clinical", "pathology", "radiology", "molecular", "multimodal"]
    embeddings_dict = {}

    for modality in modalities:
        embedding_file = os.path.join(EMBEDDINGS_DIR, f"{modality}_embeddings.npy")
        if os.path.exists(embedding_file):
            full_embeddings = np.load(embedding_file)

            # Filter embeddings to match patients with valid survival data
            valid_indices = patients_df["original_index"].values
            filtered_embeddings = full_embeddings[valid_indices]
            embeddings_dict[modality] = filtered_embeddings
            print(
                f"Loaded {modality} embeddings with shape {full_embeddings.shape} -> {filtered_embeddings.shape}"
            )

    # Remove the temporary index column
    patients_df = patients_df.drop(columns=["original_index"])

    return patients_df, embeddings_dict, survival_days_col


def save_figure(fig, modality, figure_type, name, formats=["svg", "pdf"]):
    """
    Save a figure in multiple formats to the appropriate directory.

    Parameters:
    - fig: matplotlib figure object
    - modality: which data modality (clinical, radiology, etc.)
    - figure_type: type of figure (kaplan_meier, stratified_curves, etc.)
    - name: filename without extension
    - formats: list of file formats to save
    """
    # Get the appropriate directory
    save_dir = DIRS["figures"].get(figure_type, DIRS["figures"]["comparison_plots"])

    # Create full filename
    base_filename = f"{modality}_{name}" if modality else name

    # Save in each format
    for fmt in formats:
        filename = os.path.join(save_dir, f"{base_filename}.{fmt}")
        dpi = 600 if fmt == "pdf" else 300
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", format=fmt)

    plt.close(fig)


def plot_survival_curves(patients_df, survival_days_col, modality, risk_scores=None):
    """
    Plot Kaplan-Meier survival curve for all patients.
    If risk_scores are provided, stratify patients into risk groups.
    """
    fig = plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    if risk_scores is not None:
        # Stratify patients into risk groups (tertiles)
        risk_tertiles = pd.qcut(
            risk_scores, q=3, labels=["Low", "Intermediate", "High"]
        )
        patients_with_risk = patients_df.copy()
        patients_with_risk["risk_group"] = risk_tertiles

        # Define color scheme for risk groups
        risk_colors = {
            "High": "#317EC2",  # Blue
            "Intermediate": "#E6862AF6",  # Orange
            "Low": "#C13930",  # Red
        }

        # Plot KM curves for each risk group with specific colors
        for risk_group in ["Low", "Intermediate", "High"]:
            group_data = patients_with_risk[
                patients_with_risk["risk_group"] == risk_group
            ]
            if len(group_data) > 0:
                kmf.fit(
                    group_data[survival_days_col],
                    group_data["event"],
                    label=f"{risk_group} risk",
                )
                kmf.plot(ci_show=False, color=risk_colors[risk_group], linewidth=2.5)

        figure_type = "stratified_curves"
        name = "stratified_kaplan_meier"
    else:
        # Fit and plot a single KM curve for all patients
        kmf.fit(
            patients_df[survival_days_col], patients_df["event"], label="All patients"
        )
        kmf.plot()

        figure_type = "kaplan_meier"
        name = "kaplan_meier_curve"

    plt.ylim(0, 1.05)
    plt.xlim(0, patients_df[survival_days_col].max())
    # remove the top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    # remove the legend completely
    plt.gca().get_legend().remove()
    # remove x and y labels
    plt.xlabel("")
    plt.ylabel("")

    # remove axes
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    # Clean up figure style for BioRender import
    plt.tight_layout()

    # Save figures
    save_figure(fig, modality, figure_type, name)


def run_cox_model(modality, embeddings, patients_df, survival_days_col):
    """Run a Cox Proportional Hazards model."""
    print(f"\nRunning Cox PH model for {modality} embeddings...")

    # Prepare the data
    X = embeddings

    # Flatten multi-dimensional embeddings if necessary
    original_shape = X.shape
    if len(original_shape) > 2:
        print(f"Flattening {modality} embeddings from {original_shape} to 2D...")
        X = X.reshape(original_shape[0], -1)
        print(f"Flattened shape: {X.shape}")

    y_time = patients_df[survival_days_col].values
    y_event = patients_df["event"].values

    # Split the data
    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = (
        train_test_split(
            X,
            y_time,
            y_event,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y_event,
        )
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Prepare data for Cox model - using PCA for high-dimensional data
    reduce_to = 50
    if X_train.shape[1] > reduce_to:
        pca = PCA(n_components=reduce_to)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        print(
            f"Reduced dimensions from {X_train.shape[1]} to {reduce_to} for Cox PH model"
        )

        # Save PCA object for later use
        with open(os.path.join(DIRS["models"], f"{modality}_pca.pkl"), "wb") as f:
            pickle.dump(pca, f)
    else:
        X_train_pca = X_train
        X_test_pca = X_test

    # Create dataframes for CoxPH
    train_df = pd.DataFrame(
        data=X_train_pca, columns=[f"feature_{i}" for i in range(X_train_pca.shape[1])]
    )
    train_df["time"] = y_time_train
    train_df["event"] = y_event_train

    test_df = pd.DataFrame(
        data=X_test_pca, columns=[f"feature_{i}" for i in range(X_test_pca.shape[1])]
    )
    test_df["time"] = y_time_test
    test_df["event"] = y_event_test

    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col="time", event_col="event")

    # Calculate C-index
    train_c_index = concordance_index(
        train_df["time"], -cph.predict_partial_hazard(train_df), train_df["event"]
    )

    test_c_index = concordance_index(
        test_df["time"], -cph.predict_partial_hazard(test_df), test_df["event"]
    )

    print(f"Cox PH model for {modality}:")
    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Test C-index: {test_c_index:.4f}")

    # Save the predictions to the data directory
    test_df["risk_score"] = cph.predict_partial_hazard(test_df).values
    test_predictions = test_df[["time", "event", "risk_score"]]
    test_predictions.to_csv(
        os.path.join(DIRS["data"], f"{modality}_cox_test_predictions.csv"), index=False
    )

    # Save the Cox model
    with open(os.path.join(DIRS["models"], f"{modality}_cox_model.pkl"), "wb") as f:
        pickle.dump(cph, f)

    # Save the scaler
    with open(os.path.join(DIRS["models"], f"{modality}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Plot stratified KM curves based on test predictions
    test_patients = pd.DataFrame(
        {survival_days_col: y_time_test, "event": y_event_test}
    )
    risk_scores_np = test_df["risk_score"].values
    plot_survival_curves(
        test_patients, survival_days_col, f"{modality}", risk_scores_np
    )

    return {"model": cph, "train_c_index": train_c_index, "test_c_index": test_c_index}


def run_random_survival_forest(modality, embeddings, patients_df, survival_days_col):
    """Run a Random Survival Forest model."""
    print(f"\nRunning Random Survival Forest for {modality} embeddings...")

    # Prepare the data
    X = embeddings

    # Flatten multi-dimensional embeddings if necessary
    original_shape = X.shape
    if len(original_shape) > 2:
        print(f"Flattening {modality} embeddings from {original_shape} to 2D...")
        X = X.reshape(original_shape[0], -1)
        print(f"Flattened shape: {X.shape}")

    y_time = patients_df[survival_days_col].values
    y_event = patients_df["event"].values

    # Create structured array for sksurv
    y = np.array(
        [(bool(e), t) for e, t in zip(y_event, y_time)],
        dtype=[("event", bool), ("time", float)],
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y_event
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply PCA if needed for high-dimensional data
    reduce_to = 50
    if X_train.shape[1] > reduce_to:
        pca = PCA(n_components=reduce_to)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"Reduced dimensions from {X.shape[1]} to {reduce_to} for RSF model")

        # Save PCA object for later use
        with open(os.path.join(DIRS["models"], f"{modality}_rsf_pca.pkl"), "wb") as f:
            pickle.dump(pca, f)

    # Train Random Survival Forest
    start_time = time()
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_features="log2",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    rsf.fit(X_train, y_train)
    training_time = time() - start_time

    # Make predictions
    test_risk_scores = rsf.predict(X_test)

    # Calculate C-index
    train_c_index = rsf.score(X_train, y_train)
    test_c_index = rsf.score(X_test, y_test)

    print(f"Random Survival Forest for {modality}:")
    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Test C-index: {test_c_index:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    # Save the model
    with open(os.path.join(DIRS["models"], f"{modality}_rsf_model.pkl"), "wb") as f:
        pickle.dump(rsf, f)

    # Save the scaler
    with open(os.path.join(DIRS["models"], f"{modality}_rsf_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save predictions
    test_predictions = pd.DataFrame(
        {
            "time": [t[1] for t in y_test],
            "event": [t[0] for t in y_test],
            "risk_score": test_risk_scores,
        }
    )
    test_predictions.to_csv(
        os.path.join(DIRS["data"], f"{modality}_rsf_test_predictions.csv"), index=False
    )

    # Plot stratified KM curves based on test predictions
    test_patients = pd.DataFrame(
        {survival_days_col: [t[1] for t in y_test], "event": [t[0] for t in y_test]}
    )
    plot_survival_curves(
        test_patients, survival_days_col, f"{modality}_rsf", test_risk_scores
    )

    return {
        "model": rsf,
        "train_c_index": train_c_index,
        "test_c_index": test_c_index,
        "training_time": training_time,
    }


def run_deepsurv_model(modality, embeddings, patients_df, survival_days_col):
    """Run a DeepSurv model for survival prediction."""
    print(f"\nRunning DeepSurv model for {modality} embeddings...")

    # Define the DeepSurv model
    class DeepSurv(nn.Module):
        def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
            super(DeepSurv, self).__init__()

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.Dropout(0.3))
                prev_dim = hidden_dim

            # Output layer (risk score)
            layers.append(nn.Linear(prev_dim, 1))

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Custom loss function for survival analysis
    def neg_log_likelihood_loss(risk_pred, y_time, y_event):
        """
        Compute negative log likelihood for Cox model

        Args:
            risk_pred: predicted risk scores (higher score means higher risk)
            y_time: survival times
            y_event: event indicators (1 if event occurred, 0 if censored)
        """
        # Sort data by survival time (descending)
        idx = torch.argsort(y_time, descending=True)
        risk_pred = risk_pred[idx]
        y_event = y_event[idx]

        # Calculate log partial likelihood
        log_risk = risk_pred
        cum_sums = torch.logcumsumexp(log_risk, dim=0)
        log_partial_likelihood = log_risk - cum_sums

        # Only consider events, not censored data
        neg_log_likelihood = -torch.sum(log_partial_likelihood * y_event)
        return neg_log_likelihood

    # Prepare the data
    X = embeddings

    # Flatten multi-dimensional embeddings if necessary
    original_shape = X.shape
    if len(original_shape) > 2:
        print(f"Flattening {modality} embeddings from {original_shape} to 2D...")
        X = X.reshape(original_shape[0], -1)
        print(f"Flattened shape: {X.shape}")

    y_time = patients_df[survival_days_col].values
    y_event = patients_df["event"].values

    # Split the data
    X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = (
        train_test_split(
            X,
            y_time,
            y_event,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y_event,
        )
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to torch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_time_train_tensor = torch.FloatTensor(y_time_train)
    y_time_test_tensor = torch.FloatTensor(y_time_test)
    y_event_train_tensor = torch.FloatTensor(y_event_train)
    y_event_test_tensor = torch.FloatTensor(y_event_test)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        X_train_tensor, y_time_train_tensor, y_event_train_tensor
    )
    test_dataset = TensorDataset(X_test_tensor, y_time_test_tensor, y_event_test_tensor)

    batch_size = min(64, len(X_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_dim = X_train.shape[1]
    model = DeepSurv(input_dim=input_dim)

    # Set up training parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 100
    patience = 10  # For early stopping

    # Training loop
    start_time = time()

    best_c_index = 0
    best_model = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_x, batch_time, batch_event in train_loader:
            optimizer.zero_grad()
            risk_pred = model(batch_x).squeeze()
            loss = neg_log_likelihood_loss(risk_pred, batch_time, batch_event)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            # Compute risk scores for all test data
            test_risk_pred = model(X_test_tensor).squeeze().numpy()

            # Calculate C-index
            current_c_index = concordance_index(
                y_time_test, -test_risk_pred, y_event_test
            )

            if current_c_index > best_c_index:
                best_c_index = current_c_index
                best_model = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}, C-index: {current_c_index:.4f}"
            )

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    # Final evaluation
    model.eval()
    training_time = time() - start_time

    with torch.no_grad():
        # Training set evaluation
        train_risk_pred = model(X_train_tensor).squeeze().numpy()
        train_c_index = concordance_index(y_time_train, -train_risk_pred, y_event_train)

        # Test set evaluation
        test_risk_pred = model(X_test_tensor).squeeze().numpy()
        test_c_index = concordance_index(y_time_test, -test_risk_pred, y_event_test)

    print(f"DeepSurv model for {modality}:")
    print(f"Train C-index: {train_c_index:.4f}")
    print(f"Test C-index: {test_c_index:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(DIRS["models"], f"{modality}_deepsurv_model.pt"),
    )

    # Save the scaler
    with open(
        os.path.join(DIRS["models"], f"{modality}_deepsurv_scaler.pkl"), "wb"
    ) as f:
        pickle.dump(scaler, f)

    # Save predictions
    test_predictions = pd.DataFrame(
        {"time": y_time_test, "event": y_event_test, "risk_score": test_risk_pred}
    )
    test_predictions.to_csv(
        os.path.join(DIRS["data"], f"{modality}_deepsurv_test_predictions.csv"),
        index=False,
    )

    # Plot stratified KM curves based on test predictions
    test_patients = pd.DataFrame(
        {survival_days_col: y_time_test, "event": y_event_test}
    )
    plot_survival_curves(
        test_patients, survival_days_col, f"{modality}_deepsurv", test_risk_pred
    )

    return {
        "model": model,
        "train_c_index": train_c_index,
        "test_c_index": test_c_index,
        "training_time": training_time,
    }


def run_cross_validation(
    modality, embeddings, patients_df, survival_days_col, n_folds=5
):
    """Run k-fold cross-validation for multiple survival models."""
    print(f"\nRunning {n_folds}-fold cross-validation for {modality} embeddings...")

    # Prepare the data
    X = embeddings

    # Flatten multi-dimensional embeddings if necessary
    original_shape = X.shape
    if len(original_shape) > 2:
        print(f"Flattening {modality} embeddings from {original_shape} to 2D...")
        X = X.reshape(original_shape[0], -1)
        print(f"Flattened shape: {X.shape}")

    y_time = patients_df[survival_days_col].values
    y_event = patients_df["event"].values

    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    # Initialize result storage
    cv_results = {
        "cox": {"c_index": [], "time": []},
        "rsf": {"c_index": [], "time": []},
    }

    # For tracking the best model
    best_model = {"model_type": None, "fold": None, "c_index": 0, "model": None}

    fold = 1
    for train_idx, test_idx in skf.split(X, y_event):
        print(f"\nFold {fold}/{n_folds}")

        # Split data for this fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_time_train, y_time_test = y_time[train_idx], y_time[test_idx]
        y_event_train, y_event_test = y_event[train_idx], y_event[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Reduce dimensions if needed
        reduce_to = 50
        if X_train_scaled.shape[1] > reduce_to:
            pca = PCA(n_components=reduce_to)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)
            print(f"Reduced dimensions from {X.shape[1]} to {reduce_to}")

        # 1. Cox PH Model
        start_time = time()

        train_features_df = pd.DataFrame(
            data=X_train_scaled,
            columns=[f"feature_{i}" for i in range(X_train_scaled.shape[1])],
        )
        train_features_df["time"] = y_time_train
        train_features_df["event"] = y_event_train

        test_features_df = pd.DataFrame(
            data=X_test_scaled,
            columns=[f"feature_{i}" for i in range(X_test_scaled.shape[1])],
        )
        test_features_df["time"] = y_time_test
        test_features_df["event"] = y_event_test

        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(train_features_df, duration_col="time", event_col="event")

        # Calculate C-index
        cox_c_index = concordance_index(
            test_features_df["time"],
            -cph.predict_partial_hazard(test_features_df),
            test_features_df["event"],
        )
        cox_time = time() - start_time

        cv_results["cox"]["c_index"].append(cox_c_index)
        cv_results["cox"]["time"].append(cox_time)

        if cox_c_index > best_model["c_index"]:
            best_model = {
                "model_type": "cox",
                "fold": fold,
                "c_index": cox_c_index,
                "model": cph,
                "scaler": scaler,
            }
            if X_train.shape[1] > reduce_to:
                best_model["pca"] = pca

        print(f"Cox PH - Fold {fold} C-index: {cox_c_index:.4f}, Time: {cox_time:.2f}s")

        # 2. Random Survival Forest
        start_time = time()

        # Create structured array for RSF
        y_train = np.array(
            [(bool(e), t) for e, t in zip(y_event_train, y_time_train)],
            dtype=[("event", bool), ("time", float)],
        )
        y_test = np.array(
            [(bool(e), t) for e, t in zip(y_event_test, y_time_test)],
            dtype=[("event", bool), ("time", float)],
        )

        # Train RSF model
        rsf = RandomSurvivalForest(
            n_estimators=100,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        rsf.fit(X_train_scaled, y_train)

        # Calculate C-index
        rsf_c_index = rsf.score(X_test_scaled, y_test)
        rsf_time = time() - start_time

        cv_results["rsf"]["c_index"].append(rsf_c_index)
        cv_results["rsf"]["time"].append(rsf_time)

        if rsf_c_index > best_model["c_index"]:
            best_model = {
                "model_type": "rsf",
                "fold": fold,
                "c_index": rsf_c_index,
                "model": rsf,
                "scaler": scaler,
            }
            if X_train.shape[1] > reduce_to:
                best_model["pca"] = pca

        print(f"RSF - Fold {fold} C-index: {rsf_c_index:.4f}, Time: {rsf_time:.2f}s")
        fold += 1

    # Calculate average results
    for model_type in cv_results:
        mean_c_index = np.mean(cv_results[model_type]["c_index"])
        std_c_index = np.std(cv_results[model_type]["c_index"])
        mean_time = np.mean(cv_results[model_type]["time"])

        print(f"\n{model_type.upper()} CV Results:")
        print(f"Mean C-index: {mean_c_index:.4f} ± {std_c_index:.4f}")
        print(f"Mean Time: {mean_time:.2f}s")

    # Save CV results
    cv_results_df = pd.DataFrame(
        {"Model": [], "Fold": [], "C-index": [], "Time (s)": []}
    )

    for model_type in cv_results:
        for fold_idx in range(n_folds):
            new_row = pd.DataFrame(
                {
                    "Model": [model_type],
                    "Fold": [fold_idx + 1],
                    "C-index": [cv_results[model_type]["c_index"][fold_idx]],
                    "Time (s)": [cv_results[model_type]["time"][fold_idx]],
                }
            )
            cv_results_df = pd.concat([cv_results_df, new_row], ignore_index=True)

    cv_results_df.to_csv(
        os.path.join(DIRS["data"], f"{modality}_cv_results.csv"), index=False
    )

    # Plot CV results
    fig = plt.figure(figsize=(12, 6))

    # Box plot for C-index across folds
    model_names = list(cv_results.keys())
    c_indices = [cv_results[model]["c_index"] for model in model_names]

    plt.boxplot(c_indices, labels=[m.upper() for m in model_names])
    plt.title(f"Cross-Validation Results - {modality.capitalize()}")
    plt.ylabel("C-index")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add individual points
    for i, c_index_list in enumerate(c_indices):
        x = np.random.normal(i + 1, 0.04, size=len(c_index_list))
        plt.plot(x, c_index_list, "o", alpha=0.6)

    plt.axhline(
        y=0.5, color="r", linestyle="--", alpha=0.3, label="Random (C-index=0.5)"
    )
    plt.legend()

    plt.tight_layout()

    # Save the figure
    save_figure(fig, modality, "cross_validation", "cv_comparison")

    # Save the best model
    print(
        f"\nBest model: {best_model['model_type'].upper()} from fold {best_model['fold']} with C-index {best_model['c_index']:.4f}"
    )

    if best_model["model_type"] == "cox":
        with open(
            os.path.join(DIRS["models"], f"{modality}_best_cox_model.pkl"), "wb"
        ) as f:
            pickle.dump(best_model["model"], f)
    elif best_model["model_type"] == "rsf":
        with open(
            os.path.join(DIRS["models"], f"{modality}_best_rsf_model.pkl"), "wb"
        ) as f:
            pickle.dump(best_model["model"], f)

    # Save the scaler
    with open(os.path.join(DIRS["models"], f"{modality}_best_scaler.pkl"), "wb") as f:
        pickle.dump(best_model["scaler"], f)

    # Save PCA if used
    if "pca" in best_model:
        with open(os.path.join(DIRS["models"], f"{modality}_best_pca.pkl"), "wb") as f:
            pickle.dump(best_model["pca"], f)

    return {"cv_results": cv_results, "best_model": best_model}


def compare_models(model_name, results_dict):
    """Compare the performance of different modalities for a specific model."""
    fig = plt.figure(figsize=(12, 6))

    modalities = list(results_dict.keys())
    test_c_indices = [results_dict[m]["test_c_index"] for m in modalities]

    # Sort by performance
    sorted_indices = np.argsort(test_c_indices)[::-1]  # descending
    sorted_modalities = [modalities[i] for i in sorted_indices]
    sorted_c_indices = [test_c_indices[i] for i in sorted_indices]

    # Create bar chart
    plt.bar(range(len(sorted_modalities)), sorted_c_indices, color="steelblue")

    plt.xlabel("Modality")
    plt.ylabel("Test C-index")
    plt.title(f"{model_name} Survival Analysis Performance by Modality")
    plt.xticks(
        range(len(sorted_modalities)), [m.capitalize() for m in sorted_modalities]
    )
    plt.ylim(0.4, 1.0)  # C-index ranges from 0.5 (random) to 1.0 (perfect)
    plt.axhline(
        y=0.5, color="r", linestyle="-", alpha=0.3, label="Random (C-index=0.5)"
    )
    plt.legend()

    # Add value labels on top of bars
    for i, v in enumerate(sorted_c_indices):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()

    # Save the figure using the utility function
    model_name_for_file = model_name.lower().replace(" ", "_")
    save_figure(fig, None, "comparison_plots", f"{model_name_for_file}_comparison")

    # Create comparison table and save to the data directory
    comparison_df = pd.DataFrame(
        {
            "Modality": [m.capitalize() for m in modalities],
            f"{model_name} C-index": test_c_indices,
        }
    )

    comparison_df.to_csv(
        os.path.join(DIRS["data"], f"{model_name_for_file}_comparison.csv"),
        index=False,
    )
    return comparison_df


def compare_modalities(results_dict):
    """Compare the performance of different modalities."""
    fig = plt.figure(figsize=(12, 6))

    modalities = list(results_dict.keys())
    test_c_indices = [results_dict[m]["test_c_index"] for m in modalities]

    # Sort by performance
    sorted_indices = np.argsort(test_c_indices)[::-1]  # descending
    sorted_modalities = [modalities[i] for i in sorted_indices]
    sorted_c_indices = [test_c_indices[i] for i in sorted_indices]

    # Create bar chart
    plt.bar(range(len(sorted_modalities)), sorted_c_indices, color="steelblue")

    plt.xlabel("Modality")
    plt.ylabel("Test C-index")
    plt.title("Cox PH Survival Analysis Performance by Modality")
    plt.xticks(
        range(len(sorted_modalities)), [m.capitalize() for m in sorted_modalities]
    )
    plt.ylim(0.4, 1.0)  # C-index ranges from 0.5 (random) to 1.0 (perfect)
    plt.axhline(
        y=0.5, color="r", linestyle="-", alpha=0.3, label="Random (C-index=0.5)"
    )
    plt.legend()

    # Add value labels on top of bars
    for i, v in enumerate(sorted_c_indices):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()

    # Save the figure
    save_figure(fig, None, "comparison_plots", "modality_comparison")

    # Create comparison table and save to the data directory
    comparison_df = pd.DataFrame(
        {
            "Modality": [m.capitalize() for m in modalities],
            "Cox PH C-index": test_c_indices,
        }
    )

    comparison_df.to_csv(
        os.path.join(DIRS["data"], "modality_comparison.csv"), index=False
    )
    return comparison_df


def main():
    """Main function to run the survival analysis pipeline."""
    # Create the directory structure first
    global DIRS
    DIRS = create_directory_structure()

    # Load the patient data and embeddings
    patients_df, embeddings_dict, survival_days_col = load_survival_data()

    # Store results for comparison
    results_dict = {}
    rsf_results_dict = {}
    deepsurv_results_dict = {}  # New dictionary for DeepSurv results
    cv_results_dict = {}

    # For each modality, run survival models
    for modality, embeddings in embeddings_dict.items():
        print(f"\n{'=' * 80}\nAnalyzing {modality} embeddings\n{'=' * 80}")

        # Ensure patients_df and embeddings are aligned
        assert len(patients_df) == len(embeddings), (
            f"Number of patients ({len(patients_df)}) doesn't match embeddings ({len(embeddings)})"
        )

        # Plot baseline survival curves for all patients
        plot_survival_curves(patients_df, survival_days_col, modality)

        # Run Cox model
        cox_results = run_cox_model(
            modality=modality,
            embeddings=embeddings,
            patients_df=patients_df,
            survival_days_col=survival_days_col,
        )

        # Store results
        results_dict[modality] = {
            "train_c_index": cox_results["train_c_index"],
            "test_c_index": cox_results["test_c_index"],
        }

        # Run Random Survival Forest
        rsf_results = run_random_survival_forest(
            modality=modality,
            embeddings=embeddings,
            patients_df=patients_df,
            survival_days_col=survival_days_col,
        )

        # Store RSF results
        rsf_results_dict[modality] = {
            "train_c_index": rsf_results["train_c_index"],
            "test_c_index": rsf_results["test_c_index"],
            "training_time": rsf_results["training_time"],
        }

        # Run DeepSurv model
        deepsurv_results = run_deepsurv_model(
            modality=modality,
            embeddings=embeddings,
            patients_df=patients_df,
            survival_days_col=survival_days_col,
        )

        # Store DeepSurv results
        deepsurv_results_dict[modality] = {
            "train_c_index": deepsurv_results["train_c_index"],
            "test_c_index": deepsurv_results["test_c_index"],
            "training_time": deepsurv_results["training_time"],
        }

        # Run cross-validation to get more robust performance estimates
        cv_results = run_cross_validation(
            modality=modality,
            embeddings=embeddings,
            patients_df=patients_df,
            survival_days_col=survival_days_col,
            n_folds=5,
        )

        cv_results_dict[modality] = cv_results

    print(f"\n{'=' * 80}\nSurvival Analysis Summary:\n{'=' * 80}")

    # Compare modalities using Cox model
    cox_comparison_df = compare_modalities(results_dict)
    print("\nCox PH Modality Comparison:")
    print(cox_comparison_df)

    # Compare modalities using RSF
    if rsf_results_dict:
        rsf_comparison_df = compare_models("Random Survival Forest", rsf_results_dict)
        print("\nRandom Survival Forest Modality Comparison:")
        print(rsf_comparison_df)

    # Compare modalities using DeepSurv
    if deepsurv_results_dict:
        deepsurv_comparison_df = compare_models("DeepSurv", deepsurv_results_dict)
        print("\nDeepSurv Modality Comparison:")
        print(deepsurv_comparison_df)

    # Save a summary of all results
    summary_df = pd.DataFrame(index=embeddings_dict.keys())

    # Add Cox results
    for modality in results_dict:
        summary_df.loc[modality, "Cox_Train_C-index"] = results_dict[modality][
            "train_c_index"
        ]
        summary_df.loc[modality, "Cox_Test_C-index"] = results_dict[modality][
            "test_c_index"
        ]

    # Add RSF results
    for modality in rsf_results_dict:
        summary_df.loc[modality, "RSF_Train_C-index"] = rsf_results_dict[modality][
            "train_c_index"
        ]
        summary_df.loc[modality, "RSF_Test_C-index"] = rsf_results_dict[modality][
            "test_c_index"
        ]
        summary_df.loc[modality, "RSF_Training_Time"] = rsf_results_dict[modality][
            "training_time"
        ]

    # Add DeepSurv results
    for modality in deepsurv_results_dict:
        summary_df.loc[modality, "DeepSurv_Train_C-index"] = deepsurv_results_dict[
            modality
        ]["train_c_index"]
        summary_df.loc[modality, "DeepSurv_Test_C-index"] = deepsurv_results_dict[
            modality
        ]["test_c_index"]
        summary_df.loc[modality, "DeepSurv_Training_Time"] = deepsurv_results_dict[
            modality
        ]["training_time"]

    # Format and save summary
    summary_df = summary_df.round(4)
    summary_df.index.name = "Modality"
    summary_df.to_csv(os.path.join(DIRS["data"], "survival_analysis_summary.csv"))

    # Create a long format summary with Modality, Model, C-index columns
    long_summary = []

    # Add Cox results
    for modality in results_dict:
        long_summary.append(
            {
                "Modality": modality,
                "Model": "Cox PH",
                "C-index": results_dict[modality]["test_c_index"],
            }
        )

    # Add RSF results
    for modality in rsf_results_dict:
        long_summary.append(
            {
                "Modality": modality,
                "Model": "RSF",
                "C-index": rsf_results_dict[modality]["test_c_index"],
            }
        )

    # Add DeepSurv results
    for modality in deepsurv_results_dict:
        long_summary.append(
            {
                "Modality": modality,
                "Model": "DeepSurv",
                "C-index": deepsurv_results_dict[modality]["test_c_index"],
            }
        )

    # Create and save the long format summary DataFrame
    long_summary_df = pd.DataFrame(long_summary)
    long_summary_df = long_summary_df.round(4)
    long_summary_df.to_csv(
        os.path.join(DIRS["data"], "survival_model_comparison.csv"), index=False
    )

    print("\nSurvival analysis completed. Results saved to:")
    print(f"- Figures: {DIRS['figures']}")
    print(f"- Models: {DIRS['models']}")
    print(f"- Data: {DIRS['data']}")


if __name__ == "__main__":
    main()
