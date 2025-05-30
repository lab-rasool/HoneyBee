import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)
from sklearn.manifold import TSNE
import os
import random
from tqdm.auto import tqdm
import warnings
from data import load_embeddings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define constants
N_RUNS = 10
TEST_SIZE = 0.2
N_ESTIMATORS = 100


# Function to run classification experiments
def run_classification_experiment(X, y, modality_name):
    """Run classification experiments with multiple random seeds."""
    # Check if X has more than 2 dimensions and flatten if necessary
    original_shape = X.shape
    if len(X.shape) > 2:
        print(f"Reshaping {modality_name} embeddings from {original_shape} to 2D")
        # Properly flatten all dimensions except the first (samples) regardless of dimensionality
        X = X.reshape(X.shape[0], -1)
        print(f"New shape: {X.shape}")

    # Handle NaN values which can cause issues with the classifier
    if np.isnan(X).any():
        print(f"Replacing NaN values in {modality_name} embeddings")
        X = np.nan_to_num(X)

    # Rest of the function remains the same
    accuracies = []
    random_seeds = np.random.randint(0, 10000, size=N_RUNS)

    # Initialize variables to store best model and predictions
    best_accuracy = 0
    best_y_test = None
    best_y_pred = None

    for seed in tqdm(random_seeds, desc=f"Running {modality_name} Classification"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed
        )
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # Store results of best run
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_y_test = y_test
            best_y_pred = y_pred

        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return {
        "modality": modality_name,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "best_y_test": best_y_test,
        "best_y_pred": best_y_pred,
    }


# Run classification experiments for all modalities
def run_all_classification_experiments():
    """Run classification experiments for all modalities."""
    embeddings = load_embeddings()
    results = {}
    confusion_matrices = {}

    for modality, data in embeddings.items():
        print(f"Running classification for {modality} embeddings...")
        result = run_classification_experiment(data["X"], data["y"], modality)
        results[modality] = result

        # Create confusion matrix for the best run
        if result["best_y_test"] is not None:
            cm = confusion_matrix(result["best_y_test"], result["best_y_pred"])
            confusion_matrices[modality] = {
                "cm": cm,
                "labels": np.unique(result["best_y_test"]),
            }

    return results, confusion_matrices


# Plot classification results
def plot_classification_results(results):
    """Create bar chart of classification accuracy for different modalities."""
    modalities = list(results.keys())
    accuracies = [results[mod]["mean_accuracy"] for mod in modalities]
    errors = [results[mod]["std_accuracy"] for mod in modalities]

    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modalities, accuracies, yerr=errors, capsize=10)

    # Add labels and formatting
    plt.ylabel("Classification Accuracy")
    plt.title("Cancer Type Classification Accuracy by Embedding Modality")
    plt.ylim(0, 1.0)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("classification_accuracy.png", dpi=300)
    plt.close()

    return "classification_accuracy.png"


# Plot confusion matrix
def plot_confusion_matrix(confusion_matrices, modality):
    """Plot confusion matrix for the best model of a specific modality."""
    if modality not in confusion_matrices:
        print(f"No confusion matrix available for {modality}")
        return None

    cm_data = confusion_matrices[modality]
    cm = cm_data["cm"]
    labels = cm_data["labels"]

    # Normalize confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {modality.capitalize()} Embeddings")
    plt.tight_layout()

    # Save plot
    filename = f"./figures/{modality}_confusion_matrix.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    return filename


# Create t-SNE visualization
def create_tsne_visualization(embeddings, modality):
    """Create t-SNE visualization for embeddings of a specific modality."""
    if modality not in embeddings:
        print(f"No embeddings available for {modality}")
        return None

    X = embeddings[modality]["X"]
    y = embeddings[modality]["y"]

    # Check if X has more than 2 dimensions and flatten if necessary
    original_shape = X.shape
    if len(X.shape) > 2:
        print(f"Reshaping {modality} embeddings from {original_shape} to 2D")
        # Properly flatten all dimensions except the first (samples)
        X = X.reshape(X.shape[0], -1)
        print(f"New shape: {X.shape}")

    # Handle NaN values which can cause issues with t-SNE
    if np.isnan(X).any():
        print(f"Replacing NaN values in {modality} embeddings")
        X = np.nan_to_num(X)

    # Sample data if too large (t-SNE is computationally intensive)
    if len(X) > 5000:
        indices = np.random.choice(len(X), 5000, replace=False)
        X = X[indices]
        y = y[indices]

    # Run t-SNE
    print(f"Running t-SNE for {modality} embeddings...")

    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X)

    # Create plot
    plt.figure(figsize=(10, 8))

    # Get unique cancer types
    unique_cancers = np.unique(y)

    # Plot each cancer type with a different color
    for cancer_type in unique_cancers:
        indices = y == cancer_type
        plt.scatter(
            X_reduced[indices, 0],
            X_reduced[indices, 1],
            label=cancer_type,
            alpha=0.7,
            s=30,
        )

    plt.title(f"t-SNE Visualization of {modality.capitalize()} Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Add legend only if there are fewer than 20 cancer types
    if len(unique_cancers) < 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save plot
    filename = f"./figures/{modality}_tsne.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    return filename, X_reduced, y


# Main execution function
def main():
    """Main execution function."""
    # Create output directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # Load embeddings
    embeddings = load_embeddings()

    # Run classification experiments
    results, confusion_matrices = run_all_classification_experiments()

    # Plot classification results
    classification_plot = plot_classification_results(results)
    print(f"Classification results plot saved to {classification_plot}")

    # Plot confusion matrix for all modalities
    for modality in results.keys():
        confusion_plot = plot_confusion_matrix(confusion_matrices, modality)
        if confusion_plot:
            print(f"Confusion matrix plot saved to {confusion_plot}")

    # Create t-SNE visualizations
    for modality in embeddings.keys():
        tsne_plot, _, _ = create_tsne_visualization(embeddings, modality)
        if tsne_plot:
            print(f"t-SNE visualization saved to {tsne_plot}")

    print("Analysis complete!")


if __name__ == "__main__":
    main()
