import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
import faiss
import os
import random
import json
from tqdm.auto import tqdm
from matplotlib.gridspec import GridSpec
import matplotlib.colors
import warnings
import torch
from data import load_embeddings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define these constants at the top level of the file for consistency
CUSTOM_COLORS = ["#FF6060", "#6CC2F5", "#FF9A60", "#FF60B3", "#C260FF", "#60FFA0"]
# square, circle, star, triangle up, pentagon, triangle down
CUSTOM_MARKERS = ["s", "o", "*", "^", "p", "v"]

# Define constants
N_RUNS = 10
TEST_SIZE = 0.2
N_ESTIMATORS = 100
NUM_TRIALS = 100
K_VALUES = [1, 5, 10, 20, 50]

# Map modalities to consistent colors
modality_colors = {
    "clinical": CUSTOM_COLORS[0],
    "molecular": CUSTOM_COLORS[1],
    "pathology": CUSTOM_COLORS[2],
    "radiology": CUSTOM_COLORS[3],
    "multimodal": CUSTOM_COLORS[4],
}


# ============== ANALYSIS FUNCTIONS ==============
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

    accuracies = []
    random_seeds = np.random.randint(0, 10000, size=N_RUNS)

    # Initialize variables to store best model and predictions
    best_accuracy = 0
    best_y_test = None
    best_y_pred = None

    for seed in tqdm(random_seeds, desc=f"Running {modality_name} Classification"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
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


def run_all_classification_experiments(embeddings):
    """Run classification experiments for all modalities."""
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
                "cm": cm.tolist(),  # Convert to list for JSON serialization
                "labels": np.unique(result["best_y_test"]).tolist(),
            }

    return results, confusion_matrices


def run_retrieval_benchmark(X_embeddings, y_labels, k_max=50):
    """Run retrieval benchmark for given embeddings."""
    # Check if X has more than 2 dimensions and flatten if necessary
    original_shape = X_embeddings.shape
    if len(X_embeddings.shape) > 2:
        print(f"Reshaping embeddings from {original_shape} to 2D")
        # Properly flatten all dimensions except the first (samples)
        X_embeddings = X_embeddings.reshape(X_embeddings.shape[0], -1)
        print(f"New shape: {X_embeddings.shape}")

    # Handle NaN values which can cause issues
    if np.isnan(X_embeddings).any():
        print("Replacing NaN values in embeddings")
        X_embeddings = np.nan_to_num(X_embeddings)

    # Build the FAISS index
    dimension = X_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    index.add(X_embeddings.astype(np.float32))  # FAISS requires float32

    # Prepare results container for each k value
    results = {k: [] for k in range(1, k_max + 1)}

    # Run trials
    for _ in tqdm(range(NUM_TRIALS), desc="Running retrieval trials"):
        # Pick a random embedding for evaluation
        random_index = random.randint(0, len(X_embeddings) - 1)
        query_embedding = X_embeddings[random_index]
        query_project_id = y_labels[random_index]

        # Retrieve the k closest embeddings
        distances, indices = index.search(
            query_embedding.reshape(1, -1).astype(np.float32), k=k_max + 1
        )  # k+1 to include the query itself

        # Remove the first result (the query itself)
        indices = indices[0][1:]

        # Calculate precision@k for each k value
        for k in range(1, k_max + 1):
            k_indices = indices[:k]
            matching_count = sum(
                y_labels[int(idx)] == query_project_id for idx in k_indices
            )
            precision_at_k = matching_count / k
            results[k].append(precision_at_k)

    # Calculate average precision@k
    avg_results = {k: np.mean(v) for k, v in results.items()}
    std_results = {k: np.std(v) for k, v in results.items()}

    return avg_results, std_results


def run_all_retrieval_experiments(embeddings):
    """Run retrieval experiments for all modalities."""
    results = {}

    for modality, data in embeddings.items():
        print(f"Running retrieval benchmark for {modality} embeddings...")
        avg_results, std_results = run_retrieval_benchmark(
            data["X"], data["y"], k_max=max(K_VALUES)
        )

        results[modality] = {"avg": avg_results, "std": std_results}

    return results


def create_tsne_visualization(X, y, title):
    """Create t-SNE visualization for embeddings."""
    # Check if X has more than 2 dimensions and flatten if necessary
    original_shape = X.shape
    if len(X.shape) > 2:
        print(f"Reshaping {title} embeddings from {original_shape} to 2D")
        # Properly flatten all dimensions except the first (samples)
        X = X.reshape(X.shape[0], -1)
        print(f"New shape: {X.shape}")

    # Handle NaN values which can cause issues with t-SNE
    if np.isnan(X).any():
        print(f"Replacing NaN values in {title} embeddings")
        X = np.nan_to_num(X)

    # Sample data if too large (t-SNE is computationally intensive)
    if len(X) > 5000:
        indices = np.random.choice(len(X), 5000, replace=False)
        X = X[indices]
        y = y[indices]

    # Run t-SNE
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X)

    return X_reduced, y


def export_plot_with_layers(
    X_reduced, y_labels, filename, unique_labels, include_legend=False
):
    """Export plot with separate layers for each label - easier to edit in vector editors"""
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(10, 8))

    # Get all possible color-marker combinations
    combinations = []
    for color in CUSTOM_COLORS:
        for marker in CUSTOM_MARKERS:
            combinations.append((color, marker))

    # Shuffle to get varied combinations
    random.seed(42)  # For reproducibility
    random.shuffle(combinations)

    # Plot each label with a unique combination
    for i, label in enumerate(unique_labels):
        if i < len(combinations):
            color, marker = combinations[i]
        else:
            # Fallback if we have more labels than combinations
            combo_idx = i % len(combinations)
            color, marker = combinations[combo_idx]
            print(f"Warning: Reusing combination for label {label}")

        base_color = np.array(matplotlib.colors.to_rgb(color))
        edge_color = base_color * 0.7

        indices = np.where(y_labels == label)[0]
        plt.scatter(
            X_reduced[indices, 0],
            X_reduced[indices, 1],
            c=[color],
            label=str(label),
            s=50,
            marker=marker,
            edgecolors=[edge_color],
            linewidths=0.5,
            alpha=0.9,
        )

    plt.gca().set_axis_off()
    plt.xlim(-125, 125)
    plt.ylim(-125, 125)
    plt.tight_layout()
    fig.savefig(f"{filename}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.png", format="png", dpi=300, bbox_inches="tight")

    return fig


def save_legend_separately(unique_labels, filename="legend"):
    """Save the legend as a separate file"""
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(figsize=(5, 8))

    # Generate the same color-marker combinations as in the main plot function
    combinations = []
    for color in CUSTOM_COLORS:
        for marker in CUSTOM_MARKERS:
            combinations.append((color, marker))

    # Use same seed for reproducibility
    random.seed(42)
    random.shuffle(combinations)

    # Create dummy scatter points using the same assignment logic
    for i, label in enumerate(unique_labels):
        if i < len(combinations):
            color, marker = combinations[i]
        else:
            # Fallback if we have more labels than combinations
            combo_idx = i % len(combinations)
            color, marker = combinations[combo_idx]

        base_color = np.array(matplotlib.colors.to_rgb(color))
        edge_color = base_color * 0.7

        ax.scatter(
            [0],
            [0],  # Position doesn't matter
            c=[color],
            label=str(label),
            s=50,
            marker=marker,
            edgecolors=[edge_color],
            linewidths=0.5,
            alpha=0.9,
        )

    # Hide the actual plot
    ax.set_axis_off()

    # Create the legend
    legend = ax.legend(
        title="Cancer Type",
        bbox_to_anchor=(0.5, 0.5),
        loc="center",
        frameon=True,
        title_fontsize=14,
    )

    # Save just the legend
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"{filename}.svg", format="svg", bbox_inches=legend_bbox)
    fig.savefig(f"{filename}.pdf", format="pdf", bbox_inches=legend_bbox)
    fig.savefig(f"{filename}.png", format="png", dpi=300, bbox_inches=legend_bbox)

    return fig


# ============== FIGURE GENERATION FUNCTIONS ==============
def create_figure_2(
    classification_results, confusion_data, retrieval_results, embeddings
):
    """
    Create Figure 2: Embedding Performance on Downstream Tasks
    Panel A: Bar chart showing classification accuracy
    Panel B: Confusion matrix for the best-performing model
    Panel C: Retrieval heatmap showing precision@k across modalities
    Panel D: t-SNE visualization of clinical embeddings
    """
    os.makedirs("figures", exist_ok=True)

    # Set global font properties - increase all font sizes
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 14  # Base font size
    plt.rcParams["axes.titlesize"] = 18  # Panel titles
    plt.rcParams["axes.labelsize"] = 16  # Axis labels
    plt.rcParams["xtick.labelsize"] = 14  # X-axis tick labels
    plt.rcParams["ytick.labelsize"] = 14  # Y-axis tick labels
    plt.rcParams["legend.fontsize"] = 14  # Legend text size

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)

    # Panel A: Classification accuracy bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    modalities = list(classification_results.keys())
    accuracies = [classification_results[mod]["mean_accuracy"] for mod in modalities]
    errors = [classification_results[mod]["std_accuracy"] for mod in modalities]

    bars = ax1.bar(
        modalities,
        accuracies,
        yerr=errors,
        capsize=10,
        color=[modality_colors.get(m, CUSTOM_COLORS[0]) for m in modalities],
    )
    ax1.set_ylabel("Classification Accuracy", fontsize=16)
    ax1.set_title("A. Cancer Type Classification Accuracy", fontsize=18)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis="both", labelsize=14)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    # Panel B: Confusion matrix for the best-performing model
    ax2 = fig.add_subplot(gs[0, 1])
    best_modality = max(
        classification_results, key=lambda x: classification_results[x]["mean_accuracy"]
    )
    cm_data = confusion_data[best_modality]
    cm = np.array(cm_data["cm"])
    labels = np.array(cm_data["labels"])

    # Remove "TCGA-" prefix from labels for cleaner visualization
    display_labels = np.array(
        [
            label.replace("TCGA-", "")
            if isinstance(label, str) and label.startswith("TCGA-")
            else label
            for label in labels
        ]
    )

    # Normalize confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax2,
        cbar=False,
        annot_kws={"size": 12},  # Increase annotation text size
    )
    ax2.set_xlabel("Predicted Label", fontsize=16)
    ax2.set_ylabel("True Label", fontsize=16)
    ax2.set_title(
        f"B. Confusion Matrix for {best_modality.capitalize()} Embeddings", fontsize=18
    )
    ax2.tick_params(axis="both", labelsize=14)

    # Panel C: Retrieval precision@k heatmap
    ax3 = fig.add_subplot(gs[1, 0])

    # Prepare data for heatmap
    modalities = list(retrieval_results.keys())
    data_matrix = np.zeros((len(modalities), len(K_VALUES)))

    # Fill the data matrix
    for i, modality in enumerate(modalities):
        avg_results = retrieval_results[modality]["avg"]
        for j, k in enumerate(K_VALUES):
            k_str = str(k) if isinstance(next(iter(avg_results.keys())), str) else k
            data_matrix[i, j] = avg_results[k_str]

    # Create heatmap within the subplot
    sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=K_VALUES,
        yticklabels=[m.capitalize() for m in modalities],
        ax=ax3,
        annot_kws={"size": 12},  # Increase annotation text size
    )

    ax3.set_xlabel("k value", fontsize=16)
    ax3.set_ylabel("Modality", fontsize=16)
    ax3.set_title("C. Precision@k Across Modalities", fontsize=18)
    ax3.tick_params(axis="both", labelsize=14)

    # Panel D: t-SNE visualization of clinical embeddings
    ax4 = fig.add_subplot(gs[1, 1])

    if "clinical" in embeddings:
        # Run t-SNE for clinical embeddings
        clinical_X = embeddings["clinical"]["X"]
        clinical_y = embeddings["clinical"]["y"]
        X_reduced, y_reduced = create_tsne_visualization(
            clinical_X, clinical_y, "Clinical"
        )

        # Get unique cancer types
        unique_cancers = np.unique(y_reduced)

        # Plot using color-marker combinations
        combinations = []
        for color in CUSTOM_COLORS:
            for marker in CUSTOM_MARKERS:
                combinations.append((color, marker))

        random.seed(42)
        random.shuffle(combinations)

        for i, cancer_type in enumerate(unique_cancers):
            if i < len(combinations):
                color, marker = combinations[i]
            else:
                combo_idx = i % len(combinations)
                color, marker = combinations[combo_idx]

            base_color = np.array(matplotlib.colors.to_rgb(color))
            edge_color = base_color * 0.7

            indices = y_reduced == cancer_type
            ax4.scatter(
                X_reduced[indices, 0],
                X_reduced[indices, 1],
                label=str(cancer_type),
                color=color,
                marker=marker,
                s=50,
                alpha=0.9,
                edgecolors=edge_color,
                linewidths=0.5,
            )

        ax4.set_title("D. t-SNE Visualization of Clinical Embeddings", fontsize=18)
        ax4.set_xlabel("t-SNE 1", fontsize=16)
        ax4.set_ylabel("t-SNE 2", fontsize=16)
        ax4.set_axis_off()

    plt.tight_layout()

    # Save figure
    plt.savefig(
        "figures/figure2_embedding_performance.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig("figures/figure2_embedding_performance.pdf", bbox_inches="tight")
    plt.savefig("figures/figure2_embedding_performance.svg", bbox_inches="tight")
    plt.close()

    return "figures/figure2_embedding_performance.png"


def extract_precision_at_k(results, k=10):
    """
    Extract precision@k values from retrieval results, handling different key formats.
    """
    precision_at_k = {}

    for modality, result in results.items():
        avg_results = result.get("avg", {})
        # Try different potential key formats
        value = None

        # Check integer key
        if k in avg_results:
            value = avg_results[k]
        # Check string key
        elif str(k) in avg_results:
            value = avg_results[str(k)]
        # Check float key
        elif float(k) in avg_results:
            value = avg_results[float(k)]
        # Check string float key
        elif f"{float(k)}" in avg_results:
            value = avg_results[f"{float(k)}"]

        precision_at_k[modality] = value

    return precision_at_k


def plot_classification_results(classification_results):
    """Create bar chart of classification accuracy for different modalities."""
    plt.rcParams["font.family"] = "Arial"
    modalities = list(classification_results.keys())
    accuracies = [classification_results[mod]["mean_accuracy"] for mod in modalities]
    errors = [classification_results[mod]["std_accuracy"] for mod in modalities]

    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        modalities,
        accuracies,
        yerr=errors,
        capsize=10,
        color=[modality_colors.get(m, CUSTOM_COLORS[0]) for m in modalities],
    )

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
    plt.savefig("figures/classification_accuracy.png", dpi=300)
    plt.savefig("figures/classification_accuracy.svg", format="svg")
    plt.savefig("figures/classification_accuracy.pdf", format="pdf")
    plt.close()

    return "figures/classification_accuracy.png"


def plot_confusion_matrix(confusion_matrices, modality):
    """Plot confusion matrix for a specific modality."""
    plt.rcParams["font.family"] = "Arial"
    if modality not in confusion_matrices:
        print(f"No confusion matrix available for {modality}")
        return None

    cm_data = confusion_matrices[modality]
    cm = np.array(cm_data["cm"])
    labels = np.array(cm_data["labels"])

    # Remove "TCGA-" prefix from labels for cleaner visualization
    display_labels = np.array(
        [
            label.replace("TCGA-", "")
            if isinstance(label, str) and label.startswith("TCGA-")
            else label
            for label in labels
        ]
    )

    # Normalize confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=False,  # Add this line to remove the color bar
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {modality.capitalize()} Embeddings")
    plt.tight_layout()

    # Save plot
    filename = f"figures/{modality}_confusion_matrix"
    plt.savefig(f"{filename}.png", dpi=300)
    plt.savefig(f"{filename}.svg", format="svg")
    plt.savefig(f"{filename}.pdf", format="pdf")
    plt.close()

    return f"{filename}.png"


def plot_retrieval_results(retrieval_results, k_values=K_VALUES):
    """Create precision@k curves for different modalities."""
    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(10, 6))

    for i, (modality, data) in enumerate(retrieval_results.items()):
        avg_results = data["avg"]
        std_results = data["std"]

        precisions = [
            avg_results[
                str(k) if isinstance(next(iter(avg_results.keys())), str) else k
            ]
            for k in k_values
        ]
        errors = [
            std_results[
                str(k) if isinstance(next(iter(std_results.keys())), str) else k
            ]
            for k in k_values
        ]

        color = modality_colors.get(modality, CUSTOM_COLORS[i % len(CUSTOM_COLORS)])
        marker = CUSTOM_MARKERS[i % len(CUSTOM_MARKERS)]

        plt.plot(
            k_values,
            precisions,
            marker=marker,
            label=modality.capitalize(),
            color=color,
            linewidth=2,
        )

        # Add error bands
        plt.fill_between(
            k_values,
            [p - e for p, e in zip(precisions, errors)],
            [p + e for p, e in zip(precisions, errors)],
            alpha=0.2,
            color=color,
        )

    plt.xlabel("k")
    plt.ylabel("Precision@k")
    plt.title("Retrieval Performance by Embedding Modality")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Mark specific k values
    for k in [5, 10, 20]:
        if k in k_values:
            plt.axvline(x=k, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save plot
    filename = "figures/retrieval_precision_at_k"
    plt.savefig(f"{filename}.png", dpi=300)
    plt.savefig(f"{filename}.svg", format="svg")
    plt.savefig(f"{filename}.pdf", format="pdf")
    plt.close()

    return f"{filename}.png"


def plot_precision_at_k_bar(retrieval_results, k=10):
    """Create bar chart of precision@k for different modalities."""
    plt.rcParams["font.family"] = "Arial"

    # Extract precision@k values for each modality
    precision_dict = extract_precision_at_k(retrieval_results, k)

    modalities = list(precision_dict.keys())
    precisions = [precision_dict[mod] for mod in modalities]

    # Get standard deviations
    errors = []
    for modality in modalities:
        std_results = retrieval_results[modality]["std"]
        # Handle different key formats
        if k in std_results:
            errors.append(std_results[k])
        elif str(k) in std_results:
            errors.append(std_results[str(k)])
        else:
            errors.append(0.0)  # Default if no std dev is found

    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        modalities,
        precisions,
        yerr=errors,
        capsize=10,
        color=[modality_colors.get(m, CUSTOM_COLORS[0]) for m in modalities],
    )

    # Add labels and formatting
    plt.ylabel(f"Precision@{k}")
    plt.title(f"Retrieval Precision@{k} by Embedding Modality")
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

    # Save plot
    filename = f"figures/precision_at_{k}_bar"
    plt.savefig(f"{filename}.png", dpi=300)
    plt.savefig(f"{filename}.svg", format="svg")
    plt.savefig(f"{filename}.pdf", format="pdf")
    plt.close()

    return f"{filename}.png"


def create_tsne_plots(embeddings):
    """Create t-SNE visualizations for all modalities."""
    filenames = []

    for modality, data in embeddings.items():
        print(f"Creating t-SNE visualization for {modality} embeddings...")

        # Run t-SNE
        X_reduced, y_reduced = create_tsne_visualization(data["X"], data["y"], modality)

        # Get unique cancer types
        unique_cancers = np.unique(y_reduced)

        # Use export_plot_with_layers function for consistent style
        export_plot_with_layers(
            X_reduced=X_reduced,
            y_labels=y_reduced,
            filename=f"figures/{modality}_tsne",
            unique_labels=unique_cancers,
        )

        # Also save legend separately
        save_legend_separately(
            unique_labels=unique_cancers, filename=f"figures/{modality}_legend"
        )

        filenames.append(f"figures/{modality}_tsne.png")

    return filenames


def plot_retrieval_heatmap(retrieval_results, k_values=K_VALUES):
    """Create heatmap visualization of precision@k for all modalities."""
    plt.rcParams["font.family"] = "Arial"

    # Prepare data for heatmap
    modalities = list(retrieval_results.keys())
    data_matrix = np.zeros((len(modalities), len(k_values)))

    # Fill the data matrix
    for i, modality in enumerate(modalities):
        avg_results = retrieval_results[modality]["avg"]
        for j, k in enumerate(k_values):
            k_str = str(k) if isinstance(next(iter(avg_results.keys())), str) else k
            data_matrix[i, j] = avg_results[k_str]

    # Create heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=k_values,
        yticklabels=[m.capitalize() for m in modalities],
    )

    plt.xlabel("k value")
    plt.ylabel("Modality")
    plt.title("Precision@k Across Modalities")
    plt.tight_layout()

    # Save the figure
    filename = "figures/retrieval_heatmap"
    plt.savefig(f"{filename}.png", dpi=300)
    plt.savefig(f"{filename}.svg", format="svg")
    plt.savefig(f"{filename}.pdf", format="pdf")
    plt.close()

    return f"{filename}.png"


def create_additional_figures(
    classification_results, confusion_matrices, retrieval_results, embeddings
):
    """Create all additional figures from cancer-classification.py and similarity-retrieval.py."""
    print("\n=== Generating Additional Figures ===")

    # Create output directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # 1. Classification accuracy bar chart
    classification_plot = plot_classification_results(classification_results)
    print(f"Classification results plot saved to {classification_plot}")

    # 2. Confusion matrices for each modality
    for modality in classification_results.keys():
        confusion_plot = plot_confusion_matrix(confusion_matrices, modality)
        if confusion_plot:
            print(f"Confusion matrix plot saved to {confusion_plot}")

    # 3. Retrieval precision@k curves
    retrieval_plot = plot_retrieval_results(retrieval_results)
    print(f"Retrieval precision@k plot saved to {retrieval_plot}")

    # 4. Precision@k bar charts for k=5, 10, 20
    for k in [5, 10, 20]:
        precision_bar_plot = plot_precision_at_k_bar(retrieval_results, k)
        print(f"Precision@{k} bar chart saved to {precision_bar_plot}")

    # 5. t-SNE visualizations for all modalities
    tsne_plots = create_tsne_plots(embeddings)
    for plot in tsne_plots:
        print(f"t-SNE visualization saved to {plot}")

    # New retrieval visualizations
    heatmap_plot = plot_retrieval_heatmap(retrieval_results)
    print(f"Retrieval heatmap saved to {heatmap_plot}")

    return True


# ============== MAIN EXECUTION FUNCTION ==============
def main():
    """Run analysis, or if results exist, just generate the figure and paper text."""
    # Create output directories if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Check if results already exist
    results_exist = all(
        [
            os.path.exists("results/classification_results.json"),
            os.path.exists("results/confusion_matrices.json"),
            os.path.exists("results/retrieval_results.json"),
        ]
    )

    if results_exist:
        print("Results files already exist. Skipping analysis and generating plots...")
        # Load classification results
        with open("results/classification_results.json", "r") as f:
            classification_results = json.load(f)

        # Load confusion matrices
        with open("results/confusion_matrices.json", "r") as f:
            confusion_matrices = json.load(f)

        # Load retrieval results
        with open("results/retrieval_results.json", "r") as f:
            retrieval_results = json.load(f)

        # Load embeddings (needed for t-SNE visualization)
        embeddings = load_embeddings()
    else:
        # Load embeddings
        print("Loading embeddings...")
        embeddings = load_embeddings()

        # Run classification experiments
        print("\n=== Running Classification Experiments ===")
        classification_results, confusion_matrices = run_all_classification_experiments(
            embeddings
        )

        # Save classification results
        with open("results/classification_results.json", "w") as f:
            # Convert numpy values to Python native types for JSON serialization
            serializable_results = {}
            for modality, result in classification_results.items():
                serializable_results[modality] = {
                    "modality": result["modality"],
                    "mean_accuracy": float(result["mean_accuracy"]),
                    "std_accuracy": float(result["std_accuracy"]),
                }
            json.dump(serializable_results, f, indent=2)

        # Save confusion matrices
        with open("results/confusion_matrices.json", "w") as f:
            json.dump(confusion_matrices, f, indent=2)

        # Run retrieval experiments
        print("\n=== Running Retrieval Experiments ===")
        retrieval_results = run_all_retrieval_experiments(embeddings)

        # Save retrieval results
        with open("results/retrieval_results.json", "w") as f:
            # Convert numeric keys to strings for JSON
            serializable_results = {}
            for modality, result in retrieval_results.items():
                serializable_results[modality] = {
                    "avg": {str(k): float(v) for k, v in result["avg"].items()},
                    "std": {str(k): float(v) for k, v in result["std"].items()},
                }
            json.dump(serializable_results, f, indent=2)

    # Generate main figure
    print("\n=== Generating Figure 2 ===")
    figure_path = create_figure_2(
        classification_results, confusion_matrices, retrieval_results, embeddings
    )
    print(f"Figure saved to: {figure_path}")

    # Generate all additional figures from both cancer-classification.py and similarity-retrieval.py
    create_additional_figures(
        classification_results, confusion_matrices, retrieval_results, embeddings
    )

    # Generate text for paper with actual values
    print("\n=== Generating Updated Text for Paper ===")
    print("Classification Results:")
    for modality, result in classification_results.items():
        print(
            f"  {modality}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}"
        )
    print("\nConfusion Matrices:")
    for modality, cm_data in confusion_matrices.items():
        cm = np.array(cm_data["cm"])
        labels = np.array(cm_data["labels"])
        print(f"  {modality}:")
        print("    Labels:", labels)
        print("    Confusion Matrix:\n", cm)
    print("\nRetrieval Results:")
    for modality, result in retrieval_results.items():
        avg_results = result["avg"]
        std_results = result["std"]
        print(f"  {modality}:")
        for k in K_VALUES:
            k_str = str(k)
            if k_str in avg_results:
                avg_value = avg_results[k_str]
                std_value = std_results[k_str]
                print(f"    Precision@{k}: {avg_value:.4f} ± {std_value:.4f}")


if __name__ == "__main__":
    main()
