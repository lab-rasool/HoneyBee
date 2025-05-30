from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModel,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors
import seaborn as sns
from tqdm.auto import tqdm
import os
import faiss
import random
import json
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define constants
BASE_DIR = "./clinical_benchmarking_results/"
CUSTOM_COLORS = ["#FF6060", "#6CC2F5", "#FF9A60", "#FF60B3", "#C260FF", "#60FFA0"]
CUSTOM_MARKERS = ["s", "o", "*", "^", "p", "v"]
N_RUNS = 10
NUM_TRIALS = 100
K_VALUES = [1, 5, 10, 20, 50]

# Create directories for organization
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

models_dir = os.path.join(BASE_DIR, "models")
data_dir = os.path.join(BASE_DIR, "data")
plots_dir = os.path.join(BASE_DIR, "plots")

for directory in [models_dir, data_dir, plots_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


def compute_metrics(eval_pred):
    """Compute evaluation metrics for model training."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def prepare_data_and_tokenize(data, tokenizer):
    """Prepare data for model training and tokenize text."""
    # Ensure 'project_id' is used as a label and it is in an appropriate format
    unique_labels = data["project_id"].unique().tolist()
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    data["labels"] = data["project_id"].map(label_dict)

    # Convert DataFrame to a Dataset object
    dataset = Dataset.from_pandas(data[["text", "labels"]])

    # Tokenization function to prepare input data
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )
        model_inputs["labels"] = examples["labels"]
        return model_inputs

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Split the dataset
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
    return tokenized_datasets["train"], tokenized_datasets["test"]


def train_and_save_model(pretrained_model, data, output_model_dir):
    """Fine-tune and save a model on the clinical data."""
    # Set the output model directory
    MODEL_SAVE_FOLDER_NAME = output_model_dir

    # Unique labels
    unique_labels = data["project_id"].unique().tolist()

    # LoRA configuration
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load quantized model first
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=len(unique_labels),
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    # Apply the LoRA adapter
    lora_model = get_peft_model(model, config)

    # Prepare and tokenize data
    train_dataset, test_dataset = prepare_data_and_tokenize(data, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_FOLDER_NAME + "/results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    train_result = trainer.train()

    # Print training results
    print("Training completed. Here are the stats:")
    print(train_result)

    # Evaluate the model
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    print(classification_report(test_dataset["labels"], y_pred))

    # Save the model
    trainer.model.save_pretrained(MODEL_SAVE_FOLDER_NAME)
    trainer.save_model(MODEL_SAVE_FOLDER_NAME)
    trainer.model.config.save_pretrained(MODEL_SAVE_FOLDER_NAME)

    return trainer, train_dataset, test_dataset


def generate_or_load_pt_embeddings(pre_trained_model, data, output_dir):
    """Generate embeddings using a pre-trained model or load from disk if available."""
    if not os.path.exists(output_dir):
        print(f"Generating pre-trained embeddings for {pre_trained_model}...")
        dataset = Dataset.from_pandas(data)
        model = AutoModel.from_pretrained(pre_trained_model)
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
        model.eval()
        model.to(device)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=512
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(tokenized_dataset, batch_size=16)
        embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc=f"Generating Embeddings for {pre_trained_model}"
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)

        data["embeddings"] = list(map(list, embeddings))
        data.to_parquet(output_dir)

    pre_trained_data = pd.read_parquet(output_dir)
    return pre_trained_data


def generate_or_load_ft_embeddings(
    pre_trained_model, fine_tuned_model, data, output_dir
):
    """Generate embeddings using a fine-tuned model or load from disk if available."""
    if not os.path.exists(output_dir):
        print(f"Generating fine-tuned embeddings for {pre_trained_model}...")
        dataset = Dataset.from_pandas(data)

        # Load the fine-tuned model
        model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model)

        # Extract the base model (without classification head) for embeddings
        base_model = model.base_model if hasattr(model, "base_model") else model

        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
        base_model.eval()
        base_model.to(device)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=512
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(tokenized_dataset, batch_size=16)
        embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc=f"Generating Embeddings for {fine_tuned_model}"
            ):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # Use the base model to get embeddings, not the classification head
                outputs = base_model(input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)

        data["embeddings"] = list(map(list, embeddings))
        data.to_parquet(output_dir)

    fine_tuned_data = pd.read_parquet(output_dir)
    return fine_tuned_data


def run_classification_experiments(n_runs, X, y, study):
    """Run classification experiments with RandomForest and report performance."""
    accuracies = []
    random_seeds = np.random.randint(0, 10000, size=n_runs)

    # Initialize variables to store best model and predictions
    best_accuracy = 0
    best_y_test = None
    best_y_pred = None

    for seed in tqdm(random_seeds, desc=f"Running {study} Classification"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
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

    print(f"{study} Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

    return {
        "model": study,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "best_y_test": best_y_test,
        "best_y_pred": best_y_pred,
    }


def run_retrieval_benchmark(X_embeddings, y_labels, k_max=50, num_trials=NUM_TRIALS):
    """Run retrieval benchmark for given embeddings."""
    # Build the FAISS index
    dimension = X_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    index.add(X_embeddings.astype(np.float32))  # FAISS requires float32

    # Prepare results container for each k value
    results = {k: [] for k in range(1, k_max + 1)}

    # Run trials
    for _ in tqdm(range(num_trials), desc="Running retrieval trials"):
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


def multi_run_benchmark(X_embeddings, y_labels, num_trials=NUM_TRIALS, num_runs=N_RUNS):
    """Run multiple retrieval benchmarks and report aggregate statistics."""
    results = []
    for _ in tqdm(range(num_runs), desc="Running multiple benchmarks"):
        avg_results, _ = run_retrieval_benchmark(
            X_embeddings, y_labels, k_max=max(K_VALUES), num_trials=num_trials
        )
        results.append(avg_results[10])  # Use precision@10 as the metric

    mean_result = np.mean(results)
    std_result = np.std(results)

    return mean_result, std_result


def generate_tsne_visualization(X, y, title):
    """Create t-SNE visualization for embeddings."""
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X)
    return X_reduced, y


def export_plot_with_layers(X_reduced, y_labels, filename, unique_labels):
    """Export plot with separate layers for each label."""
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(10, 8))

    # Get combinations of colors and markers
    combinations = []
    for color in CUSTOM_COLORS:
        for marker in CUSTOM_MARKERS:
            combinations.append((color, marker))

    # Shuffle to get varied combinations
    random.seed(42)
    random.shuffle(combinations)

    # Plot each label with a unique combination
    for i, label in enumerate(unique_labels):
        if i < len(combinations):
            color, marker = combinations[i]
        else:
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
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{filename}.svg", format="svg", bbox_inches="tight")
    plt.savefig(f"{filename}.pdf", format="pdf", bbox_inches="tight")

    return fig


def save_legend_separately(unique_labels, filename="legend"):
    """Save the legend as a separate file."""
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(5, 8))

    # Generate combinations of colors and markers
    combinations = []
    for color in CUSTOM_COLORS:
        for marker in CUSTOM_MARKERS:
            combinations.append((color, marker))

    # Use same seed for reproducibility
    random.seed(42)
    random.shuffle(combinations)

    # Create dummy scatter points
    for i, label in enumerate(unique_labels):
        if i < len(combinations):
            color, marker = combinations[i]
        else:
            combo_idx = i % len(combinations)
            color, marker = combinations[combo_idx]

        base_color = np.array(matplotlib.colors.to_rgb(color))
        edge_color = base_color * 0.7

        ax.scatter(
            [0],
            [0],
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
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches=legend_bbox)
    fig.savefig(f"{filename}.svg", format="svg", bbox_inches=legend_bbox)
    fig.savefig(f"{filename}.pdf", format="pdf", bbox_inches=legend_bbox)

    return fig


def generate_tsne_plot(X, y, study, save=True):
    """Generate t-SNE plot with separate layers and legend."""
    # Generate tsne embeddings
    X_reduced, y_reduced = generate_tsne_visualization(X, y, study)
    unique_projects = np.unique(y)

    # Export the plot with separate layers
    export_plot_with_layers(
        X_reduced=X_reduced,
        y_labels=y,
        filename=f"{plots_dir}/{study}_tsne_plot",
        unique_labels=unique_projects,
    )

    # Save legend separately
    save_legend_separately(
        unique_labels=unique_projects,
        filename=f"{plots_dir}/{study}_legend",
    )

    if save:
        df = pd.DataFrame(
            {
                "x": X_reduced[:, 0],
                "project_id": y,
                "y": X_reduced[:, 1],
            }
        )
        df.to_csv(f"{plots_dir}/{study}_tsne.csv", index=False)


def plot_classification_results(classification_results):
    """Create bar chart of classification accuracy for different models."""
    plt.figure(figsize=(10, 6))

    models = [result["model"] for result in classification_results.values()]
    accuracies = [result["mean_accuracy"] for result in classification_results.values()]
    errors = [result["std_accuracy"] for result in classification_results.values()]

    # Define consistent colors for each model
    colors = {
        "[UFNLP/gatortron-base] Pre-trained": CUSTOM_COLORS[0],
        "[UFNLP/gatortron-base] Fine-tuned": CUSTOM_COLORS[1],
        "[bert-base-uncased] Pre-trained": CUSTOM_COLORS[2],
        "[bert-base-uncased] Fine-tuned": CUSTOM_COLORS[3],
    }

    bar_colors = [colors.get(model, CUSTOM_COLORS[0]) for model in models]

    bars = plt.bar(models, accuracies, yerr=errors, capsize=10, color=bar_colors)

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

    plt.ylabel("Classification Accuracy")
    plt.title("Cancer Type Classification Accuracy by Model")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{plots_dir}/classification_accuracy.png", dpi=300)
    plt.savefig(f"{plots_dir}/classification_accuracy.svg", format="svg")
    plt.savefig(f"{plots_dir}/classification_accuracy.pdf", format="pdf")
    plt.close()


def plot_confusion_matrix(confusion_matrices):
    """Plot confusion matrix for the best-performing model."""
    best_model = max(
        confusion_matrices, key=lambda x: confusion_matrices[x]["accuracy"]
    )

    cm_data = confusion_matrices[best_model]
    cm = cm_data["cm"]
    labels = cm_data["labels"]

    # Remove "TCGA-" prefix for cleaner display
    display_labels = [
        label.replace("TCGA-", "")
        if isinstance(label, str) and label.startswith("TCGA-")
        else label
        for label in labels
    ]

    # Normalize confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=False,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {best_model}")
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{plots_dir}/{best_model}_confusion_matrix.png", dpi=300)
    plt.savefig(f"{plots_dir}/{best_model}_confusion_matrix.svg", format="svg")
    plt.savefig(f"{plots_dir}/{best_model}_confusion_matrix.pdf", format="pdf")
    plt.close()


def plot_retrieval_results(retrieval_results, k_values=K_VALUES):
    """Create precision@k curves for different models."""
    plt.figure(figsize=(10, 6))

    # Define consistent colors and markers for each model
    models = list(retrieval_results.keys())
    colors = {
        "[UFNLP/gatortron-base] Pre-trained": CUSTOM_COLORS[0],
        "[UFNLP/gatortron-base] Fine-tuned": CUSTOM_COLORS[1],
        "[bert-base-uncased] Pre-trained": CUSTOM_COLORS[2],
        "[bert-base-uncased] Fine-tuned": CUSTOM_COLORS[3],
    }

    markers = {
        "[UFNLP/gatortron-base] Pre-trained": CUSTOM_MARKERS[0],
        "[UFNLP/gatortron-base] Fine-tuned": CUSTOM_MARKERS[1],
        "[bert-base-uncased] Pre-trained": CUSTOM_MARKERS[2],
        "[bert-base-uncased] Fine-tuned": CUSTOM_MARKERS[3],
    }

    for i, model in enumerate(models):
        avg_results = retrieval_results[model]["avg"]
        std_results = retrieval_results[model]["std"]

        precisions = [avg_results[k] for k in k_values]
        errors = [std_results[k] for k in k_values]

        color = colors.get(model, CUSTOM_COLORS[i % len(CUSTOM_COLORS)])
        marker = markers.get(model, CUSTOM_MARKERS[i % len(CUSTOM_MARKERS)])

        plt.plot(
            k_values,
            precisions,
            marker=marker,
            label=model,
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
    plt.title("Retrieval Performance by Model")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Mark specific k values
    for k in [5, 10, 20]:
        if k in k_values:
            plt.axvline(x=k, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{plots_dir}/retrieval_precision_at_k.png", dpi=300)
    plt.savefig(f"{plots_dir}/retrieval_precision_at_k.svg", format="svg")
    plt.savefig(f"{plots_dir}/retrieval_precision_at_k.pdf", format="pdf")
    plt.close()


def plot_retrieval_heatmap(retrieval_results, k_values=K_VALUES):
    """Create heatmap visualization of precision@k for all models."""
    plt.figure(figsize=(12, 8))

    # Prepare data for heatmap
    models = list(retrieval_results.keys())
    data_matrix = np.zeros((len(models), len(k_values)))

    # Fill the data matrix
    for i, model in enumerate(models):
        avg_results = retrieval_results[model]["avg"]
        for j, k in enumerate(k_values):
            data_matrix[i, j] = avg_results[k]

    # Create heatmap
    ax = sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=k_values,
        yticklabels=models,
    )

    plt.xlabel("k value")
    plt.ylabel("Model")
    plt.title("Precision@k Across Models")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{plots_dir}/retrieval_heatmap.png", dpi=300)
    plt.savefig(f"{plots_dir}/retrieval_heatmap.svg", format="svg")
    plt.savefig(f"{plots_dir}/retrieval_heatmap.pdf", format="pdf")
    plt.close()


def create_comparison_figure(
    classification_results, confusion_matrices, retrieval_results, embeddings_dict
):
    """Create a comprehensive comparison figure."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)

    # Panel A: Classification accuracy bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    models = [result["model"] for result in classification_results.values()]
    accuracies = [result["mean_accuracy"] for result in classification_results.values()]
    errors = [result["std_accuracy"] for result in classification_results.values()]

    colors = {
        "[UFNLP/gatortron-base] Pre-trained": CUSTOM_COLORS[0],
        "[UFNLP/gatortron-base] Fine-tuned": CUSTOM_COLORS[1],
        "[bert-base-uncased] Pre-trained": CUSTOM_COLORS[2],
        "[bert-base-uncased] Fine-tuned": CUSTOM_COLORS[3],
    }

    bar_colors = [colors.get(model, CUSTOM_COLORS[0]) for model in models]

    bars = ax1.bar(
        range(len(models)), accuracies, yerr=errors, capsize=10, color=bar_colors
    )
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.split("] ")[1] for m in models], rotation=45, ha="right")
    ax1.set_ylabel("Classification Accuracy", fontsize=16)
    ax1.set_title("A. Cancer Type Classification Accuracy", fontsize=18)
    ax1.set_ylim(0, 1.0)

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

    # Panel B: Confusion matrix for the best model
    ax2 = fig.add_subplot(gs[0, 1])
    best_model = max(
        confusion_matrices, key=lambda x: confusion_matrices[x]["accuracy"]
    )

    cm_data = confusion_matrices[best_model]
    cm = cm_data["cm"]
    labels = cm_data["labels"]

    # Remove "TCGA-" prefix
    display_labels = [
        label.replace("TCGA-", "")
        if isinstance(label, str) and label.startswith("TCGA-")
        else label
        for label in labels
    ]

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
        annot_kws={"size": 12},
    )
    ax2.set_xlabel("Predicted Label", fontsize=16)
    ax2.set_ylabel("True Label", fontsize=16)
    ax2.set_title(f"B. Confusion Matrix for {best_model.split('] ')[1]}", fontsize=18)

    # Panel C: Retrieval precision@k heatmap
    ax3 = fig.add_subplot(gs[1, 0])

    # Prepare data for heatmap
    models = list(retrieval_results.keys())
    data_matrix = np.zeros((len(models), len(K_VALUES)))

    # Fill the data matrix
    for i, model in enumerate(models):
        avg_results = retrieval_results[model]["avg"]
        for j, k in enumerate(K_VALUES):
            data_matrix[i, j] = avg_results[k]

    # Create heatmap
    sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=K_VALUES,
        yticklabels=[m.split("] ")[1] for m in models],
        ax=ax3,
        annot_kws={"size": 12},
    )

    ax3.set_xlabel("k value", fontsize=16)
    ax3.set_ylabel("Model", fontsize=16)
    ax3.set_title("C. Precision@k Across Models", fontsize=18)

    # Panel D: t-SNE visualization of best model's embeddings
    ax4 = fig.add_subplot(gs[1, 1])

    # Find best performing model for retrieval
    best_retrieval_model = max(
        retrieval_results,
        key=lambda x: retrieval_results[x]["avg"][10],  # Using precision@10
    )

    # Get embeddings for best model
    X = embeddings_dict[best_retrieval_model]["X"]
    y = embeddings_dict[best_retrieval_model]["y"]

    # Run t-SNE
    X_reduced, y_reduced = generate_tsne_visualization(X, y, best_retrieval_model)

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
            label=str(cancer_type).replace("TCGA-", ""),
            color=color,
            marker=marker,
            s=50,
            alpha=0.9,
            edgecolors=edge_color,
            linewidths=0.5,
        )

    ax4.set_title(
        f"D. t-SNE Visualization of {best_retrieval_model.split('] ')[1]} Embeddings",
        fontsize=18,
    )
    ax4.set_axis_off()

    plt.tight_layout()

    # Save figure
    plt.savefig(
        f"{plots_dir}/model_comparison_figure.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(f"{plots_dir}/model_comparison_figure.pdf", bbox_inches="tight")
    plt.savefig(f"{plots_dir}/model_comparison_figure.svg", bbox_inches="tight")
    plt.close()


def main():
    """Main function to run the entire benchmarking pipeline."""
    from transformers import TrainingArguments, Trainer

    print("Starting clinical data language model benchmarking...")

    # Load the Clinical Data
    print("Loading TCGA clinical data...")
    try:
        clinical_data = load_dataset(
            "Lab-Rasool/TCGA", "clinical", split="gatortron"
        ).to_pandas()
        print(f"Loaded {len(clinical_data)} clinical samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Ensure data has 'text' column (some datasets use different column names)
    if "text" not in clinical_data.columns:
        # Try to find a suitable text column
        for col in clinical_data.columns:
            if (
                "note" in col.lower()
                or "text" in col.lower()
                or "description" in col.lower()
            ):
                print(f"Using '{col}' as text column")
                clinical_data["text"] = clinical_data[col]
                break

        if "text" not in clinical_data.columns:
            # As a last resort, concatenate non-ID columns
            print("No specific text column found, concatenating non-ID columns")
            non_id_cols = [
                col for col in clinical_data.columns if not col.endswith("_id")
            ]
            clinical_data["text"] = clinical_data[non_id_cols].apply(
                lambda row: " ".join(str(val) for val in row if val is not None), axis=1
            )

    # Check if project_id exists, if not use disease_type if available
    if "project_id" not in clinical_data.columns:
        if "disease_type" in clinical_data.columns:
            print("Using 'disease_type' as project_id")
            clinical_data["project_id"] = clinical_data["disease_type"]
        else:
            print("No project_id or disease_type column found, creating dummy labels")
            clinical_data["project_id"] = "unknown"

    # Remove rows with null values in important columns
    clinical_data = clinical_data.dropna(subset=["text", "project_id"])
    print(f"After data cleaning: {len(clinical_data)} samples")

    # Fine-tune models with proper paths
    gatortron_model_path = os.path.join(models_dir, "gatortron-base-clinical")
    bert_model_path = os.path.join(models_dir, "bert-base-uncased-clinical")

    # Define paths for embedding files
    gatortron_pt_data_path = os.path.join(data_dir, "gatortron-pretrained-data.parquet")
    gatortron_ft_data_path = os.path.join(data_dir, "gatortron-finetuned-data.parquet")
    bert_pt_data_path = os.path.join(data_dir, "bert-pretrained-data.parquet")
    bert_ft_data_path = os.path.join(data_dir, "bert-finetuned-data.parquet")

    # Check if models already exist, if not train them
    if not os.path.exists(gatortron_model_path):
        print("Fine-tuning Gatortron model...")
        train_and_save_model(
            pretrained_model="UFNLP/gatortron-base",
            data=clinical_data,
            output_model_dir=gatortron_model_path,
        )
    else:
        print(f"Gatortron fine-tuned model already exists at {gatortron_model_path}")

    if not os.path.exists(bert_model_path):
        print("Fine-tuning BERT model...")
        train_and_save_model(
            pretrained_model="bert-base-uncased",
            data=clinical_data,
            output_model_dir=bert_model_path,
        )
    else:
        print(f"BERT fine-tuned model already exists at {bert_model_path}")

    # Generate embeddings
    print("\n=== Generating Embeddings ===")

    # Gatortron pre-trained
    print("Processing Gatortron pre-trained embeddings...")
    gatortron_pretrained_data = generate_or_load_pt_embeddings(
        pre_trained_model="UFNLP/gatortron-base",
        data=clinical_data,
        output_dir=gatortron_pt_data_path,
    )
    X_gatortron_pretrained = np.array(
        [
            np.frombuffer(e, dtype=np.float32)
            for e in gatortron_pretrained_data["embeddings"]
        ]
    )
    y_gatortron_pretrained = gatortron_pretrained_data["project_id"].values

    # Gatortron fine-tuned
    print("Processing Gatortron fine-tuned embeddings...")
    gatortron_finetuned_data = generate_or_load_ft_embeddings(
        pre_trained_model="UFNLP/gatortron-base",
        fine_tuned_model=gatortron_model_path,
        data=clinical_data,
        output_dir=gatortron_ft_data_path,
    )
    X_gatortron_finetuned = np.array(
        [
            np.frombuffer(e, dtype=np.float32)
            for e in gatortron_finetuned_data["embeddings"]
        ]
    )
    y_gatortron_finetuned = gatortron_finetuned_data["project_id"].values

    # BERT pre-trained
    print("Processing BERT pre-trained embeddings...")
    bert_pretrained_data = generate_or_load_pt_embeddings(
        pre_trained_model="bert-base-uncased",
        data=clinical_data,
        output_dir=bert_pt_data_path,
    )
    X_bert_pretrained = np.array(
        [np.frombuffer(e, dtype=np.float32) for e in bert_pretrained_data["embeddings"]]
    )
    y_bert_pretrained = bert_pretrained_data["project_id"].values

    # BERT fine-tuned
    print("Processing BERT fine-tuned embeddings...")
    bert_finetuned_data = generate_or_load_ft_embeddings(
        pre_trained_model="bert-base-uncased",
        fine_tuned_model=bert_model_path,
        data=clinical_data,
        output_dir=bert_ft_data_path,
    )
    X_bert_finetuned = np.array(
        [np.frombuffer(e, dtype=np.float32) for e in bert_finetuned_data["embeddings"]]
    )
    y_bert_finetuned = bert_finetuned_data["project_id"].values

    # Create dictionary of embeddings for visualization
    embeddings_dict = {
        "[UFNLP/gatortron-base] Pre-trained": {
            "X": X_gatortron_pretrained,
            "y": y_gatortron_pretrained,
        },
        "[UFNLP/gatortron-base] Fine-tuned": {
            "X": X_gatortron_finetuned,
            "y": y_gatortron_finetuned,
        },
        "[bert-base-uncased] Pre-trained": {
            "X": X_bert_pretrained,
            "y": y_bert_pretrained,
        },
        "[bert-base-uncased] Fine-tuned": {
            "X": X_bert_finetuned,
            "y": y_bert_finetuned,
        },
    }

    # Run classification experiments
    print("\n=== Running Classification Experiments ===")
    classification_results = {}

    classification_results["gatortron_pt"] = run_classification_experiments(
        n_runs=N_RUNS,
        X=X_gatortron_pretrained,
        y=y_gatortron_pretrained,
        study="[UFNLP/gatortron-base] Pre-trained",
    )

    classification_results["gatortron_ft"] = run_classification_experiments(
        n_runs=N_RUNS,
        X=X_gatortron_finetuned,
        y=y_gatortron_finetuned,
        study="[UFNLP/gatortron-base] Fine-tuned",
    )

    classification_results["bert_pt"] = run_classification_experiments(
        n_runs=N_RUNS,
        X=X_bert_pretrained,
        y=y_bert_pretrained,
        study="[bert-base-uncased] Pre-trained",
    )

    classification_results["bert_ft"] = run_classification_experiments(
        n_runs=N_RUNS,
        X=X_bert_finetuned,
        y=y_bert_finetuned,
        study="[bert-base-uncased] Fine-tuned",
    )

    # Create confusion matrices for the best runs
    confusion_matrices = {}
    for model, result in classification_results.items():
        if result["best_y_test"] is not None and result["best_y_pred"] is not None:
            cm = confusion_matrix(result["best_y_test"], result["best_y_pred"])
            confusion_matrices[result["model"]] = {
                "cm": cm,
                "labels": np.unique(result["best_y_test"]),
                "accuracy": result["mean_accuracy"],
            }

    # Run retrieval experiments
    print("\n=== Running Retrieval Experiments ===")
    retrieval_results = {}

    # Gatortron pre-trained
    print("Evaluating Gatortron pre-trained retrieval...")
    gatortron_pt_avg, gatortron_pt_std = run_retrieval_benchmark(
        X_gatortron_pretrained, y_gatortron_pretrained, k_max=max(K_VALUES)
    )
    retrieval_results["[UFNLP/gatortron-base] Pre-trained"] = {
        "avg": gatortron_pt_avg,
        "std": {k: gatortron_pt_std[k] for k in gatortron_pt_avg},
    }

    # Gatortron fine-tuned
    print("Evaluating Gatortron fine-tuned retrieval...")
    gatortron_ft_avg, gatortron_ft_std = run_retrieval_benchmark(
        X_gatortron_finetuned, y_gatortron_finetuned, k_max=max(K_VALUES)
    )
    retrieval_results["[UFNLP/gatortron-base] Fine-tuned"] = {
        "avg": gatortron_ft_avg,
        "std": {k: gatortron_ft_std[k] for k in gatortron_ft_avg},
    }

    # BERT pre-trained
    print("Evaluating BERT pre-trained retrieval...")
    bert_pt_avg, bert_pt_std = run_retrieval_benchmark(
        X_bert_pretrained, y_bert_pretrained, k_max=max(K_VALUES)
    )
    retrieval_results["[bert-base-uncased] Pre-trained"] = {
        "avg": bert_pt_avg,
        "std": {k: bert_pt_std[k] for k in bert_pt_avg},
    }

    # BERT fine-tuned
    print("Evaluating BERT fine-tuned retrieval...")
    bert_ft_avg, bert_ft_std = run_retrieval_benchmark(
        X_bert_finetuned, y_bert_finetuned, k_max=max(K_VALUES)
    )
    retrieval_results["[bert-base-uncased] Fine-tuned"] = {
        "avg": bert_ft_avg,
        "std": {k: bert_ft_std[k] for k in bert_ft_avg},
    }

    # Generate t-SNE visualizations
    print("\n=== Generating t-SNE Visualizations ===")
    generate_tsne_plot(
        X_gatortron_pretrained,
        y_gatortron_pretrained,
        study="gatortron-pretrained",
    )

    generate_tsne_plot(
        X_gatortron_finetuned,
        y_gatortron_finetuned,
        study="gatortron-finetuned",
    )

    generate_tsne_plot(
        X_bert_pretrained,
        y_bert_pretrained,
        study="bert-pretrained",
    )

    generate_tsne_plot(
        X_bert_finetuned,
        y_bert_finetuned,
        study="bert-finetuned",
    )

    # Create visualization plots
    print("\n=== Creating Visualization Plots ===")
    plot_classification_results(classification_results)
    plot_confusion_matrix(confusion_matrices)
    plot_retrieval_results(retrieval_results)
    plot_retrieval_heatmap(retrieval_results)

    # Create comprehensive comparison figure
    create_comparison_figure(
        classification_results, confusion_matrices, retrieval_results, embeddings_dict
    )

    # Save results to CSV file
    results_df = pd.DataFrame(
        {
            "Model": [result["model"] for result in classification_results.values()],
            "Classification_Accuracy": [
                result["mean_accuracy"] for result in classification_results.values()
            ],
            "Classification_Std": [
                result["std_accuracy"] for result in classification_results.values()
            ],
            "Retrieval_Precision@10": [
                retrieval_results[result["model"]]["avg"][10]
                if 10 in retrieval_results[result["model"]]["avg"]
                else retrieval_results[result["model"]]["avg"]["10"]
                for result in classification_results.values()
            ],
            "Retrieval_Std@10": [
                retrieval_results[result["model"]]["std"][10]
                if 10 in retrieval_results[result["model"]]["std"]
                else retrieval_results[result["model"]]["std"]["10"]
                for result in classification_results.values()
            ],
        }
    )

    results_df.to_csv(os.path.join(BASE_DIR, "benchmarking_results.csv"), index=False)
    print(f"Results saved to {os.path.join(BASE_DIR, 'benchmarking_results.csv')}")

    # Print final results
    print("\n=== Final Results ===")
    print(results_df)
    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
