from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModel,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from datasets import Dataset
import numpy as np
import pandas as pd
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import os
import faiss
import random
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define these constants at the top level of the file for consistency
CUSTOM_COLORS = ["#FF6060", "#6CC2F5", "#FF9A60", "#FF60B3", "#C260FF", "#60FFA0"]
# square, circle, star, triangle up, pentagon, triangle down
CUSTOM_MARKERS = ["s", "o", "*", "^", "p", "v"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def prepare_data_and_tokenize(data, tokenizer):
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
        model_inputs["labels"] = examples[
            "labels"
        ]  # Ensure labels are included for the model to compute loss
        return model_inputs

    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Split the dataset
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)
    return tokenized_datasets["train"], tokenized_datasets["test"]


def train_and_save_model(pretrained_model, data, output_model_dir):
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

    # Apply the LoRA adapter properly
    lora_model = get_peft_model(model, config)

    # Prepare and tokenize data
    train_dataset, test_dataset = prepare_data_and_tokenize(data, tokenizer)

    # Updated training arguments
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

    trainer.model.save_pretrained(MODEL_SAVE_FOLDER_NAME)
    trainer.save_model(MODEL_SAVE_FOLDER_NAME)
    trainer.model.config.save_pretrained(MODEL_SAVE_FOLDER_NAME)


def generate_or_load_ft_embeddings(
    pre_trained_model, fine_tuned_model, data, output_dir
):
    if not os.path.exists(output_dir):
        dataset = Dataset.from_pandas(data)
        # The key fix: Load the fine-tuned model properly
        # Instead of using AutoModel, we should load the AutoModelForSequenceClassification
        # since that's what was saved during fine-tuning
        model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model)

        # Extract the base model (without classification head) for embeddings
        # Get the transformer base model for embeddings
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
            for batch in tqdm(dataloader, desc="Generating Embeddings"):
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


def generate_or_load_pt_embeddings(pre_trained_model, data, output_dir):
    if not os.path.exists(output_dir):
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
            for batch in tqdm(dataloader, desc="Generating Embeddings"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)
        data["embeddings"] = list(map(list, embeddings))
        data.to_parquet(output_dir)
    pre_trained_data = pd.read_parquet(output_dir)
    return pre_trained_data


def run_experiments(n_runs, X, y, study):
    accuracies = []
    random_seeds = np.random.randint(0, 10000, size=n_runs)
    for seed in tqdm(random_seeds, desc=f"Running {study} Experiments"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"{study} Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    return mean_accuracy, std_accuracy


def export_plot_with_layers(
    X_reduced, y_labels, filename, palette, unique_projects, include_legend=True
):
    """Export plot with separate layers for each project ID - easier to edit in vector editors"""
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(10, 8))

    global CUSTOM_COLORS, CUSTOM_MARKERS
    # Get all possible color-marker combinations
    combinations = []
    for color in CUSTOM_COLORS:
        for marker in CUSTOM_MARKERS:
            combinations.append((color, marker))

    # Shuffle to get varied combinations
    random.seed(42)  # For reproducibility
    random.shuffle(combinations)

    # Plot each project with a unique combination
    for i, project in enumerate(unique_projects):
        if i < len(combinations):
            color, marker = combinations[i]
        else:
            # Fallback if we have more projects than combinations
            combo_idx = i % len(combinations)
            color, marker = combinations[combo_idx]
            print(f"Warning: Reusing combination for project {project}")

        base_color = np.array(matplotlib.colors.to_rgb(color))
        edge_color = base_color * 0.7

        indices = np.where(y_labels == project)[0]
        plt.scatter(
            X_reduced[indices, 0],
            X_reduced[indices, 1],
            c=[color],
            label=str(project),
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

    return fig


def save_legend_separately(unique_projects, palette, markers, filename="legend"):
    """Save the legend as a separate file"""
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots(figsize=(5, 8))

    # Use the global constants
    global CUSTOM_COLORS, CUSTOM_MARKERS

    # Generate the same color-marker combinations as in the main plot function
    combinations = []
    for color in CUSTOM_COLORS:
        for marker in CUSTOM_MARKERS:
            combinations.append((color, marker))

    # Use same seed for reproducibility
    random.seed(42)
    random.shuffle(combinations)

    # Create dummy scatter points using the same assignment logic
    for i, project in enumerate(unique_projects):
        if i < len(combinations):
            color, marker = combinations[i]
        else:
            # Fallback if we have more projects than combinations
            combo_idx = i % len(combinations)
            color, marker = combinations[combo_idx]

        base_color = np.array(matplotlib.colors.to_rgb(color))
        edge_color = base_color * 0.7

        ax.scatter(
            [0],
            [0],  # Position doesn't matter
            c=[color],
            label=str(project),
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
        title="Project ID",
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

    return fig


def generate_tsne_plot(X, y, study, save=True):
    tsne = TSNE(n_components=2, random_state=42)
    unique_projects = np.unique(y)

    # No need to define palette and markers that won't be used
    # Simply pass the constants directly to the functions

    # Generate tsne embeddings
    X_reduced = tsne.fit_transform(X)

    # Export the plot with separate layers
    export_plot_with_layers(
        X_reduced=X_reduced,
        y_labels=y,
        filename=f"{study}_tsne_plot",
        palette=CUSTOM_COLORS,  # This will be unused but kept for parameter consistency
        unique_projects=unique_projects,
        include_legend=False,
    )

    save_legend_separately(
        unique_projects=unique_projects,
        palette=CUSTOM_COLORS,  # This will be unused but kept for parameter consistency
        markers=CUSTOM_MARKERS,  # This will be unused but kept for parameter consistency
        filename=f"{study}_legend",
    )

    if save:
        df = pd.DataFrame(
            {
                "x": X_reduced[:, 0],
                "project_id": y,
                "y": X_reduced[:, 1],
            }
        )
        df.to_csv(f"{study}_tsne.csv", index=False)


# Function to perform similarity search and evaluate matching project_ids
def run_benchmark(X_embeddings, y_labels, num_trials=10):
    # Build the FAISS index
    dimension = X_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    index.add(X_embeddings)

    total_matching_count = 0

    for _ in range(num_trials):
        # Pick a random patient embedding for evaluation
        random_index = random.randint(0, len(X_embeddings) - 1)
        query_embedding = X_embeddings[random_index]
        query_project_id = y_labels[random_index]

        # Retrieve the next 10 closest patients
        distances, indices = index.search(
            query_embedding.reshape(1, -1), k=11
        )  # k=11 to include the query patient itself

        # Remove the first index (the query patient itself)
        indices = indices[0][1:]
        distances = distances[0][1:]

        # Check for matching "project_id"
        matching_count = 0
        for idx in indices:
            if y_labels[int(idx)] == query_project_id:  # Convert idx to int
                matching_count += 1

        total_matching_count += matching_count

    average_matching_count = total_matching_count / num_trials
    average_matching_count = (
        average_matching_count / 10
    )  # Divide by 10 to get the percentage
    return average_matching_count


def multi_run_benchmark(X_embeddings, y_labels, num_trials=10, num_runs=10):
    results = []
    for _ in tqdm(range(num_runs), leave=False):
        results.append(run_benchmark(X_embeddings, y_labels, num_trials))
    return results


def main():
    BASE_DIR = "./benchmarking_analysis_results/"
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    # Create subdirectories for organization
    models_dir = os.path.join(BASE_DIR, "models")
    data_dir = os.path.join(BASE_DIR, "data")
    plots_dir = os.path.join(BASE_DIR, "plots")

    for directory in [models_dir, data_dir, plots_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Load the Clinical Data
    clinical_data = load_dataset(
        "Lab-Rasool/TCGA", "clinical", split="gatortron"
    ).to_pandas()

    # Fine-tune models with proper paths
    gatortron_model_path = os.path.join(models_dir, "gatortron-base-tcga")
    bert_model_path = os.path.join(models_dir, "bert-base-uncased-tcga")

    if not os.path.exists(gatortron_model_path):
        train_and_save_model(
            pretrained_model="UFNLP/gatortron-base",
            data=clinical_data,
            output_model_dir=gatortron_model_path,
        )

    if not os.path.exists(bert_model_path):
        train_and_save_model(
            pretrained_model="bert-base-uncased",
            data=clinical_data,
            output_model_dir=bert_model_path,
        )

    # Generate embeddings with proper paths
    gatortron_pt_data_path = os.path.join(data_dir, "gatortron-pretrained-data.parquet")
    gatortron_ft_data_path = os.path.join(data_dir, "gatortron-finetuned-data.parquet")
    bert_pt_data_path = os.path.join(data_dir, "bert-pretrained-data.parquet")
    bert_ft_data_path = os.path.join(data_dir, "bert-finetuned-data.parquet")

    # GatorTron Pre-trained
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
    (
        classification_gatortron_pretrained_accuracy,
        classification_gatortron_pretrained_std,
    ) = run_experiments(
        n_runs=10,
        study="[UFNLP/gatortron-base] Pre-trained",
        X=X_gatortron_pretrained,
        y=y_gatortron_pretrained,
    )

    # GatorTron Fine-tuned
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
    (
        classification_gatortron_finetuned_accuracy,
        classification_gatortron_finetuned_std,
    ) = run_experiments(
        n_runs=10,
        study="[UFNLP/gatortron-base] Fine-tuned",
        X=X_gatortron_finetuned,
        y=y_gatortron_finetuned,
    )

    # BERT Pre-trained
    bert_pretrained_data = generate_or_load_pt_embeddings(
        pre_trained_model="bert-base-uncased",
        data=clinical_data,
        output_dir=bert_pt_data_path,
    )
    X_bert_pretrained = np.array(
        [
            np.frombuffer(e, dtype=np.float32)
            for e in bert_pretrained_data["embeddings"]
        ]  # Consistent key name
    )
    y_bert_pretrained = bert_pretrained_data["project_id"].values
    classification_bert_pretrained_accuracy, classification_bert_pretrained_std = (
        run_experiments(
            n_runs=10,
            study="[bert-base-uncased] Pre-trained",
            X=X_bert_pretrained,
            y=y_bert_pretrained,
        )
    )

    # BERT Fine-tuned
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
    classification_bert_finetuned_accuracy, classification_bert_finetuned_std = (
        run_experiments(
            n_runs=10,
            study="[bert-base-uncased] Fine-tuned",
            X=X_bert_finetuned,
            y=y_bert_finetuned,
        )
    )

    # Generate t-SNE Visualizations with proper paths
    generate_tsne_plot(
        X_gatortron_pretrained,
        y_gatortron_pretrained,
        study=os.path.join(plots_dir, "gatortron-pretrained"),
    )
    generate_tsne_plot(
        X_gatortron_finetuned,
        y_gatortron_finetuned,
        study=os.path.join(plots_dir, "gatortron-finetuned"),
    )
    generate_tsne_plot(
        X_bert_pretrained,
        y_bert_pretrained,
        study=os.path.join(plots_dir, "bert-pretrained"),
    )
    generate_tsne_plot(
        X_bert_finetuned,
        y_bert_finetuned,
        study=os.path.join(plots_dir, "bert-finetuned"),
    )

    # Retrieval Benchmarking
    num_trials = 100
    num_runs = 100
    results_gatortron_pretrained = multi_run_benchmark(
        X_gatortron_pretrained,
        y_gatortron_pretrained,
        num_trials=num_trials,
        num_runs=num_runs,
    )
    print(
        f"Gatortron pretrained embeddings: {np.mean(results_gatortron_pretrained):.4f} ± {np.std(results_gatortron_pretrained):.4f}"
    )
    results_gatortron_finetuned = multi_run_benchmark(
        X_gatortron_finetuned,
        y_gatortron_finetuned,
        num_trials=num_trials,
        num_runs=num_runs,
    )
    print(
        f"Gatortron fine-tuned embeddings: {np.mean(results_gatortron_finetuned):.4f} ± {np.std(results_gatortron_finetuned):.4f}"
    )
    results_bert_pretrained = multi_run_benchmark(
        X_bert_pretrained, y_bert_pretrained, num_trials=num_trials, num_runs=num_runs
    )
    print(
        f"BERT pretrained embeddings: {np.mean(results_bert_pretrained):.4f} ± {np.std(results_bert_pretrained):.4f}"
    )

    results_bert_finetuned = multi_run_benchmark(
        X_bert_finetuned, y_bert_finetuned, num_trials=num_trials, num_runs=num_runs
    )
    print(
        f"BERT fine-tuned embeddings: {np.mean(results_bert_finetuned):.4f} ± {np.std(results_bert_finetuned):.4f}"
    )

    # save the results to a CSV file for both classification and retrieval (acc and std)
    results_df = pd.DataFrame(
        {
            "Model": [
                "Gatortron Pretrained",
                "Gatortron Finetuned",
                "BERT Pretrained",
                "BERT Finetuned",
            ],
            "Classification Accuracy": [
                classification_gatortron_pretrained_accuracy,
                classification_gatortron_finetuned_accuracy,
                classification_bert_pretrained_accuracy,
                classification_bert_finetuned_accuracy,
            ],
            "Classification Std": [
                classification_gatortron_pretrained_std,
                classification_gatortron_finetuned_std,
                classification_bert_pretrained_std,
                classification_bert_finetuned_std,
            ],
            "Retrieval Accuracy": [
                np.mean(results_gatortron_pretrained),
                np.mean(results_gatortron_finetuned),
                np.mean(results_bert_pretrained),
                np.mean(results_bert_finetuned),
            ],
            "Retrieval Std": [
                np.std(results_gatortron_pretrained),
                np.std(results_gatortron_finetuned),
                np.std(results_bert_pretrained),
                np.std(results_bert_finetuned),
            ],
        }
    )
    results_df.to_csv(os.path.join(BASE_DIR, "benchmarking_results.csv"), index=False)


if __name__ == "__main__":
    main()
