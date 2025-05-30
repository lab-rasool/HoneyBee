import numpy as np
from datasets import load_dataset
import random
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import pickle

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def load_gatortron_model():
    """
    Load and return the Gatortron model, tokenizer, and configuration.
    """
    print("Loading Gatortron model...")
    tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-base")
    config = AutoConfig.from_pretrained("UFNLP/gatortron-base")
    model = AutoModel.from_pretrained("UFNLP/gatortron-base")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Gatortron model loaded on {device}")

    return model, tokenizer, config, device


def generate_embeddings(texts, model, tokenizer, device, batch_size=8, max_length=512):
    """
    Generate embeddings for a list of texts using the Gatortron model.

    Args:
        texts (list): List of text strings
        model: Gatortron model
        tokenizer: Gatortron tokenizer
        device: Device to run the model on
        batch_size (int): Batch size for processing
        max_length (int): Maximum token length for the tokenizer

    Returns:
        numpy.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
    """
    model.eval()
    embeddings = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i : i + batch_size]

        # Handle None or NaN values
        batch_texts = [str(text) if text is not None else "" for text in batch_texts]

        # Tokenize
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        # Generate embeddings
        with torch.no_grad():
            output = model(**encoded_input)

        # Use the CLS token embedding (first token) as the sentence embedding
        batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


def get_id_column(df):
    """
    Determine the appropriate ID column in a dataframe.
    Check for common patient ID column names and return the first one found.
    """
    possible_id_columns = [
        "case_submitter_id",
        "PatientID",
        "patient_id",
        "case_id",
        "ID",
    ]

    for col in possible_id_columns:
        if col in df.columns:
            return col

    # If no known ID column is found, print all columns and raise an error
    print(f"Available columns: {df.columns.tolist()}")
    raise ValueError("No patient ID column found in dataframe.")


def get_project_id_column(df):
    """
    Determine the appropriate project ID column in a dataframe.
    Check for common project ID column names and return the first one found.
    """
    possible_project_columns = [
        "project_id",
        "ProjectID",
        "project",
    ]

    for col in possible_project_columns:
        if col in df.columns:
            return col

    return None  # Return None if no project ID column is found


def process_clinical_embeddings():
    """
    Process clinical embeddings from the TCGA dataset using Gatortron.
    If embeddings were previously saved, load them instead of regenerating.
    """
    print("Processing clinical embeddings...")

    # Declare global variable at the beginning of the function
    global patient_to_project_mapping

    # Try to load saved embeddings first
    saved_embeddings = load_saved_embeddings("clinical")
    if saved_embeddings is not None:
        # Store the patient to project mapping for other modalities to use
        patient_to_project_mapping = dict(
            zip(saved_embeddings["patient_ids"], saved_embeddings["y"])
        )
        return saved_embeddings

    print("Loading clinical data...")
    try:
        # Load clinical data
        clinical_data = load_dataset(
            "Lab-Rasool/TCGA", "clinical", split="gatortron"
        ).to_pandas()
        print(f"Loaded {len(clinical_data)} clinical samples")

        # Get ID column
        clinical_id_col = get_id_column(clinical_data)

        # Get project ID column
        project_id_col = get_project_id_column(clinical_data)
        if not project_id_col:
            print("Warning: No project ID column found in clinical data")
            clinical_data["project_id"] = "unknown"
            project_id_col = "project_id"

        # Create patient_id to project_id mapping
        # No need to redeclare global here
        patient_to_project_mapping = dict(
            zip(clinical_data[clinical_id_col], clinical_data[project_id_col])
        )

        # Get text column (assuming clinical notes are in a column named 'notes' or similar)
        text_column = None
        possible_text_columns = ["notes", "clinical_notes", "text", "clinical_text"]
        for col in possible_text_columns:
            if col in clinical_data.columns:
                text_column = col
                break

        if text_column is None:
            print(
                "Warning: No text column found in clinical data. Using first string column as fallback."
            )
            # Find first string column as fallback
            for col in clinical_data.columns:
                if clinical_data[col].dtype == "object" and not col.endswith("_id"):
                    text_column = col
                    print(f"Using {text_column} as text column")
                    break

        if text_column is None:
            print("Error: Cannot find suitable text column in clinical data")
            return None

        # Remove rows with null text
        clinical_data = clinical_data.dropna(subset=[text_column])
        print(f"After removing null text: {len(clinical_data)} clinical samples")

        # Load model and generate embeddings
        model, tokenizer, config, device = load_gatortron_model()
        clinical_embeddings = generate_embeddings(
            clinical_data[text_column].tolist(), model, tokenizer, device
        )

        # Save the embeddings before returning
        embeddings_data = {
            "X": clinical_embeddings,
            "y": clinical_data[project_id_col].values,
            "patient_ids": clinical_data[clinical_id_col].values,
        }
        save_embeddings(embeddings_data, "clinical")

        return embeddings_data
    except Exception as e:
        print(f"Error processing clinical embeddings: {e}")
        return None


def process_pathology_embeddings(target_patient_ids=None):
    """
    Process pathology embeddings from the TCGA dataset using Gatortron.
    If embeddings were previously saved, load them instead of regenerating.
    """
    print("Processing pathology embeddings...")

    # Try to load saved embeddings first
    saved_embeddings = load_saved_embeddings("pathology", target_patient_ids)
    if saved_embeddings is not None:
        return saved_embeddings

    print("Loading pathology data...")
    try:
        # Load pathology data
        pathology_data = load_dataset(
            "Lab-Rasool/TCGA", "pathology_report", split="gatortron"
        ).to_pandas()
        print(f"Loaded {len(pathology_data)} pathology samples")

        # Get ID column
        pathology_id_col = get_id_column(pathology_data)

        # Filter by patient IDs if provided
        if target_patient_ids is not None:
            original_count = len(pathology_data)
            pathology_data = pathology_data[
                pathology_data[pathology_id_col].isin(target_patient_ids)
            ]
            print(
                f"After filtering by patient IDs: {len(pathology_data)} pathology samples (from {original_count})"
            )

        # Get text column
        text_column = None
        possible_text_columns = ["report_text", "pathology_text", "text", "report"]
        for col in possible_text_columns:
            if col in pathology_data.columns:
                text_column = col
                break

        if text_column is None:
            print(
                "Warning: No text column found in pathology data. Using first string column as fallback."
            )
            # Find first string column as fallback
            for col in pathology_data.columns:
                if pathology_data[col].dtype == "object" and not col.endswith("_id"):
                    text_column = col
                    print(f"Using {text_column} as text column")
                    break

        if text_column is None:
            print("Error: Cannot find suitable text column in pathology data")
            return None

        # Remove rows with null text
        pathology_data = pathology_data.dropna(subset=[text_column])
        print(f"After removing null text: {len(pathology_data)} pathology samples")

        if len(pathology_data) == 0:
            print("No valid pathology text found.")
            return None

        # Add project_id using patient_to_project_mapping
        if "patient_to_project_mapping" in globals():
            pathology_data["project_id"] = (
                pathology_data[pathology_id_col]
                .map(patient_to_project_mapping)
                .fillna("unknown")
            )
        else:
            print("Warning: No patient_to_project_mapping available for pathology data")
            pathology_data["project_id"] = "unknown"

        # Load model and generate embeddings
        model, tokenizer, config, device = load_gatortron_model()
        pathology_embeddings = generate_embeddings(
            pathology_data[text_column].tolist(), model, tokenizer, device
        )

        # Save the embeddings before returning
        embeddings_data = {
            "X": pathology_embeddings,
            "y": pathology_data["project_id"].values,
            "patient_ids": pathology_data[pathology_id_col].values,
        }
        save_embeddings(embeddings_data, "pathology", target_patient_ids)

        return embeddings_data
    except Exception as e:
        print(f"Error processing pathology embeddings: {e}")
        return None


def process_radiology_embeddings(target_patient_ids=None):
    """
    Process radiology embeddings from the TCGA dataset.
    """
    print("Loading radiology embeddings...")
    try:
        radiology_data = load_dataset(
            "Lab-Rasool/TCGA", "radiology", split="radimagenet"
        ).to_pandas()
        print(f"Loaded {len(radiology_data)} radiology samples")

        # Get ID column
        radiology_id_col = get_id_column(radiology_data)

        # Filter by patient IDs if provided
        if target_patient_ids is not None:
            original_count = len(radiology_data)
            radiology_data = radiology_data[
                radiology_data[radiology_id_col].isin(target_patient_ids)
            ]
            print(
                f"After filtering by patient IDs: {len(radiology_data)} radiology samples (from {original_count})"
            )

        # Remove rows with null embeddings
        radiology_data = radiology_data.dropna(subset=["embedding"])
        print(
            f"After removing null embeddings: {len(radiology_data)} radiology samples"
        )

        if len(radiology_data) == 0:
            print("No valid radiology embeddings found.")
            return None

        # Add project_id using patient_to_project_mapping
        if "patient_to_project_mapping" in globals():
            radiology_data["project_id"] = (
                radiology_data[radiology_id_col]
                .map(patient_to_project_mapping)
                .fillna("unknown")
            )
        else:
            print("Warning: No patient_to_project_mapping available for radiology data")
            radiology_data["project_id"] = "unknown"

        # Process embeddings
        processed_embeddings = []

        if "embedding_shape" in radiology_data.columns:
            print("Using embedding_shape for radiology embeddings")
            for idx, row in tqdm(
                radiology_data.iterrows(),
                desc="Processing radiology embeddings",
                total=len(radiology_data),
            ):
                try:
                    emb = np.frombuffer(row["embedding"], dtype=np.float32)
                    shape = row["embedding_shape"]

                    # Sometimes embedding_shape is a string, check and convert if needed
                    if isinstance(shape, str):
                        shape = eval(shape)  # Convert string to tuple

                    reshaped_emb = emb.reshape(shape)

                    # If multi-dimensional, flatten to 1D for consistency
                    if len(reshaped_emb.shape) > 1:
                        # Take mean along all dimensions except the last
                        flattened_emb = np.mean(
                            reshaped_emb, axis=tuple(range(len(reshaped_emb.shape) - 1))
                        )
                    else:
                        flattened_emb = reshaped_emb

                    processed_embeddings.append(flattened_emb)
                except Exception as e:
                    print(f"Error processing radiology embedding at index {idx}: {e}")
                    if len(processed_embeddings) > 0:
                        processed_embeddings.append(
                            np.zeros_like(processed_embeddings[0])
                        )
                    else:
                        processed_embeddings.append(np.zeros(1000, dtype=np.float32))
        else:
            print("No embedding_shape for radiology embeddings, using raw buffers")
            for idx, row in tqdm(
                radiology_data.iterrows(),
                desc="Processing radiology embeddings",
                total=(len(radiology_data)),
            ):
                try:
                    # Check if embedding is a list
                    if isinstance(row["embedding"], list):
                        # Process list of embeddings
                        slice_embs = []
                        for e in row["embedding"]:
                            if e is not None:
                                slice_embs.append(np.frombuffer(e, dtype=np.float32))

                        if slice_embs:
                            # Average embeddings
                            avg_emb = np.mean(slice_embs, axis=0)
                            processed_embeddings.append(avg_emb)
                        else:
                            # Default empty embedding
                            processed_embeddings.append(
                                np.zeros(1000, dtype=np.float32)
                            )
                    else:
                        # Process single embedding
                        emb = np.frombuffer(row["embedding"], dtype=np.float32)
                        processed_embeddings.append(emb)
                except Exception as e:
                    print(f"Error processing radiology embedding at index {idx}: {e}")
                    if len(processed_embeddings) > 0:
                        processed_embeddings.append(
                            np.zeros_like(processed_embeddings[0])
                        )
                    else:
                        processed_embeddings.append(np.zeros(1000, dtype=np.float32))

        # Ensure all embeddings have the same shape
        from collections import Counter

        lengths = [len(emb) for emb in processed_embeddings]
        if len(set(lengths)) > 1:
            print(
                f"Found {len(set(lengths))} different embedding lengths: {set(lengths)}"
            )
            # Standardize to the most common length
            length_counts = Counter(lengths)
            most_common_length = length_counts.most_common(1)[0][0]
            print(f"Standardizing all embeddings to length {most_common_length}")

            standardized_embeddings = []
            for emb in processed_embeddings:
                if len(emb) < most_common_length:
                    # Pad with zeros
                    padded = np.zeros(most_common_length, dtype=np.float32)
                    padded[: len(emb)] = emb
                    standardized_embeddings.append(padded)
                else:
                    # Truncate
                    standardized_embeddings.append(emb[:most_common_length])

            processed_embeddings = standardized_embeddings

        embeddings_array = np.array(processed_embeddings)

        return {
            "X": embeddings_array,
            "y": radiology_data["project_id"].values,
            "patient_ids": radiology_data[radiology_id_col].values,
        }
    except Exception as e:
        print(f"Could not load radiology embeddings: {e}")
        return None


def process_molecular_embeddings(target_patient_ids=None):
    """
    Process molecular embeddings from the TCGA dataset.
    """
    print("Loading molecular embeddings...")
    try:
        molecular_data = load_dataset(
            "Lab-Rasool/TCGA", "molecular", split="senmo"
        ).to_pandas()
        print(f"Loaded {len(molecular_data)} molecular samples")

        # Get ID column
        molecular_id_col = get_id_column(molecular_data)

        # Filter by patient IDs if provided
        if target_patient_ids is not None:
            original_count = len(molecular_data)
            molecular_data = molecular_data[
                molecular_data[molecular_id_col].isin(target_patient_ids)
            ]
            print(
                f"After filtering by patient IDs: {len(molecular_data)} molecular samples (from {original_count})"
            )

        # Determine embedding column
        if "embedding" in molecular_data.columns:
            embedding_col = "embedding"
        elif "Embeddings" in molecular_data.columns:
            embedding_col = "Embeddings"
        else:
            # Try to find a column with 'embed' in the name
            embed_cols = [
                col for col in molecular_data.columns if "embed" in col.lower()
            ]
            if embed_cols:
                embedding_col = embed_cols[0]
                print(f"Using {embedding_col} for molecular embeddings")
            else:
                print("No embedding column found in molecular data")
                return None

        # Remove rows with null embeddings
        molecular_data = molecular_data.dropna(subset=[embedding_col])
        print(
            f"After removing null embeddings: {len(molecular_data)} molecular samples"
        )

        if len(molecular_data) == 0:
            print("No valid molecular embeddings found.")
            return None

        # Add project_id using patient_to_project_mapping
        if "patient_to_project_mapping" in globals():
            molecular_data["project_id"] = (
                molecular_data[molecular_id_col]
                .map(patient_to_project_mapping)
                .fillna("unknown")
            )
        else:
            print("Warning: No patient_to_project_mapping available for molecular data")
            molecular_data["project_id"] = "unknown"

        # Process embeddings
        processed_embeddings = []

        if "embedding_shape" in molecular_data.columns:
            print("Using embedding_shape for molecular embeddings")
            for idx, row in tqdm(
                molecular_data.iterrows(),
                desc="Processing molecular embeddings",
                total=len(molecular_data),
            ):
                try:
                    emb = np.frombuffer(row[embedding_col], dtype=np.float32)
                    shape = row["embedding_shape"]
                    reshaped_emb = emb.reshape(shape)
                    processed_embeddings.append(reshaped_emb)
                except Exception as e:
                    print(f"Error processing molecular embedding at index {idx}: {e}")
                    if len(processed_embeddings) > 0:
                        processed_embeddings.append(
                            np.zeros_like(processed_embeddings[0])
                        )
                    else:
                        processed_embeddings.append(np.zeros(48, dtype=np.float32))
        else:
            print("No embedding_shape for molecular embeddings, using raw buffers")
            for idx, row in tqdm(
                molecular_data.iterrows(),
                desc="Processing molecular embeddings",
                total=len(molecular_data),
            ):
                try:
                    emb = np.frombuffer(row[embedding_col], dtype=np.float32)
                    processed_embeddings.append(emb)
                except Exception as e:
                    print(f"Error processing molecular embedding at index {idx}: {e}")
                    if len(processed_embeddings) > 0:
                        processed_embeddings.append(
                            np.zeros_like(processed_embeddings[0])
                        )
                    else:
                        processed_embeddings.append(np.zeros(48, dtype=np.float32))

        # Ensure all embeddings have the same shape
        embeddings_array = np.array(processed_embeddings)

        return {
            "X": embeddings_array,
            "y": molecular_data["project_id"].values,
            "patient_ids": molecular_data[molecular_id_col].values,
        }
    except Exception as e:
        print(f"Could not load molecular embeddings: {e}")
        return None


def get_embeddings_path(modality, target_patient_ids=None):
    """
    Get the file path for saved embeddings.

    Args:
        modality (str): The modality type (e.g., 'clinical', 'pathology')
        target_patient_ids (list, optional): List of target patient IDs if filtering

    Returns:
        str: Path to the embeddings file
    """
    # Create embeddings directory if it doesn't exist
    embeddings_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "embeddings"
    )
    os.makedirs(embeddings_dir, exist_ok=True)

    target_patient_ids = None

    if target_patient_ids is None:
        return os.path.join(embeddings_dir, f"{modality}_embeddings.pkl")
    else:
        # Use hash of patient IDs for filtered embeddings
        patient_hash = hash(frozenset(target_patient_ids))
        return os.path.join(embeddings_dir, f"{modality}_filtered_{patient_hash}.pkl")


def save_embeddings(embeddings_data, modality, target_patient_ids=None):
    """
    Save embeddings data to disk.

    Args:
        embeddings_data (dict): Dictionary containing embeddings data
        modality (str): The modality type (e.g., 'clinical', 'pathology')
        target_patient_ids (list, optional): List of target patient IDs if filtering
    """
    try:
        embeddings_path = get_embeddings_path(modality, target_patient_ids)
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings_data, f)
        print(f"Saved {modality} embeddings to {embeddings_path}")
    except Exception as e:
        print(f"Error saving {modality} embeddings: {e}")


def load_saved_embeddings(modality, target_patient_ids=None):
    """
    Load embeddings data from disk if available.

    Args:
        modality (str): The modality type (e.g., 'clinical', 'pathology')
        target_patient_ids (list, optional): List of target patient IDs if filtering

    Returns:
        dict or None: Embeddings data if available, None otherwise
    """
    embeddings_path = get_embeddings_path(modality, target_patient_ids)

    if os.path.exists(embeddings_path):
        try:
            with open(embeddings_path, "rb") as f:
                embeddings_data = pickle.load(f)
            print(f"Loaded saved {modality} embeddings from {embeddings_path}")
            return embeddings_data
        except Exception as e:
            print(f"Error loading saved {modality} embeddings: {e}")

    return None


# Function to load embeddings for all modalities
def load_embeddings():
    """Load embeddings for each modality using robust processing methods."""
    print("Loading embeddings from Hugging Face datasets...")

    # Dictionary to store embeddings for each modality
    embeddings = {}

    # Load clinical embeddings
    clinical_data = process_clinical_embeddings()
    if clinical_data:
        embeddings["clinical"] = clinical_data

    # Get patient IDs from clinical data to filter other modalities
    target_patient_ids = set(clinical_data["patient_ids"]) if clinical_data else None
    print(
        f"Using {len(target_patient_ids) if target_patient_ids else 0} patient IDs for filtering other modalities"
    )

    # Load pathology embeddings
    pathology_data = process_pathology_embeddings(target_patient_ids)
    if pathology_data:
        embeddings["pathology"] = pathology_data

    # Load radiology embeddings
    radiology_data = process_radiology_embeddings(target_patient_ids)
    if radiology_data:
        embeddings["radiology"] = radiology_data

    # Load molecular embeddings
    molecular_data = process_molecular_embeddings(target_patient_ids)
    if molecular_data:
        embeddings["molecular"] = molecular_data

    # Create multimodal embeddings for patients with data in multiple modalities
    if "clinical" in embeddings:
        print("Creating multimodal embeddings...")

        # Get patient IDs from clinical data
        clinical_patients = set(embeddings["clinical"]["patient_ids"])

        # Count patients with data in multiple modalities
        modality_counts = {}
        for patient_id in clinical_patients:
            count = sum(
                1
                for modality in embeddings
                if patient_id in embeddings[modality]["patient_ids"]
            )
            modality_counts[patient_id] = count

        # Select patients with data in at least 2 modalities
        multimodal_patients = [
            pid for pid, count in modality_counts.items() if count >= 2
        ]

        if multimodal_patients:
            print(
                f"Creating multimodal embeddings for {len(multimodal_patients)} patients"
            )

            # Create mapping of patient_id to index for each modality
            patient_indices = {}
            for modality in embeddings:
                patient_indices[modality] = {
                    pid: idx
                    for idx, pid in enumerate(embeddings[modality]["patient_ids"])
                }

            # Create multimodal embeddings
            multimodal_X = []
            multimodal_y = []
            multimodal_patients = []

            for patient_id in multimodal_patients:
                patient_embeddings = []
                patient_label = None

                # Collect embeddings from each modality for this patient
                for modality in embeddings:
                    if patient_id in patient_indices[modality]:
                        idx = patient_indices[modality][patient_id]
                        patient_embeddings.append(embeddings[modality]["X"][idx])
                        # Use label from any available modality (should be the same across modalities)
                        if patient_label is None:
                            patient_label = embeddings[modality]["y"][idx]

                # Create multimodal embedding by scaling and concatenating
                if patient_embeddings:
                    # Scale each modality separately
                    scaled_embeddings = [
                        StandardScaler().fit_transform(emb.reshape(1, -1))[0]
                        for emb in patient_embeddings
                    ]
                    # Concatenate
                    multimodal_X.append(np.concatenate(scaled_embeddings))
                    multimodal_y.append(patient_label)
                    multimodal_patients.append(patient_id)

            if multimodal_X:
                embeddings["multimodal"] = {
                    "X": np.array(multimodal_X),
                    "y": np.array(multimodal_y),
                    "patient_ids": np.array(multimodal_patients),
                }
                print(
                    f"Created multimodal embeddings with shape: {embeddings['multimodal']['X'].shape}"
                )
        else:
            print(
                "No patients with data in multiple modalities found, skipping multimodal embeddings"
            )

    return embeddings
