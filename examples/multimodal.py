import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import os
import torch
from collections import Counter
from sklearn.manifold import TSNE
import warnings

# suppress all warnings
warnings.filterwarnings("ignore")

# Set the random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Constants
OUTPUT_DIR = "multimodal_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_id_column(df):
    """
    Determine the appropriate ID column in a dataframe.
    Check for common patient ID column names and return the first one found.
    """
    possible_id_columns = ["case_submitter_id", "PatientID", "patient_id", "ID"]

    for col in possible_id_columns:
        if col in df.columns:
            return col

    # If no known ID column is found, print all columns and raise an error
    print(f"Available columns: {df.columns.tolist()}")
    raise ValueError("No patient ID column found in dataframe.")


def process_clinical_embeddings():
    """
    Process clinical embeddings using the embedding_shape field if available
    """
    print("Loading clinical data...")
    clinical_data = load_dataset(
        "Lab-Rasool/TCGA", "clinical", split="gatortron"
    ).to_pandas()
    print(f"Loaded {len(clinical_data)} total clinical samples")

    # Print the available cancer types
    if "project_id" in clinical_data.columns:
        cancer_types = clinical_data["project_id"].unique()
        print(f"Available cancer types: {cancer_types}")
        print(f"Total number of cancer types: {len(cancer_types)}")
        print(f"Cancer type distribution: {clinical_data['project_id'].value_counts()}")

    # Remove rows with null embeddings
    clinical_data = clinical_data.dropna(subset=["embedding"])
    print(
        f"After removing null embeddings, {len(clinical_data)} clinical samples remain"
    )

    # Get ID column
    clinical_id_col = get_id_column(clinical_data)

    # Process embeddings
    processed_embeddings = []

    if "embedding_shape" in clinical_data.columns:
        print("Using embedding_shape for clinical embeddings")
        for idx, row in tqdm(
            clinical_data.iterrows(),
            desc="Processing clinical embeddings",
            total=len(clinical_data),
        ):
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                shape = row["embedding_shape"]
                reshaped_emb = emb.reshape(shape)
                processed_embeddings.append(reshaped_emb)
            except Exception as e:
                print(f"Error processing clinical embedding at index {idx}: {e}")
                # Try to find a valid embedding to determine shape
                if len(processed_embeddings) > 0:
                    processed_embeddings.append(np.zeros_like(processed_embeddings[0]))
                else:
                    processed_embeddings.append(np.zeros(1024, dtype=np.float32))
    else:
        print("No embedding_shape for clinical embeddings, using raw buffers")
        for idx, row in tqdm(
            clinical_data.iterrows(),
            desc="Processing clinical embeddings",
            total=len(clinical_data),
        ):
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                processed_embeddings.append(emb)
            except Exception as e:
                print(f"Error processing clinical embedding at index {idx}: {e}")
                if len(processed_embeddings) > 0:
                    processed_embeddings.append(np.zeros_like(processed_embeddings[0]))
                else:
                    processed_embeddings.append(np.zeros(1024, dtype=np.float32))

    # Create DataFrame with patient IDs and embeddings
    clinical_df = pd.DataFrame(
        {
            "patient_id": clinical_data[clinical_id_col],
            "cancer_type": clinical_data["project_id"],
            "modality": "clinical",
        }
    )
    clinical_df["embeddings"] = processed_embeddings

    return clinical_df


def process_pathology_embeddings(target_patient_ids=None):
    """
    Process pathology embeddings using the embedding_shape field if available
    """
    print("Loading pathology report data...")
    pathology_data = load_dataset(
        "Lab-Rasool/TCGA", "pathology_report", split="gatortron"
    ).to_pandas()
    print(f"Loaded {len(pathology_data)} total pathology samples")

    # Get ID column
    pathology_id_col = get_id_column(pathology_data)

    # Filter by patient IDs if provided
    if target_patient_ids:
        original_count = len(pathology_data)
        pathology_data = pathology_data[
            pathology_data[pathology_id_col].isin(target_patient_ids)
        ]
        print(
            f"After filtering by patient IDs, found {len(pathology_data)} pathology samples out of {original_count}"
        )

    # Remove rows with null embeddings
    pathology_data = pathology_data.dropna(subset=["embedding"])
    print(
        f"After removing null embeddings, {len(pathology_data)} pathology samples remain"
    )

    # Process embeddings
    processed_embeddings = []

    if "embedding_shape" in pathology_data.columns:
        print("Using embedding_shape for pathology embeddings")
        for idx, row in tqdm(
            pathology_data.iterrows(),
            desc="Processing pathology embeddings",
            total=len(pathology_data),
        ):
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                shape = row["embedding_shape"]
                reshaped_emb = emb.reshape(shape)
                processed_embeddings.append(reshaped_emb)
            except Exception as e:
                print(f"Error processing pathology embedding at index {idx}: {e}")
                if len(processed_embeddings) > 0:
                    processed_embeddings.append(np.zeros_like(processed_embeddings[0]))
                else:
                    processed_embeddings.append(np.zeros(1024, dtype=np.float32))
    else:
        print("No embedding_shape for pathology embeddings, using raw buffers")
        for idx, row in tqdm(
            pathology_data.iterrows(),
            desc="Processing pathology embeddings",
            total=len(pathology_data),
        ):
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                processed_embeddings.append(emb)
            except Exception as e:
                print(f"Error processing pathology embedding at index {idx}: {e}")
                if len(processed_embeddings) > 0:
                    processed_embeddings.append(np.zeros_like(processed_embeddings[0]))
                else:
                    processed_embeddings.append(np.zeros(1024, dtype=np.float32))

    # Create DataFrame with patient IDs and embeddings
    pathology_df = pd.DataFrame(
        {"patient_id": pathology_data[pathology_id_col], "modality": "pathology_report"}
    )
    pathology_df["embeddings"] = processed_embeddings

    return pathology_df


def process_radiology_embeddings(target_patient_ids=None):
    """
    Process radiology embeddings using the embedding_shape field to correctly reshape data
    """
    print("Loading radiology data...")
    radiology_data = load_dataset(
        "Lab-Rasool/TCGA", "radiology", split="radimagenet"
    ).to_pandas()
    print(f"Loaded {len(radiology_data)} total radiology samples")

    # Get ID column
    radiology_id_col = get_id_column(radiology_data)

    # Filter by patient IDs if provided
    if target_patient_ids:
        original_count = len(radiology_data)
        radiology_data = radiology_data[
            radiology_data[radiology_id_col].isin(target_patient_ids)
        ]
        print(
            f"After filtering by patient IDs, found {len(radiology_data)} radiology samples out of {original_count}"
        )

    # Remove rows with null embeddings
    radiology_data = radiology_data.dropna(subset=["embedding"])
    print(
        f"After removing null embeddings, {len(radiology_data)} radiology samples remain"
    )

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
                    processed_embeddings.append(np.zeros_like(processed_embeddings[0]))
                else:
                    processed_embeddings.append(np.zeros(1000, dtype=np.float32))
    else:
        print("No embedding_shape for radiology embeddings, using raw buffers")
        for idx, row in tqdm(
            radiology_data.iterrows(),
            desc="Processing radiology embeddings",
            total=len(radiology_data),
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
                        processed_embeddings.append(np.zeros(1000, dtype=np.float32))
                else:
                    # Process single embedding
                    emb = np.frombuffer(row["embedding"], dtype=np.float32)
                    processed_embeddings.append(emb)
            except Exception as e:
                print(f"Error processing radiology embedding at index {idx}: {e}")
                if len(processed_embeddings) > 0:
                    processed_embeddings.append(np.zeros_like(processed_embeddings[0]))
                else:
                    processed_embeddings.append(np.zeros(1000, dtype=np.float32))

    # Ensure all embeddings have the same length
    lengths = [len(emb) for emb in processed_embeddings]
    unique_lengths = set(lengths)

    if len(unique_lengths) > 1:
        print(
            f"Found {len(unique_lengths)} different embedding lengths: {unique_lengths}"
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

    # Create DataFrame with patient IDs and embeddings
    radiology_df = pd.DataFrame(
        {"patient_id": radiology_data[radiology_id_col], "modality": "radiology"}
    )

    # Add cancer_type if available
    if "project_id" in radiology_data.columns:
        radiology_df["cancer_type"] = radiology_data["project_id"]

    radiology_df["embeddings"] = processed_embeddings

    return radiology_df


def process_molecular_embeddings(target_patient_ids=None):
    """
    Process molecular embeddings using the embedding_shape field if available
    """
    print("Loading molecular data...")
    molecular_data = load_dataset(
        "Lab-Rasool/TCGA", "molecular", split="senmo"
    ).to_pandas()
    print(f"Loaded {len(molecular_data)} total molecular samples")
    print(f"Molecular data columns: {molecular_data.columns.tolist()}")

    # Get ID column
    molecular_id_col = get_id_column(molecular_data)

    # Filter by patient IDs if provided
    if target_patient_ids:
        original_count = len(molecular_data)
        molecular_data = molecular_data[
            molecular_data[molecular_id_col].isin(target_patient_ids)
        ]
        print(
            f"After filtering by patient IDs, found {len(molecular_data)} molecular samples out of {original_count}"
        )

    # Determine embedding column
    if "embedding" in molecular_data.columns:
        embedding_col = "embedding"
    elif "Embeddings" in molecular_data.columns:
        embedding_col = "Embeddings"
    else:
        # Try to find a column with 'embed' in the name
        embed_cols = [col for col in molecular_data.columns if "embed" in col.lower()]
        if embed_cols:
            embedding_col = embed_cols[0]
            print(f"Using {embedding_col} for molecular embeddings")
        else:
            print("No embedding column found in molecular data")
            return pd.DataFrame(
                columns=["patient_id", "modality", "cancer_type", "embeddings"]
            )

    # Remove rows with null embeddings
    molecular_data = molecular_data.dropna(subset=[embedding_col])
    print(
        f"After removing null embeddings, {len(molecular_data)} molecular samples remain"
    )

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
                    processed_embeddings.append(np.zeros_like(processed_embeddings[0]))
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
                    processed_embeddings.append(np.zeros_like(processed_embeddings[0]))
                else:
                    processed_embeddings.append(np.zeros(48, dtype=np.float32))

    # Create DataFrame with patient IDs and embeddings
    molecular_df = pd.DataFrame(
        {"patient_id": molecular_data[molecular_id_col], "modality": "molecular"}
    )

    # Add cancer_type if available
    if "project_id" in molecular_data.columns:
        molecular_df["cancer_type"] = molecular_data["project_id"]

    molecular_df["embeddings"] = processed_embeddings

    return molecular_df


def load_multimodal_data():
    """
    Load and process embeddings from all modalities
    """
    # Process clinical data first
    clinical_df = process_clinical_embeddings()

    # Get patient IDs from clinical data to filter other modalities
    target_patient_ids = set(clinical_df["patient_id"])
    print(f"Found {len(target_patient_ids)} unique patients in clinical data")

    # Process other modalities with patient ID filtering
    pathology_df = process_pathology_embeddings(target_patient_ids)
    radiology_df = process_radiology_embeddings(target_patient_ids)
    molecular_df = process_molecular_embeddings(target_patient_ids)

    # Add cancer type to modalities that don't have it
    # Get mapping from clinical data
    patient_to_cancer = dict(zip(clinical_df["patient_id"], clinical_df["cancer_type"]))

    # Update pathology cancer type
    if "cancer_type" not in pathology_df.columns:
        pathology_df["cancer_type"] = pathology_df["patient_id"].map(
            lambda x: patient_to_cancer.get(x, "Unknown")
        )

    # Update radiology cancer type if needed
    if "cancer_type" not in radiology_df.columns:
        radiology_df["cancer_type"] = radiology_df["patient_id"].map(
            lambda x: patient_to_cancer.get(x, "Unknown")
        )

    # Update molecular cancer type if needed
    if "cancer_type" not in molecular_df.columns:
        molecular_df["cancer_type"] = molecular_df["patient_id"].map(
            lambda x: patient_to_cancer.get(x, "Unknown")
        )

    # Print summary
    print("\nFinal summary:")
    print(f"Clinical data: {len(clinical_df)} samples")
    print(f"Pathology data: {len(pathology_df)} samples")
    print(f"Radiology data: {len(radiology_df)} samples")
    print(f"Molecular data: {len(molecular_df)} samples")

    # Print cancer type distribution
    print("\nCancer type distribution in clinical data:")
    print(clinical_df["cancer_type"].value_counts())

    return clinical_df, pathology_df, radiology_df, molecular_df


def align_patient_data(clinical_df, pathology_df, radiology_df, molecular_df):
    """
    Find patients that have data in all or multiple modalities and align their data
    """
    print("Aligning patient data across modalities...")

    # Find common patients across modalities
    clinical_patients = set(clinical_df["patient_id"])
    pathology_patients = set(pathology_df["patient_id"])
    radiology_patients = set(radiology_df["patient_id"])
    molecular_patients = set(molecular_df["patient_id"])

    print(f"Clinical patients: {len(clinical_patients)}")
    print(f"Pathology patients: {len(pathology_patients)}")
    print(f"Radiology patients: {len(radiology_patients)}")
    print(f"Molecular patients: {len(molecular_patients)}")

    # Find patients with data in multiple modalities
    print(
        f"Patients in clinical and pathology: {len(clinical_patients & pathology_patients)}"
    )
    print(
        f"Patients in clinical and radiology: {len(clinical_patients & radiology_patients)}"
    )
    print(
        f"Patients in clinical and molecular: {len(clinical_patients & molecular_patients)}"
    )

    # Find patients with data in all modalities
    all_modalities = (
        clinical_patients & pathology_patients & radiology_patients & molecular_patients
    )
    print(f"Found {len(all_modalities)} patients with data in all four modalities")

    # Choose patients for analysis
    if len(all_modalities) >= 50:  # If we have enough patients with all modalities
        print(f"Using {len(all_modalities)} patients with data in all modalities")
        common_patients = all_modalities
    else:
        # Find patients with data in at least 3 modalities
        at_least_three = (
            (clinical_patients & pathology_patients & radiology_patients)
            | (clinical_patients & pathology_patients & molecular_patients)
            | (clinical_patients & radiology_patients & molecular_patients)
        )

        if len(at_least_three) >= 50:
            print(
                f"Using {len(at_least_three)} patients with data in at least 3 modalities"
            )
            common_patients = at_least_three
        else:
            # Use patients with clinical data + at least one other modality
            clinical_plus = (
                (clinical_patients & pathology_patients)
                | (clinical_patients & radiology_patients)
                | (clinical_patients & molecular_patients)
            )

            if len(clinical_plus) >= 50:
                print(
                    f"Using {len(clinical_plus)} patients with clinical data + at least one other modality"
                )
                common_patients = clinical_plus
            else:
                # Fallback to just clinical patients
                print(f"Using {len(clinical_patients)} patients with clinical data")
                common_patients = clinical_patients

    # Get cancer type distribution for chosen patients
    patient_cancer_types = clinical_df[clinical_df["patient_id"].isin(common_patients)][
        "cancer_type"
    ].value_counts()
    print("\nCancer type distribution for selected patients:")
    print(patient_cancer_types)

    # Filter dataframes to keep only common patients
    clinical_filtered = clinical_df[clinical_df["patient_id"].isin(common_patients)]
    pathology_filtered = pathology_df[pathology_df["patient_id"].isin(common_patients)]
    radiology_filtered = radiology_df[radiology_df["patient_id"].isin(common_patients)]
    molecular_filtered = molecular_df[molecular_df["patient_id"].isin(common_patients)]

    # Create aligned dataframe with consistent patient ordering
    common_patients_list = sorted(list(common_patients))
    aligned_data = pd.DataFrame({"patient_id": common_patients_list})

    # Merge cancer type
    aligned_data = aligned_data.merge(
        clinical_filtered[["patient_id", "cancer_type"]], on="patient_id", how="left"
    )

    # Fill missing cancer types
    for df in [pathology_filtered, radiology_filtered, molecular_filtered]:
        if len(df) > 0:
            missing_mask = aligned_data["cancer_type"].isna()
            if missing_mask.any():
                for idx, row in aligned_data[missing_mask].iterrows():
                    patient_match = df[df["patient_id"] == row["patient_id"]]
                    if (
                        len(patient_match) > 0
                        and "cancer_type" in patient_match.columns
                    ):
                        aligned_data.loc[idx, "cancer_type"] = patient_match[
                            "cancer_type"
                        ].iloc[0]

    # Fill any still missing cancer types with "Unknown"
    missing_mask = aligned_data["cancer_type"].isna()
    if missing_mask.any():
        print(
            f"Still missing cancer type for {missing_mask.sum()} patients. Using 'Unknown'."
        )
        aligned_data.loc[missing_mask, "cancer_type"] = "Unknown"

    # Get embedding shapes for defaults
    clinical_embedding_shape = None
    pathology_embedding_shape = None
    radiology_embedding_shape = None
    molecular_embedding_shape = None

    if len(clinical_filtered) > 0:
        clinical_embedding_shape = clinical_filtered["embeddings"].iloc[0].shape
    if len(pathology_filtered) > 0:
        pathology_embedding_shape = pathology_filtered["embeddings"].iloc[0].shape
    if len(radiology_filtered) > 0:
        radiology_embedding_shape = radiology_filtered["embeddings"].iloc[0].shape
    if len(molecular_filtered) > 0:
        molecular_embedding_shape = molecular_filtered["embeddings"].iloc[0].shape

    # Use default shapes if not found
    if clinical_embedding_shape is None:
        clinical_embedding_shape = (1024,)
    if pathology_embedding_shape is None:
        pathology_embedding_shape = (1024,)
    if radiology_embedding_shape is None:
        radiology_embedding_shape = (1000,)
    if molecular_embedding_shape is None:
        molecular_embedding_shape = (48,)

    # Initialize embedding columns as object type to hold arrays
    aligned_data["clinical_embedding"] = None
    aligned_data["pathology_embedding"] = None
    aligned_data["radiology_embedding"] = None
    aligned_data["molecular_embedding"] = None

    # Add embeddings for each modality
    for patient_id in tqdm(common_patients_list, desc="Aligning embeddings"):
        # Get the index for this patient
        patient_idx = aligned_data.index[
            aligned_data["patient_id"] == patient_id
        ].tolist()[0]

        # Add clinical embeddings
        clinical_match = clinical_filtered[
            clinical_filtered["patient_id"] == patient_id
        ]
        if len(clinical_match) > 0:
            aligned_data.at[patient_idx, "clinical_embedding"] = clinical_match[
                "embeddings"
            ].values[0]
        else:
            aligned_data.at[patient_idx, "clinical_embedding"] = np.zeros(
                clinical_embedding_shape
            )

        # Add pathology embeddings
        pathology_match = pathology_filtered[
            pathology_filtered["patient_id"] == patient_id
        ]
        if len(pathology_match) > 0:
            aligned_data.at[patient_idx, "pathology_embedding"] = pathology_match[
                "embeddings"
            ].values[0]
        else:
            aligned_data.at[patient_idx, "pathology_embedding"] = np.zeros(
                pathology_embedding_shape
            )

        # Add radiology embeddings
        radiology_match = radiology_filtered[
            radiology_filtered["patient_id"] == patient_id
        ]
        if len(radiology_match) > 0:
            aligned_data.at[patient_idx, "radiology_embedding"] = radiology_match[
                "embeddings"
            ].values[0]
        else:
            aligned_data.at[patient_idx, "radiology_embedding"] = np.zeros(
                radiology_embedding_shape
            )

        # Add molecular embeddings
        molecular_match = molecular_filtered[
            molecular_filtered["patient_id"] == patient_id
        ]
        if len(molecular_match) > 0:
            aligned_data.at[patient_idx, "molecular_embedding"] = molecular_match[
                "embeddings"
            ].values[0]
        else:
            aligned_data.at[patient_idx, "molecular_embedding"] = np.zeros(
                molecular_embedding_shape
            )

    return aligned_data


def create_multimodal_embeddings(aligned_data):
    """
    Create integrated multimodal embeddings by concatenating modality embeddings
    """
    print("Creating multimodal embeddings...")

    # Handle multi-dimensional embeddings by extracting and flattening if needed
    clinical_embeddings = []
    pathology_embeddings = []
    radiology_embeddings = []
    molecular_embeddings = []

    for _, row in aligned_data.iterrows():
        # Process clinical embedding
        emb = row["clinical_embedding"]
        if emb is None:
            clinical_embeddings.append(np.zeros(1024))
        elif len(np.array(emb).shape) > 1:
            # If multi-dimensional, flatten or take mean
            emb_array = np.array(emb)
            if len(emb_array.shape) == 2:
                # Take mean along first dimension for 2D arrays
                clinical_embeddings.append(np.mean(emb_array, axis=0))
            else:
                # For higher dimensions, flatten to 1D
                clinical_embeddings.append(np.array(emb).flatten())
        else:
            clinical_embeddings.append(np.array(emb))

        # Process pathology embedding
        emb = row["pathology_embedding"]
        if emb is None:
            pathology_embeddings.append(np.zeros(1024))
        elif len(np.array(emb).shape) > 1:
            emb_array = np.array(emb)
            if len(emb_array.shape) == 2:
                pathology_embeddings.append(np.mean(emb_array, axis=0))
            else:
                pathology_embeddings.append(np.array(emb).flatten())
        else:
            pathology_embeddings.append(np.array(emb))

        # Process radiology embedding
        emb = row["radiology_embedding"]
        if emb is None:
            radiology_embeddings.append(np.zeros(1000))
        elif len(np.array(emb).shape) > 1:
            emb_array = np.array(emb)
            if len(emb_array.shape) == 2:
                radiology_embeddings.append(np.mean(emb_array, axis=0))
            else:
                radiology_embeddings.append(np.array(emb).flatten())
        else:
            radiology_embeddings.append(np.array(emb))

        # Process molecular embedding
        emb = row["molecular_embedding"]
        if emb is None:
            molecular_embeddings.append(np.zeros(48))
        elif len(np.array(emb).shape) > 1:
            emb_array = np.array(emb)
            if len(emb_array.shape) == 2:
                molecular_embeddings.append(np.mean(emb_array, axis=0))
            else:
                molecular_embeddings.append(np.array(emb).flatten())
        else:
            molecular_embeddings.append(np.array(emb))

    # Convert lists to numpy arrays
    clinical_embeddings = np.array(clinical_embeddings)
    pathology_embeddings = np.array(pathology_embeddings)
    radiology_embeddings = np.array(radiology_embeddings)
    molecular_embeddings = np.array(molecular_embeddings)

    # Print shapes for debugging
    print(f"Clinical embeddings shape: {clinical_embeddings.shape}")
    print(f"Pathology embeddings shape: {pathology_embeddings.shape}")
    print(f"Radiology embeddings shape: {radiology_embeddings.shape}")
    print(f"Molecular embeddings shape: {molecular_embeddings.shape}")

    # Normalize each modality separately
    clinical_scaled = StandardScaler().fit_transform(clinical_embeddings)
    pathology_scaled = StandardScaler().fit_transform(pathology_embeddings)
    radiology_scaled = StandardScaler().fit_transform(radiology_embeddings)
    molecular_scaled = StandardScaler().fit_transform(molecular_embeddings)

    # Create multimodal embeddings by concatenation
    multimodal_embeddings = np.hstack(
        [clinical_scaled, pathology_scaled, radiology_scaled, molecular_scaled]
    )

    print(f"Multimodal embeddings shape: {multimodal_embeddings.shape}")

    return multimodal_embeddings


def save_legend_separately(unique_projects, palette, markers, filename="legend"):
    """Save the legend as a separate file with a horizontal layout and improved marker visibility"""
    plt.rcParams["font.family"] = "Arial"

    # Create a wider figure for a horizontal legend
    fig, ax = plt.subplots(figsize=(12, 4))  # Made slightly larger for better spacing

    # Create dummy scatter points with improved visibility
    for i, project in enumerate(unique_projects):
        marker = markers[i % len(markers)]

        # Fix: Convert color string to RGB values first if it's a string
        if isinstance(palette[i], str):
            from matplotlib.colors import to_rgb

            base_color = np.array(to_rgb(palette[i]))
            edge_color = base_color * 0.7
        else:
            base_color = np.array(palette[i])
            edge_color = base_color * 0.7

        # Increase size and linewidth for better visibility, especially for thin markers
        marker_size = 80  # Increased from 50
        line_width = 1.5  # Increased from 0.5

        # For thin markers like '+', 'x', '|', '_', increase size even more
        if marker in ["+", "x", "|", "_", ",", ".", "1", "2", "3", "4"]:
            marker_size = 125
            line_width = 2.0

        ax.scatter(
            [0],
            [0],  # Position doesn't matter
            c=[palette[i]],
            label=str(project),
            s=marker_size,
            marker=marker,
            edgecolors=[edge_color],
            linewidths=line_width,
            alpha=1.0,  # Increased from 0.9 for better visibility
        )

    # Rest of the function remains the same
    # Hide the actual plot
    ax.set_axis_off()

    # Create the legend with horizontal orientation and improved spacing
    legend = ax.legend(
        bbox_to_anchor=(0.5, 0.5),
        loc="center",
        frameon=True,
        title_fontsize=14,
        ncol=min(5, len(unique_projects)),  # Make it multi-column for horizontal layout
        columnspacing=1.25,  # Increased from 1.0 for better spacing
        handletextpad=0.7,  # Increased from 0.5 for more space between marker and text
        handlelength=2.0,  # Explicitly set handle length
        markerscale=1.0,  # Scale up the markers in the legend
    )

    # Improve legend aesthetics
    legend.get_frame().set_linewidth(0.8)  # Add a border to the legend
    legend.get_frame().set_edgecolor("lightgray")

    # Save just the legend with padding
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(f"{filename}.svg", format="svg", bbox_inches=legend_bbox, dpi=600)
    fig.savefig(f"{filename}.pdf", format="pdf", bbox_inches=legend_bbox, dpi=600)

    print(
        f"Saved horizontal legend with improved marker visibility to {filename}.svg and {filename}.pdf"
    )
    return fig


def visualize_multimodal_integration_tsne(aligned_data, multimodal_embeddings):
    """
    Visualize the integration of multimodal data using t-SNE in 2D with separate
    layers for each cancer type for better editing in vector editors.
    Each plot is saved as an individual square figure.
    """
    print(
        "Visualizing multimodal integration with t-SNE in 2D using layered approach..."
    )

    # Set font to Arial for consistency
    plt.rcParams["font.family"] = "Arial"

    # Handle multi-dimensional embeddings by extracting and flattening if needed
    clinical_embeddings = []
    pathology_embeddings = []
    radiology_embeddings = []
    molecular_embeddings = []

    for _, row in aligned_data.iterrows():
        # Process clinical embedding
        emb = row["clinical_embedding"]
        if emb is None:
            clinical_embeddings.append(np.zeros(1024))
        elif len(np.array(emb).shape) > 1:
            # If multi-dimensional, flatten or take mean
            emb_array = np.array(emb)
            if len(emb_array.shape) == 2:
                # Take mean along first dimension for 2D arrays
                clinical_embeddings.append(np.mean(emb_array, axis=0))
            else:
                # For higher dimensions, flatten to 1D
                clinical_embeddings.append(np.array(emb).flatten())
        else:
            clinical_embeddings.append(np.array(emb))

        # Process pathology embedding
        emb = row["pathology_embedding"]
        if emb is None:
            pathology_embeddings.append(np.zeros(1024))
        elif len(np.array(emb).shape) > 1:
            emb_array = np.array(emb)
            if len(emb_array.shape) == 2:
                pathology_embeddings.append(np.mean(emb_array, axis=0))
            else:
                pathology_embeddings.append(np.array(emb).flatten())
        else:
            pathology_embeddings.append(np.array(emb))

        # Process radiology embedding
        emb = row["radiology_embedding"]
        if emb is None:
            radiology_embeddings.append(np.zeros(1000))
        elif len(np.array(emb).shape) > 1:
            emb_array = np.array(emb)
            if len(emb_array.shape) == 2:
                radiology_embeddings.append(np.mean(emb_array, axis=0))
            else:
                radiology_embeddings.append(np.array(emb).flatten())
        else:
            radiology_embeddings.append(np.array(emb))

        # Process molecular embedding
        emb = row["molecular_embedding"]
        if emb is None:
            molecular_embeddings.append(np.zeros(48))
        elif len(np.array(emb).shape) > 1:
            emb_array = np.array(emb)
            if len(emb_array.shape) == 2:
                molecular_embeddings.append(np.mean(emb_array, axis=0))
            else:
                molecular_embeddings.append(np.array(emb).flatten())
        else:
            molecular_embeddings.append(np.array(emb))

    # Convert lists to numpy arrays
    clinical_embeddings = np.array(clinical_embeddings)
    pathology_embeddings = np.array(pathology_embeddings)
    radiology_embeddings = np.array(radiology_embeddings)
    molecular_embeddings = np.array(molecular_embeddings)

    print(f"Clinical embeddings processed shape: {clinical_embeddings.shape}")
    print(f"Pathology embeddings processed shape: {pathology_embeddings.shape}")
    print(f"Radiology embeddings processed shape: {radiology_embeddings.shape}")
    print(f"Molecular embeddings processed shape: {molecular_embeddings.shape}")

    # Function to safely apply t-SNE with proper error handling
    def safe_tsne_transform(data, name, perplexity=30):
        # Apply StandardScaler, ensuring no NaNs
        scaled_data = StandardScaler().fit_transform(data)
        scaled_data = np.nan_to_num(scaled_data, nan=0.0)

        # Adjust perplexity if needed (t-SNE requires perplexity < n_samples)
        n_samples = scaled_data.shape[0]
        if perplexity >= n_samples:
            perplexity = max(5, n_samples // 4)
            print(f"Adjusted perplexity to {perplexity} for {name} embeddings")

        # Apply t-SNE with 2 components
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            random_state=42,
            n_jobs=-1,
        )
        transformed = tsne.fit_transform(scaled_data)
        return transformed

    print("Applying t-SNE to clinical embeddings...")
    clinical_tsne = safe_tsne_transform(clinical_embeddings, "clinical")

    print("Applying t-SNE to pathology embeddings...")
    pathology_tsne = safe_tsne_transform(pathology_embeddings, "pathology")

    print("Applying t-SNE to radiology embeddings...")
    radiology_tsne = safe_tsne_transform(radiology_embeddings, "radiology")

    print("Applying t-SNE to molecular embeddings...")
    molecular_tsne = safe_tsne_transform(molecular_embeddings, "molecular")

    print("Applying t-SNE to multimodal embeddings...")
    multimodal_tsne = safe_tsne_transform(multimodal_embeddings, "multimodal")

    # Get unique cancer types
    cancer_types = aligned_data["cancer_type"].unique()
    plot_cancer_types = cancer_types

    # Get unique color and marker combinations
    colors, markers = get_unique_color_marker_combos(len(plot_cancer_types))

    # Function to create and save individual square plot
    def create_and_save_square_plot(data, y_labels, title, modality_name):
        # Create a square figure
        fig, ax = plt.subplots(figsize=(8, 8), dpi=600)

        # Create layered plot with separate layers for each cancer type
        for i, cancer_type in enumerate(plot_cancer_types):
            mask = y_labels == cancer_type
            if np.sum(mask) > 0:
                # Create a darker version of the base color for edge
                from matplotlib.colors import to_rgb

                base_color = to_rgb(colors[i])
                edge_color = tuple(c * 0.7 for c in base_color)  # Make it darker

                # Use the assigned marker type
                marker = markers[i]

                # Determine point size and line width based on marker type
                marker_size = 100
                line_width = 1.0

                # Adjust size for specific markers (star)
                if marker in ["*", "p", "P"]:
                    marker_size = 120
                    line_width = 1.2

                ax.scatter(
                    data[mask, 0],
                    data[mask, 1],
                    c=[colors[i]],
                    label=str(cancer_type),
                    s=marker_size,
                    marker=marker,
                    edgecolors=[edge_color],
                    linewidths=line_width,
                    alpha=0.9,
                )

        # Turn off the axes
        ax.axis("off")
        # Use consistent axis limits across all plots
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)

        # Save the figure without legend
        plt.tight_layout()
        fig.savefig(
            f"{OUTPUT_DIR}/{modality_name}_tsne_2d.svg",
            format="svg",
            bbox_inches="tight",
        )
        fig.savefig(
            f"{OUTPUT_DIR}/{modality_name}_tsne_2d.pdf",
            format="pdf",
            bbox_inches="tight",
        )

        return fig

    # Create and save individual plots for each modality
    clinical_fig = create_and_save_square_plot(
        clinical_tsne, aligned_data["cancer_type"].values, "Clinical", "clinical"
    )

    pathology_fig = create_and_save_square_plot(
        pathology_tsne,
        aligned_data["cancer_type"].values,
        "Pathology Report",
        "pathology",
    )

    molecular_fig = create_and_save_square_plot(
        molecular_tsne, aligned_data["cancer_type"].values, "Molecular", "molecular"
    )

    radiology_fig = create_and_save_square_plot(
        radiology_tsne, aligned_data["cancer_type"].values, "Radiology", "radiology"
    )

    multimodal_fig = create_and_save_square_plot(
        multimodal_tsne,
        aligned_data["cancer_type"].values,
        "Multimodal Integration",
        "multimodal",
    )

    # Now save a separate square legend file
    legend_fig = save_legend_separately(
        unique_projects=plot_cancer_types,
        palette=colors,
        markers=markers,
        filename=f"{OUTPUT_DIR}/multimodal_tsne_legend",
    )

    # Save data to CSV for further analysis or visualization
    for modality, data in [
        ("clinical", clinical_tsne),
        ("pathology", pathology_tsne),
        ("radiology", radiology_tsne),
        ("molecular", molecular_tsne),
        ("multimodal", multimodal_tsne),
    ]:
        df = pd.DataFrame(
            {
                "x": data[:, 0],
                "cancer_type": aligned_data["cancer_type"].values,
                "y": data[:, 1],
            }
        )
        df.to_csv(f"{OUTPUT_DIR}/{modality}_tsne.csv", index=False)

    print(
        f"t-SNE visualizations saved to {OUTPUT_DIR} in vector (SVG/PDF) and CSV formats"
    )

    # Return all the figures
    return [
        clinical_fig,
        pathology_fig,
        molecular_fig,
        radiology_fig,
        multimodal_fig,
        legend_fig,
    ]


def save_embeddings_with_clinical_data(aligned_data, multimodal_embeddings):
    """
    Save embeddings vectors for each modality and multimodal embeddings
    along with clinical data for each patient for downstream model training.
    Includes ALL clinical information for each patient.
    """
    print("Saving embeddings with clinical data for downstream models...")

    # Create a directory for saved embeddings
    embeddings_dir = os.path.join(OUTPUT_DIR, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    # Fetch original clinical data to get all available clinical fields
    print("Loading full clinical dataset to extract detailed patient information...")
    full_clinical_data = load_dataset(
        "Lab-Rasool/TCGA", "clinical", split="gatortron"
    ).to_pandas()

    # Get ID column
    clinical_id_col = get_id_column(full_clinical_data)

    # Extract patients from our aligned dataset
    patient_ids = aligned_data["patient_id"].tolist()

    # Filter full clinical data to include only our patients
    detailed_clinical = full_clinical_data[
        full_clinical_data[clinical_id_col].isin(patient_ids)
    ]

    # Include ALL clinical fields EXCEPT embedding-related ones
    exclude_columns = ["embedding", "embedding_shape"]
    clinical_fields = [
        col
        for col in detailed_clinical.columns
        if col not in exclude_columns and not col.startswith("embed")
    ]

    print(f"Including {len(clinical_fields)} clinical features for downstream modeling")

    # First rename the ID column in detailed_clinical to "patient_id"
    # Make a copy to avoid SettingWithCopyWarning
    detailed_clinical = detailed_clinical.copy()

    # Create mapping from detailed_clinical to result_df
    result_df = pd.DataFrame({"patient_id": patient_ids})

    # Process one column at a time to handle any issues
    for col in clinical_fields:
        if col == clinical_id_col:
            # Skip the ID column as we'll handle it separately
            continue

        try:
            # Add this column to result_df, using the merge on ID
            col_df = detailed_clinical[[clinical_id_col, col]].rename(
                columns={clinical_id_col: "patient_id"}
            )
            result_df = result_df.merge(col_df, on="patient_id", how="left")
        except Exception as e:
            print(f"Warning: Could not process clinical column {col}: {e}")

    # Ensure cancer_type is included (from aligned_data if not in clinical data)
    if "cancer_type" not in result_df.columns and "project_id" in result_df.columns:
        result_df = result_df.rename(columns={"project_id": "cancer_type"})
    elif "cancer_type" not in result_df.columns:
        result_df["cancer_type"] = aligned_data["cancer_type"]

    # Fill missing values with reasonable defaults
    for col in result_df.columns:
        if col == "patient_id":
            continue
        try:
            if result_df[col].dtype == "object":
                result_df[col] = result_df[col].fillna("Unknown")
        except Exception as e:
            print(f"Warning: Could not fill missing values for {col}: {e}")
            pass

    # Save individual modality embeddings as numpy arrays
    modalities = [
        "clinical_embedding",
        "pathology_embedding",
        "radiology_embedding",
        "molecular_embedding",
    ]

    # Create embedding arrays for each modality
    for modality in modalities:
        # Extract embeddings and save as numpy arrays
        print(f"Processing {modality}...")
        try:
            embeddings_array = np.stack(aligned_data[modality].values)
            np.save(
                os.path.join(
                    embeddings_dir, f"{modality.split('_')[0]}_embeddings.npy"
                ),
                embeddings_array,
            )

            # Add a pointer to the saved file in the dataframe
            result_df[f"{modality}_file"] = f"{modality.split('_')[0]}_embeddings.npy"
        except Exception as e:
            print(f"Warning: Error processing {modality}: {e}")

    # Save multimodal embeddings
    print("Saving multimodal embeddings...")
    np.save(
        os.path.join(embeddings_dir, "multimodal_embeddings.npy"), multimodal_embeddings
    )
    result_df["multimodal_embedding_file"] = "multimodal_embeddings.npy"

    # Save the metadata CSV with all clinical information
    result_df.to_csv(
        os.path.join(embeddings_dir, "patient_data_with_embeddings.csv"), index=False
    )

    # Save a compact version with patient IDs and indices for easy lookup
    patient_index_df = pd.DataFrame(
        {
            "index": range(len(aligned_data)),
            "patient_id": aligned_data["patient_id"],
            "cancer_type": aligned_data["cancer_type"],
        }
    )
    patient_index_df.to_csv(
        os.path.join(embeddings_dir, "patient_index_mapping.csv"), index=False
    )

    # Save a README file explaining the data structure
    with open(os.path.join(embeddings_dir, "README.md"), "w") as f:
        f.write("# Multimodal Cancer Data Embeddings\n\n")
        f.write("## Data Structure\n\n")
        f.write(
            "- **patient_data_with_embeddings.csv**: Contains ALL patient clinical data and references to embedding files\n"
        )
        f.write(
            "- **patient_index_mapping.csv**: Simple mapping between array indices and patient IDs\n"
        )
        f.write(
            "- **[modality]_embeddings.npy**: NumPy arrays containing embeddings for each modality\n"
        )
        f.write(
            "- **multimodal_embeddings.npy**: Combined multimodal embeddings created by concatenating scaled modality embeddings\n\n"
        )
        f.write("## Dataset Statistics\n\n")
        f.write(f"- Total patients: {len(result_df)}\n")
        f.write(
            f"- Cancer types: {', '.join(sorted(result_df['cancer_type'].unique()))}\n"
        )
        f.write("- Embedding dimensions:\n")
        for modality in modalities:
            try:
                shape = np.stack(aligned_data[modality].values).shape[1]
                f.write(f"  - {modality.split('_')[0]}: {shape}\n")
            except Exception as e:
                print(f"Error processing {modality} shape: {e}")
                f.write(f"  - {modality.split('_')[0]}: unknown\n")

        f.write(f"  - multimodal: {multimodal_embeddings.shape[1]}\n")

    print(f"Saved all embeddings and metadata to {embeddings_dir}/")
    print(f"- Multimodal embeddings shape: {multimodal_embeddings.shape}")
    print(f"- Number of patients: {len(result_df)}")
    print(
        f"- Clinical features included: {len(result_df.columns) - 1}"
    )  # Excluding patient_id
    print(f"- Available cancer types: {sorted(result_df['cancer_type'].unique())}")

    return embeddings_dir


def get_unique_color_marker_combos(num_types):
    """
    Generate unique color and marker combinations using predefined sets
    of visually distinctive colors and markers.
    """
    # Define these constants at the top level of the file for consistency
    CUSTOM_COLORS = ["#FF6060", "#6CC2F5", "#FF9A60", "#FF60B3", "#C260FF", "#60FFA0"]

    # square, circle, star, triangle up, pentagon, triangle down
    CUSTOM_MARKERS = ["s", "o", "*", "^", "p", "v"]
    # Define visually distinct colors (more vibrant than current muted ones)
    colors = CUSTOM_COLORS  # Use the globally defined colors

    # Define distinct marker shapes
    markers = CUSTOM_MARKERS  # Use the globally defined markers

    # Generate combinations
    result_colors = []
    result_markers = []

    # For combinations up to colors × markers
    needed_combos = min(num_types, len(colors) * len(markers))

    for i in range(needed_combos):
        color_idx = i % len(colors)
        marker_idx = (i // len(colors)) % len(markers)

        result_colors.append(colors[color_idx])
        result_markers.append(markers[marker_idx])

    # If we need more than the basic combinations
    if num_types > len(colors) * len(markers):
        # Generate additional combinations with variations
        from matplotlib.colors import to_rgb, rgb2hex

        for i in range(len(colors) * len(markers), num_types):
            color_idx = i % len(colors)
            marker_idx = (i // len(colors)) % len(markers)
            variation = i // (len(colors) * len(markers))

            # Use color with slight variation
            base_color = to_rgb(colors[color_idx])

            # Create variations: lighter, darker, more saturated
            if variation == 0:
                # Slightly lighter
                color = tuple(min(1.0, c * 1.2) for c in base_color)
            elif variation == 1:
                # Slightly darker
                color = tuple(c * 0.8 for c in base_color)
            else:
                # Adjust saturation (move closer or further from gray)
                gray = sum(base_color) / 3
                color = tuple(gray + (c - gray) * 1.2 for c in base_color)
                # Clamp values
                color = tuple(max(0, min(1, c)) for c in color)

            result_colors.append(rgb2hex(color))
            result_markers.append(markers[marker_idx])

    return result_colors, result_markers


def perform_clustering_analysis(
    aligned_data,
    multimodal_embeddings,
    clinical_embeddings,
    molecular_embeddings,
    pathology_embeddings,
    radiology_embeddings,
):
    """
    Perform comprehensive clustering analysis on different embedding modalities and
    calculate metrics for paper placeholders
    """
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    from collections import Counter
    import numpy as np
    import pandas as pd
    import os

    print("Performing clustering analysis...")

    # Get cancer type labels
    cancer_types = aligned_data["cancer_type"].values

    # Get unique cancer types
    unique_cancer_types = sorted(list(set(cancer_types)))
    cancer_type_to_idx = {cancer: idx for idx, cancer in enumerate(unique_cancer_types)}

    # Convert cancer types to numerical labels for clustering evaluation
    numeric_labels = np.array([cancer_type_to_idx[cancer] for cancer in cancer_types])

    # Create a mapping for specific cancer types mentioned in the paper
    kirc_mask = cancer_types == "TCGA-KIRC"
    ov_mask = cancer_types == "TCGA-OV"
    brca_mask = cancer_types == "TCGA-BRCA"

    # Ensure all embeddings are properly scaled and flattened if needed
    scaled_embeddings = {}

    # Function to safely flatten and scale embeddings
    def process_embedding_for_clustering(embeddings, name):
        try:
            # Check the shape of embeddings
            if len(embeddings.shape) > 2:
                print(
                    f"Warning: {name} embeddings have shape {embeddings.shape}, flattening all dimensions except the first"
                )
                # Flatten all dimensions except the first one (samples)
                n_samples = embeddings.shape[0]
                flattened = embeddings.reshape(n_samples, -1)
                print(f"Flattened {name} embeddings to shape {flattened.shape}")
                embeddings = flattened

            # Check for NaN or infinite values
            if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                print(
                    f"Warning: {name} embeddings contain NaN or infinite values, replacing with zeros"
                )
                embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply scaling
            scaled = StandardScaler().fit_transform(embeddings)
            return scaled
        except Exception as e:
            print(f"Error processing {name} embeddings: {e}")
            # Return original embeddings if processing fails
            # Make sure it's 2D
            if len(embeddings.shape) == 1:
                return embeddings.reshape(-1, 1)
            return embeddings

    # Process each embedding type
    scaled_embeddings["clinical"] = process_embedding_for_clustering(
        clinical_embeddings, "clinical"
    )
    scaled_embeddings["molecular"] = process_embedding_for_clustering(
        molecular_embeddings, "molecular"
    )
    scaled_embeddings["pathology"] = process_embedding_for_clustering(
        pathology_embeddings, "pathology"
    )
    scaled_embeddings["radiology"] = process_embedding_for_clustering(
        radiology_embeddings, "radiology"
    )
    scaled_embeddings["multimodal"] = process_embedding_for_clustering(
        multimodal_embeddings, "multimodal"
    )

    # 1. Calculate silhouette scores for each embedding space
    silhouette_scores = {}
    for modality, embeddings in scaled_embeddings.items():
        # Skip if too few samples or NaN values
        if len(embeddings) <= len(unique_cancer_types) or np.isnan(embeddings).any():
            print(
                f"Warning: Cannot calculate silhouette score for {modality} - insufficient data or NaN values"
            )
            silhouette_scores[modality] = 0
            continue

        try:
            score = silhouette_score(embeddings, numeric_labels)
            silhouette_scores[modality] = score
            print(f"Silhouette score for {modality}: {score:.4f}")
        except Exception as e:
            print(f"Error calculating silhouette score for {modality}: {e}")
            silhouette_scores[modality] = 0

    # 2. Calculate cluster metrics using KMeans
    kmeans_results = {}
    cluster_assignments = {}

    for modality, embeddings in scaled_embeddings.items():
        try:
            # Apply KMeans clustering with number of clusters = number of cancer types
            kmeans = KMeans(
                n_clusters=len(unique_cancer_types), random_state=42, n_init=10
            )
            clusters = kmeans.fit_predict(embeddings)
            cluster_assignments[modality] = clusters

            # Store KMeans model for further analysis
            kmeans_results[modality] = kmeans
        except Exception as e:
            print(f"Error in KMeans clustering for {modality}: {e}")
            cluster_assignments[modality] = np.zeros(len(numeric_labels))

    # 3. Calculate inter-cluster and intra-cluster distances
    distance_metrics = {}

    for modality, embeddings in scaled_embeddings.items():
        distance_metrics[modality] = {"intra_cluster": {}, "inter_cluster": {}}

        # Calculate intra-cluster distances for specific cancer types
        for cancer_type, mask in [
            ("KIRC", kirc_mask),
            ("OV", ov_mask),
            ("BRCA", brca_mask),
        ]:
            if sum(mask) >= 2:  # Need at least 2 samples to calculate distances
                # Get embeddings for this cancer type
                cancer_embeddings = embeddings[mask]

                # Calculate pairwise distances within cluster
                nn = NearestNeighbors(n_neighbors=min(len(cancer_embeddings), 5))
                nn.fit(cancer_embeddings)
                distances, _ = nn.kneighbors(cancer_embeddings)

                # Average distance to nearest neighbors (excluding self)
                intra_distance = (
                    np.mean(distances[:, 1:]) if distances.shape[1] > 1 else 0
                )
                distance_metrics[modality]["intra_cluster"][cancer_type] = (
                    intra_distance
                )

        # Calculate inter-cluster distances (distance between cluster centroids)
        if modality in kmeans_results:
            centroids = kmeans_results[modality].cluster_centers_

            for i, cancer_type_i in enumerate(unique_cancer_types):
                for j, cancer_type_j in enumerate(unique_cancer_types):
                    if i < j:  # Only calculate once for each pair
                        centroid_i = centroids[i]
                        centroid_j = centroids[j]
                        distance = np.linalg.norm(centroid_i - centroid_j)

                        # Store distance between cancer types
                        key = f"{cancer_type_i}_{cancer_type_j}"
                        distance_metrics[modality]["inter_cluster"][key] = distance

    # 4. Calculate misclassification rates
    misclassification_rates = {}

    for modality, clusters in cluster_assignments.items():
        misclassification_rates[modality] = {}

        # Create mapping from cluster ID to majority cancer type
        cluster_to_cancer = {}
        for cluster_id in range(len(unique_cancer_types)):
            mask = clusters == cluster_id
            if sum(mask) > 0:
                cluster_cancers = cancer_types[mask]
                counter = Counter(cluster_cancers)
                majority_cancer = counter.most_common(1)[0][0]
                cluster_to_cancer[cluster_id] = majority_cancer

        # Calculate misclassification rate for each cancer type
        for cancer_type, mask in [
            ("KIRC", kirc_mask),
            ("OV", ov_mask),
            ("BRCA", brca_mask),
        ]:
            if sum(mask) > 0:
                # Get assigned clusters for this cancer type
                cancer_clusters = clusters[mask]

                # Count correctly classified samples
                correct = 0
                for cluster_id in cancer_clusters:
                    if (
                        cluster_id in cluster_to_cancer
                        and cluster_to_cancer[cluster_id] == f"TCGA-{cancer_type}"
                    ):
                        correct += 1

                # Calculate misclassification rate
                misc_rate = 100 * (1 - correct / sum(mask))
                misclassification_rates[modality][cancer_type] = misc_rate

    # 5. Calculate Normalized Mutual Information (NMI)
    nmi_scores = {}

    for modality, clusters in cluster_assignments.items():
        try:
            # Calculate NMI between cluster assignments and true cancer types
            nmi = normalized_mutual_info_score(numeric_labels, clusters)
            nmi_scores[modality] = nmi
            print(f"NMI score for {modality}: {nmi:.4f}")
        except Exception as e:
            print(f"Error calculating NMI for {modality}: {e}")
            nmi_scores[modality] = 0

    # Save all metrics to CSV
    metrics_dir = os.path.join(OUTPUT_DIR, "clustering_metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Save silhouette scores
    pd.DataFrame(
        silhouette_scores.items(), columns=["Modality", "Silhouette_Score"]
    ).to_csv(os.path.join(metrics_dir, "silhouette_scores.csv"), index=False)

    # Save NMI scores
    pd.DataFrame(nmi_scores.items(), columns=["Modality", "NMI_Score"]).to_csv(
        os.path.join(metrics_dir, "nmi_scores.csv"), index=False
    )

    # Save misclassification rates
    misc_df = pd.DataFrame(
        columns=["Modality", "Cancer_Type", "Misclassification_Rate"]
    )
    for modality, rates in misclassification_rates.items():
        for cancer_type, rate in rates.items():
            misc_df = pd.concat(
                [
                    misc_df,
                    pd.DataFrame(
                        {
                            "Modality": [modality],
                            "Cancer_Type": [cancer_type],
                            "Misclassification_Rate": [rate],
                        }
                    ),
                ],
                ignore_index=True,
            )
    misc_df.to_csv(
        os.path.join(metrics_dir, "misclassification_rates.csv"), index=False
    )

    # Save distance metrics
    intra_df = pd.DataFrame(
        columns=["Modality", "Cancer_Type", "Intra_Cluster_Distance"]
    )
    for modality, metrics in distance_metrics.items():
        for cancer_type, distance in metrics["intra_cluster"].items():
            intra_df = pd.concat(
                [
                    intra_df,
                    pd.DataFrame(
                        {
                            "Modality": [modality],
                            "Cancer_Type": [cancer_type],
                            "Intra_Cluster_Distance": [distance],
                        }
                    ),
                ],
                ignore_index=True,
            )
    intra_df.to_csv(
        os.path.join(metrics_dir, "intra_cluster_distances.csv"), index=False
    )

    inter_df = pd.DataFrame(
        columns=["Modality", "Cancer_Type_Pair", "Inter_Cluster_Distance"]
    )
    for modality, metrics in distance_metrics.items():
        for cancer_pair, distance in metrics["inter_cluster"].items():
            inter_df = pd.concat(
                [
                    inter_df,
                    pd.DataFrame(
                        {
                            "Modality": [modality],
                            "Cancer_Type_Pair": [cancer_pair],
                            "Inter_Cluster_Distance": [distance],
                        }
                    ),
                ],
                ignore_index=True,
            )
    inter_df.to_csv(
        os.path.join(metrics_dir, "inter_cluster_distances.csv"), index=False
    )

    # Return a dictionary with all the metrics needed for the paper
    paper_metrics = {
        "silhouette": silhouette_scores,
        "nmi": nmi_scores,
        "misclassification": {
            "clinical": {
                "OV": misclassification_rates.get("clinical", {}).get("OV", 0),
                "BRCA": misclassification_rates.get("clinical", {}).get("BRCA", 0),
            },
            "multimodal": {
                "OV": misclassification_rates.get("multimodal", {}).get("OV", 0),
                "BRCA": misclassification_rates.get("multimodal", {}).get("BRCA", 0),
            },
        },
        "distances": {
            "radiology": {
                "inter_KIRC": np.mean(
                    [
                        v
                        for k, v in distance_metrics.get("radiology", {})
                        .get("inter_cluster", {})
                        .items()
                        if "KIRC" in k
                    ]
                )
            },
            "multimodal": {
                "intra_KIRC": distance_metrics.get("multimodal", {})
                .get("intra_cluster", {})
                .get("KIRC", 0),
                "inter_KIRC": np.mean(
                    [
                        v
                        for k, v in distance_metrics.get("multimodal", {})
                        .get("inter_cluster", {})
                        .items()
                        if "KIRC" in k
                    ]
                ),
            },
        },
    }

    # Print the metrics for the paper
    print("\nMetrics for paper placeholders:")
    print(
        f"Silhouette scores: clinical = {paper_metrics['silhouette'].get('clinical', 0):.4f}, "
        f"molecular = {paper_metrics['silhouette'].get('molecular', 0):.4f}, "
        f"pathology = {paper_metrics['silhouette'].get('pathology', 0):.4f}, "
        f"radiology = {paper_metrics['silhouette'].get('radiology', 0):.4f}, "
        f"multimodal = {paper_metrics['silhouette'].get('multimodal', 0):.4f}"
    )

    print(
        f"NMI scores: clinical = {paper_metrics['nmi'].get('clinical', 0):.4f}, "
        f"molecular = {paper_metrics['nmi'].get('molecular', 0):.4f}, "
        f"pathology = {paper_metrics['nmi'].get('pathology', 0):.4f}, "
        f"radiology = {paper_metrics['nmi'].get('radiology', 0):.4f}, "
        f"multimodal = {paper_metrics['nmi'].get('multimodal', 0):.4f}"
    )

    print(
        f"Misclassification rates: clinical OV = {paper_metrics['misclassification']['clinical']['OV']:.2f}%, "
        f"clinical BRCA = {paper_metrics['misclassification']['clinical']['BRCA']:.2f}%, "
        f"multimodal OV = {paper_metrics['misclassification']['multimodal']['OV']:.2f}%, "
        f"multimodal BRCA = {paper_metrics['misclassification']['multimodal']['BRCA']:.2f}%"
    )

    print(
        f"KIRC inter-cluster distance in radiology = {paper_metrics['distances']['radiology']['inter_KIRC']:.4f}"
    )
    print(
        f"KIRC intra-cluster distance in multimodal = {paper_metrics['distances']['multimodal']['intra_KIRC']:.4f}"
    )
    print(
        f"KIRC inter-cluster distance in multimodal = {paper_metrics['distances']['multimodal']['inter_KIRC']:.4f}"
    )

    # Calculate improvement percentage for silhouette score
    multimodal_silhouette = paper_metrics["silhouette"].get("multimodal", 0)
    next_best_silhouette = max(
        [v for k, v in paper_metrics["silhouette"].items() if k != "multimodal"] or [0]
    )
    if next_best_silhouette > 0:
        improvement_percent = (
            100 * (multimodal_silhouette - next_best_silhouette) / next_best_silhouette
        )
        print(f"Silhouette score improvement: {improvement_percent:.2f}%")
    else:
        print("Could not calculate improvement percentage - no valid comparison")

    return paper_metrics


def main():
    """
    Main function to run the multimodal data integration analysis
    """
    # Load data from all modalities
    clinical_df, pathology_df, radiology_df, molecular_df = load_multimodal_data()

    # Align patient data across modalities
    aligned_data = align_patient_data(
        clinical_df, pathology_df, radiology_df, molecular_df
    )

    # Create multimodal embeddings
    multimodal_embeddings = create_multimodal_embeddings(aligned_data)

    # Extract individual modality embeddings for clustering analysis
    # Use a safe extraction method that handles potential issues
    def safely_extract_embeddings(data_column, default_shape=(1,)):
        try:
            # First attempt to convert to a list of arrays
            embeddings_list = list(data_column.values)

            # Check if any embeddings are None
            embeddings_list = [
                e if e is not None else np.zeros(default_shape) for e in embeddings_list
            ]

            # Convert list of arrays to a single array
            embeddings_array = np.stack(embeddings_list)

            print(f"Extracted embeddings with shape: {embeddings_array.shape}")
            return embeddings_array
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            # Return a dummy array if extraction fails
            return np.zeros((len(data_column), *default_shape))

    # Extract embeddings for each modality
    clinical_embeddings = safely_extract_embeddings(aligned_data["clinical_embedding"])
    pathology_embeddings = safely_extract_embeddings(
        aligned_data["pathology_embedding"]
    )
    radiology_embeddings = safely_extract_embeddings(
        aligned_data["radiology_embedding"]
    )
    molecular_embeddings = safely_extract_embeddings(
        aligned_data["molecular_embedding"]
    )

    # Perform clustering analysis
    clustering_metrics = perform_clustering_analysis(
        aligned_data,
        multimodal_embeddings,
        clinical_embeddings,
        molecular_embeddings,
        pathology_embeddings,
        radiology_embeddings,
    )

    # Save the clustering metrics to a JSON file for easy reference
    import json

    # Convert numpy values to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    serializable_metrics = convert_to_serializable(clustering_metrics)

    with open(os.path.join(OUTPUT_DIR, "paper_metrics.json"), "w") as f:
        json.dump(serializable_metrics, f, indent=2)

    # Save embeddings and clinical data for downstream models
    embeddings_dir = save_embeddings_with_clinical_data(
        aligned_data, multimodal_embeddings
    )
    print(f"Embeddings and clinical data saved to {embeddings_dir}")

    # Visualize multimodal integration with t-SNE
    visualize_multimodal_integration_tsne(aligned_data, multimodal_embeddings)

    print(f"Multimodal integration analysis complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
