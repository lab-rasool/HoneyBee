import numpy as np
import os
import pickle


def get_embeddings_path(modality):
    """
    Get the file path for saved embeddings.
    
    Args:
        modality (str): The modality type (e.g., 'clinical', 'pathology')
        
    Returns:
        str: Path to the embeddings file
    """
    # Use the embeddings from the shared data directory
    embeddings_dir = "/mnt/f/Projects/HoneyBee/results/shared_data/embeddings"
    return os.path.join(embeddings_dir, f"{modality}_embeddings.pkl")


def load_saved_embeddings(modality):
    """
    Load embeddings data from disk if available.
    
    Args:
        modality (str): The modality type (e.g., 'clinical', 'pathology')
        
    Returns:
        dict or None: Embeddings data if available, None otherwise
    """
    embeddings_path = get_embeddings_path(modality)
    
    if os.path.exists(embeddings_path):
        try:
            with open(embeddings_path, "rb") as f:
                embeddings_data = pickle.load(f)
            print(f"Loaded saved {modality} embeddings from {embeddings_path}")
            return embeddings_data
        except Exception as e:
            print(f"Error loading saved {modality} embeddings: {e}")
    
    return None


def load_embeddings():
    """Load embeddings for all modalities."""
    print("Loading embeddings for retrieval...")
    
    # Dictionary to store embeddings for each modality
    embeddings = {}
    
    # List of modalities to load
    modalities = ["clinical", "pathology", "radiology", "molecular"]
    
    for modality in modalities:
        data = load_saved_embeddings(modality)
        if data:
            embeddings[modality] = data
            print(f"Loaded {modality} embeddings with shape: {data['X'].shape}")
    
    # Create multimodal embeddings if we have multiple modalities
    if len(embeddings) >= 2 and "clinical" in embeddings:
        print("Creating multimodal embeddings...")
        
        # Get patient IDs from clinical data as reference
        clinical_patients = set(embeddings["clinical"]["patient_ids"])
        
        # Create mapping of patient_id to index for each modality
        patient_indices = {}
        for modality in embeddings:
            patient_indices[modality] = {
                pid: idx
                for idx, pid in enumerate(embeddings[modality]["patient_ids"])
            }
        
        # Find patients with data in multiple modalities
        multimodal_patients = []
        for patient_id in clinical_patients:
            count = sum(
                1
                for modality in embeddings
                if patient_id in patient_indices[modality]
            )
            if count >= 2:
                multimodal_patients.append(patient_id)
        
        if multimodal_patients:
            print(f"Creating multimodal embeddings for {len(multimodal_patients)} patients")
            
            # Create multimodal embeddings
            multimodal_X = []
            multimodal_y = []
            multimodal_patient_ids = []
            
            for patient_id in multimodal_patients:
                patient_embeddings = []
                patient_label = None
                
                # Collect embeddings from each modality for this patient
                for modality in embeddings:
                    if patient_id in patient_indices[modality]:
                        idx = patient_indices[modality][patient_id]
                        # Flatten the embedding if necessary
                        emb = embeddings[modality]["X"][idx]
                        if len(emb.shape) > 1:
                            emb = emb.flatten()
                        patient_embeddings.append(emb)
                        # Use label from any available modality
                        if patient_label is None:
                            patient_label = embeddings[modality]["y"][idx]
                
                # Concatenate embeddings
                if patient_embeddings:
                    multimodal_X.append(np.concatenate(patient_embeddings))
                    multimodal_y.append(patient_label)
                    multimodal_patient_ids.append(patient_id)
            
            if multimodal_X:
                embeddings["multimodal"] = {
                    "X": np.array(multimodal_X),
                    "y": np.array(multimodal_y),
                    "patient_ids": np.array(multimodal_patient_ids),
                }
                print(f"Created multimodal embeddings with shape: {embeddings['multimodal']['X'].shape}")
    
    return embeddings