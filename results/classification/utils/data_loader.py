import numpy as np
import os
import pickle
from .multimodal_fusion import create_multimodal_embeddings


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


def load_embeddings(fusion_method="concat"):
    """Load embeddings for all modalities."""
    print("Loading embeddings...")
    
    # Dictionary to store embeddings for each modality
    embeddings = {}
    
    # List of modalities to load
    modalities = ["clinical", "pathology", "radiology", "molecular", "wsi"]
    
    for modality in modalities:
        data = load_saved_embeddings(modality)
        if data:
            embeddings[modality] = data
            print(f"Loaded {modality} embeddings with shape: {data['X'].shape}")
    
    # Create multimodal embeddings if we have multiple modalities
    if len(embeddings) >= 2:
        multimodal_result = create_multimodal_embeddings(embeddings, fusion_method)
        if multimodal_result:
            embeddings["multimodal"] = multimodal_result
    
    return embeddings