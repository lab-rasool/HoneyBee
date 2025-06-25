import numpy as np
from itertools import product


def kronecker_product_fusion(embeddings_list):
    """
    Fuse embeddings using Kronecker Product (KP).
    Uses a simplified version that creates pairwise interactions between modalities.
    
    Args:
        embeddings_list: List of embedding arrays for available modalities
        
    Returns:
        Fused embedding using KP with fixed dimension
    """
    if len(embeddings_list) == 0:
        raise ValueError("No embeddings provided")
    
    if len(embeddings_list) == 1:
        return embeddings_list[0]
    
    # Concatenate all embeddings first
    concatenated = np.concatenate(embeddings_list)
    
    # Create pairwise interactions between modalities
    interactions = []
    n_interactions_per_pair = 100  # Fixed number of interactions per modality pair
    
    for i in range(len(embeddings_list)):
        for j in range(i + 1, len(embeddings_list)):
            emb1 = embeddings_list[i]
            emb2 = embeddings_list[j]
            
            # Sample indices for interactions
            n_samples = min(n_interactions_per_pair, len(emb1), len(emb2))
            
            if n_samples > 0:
                idx1 = np.random.choice(len(emb1), size=n_samples, replace=False)
                idx2 = np.random.choice(len(emb2), size=n_samples, replace=False)
                
                # Create interaction features
                interaction = emb1[idx1] * emb2[idx2]
                interactions.append(interaction)
    
    # Combine original features with interactions
    if interactions:
        all_interactions = np.concatenate(interactions)
        result = np.concatenate([concatenated, all_interactions])
    else:
        result = concatenated
    
    return result


def mean_pool_fusion(embeddings_list):
    """
    Fuse embeddings using Mean Pooling.
    
    Args:
        embeddings_list: List of embedding arrays for available modalities
        
    Returns:
        Fused embedding using mean pooling
    """
    if len(embeddings_list) == 0:
        raise ValueError("No embeddings provided")
    
    # Ensure all embeddings have the same dimension for mean pooling
    # We'll project them to a common dimension if needed
    target_dim = max(len(emb) for emb in embeddings_list)
    
    projected_embeddings = []
    for emb in embeddings_list:
        if len(emb) < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:len(emb)] = emb
            projected_embeddings.append(padded)
        elif len(emb) > target_dim:
            # Truncate (or could use PCA/random projection)
            projected_embeddings.append(emb[:target_dim])
        else:
            projected_embeddings.append(emb)
    
    # Stack and compute mean
    stacked = np.stack(projected_embeddings)
    return np.mean(stacked, axis=0)


def concatenation_fusion(embeddings_list, modality_dims, available_modalities, all_modalities):
    """
    Fuse embeddings using Concatenation with zero-padding for missing modalities.
    
    Args:
        embeddings_list: List of embedding arrays for available modalities
        modality_dims: Dictionary of dimensions for each modality
        available_modalities: List of available modalities for this patient
        all_modalities: List of all possible modalities
        
    Returns:
        Fused embedding using concatenation with consistent dimension
    """
    if len(embeddings_list) == 0:
        raise ValueError("No embeddings provided")
    
    # Create a consistent order and pad missing modalities with zeros
    fused = []
    emb_idx = 0
    
    for modality in sorted(all_modalities):
        if modality in available_modalities:
            fused.append(embeddings_list[emb_idx])
            emb_idx += 1
        else:
            # Pad with zeros for missing modality
            fused.append(np.zeros(modality_dims[modality]))
    
    return np.concatenate(fused)


def create_multimodal_embeddings(embeddings, fusion_method="concat"):
    """
    Create multimodal embeddings using specified fusion method.
    
    Args:
        embeddings: Dictionary of embeddings for each modality
        fusion_method: One of "concat", "mean_pool", or "kp"
        
    Returns:
        Dictionary with multimodal embeddings
    """
    print(f"Creating multimodal embeddings using {fusion_method} fusion...")
    
    # Get all available modalities (excluding any existing multimodal)
    modalities = [m for m in embeddings.keys() if m != "multimodal"]
    
    if len(modalities) < 2:
        print("Need at least 2 modalities for multimodal fusion")
        return None
    
    # Create mapping of patient_id to index for each modality
    patient_indices = {}
    for modality in modalities:
        patient_indices[modality] = {
            pid: idx
            for idx, pid in enumerate(embeddings[modality]["patient_ids"])
        }
    
    # Find all unique patients
    all_patients = set()
    for modality in modalities:
        all_patients.update(embeddings[modality]["patient_ids"])
    
    # Get modality dimensions
    modality_dims = {}
    for modality in modalities:
        sample_emb = embeddings[modality]["X"][0]
        if len(sample_emb.shape) > 1:
            sample_emb = sample_emb.flatten()
        modality_dims[modality] = len(sample_emb)
    
    print(f"Modality dimensions: {modality_dims}")
    
    # Process each patient
    multimodal_X = []
    multimodal_y = []
    multimodal_patient_ids = []
    modality_availability = []
    
    for patient_id in sorted(all_patients):
        # Collect available embeddings for this patient
        patient_embeddings = []
        available_modalities = []
        patient_label = None
        
        for modality in sorted(modalities):  # Sort for consistency
            if patient_id in patient_indices[modality]:
                idx = patient_indices[modality][patient_id]
                emb = embeddings[modality]["X"][idx]
                
                # Flatten if necessary
                if len(emb.shape) > 1:
                    emb = emb.flatten()
                
                patient_embeddings.append(emb)
                available_modalities.append(modality)
                
                # Use label from any available modality
                if patient_label is None:
                    patient_label = embeddings[modality]["y"][idx]
        
        # Only process if at least 2 modalities are available
        if len(available_modalities) >= 2:
            try:
                # Apply fusion method
                if fusion_method == "concat":
                    fused = concatenation_fusion(patient_embeddings, modality_dims, 
                                                available_modalities, modalities)
                elif fusion_method == "mean_pool":
                    # For mean pool, we need all modalities to have same dim, so pad missing ones
                    padded_embeddings = []
                    emb_idx = 0
                    for modality in sorted(modalities):
                        if modality in available_modalities:
                            padded_embeddings.append(patient_embeddings[emb_idx])
                            emb_idx += 1
                        else:
                            # Pad with zeros for missing modality
                            padded_embeddings.append(np.zeros(modality_dims[modality]))
                    fused = mean_pool_fusion(padded_embeddings)
                elif fusion_method == "kp":
                    # For KP, also pad missing modalities
                    padded_embeddings = []
                    emb_idx = 0
                    for modality in sorted(modalities):
                        if modality in available_modalities:
                            padded_embeddings.append(patient_embeddings[emb_idx])
                            emb_idx += 1
                        else:
                            # Pad with zeros for missing modality
                            padded_embeddings.append(np.zeros(modality_dims[modality]))
                    fused = kronecker_product_fusion(padded_embeddings)
                else:
                    raise ValueError(f"Unknown fusion method: {fusion_method}")
                
                multimodal_X.append(fused)
                multimodal_y.append(patient_label)
                multimodal_patient_ids.append(patient_id)
                modality_availability.append(available_modalities)
                
            except Exception as e:
                print(f"Error fusing embeddings for patient {patient_id}: {e}")
    
    if multimodal_X:
        # Convert to numpy arrays
        multimodal_X = np.array(multimodal_X)
        
        # Print statistics
        print(f"Created multimodal embeddings for {len(multimodal_X)} patients")
        print(f"Embedding dimension: {multimodal_X.shape[1]}")
        
        # Count modality combinations
        from collections import Counter
        combo_counts = Counter([",".join(sorted(mods)) for mods in modality_availability])
        print("\nModality combinations:")
        for combo, count in combo_counts.most_common():
            print(f"  {combo}: {count} patients")
        
        return {
            "X": multimodal_X,
            "y": np.array(multimodal_y),
            "patient_ids": np.array(multimodal_patient_ids),
            "modality_availability": modality_availability
        }
    
    return None