"""
RadImageNet Demo Script

Comprehensive demonstration of RadImageNet integration for medical image analysis,
including feature extraction, multi-modal fusion, and transfer learning examples.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Import radiology modules
from radiology.data_management import load_dicom_series
from radiology.preprocessing import preprocess_ct, WindowLevelAdjuster
from radiology.ai_integration import (
    RadImageNetProcessor, 
    MultiModalFusion,
    create_embedding_model,
    process_2d_slices,
    process_3d_volume
)
from radiology.utils import visualize_slices, create_montage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_radimagenet_models():
    """Demonstrate different RadImageNet model architectures"""
    
    logger.info("=" * 50)
    logger.info("RadImageNet Model Comparison")
    logger.info("=" * 50)
    
    # Load sample CT data
    ct_dir = Path("../samples/CT")
    if not ct_dir.exists():
        logger.error(f"CT samples not found at {ct_dir}")
        return
    
    # Load CT volume
    ct_volume, metadata = load_dicom_series(ct_dir)
    logger.info(f"Loaded CT volume: {ct_volume.shape}")
    
    # Preprocess with lung window
    ct_processed = preprocess_ct(ct_volume, window='lung', normalize=True)
    
    # Test different RadImageNet models
    models = ['densenet121', 'resnet50', 'inception_v3']
    embeddings = {}
    processing_times = {}
    
    for model_name in models:
        logger.info(f"\nTesting {model_name}...")
        
        # Create processor
        processor = RadImageNetProcessor(model_name=model_name, pretrained=True)
        
        # Measure processing time
        import time
        start_time = time.time()
        
        # Generate embeddings
        embedding = processor.generate_embeddings(ct_processed, mode='2d')
        
        processing_time = time.time() - start_time
        processing_times[model_name] = processing_time
        
        embeddings[model_name] = embedding
        
        logger.info(f"  Embedding shape: {embedding.shape}")
        logger.info(f"  Processing time: {processing_time:.3f} seconds")
        logger.info(f"  GPU used: {torch.cuda.is_available()}")
    
    # Compare embeddings
    logger.info("\nEmbedding Statistics:")
    for model_name, embedding in embeddings.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Mean: {embedding.mean():.4f}")
        logger.info(f"  Std: {embedding.std():.4f}")
        logger.info(f"  Min: {embedding.min():.4f}")
        logger.info(f"  Max: {embedding.max():.4f}")
    
    return embeddings, ct_processed


def demonstrate_feature_extraction():
    """Demonstrate multi-scale feature extraction"""
    
    logger.info("\n" + "=" * 50)
    logger.info("Multi-Scale Feature Extraction")
    logger.info("=" * 50)
    
    # Load sample data
    ct_dir = Path("../samples/CT")
    ct_volume, metadata = load_dicom_series(ct_dir)
    ct_processed = preprocess_ct(ct_volume, window='lung')
    
    # Create processor
    processor = RadImageNetProcessor('densenet121')
    
    # Extract features from multiple layers
    features = processor.extract_features(ct_processed)
    
    # Visualize feature maps
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (layer_name, feature_map) in enumerate(features.items()):
        if idx >= 4:
            break
        
        # Average across channels for visualization
        if len(feature_map.shape) == 4:  # Batch x Channels x H x W
            avg_feature = feature_map[0].mean(axis=0)
        else:
            avg_feature = feature_map.mean(axis=0)
        
        im = axes[idx].imshow(avg_feature, cmap='viridis')
        axes[idx].set_title(f'{layer_name} (avg across channels)')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx])
    
    plt.tight_layout()
    plt.savefig('results_radimagenet_features.png', dpi=150, bbox_inches='tight')
    logger.info("Feature maps saved to: results_radimagenet_features.png")
    
    # Analyze feature statistics
    logger.info("\nFeature Map Statistics:")
    for layer_name, feature_map in features.items():
        logger.info(f"\n{layer_name}:")
        logger.info(f"  Shape: {feature_map.shape}")
        logger.info(f"  Mean activation: {feature_map.mean():.4f}")
        logger.info(f"  Sparsity: {(feature_map == 0).mean():.2%}")


def demonstrate_3d_processing():
    """Demonstrate 3D volume processing strategies"""
    
    logger.info("\n" + "=" * 50)
    logger.info("3D Volume Processing Strategies")
    logger.info("=" * 50)
    
    # Load sample data
    ct_dir = Path("../samples/CT")
    ct_volume, metadata = load_dicom_series(ct_dir)
    ct_processed = preprocess_ct(ct_volume, window='lung')
    
    # Create processor
    processor = RadImageNetProcessor('densenet121')
    
    # Strategy 1: Single middle slice
    logger.info("\nStrategy 1: Single middle slice")
    embedding_middle = processor.generate_embeddings(ct_processed, mode='2d')
    logger.info(f"  Embedding shape: {embedding_middle.shape}")
    
    # Strategy 2: Multiple key slices
    logger.info("\nStrategy 2: Multiple key slices")
    key_slices = [0, ct_volume.shape[0]//4, ct_volume.shape[0]//2, 
                  3*ct_volume.shape[0]//4, ct_volume.shape[0]-1]
    embeddings_key = process_2d_slices(ct_processed, slice_indices=key_slices)
    logger.info(f"  Embeddings shape: {embeddings_key.shape}")
    
    # Strategy 3: Full 3D with mean aggregation
    logger.info("\nStrategy 3: Full 3D with mean aggregation")
    embedding_3d_mean = processor.generate_embeddings(ct_processed, mode='3d', aggregation='mean')
    logger.info(f"  Embedding shape: {embedding_3d_mean.shape}")
    
    # Strategy 4: Full 3D with max aggregation
    logger.info("\nStrategy 4: Full 3D with max aggregation")
    embedding_3d_max = processor.generate_embeddings(ct_processed, mode='3d', aggregation='max')
    logger.info(f"  Embedding shape: {embedding_3d_max.shape}")
    
    # Compare strategies
    logger.info("\nComparing 3D processing strategies:")
    
    # Calculate similarities
    strategies = {
        'Middle Slice': embedding_middle,
        'Key Slices Mean': embeddings_key.mean(axis=0),
        '3D Mean': embedding_3d_mean,
        '3D Max': embedding_3d_max
    }
    
    # Compute pairwise similarities
    logger.info("\nCosine Similarities between strategies:")
    for name1, emb1 in strategies.items():
        for name2, emb2 in strategies.items():
            if name1 < name2:  # Avoid duplicates
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                logger.info(f"  {name1} vs {name2}: {similarity:.3f}")
    
    return strategies


def demonstrate_multimodal_fusion():
    """Demonstrate multi-modal fusion capabilities"""
    
    logger.info("\n" + "=" * 50)
    logger.info("Multi-Modal Fusion Demo")
    logger.info("=" * 50)
    
    # For demonstration, we'll use the same CT data with different windows
    # In practice, these would be different modalities (CT, MRI, PET)
    ct_dir = Path("../samples/CT")
    ct_volume, metadata = load_dicom_series(ct_dir)
    
    # Simulate different modalities with different preprocessing
    logger.info("Simulating multi-modal data with different CT windows...")
    
    modality_images = {
        'CT_lung': preprocess_ct(ct_volume, window='lung'),
        'CT_bone': preprocess_ct(ct_volume, window='bone'),
        'CT_soft_tissue': preprocess_ct(ct_volume, window='soft_tissue')
    }
    
    # Test different fusion methods
    fusion_methods = ['concatenate', 'attention', 'learned']
    fusion_results = {}
    
    for method in fusion_methods:
        logger.info(f"\nTesting {method} fusion...")
        
        # Create fusion processor
        fusion = MultiModalFusion(fusion_method=method)
        
        # Add modality processors
        for modality in modality_images.keys():
            fusion.add_modality(modality, model_name='densenet121')
        
        # Perform fusion
        fused_embedding = fusion.fuse(modality_images)
        fusion_results[method] = fused_embedding
        
        logger.info(f"  Fused embedding shape: {fused_embedding.shape}")
        logger.info(f"  Fused embedding mean: {fused_embedding.mean():.4f}")
        logger.info(f"  Fused embedding std: {fused_embedding.std():.4f}")
    
    return fusion_results


def demonstrate_embedding_visualization():
    """Visualize embeddings using dimensionality reduction"""
    
    logger.info("\n" + "=" * 50)
    logger.info("Embedding Visualization")
    logger.info("=" * 50)
    
    # For demonstration, generate embeddings from multiple slices
    ct_dir = Path("../samples/CT")
    ct_volume, metadata = load_dicom_series(ct_dir)
    ct_processed = preprocess_ct(ct_volume, window='lung')
    
    # Generate embeddings from different slices
    processor = RadImageNetProcessor('densenet121')
    
    # Sample slices from different regions
    n_samples = min(30, ct_volume.shape[0])
    slice_indices = np.linspace(0, ct_volume.shape[0]-1, n_samples, dtype=int)
    
    embeddings = []
    slice_positions = []
    
    logger.info(f"Generating embeddings from {n_samples} slices...")
    for idx in slice_indices:
        emb = processor._process_2d_slice(ct_processed[idx])
        embeddings.append(emb)
        slice_positions.append(idx / ct_volume.shape[0])  # Normalized position
    
    embeddings = np.stack(embeddings)
    
    # Dimensionality reduction
    logger.info("Applying dimensionality reduction...")
    
    # PCA
    n_components = min(50, len(embeddings) - 1)
    pca = PCA(n_components=n_components)
    embeddings_pca = pca.fit_transform(embeddings)
    logger.info(f"  PCA explained variance: {pca.explained_variance_ratio_[:5].sum():.2%} (first 5 components)")
    
    # t-SNE
    perplexity = min(30, len(embeddings) - 1)  # Default is 30, but we need less than n_samples
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings_pca)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA visualization (first 2 components)
    scatter1 = ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                          c=slice_positions, cmap='viridis', s=100)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA of RadImageNet Embeddings')
    plt.colorbar(scatter1, ax=ax1, label='Slice Position')
    
    # t-SNE visualization
    scatter2 = ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                          c=slice_positions, cmap='viridis', s=100)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('t-SNE of RadImageNet Embeddings')
    plt.colorbar(scatter2, ax=ax2, label='Slice Position')
    
    plt.tight_layout()
    plt.savefig('results_embedding_visualization.png', dpi=150, bbox_inches='tight')
    logger.info("Visualization saved to: results_embedding_visualization.png")
    
    return embeddings, embeddings_tsne


def demonstrate_transfer_learning():
    """Demonstrate using RadImageNet for transfer learning"""
    
    logger.info("\n" + "=" * 50)
    logger.info("Transfer Learning with RadImageNet")
    logger.info("=" * 50)
    
    # This demonstrates how to use RadImageNet models for transfer learning
    # In practice, you would fine-tune on your specific task
    
    # Load pretrained model
    model = create_embedding_model('densenet121', pretrained=True)
    
    # Get model architecture info
    logger.info("Model architecture summary:")
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model on device: {model.device}")
    
    # Example: Extract features for downstream task
    ct_dir = Path("../samples/CT")
    ct_volume, metadata = load_dicom_series(ct_dir)
    ct_processed = preprocess_ct(ct_volume, window='lung')
    
    # Extract features from different layers
    features = model.extract_features(ct_processed)
    
    logger.info("\nFeature extraction for transfer learning:")
    for layer_name, feat in features.items():
        logger.info(f"  {layer_name}: {feat.shape} -> {np.prod(feat.shape[1:])} features")
    
    # Demonstrate feature quality
    # In practice, these features would be used for:
    # - Classification (e.g., disease detection)
    # - Regression (e.g., severity scoring)
    # - Segmentation (as encoder features)
    # - Retrieval (similarity search)
    
    logger.info("\nPotential downstream applications:")
    logger.info("  1. Disease Classification: Use final embeddings with classifier head")
    logger.info("  2. Anomaly Detection: Use embeddings for outlier detection")
    logger.info("  3. Image Retrieval: Use embeddings for similarity search")
    logger.info("  4. Segmentation: Use intermediate features in U-Net decoder")
    logger.info("  5. Multi-task Learning: Share encoder across multiple tasks")


def main():
    """Main demonstration function"""
    
    logger.info("RadImageNet Integration Demonstration")
    logger.info("=====================================\n")
    
    # Create results directory
    results_dir = Path("results_radimagenet_demo")
    results_dir.mkdir(exist_ok=True)
    
    # 1. Model comparison
    embeddings, ct_processed = demonstrate_radimagenet_models()
    
    # 2. Feature extraction
    demonstrate_feature_extraction()
    
    # 3. 3D processing strategies
    strategies = demonstrate_3d_processing()
    
    # 4. Multi-modal fusion
    fusion_results = demonstrate_multimodal_fusion()
    
    # 5. Embedding visualization
    embeddings_array, embeddings_tsne = demonstrate_embedding_visualization()
    
    # 6. Transfer learning
    demonstrate_transfer_learning()
    
    # Generate summary report
    logger.info("\n" + "=" * 50)
    logger.info("Summary")
    logger.info("=" * 50)
    
    logger.info("\nKey Capabilities Demonstrated:")
    logger.info("✓ Multiple pre-trained RadImageNet architectures")
    logger.info("✓ GPU-accelerated processing")
    logger.info("✓ 2D and 3D volume processing")
    logger.info("✓ Multi-scale feature extraction")
    logger.info("✓ Multi-modal fusion strategies")
    logger.info("✓ Embedding visualization and analysis")
    logger.info("✓ Transfer learning applications")
    
    logger.info("\nRadImageNet provides high-quality embeddings specifically")
    logger.info("trained on radiological images, offering better performance")
    logger.info("than natural image models for medical imaging tasks.")
    
    logger.info("\nDemo complete! Check generated visualizations and results.")


if __name__ == "__main__":
    main()