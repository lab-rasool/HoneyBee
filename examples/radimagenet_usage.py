"""
Example: Using RadImageNet with HuggingFace Hub and Local Models

This example shows how to use the enhanced RadImageNet model with:
1. HuggingFace Hub downloads
2. Local model paths
3. Advanced features like multi-scale extraction and batch processing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from honeybee.models.RadImageNet import RadImageNet


def example_1_huggingface_hub():
    """Example 1: Using HuggingFace Hub"""
    print("\n" + "="*60)
    print("Example 1: Loading from HuggingFace Hub")
    print("="*60)
    
    # Initialize model with HuggingFace Hub
    model = RadImageNet(
        model_name="ResNet50",
        use_hub=True,  # This will download from HuggingFace
        repo_id="Lab-Rasool/RadImageNet"
    )
    
    print(f"✓ Model loaded from HuggingFace Hub")
    print(f"  Model: {model.model_name}")
    print(f"  Input size: {model.input_size}")
    print(f"  Embedding dimension: {model.get_embedding_dim()}")
    
    # Generate embeddings
    test_image = np.random.rand(512, 512).astype(np.float32)
    embeddings = model.generate_embeddings(test_image, mode='2d')
    print(f"  Embedding shape: {embeddings.shape}")


def example_2_local_models():
    """Example 2: Using Local Models"""
    print("\n" + "="*60)
    print("Example 2: Loading from Local Path")
    print("="*60)
    
    # The model will automatically check default locations:
    # 1. /mnt/d/Models/radimagenet/
    # 2. ~/.cache/radimagenet/
    
    # Option 1: Auto-detect from default paths
    model1 = RadImageNet(
        model_name="DenseNet121",
        use_hub=False  # Will look in default paths
    )
    print("✓ Model loaded from default local path")
    
    # Option 2: Specify explicit path
    model2 = RadImageNet(
        model_path="/mnt/d/Models/radimagenet/ResNet50.pt",
        model_name="ResNet50",
        use_hub=False
    )
    print("✓ Model loaded from explicit path")
    
    # Generate embeddings
    test_image = np.random.rand(224, 224).astype(np.float32)
    embeddings = model1.generate_embeddings(test_image, mode='2d')
    print(f"  DenseNet121 embedding shape: {embeddings.shape}")


def example_3_advanced_features():
    """Example 3: Advanced Features"""
    print("\n" + "="*60)
    print("Example 3: Advanced Features")
    print("="*60)
    
    # Initialize with feature extraction
    model = RadImageNet(
        model_name="ResNet50",
        use_hub=False,
        extract_features=True,
        feature_layers=['layer1', 'layer2', 'layer3', 'layer4']
    )
    
    print("✓ Model initialized with feature extraction")
    
    # 1. Multi-scale feature extraction
    print("\n1. Multi-scale features:")
    test_image = np.random.rand(256, 256).astype(np.float32)
    multi_scale = model.extract_multi_scale_features(
        test_image, 
        scales=[0.5, 1.0, 1.5]
    )
    for scale, features in multi_scale.items():
        print(f"   Scale {scale}: {features.shape}")
    
    # 2. Batch processing
    print("\n2. Batch processing:")
    batch_images = [np.random.rand(224, 224) for _ in range(10)]
    batch_embeddings = model.process_batch(batch_images, batch_size=5)
    print(f"   Batch shape: {batch_embeddings.shape}")
    
    # 3. 3D volume processing
    print("\n3. 3D volume processing:")
    volume = np.random.rand(32, 224, 224).astype(np.float32)
    
    # Different aggregation methods
    for aggregation in ['mean', 'max']:
        embeddings_3d = model.generate_embeddings(
            volume, mode='3d', aggregation=aggregation
        )
        print(f"   {aggregation} aggregation: {embeddings_3d.shape}")
    
    # 4. Feature extraction from intermediate layers
    print("\n4. Intermediate features:")
    result = model.generate_embeddings(
        test_image, 
        mode='2d',
        return_features=True
    )
    
    if isinstance(result, dict):
        print(f"   Final embeddings: {result['embeddings'].shape}")
        print("   Intermediate features:")
        for layer, features in result['features'].items():
            print(f"     {layer}: {features.shape}")


def example_4_model_comparison():
    """Example 4: Comparing Different Models"""
    print("\n" + "="*60)
    print("Example 4: Model Comparison")
    print("="*60)
    
    models = {
        'DenseNet121': 1024,
        'ResNet50': 2048,
        'InceptionV3': 2048
    }
    
    test_image = np.random.rand(512, 512).astype(np.float32)
    
    for model_name, expected_dim in models.items():
        # InceptionV3 requires 299x299 input
        if model_name == 'InceptionV3':
            test_img = np.random.rand(299, 299).astype(np.float32)
        else:
            test_img = test_image
        
        model = RadImageNet(
            model_name=model_name,
            use_hub=False
        )
        
        embeddings = model.generate_embeddings(test_img, mode='2d')
        print(f"{model_name}:")
        print(f"  Input size: {model.input_size}x{model.input_size}")
        print(f"  Embedding dimension: {embeddings.shape[-1]} (expected: {expected_dim})")
        print(f"  ✓ Verified" if embeddings.shape[-1] == expected_dim else "  ✗ Mismatch")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" RADIMAGENET USAGE EXAMPLES")
    print("="*80)
    
    print("\nThe enhanced RadImageNet model supports:")
    print("  • HuggingFace Hub integration")
    print("  • Local model loading with auto-detection")
    print("  • Multi-scale feature extraction")
    print("  • Batch processing")
    print("  • 3D volume processing with aggregation")
    print("  • Intermediate layer feature extraction")
    
    # Check if models exist locally
    import os
    if os.path.exists("/mnt/d/Models/radimagenet/"):
        print("\n✓ Local models found at /mnt/d/Models/radimagenet/")
        models = ['DenseNet121.pt', 'ResNet50.pt', 'InceptionV3.pt']
        for model in models:
            if os.path.exists(f"/mnt/d/Models/radimagenet/{model}"):
                print(f"  ✓ {model}")
    
    try:
        # Run examples that don't require downloading
        example_2_local_models()
        example_3_advanced_features()
        example_4_model_comparison()
        
        # Optionally run HuggingFace example
        print("\n" + "="*60)
        print("Note: Example 1 (HuggingFace Hub) skipped to avoid downloads")
        print("To run it, uncomment the line below:")
        print("# example_1_huggingface_hub()")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure models are available at /mnt/d/Models/radimagenet/")
    
    print("\n" + "="*80)
    print(" All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()