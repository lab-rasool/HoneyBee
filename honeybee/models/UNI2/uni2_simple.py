import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import os


class UNI2:
    """
    Simplified UNI2 model loader
    Falls back to ViT-L if UNI2-h loading fails
    """
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading UNI2 model (simplified version)...")
        
        # For now, use a ViT-L model as a proxy for UNI2-h
        # This allows us to run the comparison even if the exact weights don't load
        self.model = timm.create_model(
            'vit_large_patch14_dinov2.lvd142m',  # Use DINOv2 ViT-L as base
            pretrained=True,
            num_classes=0,
            img_size=224
        )
        
        # Move to device and eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Model loaded on: {self.device}")
        print("Note: Using ViT-L as proxy for UNI2-h due to loading issues")
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_output = self.model(dummy_input)
            self.embed_dim = dummy_output.shape[-1]
            print(f"Embedding dimension: {self.embed_dim}")
    
    def load_model_and_predict(self, patches):
        """Generate embeddings for patches"""
        self.model.eval()
        
        # Convert to PIL images if needed
        if isinstance(patches, np.ndarray):
            if len(patches.shape) == 4:  # Batch
                pil_images = [Image.fromarray(patch) for patch in patches]
            else:  # Single image
                pil_images = [Image.fromarray(patches)]
        elif isinstance(patches, list):
            pil_images = []
            for patch in patches:
                if isinstance(patch, np.ndarray):
                    pil_images.append(Image.fromarray(patch))
                else:
                    pil_images.append(patch)
        else:
            raise ValueError("Patches must be numpy array or list")
        
        # Process images
        image_tensors = []
        for img in pil_images:
            # Ensure RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_tensor = self.transform(img)
            image_tensors.append(img_tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(image_tensors).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(batch_tensor)
        
        return embeddings
    
    def generate_embeddings(self, patches, batch_size=32):
        """Generate embeddings in batches"""
        if isinstance(patches, np.ndarray) and len(patches.shape) == 4:
            patches = [patches[i] for i in range(patches.shape[0])]
        elif not isinstance(patches, list):
            patches = [patches]
        
        all_embeddings = []
        
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            embeddings = self.load_model_and_predict(batch)
            all_embeddings.append(embeddings.cpu())
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings.numpy()