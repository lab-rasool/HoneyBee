import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
import requests
import os
from huggingface_hub import hf_hub_download


class Virchow2:
    """
    Simplified Virchow2 model loader using timm
    Based on DINOv2 ViT-L/14 architecture
    """
    def __init__(self, model_path=None, use_hf=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Virchow2 model...")
        
        # Create DINOv2 ViT-L/14 model
        self.model = timm.create_model(
            'vit_large_patch14_reg4_dinov2.lvd142m',
            pretrained=True,  # Load DINOv2 pretrained weights as base
            num_classes=0,    # Remove classification head
            img_size=224
        )
        
        # Try to load Virchow2 weights if available
        if use_hf:
            try:
                print("Attempting to download Virchow2 weights from HuggingFace...")
                # Download model weights
                model_file = hf_hub_download(
                    repo_id="paige-ai/Virchow2",
                    filename="pytorch_model.bin",
                    cache_dir="/mnt/f/Projects/HoneyBee/models/cache"
                )
                
                # Load the weights
                state_dict = torch.load(model_file, map_location=self.device)
                # Try to load with strict=False to handle any mismatches
                self.model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded Virchow2 weights")
            except Exception as e:
                print(f"Could not load Virchow2 weights, using DINOv2 base: {e}")
        
        elif model_path and os.path.exists(model_path):
            # Load from local path
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {model_path}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Model loaded on: {self.device}")
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_output = self.model(dummy_input)
            self.embed_dim = dummy_output.shape[-1]
            print(f"Embedding dimension: {self.embed_dim}")
    
    def load_model_and_predict(self, patches):
        """
        Generate embeddings for patches
        
        Args:
            patches: numpy array of shape (N, H, W, 3) or list of PIL images
        
        Returns:
            embeddings: torch tensor of shape (N, embedding_dim)
        """
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
        """
        Generate embeddings in batches
        """
        if isinstance(patches, np.ndarray) and len(patches.shape) == 4:
            patches = [patches[i] for i in range(patches.shape[0])]
        elif not isinstance(patches, list):
            patches = [patches]
        
        all_embeddings = []
        
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            embeddings = self.load_model_and_predict(batch)
            all_embeddings.append(embeddings.cpu())
            
            if i % (batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings.numpy()