import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
import numpy as np
from PIL import Image
import os


class Virchow2:
    """
    Virchow2 model from Paige AI
    https://huggingface.co/paige-ai/Virchow2
    
    This is a DINOv2 ViT-L/14 model trained on 3.1M pathology slides
    """
    def __init__(self, model_path=None):
        # Use the HuggingFace model directly
        self.model_name = "paige-ai/Virchow2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading Virchow2 model from HuggingFace...")
        try:
            # Try loading with transformers first
            from transformers import ViTModel, ViTImageProcessor
            
            # Load as a ViT model without classification head
            self.model = ViTModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                num_labels=None,  # No classification head needed
                ignore_mismatched_sizes=True
            ).to(self.device)
            
            self.processor = ViTImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Failed to load with ViTModel, trying alternative approach: {e}")
            # Alternative: Load using timm
            import timm
            from torchvision import transforms
            
            # Virchow2 is based on DINOv2 ViT-L/14
            self.model = timm.create_model(
                'vit_large_patch14_dinov2',
                pretrained=False,
                num_classes=0,  # Remove classification head
                img_size=224
            )
            
            # Load weights if available locally
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            
            self.model = self.model.to(self.device)
            
            # Create standard preprocessing
            self.processor = None
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        self.model.eval()
        print(f"Virchow2 model loaded on: {self.device}")
        
    def load_model_and_predict(self, patches):
        """
        Generate embeddings for patches
        
        Args:
            patches: numpy array of shape (N, H, W, 3) or list of PIL images
        
        Returns:
            embeddings: torch tensor of shape (N, embedding_dim)
        """
        self.model.eval()
        
        # Convert numpy arrays to PIL images if needed
        if isinstance(patches, np.ndarray):
            if len(patches.shape) == 4:  # Batch of images
                pil_images = [Image.fromarray(patch) for patch in patches]
            else:  # Single image
                pil_images = [Image.fromarray(patches)]
        elif isinstance(patches, list):
            # Assume already PIL images or convert if numpy
            pil_images = []
            for patch in patches:
                if isinstance(patch, np.ndarray):
                    pil_images.append(Image.fromarray(patch))
                else:
                    pil_images.append(patch)
        else:
            raise ValueError("Patches must be numpy array or list of images")
        
        # Process images based on which loader we're using
        if self.processor is not None:
            # Using HuggingFace processor
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get the CLS token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            # Using timm model with manual preprocessing
            # Convert PIL images to tensors
            image_tensors = []
            for img in pil_images:
                img_tensor = self.transform(img)
                image_tensors.append(img_tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(image_tensors).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                # For timm models, forward pass returns features directly
                embeddings = self.model(batch_tensor)
        
        return embeddings
    
    def generate_embeddings(self, patches, batch_size=32):
        """
        Generate embeddings in batches for memory efficiency
        
        Args:
            patches: numpy array or list of images
            batch_size: number of patches to process at once
            
        Returns:
            embeddings: numpy array of shape (N, embedding_dim)
        """
        # Ensure patches is a list
        if isinstance(patches, np.ndarray) and len(patches.shape) == 4:
            patches = [patches[i] for i in range(patches.shape[0])]
        elif not isinstance(patches, list):
            patches = [patches]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size]
            embeddings = self.load_model_and_predict(batch)
            all_embeddings.append(embeddings.cpu())
            
            # Clear cache periodically
            if i % (batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all embeddings
        final_embeddings = torch.cat(all_embeddings, dim=0)
        
        return final_embeddings.numpy()