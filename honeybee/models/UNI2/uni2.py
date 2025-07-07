import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import login, hf_hub_download
import os


class UNI2:
    """
    Proper UNI2-h model implementation based on official documentation
    https://huggingface.co/MahmoodLab/UNI2-h
    
    ViT-H/14 with 681M parameters trained on 200M+ pathology tiles
    """
    def __init__(self, model_path=None, use_auth_token=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading UNI2-h model (proper implementation)...")
        
        # Model configuration from official documentation
        self.timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        
        if use_auth_token:
            # Use HuggingFace Hub (requires authentication)
            print("Using HuggingFace Hub (requires authentication)")
            try:
                # Create model with HF hub
                self.model = timm.create_model(
                    "hf-hub:MahmoodLab/UNI2-h", 
                    pretrained=True, 
                    **self.timm_kwargs
                )
                print("Successfully loaded from HuggingFace Hub")
            except Exception as e:
                print(f"Error loading from HF Hub: {e}")
                print("Please run: huggingface-cli login")
                raise
        else:
            # Create model architecture
            self.model = timm.create_model(
                pretrained=False, 
                **self.timm_kwargs
            )
            
            # Try to load weights
            if model_path and os.path.exists(model_path):
                print(f"Loading weights from {model_path}")
                state_dict = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=True)
                print("Successfully loaded UNI2-h weights")
            else:
                # Try to download weights
                try:
                    print("Attempting to download UNI2-h weights...")
                    local_dir = "/mnt/f/Projects/HoneyBee/models/uni2-h/"
                    os.makedirs(local_dir, exist_ok=True)
                    
                    # This requires authentication
                    weight_path = hf_hub_download(
                        "MahmoodLab/UNI2-h", 
                        filename="pytorch_model.bin", 
                        local_dir=local_dir,
                        cache_dir=local_dir
                    )
                    
                    state_dict = torch.load(weight_path, map_location="cpu")
                    self.model.load_state_dict(state_dict, strict=True)
                    print("Successfully downloaded and loaded UNI2-h weights")
                except Exception as e:
                    print(f"Could not download weights: {e}")
                    print("Using random initialization. Results will not be meaningful.")
                    print("To use proper weights, either:")
                    print("1. Run: huggingface-cli login")
                    print("2. Download weights manually and provide path")
        
        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create official transform
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        print(f"UNI2-h model loaded on: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        
        # Verify embedding dimension
        self.embed_dim = 1536  # Fixed for UNI2-h
    
    def load_model_and_predict(self, patches):
        """
        Generate embeddings for patches
        
        Args:
            patches: numpy array of shape (N, H, W, 3) or list of PIL images
        
        Returns:
            embeddings: torch tensor of shape (N, 1536)
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
        with torch.inference_mode():
            feature_emb = self.model(batch_tensor)
        
        return feature_emb
    
    def generate_embeddings(self, patches, batch_size=16):
        """
        Generate embeddings in batches (reduced batch size for ViT-H)
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
            
            # Clear cache more frequently for large model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        final_embeddings = torch.cat(all_embeddings, dim=0)
        return final_embeddings.numpy()