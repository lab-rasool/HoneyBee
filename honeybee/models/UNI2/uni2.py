import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
import os


class UNI2:
    """
    UNI2-h model from Mahmood Lab
    https://huggingface.co/MahmoodLab/UNI2-h
    
    This is a ViT-g/14 model trained on over 1.5M pathology slides
    """
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "MahmoodLab/UNI2-h"
        
        print(f"Loading UNI2-h model...")
        
        try:
            # Try to load from HuggingFace
            print("Downloading UNI2-h from HuggingFace...")
            
            # Download the model file
            model_file = hf_hub_download(
                repo_id=self.model_name,
                filename="pytorch_model.bin",
                cache_dir="/mnt/f/Projects/HoneyBee/models/cache"
            )
            
            # UNI2-h is a ViT-g/14 model with specific configuration
            # Create the model architecture matching UNI2-h
            self.model = timm.create_model(
                'vit_giant_patch14_224',  # ViT-g/14 architecture
                pretrained=False,
                num_classes=0,  # Remove classification head
                img_size=224,
                embed_dim=1536,  # UNI2-h uses 1536 embed dim
                depth=40,        # 40 transformer blocks
                num_heads=24,    # 24 attention heads
                mlp_ratio=4,
                patch_size=14,
                global_pool='avg'  # Use average pooling
            )
            
            # Load the weights
            print("Loading UNI2-h weights...")
            state_dict = torch.load(model_file, map_location=self.device)
            
            # Handle potential state dict wrapper
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove any prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove 'model.' or 'module.' prefix if present
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                elif k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            # Handle positional embedding size mismatch
            if 'pos_embed' in new_state_dict:
                pos_embed_checkpoint = new_state_dict['pos_embed']
                pos_embed_model = self.model.pos_embed
                
                if pos_embed_checkpoint.shape != pos_embed_model.shape:
                    print(f"Resizing pos_embed from {pos_embed_checkpoint.shape} to {pos_embed_model.shape}")
                    # Interpolate positional embeddings if needed
                    if pos_embed_checkpoint.shape[1] == 256 and pos_embed_model.shape[1] == 257:
                        # Add CLS token embedding
                        cls_token = pos_embed_checkpoint[:, :1, :]  # Use first token as CLS
                        spatial_pos_embed = pos_embed_checkpoint[:, 1:, :]  # Remaining are spatial
                        new_state_dict['pos_embed'] = torch.cat([cls_token, spatial_pos_embed], dim=1)
                    else:
                        # Skip loading pos_embed if incompatible
                        print("Skipping pos_embed due to shape mismatch")
                        del new_state_dict['pos_embed']
            
            # Load with strict=False to handle any remaining mismatches
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            print("Successfully loaded UNI2-h weights")
            
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
            print("Falling back to create ViT-g/14 architecture...")
            
            # Create ViT-g/14 model without pretrained weights
            self.model = timm.create_model(
                'vit_giant_patch14_224',
                pretrained=False,
                num_classes=0,
                img_size=224
            )
            
            if model_path and os.path.exists(model_path):
                print(f"Loading weights from {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
        
        # Move to device and set to eval
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
        
        print(f"UNI2-h model loaded on: {self.device}")
        
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
    
    def generate_embeddings(self, patches, batch_size=16):
        """
        Generate embeddings in batches (reduced batch size for ViT-g)
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