import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import numpy as np
import os
from typing import Optional, Union, Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RadImageNet:
    """Enhanced RadImageNet model with advanced features
    
    Features:
        - Multi-scale feature extraction
        - Batch processing capabilities
        - 2D/3D processing modes
        - Intermediate layer feature extraction
        - Flexible preprocessing pipelines
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "ResNet50",
        use_hub: bool = True,
        repo_id: str = "Lab-Rasool/RadImageNet",
        model_filename: Optional[str] = None,
        extract_features: bool = False,
        feature_layers: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize RadImageNet model.

        Args:
            model_path: Local path to model weights (optional if use_hub=True)
            model_name: Model name (DenseNet121, InceptionV3, or ResNet50)
            use_hub: Whether to load from HuggingFace Hub (default: True)
            repo_id: HuggingFace repository ID (default: Lab-Rasool/RadImageNet)
            model_filename: Specific model filename (e.g., 'ResNet50.pt', 'DenseNet121.pt')
            extract_features: Whether to enable feature extraction from intermediate layers
            feature_layers: List of layer names to extract features from
        """
        self.model_path = model_path
        self.model_name = model_name
        self.use_hub = use_hub
        self.repo_id = repo_id
        self.model_filename = model_filename
        self.extract_features = extract_features
        self.feature_layers = feature_layers or []
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features = {}  # Store intermediate features
        self.hooks = []  # Store forward hooks
        
        # Set up preprocessing
        self.input_size = self._get_input_size()
        self.preprocess = self._setup_preprocessing()
        
        # Load model
        self._load_model()
        
        # Set up feature extraction if requested
        if self.extract_features:
            self._setup_feature_extraction()

    def _load_model(self) -> None:
        # Use provided filename or determine from model name
        if self.model_filename:
            model_filename = self.model_filename
            # Extract model type from filename
            if "DenseNet121" in model_filename:
                self.model = models.densenet121(weights=None)
            elif "InceptionV3" in model_filename:
                self.model = models.inception_v3(weights=None)
            elif "ResNet50" in model_filename:
                self.model = models.resnet50(weights=None)
            else:
                raise ValueError(
                    f"Cannot determine model type from filename: {model_filename}"
                )
        else:
            # Determine model architecture from name
            if "DenseNet121" in self.model_name:
                self.model = models.densenet121(weights=None)
                model_filename = "DenseNet121.pt"
            elif "InceptionV3" in self.model_name:
                self.model = models.inception_v3(weights=None)
                model_filename = "InceptionV3.pt"
            elif "ResNet50" in self.model_name:
                self.model = models.resnet50(weights=None)
                model_filename = "ResNet50.pt"
            else:
                raise ValueError(
                    "Model not recognized. Model name must be one of 'DenseNet121', 'InceptionV3', or 'ResNet50'."
                )

        # Load weights
        if self.use_hub:
            # Download from HuggingFace Hub
            try:
                logger.info(f"Downloading {model_filename} from HuggingFace Hub...")
                self.model_path = hf_hub_download(
                    repo_id=self.repo_id, filename=model_filename
                )
                logger.info(f"Model downloaded to: {self.model_path}")
            except Exception as e:
                logger.warning(f"Error downloading from Hub: {e}")
                # Try local fallback paths
                local_paths = [
                    Path(f"/mnt/d/Models/radimagenet/{model_filename}"),
                    Path.home() / ".cache" / "radimagenet" / model_filename,
                ]
                if self.model_path:
                    local_paths.insert(0, Path(self.model_path))
                
                for path in local_paths:
                    if path.exists():
                        logger.info(f"Found model at local path: {path}")
                        self.model_path = str(path)
                        break
                else:
                    raise ValueError(
                        f"Could not download model from Hub and no local model found. "
                        f"Searched paths: {[str(p) for p in local_paths]}"
                    )
        else:
            # Not using hub - check provided path or default locations
            if not self.model_path:
                # Check default local paths
                local_paths = [
                    Path(f"/mnt/d/Models/radimagenet/{model_filename}"),
                    Path.home() / ".cache" / "radimagenet" / model_filename,
                ]
                
                for path in local_paths:
                    if path.exists():
                        logger.info(f"Found model at default path: {path}")
                        self.model_path = str(path)
                        break
                else:
                    raise ValueError(
                        f"model_path is required when use_hub=False and no model found at default locations. "
                        f"Searched: {[str(p) for p in local_paths]}"
                    )

        # Load the model weights
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            
            # Remove the final classification layer to get embeddings
            if isinstance(self.model, models.DenseNet):
                self.model.classifier = nn.Identity()
            elif isinstance(self.model, models.ResNet):
                self.model.fc = nn.Identity()
            elif isinstance(self.model, models.Inception3):
                self.model.fc = nn.Identity()
                # Also disable auxiliary classifiers
                self.model.aux_logits = False
            
            self.model.to(self.device)  # Move model to the appropriate device (GPU or CPU)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded model from: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {self.model_path}: {e}")

    def _get_input_size(self) -> int:
        """Get expected input size based on model"""
        if "InceptionV3" in self.model_name:
            return 299
        else:
            return 224
    
    def _setup_preprocessing(self) -> transforms.Compose:
        """Set up preprocessing pipeline"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _setup_feature_extraction(self):
        """Set up hooks for intermediate feature extraction"""
        # Clear existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.features.clear()
        
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        
        # Register hooks based on model type
        if isinstance(self.model, models.DenseNet):
            if not self.feature_layers or 'all' in self.feature_layers:
                self.hooks.append(self.model.features.denseblock1.register_forward_hook(get_features('denseblock1')))
                self.hooks.append(self.model.features.denseblock2.register_forward_hook(get_features('denseblock2')))
                self.hooks.append(self.model.features.denseblock3.register_forward_hook(get_features('denseblock3')))
                self.hooks.append(self.model.features.denseblock4.register_forward_hook(get_features('denseblock4')))
            else:
                for layer_name in self.feature_layers:
                    if hasattr(self.model.features, layer_name):
                        layer = getattr(self.model.features, layer_name)
                        self.hooks.append(layer.register_forward_hook(get_features(layer_name)))
        
        elif isinstance(self.model, models.ResNet):
            if not self.feature_layers or 'all' in self.feature_layers:
                self.hooks.append(self.model.layer1.register_forward_hook(get_features('layer1')))
                self.hooks.append(self.model.layer2.register_forward_hook(get_features('layer2')))
                self.hooks.append(self.model.layer3.register_forward_hook(get_features('layer3')))
                self.hooks.append(self.model.layer4.register_forward_hook(get_features('layer4')))
            else:
                for layer_name in self.feature_layers:
                    if hasattr(self.model, layer_name):
                        layer = getattr(self.model, layer_name)
                        self.hooks.append(layer.register_forward_hook(get_features(layer_name)))
    
    def generate_embeddings(self, 
                          input_data: Union[torch.Tensor, np.ndarray],
                          mode: str = '2d',
                          aggregation: str = 'mean',
                          return_features: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate embeddings from input data
        
        Args:
            input_data: Input tensor or numpy array
            mode: Processing mode ('2d' or '3d')
            aggregation: Aggregation method for 3D ('mean', 'max', 'concat')
            return_features: Whether to return intermediate features
            
        Returns:
            Embeddings tensor or dictionary with embeddings and features
        """
        if self.model is None:
            raise ValueError("Model is not loaded.")
        
        # Convert numpy to tensor if needed
        if isinstance(input_data, np.ndarray):
            input_data = self._preprocess_numpy(input_data)
        
        # Process based on mode
        if mode == '2d':
            embeddings = self._process_2d(input_data)
        elif mode == '3d':
            embeddings = self._process_3d(input_data, aggregation)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        if return_features and self.extract_features:
            return {
                'embeddings': embeddings,
                'features': {k: v.cpu() for k, v in self.features.items()}
            }
        
        return embeddings
    
    def _preprocess_numpy(self, data: np.ndarray) -> torch.Tensor:
        """Preprocess numpy array to tensor"""
        # Handle different input shapes
        if len(data.shape) == 2:  # Single 2D image
            # Convert to RGB if grayscale
            data_rgb = np.stack([data] * 3, axis=-1)
            tensor = self.preprocess(data_rgb)
            return tensor.unsqueeze(0)
        
        elif len(data.shape) == 3:  # 3D volume or RGB image
            if data.shape[-1] == 3:  # RGB image
                tensor = self.preprocess(data)
                return tensor.unsqueeze(0)
            else:  # 3D volume
                # Process each slice
                tensors = []
                for i in range(data.shape[0]):
                    slice_rgb = np.stack([data[i]] * 3, axis=-1)
                    tensor = self.preprocess(slice_rgb)
                    tensors.append(tensor)
                return torch.stack(tensors)
        
        elif len(data.shape) == 4:  # Batch or 3D RGB
            if data.shape[-1] == 3:  # Batch of RGB images
                tensors = []
                for i in range(data.shape[0]):
                    tensor = self.preprocess(data[i])
                    tensors.append(tensor)
                return torch.stack(tensors)
        
        raise ValueError(f"Unsupported input shape: {data.shape}")
    
    def _process_2d(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Process 2D input"""
        # Ensure 4D tensor [batch, channels, height, width]
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Check dimensions
        if len(input_tensor.shape) != 4:
            raise ValueError(f"Expected 4D tensor, got {len(input_tensor.shape)}D")
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            self.features.clear()  # Clear previous features
            embeddings = self.model(input_tensor)
        
        return embeddings
    
    def _process_3d(self, input_tensor: torch.Tensor, aggregation: str = 'mean') -> torch.Tensor:
        """Process 3D volume"""
        # input_tensor shape: [slices, channels, height, width]
        if len(input_tensor.shape) != 4:
            raise ValueError(f"Expected 4D tensor for 3D processing, got {len(input_tensor.shape)}D")
        
        # Process in batches for efficiency
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, input_tensor.shape[0], batch_size):
            batch = input_tensor[i:i+batch_size].to(self.device)
            
            with torch.no_grad():
                batch_embeddings = self.model(batch)
                embeddings_list.append(batch_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings_list, dim=0)
        
        # Aggregate based on method
        if aggregation == 'mean':
            return all_embeddings.mean(dim=0, keepdim=True)
        elif aggregation == 'max':
            return all_embeddings.max(dim=0, keepdim=True)[0]
        elif aggregation == 'concat':
            return all_embeddings.flatten()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    def process_batch(self, 
                     images: List[Union[np.ndarray, torch.Tensor]], 
                     batch_size: int = 32) -> torch.Tensor:
        """Process batch of images efficiently
        
        Args:
            images: List of images (numpy arrays or tensors)
            batch_size: Batch size for processing
            
        Returns:
            Batch embeddings tensor
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                if isinstance(img, np.ndarray):
                    tensor = self._preprocess_numpy(img)
                else:
                    tensor = img
                
                if len(tensor.shape) == 3:
                    tensor = tensor.unsqueeze(0)
                
                batch_tensors.append(tensor)
            
            # Stack batch
            batch = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(batch)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def extract_multi_scale_features(self, 
                                   input_data: Union[torch.Tensor, np.ndarray],
                                   scales: List[float] = [0.5, 1.0, 1.5]) -> Dict[float, torch.Tensor]:
        """Extract features at multiple scales
        
        Args:
            input_data: Input image or tensor
            scales: List of scales to process
            
        Returns:
            Dictionary mapping scale to features
        """
        multi_scale_features = {}
        
        # Convert to tensor if needed
        if isinstance(input_data, np.ndarray):
            base_tensor = self._preprocess_numpy(input_data)
        else:
            base_tensor = input_data
        
        # Ensure 4D tensor
        if len(base_tensor.shape) == 3:
            base_tensor = base_tensor.unsqueeze(0)
        
        # Process at each scale
        for scale in scales:
            if scale != 1.0:
                # Resize tensor
                scaled_size = int(self.input_size * scale)
                scaled_tensor = F.interpolate(
                    base_tensor, 
                    size=(scaled_size, scaled_size),
                    mode='bilinear',
                    align_corners=False
                )
                # Resize back to original size
                scaled_tensor = F.interpolate(
                    scaled_tensor,
                    size=(self.input_size, self.input_size),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_tensor = base_tensor
            
            # Extract features
            features = self._process_2d(scaled_tensor)
            multi_scale_features[scale] = features
        
        return multi_scale_features
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of output embeddings"""
        if "DenseNet121" in self.model_name:
            return 1024
        elif "ResNet50" in self.model_name:
            return 2048
        elif "InceptionV3" in self.model_name:
            return 2048
        else:
            # Try to infer from model
            if hasattr(self.model, 'fc'):
                if hasattr(self.model.fc, 'in_features'):
                    return self.model.fc.in_features
            elif hasattr(self.model, 'classifier'):
                if hasattr(self.model.classifier, 'in_features'):
                    return self.model.classifier.in_features
            return 1000  # Default ImageNet output
