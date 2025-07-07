"""
AI Integration Module with RadImageNet

Integrates pre-trained RadImageNet models for radiological image analysis:
- DenseNet121, ResNet50, InceptionV3 trained on radiological images
- High-quality embeddings generation
- 2D and 3D processing modes
- GPU acceleration support
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional, Dict, List
import torchvision.transforms as transforms
from torchvision import models
import logging
from pathlib import Path
import requests
import os

logger = logging.getLogger(__name__)


class RadImageNetProcessor:
    """
    RadImageNet model processor for medical image embeddings

    RadImageNet models are pre-trained on millions of radiological images
    and provide better features for medical imaging tasks compared to
    natural image models.
    """


    def __init__(
        self,
        model_name: str = "densenet121",
        pretrained: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize RadImageNet processor

        Args:
            model_name: Model architecture ('densenet121', 'resnet50', 'inception_v3')
            pretrained: Use pretrained weights
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name.lower()
        self.pretrained = pretrained

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = self._init_model()
        self.model.to(self.device)
        self.model.eval()

        # Set up preprocessing
        self.preprocess = self._get_preprocessing()

        # Embedding dimension
        self.embedding_dim = self._get_embedding_dim()

    def _init_model(self) -> nn.Module:
        """Initialize the model architecture"""
        if self.model_name == "densenet121":
            model = models.densenet121(pretrained=False)
            # Modify for single channel input
            model.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Remove classifier for embeddings
            model.classifier = nn.Identity()
            embedding_dim = 1024

        elif self.model_name == "resnet50":
            model = models.resnet50(pretrained=False)
            # Modify for single channel input
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Remove FC layer for embeddings
            model.fc = nn.Identity()
            embedding_dim = 2048

        elif self.model_name == "inception_v3":
            model = models.inception_v3(pretrained=False, aux_logits=False)
            # Modify for single channel input
            model.Conv2d_1a_3x3.conv = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, bias=False
            )
            # Remove FC layer for embeddings
            model.fc = nn.Identity()
            embedding_dim = 2048

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # Load pretrained weights if requested
        if self.pretrained:
            self._load_pretrained_weights(model)

        return model

    def _load_pretrained_weights(self, model: nn.Module):
        """Load pretrained RadImageNet weights from Hugging Face Hub"""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface-hub")
            logger.info("Falling back to ImageNet initialization...")
            self._fallback_to_imagenet(model)
            return

        # Map model names to Hugging Face file names
        model_file_map = {
            "densenet121": "DenseNet121.pt",
            "resnet50": "ResNet50.pt",
            "inception_v3": "InceptionV3.pt"
        }

        if self.model_name not in model_file_map:
            logger.error(f"Unknown model name: {self.model_name}")
            self._fallback_to_imagenet(model)
            return

        try:
            # Download weights from Hugging Face Hub
            logger.info(f"Downloading RadImageNet weights for {self.model_name} from Hugging Face Hub...")
            
            weight_file = hf_hub_download(
                repo_id="Lab-Rasool/RadImageNet",
                filename=model_file_map[self.model_name],
                repo_type="model"
            )
            
            logger.info(f"Loading RadImageNet weights from {weight_file}")
            state_dict = torch.load(weight_file, map_location=self.device)
            
            # Load the entire model if it's a complete model checkpoint
            if isinstance(state_dict, nn.Module):
                # Extract state dict from the model
                model.load_state_dict(state_dict.state_dict(), strict=False)
            else:
                # Direct state dict loading
                model.load_state_dict(state_dict, strict=False)
                
            logger.info("Successfully loaded RadImageNet weights")
            
        except Exception as e:
            logger.error(f"Failed to load RadImageNet weights: {e}")
            logger.info("Falling back to ImageNet initialization...")
            self._fallback_to_imagenet(model)

    def _fallback_to_imagenet(self, model: nn.Module):
        """Fallback to ImageNet weights when RadImageNet weights are not available"""
        if self.model_name == "densenet121":
            imagenet_model = models.densenet121(pretrained=True)
            # Copy weights except first conv layer
            state_dict = imagenet_model.state_dict()
            model_dict = model.state_dict()

            # Filter out first conv layer
            pretrained_dict = {
                k: v
                for k, v in state_dict.items()
                if k != "features.conv0.weight" and k in model_dict
            }

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

        elif self.model_name == "resnet50":
            imagenet_model = models.resnet50(pretrained=True)
            state_dict = imagenet_model.state_dict()
            model_dict = model.state_dict()

            pretrained_dict = {
                k: v
                for k, v in state_dict.items()
                if k != "conv1.weight" and k in model_dict
            }

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

        elif self.model_name == "inception_v3":
            imagenet_model = models.inception_v3(pretrained=True)
            # Remove aux_logits from the state dict
            state_dict = imagenet_model.state_dict()
            model_dict = model.state_dict()

            # Filter out first conv layer and aux classifier
            pretrained_dict = {
                k: v
                for k, v in state_dict.items()
                if k != "Conv2d_1a_3x3.conv.weight"
                and not k.startswith("AuxLogits")
                and k in model_dict
            }

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)

    def _get_preprocessing(self) -> transforms.Compose:
        """Get preprocessing pipeline for the model"""
        # RadImageNet models expect:
        # - Single channel input (grayscale)
        # - Normalized to [0, 1] or standardized
        # - Specific input size based on architecture

        if self.model_name == "inception_v3":
            input_size = 299
        else:
            input_size = 224

        return transforms.Compose(
            [
                transforms.ToPILImage(mode="L"),  # Grayscale
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
            ]
        )

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension for the model"""
        dims = {"densenet121": 1024, "resnet50": 2048, "inception_v3": 2048}
        return dims[self.model_name]

    def generate_embeddings(
        self, image: np.ndarray, mode: str = "2d", aggregation: str = "mean"
    ) -> np.ndarray:
        """
        Generate embeddings from medical image

        Args:
            image: Input image (2D or 3D)
            mode: Processing mode ('2d' for single slice, '3d' for volume)
            aggregation: How to aggregate 3D embeddings ('mean', 'max', 'concat')
        """
        if mode == "2d":
            if len(image.shape) == 3:
                # Process middle slice for 3D volume
                middle_slice = image.shape[0] // 2
                return self._process_2d_slice(image[middle_slice])
            else:
                return self._process_2d_slice(image)

        elif mode == "3d":
            if len(image.shape) != 3:
                raise ValueError("3D mode requires 3D volume")
            return self._process_3d_volume(image, aggregation)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _process_2d_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Process single 2D slice"""
        # Normalize to 0-255 range for preprocessing
        slice_norm = self._normalize_slice(slice_2d)

        # Preprocess
        input_tensor = self.preprocess(slice_norm)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Generate embedding
        with torch.no_grad():
            embedding = self.model(input_batch)

        return embedding.cpu().numpy().squeeze()

    def _process_3d_volume(
        self, volume: np.ndarray, aggregation: str = "mean"
    ) -> np.ndarray:
        """Process 3D volume"""
        embeddings = []

        # Process each slice
        for i in range(volume.shape[0]):
            slice_embedding = self._process_2d_slice(volume[i])
            embeddings.append(slice_embedding)

        embeddings = np.stack(embeddings)

        # Aggregate embeddings
        if aggregation == "mean":
            return embeddings.mean(axis=0)
        elif aggregation == "max":
            return embeddings.max(axis=0)
        elif aggregation == "concat":
            return embeddings.flatten()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def _normalize_slice(self, slice_2d: np.ndarray) -> np.ndarray:
        """Normalize slice to 0-255 range"""
        # Handle different input ranges
        if slice_2d.min() < 0:
            # Likely HU values, window to soft tissue
            windowed = np.clip(slice_2d, -150, 250)
            normalized = (windowed - windowed.min()) / (
                windowed.max() - windowed.min() + 1e-8
            )
        else:
            # Already positive, just normalize
            normalized = (slice_2d - slice_2d.min()) / (
                slice_2d.max() - slice_2d.min() + 1e-8
            )

        return (normalized * 255).astype(np.uint8)

    def process_batch(
        self, images: List[np.ndarray], batch_size: int = 32
    ) -> np.ndarray:
        """Process batch of images efficiently"""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                if len(img.shape) == 3:
                    img = img[img.shape[0] // 2]  # Middle slice

                img_norm = self._normalize_slice(img)
                tensor = self.preprocess(img_norm)
                batch_tensors.append(tensor)

            # Stack into batch
            batch = torch.stack(batch_tensors).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(batch)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def extract_features(
        self, image: np.ndarray, layer_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from multiple layers

        Args:
            image: Input image
            layer_name: Specific layer to extract from (if None, extract from multiple)
        """
        features = {}

        # Register hooks to extract features
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach().cpu().numpy()

            return hook

        # Register hooks based on model
        if self.model_name == "densenet121":
            hooks = [
                self.model.features.denseblock1.register_forward_hook(
                    get_features("block1")
                ),
                self.model.features.denseblock2.register_forward_hook(
                    get_features("block2")
                ),
                self.model.features.denseblock3.register_forward_hook(
                    get_features("block3")
                ),
                self.model.features.denseblock4.register_forward_hook(
                    get_features("block4")
                ),
            ]
        elif self.model_name == "resnet50":
            hooks = [
                self.model.layer1.register_forward_hook(get_features("layer1")),
                self.model.layer2.register_forward_hook(get_features("layer2")),
                self.model.layer3.register_forward_hook(get_features("layer3")),
                self.model.layer4.register_forward_hook(get_features("layer4")),
            ]

        # Process image
        if len(image.shape) == 3:
            image = image[image.shape[0] // 2]

        img_norm = self._normalize_slice(image)
        input_tensor = self.preprocess(img_norm).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Filter by layer name if specified
        if layer_name and layer_name in features:
            return {layer_name: features[layer_name]}

        return features


def create_embedding_model(
    model_name: str = "densenet121", pretrained: bool = True
) -> RadImageNetProcessor:
    """Create RadImageNet embedding model"""
    return RadImageNetProcessor(model_name, pretrained)


def generate_embeddings(
    image: np.ndarray, model_name: str = "densenet121", mode: str = "2d"
) -> np.ndarray:
    """Generate embeddings using RadImageNet"""
    processor = RadImageNetProcessor(model_name)
    return processor.generate_embeddings(image, mode)


def load_pretrained_model(
    model_name: str = "densenet121", device: Optional[str] = None
) -> nn.Module:
    """Load pretrained RadImageNet model"""
    processor = RadImageNetProcessor(model_name, device=device)
    return processor.model


def process_2d_slices(
    volume: np.ndarray,
    model_name: str = "densenet121",
    slice_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Process specific 2D slices from volume"""
    processor = RadImageNetProcessor(model_name)

    if slice_indices is None:
        # Process every 10th slice by default
        slice_indices = list(range(0, volume.shape[0], 10))

    embeddings = []
    for idx in slice_indices:
        if idx < volume.shape[0]:
            embedding = processor._process_2d_slice(volume[idx])
            embeddings.append(embedding)

    return np.stack(embeddings)


def process_3d_volume(
    volume: np.ndarray, model_name: str = "densenet121", aggregation: str = "mean"
) -> np.ndarray:
    """Process entire 3D volume"""
    processor = RadImageNetProcessor(model_name)
    return processor._process_3d_volume(volume, aggregation)


class MultiModalFusion:
    """Fusion of multiple imaging modalities using RadImageNet"""

    def __init__(self, fusion_method: str = "concatenate"):
        """
        Initialize multi-modal fusion

        Args:
            fusion_method: How to fuse modalities ('concatenate', 'attention', 'learned')
        """
        self.fusion_method = fusion_method
        self.processors = {}

    def add_modality(self, modality_name: str, model_name: str = "densenet121"):
        """Add a modality processor"""
        self.processors[modality_name] = RadImageNetProcessor(model_name)

    def fuse(self, modality_images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse multiple modality images

        Args:
            modality_images: Dictionary mapping modality name to image
        """
        embeddings = {}

        # Generate embeddings for each modality
        for modality, image in modality_images.items():
            if modality in self.processors:
                embedding = self.processors[modality].generate_embeddings(image)
                embeddings[modality] = embedding
            else:
                logger.warning(f"No processor for modality: {modality}")

        # Fuse embeddings
        if self.fusion_method == "concatenate":
            return np.concatenate(list(embeddings.values()))

        elif self.fusion_method == "attention":
            # Simple attention-based fusion
            embeddings_array = np.stack(list(embeddings.values()))

            # Compute attention weights (simplified)
            attention_scores = np.sum(embeddings_array**2, axis=1)
            # Normalize scores to prevent overflow
            attention_scores = attention_scores / attention_scores.max()
            attention_weights = np.exp(attention_scores) / np.sum(
                np.exp(attention_scores)
            )

            # Weighted sum
            fused = np.sum(embeddings_array * attention_weights[:, np.newaxis], axis=0)
            return fused

        elif self.fusion_method == "learned":
            # Placeholder for learned fusion
            logger.warning("Learned fusion not implemented, using concatenation")
            return np.concatenate(list(embeddings.values()))

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


if __name__ == "__main__":
    # Example usage
    import sys
    from data_management import load_medical_image
    from preprocessing import preprocess_ct

    if len(sys.argv) > 1:
        # Load and preprocess image
        image, metadata = load_medical_image(sys.argv[1])

        if metadata.modality == "CT":
            image = preprocess_ct(image, window="lung")

        # Generate embeddings
        processor = RadImageNetProcessor("densenet121")

        # 2D embedding from middle slice
        embedding_2d = processor.generate_embeddings(image, mode="2d")
        print(f"2D embedding shape: {embedding_2d.shape}")

        # 3D embedding if volume
        if len(image.shape) == 3:
            embedding_3d = processor.generate_embeddings(image, mode="3d")
            print(f"3D embedding shape: {embedding_3d.shape}")

        # Extract multi-scale features
        features = processor.extract_features(image)
        for layer, feat in features.items():
            print(f"{layer} features shape: {feat.shape}")
