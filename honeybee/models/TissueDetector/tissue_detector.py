import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from huggingface_hub import hf_hub_download


class TissueDetector:
    def __init__(self, model_path=None, device="cuda"):
        self.device = torch.device(device)
        
        # If no model path provided, download the weights automatically
        if model_path is None:
            model_path = self._download_weights()
        
        self.model = self._load_model(model_path)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _download_weights(self):
        """Download model weights from HuggingFace Hub if not available locally."""
        # First check if weights exist locally
        local_weights_dir = Path(__file__).parent / "weights"
        local_weights_path = local_weights_dir / "deep-tissue-detector_densenet_state-dict.pt"

        if local_weights_path.exists():
            print(f"Using existing local weights from {local_weights_path}")
            return str(local_weights_path)

        # Try to download from HuggingFace Hub
        print("Downloading tissue detector weights from HuggingFace Hub...")
        try:
            # Try SafeTensors format first (recommended)
            try:
                model_path = hf_hub_download(
                    repo_id="Lab-Rasool/tissue-detector",
                    filename="model.safetensors",
                    cache_dir=local_weights_dir.parent / ".cache"
                )
                print(f"✓ Downloaded SafeTensors weights from HuggingFace Hub")
                return model_path
            except Exception as e:
                print(f"SafeTensors not available, trying PyTorch format: {e}")

            # Fallback to PyTorch format
            model_path = hf_hub_download(
                repo_id="Lab-Rasool/tissue-detector",
                filename="deep-tissue-detector_densenet_state-dict.pt",
                cache_dir=local_weights_dir.parent / ".cache"
            )
            print(f"✓ Downloaded PyTorch weights from HuggingFace Hub")
            return model_path

        except Exception as e:
            raise RuntimeError(
                f"Failed to download model weights from HuggingFace Hub: {e}\n"
                f"Please ensure you have internet connectivity and huggingface_hub is installed.\n"
                f"You can install it with: pip install huggingface_hub"
            )

    def _load_model(self, model_path):
        """Load DenseNet121 model with custom classifier."""
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 3)

        # Load weights based on file format
        model_path_str = str(model_path)
        if model_path_str.endswith('.safetensors'):
            # Load SafeTensors format
            try:
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                model.load_state_dict(state_dict)
                print("✓ Loaded model from SafeTensors format")
            except ImportError:
                raise RuntimeError(
                    "SafeTensors format detected but safetensors package not installed.\n"
                    "Install it with: pip install safetensors"
                )
        else:
            # Load PyTorch format
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print("✓ Loaded model from PyTorch format")

        return model.to(self.device).eval()
