import torch
import torch.nn as nn
from torchvision import models, transforms
import requests
from pathlib import Path


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
        """Download model weights from GitHub if not available locally."""
        weights_url = "https://github.com/markowetzlab/slidl/raw/main/slidl/models/deep-tissue-detector_densenet_state-dict.pt"
        weights_dir = Path(__file__).parent / "weights"
        weights_path = weights_dir / "deep-tissue-detector_densenet_state-dict.pt"
        
        # Create weights directory if it doesn't exist
        weights_dir.mkdir(exist_ok=True)
        
        # Download weights if file doesn't exist
        if not weights_path.exists():
            print(f"Downloading model weights from {weights_url}...")
            try:
                response = requests.get(weights_url, stream=True)
                response.raise_for_status()
                
                with open(weights_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"Model weights downloaded successfully to {weights_path}")
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download model weights: {e}")
        else:
            print(f"Using existing model weights from {weights_path}")
        
        return str(weights_path)

    def _load_model(self, model_path):
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device).eval()
