import torch
import torch.nn as nn
from torchvision import models, transforms


class TissueDetector:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self, model_path):
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model.to(self.device).eval()
