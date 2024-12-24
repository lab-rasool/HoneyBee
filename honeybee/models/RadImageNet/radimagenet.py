import torch
import torch.nn as nn
from torchvision import models


class RadImageNet:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        if "DenseNet121" in self.model_path:
            self.model = models.densenet121(weights=None)
        elif "InceptionV3" in self.model_path:
            self.model = models.inception_v3(weights=None)
        elif "ResNet50" in self.model_path:
            self.model = models.resnet50(weights=None)
        else:
            raise ValueError(
                "Model not recognized. Ensure the model path contains one of 'DenseNet121', 'InceptionV3', or 'ResNet50'."
            )

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)  # Move model to the appropriate device (GPU or CPU)
        self.model.eval()  # Set to evaluation mode

    def load_model_and_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Check input dimensions
        if self.model is None:
            raise ValueError("Model is not loaded.")
        if len(input_tensor.shape) != 4:
            raise ValueError(
                "Input tensor must be of shape [slices, 3, 224 or 299, 224 or 299]."
            )

        # Handle input size for InceptionV3
        if isinstance(self.model, models.Inception3):
            if input_tensor.shape[2:] != (299, 299):
                raise ValueError(
                    "InceptionV3 model requires input size of [slices, 3, 299, 299]."
                )
        else:
            if input_tensor.shape[2:] != (224, 224):
                raise ValueError(
                    "DenseNet121 and ResNet50 models require input size of [slices, 3, 224, 224]."
                )

        # Move input tensor to the same device as the model
        input_tensor = input_tensor.to(self.device)

        # Forward pass to generate embeddings
        with torch.no_grad():
            feature_emb = self.model(input_tensor)

        return feature_emb
