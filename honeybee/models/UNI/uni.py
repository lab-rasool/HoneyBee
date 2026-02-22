import warnings

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms


class UNI:
    def __init__(self, model_path) -> None:
        warnings.warn(
            "UNI is deprecated. Use PathologyProcessor(model='uni') with the registry "
            "system or honeybee.models.registry.load_model('uni'). "
            "This class will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
        self.model = model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        print(f"UNI model loaded on: {self.device}")

    def load_model_and_predict(self, patches):
        self.model.eval()

        if isinstance(patches, np.ndarray):
            if len(patches.shape) == 4:
                pil_images = [Image.fromarray(patch) for patch in patches]
            else:
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

        image_tensors = []
        for img in pil_images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_tensors.append(self.transform(img))

        batch_tensor = torch.stack(image_tensors).to(self.device)

        with torch.inference_mode():
            feature_emb = self.model(batch_tensor)

        return feature_emb
