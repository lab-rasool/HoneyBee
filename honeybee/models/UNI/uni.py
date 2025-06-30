import timm
import torch


class UNI:
    def __init__(self, model_path) -> None:
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(model_path, map_location=self.device), strict=True
        )
        self.model = model.to(self.device)
        print(f"UNI model loaded on: {self.device}")

    def load_model_and_predict(self, patches):
        self.model.eval()

        # convert patches from ndarray to torch.Tensor and move to GPU
        patches = torch.tensor(patches, dtype=torch.float32).to(self.device)
        patches = patches.permute(0, 3, 1, 2)

        # use patches as input
        # patches should be a torch.Tensor with shape [batch_size, 3, 224, 224]
        with torch.inference_mode():
            feature_emb = self.model(patches)

        return feature_emb
