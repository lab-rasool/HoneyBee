import timm
import torch


class UNI:
    def __init__(self) -> None:
        pass

    def load_model_and_predict(self, model_path, patches):
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        model.load_state_dict(
            torch.load(model_path, map_location=map_location), strict=True
        )
        model.eval()

        # convert patches from ndarray to torch.Tensor
        patches = torch.tensor(patches, dtype=torch.float32)
        patches = patches.permute(0, 3, 1, 2)
        print(patches.shape)

        # use patches as input
        # patches should be a torch.Tensor with shape [batch_size, 3, 224, 224]
        with torch.inference_mode():
            feature_emb = model(patches)

        return feature_emb
