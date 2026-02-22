import os
import subprocess
import warnings

import cv2
import numpy as np
import onnxruntime as ort

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class REMEDIS:
    def __init__(self, model_path=None):
        warnings.warn(
            "REMEDIS is deprecated. Use PathologyProcessor(model='remedis') with the "
            "registry system or honeybee.models.registry.load_model('remedis'). "
            "This class will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model_path = model_path

    @staticmethod
    def _preprocess(patches):
        """Resize to 224x224 and apply ImageNet normalization (NHWC float32)."""
        out = np.empty((len(patches), 224, 224, 3), dtype=np.float32)
        for i, patch in enumerate(patches):
            resized = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
            out[i] = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        return out

    def load_model_and_predict(self, patches):
        if self.model_path is None:
            raise ValueError("model_path is required for REMEDIS inference")
        patches = self._preprocess(patches)
        sess = ort.InferenceSession(
            self.model_path,
            providers=[
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "gpu_mem_limit": 24 * 1024 * 1024 * 1024,  # 24GB
                    },
                ),
                "CPUExecutionProvider",
            ],
        )
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run([label_name], {input_name: patches})[0]
        return pred_onnx

    def convert_models_to_onnx(self, source_dir, target_dir):
        """
        Example usage:
        convert_models_to_onnx(
            source_dir="/mnt/d/Models/REMEDIS/Pretrained-Weights",
            target_dir="/mnt/d/Models/REMEDIS/onnx",
        )
        """

        os.makedirs(target_dir, exist_ok=True)
        for model_name in os.listdir(source_dir):
            model_path = os.path.join(source_dir, model_name)
            onnx_model_path = os.path.join(target_dir, f"{model_name}.onnx")
            if os.path.exists(onnx_model_path):
                print(f"Skipping {model_name} as it already exists.")
                continue
            if os.path.isdir(model_path):
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "tf2onnx.convert",
                        "--saved-model",
                        model_path,
                        "--output",
                        onnx_model_path,
                    ]
                )
                print(f"Converted {model_name} to ONNX format.")
