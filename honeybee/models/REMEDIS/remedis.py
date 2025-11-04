import os
import subprocess

import onnxruntime as ort


class REMEDIS:
    def __init__(self):
        pass

    def load_model_and_predict(model_path, patches):
        sess = ort.InferenceSession(
            model_path,
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
