import imageio.v3 as iio
import numpy as np
from skimage.color import gray2rgb
from skimage.transform import resize


class Scan:
    def __init__(self, path):
        self.path = path

    def prepare_volume_or_image(self, vol, target_size=(224, 224)):
        # Normalize the volume or image first
        vol = vol / np.max(vol)

        if vol.ndim == 3:  # If the input is a volume
            # Resize each slice and convert to RGB
            resized_and_colored = np.array(
                [
                    gray2rgb(resize(slice, target_size, anti_aliasing=True))
                    for slice in vol
                ]
            )
        elif vol.ndim == 2:  # If the input is a single image
            resized = resize(vol, target_size, anti_aliasing=True)
            resized_and_colored = gray2rgb(resized)
            resized_and_colored = np.expand_dims(resized_and_colored, axis=0)
        else:
            raise ValueError("Unsupported volume/image shape")

        return resized_and_colored

    def load_patches(self, target_patch_size):
        print(f"Loading patches from {self.path}")
        im_read = iio.imread(self.path, plugin="DICOM")
        print(f"Loaded with shape {im_read.shape} and {im_read.ndim} dimensions")
        patches = self.prepare_volume_or_image(
            im_read, target_size=(target_patch_size, target_patch_size)
        )
        patches = patches.astype(np.float32)
        return patches
