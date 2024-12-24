import cv2
import numpy as np
import pydicom


class DICOMPreprocessor:
    def __init__(
        self,
        target_size=None,
        window_center=None,
        window_width=None,
        normalize=True,
        to_three_channels=False,
    ):
        self.target_size = target_size
        self.window_center = window_center
        self.window_width = window_width
        self.normalize = normalize
        self.to_three_channels = to_three_channels

    def load_dicom(self, dicom_path):
        dicom_data = pydicom.dcmread(dicom_path)
        return dicom_data.pixel_array.astype(np.float32)

    def apply_windowing(self, image):
        if self.window_center is not None and self.window_width is not None:
            min_value = self.window_center - self.window_width // 2
            max_value = self.window_center + self.window_width // 2
            image = np.clip(image, min_value, max_value)
            image = (image - min_value) / (max_value - min_value)
        return image

    def resize_image(self, image):
        if self.target_size is not None:
            return cv2.resize(
                image,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return image

    def pad_or_crop(self, image):
        if self.target_size is not None:
            current_size = image.shape[0]  # Assuming square images
            if current_size < self.target_size:
                pad_amount = (self.target_size - current_size) // 2
                image = np.pad(
                    image,
                    ((pad_amount, pad_amount), (pad_amount, pad_amount)),
                    mode="constant",
                )
            elif current_size > self.target_size:
                start = (current_size - self.target_size) // 2
                image = image[
                    start : start + self.target_size, start : start + self.target_size
                ]
        return image

    def normalize_image(self, image):
        if self.normalize:
            min_val = np.min(image)
            max_val = np.max(image)
            return (
                (image - min_val) / (max_val - min_val) if max_val > min_val else image
            )
        return image

    def convert_to_three_channels(self, image):
        if self.to_three_channels and image.ndim < 3:
            return np.stack((image,) * 3, axis=-1)
        return image

    def preprocess(self, dicom_path):
        image = self.load_dicom(dicom_path)
        image = self.apply_windowing(image)
        image = self.normalize_image(image)
        image = self.resize_image(image)
        image = self.pad_or_crop(image)
        image = self.convert_to_three_channels(image)
        return image
