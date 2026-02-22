"""
Radiology Preprocessing Components

Comprehensive preprocessing capabilities for medical images including:
- Denoising (NLM, TV, bilateral, PET-specific)
- Artifact reduction (metal, motion, ring, beam hardening)
- Intensity normalization
- Window/level adjustment
"""

import logging
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, median_filter
from skimage import exposure
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle

logger = logging.getLogger(__name__)


class Denoiser:
    """Advanced denoising methods for medical images

    Supported methods:
        - nlm: Non-local means denoising
        - tv: Total variation denoising
        - bilateral: Bilateral filtering
        - median: Median filtering
        - gaussian: Gaussian filtering
        - pet_specific: PET-specific iterative denoising
    """

    def __init__(self, method: str = "nlm"):
        """Initialize denoiser

        Args:
            method: Denoising method to use
        """
        self.method = method.lower()
        self.supported_methods = [
            "nlm", "tv", "bilateral", "median", "gaussian", "pet_specific", "rician", "deep",
            "dipy_nlm", "dipy_mppca",
        ]

        if self.method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported. Choose from {self.supported_methods}")

    def denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply denoising to image

        Args:
            image: Input image (2D or 3D)
            **kwargs: Method-specific parameters

        Returns:
            Denoised image
        """
        if self.method == "nlm":
            return self._nlm_denoise(image, **kwargs)
        elif self.method == "tv":
            return self._tv_denoise(image, **kwargs)
        elif self.method == "bilateral":
            return self._bilateral_denoise(image, **kwargs)
        elif self.method == "median":
            return self._median_denoise(image, **kwargs)
        elif self.method == "gaussian":
            return self._gaussian_denoise(image, **kwargs)
        elif self.method == "pet_specific":
            return self._pet_specific_denoise(image, **kwargs)
        elif self.method == "rician":
            return self._rician_denoise(image, **kwargs)
        elif self.method == "deep":
            return self._deep_denoise(image, **kwargs)
        elif self.method == "dipy_nlm":
            return self._dipy_nlm_denoise(image, **kwargs)
        elif self.method == "dipy_mppca":
            return self._dipy_mppca_denoise(image, **kwargs)

    def _nlm_denoise(
        self, image: np.ndarray, patch_size: int = 5, patch_distance: int = 6, h: float = 0.1
    ) -> np.ndarray:
        """Non-local means denoising"""
        # Normalize to 0-1 range for skimage
        img_min, img_max = image.min(), image.max()
        img_norm = (image - img_min) / (img_max - img_min + 1e-8)

        # Apply NLM
        if len(image.shape) == 2:
            denoised = denoise_nl_means(
                img_norm, patch_size=patch_size, patch_distance=patch_distance, h=h
            )
        else:
            # Process slice by slice for 3D
            denoised = np.zeros_like(img_norm)
            for i in range(image.shape[0]):
                denoised[i] = denoise_nl_means(
                    img_norm[i], patch_size=patch_size, patch_distance=patch_distance, h=h
                )

        # Restore original range
        return denoised * (img_max - img_min) + img_min

    def _tv_denoise(self, image: np.ndarray, weight: float = 0.1) -> np.ndarray:
        """Total variation denoising"""
        # Normalize
        img_min, img_max = image.min(), image.max()
        img_norm = (image - img_min) / (img_max - img_min + 1e-8)

        # Apply TV denoising
        if len(image.shape) == 2:
            denoised = denoise_tv_chambolle(img_norm, weight=weight)
        else:
            denoised = np.zeros_like(img_norm)
            for i in range(image.shape[0]):
                denoised[i] = denoise_tv_chambolle(img_norm[i], weight=weight)

        return denoised * (img_max - img_min) + img_min

    def _bilateral_denoise(
        self, image: np.ndarray, sigma_color: float = 0.05, sigma_spatial: float = 15
    ) -> np.ndarray:
        """Bilateral filtering"""
        # For 2D images
        if len(image.shape) == 2:
            # Convert to uint8 for OpenCV
            img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            denoised = cv2.bilateralFilter(img_norm, d=9, sigmaColor=75, sigmaSpace=75)
            # Convert back
            return denoised.astype(np.float32) * (image.max() - image.min()) / 255 + image.min()
        else:
            # Process slice by slice
            denoised = np.zeros_like(image)
            for i in range(image.shape[0]):
                img_norm = cv2.normalize(image[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                denoised[i] = cv2.bilateralFilter(img_norm, d=9, sigmaColor=75, sigmaSpace=75)
            return denoised.astype(np.float32) * (image.max() - image.min()) / 255 + image.min()

    def _median_denoise(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        """Median filtering"""
        return median_filter(image, size=size)

    def _gaussian_denoise(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian filtering"""
        return gaussian_filter(image, sigma=sigma)

    def _pet_specific_denoise(
        self, image: np.ndarray, iterations: int = 5, kernel_size: int = 3
    ) -> np.ndarray:
        """PET-specific denoising using iterative filtering"""
        denoised = image.copy()

        for _ in range(iterations):
            # Apply median filter to reduce noise
            denoised = median_filter(denoised, size=kernel_size)

            # Apply edge-preserving smoothing
            denoised = self._anisotropic_diffusion(denoised, iterations=10)

        return denoised

    def _anisotropic_diffusion(
        self, image: np.ndarray, iterations: int = 10, kappa: float = 50, gamma: float = 0.1
    ) -> np.ndarray:
        """Perona-Malik anisotropic diffusion"""
        img = image.copy()

        for _ in range(iterations):
            # Calculate gradients
            nabla_n = np.roll(img, -1, axis=0) - img
            nabla_s = np.roll(img, 1, axis=0) - img
            nabla_e = np.roll(img, -1, axis=1) - img
            nabla_w = np.roll(img, 1, axis=1) - img

            # Calculate diffusion coefficients
            c_n = np.exp(-((nabla_n / kappa) ** 2))
            c_s = np.exp(-((nabla_s / kappa) ** 2))
            c_e = np.exp(-((nabla_e / kappa) ** 2))
            c_w = np.exp(-((nabla_w / kappa) ** 2))

            # Update image
            img += gamma * (c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w)

        return img

    def _rician_denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Rician noise correction and denoising for MRI.

        Estimates noise sigma from background, applies Rician bias correction,
        then NLM denoising on the corrected image.
        """
        # Estimate noise sigma from lowest 25th percentile region
        background = image[image <= np.percentile(image, 25)]
        if len(background) == 0 or background.std() == 0:
            sigma = max(image.std() * 0.05, 1e-8)
        else:
            # For Rician noise, sigma ≈ mode of background / 0.655
            sigma = float(np.median(np.abs(background - np.median(background)))) / 0.655
            sigma = max(sigma, 1e-8)

        # Apply Rician bias correction: corrected = sqrt(max(0, image^2 - 2*sigma^2))
        image_float = image.astype(np.float64)
        corrected = np.sqrt(np.maximum(0, image_float**2 - 2 * sigma**2))

        # Apply NLM denoising on the bias-corrected image
        img_min, img_max = corrected.min(), corrected.max()
        img_range = img_max - img_min
        if img_range == 0:
            return corrected.astype(image.dtype)

        img_norm = (corrected - img_min) / img_range

        if len(image.shape) == 2:
            denoised = denoise_nl_means(img_norm, patch_size=5, patch_distance=6, h=0.08)
        else:
            denoised = np.zeros_like(img_norm)
            for i in range(image.shape[0]):
                denoised[i] = denoise_nl_means(img_norm[i], patch_size=5, patch_distance=6, h=0.08)

        return (denoised * img_range + img_min).astype(image.dtype)

    def _deep_denoise(
        self, image: np.ndarray, weights_path: Optional[str] = None, **kwargs
    ) -> np.ndarray:
        """Deep learning denoising using a DnCNN-style residual CNN.

        Falls back to NLM if torch is not available.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("PyTorch not available for deep denoising, falling back to NLM")
            return self._nlm_denoise(image)

        class DnCNN(nn.Module):
            """DnCNN-style residual denoising network."""

            def __init__(self, channels: int = 1, num_layers: int = 7, features: int = 64):
                super().__init__()
                layers = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(inplace=True)]
                for _ in range(num_layers - 2):
                    layers += [
                        nn.Conv2d(features, features, 3, padding=1),
                        nn.BatchNorm2d(features),
                        nn.ReLU(inplace=True),
                    ]
                layers.append(nn.Conv2d(features, channels, 3, padding=1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                noise = self.net(x)
                return x - noise  # residual learning

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DnCNN().to(device)

        if weights_path:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
        else:
            # Self-supervised Noise2Self approach using blind-spot masking
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Prepare training data from the input image itself
            if len(image.shape) == 2:
                slices = [image]
            else:
                # Use a subset of slices for efficiency
                step = max(1, image.shape[0] // 16)
                slices = [image[i] for i in range(0, image.shape[0], step)]

            img_min, img_max = image.min(), image.max()
            img_range = img_max - img_min if img_max != img_min else 1.0

            for epoch in range(30):
                for sl in slices:
                    sl_norm = (sl.astype(np.float32) - img_min) / img_range
                    tensor = torch.from_numpy(sl_norm).unsqueeze(0).unsqueeze(0).to(device)

                    # Blind-spot masking: mask random pixels
                    mask = (torch.rand_like(tensor) > 0.3).float()
                    masked_input = tensor * mask

                    output = model(masked_input)
                    # Loss only on masked-out pixels
                    loss = ((output - tensor) ** 2 * (1 - mask)).sum() / (
                        (1 - mask).sum() + 1e-8
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()

        # Apply denoising
        img_min, img_max = image.min(), image.max()
        img_range = img_max - img_min if img_max != img_min else 1.0

        def _denoise_slice(sl):
            sl_norm = (sl.astype(np.float32) - img_min) / img_range
            tensor = torch.from_numpy(sl_norm).unsqueeze(0).unsqueeze(0).to(device)
            with torch.inference_mode():
                output = model(tensor)
            return output.squeeze().cpu().numpy() * img_range + img_min

        if len(image.shape) == 2:
            return _denoise_slice(image).astype(image.dtype)
        else:
            result = np.zeros_like(image, dtype=np.float64)
            for i in range(image.shape[0]):
                result[i] = _denoise_slice(image[i])
            return result.astype(image.dtype)

    def _dipy_nlm_denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Non-local means denoising using DIPY with automatic sigma estimation.

        Falls back to scikit-image NLM if DIPY is not installed.
        """
        try:
            from dipy.denoise.nlmeans import nlmeans
            from dipy.denoise.noise_estimate import estimate_sigma

            sigma = estimate_sigma(image)
            denoised = nlmeans(image, sigma=sigma, **kwargs)
            return denoised
        except ImportError:
            logger.warning("DIPY not available, falling back to scikit-image NLM")
            return self._nlm_denoise(image)

    def _dipy_mppca_denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Marchenko-Pastur PCA denoising using DIPY.

        Best suited for diffusion MRI data. Falls back to NLM if DIPY not installed.
        """
        try:
            from dipy.denoise.localpca import mppca

            denoised, _ = mppca(image, **kwargs)
            return denoised
        except ImportError:
            logger.warning("DIPY not available, falling back to scikit-image NLM")
            return self._nlm_denoise(image)


class IntensityNormalizer:
    """Intensity normalization methods for medical images

    Supported methods:
        - zscore: Z-score normalization
        - minmax: Min-max scaling
        - percentile: Percentile-based normalization
        - histogram: Histogram equalization
    """

    def __init__(self, method: str = "zscore"):
        """Initialize normalizer

        Args:
            method: Normalization method to use
        """
        self.method = method.lower()
        self.supported_methods = [
            "zscore", "minmax", "percentile", "histogram", "ct_normalize", "whitestripe",
            "torchio_znorm", "torchio_histogram", "monai_znorm",
        ]

        if self.method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported. Choose from {self.supported_methods}")

    def normalize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply normalization

        Args:
            image: Input image
            **kwargs: Method-specific parameters

        Returns:
            Normalized image
        """
        if self.method == "zscore":
            return self._zscore_normalize(image, **kwargs)
        elif self.method == "minmax":
            return self._minmax_normalize(image, **kwargs)
        elif self.method == "percentile":
            return self._percentile_normalize(image, **kwargs)
        elif self.method == "histogram":
            return self._histogram_normalize(image, **kwargs)
        elif self.method == "ct_normalize":
            return self._ct_normalize(image, **kwargs)
        elif self.method == "whitestripe":
            return self._whitestripe_normalize(image, **kwargs)
        elif self.method == "torchio_znorm":
            return self._torchio_znorm(image, **kwargs)
        elif self.method == "torchio_histogram":
            return self._torchio_histogram(image, **kwargs)
        elif self.method == "monai_znorm":
            return self._monai_znorm(image, **kwargs)

    def _zscore_normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Z-score normalization"""
        if mask is not None:
            mean = image[mask > 0].mean()
            std = image[mask > 0].std()
        else:
            mean = image.mean()
            std = image.std()

        return (image - mean) / (std + 1e-8)

    def _minmax_normalize(
        self, image: np.ndarray, out_min: float = 0.0, out_max: float = 1.0
    ) -> np.ndarray:
        """Min-max normalization"""
        in_min, in_max = image.min(), image.max()
        normalized = (image - in_min) / (in_max - in_min + 1e-8)
        return normalized * (out_max - out_min) + out_min

    def _percentile_normalize(
        self, image: np.ndarray, lower: float = 1.0, upper: float = 99.0
    ) -> np.ndarray:
        """Percentile-based normalization"""
        p_lower = np.percentile(image, lower)
        p_upper = np.percentile(image, upper)

        # Clip and normalize
        clipped = np.clip(image, p_lower, p_upper)
        return (clipped - p_lower) / (p_upper - p_lower + 1e-8)

    def _histogram_normalize(self, image: np.ndarray, nbins: int = 256) -> np.ndarray:
        """Histogram equalization"""
        # For 2D images
        if len(image.shape) == 2:
            return exposure.equalize_hist(image, nbins=nbins)
        else:
            # Process slice by slice
            normalized = np.zeros_like(image)
            for i in range(image.shape[0]):
                normalized[i] = exposure.equalize_hist(image[i], nbins=nbins)
            return normalized

    def _ct_normalize(
        self, image: np.ndarray, hu_min: float = -1024.0, hu_max: float = 3071.0
    ) -> np.ndarray:
        """CT normalization: clip to HU range then scale to [0, 1]."""
        clipped = np.clip(image, hu_min, hu_max)
        return (clipped - hu_min) / (hu_max - hu_min)

    def _whitestripe_normalize(self, image: np.ndarray, width: float = 0.05) -> np.ndarray:
        """White stripe normalization for MRI.

        Estimates the white matter peak from the intensity histogram
        and normalizes relative to it.
        """
        # Compute histogram
        nonzero = image[image > 0].flatten()
        if len(nonzero) == 0:
            return image.copy()

        hist, bin_edges = np.histogram(nonzero, bins=200)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth histogram
        from scipy.ndimage import gaussian_filter1d

        smoothed = gaussian_filter1d(hist.astype(float), sigma=3)

        # Find the largest peak (likely WM for T1)
        peak_idx = np.argmax(smoothed)
        wm_peak = bin_centers[peak_idx]

        # Define white stripe as region around peak
        stripe_lower = wm_peak * (1 - width)
        stripe_upper = wm_peak * (1 + width)

        # Get voxels in the white stripe
        stripe_mask = (image >= stripe_lower) & (image <= stripe_upper) & (image > 0)
        if stripe_mask.sum() == 0:
            return image.copy()

        stripe_mean = image[stripe_mask].mean()
        stripe_std = image[stripe_mask].std()

        if stripe_std == 0:
            return image.copy()

        return (image - stripe_mean) / stripe_std

    def _torchio_znorm(self, image: np.ndarray, masking_method: str = "mean", **kwargs) -> np.ndarray:
        """Z-normalization using TorchIO with optional masking.

        Falls back to built-in z-score normalization if TorchIO not installed.
        """
        try:
            import torch
            import torchio as tio

            # Wrap as TorchIO ScalarImage (expects 4D: C, D, H, W)
            if image.ndim == 2:
                tensor = torch.from_numpy(image[np.newaxis, np.newaxis].astype(np.float32))
            elif image.ndim == 3:
                tensor = torch.from_numpy(image[np.newaxis].astype(np.float32))
            else:
                tensor = torch.from_numpy(image.astype(np.float32))

            subject = tio.Subject(image=tio.ScalarImage(tensor=tensor))
            transform = tio.ZNormalization(masking_method=masking_method)
            transformed = transform(subject)
            result = transformed.image.numpy().squeeze()

            # Restore original shape
            if result.shape != image.shape:
                result = result.reshape(image.shape)
            return result
        except ImportError:
            logger.warning("TorchIO not available, falling back to built-in z-score")
            return self._zscore_normalize(image)

    def _torchio_histogram(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Histogram standardization using TorchIO.

        Falls back to built-in histogram equalization if TorchIO not installed.
        """
        try:
            import torch
            import torchio as tio

            if image.ndim == 2:
                tensor = torch.from_numpy(image[np.newaxis, np.newaxis].astype(np.float32))
            elif image.ndim == 3:
                tensor = torch.from_numpy(image[np.newaxis].astype(np.float32))
            else:
                tensor = torch.from_numpy(image.astype(np.float32))

            subject = tio.Subject(image=tio.ScalarImage(tensor=tensor))
            transform = tio.HistogramStandardization({"image": np.linspace(0, 1, 100)})
            transformed = transform(subject)
            result = transformed.image.numpy().squeeze()

            if result.shape != image.shape:
                result = result.reshape(image.shape)
            return result
        except ImportError:
            logger.warning("TorchIO not available, falling back to built-in histogram equalization")
            return self._histogram_normalize(image)

    def _monai_znorm(self, image: np.ndarray, nonzero: bool = True, **kwargs) -> np.ndarray:
        """Z-normalization using MONAI with nonzero masking.

        Falls back to built-in z-score normalization if MONAI not installed.
        """
        try:
            from monai.transforms import NormalizeIntensity

            transform = NormalizeIntensity(nonzero=nonzero)
            result = transform(image.astype(np.float32))
            if hasattr(result, "numpy"):
                result = result.numpy()
            return np.asarray(result)
        except ImportError:
            logger.warning("MONAI not available, falling back to built-in z-score")
            return self._zscore_normalize(image)


class WindowLevelAdjuster:
    """Window/level adjustment for medical images

    Provides preset window settings for common imaging protocols
    and automatic window/level calculation.
    """

    # Predefined window settings
    PRESETS = {
        "lung": {"center": -600, "width": 1500},
        "abdomen": {"center": 50, "width": 350},
        "bone": {"center": 400, "width": 2000},
        "brain": {"center": 40, "width": 80},
        "soft_tissue": {"center": 50, "width": 400},
        "liver": {"center": 60, "width": 150},
        "mediastinum": {"center": 50, "width": 350},
        "stroke": {"center": 35, "width": 40},
        "cta": {"center": 300, "width": 600},
        "pet": {"center": 2.5, "width": 5.0},
    }

    def __init__(self):
        self.current_window = None
        self.current_level = None

    def adjust(
        self,
        image: np.ndarray,
        window: Union[float, str] = None,
        level: float = None,
        output_range: Tuple[float, float] = (0, 255),
    ) -> np.ndarray:
        """Apply window/level adjustment

        Args:
            image: Input image
            window: Window width or preset name
            level: Window center/level
            output_range: Output intensity range

        Returns:
            Windowed image
        """
        # Handle presets
        if isinstance(window, str):
            if window.lower() not in self.PRESETS:
                raise ValueError(f"Unknown preset: {window}")
            preset = self.PRESETS[window.lower()]
            window = preset["width"]
            level = preset["center"]

        # Use defaults if not provided
        if window is None or level is None:
            # Auto window/level
            level = np.median(image)
            window = np.percentile(image, 95) - np.percentile(image, 5)

        # Apply windowing
        min_val = level - window / 2
        max_val = level + window / 2

        # Clip and scale
        windowed = np.clip(image, min_val, max_val)
        scaled = (windowed - min_val) / (max_val - min_val)

        # Scale to output range
        return scaled * (output_range[1] - output_range[0]) + output_range[0]

    def get_auto_window(
        self, image: np.ndarray, percentile_range: Tuple[float, float] = (5, 95)
    ) -> Dict[str, float]:
        """Calculate automatic window/level from image statistics

        Args:
            image: Input image
            percentile_range: Percentiles for window calculation

        Returns:
            Dictionary with 'center' and 'width' keys
        """
        p_low, p_high = np.percentile(image, percentile_range)

        center = (p_low + p_high) / 2
        width = p_high - p_low

        return {"center": center, "width": width}


class ArtifactReducer:
    """Artifact reduction for medical images

    Supported artifacts:
        - metal: Metal artifacts in CT
        - motion: Motion artifacts
        - ring: Ring artifacts in CT
        - beam_hardening: Beam hardening artifacts
    """

    def __init__(self):
        self.methods = ["metal", "motion", "ring", "beam_hardening"]

    def reduce_artifacts(
        self, image: np.ndarray, artifact_type: str = "metal", **kwargs
    ) -> np.ndarray:
        """Apply artifact reduction

        Args:
            image: Input image
            artifact_type: Type of artifact to reduce
            **kwargs: Method-specific parameters

        Returns:
            Image with reduced artifacts
        """
        if artifact_type == "metal":
            return self._reduce_metal_artifacts(image, **kwargs)
        elif artifact_type == "motion":
            return self._reduce_motion_artifacts(image, **kwargs)
        elif artifact_type == "ring":
            return self._reduce_ring_artifacts(image, **kwargs)
        elif artifact_type == "beam_hardening":
            return self._reduce_beam_hardening(image, **kwargs)
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

    def _reduce_metal_artifacts(
        self, image: np.ndarray, threshold: float = 3000, interpolation_method: str = "linear"
    ) -> np.ndarray:
        """Metal artifact reduction for CT"""
        # Identify metal regions
        metal_mask = image > threshold

        # Create interpolation mask
        non_metal_mask = ~metal_mask

        # Interpolate metal regions
        if len(image.shape) == 2:
            # Use inpainting for 2D
            result = image.copy()
            metal_regions = metal_mask.astype(np.uint8) * 255

            # Convert to appropriate dtype for OpenCV
            img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # Inpaint metal regions
            inpainted = cv2.inpaint(img_norm, metal_regions, 3, cv2.INPAINT_TELEA)

            # Convert back and blend
            inpainted_float = (
                inpainted.astype(np.float32) * (image.max() - image.min()) / 255 + image.min()
            )
            result[metal_mask] = inpainted_float[metal_mask]

            return result
        else:
            # Process slice by slice for 3D
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = self._reduce_metal_artifacts(image[i], threshold, interpolation_method)
            return result

    def _reduce_motion_artifacts(self, image: np.ndarray, method: str = "averaging") -> np.ndarray:
        """Motion artifact reduction"""
        if method == "averaging":
            # Simple temporal averaging (placeholder)
            return gaussian_filter(image, sigma=1.0)
        else:
            # More sophisticated methods would require multiple frames
            logger.warning("Advanced motion correction not implemented")
            return image

    def _reduce_ring_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Ring artifact reduction using polar-domain radial profile subtraction.

        Converts to polar coordinates, estimates the ring pattern as the
        median radial profile across all angles, subtracts only that bias
        from the original image, and converts back to Cartesian.
        """
        if len(image.shape) != 2:
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = self._reduce_ring_artifacts(image[i])
            return result

        h, w = image.shape
        center = (w / 2.0, h / 2.0)  # cv2 uses (x, y)
        max_radius = int(np.sqrt(center[0] ** 2 + center[1] ** 2))

        img32 = image.astype(np.float32)

        # Forward polar transform (output rows=angle, cols=radius)
        polar = cv2.warpPolar(
            img32, (360, max_radius), center, max_radius,
            cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR,
        )

        # Ring pattern = radial profile (median across all angles for each radius)
        ring_profile = np.median(polar, axis=0, keepdims=True)  # shape (1, max_radius)

        # Build a full polar image of the ring pattern and warp it back
        ring_polar = np.broadcast_to(ring_profile, polar.shape).astype(np.float32).copy()
        cart_correction = cv2.warpPolar(
            ring_polar, (h, w), center, max_radius,
            cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR,
        )

        return image - cart_correction

    def _reduce_beam_hardening(
        self, image: np.ndarray, correction_factor: float = 0.1
    ) -> np.ndarray:
        """Beam hardening correction"""
        # Simple polynomial correction
        # More sophisticated methods would use water correction or dual-energy CT
        corrected = image - correction_factor * (image**2) / (image.max() ** 2)
        return corrected


def preprocess_ct(
    image: np.ndarray,
    denoise: bool = True,
    normalize: bool = True,
    window: str = "lung",
    reduce_artifacts: bool = False,
) -> np.ndarray:
    """Complete CT preprocessing pipeline

    Args:
        image: Input CT image
        denoise: Apply denoising
        normalize: Apply normalization
        window: Window preset or None for auto
        reduce_artifacts: Apply artifact reduction

    Returns:
        Preprocessed image
    """
    result = image.copy()

    # Denoise
    if denoise:
        denoiser = Denoiser(method="bilateral")
        result = denoiser.denoise(result)

    # Reduce artifacts
    if reduce_artifacts:
        artifact_reducer = ArtifactReducer()
        result = artifact_reducer.reduce_artifacts(result, artifact_type="metal")

    # Window/level adjustment
    windower = WindowLevelAdjuster()
    result = windower.adjust(result, window=window)

    # Normalize
    if normalize:
        normalizer = IntensityNormalizer(method="minmax")
        result = normalizer.normalize(result, out_min=0, out_max=1)

    return result


def preprocess_mri(
    image: np.ndarray, denoise: bool = True, bias_correction: bool = True, normalize: bool = True
) -> np.ndarray:
    """Complete MRI preprocessing pipeline

    Args:
        image: Input MRI image
        denoise: Apply denoising
        bias_correction: Apply N4 bias field correction
        normalize: Apply normalization

    Returns:
        Preprocessed image
    """
    result = image.copy()

    # Denoise
    if denoise:
        denoiser = Denoiser(method="nlm")
        result = denoiser.denoise(result)

    # Bias field correction
    if bias_correction:
        # Use SimpleITK for N4 bias correction
        sitk_image = sitk.GetImageFromArray(result)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(sitk_image)
        result = sitk.GetArrayFromImage(corrected)

    # Normalize
    if normalize:
        normalizer = IntensityNormalizer(method="zscore")
        result = normalizer.normalize(result)

    return result


def preprocess_pet(
    image: np.ndarray,
    denoise: bool = True,
    normalize: bool = True,
    suv_conversion: bool = False,
    body_weight: float = None,
    injected_dose: float = None,
) -> np.ndarray:
    """Complete PET preprocessing pipeline

    Args:
        image: Input PET image
        denoise: Apply denoising
        normalize: Apply normalization
        suv_conversion: Convert to SUV
        body_weight: Patient body weight in kg (for SUV)
        injected_dose: Injected dose in Bq (for SUV)

    Returns:
        Preprocessed image
    """
    result = image.copy()

    # SUV conversion if requested
    if suv_conversion and body_weight and injected_dose:
        # SUV = pixel_value * body_weight(g) / injected_dose(Bq)
        result = result * (body_weight * 1000) / injected_dose

    # Denoise
    if denoise:
        denoiser = Denoiser(method="pet_specific")
        result = denoiser.denoise(result)

    # Normalize
    if normalize:
        normalizer = IntensityNormalizer(method="percentile")
        result = normalizer.normalize(result, lower=5, upper=95)

    return result


class HUClipper:
    """Clip Hounsfield Unit values to clinically relevant ranges.

    Useful for removing scanner-specific outliers before windowing.
    """

    # Common anatomical presets
    PRESETS = {
        "default": (-1024, 3071),
        "soft_tissue": (-200, 400),
        "lung": (-1100, -200),
        "bone": (200, 3071),
        "brain": (-100, 200),
    }

    def clip(
        self,
        image: np.ndarray,
        hu_min: float = -1024,
        hu_max: float = 3071,
        preset: Optional[str] = None,
    ) -> np.ndarray:
        """Clip HU values to specified range.

        Args:
            image: CT image in HU
            hu_min: Minimum HU value
            hu_max: Maximum HU value
            preset: Optional anatomy preset name

        Returns:
            Clipped image
        """
        if preset:
            if preset not in self.PRESETS:
                raise ValueError(
                    f"Unknown preset: {preset}. Choose from {list(self.PRESETS.keys())}"
                )
            hu_min, hu_max = self.PRESETS[preset]

        return np.clip(image, hu_min, hu_max)


class VoxelResampler:
    """Standalone voxel resampling utility using scipy (no SimpleITK required).

    Resamples 3D volumes to target spacing using scipy.ndimage.zoom.
    """

    def resample(
        self,
        image: np.ndarray,
        current_spacing: Tuple[float, ...],
        target_spacing: Tuple[float, ...],
        order: int = 1,
    ) -> np.ndarray:
        """Resample volume to target spacing.

        Args:
            image: Input volume (2D or 3D)
            current_spacing: Current voxel spacing
            target_spacing: Target voxel spacing
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)

        Returns:
            Resampled volume
        """
        from scipy.ndimage import zoom

        # Calculate zoom factors
        zoom_factors = tuple(cs / ts for cs, ts in zip(current_spacing, target_spacing))

        # Ensure zoom factors match image dimensions
        if len(zoom_factors) != len(image.shape):
            # Pad or truncate zoom factors to match
            if len(image.shape) > len(zoom_factors):
                zoom_factors = zoom_factors + (1.0,) * (len(image.shape) - len(zoom_factors))
            else:
                zoom_factors = zoom_factors[: len(image.shape)]

        return zoom(image, zoom_factors, order=order)
