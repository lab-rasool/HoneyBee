"""
Radiology Preprocessing Components

Comprehensive preprocessing capabilities for medical images including:
- Denoising (NLM, TV, bilateral, PET-specific, Rician, deep learning, DIPY)
- Artifact reduction (metal, motion, ring, beam hardening)
- Intensity normalization (z-score, min-max, percentile, histogram, WhiteStripe, TorchIO, MONAI)
- Window/level adjustment with clinical presets
- HU clipping and voxel resampling
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class Denoiser:
    """Advanced denoising methods for medical images.

    Supported methods:
        - nlm: Non-local means denoising
        - tv: Total variation denoising
        - bilateral: Bilateral filtering
        - median: Median filtering
        - gaussian: Gaussian filtering
        - pet_specific: PET-specific iterative denoising
        - rician: Rician noise correction for MRI
        - deep: DnCNN-style deep denoising
        - dipy_nlm: DIPY non-local means
        - dipy_mppca: DIPY Marchenko-Pastur PCA
    """

    def __init__(self, method: str = "nlm"):
        self.method = method.lower()
        self.supported_methods = [
            "nlm",
            "tv",
            "bilateral",
            "median",
            "gaussian",
            "pet_specific",
            "rician",
            "deep",
            "dipy_nlm",
            "dipy_mppca",
        ]

        if self.method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported. Choose from {self.supported_methods}")

    def denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply denoising to image.

        Args:
            image: Input image (2D or 3D)
            **kwargs: Method-specific parameters

        Returns:
            Denoised image
        """
        dispatch = {
            "nlm": self._nlm_denoise,
            "tv": self._tv_denoise,
            "bilateral": self._bilateral_denoise,
            "median": self._median_denoise,
            "gaussian": self._gaussian_denoise,
            "pet_specific": self._pet_specific_denoise,
            "rician": self._rician_denoise,
            "deep": self._deep_denoise,
            "dipy_nlm": self._dipy_nlm_denoise,
            "dipy_mppca": self._dipy_mppca_denoise,
        }
        return dispatch[self.method](image, **kwargs)

    def _nlm_denoise(
        self, image: np.ndarray, patch_size: int = 5, patch_distance: int = 6, h: float = 0.1
    ) -> np.ndarray:
        """Non-local means denoising."""
        from skimage.restoration import denoise_nl_means

        img_min, img_max = image.min(), image.max()
        img_norm = (image - img_min) / (img_max - img_min + 1e-8)

        if len(image.shape) == 2:
            denoised = denoise_nl_means(
                img_norm, patch_size=patch_size, patch_distance=patch_distance, h=h
            )
        else:
            denoised = np.zeros_like(img_norm)
            for i in range(image.shape[0]):
                denoised[i] = denoise_nl_means(
                    img_norm[i], patch_size=patch_size, patch_distance=patch_distance, h=h
                )

        return denoised * (img_max - img_min) + img_min

    def _tv_denoise(self, image: np.ndarray, weight: float = 0.1) -> np.ndarray:
        """Total variation denoising."""
        from skimage.restoration import denoise_tv_chambolle

        img_min, img_max = image.min(), image.max()
        img_norm = (image - img_min) / (img_max - img_min + 1e-8)

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
        """Bilateral filtering."""
        import cv2

        if len(image.shape) == 2:
            img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            denoised = cv2.bilateralFilter(img_norm, d=9, sigmaColor=75, sigmaSpace=75)
            return denoised.astype(np.float32) * (image.max() - image.min()) / 255 + image.min()
        else:
            denoised = np.zeros_like(image)
            for i in range(image.shape[0]):
                img_norm = cv2.normalize(image[i], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                denoised[i] = cv2.bilateralFilter(img_norm, d=9, sigmaColor=75, sigmaSpace=75)
            return denoised.astype(np.float32) * (image.max() - image.min()) / 255 + image.min()

    def _median_denoise(self, image: np.ndarray, size: int = 3) -> np.ndarray:
        """Median filtering."""
        from scipy.ndimage import median_filter

        return median_filter(image, size=size)

    def _gaussian_denoise(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gaussian filtering."""
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(image, sigma=sigma)

    def _pet_specific_denoise(
        self, image: np.ndarray, iterations: int = 5, kernel_size: int = 3
    ) -> np.ndarray:
        """PET-specific denoising using iterative filtering."""
        from scipy.ndimage import median_filter

        denoised = image.copy()
        for _ in range(iterations):
            denoised = median_filter(denoised, size=kernel_size)
            denoised = self._anisotropic_diffusion(denoised, iterations=10)
        return denoised

    def _anisotropic_diffusion(
        self, image: np.ndarray, iterations: int = 10, kappa: float = 50, gamma: float = 0.1
    ) -> np.ndarray:
        """Perona-Malik anisotropic diffusion."""
        img = image.copy()
        for _ in range(iterations):
            nabla_n = np.roll(img, -1, axis=0) - img
            nabla_s = np.roll(img, 1, axis=0) - img
            nabla_e = np.roll(img, -1, axis=1) - img
            nabla_w = np.roll(img, 1, axis=1) - img
            c_n = np.exp(-((nabla_n / kappa) ** 2))
            c_s = np.exp(-((nabla_s / kappa) ** 2))
            c_e = np.exp(-((nabla_e / kappa) ** 2))
            c_w = np.exp(-((nabla_w / kappa) ** 2))
            img += gamma * (c_n * nabla_n + c_s * nabla_s + c_e * nabla_e + c_w * nabla_w)
        return img

    def _rician_denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Rician noise correction and denoising for MRI."""
        from skimage.restoration import denoise_nl_means

        background = image[image <= np.percentile(image, 25)]
        if len(background) == 0 or background.std() == 0:
            sigma = max(image.std() * 0.05, 1e-8)
        else:
            sigma = float(np.median(np.abs(background - np.median(background)))) / 0.655
            sigma = max(sigma, 1e-8)

        image_float = image.astype(np.float64)
        corrected = np.sqrt(np.maximum(0, image_float**2 - 2 * sigma**2))

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

        class _DnCNN(nn.Module):
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
                return x - noise

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _DnCNN().to(device)

        if weights_path:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.set_to_inference_mode()
        else:
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            if len(image.shape) == 2:
                slices = [image]
            else:
                step = max(1, image.shape[0] // 16)
                slices = [image[i] for i in range(0, image.shape[0], step)]

            img_min, img_max = image.min(), image.max()
            img_range = img_max - img_min if img_max != img_min else 1.0

            for _epoch in range(30):
                for sl in slices:
                    sl_norm = (sl.astype(np.float32) - img_min) / img_range
                    tensor = torch.from_numpy(sl_norm).unsqueeze(0).unsqueeze(0).to(device)
                    mask = (torch.rand_like(tensor) > 0.3).float()
                    masked_input = tensor * mask
                    output = model(masked_input)
                    loss = ((output - tensor) ** 2 * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        model.train(False)

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
        """Non-local means denoising using DIPY.

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

        Falls back to NLM if DIPY not installed.
        """
        try:
            from dipy.denoise.localpca import mppca

            denoised, _ = mppca(image, **kwargs)
            return denoised
        except ImportError:
            logger.warning("DIPY not available, falling back to scikit-image NLM")
            return self._nlm_denoise(image)


class IntensityNormalizer:
    """Intensity normalization methods for medical images.

    Supported methods:
        - zscore: Z-score normalization
        - minmax: Min-max scaling
        - percentile: Percentile-based normalization
        - histogram: Histogram equalization
        - ct_normalize: CT HU range normalization
        - whitestripe: White stripe MRI normalization
        - torchio_znorm: TorchIO Z-normalization
        - torchio_histogram: TorchIO histogram standardization
        - monai_znorm: MONAI Z-normalization
    """

    def __init__(self, method: str = "zscore"):
        self.method = method.lower()
        self.supported_methods = [
            "zscore",
            "minmax",
            "percentile",
            "histogram",
            "ct_normalize",
            "whitestripe",
            "torchio_znorm",
            "torchio_histogram",
            "monai_znorm",
        ]

        if self.method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported. Choose from {self.supported_methods}")

    def normalize(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply normalization.

        Args:
            image: Input image
            **kwargs: Method-specific parameters

        Returns:
            Normalized image
        """
        dispatch = {
            "zscore": self._zscore_normalize,
            "minmax": self._minmax_normalize,
            "percentile": self._percentile_normalize,
            "histogram": self._histogram_normalize,
            "ct_normalize": self._ct_normalize,
            "whitestripe": self._whitestripe_normalize,
            "torchio_znorm": self._torchio_znorm,
            "torchio_histogram": self._torchio_histogram,
            "monai_znorm": self._monai_znorm,
        }
        return dispatch[self.method](image, **kwargs)

    def _zscore_normalize(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Z-score normalization."""
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
        """Min-max normalization."""
        in_min, in_max = image.min(), image.max()
        normalized = (image - in_min) / (in_max - in_min + 1e-8)
        return normalized * (out_max - out_min) + out_min

    def _percentile_normalize(
        self, image: np.ndarray, lower: float = 1.0, upper: float = 99.0
    ) -> np.ndarray:
        """Percentile-based normalization."""
        p_lower = np.percentile(image, lower)
        p_upper = np.percentile(image, upper)
        clipped = np.clip(image, p_lower, p_upper)
        return (clipped - p_lower) / (p_upper - p_lower + 1e-8)

    def _histogram_normalize(self, image: np.ndarray, nbins: int = 256) -> np.ndarray:
        """Histogram equalization."""
        from skimage import exposure

        if len(image.shape) == 2:
            return exposure.equalize_hist(image, nbins=nbins)
        else:
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
        """White stripe normalization for MRI."""
        from scipy.ndimage import gaussian_filter1d

        nonzero = image[image > 0].flatten()
        if len(nonzero) == 0:
            return image.copy()

        hist, bin_edges = np.histogram(nonzero, bins=200)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(hist.astype(float), sigma=3)
        peak_idx = np.argmax(smoothed)
        wm_peak = bin_centers[peak_idx]

        stripe_lower = wm_peak * (1 - width)
        stripe_upper = wm_peak * (1 + width)
        stripe_mask = (image >= stripe_lower) & (image <= stripe_upper) & (image > 0)
        if stripe_mask.sum() == 0:
            return image.copy()

        stripe_mean = image[stripe_mask].mean()
        stripe_std = image[stripe_mask].std()
        if stripe_std == 0:
            return image.copy()

        return (image - stripe_mean) / stripe_std

    def _torchio_znorm(
        self, image: np.ndarray, masking_method: str = "mean", **kwargs
    ) -> np.ndarray:
        """Z-normalization using TorchIO.

        Falls back to built-in z-score if TorchIO not installed.
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
            transform = tio.ZNormalization(masking_method=masking_method)
            transformed = transform(subject)
            result = transformed.image.numpy().squeeze()

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
        """Z-normalization using MONAI.

        Falls back to built-in z-score if MONAI not installed.
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
    """Window/level adjustment for medical images with clinical presets."""

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
        """Apply window/level adjustment.

        Args:
            image: Input image
            window: Window width or preset name
            level: Window center/level
            output_range: Output intensity range

        Returns:
            Windowed image
        """
        if isinstance(window, str):
            if window.lower() not in self.PRESETS:
                raise ValueError(f"Unknown preset: {window}")
            preset = self.PRESETS[window.lower()]
            window = preset["width"]
            level = preset["center"]

        if window is None or level is None:
            level = np.median(image)
            window = np.percentile(image, 95) - np.percentile(image, 5)

        min_val = level - window / 2
        max_val = level + window / 2
        windowed = np.clip(image, min_val, max_val)
        scaled = (windowed - min_val) / (max_val - min_val)
        return scaled * (output_range[1] - output_range[0]) + output_range[0]

    def get_auto_window(
        self, image: np.ndarray, percentile_range: Tuple[float, float] = (5, 95)
    ) -> Dict[str, float]:
        """Calculate automatic window/level from image statistics."""
        p_low, p_high = np.percentile(image, percentile_range)
        center = (p_low + p_high) / 2
        width = p_high - p_low
        return {"center": center, "width": width}


class ArtifactReducer:
    """Artifact reduction for medical images.

    Supported artifacts: metal, motion, ring, beam_hardening
    """

    def __init__(self):
        self.methods = ["metal", "motion", "ring", "beam_hardening"]

    def reduce_artifacts(
        self, image: np.ndarray, artifact_type: str = "metal", **kwargs
    ) -> np.ndarray:
        """Apply artifact reduction.

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
        """Metal artifact reduction for CT."""
        import cv2

        metal_mask = image > threshold

        if len(image.shape) == 2:
            result = image.copy()
            metal_regions = metal_mask.astype(np.uint8) * 255
            img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            inpainted = cv2.inpaint(img_norm, metal_regions, 3, cv2.INPAINT_TELEA)
            inpainted_float = (
                inpainted.astype(np.float32) * (image.max() - image.min()) / 255 + image.min()
            )
            result[metal_mask] = inpainted_float[metal_mask]
            return result
        else:
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = self._reduce_metal_artifacts(image[i], threshold, interpolation_method)
            return result

    def _reduce_motion_artifacts(self, image: np.ndarray, method: str = "averaging") -> np.ndarray:
        """Motion artifact reduction."""
        from scipy.ndimage import gaussian_filter

        if method == "averaging":
            return gaussian_filter(image, sigma=1.0)
        else:
            logger.warning("Advanced motion correction not implemented")
            return image

    def _reduce_ring_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Ring artifact reduction using polar-domain radial profile subtraction."""
        import cv2

        if len(image.shape) != 2:
            result = np.zeros_like(image)
            for i in range(image.shape[0]):
                result[i] = self._reduce_ring_artifacts(image[i])
            return result

        h, w = image.shape
        center = (w / 2.0, h / 2.0)
        max_radius = int(np.sqrt(center[0] ** 2 + center[1] ** 2))
        img32 = image.astype(np.float32)

        polar = cv2.warpPolar(
            img32,
            (360, max_radius),
            center,
            max_radius,
            cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR,
        )

        ring_profile = np.median(polar, axis=0, keepdims=True)
        ring_polar = np.broadcast_to(ring_profile, polar.shape).astype(np.float32).copy()
        cart_correction = cv2.warpPolar(
            ring_polar,
            (h, w),
            center,
            max_radius,
            cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR,
        )

        return image - cart_correction

    def _reduce_beam_hardening(
        self, image: np.ndarray, correction_factor: float = 0.1
    ) -> np.ndarray:
        """Beam hardening correction."""
        corrected = image - correction_factor * (image**2) / (image.max() ** 2)
        return corrected


class HUClipper:
    """Clip Hounsfield Unit values to clinically relevant ranges."""

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
    """Standalone voxel resampling utility using scipy."""

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

        zoom_factors = tuple(cs / ts for cs, ts in zip(current_spacing, target_spacing))
        if len(zoom_factors) != len(image.shape):
            if len(image.shape) > len(zoom_factors):
                zoom_factors = zoom_factors + (1.0,) * (len(image.shape) - len(zoom_factors))
            else:
                zoom_factors = zoom_factors[: len(image.shape)]
        return zoom(image, zoom_factors, order=order)


# ============================================================
# Module-level pipeline functions
# ============================================================


def preprocess_ct(
    image: np.ndarray,
    denoise: bool = True,
    normalize: bool = True,
    window: str = "lung",
    reduce_artifacts: bool = False,
) -> np.ndarray:
    """Complete CT preprocessing pipeline."""
    result = image.copy()

    if denoise:
        denoiser = Denoiser(method="bilateral")
        result = denoiser.denoise(result)

    if reduce_artifacts:
        artifact_reducer = ArtifactReducer()
        result = artifact_reducer.reduce_artifacts(result, artifact_type="metal")

    windower = WindowLevelAdjuster()
    result = windower.adjust(result, window=window)

    if normalize:
        normalizer = IntensityNormalizer(method="minmax")
        result = normalizer.normalize(result, out_min=0, out_max=1)

    return result


def preprocess_mri(
    image: np.ndarray, denoise: bool = True, bias_correction: bool = True, normalize: bool = True
) -> np.ndarray:
    """Complete MRI preprocessing pipeline."""
    import SimpleITK as sitk

    result = image.copy()

    if denoise:
        denoiser = Denoiser(method="nlm")
        result = denoiser.denoise(result)

    if bias_correction:
        sitk_image = sitk.GetImageFromArray(result)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected = corrector.Execute(sitk_image)
        result = sitk.GetArrayFromImage(corrected)

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
    """Complete PET preprocessing pipeline."""
    result = image.copy()

    if suv_conversion and body_weight and injected_dose:
        result = result * (body_weight * 1000) / injected_dose

    if denoise:
        denoiser = Denoiser(method="pet_specific")
        result = denoiser.denoise(result)

    if normalize:
        normalizer = IntensityNormalizer(method="percentile")
        result = normalizer.normalize(result, lower=5, upper=95)

    return result
