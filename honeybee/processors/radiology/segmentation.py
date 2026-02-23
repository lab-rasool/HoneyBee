"""
Segmentation Module for Medical Images

Implements segmentation algorithms for medical images:
- LungmaskSegmenter: Lung segmentation using lungmask (default, no model path needed)
- TotalSegmentatorWrapper: Multi-organ segmentation (117 structures)
- NNUNetSegmenter: nnU-Net v2 based segmentation for CT/MRI
- PETSegmenter: SUV-based metabolic volume segmentation for PET
- detect_nodules: Standalone LoG blob detection for lung nodule detection
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default label maps for common nnU-Net tasks
DEFAULT_LABEL_MAPS: Dict[str, Dict[int, str]] = {
    "lung": {
        1: "left_lung",
        2: "right_lung",
    },
    "total_organs": {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "pancreas",
        8: "adrenal_gland_right",
        9: "adrenal_gland_left",
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
    },
    "brain": {
        1: "brain",
    },
    "brain_tumor": {
        1: "tumor",
    },
}


class LungmaskSegmenter:
    """Lung segmentation using the lungmask library.

    Default segmentation backend - no model paths needed.

    Args:
        modelname: Model variant ('R231', 'LTRCLobes', 'R231CovidWeb')
        force_cpu: Force CPU inference
        batch_size: Batch size for inference
    """

    def __init__(
        self,
        modelname: str = "R231",
        force_cpu: bool = False,
        batch_size: int = 20,
    ):
        self._modelname = modelname
        self._force_cpu = force_cpu
        self._batch_size = batch_size
        self._inferer = None

    def _get_inferer(self):
        """Lazy-load the LMInferer."""
        if self._inferer is None:
            from lungmask import LMInferer

            self._inferer = LMInferer(
                modelname=self._modelname,
                force_cpu=self._force_cpu,
                batch_size=self._batch_size,
            )
        return self._inferer

    def segment(
        self, image: np.ndarray, spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """Segment lungs from CT volume.

        Args:
            image: CT volume (D, H, W) in HU.
            spacing: Voxel spacing (z, y, x) in mm.

        Returns:
            Binary lung mask (uint8). 0=background, 1=right lung, 2=left lung.
        """
        import SimpleITK as sitk

        inferer = self._get_inferer()

        # Convert numpy to SimpleITK image
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        sitk_image.SetSpacing(tuple(reversed(spacing[:3])))  # SimpleITK uses x,y,z

        segmentation = inferer.apply(sitk_image)
        return segmentation


class TotalSegmentatorWrapper:
    """Multi-organ segmentation using TotalSegmentator (117 structures).

    Args:
        fast: Use low-resolution fast mode
        device: Device for inference ('gpu' or 'cpu')
    """

    def __init__(self, fast: bool = False, device: str = "gpu"):
        self._fast = fast
        self._device = device

    def segment(
        self,
        image: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        task: str = "total",
    ) -> Dict[str, np.ndarray]:
        """Segment multiple organs from CT volume.

        Args:
            image: CT volume (D, H, W) in HU.
            spacing: Voxel spacing (z, y, x) in mm.
            task: Segmentation task ('total', 'lung_vessels', 'body', etc.)

        Returns:
            Dict mapping organ names to binary masks.
        """
        import tempfile

        import nibabel as nib
        import SimpleITK as sitk

        # Write to temp NIfTI for TotalSegmentator
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.nii.gz"
            output_path = Path(tmpdir) / "output"

            sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
            sitk_image.SetSpacing(tuple(reversed(spacing[:3])))
            sitk.WriteImage(sitk_image, str(input_path))

            from totalsegmentator.python_api import totalsegmentator

            totalsegmentator(
                input=str(input_path),
                output=str(output_path),
                task=task,
                fast=self._fast,
                device=self._device,
            )

            # Read results
            results: Dict[str, np.ndarray] = {}
            output_dir = Path(output_path)
            if output_dir.exists():
                for nifti_file in output_dir.glob("*.nii.gz"):
                    organ_name = nifti_file.stem.replace(".nii", "")
                    mask_img = nib.load(str(nifti_file))
                    results[organ_name] = mask_img.get_fdata().astype(np.uint8)

        return results


class NNUNetSegmenter:
    """nnU-Net v2 based segmentation for CT and MRI images.

    Args:
        model_paths: Mapping of task names to nnU-Net model folder paths.
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        use_mirroring: Enable test-time augmentation via mirroring.
        tile_step_size: Overlap fraction for sliding-window inference.
        verbose: Enable verbose nnU-Net logging.
        label_maps: Custom label maps per task.
    """

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        device: str = "cuda",
        use_mirroring: bool = True,
        tile_step_size: float = 0.5,
        verbose: bool = False,
        label_maps: Optional[Dict[str, Dict[int, str]]] = None,
    ):
        self._model_paths: Dict[str, str] = dict(model_paths) if model_paths else {}
        self._device = device
        self._use_mirroring = use_mirroring
        self._tile_step_size = tile_step_size
        self._verbose = verbose
        self._predictors: Dict[str, object] = {}

        self._label_maps: Dict[str, Dict[int, str]] = dict(DEFAULT_LABEL_MAPS)
        if label_maps:
            self._label_maps.update(label_maps)

    @property
    def available_tasks(self) -> List[str]:
        """Return list of configured task names."""
        return list(self._model_paths.keys())

    def set_model_path(self, task: str, path: str) -> None:
        """Register or update the model path for a task."""
        self._model_paths[task] = path
        self._predictors.pop(task, None)

    def set_label_map(self, task: str, label_map: Dict[int, str]) -> None:
        """Set or override the label map for a task."""
        self._label_maps[task] = label_map

    def get_label_map(self, task: str) -> Dict[int, str]:
        """Return the label map for a task."""
        if task not in self._label_maps and task in self._model_paths:
            self._label_maps[task] = self._read_dataset_json_labels(task)
        return self._label_maps.get(task, {})

    def segment_lungs(
        self,
        image: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        task: str = "lung",
    ) -> np.ndarray:
        """Segment lungs from CT volume.

        Args:
            image: CT volume (D, H, W) or 2D slice (H, W) in HU.
            spacing: Voxel spacing (z, y, x) in mm.
            task: nnU-Net task name.

        Returns:
            Binary lung mask (union of all lung labels).
        """
        seg_map = self.predict_raw(image, spacing, task)
        return (seg_map > 0).astype(np.uint8)

    def segment_organs(
        self,
        image: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        organs: Optional[List[str]] = None,
        task: str = "total_organs",
    ) -> Dict[str, np.ndarray]:
        """Segment multiple organs from CT volume.

        Args:
            image: CT volume (D, H, W) in HU.
            spacing: Voxel spacing (z, y, x) in mm.
            organs: Filter to these organ names. None returns all.
            task: nnU-Net task name.

        Returns:
            Dict mapping organ names to binary masks.
        """
        seg_map = self.predict_raw(image, spacing, task)
        label_map = self.get_label_map(task)

        results: Dict[str, np.ndarray] = {}
        for label_id, name in label_map.items():
            if organs is not None and name not in organs:
                continue
            results[name] = (seg_map == label_id).astype(np.uint8)
        return results

    def extract_brain(
        self,
        image: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        task: str = "brain",
    ) -> np.ndarray:
        """Extract brain mask from MRI volume."""
        seg_map = self.predict_raw(image, spacing, task)
        return (seg_map > 0).astype(np.uint8)

    def segment_tumor(
        self,
        image: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        task: str = "brain_tumor",
    ) -> np.ndarray:
        """Segment tumor from CT or MRI volume."""
        seg_map = self.predict_raw(image, spacing, task)
        return (seg_map > 0).astype(np.uint8)

    def predict_raw(
        self,
        image: np.ndarray,
        spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
        task: str = "lung",
    ) -> np.ndarray:
        """Run nnU-Net inference and return the raw integer segmentation map."""
        predictor = self._get_predictor(task)
        data, props = self._prepare_input(image, spacing)
        seg = predictor.predict_single_npy_array(data, props, None, None, False)
        if seg.ndim > image.ndim:
            seg = seg.squeeze()
        if image.ndim == 2 and seg.ndim == 3:
            seg = seg[0]
        return seg

    def _get_predictor(self, task: str):
        """Return a cached nnUNetPredictor for the given task."""
        if task in self._predictors:
            return self._predictors[task]

        if task not in self._model_paths:
            raise ValueError(
                f"No model path configured for task '{task}'. "
                f"Available tasks: {self.available_tasks}. "
                f"Use set_model_path('{task}', '/path/to/model') to configure."
            )

        model_dir = Path(self._model_paths[task])
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(
            tile_step_size=self._tile_step_size,
            use_mirroring=self._use_mirroring,
            verbose=self._verbose,
            verbose_preprocessing=self._verbose,
            allow_tqdm=self._verbose,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_dir),
            use_folds="all",
            checkpoint_name="checkpoint_final.pth",
        )
        predictor.network = predictor.network.to(self._device)

        self._predictors[task] = predictor
        logger.info(f"Loaded nnU-Net predictor for task '{task}' from {model_dir}")
        return predictor

    def _prepare_input(
        self, image: np.ndarray, spacing: Tuple[float, ...]
    ) -> Tuple[np.ndarray, dict]:
        """Convert numpy image to nnU-Net input format."""
        arr = image.astype(np.float32)

        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
            spacing = (1.0,) + tuple(spacing[-2:]) if len(spacing) >= 2 else (1.0, 1.0, 1.0)

        arr = arr[np.newaxis, ...]
        spacing_xyz = tuple(reversed(spacing[:3]))
        props = {"spacing": list(spacing_xyz)}
        return arr, props

    def _read_dataset_json_labels(self, task: str) -> Dict[int, str]:
        """Read label names from dataset.json in model folder."""
        if task not in self._model_paths:
            return {}
        dataset_json = Path(self._model_paths[task]) / "dataset.json"
        if not dataset_json.exists():
            return {}
        try:
            with open(dataset_json) as f:
                data = json.load(f)
            labels = data.get("labels", {})
            return {int(k): v for k, v in labels.items() if k != "0" and k != "background"}
        except Exception:
            logger.warning(f"Failed to read label map from {dataset_json}")
            return {}


# ============================================================
# Standalone nodule detection
# ============================================================


def detect_nodules(
    image: np.ndarray,
    lung_mask: np.ndarray,
    min_size: float = 3.0,
    max_size: float = 30.0,
) -> List[Dict]:
    """Detect lung nodules using multi-scale LoG blob detection.

    Args:
        image: CT image in HU.
        lung_mask: Pre-computed binary lung mask (required).
        min_size: Minimum nodule diameter in mm.
        max_size: Maximum nodule diameter in mm.

    Returns:
        List of dicts with keys: position, radius, diameter, intensity.
    """
    if lung_mask is None:
        raise ValueError("lung_mask is required.")

    lung_region = image * lung_mask

    sigma_min = min_size / 2.355
    sigma_max = max_size / 2.355
    sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), num=10)

    if image.ndim == 2:
        blobs = _detect_blobs_2d(lung_region, sigmas, lung_mask)
    else:
        blobs = _detect_blobs_3d(lung_region, sigmas, lung_mask)

    nodules = []
    for blob in blobs:
        nodule = {
            "position": blob[:3] if image.ndim == 3 else blob[:2],
            "radius": blob[-1],
            "diameter": blob[-1] * 2,
            "intensity": image[tuple(map(int, blob[: image.ndim]))],
        }
        nodules.append(nodule)

    return nodules


def _detect_blobs_2d(image: np.ndarray, sigmas: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Detect blobs in 2D using LoG."""
    from scipy import ndimage
    from skimage.feature import peak_local_max

    log_images = []
    for sigma in sigmas:
        log_image = ndimage.gaussian_laplace(image, sigma=sigma) * sigma**2
        log_images.append(log_image)

    log_stack = np.stack(log_images, axis=-1)

    local_maxima = peak_local_max(
        -log_stack.max(axis=-1),
        min_distance=int(sigmas.min()),
        threshold_abs=0.1,
        exclude_border=False,
    )

    blobs = []
    for peak in local_maxima:
        if mask[peak[0], peak[1]]:
            scale_idx = log_stack[peak[0], peak[1]].argmax()
            radius = sigmas[scale_idx] * np.sqrt(2)
            blobs.append([peak[0], peak[1], radius])

    return np.array(blobs) if blobs else np.array([])


def _detect_blobs_3d(image: np.ndarray, sigmas: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Detect blobs in 3D volume (slice-by-slice + merge)."""
    all_blobs = []
    for z in range(image.shape[0]):
        if mask[z].any():
            blobs_2d = _detect_blobs_2d(image[z], sigmas, mask[z])
            for blob in blobs_2d:
                all_blobs.append([z, blob[0], blob[1], blob[2]])

    if all_blobs:
        return _merge_3d_blobs(np.array(all_blobs))
    return np.array([])


def _merge_3d_blobs(blobs: np.ndarray, distance_threshold: float = 5.0) -> np.ndarray:
    """Merge nearby blobs in 3D."""
    if len(blobs) == 0:
        return blobs

    merged = []
    used = np.zeros(len(blobs), dtype=bool)

    for i in range(len(blobs)):
        if used[i]:
            continue
        distances = np.sqrt(np.sum((blobs[:, :3] - blobs[i][:3]) ** 2, axis=1))
        nearby = distances < distance_threshold

        cluster_blobs = blobs[nearby]
        mean_pos = cluster_blobs[:, :3].mean(axis=0)
        max_radius = cluster_blobs[:, 3].max()

        merged.append(list(mean_pos) + [max_radius])
        used[nearby] = True

    return np.array(merged)


# ============================================================
# PET Segmenter
# ============================================================


class PETSegmenter:
    """PET-specific segmentation algorithms."""

    def __init__(self):
        self.suv_thresholds = {
            "fixed": 2.5,
            "liver": 1.5,
            "blood": 2.0,
        }

    def segment_metabolic_volume(
        self,
        image: np.ndarray,
        method: str = "fixed",
        threshold: Optional[float] = None,
        reference_region: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Segment metabolically active tumor volume.

        Args:
            image: PET image (preferably in SUV)
            method: Thresholding method ('fixed', 'adaptive', 'gradient')
            threshold: Manual threshold override
            reference_region: Mask for reference region
        """
        if method == "fixed":
            return self._fixed_threshold_segmentation(image, threshold)
        elif method == "adaptive":
            return self._adaptive_threshold_segmentation(image, reference_region)
        elif method == "gradient":
            return self._gradient_based_segmentation(image)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _fixed_threshold_segmentation(
        self, image: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """Fixed SUV threshold segmentation."""
        from scipy.ndimage import binary_fill_holes
        from skimage import morphology

        if threshold is None:
            threshold = self.suv_thresholds["fixed"]

        mask = image > threshold
        mask = morphology.remove_small_objects(mask, min_size=10)
        mask = binary_fill_holes(mask)
        return mask

    def _adaptive_threshold_segmentation(
        self, image: np.ndarray, reference_region: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Adaptive threshold based on reference region."""
        from scipy.ndimage import binary_fill_holes
        from skimage import morphology

        if reference_region is None:
            reference_region = self._estimate_liver_region(image)

        if reference_region.any():
            reference_mean = image[reference_region].mean()
            threshold = reference_mean * self.suv_thresholds["liver"]
        else:
            threshold = self.suv_thresholds["fixed"]

        mask = image > threshold
        mask = morphology.remove_small_objects(mask, min_size=10)
        mask = binary_fill_holes(mask)
        return mask

    def _gradient_based_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Gradient-based segmentation for PET."""
        from scipy.ndimage import label
        from skimage import morphology
        from skimage.filters import gaussian
        from skimage.segmentation import watershed

        smoothed = gaussian(image, sigma=2.0)
        high_uptake = image > np.percentile(image, 90)
        markers = label(high_uptake)[0]
        labels = watershed(-smoothed, markers, mask=image > np.percentile(image, 70))
        mask = labels > 0
        mask = morphology.remove_small_objects(mask, min_size=20)
        return mask

    def _estimate_liver_region(self, image: np.ndarray) -> np.ndarray:
        """Estimate liver region for reference (simplified)."""
        from scipy.ndimage import label

        liver_range = (np.percentile(image, 40), np.percentile(image, 60))
        liver_mask = (image > liver_range[0]) & (image < liver_range[1])

        if len(image.shape) == 3:
            z_center = image.shape[0] // 2
            liver_mask[: z_center // 2] = False
            liver_mask[int(z_center * 1.5) :] = False

        labeled, num_features = label(liver_mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            largest = component_sizes[1:].argmax() + 1
            liver_mask = labeled == largest

        return liver_mask

    def calculate_suv_metrics(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate SUV metrics for segmented region."""
        if not mask.any():
            return {
                "suv_max": 0.0,
                "suv_mean": 0.0,
                "suv_peak": 0.0,
                "mtv": 0.0,
                "tlg": 0.0,
            }

        values = image[mask]
        suv_max = values.max()
        suv_mean = values.mean()

        max_loc = np.unravel_index(np.argmax(image * mask), image.shape)
        suv_peak = self._calculate_suv_peak(image, max_loc)

        mtv = mask.sum()
        tlg = suv_mean * mtv

        return {
            "suv_max": float(suv_max),
            "suv_mean": float(suv_mean),
            "suv_peak": float(suv_peak),
            "mtv": float(mtv),
            "tlg": float(tlg),
        }

    def _calculate_suv_peak(
        self, image: np.ndarray, center: Tuple[int, ...], radius: int = 6
    ) -> float:
        """Calculate SUV peak in sphere around point."""
        if len(image.shape) == 2:
            y, x = np.ogrid[: image.shape[0], : image.shape[1]]
            mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius**2
        else:
            z, y, x = np.ogrid[: image.shape[0], : image.shape[1], : image.shape[2]]
            mask = (x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2 <= radius**2

        if mask.any():
            return image[mask].mean()
        else:
            return image[center]
