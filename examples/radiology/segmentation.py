"""
Segmentation Module for Medical Images

Implements various segmentation algorithms for CT, MRI, and PET images:
- Lung segmentation for CT
- Multi-organ segmentation
- Nodule detection
- Brain extraction for MRI
- Metabolic volume segmentation for PET
"""

import numpy as np
from typing import Union, Tuple, Optional, Dict, List
import cv2
from scipy import ndimage
from scipy.ndimage import label, binary_fill_holes, binary_erosion, binary_dilation
from skimage import morphology, measure, segmentation, feature
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)


class CTSegmenter:
    """CT-specific segmentation algorithms"""
    
    def __init__(self):
        self.lung_threshold = -320  # HU threshold for lung
        self.organ_thresholds = {
            'liver': (-50, 150),
            'spleen': (40, 130),
            'kidney': (30, 120),
            'bone': (200, 3000),
            'muscle': (0, 100),
            'fat': (-150, -30)
        }
    
    def segment_lungs(self, image: np.ndarray, 
                     enhanced: bool = True) -> np.ndarray:
        """
        Enhanced lung segmentation for CT
        
        Args:
            image: CT image in HU
            enhanced: Use enhanced segmentation with airway removal
        """
        if len(image.shape) == 2:
            return self._segment_lungs_2d(image, enhanced)
        else:
            # Process 3D volume
            lung_mask = np.zeros_like(image, dtype=bool)
            for i in range(image.shape[0]):
                lung_mask[i] = self._segment_lungs_2d(image[i], enhanced)
            
            # Apply 3D morphological operations for consistency
            if enhanced:
                lung_mask = self._refine_3d_lung_mask(lung_mask)
            
            return lung_mask
    
    def _segment_lungs_2d(self, image: np.ndarray, enhanced: bool) -> np.ndarray:
        """Segment lungs in 2D slice"""
        # Threshold for air/lung
        binary = image < self.lung_threshold
        
        # Remove small components
        binary = morphology.remove_small_objects(binary, min_size=100)
        
        # Fill holes
        binary = binary_fill_holes(binary)
        
        # Separate left and right lungs
        labeled, num_features = label(binary)
        
        # Find lung regions (usually the two largest components)
        regions = measure.regionprops(labeled)
        regions.sort(key=lambda x: x.area, reverse=True)
        
        # Create lung mask
        lung_mask = np.zeros_like(image, dtype=bool)
        
        if len(regions) >= 2:
            # Use two largest regions as lungs
            for region in regions[:2]:
                lung_mask[labeled == region.label] = True
        elif len(regions) == 1:
            # Single lung visible
            lung_mask[labeled == regions[0].label] = True
        
        if enhanced:
            # Remove airways and vessels
            lung_mask = self._remove_airways(lung_mask, image)
        
        return lung_mask
    
    def _remove_airways(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Remove airways from lung mask"""
        # Find very bright regions within lungs (airways)
        airways = (image > -950) & mask
        
        # Morphological operations to clean up
        airways = morphology.opening(airways, morphology.disk(2))
        
        # Remove airways from mask
        refined_mask = mask & ~airways
        
        # Fill small holes that might have been created
        refined_mask = morphology.remove_small_holes(refined_mask, area_threshold=50)
        
        return refined_mask
    
    def _refine_3d_lung_mask(self, mask: np.ndarray) -> np.ndarray:
        """Refine 3D lung mask for consistency across slices"""
        # Remove small 3D components
        labeled, num_features = label(mask)
        
        # Keep only large connected components
        component_sizes = np.bincount(labeled.ravel())
        large_components = component_sizes > 1000  # Adjust threshold as needed
        large_components[0] = False  # Background
        
        refined_mask = large_components[labeled]
        
        # Smooth boundaries in 3D
        refined_mask = binary_erosion(refined_mask, iterations=1)
        refined_mask = binary_dilation(refined_mask, iterations=2)
        refined_mask = binary_erosion(refined_mask, iterations=1)
        
        return refined_mask
    
    def segment_organs(self, image: np.ndarray, 
                      organs: List[str] = ['liver', 'spleen', 'kidney']) -> Dict[str, np.ndarray]:
        """
        Multi-organ segmentation
        
        Args:
            image: CT image in HU
            organs: List of organs to segment
        """
        results = {}
        
        for organ in organs:
            if organ in self.organ_thresholds:
                mask = self._segment_organ(image, organ)
                results[organ] = mask
            else:
                logger.warning(f"Unknown organ: {organ}")
        
        return results
    
    def _segment_organ(self, image: np.ndarray, organ: str) -> np.ndarray:
        """Segment specific organ based on HU range"""
        low, high = self.organ_thresholds[organ]
        
        # Initial threshold
        mask = (image >= low) & (image <= high)
        
        # Morphological cleanup
        mask = morphology.remove_small_objects(mask, min_size=500)
        mask = morphology.remove_small_holes(mask, area_threshold=200)
        
        # Organ-specific refinements
        if organ == 'liver':
            mask = self._refine_liver_segmentation(mask, image)
        elif organ == 'kidney':
            mask = self._refine_kidney_segmentation(mask, image)
        
        return mask
    
    def _refine_liver_segmentation(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Refine liver segmentation"""
        # Liver is typically the largest organ in abdomen
        labeled, num_features = label(mask)
        
        if num_features > 0:
            # Keep largest component
            component_sizes = np.bincount(labeled.ravel())
            largest_component = component_sizes[1:].argmax() + 1
            mask = labeled == largest_component
        
        # Smooth boundaries
        mask = morphology.binary_closing(mask, morphology.disk(3))
        mask = morphology.binary_opening(mask, morphology.disk(2))
        
        return mask
    
    def _refine_kidney_segmentation(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Refine kidney segmentation (usually two separate regions)"""
        labeled, num_features = label(mask)
        
        if num_features >= 2:
            # Keep two largest components (left and right kidney)
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0  # Ignore background
            
            # Get indices of two largest components
            largest_two = np.argpartition(component_sizes, -2)[-2:]
            
            refined_mask = np.zeros_like(mask)
            for comp in largest_two:
                if comp > 0:  # Skip background
                    refined_mask |= (labeled == comp)
            
            return refined_mask
        
        return mask
    
    def detect_nodules(self, image: np.ndarray, 
                      lung_mask: Optional[np.ndarray] = None,
                      min_size: float = 3.0,
                      max_size: float = 30.0) -> List[Dict]:
        """
        Detect lung nodules
        
        Args:
            image: CT image
            lung_mask: Pre-computed lung mask
            min_size: Minimum nodule diameter in mm
            max_size: Maximum nodule diameter in mm
        """
        if lung_mask is None:
            lung_mask = self.segment_lungs(image)
        
        # Apply lung mask
        lung_region = image * lung_mask
        
        # Multi-scale LoG filter for blob detection
        # Assuming 1mm pixel spacing (should be adjusted based on actual spacing)
        sigma_min = min_size / 2.355  # FWHM to sigma
        sigma_max = max_size / 2.355
        
        # Generate sigmas for multi-scale detection
        sigmas = np.logspace(np.log10(sigma_min), np.log10(sigma_max), num=10)
        
        # Detect blobs
        if len(image.shape) == 2:
            blobs = self._detect_blobs_2d(lung_region, sigmas, lung_mask)
        else:
            blobs = self._detect_blobs_3d(lung_region, sigmas, lung_mask)
        
        # Convert to nodule information
        nodules = []
        for blob in blobs:
            nodule = {
                'position': blob[:3] if len(image.shape) == 3 else blob[:2],
                'radius': blob[-1],
                'diameter': blob[-1] * 2,
                'intensity': image[tuple(map(int, blob[:len(image.shape)]))],
            }
            nodules.append(nodule)
        
        return nodules
    
    def _detect_blobs_2d(self, image: np.ndarray, sigmas: np.ndarray, 
                        mask: np.ndarray) -> np.ndarray:
        """Detect blobs in 2D using LoG"""
        # Apply Laplacian of Gaussian
        log_images = []
        
        for sigma in sigmas:
            # LoG filter
            log_image = ndimage.gaussian_laplace(image, sigma=sigma) * sigma ** 2
            log_images.append(log_image)
        
        # Stack and find local maxima
        log_stack = np.stack(log_images, axis=-1)
        
        # Find local maxima
        local_maxima = peak_local_max(
            -log_stack.max(axis=-1),  # Negative for bright blobs
            min_distance=int(sigmas.min()),
            threshold_abs=0.1,
            exclude_border=False
        )
        
        # Filter by mask
        blobs = []
        for peak in local_maxima:
            if mask[peak[0], peak[1]]:
                # Find which scale gave maximum response
                scale_idx = log_stack[peak[0], peak[1]].argmax()
                radius = sigmas[scale_idx] * np.sqrt(2)
                blobs.append([peak[0], peak[1], radius])
        
        return np.array(blobs)
    
    def _detect_blobs_3d(self, image: np.ndarray, sigmas: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
        """Detect blobs in 3D volume"""
        # For 3D, we'll process slice by slice and then combine
        all_blobs = []
        
        for z in range(image.shape[0]):
            if mask[z].any():
                blobs_2d = self._detect_blobs_2d(image[z], sigmas, mask[z])
                
                # Add z coordinate
                for blob in blobs_2d:
                    all_blobs.append([z, blob[0], blob[1], blob[2]])
        
        # Merge nearby blobs in 3D
        if all_blobs:
            blobs_3d = self._merge_3d_blobs(np.array(all_blobs))
            return blobs_3d
        
        return np.array([])
    
    def _merge_3d_blobs(self, blobs: np.ndarray, distance_threshold: float = 5.0) -> np.ndarray:
        """Merge nearby blobs in 3D"""
        if len(blobs) == 0:
            return blobs
        
        # Simple clustering based on distance
        merged = []
        used = np.zeros(len(blobs), dtype=bool)
        
        for i in range(len(blobs)):
            if used[i]:
                continue
            
            # Find all blobs within threshold
            distances = np.sqrt(np.sum((blobs[:, :3] - blobs[i][:3])**2, axis=1))
            nearby = distances < distance_threshold
            
            # Average position and max radius
            cluster_blobs = blobs[nearby]
            mean_pos = cluster_blobs[:, :3].mean(axis=0)
            max_radius = cluster_blobs[:, 3].max()
            
            merged.append(list(mean_pos) + [max_radius])
            used[nearby] = True
        
        return np.array(merged)
    
    def segment_tumor(self, image: np.ndarray, seed_point: Tuple[int, ...],
                     lower_threshold: float = -100,
                     upper_threshold: float = 200) -> np.ndarray:
        """
        Segment tumor using region growing from seed point
        
        Args:
            image: CT image
            seed_point: Starting point (y, x) or (z, y, x)
            lower_threshold: Lower HU threshold
            upper_threshold: Upper HU threshold
        """
        # Convert to SimpleITK for region growing
        sitk_image = sitk.GetImageFromArray(image)
        
        # Region growing
        seg = sitk.ConnectedThreshold(
            sitk_image,
            seedList=[seed_point[::-1]],  # SimpleITK uses (x, y, z) order
            lower=lower_threshold,
            upper=upper_threshold,
            replaceValue=1
        )
        
        # Convert back to numpy
        mask = sitk.GetArrayFromImage(seg).astype(bool)
        
        # Morphological refinement
        mask = morphology.binary_closing(mask, morphology.ball(2))
        mask = morphology.remove_small_holes(mask, area_threshold=64)
        
        return mask


class MRISegmenter:
    """MRI-specific segmentation algorithms"""
    
    def __init__(self):
        self.brain_templates = {
            't1': {'threshold_factor': 0.5},
            't2': {'threshold_factor': 0.6},
            'flair': {'threshold_factor': 0.4}
        }
    
    def extract_brain(self, image: np.ndarray, 
                     sequence: str = 't1',
                     method: str = 'threshold') -> np.ndarray:
        """
        Brain extraction (skull stripping)
        
        Args:
            image: MRI image
            sequence: MRI sequence type ('t1', 't2', 'flair')
            method: Extraction method ('threshold', 'watershed', 'morphological')
        """
        if method == 'threshold':
            return self._threshold_brain_extraction(image, sequence)
        elif method == 'watershed':
            return self._watershed_brain_extraction(image)
        elif method == 'morphological':
            return self._morphological_brain_extraction(image)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _threshold_brain_extraction(self, image: np.ndarray, sequence: str) -> np.ndarray:
        """Simple threshold-based brain extraction"""
        # Get threshold factor for sequence
        factor = self.brain_templates.get(sequence, {}).get('threshold_factor', 0.5)
        
        # Calculate threshold
        threshold = threshold_otsu(image) * factor
        
        # Initial brain mask
        brain_mask = image > threshold
        
        # Remove small components
        brain_mask = morphology.remove_small_objects(brain_mask, min_size=1000)
        
        # Fill holes
        brain_mask = binary_fill_holes(brain_mask)
        
        # Get largest connected component (brain)
        labeled, num_features = label(brain_mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            largest_component = component_sizes[1:].argmax() + 1
            brain_mask = labeled == largest_component
        
        # Smooth boundaries
        brain_mask = morphology.binary_closing(brain_mask, morphology.ball(5))
        brain_mask = morphology.binary_opening(brain_mask, morphology.ball(3))
        
        return brain_mask
    
    def _watershed_brain_extraction(self, image: np.ndarray) -> np.ndarray:
        """Watershed-based brain extraction"""
        # Preprocessing
        smoothed = gaussian(image, sigma=1.0)
        
        # Calculate gradient
        gradient = ndimage.morphological_gradient(smoothed, size=3)
        
        # Markers for watershed
        markers = np.zeros_like(image, dtype=int)
        
        # Background markers (dark regions)
        markers[image < np.percentile(image, 10)] = 1
        
        # Foreground markers (bright regions)
        markers[image > np.percentile(image, 70)] = 2
        
        # Apply watershed
        labels = watershed(gradient, markers)
        
        # Brain is typically the foreground
        brain_mask = labels == 2
        
        # Clean up
        brain_mask = morphology.remove_small_objects(brain_mask, min_size=5000)
        brain_mask = binary_fill_holes(brain_mask)
        
        return brain_mask
    
    def _morphological_brain_extraction(self, image: np.ndarray) -> np.ndarray:
        """Morphological operations-based brain extraction"""
        # Initial threshold
        threshold = np.percentile(image, 30)
        brain_mask = image > threshold
        
        # Series of morphological operations
        # Remove thin connections
        brain_mask = morphology.binary_opening(brain_mask, morphology.ball(2))
        
        # Fill holes
        brain_mask = binary_fill_holes(brain_mask)
        
        # Remove small objects
        brain_mask = morphology.remove_small_objects(brain_mask, min_size=10000)
        
        # Get convex hull of the brain
        if len(image.shape) == 2:
            brain_mask = morphology.convex_hull_image(brain_mask)
        else:
            # Process slice by slice for 3D
            for i in range(brain_mask.shape[0]):
                if brain_mask[i].any():
                    brain_mask[i] = morphology.convex_hull_image(brain_mask[i])
        
        # Final smoothing
        brain_mask = morphology.binary_closing(brain_mask, morphology.ball(3))
        
        return brain_mask
    
    def segment_white_matter(self, image: np.ndarray, 
                           brain_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Segment white matter from T1-weighted MRI"""
        if brain_mask is None:
            brain_mask = self.extract_brain(image, sequence='t1')
        
        # Apply brain mask
        brain_region = image * brain_mask
        
        # White matter is typically bright in T1
        # Use multi-threshold approach
        thresholds = threshold_multiotsu(brain_region[brain_mask], classes=3)
        
        # White matter is the brightest class
        wm_mask = (brain_region > thresholds[1]) & brain_mask
        
        # Clean up
        wm_mask = morphology.remove_small_objects(wm_mask, min_size=100)
        wm_mask = morphology.binary_closing(wm_mask, morphology.ball(2))
        
        return wm_mask
    
    def segment_gray_matter(self, image: np.ndarray,
                          brain_mask: Optional[np.ndarray] = None,
                          wm_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Segment gray matter from T1-weighted MRI"""
        if brain_mask is None:
            brain_mask = self.extract_brain(image, sequence='t1')
        
        if wm_mask is None:
            wm_mask = self.segment_white_matter(image, brain_mask)
        
        # Apply brain mask
        brain_region = image * brain_mask
        
        # Gray matter is intermediate intensity
        thresholds = threshold_multiotsu(brain_region[brain_mask], classes=3)
        
        # Gray matter is between CSF and white matter
        gm_mask = (brain_region > thresholds[0]) & (brain_region <= thresholds[1]) & brain_mask
        
        # Remove white matter regions
        gm_mask = gm_mask & ~wm_mask
        
        # Clean up
        gm_mask = morphology.remove_small_objects(gm_mask, min_size=50)
        
        return gm_mask


class PETSegmenter:
    """PET-specific segmentation algorithms"""
    
    def __init__(self):
        self.suv_thresholds = {
            'fixed': 2.5,
            'liver': 1.5,  # Multiplier for liver mean
            'blood': 2.0   # Multiplier for blood pool
        }
    
    def segment_metabolic_volume(self, image: np.ndarray,
                               method: str = 'fixed',
                               threshold: Optional[float] = None,
                               reference_region: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segment metabolically active tumor volume
        
        Args:
            image: PET image (preferably in SUV)
            method: Thresholding method ('fixed', 'adaptive', 'gradient')
            threshold: Manual threshold override
            reference_region: Mask for reference region (liver, blood pool)
        """
        if method == 'fixed':
            return self._fixed_threshold_segmentation(image, threshold)
        elif method == 'adaptive':
            return self._adaptive_threshold_segmentation(image, reference_region)
        elif method == 'gradient':
            return self._gradient_based_segmentation(image)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _fixed_threshold_segmentation(self, image: np.ndarray,
                                    threshold: Optional[float] = None) -> np.ndarray:
        """Fixed SUV threshold segmentation"""
        if threshold is None:
            threshold = self.suv_thresholds['fixed']
        
        # Simple thresholding
        mask = image > threshold
        
        # Remove small regions
        mask = morphology.remove_small_objects(mask, min_size=10)
        
        # Fill holes
        mask = binary_fill_holes(mask)
        
        return mask
    
    def _adaptive_threshold_segmentation(self, image: np.ndarray,
                                       reference_region: Optional[np.ndarray] = None) -> np.ndarray:
        """Adaptive threshold based on reference region"""
        if reference_region is None:
            # Auto-detect liver region (simplified)
            # In practice, this would use anatomical information or atlas
            reference_region = self._estimate_liver_region(image)
        
        # Calculate reference statistics
        if reference_region.any():
            reference_mean = image[reference_region].mean()
            threshold = reference_mean * self.suv_thresholds['liver']
        else:
            # Fallback to fixed threshold
            threshold = self.suv_thresholds['fixed']
        
        # Apply threshold
        mask = image > threshold
        
        # Clean up
        mask = morphology.remove_small_objects(mask, min_size=10)
        mask = binary_fill_holes(mask)
        
        return mask
    
    def _gradient_based_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Gradient-based segmentation for PET"""
        # Smooth image
        smoothed = gaussian(image, sigma=2.0)
        
        # Calculate gradient magnitude
        gradient = ndimage.morphological_gradient(smoothed, size=3)
        
        # Find high uptake regions
        high_uptake = image > np.percentile(image, 90)
        
        # Use watershed from high uptake regions
        markers = label(high_uptake)[0]
        
        # Watershed
        labels = watershed(-smoothed, markers, mask=image > np.percentile(image, 70))
        
        # Create mask from labeled regions
        mask = labels > 0
        
        # Clean up
        mask = morphology.remove_small_objects(mask, min_size=20)
        
        return mask
    
    def _estimate_liver_region(self, image: np.ndarray) -> np.ndarray:
        """Estimate liver region for reference (simplified)"""
        # This is a very simplified approach
        # In practice, you would use anatomical priors or registration
        
        # Liver typically has moderate uptake
        liver_range = (np.percentile(image, 40), np.percentile(image, 60))
        
        # Initial mask
        liver_mask = (image > liver_range[0]) & (image < liver_range[1])
        
        # Spatial constraints (liver is typically in upper right)
        if len(image.shape) == 3:
            # Assume axial orientation
            z_center = image.shape[0] // 2
            liver_mask[:z_center//2] = False  # Remove lower slices
            liver_mask[int(z_center*1.5):] = False  # Remove upper slices
        
        # Get largest connected component
        labeled, num_features = label(liver_mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            largest = component_sizes[1:].argmax() + 1
            liver_mask = labeled == largest
        
        return liver_mask
    
    def calculate_suv_metrics(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate SUV metrics for segmented region"""
        if not mask.any():
            return {
                'suv_max': 0.0,
                'suv_mean': 0.0,
                'suv_peak': 0.0,
                'mtv': 0.0,  # Metabolic tumor volume
                'tlg': 0.0   # Total lesion glycolysis
            }
        
        # Extract values in mask
        values = image[mask]
        
        # SUVmax
        suv_max = values.max()
        
        # SUVmean
        suv_mean = values.mean()
        
        # SUVpeak (mean of 1cm³ sphere around max)
        max_loc = np.unravel_index(np.argmax(image * mask), image.shape)
        suv_peak = self._calculate_suv_peak(image, max_loc)
        
        # MTV (metabolic tumor volume) - assuming 1mm³ voxels
        mtv = mask.sum()
        
        # TLG (total lesion glycolysis)
        tlg = suv_mean * mtv
        
        return {
            'suv_max': float(suv_max),
            'suv_mean': float(suv_mean),
            'suv_peak': float(suv_peak),
            'mtv': float(mtv),
            'tlg': float(tlg)
        }
    
    def _calculate_suv_peak(self, image: np.ndarray, center: Tuple[int, ...],
                          radius: int = 6) -> float:
        """Calculate SUV peak in sphere around point"""
        # Create sphere mask
        if len(image.shape) == 2:
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]
            mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        else:
            z, y, x = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
            mask = ((x - center[2])**2 + (y - center[1])**2 + 
                   (z - center[0])**2 <= radius**2)
        
        # Calculate mean in sphere
        if mask.any():
            return image[mask].mean()
        else:
            return image[center]


# Convenience functions
def segment_lungs(image: np.ndarray, enhanced: bool = True) -> np.ndarray:
    """Segment lungs from CT image"""
    segmenter = CTSegmenter()
    return segmenter.segment_lungs(image, enhanced)


def segment_organs(image: np.ndarray, 
                  organs: List[str] = ['liver', 'spleen', 'kidney']) -> Dict[str, np.ndarray]:
    """Segment multiple organs from CT"""
    segmenter = CTSegmenter()
    return segmenter.segment_organs(image, organs)


def detect_nodules(image: np.ndarray, lung_mask: Optional[np.ndarray] = None,
                  min_size: float = 3.0, max_size: float = 30.0) -> List[Dict]:
    """Detect lung nodules in CT"""
    segmenter = CTSegmenter()
    return segmenter.detect_nodules(image, lung_mask, min_size, max_size)


def extract_brain(image: np.ndarray, sequence: str = 't1') -> np.ndarray:
    """Extract brain from MRI"""
    segmenter = MRISegmenter()
    return segmenter.extract_brain(image, sequence)


def segment_metabolic_volume(image: np.ndarray, method: str = 'fixed') -> np.ndarray:
    """Segment metabolically active regions in PET"""
    segmenter = PETSegmenter()
    return segmenter.segment_metabolic_volume(image, method)


def threshold_multiotsu(image: np.ndarray, classes: int = 3) -> List[float]:
    """Multi-Otsu thresholding (simplified implementation)"""
    # For simplicity, using percentiles
    # In practice, use skimage.filters.threshold_multiotsu
    percentiles = np.linspace(0, 100, classes + 1)[1:-1]
    return [np.percentile(image[image > 0], p) for p in percentiles]


if __name__ == "__main__":
    # Example usage
    import sys
    from data_management import load_medical_image
    
    if len(sys.argv) > 1:
        # Load image
        image, metadata = load_medical_image(sys.argv[1])
        
        # Segment based on modality
        if metadata.modality == 'CT':
            mask = segment_lungs(image)
            print(f"Segmented lungs: {mask.sum()} voxels")
            
            # Detect nodules
            nodules = detect_nodules(image, mask)
            print(f"Found {len(nodules)} potential nodules")
            
        elif metadata.modality == 'MR':
            mask = extract_brain(image)
            print(f"Extracted brain: {mask.sum()} voxels")
            
        elif metadata.modality == 'PT':
            mask = segment_metabolic_volume(image)
            print(f"Metabolic volume: {mask.sum()} voxels")
            
            # Calculate metrics
            segmenter = PETSegmenter()
            metrics = segmenter.calculate_suv_metrics(image, mask)
            print(f"SUV metrics: {metrics}")