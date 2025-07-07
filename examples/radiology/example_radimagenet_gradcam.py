"""
Advanced Gradient-based Visualizations for RadImageNet

Implements Grad-CAM, Grad-CAM++, and other gradient-based visualization techniques
for understanding what the RadImageNet models are learning from medical images.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import cv2
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import radiology modules
from radiology.data_management import load_dicom_series
from radiology.preprocessing import preprocess_ct
from radiology.ai_integration import RadImageNetProcessor, create_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """Implementation of Grad-CAM for medical image visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_image, target_class=None):
        """Generate Grad-CAM heatmap"""
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
            
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        pooled_gradients = self.gradients.mean(dim=[2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[:, i].unsqueeze(-1).unsqueeze(-1)
            
        # Average the channels
        heatmap = self.activations.mean(dim=1).squeeze()
        
        # ReLU and normalize
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap.cpu().numpy()


class GradCAMPlusPlus:
    """Implementation of Grad-CAM++ for improved localization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_image, target_class=None):
        """Generate Grad-CAM++ heatmap"""
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
            
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calculate alpha weights
        gradients_2 = self.gradients.pow(2)
        gradients_3 = gradients_2 * self.gradients
        
        sum_activations = self.activations.sum(dim=[2, 3])
        alpha_num = gradients_2
        alpha_denom = 2 * gradients_2 + (gradients_3 * sum_activations.unsqueeze(-1).unsqueeze(-1))
        alpha = alpha_num / (alpha_denom + 1e-8)
        
        # Apply ReLU to gradients
        weights = (alpha * F.relu(self.gradients)).sum(dim=[2, 3])
        
        # Generate heatmap
        heatmap = torch.zeros(self.activations.shape[2:], device=self.activations.device)
        for i in range(self.activations.shape[1]):
            heatmap += weights[0, i] * self.activations[0, i, :, :]
            
        # Normalize
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap.cpu().numpy()


class IntegratedGradients:
    """Implementation of Integrated Gradients for feature attribution"""
    
    def __init__(self, model):
        self.model = model
        
    def generate_attributions(self, input_image, baseline=None, steps=50):
        """Generate integrated gradients attributions"""
        if baseline is None:
            baseline = torch.zeros_like(input_image)
            
        # Generate interpolated images
        alphas = torch.linspace(0, 1, steps).to(input_image.device)
        interpolated_images = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_image - baseline)
            interpolated_images.append(interpolated)
            
        interpolated_images = torch.cat(interpolated_images)
        
        # Forward pass
        self.model.eval()
        interpolated_images.requires_grad_(True)
        outputs = self.model(interpolated_images)
        
        # Get target class
        target_class = outputs[-1].argmax()
        
        # Backward pass
        one_hot = torch.zeros_like(outputs)
        one_hot[:, target_class] = 1.0
        
        outputs.backward(gradient=one_hot)
        
        # Calculate integrated gradients
        gradients = interpolated_images.grad
        avg_gradients = gradients.mean(dim=0, keepdim=True)
        
        integrated_grads = (input_image - baseline) * avg_gradients
        
        return integrated_grads.squeeze().cpu().numpy()


class SmoothGrad:
    """Implementation of SmoothGrad for noise reduction in gradients"""
    
    def __init__(self, model):
        self.model = model
        
    def generate_saliency(self, input_image, n_samples=50, noise_level=0.1):
        """Generate SmoothGrad saliency map"""
        # Generate noisy samples
        noise_std = noise_level * (input_image.max() - input_image.min())
        
        total_gradients = torch.zeros_like(input_image)
        
        for _ in range(n_samples):
            # Add noise
            noise = torch.randn_like(input_image) * noise_std
            noisy_input = input_image + noise
            noisy_input.requires_grad_(True)
            
            # Forward pass
            output = self.model(noisy_input)
            target_class = output.argmax()
            
            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()
            
            # Accumulate gradients
            total_gradients += noisy_input.grad
            
        # Average gradients
        smooth_grad = total_gradients / n_samples
        
        return smooth_grad.squeeze().cpu().numpy()


class AdvancedVisualizationSuite:
    """Comprehensive suite of advanced visualization techniques"""
    
    def __init__(self, model_name='densenet121'):
        self.processor = RadImageNetProcessor(model_name=model_name, pretrained=True)
        self.model = self.processor.model
        self.device = self.processor.device
        
        # Get target layers for visualization
        self.target_layers = self._get_target_layers()
        
    def _get_target_layers(self):
        """Get relevant layers for visualization based on model architecture"""
        layers = {}
        
        if 'densenet' in self.processor.model_name:
            layers['early'] = self.model.features.denseblock1
            layers['middle'] = self.model.features.denseblock2
            layers['late'] = self.model.features.denseblock4
        elif 'resnet' in self.processor.model_name:
            layers['early'] = self.model.layer1
            layers['middle'] = self.model.layer2
            layers['late'] = self.model.layer4
            
        return layers
    
    def create_gradcam_comparison(self, image, save_path='gradcam_comparison.png'):
        """Compare Grad-CAM across different layers"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            image_tensor = image
        image_tensor = image_tensor.to(self.device)
        
        # Display original
        ax = axes[0, 0]
        img_np = image_tensor.squeeze().cpu().numpy()
        ax.imshow(img_np, cmap='gray')
        ax.set_title('Original Image')
        ax.axis('off')
        
        # Generate Grad-CAM for different layers
        layer_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for idx, (layer_name, layer) in enumerate(self.target_layers.items()):
            if idx >= len(layer_positions):
                break
                
            pos = layer_positions[idx]
            ax = axes[pos]
            
            # Generate heatmap
            gradcam = GradCAM(self.model, layer)
            heatmap = gradcam.generate_heatmap(image_tensor)
            
            # Resize heatmap to match image size
            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            
            # Overlay heatmap on image
            ax.imshow(img_np, cmap='gray', alpha=0.5)
            im = ax.imshow(heatmap_resized, cmap='jet', alpha=0.5)
            ax.set_title(f'Grad-CAM: {layer_name}')
            ax.axis('off')
            
        plt.suptitle('Grad-CAM Visualization Across Network Layers', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Grad-CAM comparison saved to: {save_path}")
        
    def create_attribution_comparison(self, image, save_path='attributions.png'):
        """Compare different attribution methods"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Prepare image
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            image_tensor = image
        image_tensor = image_tensor.to(self.device)
        
        img_np = image_tensor.squeeze().cpu().numpy()
        
        # 1. Original image
        ax = axes[0, 0]
        ax.imshow(img_np, cmap='gray')
        ax.set_title('Original Image')
        ax.axis('off')
        
        # 2. Vanilla gradients
        ax = axes[0, 1]
        image_tensor.requires_grad_(True)
        output = self.model(image_tensor)
        target = output.argmax()
        self.model.zero_grad()
        output[0, target].backward()
        vanilla_grad = image_tensor.grad.squeeze().cpu().numpy()
        
        ax.imshow(np.abs(vanilla_grad), cmap='hot')
        ax.set_title('Vanilla Gradients')
        ax.axis('off')
        
        # 3. SmoothGrad
        ax = axes[0, 2]
        smoothgrad = SmoothGrad(self.model)
        smooth_saliency = smoothgrad.generate_saliency(image_tensor.detach())
        
        ax.imshow(np.abs(smooth_saliency), cmap='hot')
        ax.set_title('SmoothGrad')
        ax.axis('off')
        
        # 4. Integrated Gradients
        ax = axes[1, 0]
        intgrad = IntegratedGradients(self.model)
        attributions = intgrad.generate_attributions(image_tensor.detach())
        
        ax.imshow(np.abs(attributions), cmap='hot')
        ax.set_title('Integrated Gradients')
        ax.axis('off')
        
        # 5. Grad-CAM++
        ax = axes[1, 1]
        last_conv = list(self.target_layers.values())[-1]
        gradcampp = GradCAMPlusPlus(self.model, last_conv)
        heatmap_pp = gradcampp.generate_heatmap(image_tensor.detach())
        heatmap_pp_resized = cv2.resize(heatmap_pp, (img_np.shape[1], img_np.shape[0]))
        
        ax.imshow(img_np, cmap='gray', alpha=0.5)
        ax.imshow(heatmap_pp_resized, cmap='jet', alpha=0.5)
        ax.set_title('Grad-CAM++')
        ax.axis('off')
        
        # 6. Combined visualization
        ax = axes[1, 2]
        # Normalize all attributions
        combined = (np.abs(vanilla_grad) + np.abs(smooth_saliency) + np.abs(attributions)) / 3
        combined = combined / combined.max()
        
        ax.imshow(combined, cmap='hot')
        ax.set_title('Combined Attributions')
        ax.axis('off')
        
        plt.suptitle('Attribution Method Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Attribution comparison saved to: {save_path}")
        
    def create_occlusion_sensitivity_map(self, image, occlusion_size=50, stride=25,
                                       save_path='occlusion_sensitivity.png'):
        """Create occlusion sensitivity map"""
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:
            image_tensor = image
        image_tensor = image_tensor.to(self.device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(image_tensor)
            baseline_prob = F.softmax(baseline_output, dim=1).max().item()
            target_class = baseline_output.argmax().item()
        
        # Create sensitivity map
        h, w = image_tensor.shape[2:]
        sensitivity_map = np.zeros((h // stride, w // stride))
        
        for i in range(0, h - occlusion_size, stride):
            for j in range(0, w - occlusion_size, stride):
                # Create occluded image
                occluded = image_tensor.clone()
                occluded[:, :, i:i+occlusion_size, j:j+occlusion_size] = 0
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(occluded)
                    prob = F.softmax(output, dim=1)[0, target_class].item()
                
                # Calculate sensitivity
                sensitivity = baseline_prob - prob
                sensitivity_map[i // stride, j // stride] = sensitivity
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        img_np = image_tensor.squeeze().cpu().numpy()
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Sensitivity map
        sensitivity_resized = cv2.resize(sensitivity_map, (w, h))
        ax2.imshow(img_np, cmap='gray', alpha=0.5)
        im = ax2.imshow(sensitivity_resized, cmap='hot', alpha=0.7)
        ax2.set_title(f'Occlusion Sensitivity (size={occlusion_size})')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Occlusion sensitivity saved to: {save_path}")
        
    def create_layer_activation_maximization(self, save_path='activation_max.png'):
        """Visualize what maximally activates different layers"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        layer_names = list(self.target_layers.keys())[:6]
        
        for idx, layer_name in enumerate(layer_names):
            ax = axes[idx]
            
            # Create random input
            input_tensor = torch.randn(1, 1, 224, 224, requires_grad=True, device=self.device)
            
            # Optimization loop
            optimizer = torch.optim.Adam([input_tensor], lr=0.1)
            
            for _ in range(100):
                optimizer.zero_grad()
                
                # Forward pass
                features = self.model(input_tensor)
                
                # Maximize mean activation of target layer
                # This is simplified - in practice you'd hook into specific layer
                loss = -features.mean()
                
                loss.backward()
                optimizer.step()
                
                # Constrain values
                input_tensor.data = torch.clamp(input_tensor.data, -1, 1)
            
            # Visualize result
            result = input_tensor.detach().squeeze().cpu().numpy()
            ax.imshow(result, cmap='gray')
            ax.set_title(f'Max Activation: {layer_name}')
            ax.axis('off')
        
        plt.suptitle('Layer Activation Maximization', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Activation maximization saved to: {save_path}")


def create_comprehensive_gradient_report(ct_volume):
    """Generate comprehensive gradient-based visualization report"""
    logger.info("Generating Gradient-based Visualization Report")
    logger.info("=" * 50)
    
    # Create output directory
    output_dir = Path("results_gradcam_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    viz_suite = AdvancedVisualizationSuite('densenet121')
    
    # Select representative slices
    slices = [ct_volume.shape[0] // 4, ct_volume.shape[0] // 2, 3 * ct_volume.shape[0] // 4]
    
    for slice_idx in slices:
        logger.info(f"\nProcessing slice {slice_idx}...")
        
        # Preprocess slice
        ct_slice = preprocess_ct(ct_volume[slice_idx:slice_idx+1], window='lung')[0]
        
        # 1. Grad-CAM comparison
        viz_suite.create_gradcam_comparison(
            ct_slice,
            save_path=output_dir / f'gradcam_comparison_slice{slice_idx}.png'
        )
        
        # 2. Attribution comparison
        viz_suite.create_attribution_comparison(
            ct_slice,
            save_path=output_dir / f'attribution_comparison_slice{slice_idx}.png'
        )
        
        # 3. Occlusion sensitivity
        viz_suite.create_occlusion_sensitivity_map(
            ct_slice,
            save_path=output_dir / f'occlusion_sensitivity_slice{slice_idx}.png'
        )
    
    # 4. Layer activation maximization (slice-independent)
    viz_suite.create_layer_activation_maximization(
        save_path=output_dir / 'layer_activation_maximization.png'
    )
    
    logger.info(f"\nAll gradient visualizations saved to: {output_dir}")
    logger.info("Gradient visualization report complete!")


def main():
    """Main function to run gradient-based visualizations"""
    
    # Load sample CT data
    ct_dir = Path("../samples/CT")
    if not ct_dir.exists():
        logger.error(f"CT samples not found at {ct_dir}")
        logger.info("Creating synthetic data for demonstration...")
        
        # Create synthetic CT-like data
        ct_volume = np.random.randn(30, 512, 512) * 30 + 100
        # Add some structure
        for i in range(30):
            center = (256, 256)
            y, x = np.ogrid[:512, :512]
            mask = (x - center[0])**2 + (y - center[1])**2 <= (100 + i*3)**2
            ct_volume[i][mask] += 50
    else:
        ct_volume, metadata = load_dicom_series(ct_dir)
        logger.info(f"Loaded CT volume: {ct_volume.shape}")
    
    # Generate gradient-based visualizations
    create_comprehensive_gradient_report(ct_volume)
    
    logger.info("\n" + "=" * 50)
    logger.info("Gradient-based visualization demo complete!")
    logger.info("Check results_gradcam_visualizations/ for all outputs")


if __name__ == "__main__":
    main()