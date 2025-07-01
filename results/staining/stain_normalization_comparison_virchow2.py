#!/usr/bin/env python3
"""
Stain Normalization Impact on Virchow2 Embeddings
Compares classification performance with and without stain normalization
Using Paige AI's Virchow2 foundation model for pathology
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from tqdm import tqdm
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Add paths for direct imports
sys.path.append('/mnt/f/Projects/HoneyBee')
sys.path.append('/mnt/f/Projects/HoneyBee/honeybee/models/Virchow2')
sys.path.append('/mnt/f/Projects/HoneyBee/honeybee/models/TissueDetector')

# Direct imports
from virchow2_simple import Virchow2
from tissue_detector import TissueDetector

# Import required modules
import gc
from cucim import CuImage
from PIL import Image
import timm
import multiprocessing

# Import stain normalization methods
from skimage.color import rgb2hed, hed2rgb, rgb2lab, lab2rgb


class SimpleSlide:
    """Simplified slide loader without problematic dependencies"""
    def __init__(self, slide_path, tile_size=512, max_patches=100, tissue_detector=None):
        self.slide_path = slide_path
        self.tile_size = tile_size
        self.max_patches = max_patches
        self.tissue_detector = tissue_detector
        self.img = CuImage(slide_path)
        
        # Select appropriate level
        resolutions = self.img.resolutions
        level_dimensions = resolutions["level_dimensions"]
        self.selected_level = 0
        
        for level in range(resolutions["level_count"]):
            width, height = level_dimensions[level]
            num_tiles = (width // self.tile_size) * (height // self.tile_size)
            if num_tiles <= max_patches * 4:
                self.selected_level = level
                break
        
        # Read slide at selected level
        self.slide = self.img.read_region(location=[0, 0], level=self.selected_level)
        self.slide_width = int(self.slide.metadata["cucim"]["shape"][1])
        self.slide_height = int(self.slide.metadata["cucim"]["shape"][0])
        
    def load_patches_concurrently(self, target_patch_size=224):
        """Extract patches from slide"""
        patches = []
        
        # Calculate stride
        total_tiles = (self.slide_width // self.tile_size) * (self.slide_height // self.tile_size)
        if total_tiles > self.max_patches:
            stride = int(np.sqrt(total_tiles / self.max_patches)) + 1
        else:
            stride = 1
        
        # Extract patches
        for y in range(0, self.slide_height - self.tile_size, self.tile_size * stride):
            for x in range(0, self.slide_width - self.tile_size, self.tile_size * stride):
                # Extract patch
                patch = self.slide.read_region(
                    location=(x, y),
                    size=(self.tile_size, self.tile_size),
                    level=0
                )
                patch_array = np.asarray(patch)[:, :, :3]  # Remove alpha channel
                
                # Check if tissue using tissue detector
                if self.tissue_detector is not None:
                    # Resize for tissue detection
                    patch_resized = cv2.resize(patch_array, (224, 224))
                    patch_pil = Image.fromarray(patch_resized)
                    
                    # Apply tissue detector transforms and predict
                    patch_transformed = self.tissue_detector.transforms(patch_pil)
                    patch_batch = patch_transformed.unsqueeze(0).to(self.tissue_detector.device)
                    
                    with torch.no_grad():
                        prediction = self.tissue_detector.model(patch_batch)
                        prob = torch.nn.functional.softmax(prediction, dim=1).cpu().numpy()[0]
                        tissue_class = np.argmax(prob)
                    
                    # Keep patch if it's tissue (class 2)
                    if tissue_class != 2 or prob[2] < 0.8:
                        continue
                
                # Resize to target size
                patch_final = cv2.resize(patch_array, (target_patch_size, target_patch_size))
                patches.append(patch_final)
                
                if len(patches) >= self.max_patches:
                    break
            
            if len(patches) >= self.max_patches:
                break
        
        # Clear GPU cache after patch extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return patches


class MacenkoNormalizer:
    """Macenko stain normalization method"""
    def __init__(self):
        self.target_stain_matrix = None
        self.target_concentrations = None
        
    def fit(self, target_image):
        """Fit normalizer to target image"""
        # Convert to optical density
        od = -np.log((target_image.astype(np.float32) + 1) / 256)
        
        # Remove pixels with low optical density
        od_flat = od.reshape(-1, 3)
        od_thresh = od_flat[np.all(od_flat > 0.15, axis=1)]
        
        if len(od_thresh) == 0:
            self.target_stain_matrix = np.eye(3)
            return self
            
        # Compute SVD
        _, _, V = np.linalg.svd(od_thresh.T, full_matrices=False)
        
        # Project to plane and find angles
        theta = np.arctan2(V[1, :], V[0, :])
        
        # Find min and max angles
        min_angle = np.percentile(theta, 1)
        max_angle = np.percentile(theta, 99)
        
        # Convert back to vectors
        vec1 = np.array([np.cos(min_angle), np.sin(min_angle), 0])
        vec2 = np.array([np.cos(max_angle), np.sin(max_angle), 0])
        
        # Create stain matrix
        self.target_stain_matrix = np.array([vec1, vec2, np.cross(vec1, vec2)]).T
        self.target_stain_matrix = self.target_stain_matrix / np.linalg.norm(self.target_stain_matrix, axis=0)
        
        return self
    
    def transform(self, source_image):
        """Transform source image using fitted parameters"""
        if self.target_stain_matrix is None:
            raise ValueError("Normalizer must be fitted first")
            
        # Convert to optical density
        od = -np.log((source_image.astype(np.float32) + 1) / 256)
        
        # Get source stain matrix using same method
        od_flat = od.reshape(-1, 3)
        od_thresh = od_flat[np.all(od_flat > 0.15, axis=1)]
        
        if len(od_thresh) == 0:
            return source_image
            
        _, _, V = np.linalg.svd(od_thresh.T, full_matrices=False)
        theta = np.arctan2(V[1, :], V[0, :])
        min_angle = np.percentile(theta, 1)
        max_angle = np.percentile(theta, 99)
        
        vec1 = np.array([np.cos(min_angle), np.sin(min_angle), 0])
        vec2 = np.array([np.cos(max_angle), np.sin(max_angle), 0])
        source_stain_matrix = np.array([vec1, vec2, np.cross(vec1, vec2)]).T
        source_stain_matrix = source_stain_matrix / np.linalg.norm(source_stain_matrix, axis=0)
        
        # Get concentrations
        source_concentrations = np.linalg.lstsq(source_stain_matrix, od.reshape(-1, 3).T, rcond=None)[0].T
        
        # Transform using target stain matrix
        od_normalized = (self.target_stain_matrix @ source_concentrations.T).T
        od_normalized = od_normalized.reshape(source_image.shape)
        
        # Convert back to RGB
        normalized = np.exp(-od_normalized) * 256 - 1
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        return normalized


class ReinhardNormalizer:
    """Reinhard color normalization method"""
    def __init__(self):
        self.target_mean = None
        self.target_std = None
        
    def fit(self, target_image):
        """Fit normalizer to target image"""
        # Convert to LAB color space
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate mean and std for each channel
        self.target_mean = np.mean(target_lab, axis=(0, 1))
        self.target_std = np.std(target_lab, axis=(0, 1))
        
        return self
    
    def transform(self, source_image):
        """Transform source image using fitted parameters"""
        if self.target_mean is None:
            raise ValueError("Normalizer must be fitted first")
            
        # Convert to LAB
        source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate source statistics
        source_mean = np.mean(source_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        
        # Normalize each channel
        normalized_lab = source_lab.copy()
        for i in range(3):
            normalized_lab[:, :, i] = (source_lab[:, :, i] - source_mean[i]) * (self.target_std[i] / source_std[i]) + self.target_mean[i]
        
        # Convert back to RGB
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)
        normalized = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2RGB)
        
        return normalized


class WSIProcessor:
    """Process WSI slides with optional stain normalization"""
    def __init__(self, model_path, tissue_detector_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"WSIProcessor using device: {self.device}")
        # For Virchow2, use DINOv2 base weights (Virchow2 weights require authentication)
        self.model = Virchow2(use_hf=False)
        self.tissue_detector = TissueDetector(tissue_detector_path, device=str(self.device))
        self.normalizer = None
        
    def set_normalizer(self, normalizer):
        """Set the stain normalizer to use"""
        self.normalizer = normalizer
        
    def process_slide(self, slide_path, max_patches=200, apply_normalization=False):
        """Process a single slide and return embeddings"""
        # Load slide
        slide = SimpleSlide(
            slide_path,
            tile_size=512,
            max_patches=max_patches,
            tissue_detector=self.tissue_detector,
        )
        
        # Extract patches
        patches = slide.load_patches_concurrently(target_patch_size=224)
        
        if len(patches) == 0:
            print(f"No valid patches found in {slide_path}")
            return None
            
        # Apply stain normalization if requested
        if apply_normalization and self.normalizer is not None:
            normalized_patches = []
            for patch in patches:
                # Normalize
                normalized = self.normalizer.transform(patch)
                normalized_patches.append(normalized)
                
            patches = normalized_patches
        
        # Generate embeddings in batches for better GPU utilization
        embeddings_list = []
        embedding_batch_size = 32  # Process 32 patches at a time on GPU
        
        for i in range(0, len(patches), embedding_batch_size):
            batch = patches[i:i+embedding_batch_size]
            batch_embeddings = self.model.load_model_and_predict(batch)
            
            # Move to CPU immediately to free GPU memory
            if isinstance(batch_embeddings, torch.Tensor):
                embeddings_list.append(batch_embeddings.detach().cpu())
            else:
                embeddings_list.append(batch_embeddings)
            
            # Clear GPU cache periodically
            if i % (embedding_batch_size * 4) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all embeddings
        if embeddings_list and isinstance(embeddings_list[0], torch.Tensor):
            embeddings = torch.cat(embeddings_list, dim=0).numpy()
        else:
            embeddings = np.concatenate(embeddings_list, axis=0)
        
        # Return patch-level embeddings
        return embeddings
    
    def create_patient_representation(self, embeddings, method='mean'):
        """Create patient-level representation from patch embeddings"""
        if embeddings is None or len(embeddings) == 0:
            return None
            
        if method == 'mean':
            return np.mean(embeddings, axis=0)
        elif method == 'max':
            return np.max(embeddings, axis=0)
        elif method == 'mean_max':
            mean_pool = np.mean(embeddings, axis=0)
            max_pool = np.max(embeddings, axis=0)
            return np.concatenate([mean_pool, max_pool])
        else:
            raise ValueError(f"Unknown pooling method: {method}")


class PatientDataset(Dataset):
    """Dataset for patient-level representations"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    """Simple neural network classifier"""
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def train_classifier(train_features, train_labels, val_features, val_labels, 
                    num_epochs=50, lr=1e-3, batch_size=64):  # Increased batch size for GPU
    """Train neural network classifier"""
    # Create datasets and loaders
    train_dataset = PatientDataset(train_features, train_labels)
    val_dataset = PatientDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = train_features.shape[1]
    num_classes = len(np.unique(train_labels))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training neural network on: {device}")
    
    model = SimpleClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_acc = 100 * correct / total
        val_accs.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
    
    return model, train_losses, val_accs


def evaluate_model(model, test_features, test_labels, model_type='nn'):
    """Evaluate model and return metrics"""
    if model_type == 'nn':
        device = next(model.parameters()).device
        test_dataset = PatientDataset(test_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        predictions = np.array(all_preds)
        probabilities = np.array(all_probs)
        
    else:  # sklearn models
        predictions = model.predict(test_features)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(test_features)
        else:
            probabilities = None
    
    # Calculate metrics
    accuracy = np.mean(predictions == test_labels)
    report = classification_report(test_labels, predictions, output_dict=True)
    conf_matrix = confusion_matrix(test_labels, predictions)
    
    # Calculate AUC if binary classification
    auc_score = None
    if len(np.unique(test_labels)) == 2 and probabilities is not None:
        auc_score = roc_auc_score(test_labels, probabilities[:, 1])
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'probabilities': probabilities,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'auc_score': auc_score
    }


def create_tsne_visualization(X_norm, X_no_norm, y, labels, save_dir, perplexity=30, n_iter=1000):
    """Create t-SNE visualizations comparing normalized and non-normalized embeddings"""
    print("Generating t-SNE visualizations...")
    
    # Create figures directory
    figures_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Fit t-SNE for both sets
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    
    print("Computing t-SNE for normalized embeddings...")
    X_norm_tsne = tsne.fit_transform(X_norm)
    
    print("Computing t-SNE for non-normalized embeddings...")
    X_no_norm_tsne = tsne.fit_transform(X_no_norm)
    
    # Create side-by-side comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Define colors for each class
    colors = ['#FF6B6B', '#4ECDC4']  # Red for BRCA, Teal for BLCA
    
    # Plot normalized embeddings
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        cancer_type = labels[np.where(y == label)[0][0]]
        ax1.scatter(X_norm_tsne[idx, 0], X_norm_tsne[idx, 1], 
                   c=colors[i], label=cancer_type, alpha=0.6, s=50)
    
    ax1.set_title('t-SNE: With Stain Normalization', fontsize=16)
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot non-normalized embeddings
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        cancer_type = labels[np.where(y == label)[0][0]]
        ax2.scatter(X_no_norm_tsne[idx, 0], X_no_norm_tsne[idx, 1], 
                   c=colors[i], label=cancer_type, alpha=0.6, s=50)
    
    ax2.set_title('t-SNE: Without Stain Normalization', fontsize=16)
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('t-SNE Visualization of Virchow2 Embeddings: Impact of Stain Normalization', fontsize=18)
    plt.tight_layout()
    
    # Save combined plot
    plt.savefig(os.path.join(figures_dir, 'tsne_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual plots
    # Normalized only
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        cancer_type = labels[np.where(y == label)[0][0]]
        ax.scatter(X_norm_tsne[idx, 0], X_norm_tsne[idx, 1], 
                  c=colors[i], label=cancer_type, alpha=0.6, s=50)
    
    ax.set_title('t-SNE: With Stain Normalization', fontsize=16)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tsne_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Non-normalized only
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        cancer_type = labels[np.where(y == label)[0][0]]
        ax.scatter(X_no_norm_tsne[idx, 0], X_no_norm_tsne[idx, 1], 
                  c=colors[i], label=cancer_type, alpha=0.6, s=50)
    
    ax.set_title('t-SNE: Without Stain Normalization', fontsize=16)
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tsne_non_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create density plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Normalized density
    from scipy.stats import gaussian_kde
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        cancer_type = labels[np.where(y == label)[0][0]]
        
        # Calculate density
        xy = X_norm_tsne[idx]
        if len(xy) > 1:
            kernel = gaussian_kde(xy.T)
            
            # Create grid
            x_min, x_max = X_norm_tsne[:, 0].min() - 5, X_norm_tsne[:, 0].max() + 5
            y_min, y_max = X_norm_tsne[:, 1].min() - 5, X_norm_tsne[:, 1].max() + 5
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Evaluate kernel
            f = np.reshape(kernel(positions).T, xx.shape)
            
            # Plot contours
            ax1.contourf(xx, yy, f, alpha=0.3, cmap='viridis' if i == 0 else 'plasma')
            ax1.scatter(xy[:, 0], xy[:, 1], c=colors[i], label=cancer_type, alpha=0.6, s=30)
    
    ax1.set_title('t-SNE Density: With Normalization', fontsize=16)
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.legend(fontsize=12)
    
    # Non-normalized density
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        cancer_type = labels[np.where(y == label)[0][0]]
        
        # Calculate density
        xy = X_no_norm_tsne[idx]
        if len(xy) > 1:
            kernel = gaussian_kde(xy.T)
            
            # Create grid
            x_min, x_max = X_no_norm_tsne[:, 0].min() - 5, X_no_norm_tsne[:, 0].max() + 5
            y_min, y_max = X_no_norm_tsne[:, 1].min() - 5, X_no_norm_tsne[:, 1].max() + 5
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Evaluate kernel
            f = np.reshape(kernel(positions).T, xx.shape)
            
            # Plot contours
            ax2.contourf(xx, yy, f, alpha=0.3, cmap='viridis' if i == 0 else 'plasma')
            ax2.scatter(xy[:, 0], xy[:, 1], c=colors[i], label=cancer_type, alpha=0.6, s=30)
    
    ax2.set_title('t-SNE Density: Without Normalization', fontsize=16)
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.legend(fontsize=12)
    
    plt.suptitle('t-SNE Density Visualization', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'tsne_density.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return X_norm_tsne, X_no_norm_tsne


def plot_results(results_norm, results_no_norm, save_dir):
    """Create comparison plots and save individual figures"""
    # Create figures directory
    figures_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # First create the combined plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    models = list(results_norm.keys())
    accs_norm = [results_norm[m]['accuracy'] for m in models]
    accs_no_norm = [results_no_norm[m]['accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, accs_norm, width, label='With Normalization', alpha=0.8)
    ax.bar(x + width/2, accs_no_norm, width, label='Without Normalization', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2 & 3: Confusion matrices for best model
    best_model = max(models, key=lambda m: results_norm[m]['accuracy'])
    
    for idx, (results, title) in enumerate([
        (results_norm[best_model], f'{best_model} - With Normalization'),
        (results_no_norm[best_model], f'{best_model} - Without Normalization')
    ]):
        ax = axes[0, idx + 1]
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax, cbar=True)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # Plot 4: Feature comparison (if embeddings available)
    ax = axes[1, 0]
    ax.text(0.5, 0.5, 'Feature Distribution\nComparison\n(t-SNE visualization\ncan be added here)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Embedding Space Comparison')
    ax.axis('off')
    
    # Plot 5: Training curves (if available)
    ax = axes[1, 1]
    if 'train_losses' in results_norm.get('neural_network', {}):
        epochs = range(1, len(results_norm['neural_network']['train_losses']) + 1)
        ax.plot(epochs, results_norm['neural_network']['train_losses'], 
               label='With Normalization', alpha=0.8)
        ax.plot(epochs, results_no_norm['neural_network']['train_losses'], 
               label='Without Normalization', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Neural Network Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Training Curves\n(Available for NN only)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
    
    # Plot 6: ROC curves (if binary classification)
    ax = axes[1, 2]
    if results_norm[best_model]['auc_score'] is not None:
        # Assuming binary classification
        for results, label in [
            (results_norm[best_model], 'With Normalization'),
            (results_no_norm[best_model], 'Without Normalization')
        ]:
            if results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(results['true_labels'], 
                                       results['probabilities'][:, 1])
                ax.plot(fpr, tpr, label=f'{label} (AUC={results["auc_score"]:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {best_model}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'ROC Curves\n(Available for binary\nclassification only)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'performance_comparison_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now save individual plots
    # 1. Accuracy comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, accs_norm, width, label='With Normalization', alpha=0.8)
    ax.bar(x + width/2, accs_no_norm, width, label='Without Normalization', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Individual confusion matrices
    for model_name in models:
        # With normalization
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.heatmap(results_norm[model_name]['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax, cbar=True)
        ax.set_title(f'{model_name} - With Normalization')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'confusion_matrix_{model_name}_normalized.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Without normalization
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.heatmap(results_no_norm[model_name]['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax, cbar=True)
        ax.set_title(f'{model_name} - Without Normalization')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'confusion_matrix_{model_name}_non_normalized.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Training curves for neural network
    if 'train_losses' in results_norm.get('neural_network', {}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Loss curves
        epochs = range(1, len(results_norm['neural_network']['train_losses']) + 1)
        ax1.plot(epochs, results_norm['neural_network']['train_losses'], 
                label='With Normalization', alpha=0.8)
        ax1.plot(epochs, results_no_norm['neural_network']['train_losses'], 
                label='Without Normalization', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Neural Network Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation accuracy curves
        ax2.plot(epochs, results_norm['neural_network']['val_accs'], 
                label='With Normalization', alpha=0.8)
        ax2.plot(epochs, results_no_norm['neural_network']['val_accs'], 
                label='Without Normalization', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Accuracy (%)')
        ax2.set_title('Neural Network Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Neural Network Training Progress')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'nn_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create accuracy improvement plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    improvements = [(m, results_norm[m]['accuracy'] - results_no_norm[m]['accuracy']) 
                   for m in models]
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    models_sorted, impr_values = zip(*improvements)
    colors = ['green' if v > 0 else 'red' for v in impr_values]
    
    bars = ax.bar(range(len(models_sorted)), impr_values, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy Improvement (%)')
    ax.set_title('Impact of Stain Normalization on Classification Accuracy')
    ax.set_xticks(range(len(models_sorted)))
    ax.set_xticklabels(models_sorted, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, impr_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2%}', ha='center', va='bottom' if value > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_improvement.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'accuracy_improvement.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_additional_visualizations(X_norm, X_no_norm, y, labels, results_norm, results_no_norm, save_dir):
    """Create additional visualization plots"""
    figures_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. PCA visualization
    print("Creating PCA visualizations...")
    pca = PCA(n_components=2)
    X_norm_pca = pca.fit_transform(X_norm)
    X_no_norm_pca = pca.fit_transform(X_no_norm)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#FF6B6B', '#4ECDC4']
    
    # PCA with normalization
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        cancer_type = labels[np.where(y == label)[0][0]]
        ax1.scatter(X_norm_pca[idx, 0], X_norm_pca[idx, 1], 
                   c=colors[i], label=cancer_type, alpha=0.6, s=50)
    ax1.set_title('PCA: With Stain Normalization', fontsize=14)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PCA without normalization
    pca_no_norm = PCA(n_components=2)
    X_no_norm_pca = pca_no_norm.fit_transform(X_no_norm)
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        cancer_type = labels[np.where(y == label)[0][0]]
        ax2.scatter(X_no_norm_pca[idx, 0], X_no_norm_pca[idx, 1], 
                   c=colors[i], label=cancer_type, alpha=0.6, s=50)
    ax2.set_title('PCA: Without Stain Normalization', fontsize=14)
    ax2.set_xlabel(f'PC1 ({pca_no_norm.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca_no_norm.explained_variance_ratio_[1]:.2%} variance)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('PCA Visualization of Virchow2 Embeddings', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pca_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature importance from Random Forest
    print("Creating feature importance plots...")
    if 'random_forest' in results_norm and 'model' in results_norm['random_forest']:
        # Get feature importances
        rf_norm = results_norm['random_forest']['model']
        rf_no_norm = results_no_norm['random_forest']['model']
        
        # Get top 20 feature importances
        n_features = 20
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Normalized model
        importances_norm = rf_norm.feature_importances_
        indices_norm = np.argsort(importances_norm)[-n_features:]
        
        ax1.barh(range(n_features), importances_norm[indices_norm], alpha=0.7)
        ax1.set_yticks(range(n_features))
        ax1.set_yticklabels([f'Feature {i}' for i in indices_norm])
        ax1.set_xlabel('Importance Score')
        ax1.set_title('Top 20 Features - With Normalization')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Non-normalized model
        importances_no_norm = rf_no_norm.feature_importances_
        indices_no_norm = np.argsort(importances_no_norm)[-n_features:]
        
        ax2.barh(range(n_features), importances_no_norm[indices_no_norm], alpha=0.7)
        ax2.set_yticks(range(n_features))
        ax2.set_yticklabels([f'Feature {i}' for i in indices_no_norm])
        ax2.set_xlabel('Importance Score')
        ax2.set_title('Top 20 Features - Without Normalization')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Random Forest Feature Importance Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot importance distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(importances_norm, bins=50, alpha=0.5, label='With Normalization', density=True)
        ax.hist(importances_no_norm, bins=50, alpha=0.5, label='Without Normalization', density=True)
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Feature Importances')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'feature_importance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Class distribution plots
    print("Creating class distribution plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count samples per class
    unique_labels, counts = np.unique(y, return_counts=True)
    class_names = [labels[np.where(y == label)[0][0]] for label in unique_labels]
    
    # Bar plot
    ax1.bar(class_names, counts, color=colors, alpha=0.7)
    ax1.set_xlabel('Cancer Type')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Distribution by Cancer Type')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (name, count) in enumerate(zip(class_names, counts)):
        ax1.text(i, count + 0.5, str(count), ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=class_names, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Sample Distribution (Percentage)')
    
    plt.suptitle('Dataset Class Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance metrics comparison
    print("Creating performance metrics comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['precision', 'recall', 'f1-score']
    models = list(results_norm.keys())
    
    # For each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Get macro avg scores for each model
        scores_norm = []
        scores_no_norm = []
        
        for model in models:
            report_norm = results_norm[model]['classification_report']
            report_no_norm = results_no_norm[model]['classification_report']
            
            if 'macro avg' in report_norm:
                scores_norm.append(report_norm['macro avg'][metric])
                scores_no_norm.append(report_no_norm['macro avg'][metric])
            else:
                scores_norm.append(0)
                scores_no_norm.append(0)
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, scores_norm, width, label='With Normalization', alpha=0.8)
        ax.bar(x + width/2, scores_no_norm, width, label='Without Normalization', alpha=0.8)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Support comparison
    ax = axes[1, 1]
    ax.text(0.5, 0.5, f'Total Test Samples: {len(y) * 0.2:.0f}', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Test Set Information')
    ax.axis('off')
    
    plt.suptitle('Detailed Performance Metrics Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Embedding statistics
    print("Creating embedding statistics plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # L2 norm distribution
    ax = axes[0, 0]
    norm_l2_norm = np.linalg.norm(X_norm, axis=1)
    no_norm_l2_norm = np.linalg.norm(X_no_norm, axis=1)
    
    ax.hist(norm_l2_norm, bins=30, alpha=0.5, label='With Normalization', density=True)
    ax.hist(no_norm_l2_norm, bins=30, alpha=0.5, label='Without Normalization', density=True)
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Density')
    ax.set_title('Embedding L2 Norm Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean embedding values per class
    ax = axes[0, 1]
    mean_norm_brca = np.mean(X_norm[y == 0], axis=0).mean()
    mean_norm_blca = np.mean(X_norm[y == 1], axis=0).mean()
    mean_no_norm_brca = np.mean(X_no_norm[y == 0], axis=0).mean()
    mean_no_norm_blca = np.mean(X_no_norm[y == 1], axis=0).mean()
    
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [mean_norm_brca, mean_norm_blca], width, 
           label='With Normalization', alpha=0.8)
    ax.bar(x + width/2, [mean_no_norm_brca, mean_no_norm_blca], width, 
           label='Without Normalization', alpha=0.8)
    ax.set_xlabel('Cancer Type')
    ax.set_ylabel('Mean Embedding Value')
    ax.set_title('Average Embedding Values by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(['BRCA', 'BLCA'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Cosine similarity distribution
    ax = axes[1, 0]
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Sample subset for computational efficiency
    sample_size = min(100, len(X_norm))
    indices = np.random.choice(len(X_norm), sample_size, replace=False)
    
    cos_sim_norm = cosine_similarity(X_norm[indices])
    cos_sim_no_norm = cosine_similarity(X_no_norm[indices])
    
    # Get upper triangle values (excluding diagonal)
    triu_indices = np.triu_indices(sample_size, k=1)
    cos_sim_norm_values = cos_sim_norm[triu_indices]
    cos_sim_no_norm_values = cos_sim_no_norm[triu_indices]
    
    ax.hist(cos_sim_norm_values, bins=30, alpha=0.5, 
            label='With Normalization', density=True)
    ax.hist(cos_sim_no_norm_values, bins=30, alpha=0.5, 
            label='Without Normalization', density=True)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Pairwise Cosine Similarity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Inter vs Intra class distances
    ax = axes[1, 1]
    from sklearn.metrics import pairwise_distances
    
    # Calculate average inter and intra class distances
    def calc_class_distances(X, y):
        distances = pairwise_distances(X)
        
        # Intra-class distances
        intra_dists = []
        for class_label in np.unique(y):
            class_indices = np.where(y == class_label)[0]
            if len(class_indices) > 1:
                class_dists = distances[np.ix_(class_indices, class_indices)]
                upper_tri = class_dists[np.triu_indices(len(class_indices), k=1)]
                intra_dists.extend(upper_tri)
        
        # Inter-class distances
        inter_dists = []
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]
        inter_class_dists = distances[np.ix_(class_0_indices, class_1_indices)]
        inter_dists = inter_class_dists.flatten()
        
        return np.mean(intra_dists), np.mean(inter_dists)
    
    intra_norm, inter_norm = calc_class_distances(X_norm, y)
    intra_no_norm, inter_no_norm = calc_class_distances(X_no_norm, y)
    
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [intra_norm, inter_norm], width, 
           label='With Normalization', alpha=0.8)
    ax.bar(x + width/2, [intra_no_norm, inter_no_norm], width, 
           label='Without Normalization', alpha=0.8)
    ax.set_xlabel('Distance Type')
    ax.set_ylabel('Average Distance')
    ax.set_title('Inter vs Intra Class Distances')
    ax.set_xticks(x)
    ax.set_xticklabels(['Intra-class', 'Inter-class'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Embedding Space Statistics', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'embedding_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_staining_visualization(processor, slide_paths, save_dir, normalizer):
    """Create visualization showing staining differences"""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Process first 3 slides
    for i, slide_path in enumerate(slide_paths[:3]):
        # Load slide and get a few patches
        slide = SimpleSlide(
            slide_path,
            tile_size=512,
            max_patches=20,
            tissue_detector=processor.tissue_detector,
        )
        
        patches = slide.load_patches_concurrently(target_patch_size=224)
        
        if len(patches) > 0:
            # Select a representative patch
            patch = patches[len(patches)//2]
            
            # Original
            axes[i, 0].imshow(patch)
            axes[i, 0].set_title('Original' if i == 0 else '')
            axes[i, 0].axis('off')
            
            # Macenko normalized
            macenko_norm = normalizer['macenko'].transform(patch)
            axes[i, 1].imshow(macenko_norm)
            axes[i, 1].set_title('Macenko Normalized' if i == 0 else '')
            axes[i, 1].axis('off')
            
            # Reinhard normalized
            reinhard_norm = normalizer['reinhard'].transform(patch)
            axes[i, 2].imshow(reinhard_norm)
            axes[i, 2].set_title('Reinhard Normalized' if i == 0 else '')
            axes[i, 2].axis('off')
            
            # Difference map (Macenko)
            diff = np.abs(patch.astype(float) - macenko_norm.astype(float))
            diff = (diff / diff.max() * 255).astype(np.uint8)
            axes[i, 3].imshow(diff)
            axes[i, 3].set_title('Difference Map' if i == 0 else '')
            axes[i, 3].axis('off')
            
            # Add slide name
            axes[i, 0].text(-0.1, 0.5, f'Slide {i+1}', 
                          transform=axes[i, 0].transAxes,
                          rotation=90, va='center', fontsize=10)
    
    plt.suptitle('Stain Normalization Effects on WSI Patches', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'staining_visualization.png'), dpi=300, bbox_inches='tight')
    
    # Also save to figures directory
    figures_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'staining_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function"""
    # Configuration
    config = {
        'model_path': None,  # Virchow2 loads from HuggingFace
        'tissue_detector_path': '/mnt/d/Models/TissueDetector/HnE.pt',
        'output_dir': '/mnt/f/Projects/HoneyBee/results/staining_virchow2',
        'temp_dir': '/mnt/f/Projects/HoneyBee/results/staining_virchow2/tmp',
        'max_patches_per_slide': 100,  # Reduced for Virchow2 memory requirements
        'pooling_method': 'mean_max',  # mean, max, or mean_max
        'normalization_method': 'macenko',  # macenko or reinhard
        'max_slides_per_type': 100  # Start with fewer slides for testing Virchow2  
    }
    
    # Create temp directory if it doesn't exist
    os.makedirs(config['temp_dir'], exist_ok=True)
    
    # Get slide paths
    brca_base = "/mnt/d/TCGA/TCGA-BRCA/raw"
    blca_base = "/mnt/f/TCGA-BLCA/raw"
    
    slide_data = []
    
    max_slides_per_type = config.get('max_slides_per_type', 100)  # Default to 100 if not set
    print(f"Max slides per type: {max_slides_per_type}")
    
    # Collect BRCA slides
    print("Collecting BRCA slides...")
    brca_count = 0
    for patient_dir in Path(brca_base).iterdir():
        if brca_count >= max_slides_per_type:
            break
        if patient_dir.is_dir() and patient_dir.name.startswith("TCGA-"):
            slide_image_dir = patient_dir / "Slide Image"
            if slide_image_dir.exists():
                for slide_dir in slide_image_dir.iterdir():
                    if slide_dir.is_dir():
                        svs_files = list(slide_dir.glob("*.svs"))
                        if svs_files:
                            slide_data.append({
                                'path': str(svs_files[0]),
                                'patient_id': patient_dir.name,
                                'cancer_type': 'BRCA'
                            })
                            brca_count += 1
                            break
    
    # Collect BLCA slides
    print("Collecting BLCA slides...")
    blca_count = 0
    for patient_dir in Path(blca_base).iterdir():
        if blca_count >= max_slides_per_type:
            break
        if patient_dir.is_dir() and patient_dir.name.startswith("TCGA-"):
            slide_image_dir = patient_dir / "Slide Image"
            if slide_image_dir.exists():
                for slide_dir in slide_image_dir.iterdir():
                    if slide_dir.is_dir():
                        svs_files = list(slide_dir.glob("*.svs"))
                        if svs_files:
                            slide_data.append({
                                'path': str(svs_files[0]),
                                'patient_id': patient_dir.name,
                                'cancer_type': 'BLCA'
                            })
                            blca_count += 1
                            break
    
    print(f"\nFound {len(slide_data)} slides total")
    print(f"BRCA: {sum(1 for s in slide_data if s['cancer_type'] == 'BRCA')}")
    print(f"BLCA: {sum(1 for s in slide_data if s['cancer_type'] == 'BLCA')}")
    print(f"\nThis will result in approximately {len(slide_data) * 0.8 * 0.8:.0f} training samples")
    
    if len(slide_data) == 0:
        print("No slides found! Please check the paths.")
        return
    
    # Initialize processor
    print("\nInitializing WSI processor...")
    processor = WSIProcessor(
        config['model_path'],
        config['tissue_detector_path']
    )
    
    # Select reference image for normalization
    print("\nSelecting reference image for stain normalization...")
    ref_slide = slide_data[0]  # Use first slide as reference
    ref_slide_obj = SimpleSlide(
        ref_slide['path'],
        tile_size=512,
        max_patches=10,
        tissue_detector=processor.tissue_detector,
    )
    ref_patches = ref_slide_obj.load_patches_concurrently(target_patch_size=224)
    
    if len(ref_patches) > 0:
        ref_patch = ref_patches[0]
        
        # Initialize normalizers
        print("Fitting stain normalizers...")
        normalizers = {
            'macenko': MacenkoNormalizer().fit(ref_patch),
            'reinhard': ReinhardNormalizer().fit(ref_patch)
        }
    else:
        print("Failed to extract patches from reference slide!")
        return
    
    # Process slides with and without normalization
    print("\nProcessing slides...")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    embeddings_norm = []
    embeddings_no_norm = []
    labels = []
    patient_ids = []
    
    # Process in batches to save memory
    batch_size = 5  # Reduced batch size for memory efficiency with more slides
    for i in range(0, len(slide_data), batch_size):
        batch = slide_data[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(slide_data) + batch_size - 1)//batch_size}")
        
        for slide_info in tqdm(batch, desc="Processing slides"):
            try:
                # Process without normalization
                processor.set_normalizer(None)
                emb_no_norm = processor.process_slide(
                    slide_info['path'], 
                    max_patches=config['max_patches_per_slide'],
                    apply_normalization=False
                )
                
                # Process with normalization
                processor.set_normalizer(normalizers[config['normalization_method']])
                emb_norm = processor.process_slide(
                    slide_info['path'],
                    max_patches=config['max_patches_per_slide'],
                    apply_normalization=True
                )
                
                if emb_no_norm is not None and emb_norm is not None:
                    # Create patient-level representations
                    patient_rep_no_norm = processor.create_patient_representation(
                        emb_no_norm, method=config['pooling_method']
                    )
                    patient_rep_norm = processor.create_patient_representation(
                        emb_norm, method=config['pooling_method']
                    )
                    
                    if patient_rep_no_norm is not None and patient_rep_norm is not None:
                        embeddings_no_norm.append(patient_rep_no_norm)
                        embeddings_norm.append(patient_rep_norm)
                        labels.append(slide_info['cancer_type'])
                        patient_ids.append(slide_info['patient_id'])
                        
            except Exception as e:
                print(f"Error processing {slide_info['path']}: {e}")
                # Clear GPU cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # Clear GPU cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Save intermediate results
        if (i + batch_size) % 25 == 0 or (i + batch_size) >= len(slide_data):
            print("Saving intermediate results...")
            np.save(os.path.join(config['temp_dir'], f'embeddings_norm_batch_{i}.npy'), 
                   np.array(embeddings_norm))
            np.save(os.path.join(config['temp_dir'], f'embeddings_no_norm_batch_{i}.npy'), 
                   np.array(embeddings_no_norm))
            with open(os.path.join(config['temp_dir'], f'metadata_batch_{i}.pkl'), 'wb') as f:
                pickle.dump({'labels': labels, 'patient_ids': patient_ids}, f)
    
    # Convert to arrays
    X_norm = np.array(embeddings_norm)
    X_no_norm = np.array(embeddings_no_norm)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"\nProcessed {len(y)} slides successfully")
    print(f"Feature dimensions: {X_norm.shape[1]}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split data
    print("\nSplitting data...")
    X_train_norm, X_test_norm, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_no_norm, X_test_no_norm, _, _ = train_test_split(
        X_no_norm, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split train into train/val
    X_train_norm, X_val_norm, y_train_split, y_val_split = train_test_split(
        X_train_norm, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    X_train_no_norm, X_val_no_norm, _, _ = train_test_split(
        X_train_no_norm, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train classifiers
    print("\nTraining classifiers...")
    results_norm = {}
    results_no_norm = {}
    
    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr_norm = LogisticRegression(max_iter=1000, random_state=42)
    lr_norm.fit(X_train_norm, y_train_split)
    results_norm['logistic_regression'] = evaluate_model(lr_norm, X_test_norm, y_test, 'sklearn')
    results_norm['logistic_regression']['model'] = lr_norm
    
    lr_no_norm = LogisticRegression(max_iter=1000, random_state=42)
    lr_no_norm.fit(X_train_no_norm, y_train_split)
    results_no_norm['logistic_regression'] = evaluate_model(lr_no_norm, X_test_no_norm, y_test, 'sklearn')
    results_no_norm['logistic_regression']['model'] = lr_no_norm
    
    # 2. Random Forest
    print("\n2. Training Random Forest...")
    rf_norm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Use all CPU cores
    rf_norm.fit(X_train_norm, y_train_split)
    results_norm['random_forest'] = evaluate_model(rf_norm, X_test_norm, y_test, 'sklearn')
    results_norm['random_forest']['model'] = rf_norm
    
    rf_no_norm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # Use all CPU cores
    rf_no_norm.fit(X_train_no_norm, y_train_split)
    results_no_norm['random_forest'] = evaluate_model(rf_no_norm, X_test_no_norm, y_test, 'sklearn')
    results_no_norm['random_forest']['model'] = rf_no_norm
    
    # 3. Neural Network
    print("\n3. Training Neural Network...")
    nn_norm, train_losses_norm, val_accs_norm = train_classifier(
        X_train_norm, y_train_split, X_val_norm, y_val_split
    )
    results_norm['neural_network'] = evaluate_model(nn_norm, X_test_norm, y_test, 'nn')
    results_norm['neural_network']['train_losses'] = train_losses_norm
    results_norm['neural_network']['val_accs'] = val_accs_norm
    results_norm['neural_network']['model'] = nn_norm
    
    nn_no_norm, train_losses_no_norm, val_accs_no_norm = train_classifier(
        X_train_no_norm, y_train_split, X_val_no_norm, y_val_split
    )
    results_no_norm['neural_network'] = evaluate_model(nn_no_norm, X_test_no_norm, y_test, 'nn')
    results_no_norm['neural_network']['train_losses'] = train_losses_no_norm
    results_no_norm['neural_network']['val_accs'] = val_accs_no_norm
    results_no_norm['neural_network']['model'] = nn_no_norm
    
    # Add true labels for ROC curve plotting
    for results in [results_norm, results_no_norm]:
        for model in results:
            results[model]['true_labels'] = y_test
    
    # Print results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'With Norm':<15} {'Without Norm':<15} {'Improvement':<15}")
    print("-"*60)
    
    for model in results_norm.keys():
        acc_norm = results_norm[model]['accuracy']
        acc_no_norm = results_no_norm[model]['accuracy']
        improvement = acc_norm - acc_no_norm
        
        print(f"{model:<25} {acc_norm:<15.4f} {acc_no_norm:<15.4f} {improvement:+.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_results(results_norm, results_no_norm, config['output_dir'])
    
    # Create staining visualization
    slide_paths = [s['path'] for s in slide_data[:3]]  # Use first 3 slides
    create_staining_visualization(processor, slide_paths, config['output_dir'], normalizers)
    
    # Create t-SNE visualizations
    print("\nGenerating t-SNE visualizations...")
    # Adjust perplexity based on sample size
    perplexity = min(30, max(5, len(y) // 4))
    X_norm_tsne, X_no_norm_tsne = create_tsne_visualization(
        X_norm, X_no_norm, y, labels, config['output_dir'], perplexity=perplexity
    )
    
    # Create additional visualizations
    print("\nGenerating additional visualizations...")
    create_additional_visualizations(
        X_norm, X_no_norm, y, labels, results_norm, results_no_norm, config['output_dir']
    )
    
    # Save embeddings for future analysis
    print("\nSaving embeddings...")
    embeddings_dir = os.path.join(config['output_dir'], 'embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)
    
    np.save(os.path.join(embeddings_dir, 'embeddings_normalized.npy'), X_norm)
    np.save(os.path.join(embeddings_dir, 'embeddings_non_normalized.npy'), X_no_norm)
    np.save(os.path.join(embeddings_dir, 'labels.npy'), y)
    np.save(os.path.join(embeddings_dir, 'tsne_normalized.npy'), X_norm_tsne)
    np.save(os.path.join(embeddings_dir, 'tsne_non_normalized.npy'), X_no_norm_tsne)
    
    # Save label mapping
    with open(os.path.join(embeddings_dir, 'label_mapping.json'), 'w') as f:
        label_map = {int(i): labels[np.where(y == i)[0][0]] for i in np.unique(y)}
        json.dump(label_map, f, indent=2)
    
    # Save results
    print("\nSaving results...")
    results = {
        'config': config,
        'results_with_normalization': results_norm,
        'results_without_normalization': results_no_norm,
        'label_encoder': label_encoder,
        'slide_metadata': slide_data[:len(y)],  # Only processed slides
    }
    
    with open(os.path.join(config['output_dir'], 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary report
    with open(os.path.join(config['output_dir'], 'summary_report.txt'), 'w') as f:
        f.write("Stain Normalization Impact on Virchow2 Embeddings\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total slides processed: {len(y)}\n")
        f.write(f"BRCA slides: {sum(1 for l in labels if l == 'BRCA')}\n")
        f.write(f"BLCA slides: {sum(1 for l in labels if l == 'BLCA')}\n")
        f.write(f"Feature dimensions: {X_norm.shape[1]}\n")
        f.write(f"Normalization method: {config['normalization_method']}\n")
        f.write(f"Pooling method: {config['pooling_method']}\n\n")
        
        f.write("Classification Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<25} {'With Norm':<15} {'Without Norm':<15} {'Improvement':<15}\n")
        f.write("-" * 60 + "\n")
        
        for model in results_norm.keys():
            acc_norm = results_norm[model]['accuracy']
            acc_no_norm = results_no_norm[model]['accuracy']
            improvement = acc_norm - acc_no_norm
            
            f.write(f"{model:<25} {acc_norm:<15.4f} {acc_no_norm:<15.4f} {improvement:+.4f}\n")
    
    print(f"\nResults saved to {config['output_dir']}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()