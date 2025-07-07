#!/usr/bin/env python3
"""
Example of using RadImageNet models from Hugging Face Hub.
"""

import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms
from PIL import Image


def load_radimagenet_from_hub(model_name="ResNet50.pt", repo_id="Lab-Rasool/RadImageNet"):
    """Simple function to load a RadImageNet model from Hugging Face."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download model from hub
    model_path = hf_hub_download(repo_id=repo_id, filename=model_name)
    
    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    return model, device


def preprocess_medical_image(image_path):
    """Preprocess medical images for RadImageNet models."""
    # Standard RadImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0)


# Example usage
if __name__ == "__main__":
    # Load ResNet50 model
    print("Loading RadImageNet ResNet50 model...")
    model, device = load_radimagenet_from_hub("ResNet50.pt")
    
    # Example inference (uncomment if you have an image)
    # image_tensor = preprocess_medical_image("path/to/medical/image.jpg")
    # image_tensor = image_tensor.to(device)
    # 
    # with torch.no_grad():
    #     features = model(image_tensor)
    #     print(f"Features shape: {features.shape}")
    
    print("Model loaded successfully!")
    print(f"Device: {device}")
    
    # Show model architecture summary
    print(f"\nModel type: {type(model)}")
    if hasattr(model, '__name__'):
        print(f"Model name: {model.__name__}")