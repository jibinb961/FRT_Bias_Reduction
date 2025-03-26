#!/usr/bin/env python
# Script to prepare sample data for testing the application

import os
import argparse
import shutil
import requests
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_sample_images(output_dir, num_images=10):
    """
    Download sample face images for testing
    
    Args:
        output_dir (str): Directory to save the images
        num_images (int): Number of images to download
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URLs for sample images (these are example URLs, you might need to replace them)
    # Using a sample of images from publicly available datasets
    urls = [
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00000.png?raw=true",
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00001.png?raw=true",
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00002.png?raw=true",
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00003.png?raw=true",
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00004.png?raw=true",
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00005.png?raw=true",
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00006.png?raw=true",
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00007.png?raw=true",
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00008.png?raw=true",
        "https://github.com/NVlabs/ffhq-dataset/blob/master/thumbnails/00009.png?raw=true",
    ]
    
    # Limit to the requested number of images
    urls = urls[:num_images]
    
    print(f"Downloading {len(urls)} sample images...")
    
    # Download images
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Save image
            img = Image.open(io.BytesIO(response.content))
            filename = os.path.join(output_dir, f"sample_{i:02d}.png")
            img.save(filename)
            print(f"Downloaded image {i+1}/{len(urls)}: {filename}")
            
        except Exception as e:
            print(f"Error downloading image {i+1}: {e}")
    
    print(f"Downloaded {len(urls)} images to {output_dir}")

def create_dummy_model(output_dir):
    """
    Create a dummy model for testing
    
    Args:
        output_dir (str): Directory to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import the model class
        from src.models.model import FaceRecognitionModel
        
        # Create a dummy model
        model = FaceRecognitionModel(pretrained=False)
        
        # Save the model
        model_path = os.path.join(output_dir, "dummy_model.pth")
        model.save(model_path)
        
        print(f"Created dummy model: {model_path}")
        
    except Exception as e:
        print(f"Error creating dummy model: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare sample data for testing the application")
    parser.add_argument("--output_dir", type=str, default="sample_data", help="Directory to save the sample data")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to download")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save the dummy model")
    
    args = parser.parse_args()
    
    # Download sample images
    download_sample_images(args.output_dir, args.num_images)
    
    # Create dummy model
    create_dummy_model(args.model_dir)
    
    print("Sample data preparation complete!")

if __name__ == "__main__":
    main() 