#!/usr/bin/env python
# Script to download and preprocess the FairFace dataset

import os
import requests
import zipfile
import pandas as pd
from tqdm import tqdm
import argparse
import shutil

def download_file(url, destination):
    """
    Download a file from a URL to a destination path with progress bar
    """
    if os.path.exists(destination):
        print(f"File already exists at {destination}. Skipping download.")
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}") as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))

def extract_zip(zip_path, extract_path):
    """
    Extract a zip file to a destination path
    """
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        print(f"Extracting {total_files} files from {zip_path}...")
        zip_ref.extractall(extract_path)
    
    print(f"Extraction complete. Files extracted to {extract_path}")

def setup_fairface_dataset(data_dir="data", val_ratio=0.2, test_ratio=0.1):
    """
    Download, extract, and organize the FairFace dataset
    """
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    fairface_dir = os.path.join(data_dir, "fairface")
    os.makedirs(fairface_dir, exist_ok=True)
    
    # URLs for the FairFace dataset
    fairface_train_val_url = "https://drive.google.com/uc?id=1g7qNOZz9wC7OfOhcPqH1EZ5bk1UFGmlL"
    fairface_test_url = "https://drive.google.com/uc?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH"
    fairface_train_val_labels_url = "https://drive.google.com/uc?id=1wOdja-ezstMEp81tX1a-EYkFebev4h7D"
    fairface_test_labels_url = "https://drive.google.com/uc?id=1w5i9UL0_1FG7SxAExPCRDhWMuRRBjUfZ"
    
    # Download files (these direct download links may not work, Google Drive has limits)
    # In a real implementation, you might need to use gdown or a similar tool
    print("Note: This script uses direct Google Drive links which may not work due to Google's restrictions.")
    print("If downloads fail, please download the FairFace dataset manually from https://github.com/joojs/fairface")
    
    try:
        train_val_zip = os.path.join(fairface_dir, "fairface-train-val.zip")
        test_zip = os.path.join(fairface_dir, "fairface-test.zip")
        train_val_labels = os.path.join(fairface_dir, "fairface-train-val-labels.csv")
        test_labels = os.path.join(fairface_dir, "fairface-test-labels.csv")
        
        # Download files
        print("Attempting to download dataset files...")
        download_file(fairface_train_val_url, train_val_zip)
        download_file(fairface_test_url, test_zip)
        download_file(fairface_train_val_labels_url, train_val_labels)
        download_file(fairface_test_labels_url, test_labels)
        
        # Extract zip files
        extract_zip(train_val_zip, os.path.join(fairface_dir, "train-val"))
        extract_zip(test_zip, os.path.join(fairface_dir, "test"))
        
        # Organize dataset into train, val, test splits
        organize_fairface_dataset(fairface_dir, val_ratio, test_ratio)
        
        print("FairFace dataset setup complete!")
        
    except Exception as e:
        print(f"Error downloading FairFace dataset: {e}")
        print("Please download the dataset manually from https://github.com/joojs/fairface")

def organize_fairface_dataset(fairface_dir, val_ratio=0.2, test_ratio=0.1):
    """
    Organize the FairFace dataset into train, val, test splits
    """
    # Read labels
    train_val_labels_path = os.path.join(fairface_dir, "fairface-train-val-labels.csv")
    test_labels_path = os.path.join(fairface_dir, "fairface-test-labels.csv")
    
    if not os.path.exists(train_val_labels_path) or not os.path.exists(test_labels_path):
        print("Label files not found. Dataset organization skipped.")
        return
    
    train_val_df = pd.read_csv(train_val_labels_path)
    test_df = pd.read_csv(test_labels_path)
    
    # Create train/val split from train_val data
    val_size = int(len(train_val_df) * val_ratio / (1 - test_ratio))
    train_df = train_val_df.iloc[val_size:]
    val_df = train_val_df.iloc[:val_size]
    
    # Create output directories
    output_dirs = {
        "train": os.path.join(fairface_dir, "train"),
        "val": os.path.join(fairface_dir, "val"),
        "test": os.path.join(fairface_dir, "test")
    }
    
    for directory in output_dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # Save labels
    train_df.to_csv(os.path.join(fairface_dir, "train_labels.csv"), index=False)
    val_df.to_csv(os.path.join(fairface_dir, "val_labels.csv"), index=False)
    test_df.to_csv(os.path.join(fairface_dir, "test_labels.csv"), index=False)
    
    print("Dataset organization complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and set up the FairFace dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store the dataset")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    
    args = parser.parse_args()
    
    print("Starting FairFace dataset setup...")
    setup_fairface_dataset(args.data_dir, args.val_ratio, args.test_ratio) 