import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class FairFaceDataset(Dataset):
    """
    Dataset class for the FairFace dataset
    """
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Map labels to indices
        self.gender_to_idx = {"Female": 0, "Male": 1}
        
        self.age_to_idx = {
            "0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3,
            "30-39": 4, "40-49": 5, "50-59": 6, "60-69": 7, "70+": 8
        }
        
        self.race_to_idx = {
            "White": 0, "Black": 1, "Latino_Hispanic": 2, "East Asian": 3,
            "Southeast Asian": 4, "Indian": 5, "Middle Eastern": 6
        }
        
    def __len__(self):
        return len(self.face_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image file path
        img_name = os.path.join(self.img_dir, self.face_frame.iloc[idx, 0])
        
        # Read image
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a placeholder if image loading fails
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        gender = self.face_frame.iloc[idx]['gender']
        age = self.face_frame.iloc[idx]['age']
        race = self.face_frame.iloc[idx]['race']
        
        # Convert labels to indices
        gender_idx = self.gender_to_idx[gender]
        age_idx = self.age_to_idx[age]
        race_idx = self.race_to_idx[race]
        
        return {
            'image': image,
            'gender': gender_idx,
            'age': age_idx,
            'race': race_idx,
            'file_name': self.face_frame.iloc[idx, 0]
        }


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create dataloaders for training, validation, and testing
    
    Args:
        data_dir (string): Path to the FairFace dataset directory
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        Dictionary of dataloaders for train, val, and test sets
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_csv = os.path.join(data_dir, "train_labels.csv")
    val_csv = os.path.join(data_dir, "val_labels.csv")
    test_csv = os.path.join(data_dir, "test_labels.csv")
    
    train_img_dir = os.path.join(data_dir, "train-val")
    test_img_dir = os.path.join(data_dir, "test")
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(f"CSV files not found in {data_dir}. Please run the download_dataset.py script first.")
    
    train_dataset = FairFaceDataset(train_csv, train_img_dir, train_transform)
    val_dataset = FairFaceDataset(val_csv, train_img_dir, val_test_transform)
    test_dataset = FairFaceDataset(test_csv, test_img_dir, val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 