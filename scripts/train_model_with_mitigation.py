#!/usr/bin/env python
# Script to train the face recognition model with bias mitigation

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import time
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import FaceRecognitionModel
from src.data import get_dataloaders
from src.bias_mitigation.mitigator import BiasMitigator
from src.bias_mitigation.detector import BiasDetector

def train_model_with_mitigation(
    data_dir, 
    output_dir, 
    mitigation_technique="Reweighting",
    batch_size=32, 
    num_epochs=20, 
    learning_rate=0.001, 
    device=None
):
    """
    Train the face recognition model with bias mitigation
    
    Args:
        data_dir (string): Path to the dataset directory
        output_dir (string): Path to save the trained model
        mitigation_technique (string): Bias mitigation technique to use
        batch_size (int): Batch size for training
        num_epochs (int): Number of epochs to train for
        learning_rate (float): Initial learning rate
        device (string): Device to use for training (cuda or cpu)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataloaders
    try:
        dataloaders = get_dataloaders(data_dir, batch_size=batch_size)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create model
    model = FaceRecognitionModel(pretrained=True)
    model.to(device)
    
    # Initialize bias mitigator and detector
    mitigator = BiasMitigator()
    detector = BiasDetector()
    
    # Define loss functions
    gender_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.CrossEntropyLoss()
    race_criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    bias_metrics_history = []
    
    print(f"Starting training with {mitigation_technique} mitigation for {num_epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            gender_correct = 0
            age_correct = 0
            race_correct = 0
            total = 0
            
            # Collect predictions and labels for bias mitigation
            all_predictions = {'gender': [], 'age': [], 'race': []}
            all_labels = {'gender': [], 'age': [], 'race': []}
            
            # Iterate over data
            pbar = tqdm(dataloaders[phase], desc=f"{phase} batch")
            for batch in pbar:
                images = batch['image'].to(device)
                gender_labels = batch['gender'].to(device)
                age_labels = batch['age'].to(device)
                race_labels = batch['race'].to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    gender_outputs, age_outputs, race_outputs = model(images)
                    
                    # Calculate losses
                    gender_loss = gender_criterion(gender_outputs, gender_labels)
                    age_loss = age_criterion(age_outputs, age_labels)
                    race_loss = race_criterion(race_outputs, race_labels)
                    
                    # Combined loss
                    loss = gender_loss + age_loss + race_loss
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                _, gender_preds = torch.max(gender_outputs, 1)
                _, age_preds = torch.max(age_outputs, 1)
                _, race_preds = torch.max(race_outputs, 1)
                
                gender_correct += torch.sum(gender_preds == gender_labels).item()
                age_correct += torch.sum(age_preds == age_labels).item()
                race_correct += torch.sum(race_preds == race_labels).item()
                total += images.size(0)
                
                # Collect predictions and labels for bias mitigation
                all_predictions['gender'].extend(gender_preds.cpu().numpy())
                all_predictions['age'].extend(age_preds.cpu().numpy())
                all_predictions['race'].extend(race_preds.cpu().numpy())
                all_labels['gender'].extend(gender_labels.cpu().numpy())
                all_labels['age'].extend(age_labels.cpu().numpy())
                all_labels['race'].extend(race_labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'gender_acc': gender_correct / total,
                    'age_acc': age_correct / total,
                    'race_acc': race_correct / total
                })
            
            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            gender_acc = gender_correct / len(dataloaders[phase].dataset)
            age_acc = age_correct / len(dataloaders[phase].dataset)
            race_acc = race_correct / len(dataloaders[phase].dataset)
            
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Gender Acc: {gender_acc:.4f}, Age Acc: {age_acc:.4f}, Race Acc: {race_acc:.4f}")
            
            # Save losses for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
                # Update learning rate scheduler
                scheduler.step(epoch_loss)
                
                # Save best model
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    print(f"Saving best model with validation loss: {best_val_loss:.4f}")
                    model.save(os.path.join(output_dir, 'best_model.pth'))
                
                # Calculate and store bias metrics
                bias_metrics = detector.detect_bias(all_predictions, all_labels)
                bias_metrics_history.append(bias_metrics)
                
                # Print bias metrics
                print("\nBias Metrics:")
                for attr, metrics in bias_metrics.items():
                    if attr == 'gender':
                        print(f"Gender Bias - SPD: {metrics['statistical_parity_difference']:.4f}, DI: {metrics['disparate_impact']:.4f}")
                    elif attr == 'race':
                        print("Racial Bias:")
                        for race, race_metrics in metrics.items():
                            print(f"  {race} - SPD: {race_metrics['statistical_parity_difference']:.4f}, DI: {race_metrics['disparate_impact']:.4f}")
        
        # Save checkpoint every epoch
        model.save(os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
        print('-' * 60)
    
    # Calculate training time
    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    
    # Plot bias metrics over time
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(bias_metrics_history) + 1)
    
    # Plot gender bias
    gender_spd = [metrics['gender']['statistical_parity_difference'] for metrics in bias_metrics_history]
    plt.plot(epochs, gender_spd, label='Gender SPD')
    
    # Plot racial bias (average across all races)
    race_spd = []
    for metrics in bias_metrics_history:
        race_metrics = metrics['race']
        avg_spd = np.mean([m['statistical_parity_difference'] for m in race_metrics.values()])
        race_spd.append(avg_spd)
    plt.plot(epochs, race_spd, label='Average Race SPD')
    
    plt.xlabel('Epochs')
    plt.ylabel('Statistical Parity Difference')
    plt.title('Bias Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'bias_metrics_plot.png'))
    
    # Save final model
    model.save(os.path.join(output_dir, 'final_model.pth'))
    print(f"Model saved to {output_dir}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the face recognition model with bias mitigation")
    parser.add_argument("--data_dir", type=str, default="data/fairface", help="Path to the dataset directory")
    parser.add_argument("--output_dir", type=str, default="models", help="Path to save the trained model")
    parser.add_argument("--mitigation_technique", type=str, default="Reweighting", 
                      choices=["Reweighting", "Resampling", "DisparateImpactRemover"],
                      help="Bias mitigation technique to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (cuda or cpu)")
    
    args = parser.parse_args()
    
    train_model_with_mitigation(
        args.data_dir,
        args.output_dir,
        mitigation_technique=args.mitigation_technique,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    ) 