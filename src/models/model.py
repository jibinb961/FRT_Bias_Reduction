import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import os

class FaceRecognitionModel(nn.Module):
    """
    Face recognition model based on ResNet-18 architecture.
    Modified for multi-task learning to predict gender, age, and race.
    """
    def __init__(self, num_genders=2, num_age_groups=9, num_races=7, pretrained=True):
        super(FaceRecognitionModel, self).__init__()
        
        # Load pre-trained ResNet-18 model
        self.base_model = models.resnet18(pretrained=pretrained)
        
        # Get the number of features in the last layer
        num_features = self.base_model.fc.in_features
        
        # Remove the final fully connected layer
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # Add task-specific heads
        self.gender_classifier = nn.Linear(num_features, num_genders)
        self.age_classifier = nn.Linear(num_features, num_age_groups)
        self.race_classifier = nn.Linear(num_features, num_races)
        
        # Define normalization transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (gender_logits, age_logits, race_logits)
        """
        # Pass through the base model
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Get predictions from each task-specific head
        gender_logits = self.gender_classifier(x)
        age_logits = self.age_classifier(x)
        race_logits = self.race_classifier(x)
        
        return gender_logits, age_logits, race_logits
    
    def predict(self, image):
        """
        Make a prediction on a single image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dictionary with gender, age, and race predictions
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transformation
        x = self.transform(image)
        x = x.unsqueeze(0)  # Add batch dimension
        
        # Move to the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Make prediction
        with torch.no_grad():
            gender_logits, age_logits, race_logits = self.forward(x)
            
            gender_probs = torch.softmax(gender_logits, dim=1)
            age_probs = torch.softmax(age_logits, dim=1)
            race_probs = torch.softmax(race_logits, dim=1)
            
            gender_pred = torch.argmax(gender_probs, dim=1).item()
            age_pred = torch.argmax(age_probs, dim=1).item()
            race_pred = torch.argmax(race_probs, dim=1).item()
        
        # Convert indices to human-readable labels (placeholder mapping)
        gender_map = {0: "Female", 1: "Male"}
        age_map = {
            0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 
            4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "70+"
        }
        race_map = {
            0: "White", 1: "Black", 2: "Latino/Hispanic", 3: "East Asian", 
            4: "Southeast Asian", 5: "Indian", 6: "Middle Eastern"
        }
        
        return {
            "gender": gender_map[gender_pred],
            "age": age_map[age_pred],
            "race": race_map[race_pred],
            "confidence_scores": {
                "gender": gender_probs[0][gender_pred].item(),
                "age": age_probs[0][age_pred].item(),
                "race": race_probs[0][race_pred].item()
            }
        }
    
    def save(self, path):
        """Save the model to a file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_genders': self.gender_classifier.out_features,
            'num_age_groups': self.age_classifier.out_features,
            'num_races': self.race_classifier.out_features
        }, path)
    
    @classmethod
    def load(cls, path, map_location=None):
        """Load a model from a file"""
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(
            num_genders=checkpoint['num_genders'],
            num_age_groups=checkpoint['num_age_groups'],
            num_races=checkpoint['num_races'],
            pretrained=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model


def get_embedding(model, image):
    """
    Extract the feature embedding from the base model for an image
    
    Args:
        model: FaceRecognitionModel instance
        image: PIL Image or numpy array
        
    Returns:
        Feature embedding (numpy array)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply transformation
    x = model.transform(image)
    x = x.unsqueeze(0)  # Add batch dimension
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    x = x.to(device)
    
    # Extract features
    with torch.no_grad():
        features = model.base_model(x)
        features = features.view(features.size(0), -1)
    
    return features.cpu().numpy() 