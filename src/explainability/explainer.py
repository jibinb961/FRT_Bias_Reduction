import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import io

class ShapExplainer:
    """
    SHAP-based explainer for facial recognition models
    """
    def __init__(self, model, background_samples=None, device=None):
        """
        Initialize the SHAP explainer
        
        Args:
            model: Facial recognition model to explain
            background_samples: Background samples for DeepExplainer
            device: Device to use for computation (cpu or cuda)
        """
        self.model = model
        self.background_samples = background_samples
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Move model to the specified device
        self.model.to(self.device)
        
        # Initialize SHAP explainer
        self.explainer = None
        
    def initialize_explainer(self, background_samples=None):
        """
        Initialize the SHAP DeepExplainer with background samples
        
        Args:
            background_samples: Background samples for DeepExplainer. If None, uses self.background_samples
        """
        if background_samples is not None:
            self.background_samples = background_samples
            
        if self.background_samples is None:
            raise ValueError("Background samples are required to initialize the explainer")
            
        # Move background samples to the specified device
        self.background_samples = self.background_samples.to(self.device)
        
        # Wrap model for SHAP
        def model_wrapper(x):
            # We need to convert the output to the format SHAP expects
            gender_logits, age_logits, race_logits = self.model(x)
            return torch.cat([gender_logits, age_logits, race_logits], dim=1)
            
        # Initialize SHAP explainer
        self.explainer = shap.DeepExplainer(model_wrapper, self.background_samples)
        
    def explain_image(self, image, task='gender', return_attributions=False):
        """
        Generate SHAP explanations for an image
        
        Args:
            image: PIL Image or numpy array
            task: Task to explain ('gender', 'age', or 'race')
            return_attributions: Whether to return the raw SHAP values
            
        Returns:
            tuple: (visualization image, raw attributions if return_attributions=True)
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
            
        # Ensure image is in the right format
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a PIL Image or numpy array")
            
        # Apply model's transform
        x = self.model.transform(pil_image)
        x = x.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(x)
        
        # Extract SHAP values for the specified task
        task_idx = {'gender': 0, 'age': 1, 'race': 2}[task]
        
        # For multi-class outputs, we need to determine which class's explanation to show
        with torch.no_grad():
            gender_logits, age_logits, race_logits = self.model(x)
            
        logits_map = {'gender': gender_logits, 'age': age_logits, 'race': race_logits}
        task_logits = logits_map[task]
        
        # Get the predicted class
        pred_class = torch.argmax(task_logits, dim=1).item()
        
        # Get the task-specific SHAP values
        if task == 'gender':
            # Binary classification, typically 2 classes (0 and 1)
            # We get the explanation for the predicted class
            task_shap_values = shap_values[pred_class]
        elif task == 'age':
            # Multi-class, age groups
            # Offset by the number of gender classes (2)
            age_offset = 2
            task_shap_values = shap_values[age_offset + pred_class]
        elif task == 'race':
            # Multi-class, race categories
            # Offset by the number of gender (2) and age classes (9)
            race_offset = 2 + 9
            task_shap_values = shap_values[race_offset + pred_class]
            
        # Convert to numpy and move to CPU if needed
        task_shap_values = task_shap_values.cpu().numpy()
        
        # Aggregate across color channels for visualization
        attributions = np.sum(task_shap_values, axis=0)
        
        # Create visualization
        visualization = self._create_visualization(pil_image, attributions)
        
        if return_attributions:
            return visualization, attributions
        else:
            return visualization
    
    def _create_visualization(self, image, attributions):
        """
        Create a visualization of the SHAP attributions overlaid on the original image
        
        Args:
            image: PIL Image
            attributions: SHAP attributions
            
        Returns:
            PIL Image: Visualization of SHAP attributions
        """
        # Convert PIL image to numpy
        img_array = np.array(image)
        
        # Normalize attributions to [-1, 1]
        attributions = attributions / np.max(np.abs(attributions))
        
        # Create a heatmap
        heatmap = np.zeros((attributions.shape[0], attributions.shape[1], 3), dtype=np.float32)
        
        # Red for positive attributions, blue for negative
        heatmap[attributions > 0, 0] = attributions[attributions > 0]  # Red channel
        heatmap[attributions < 0, 2] = -attributions[attributions < 0]  # Blue channel
        
        # Scale to [0, 255]
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Resize to match the original image size
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Create the overlay
        alpha = 0.5
        overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
        
        # Convert back to PIL Image
        return Image.fromarray(overlay)
    
    def save_explanation(self, image, task='gender', output_dir='explanations', filename=None):
        """
        Generate and save SHAP explanation for an image
        
        Args:
            image: PIL Image or numpy array
            task: Task to explain ('gender', 'age', or 'race')
            output_dir: Directory to save the explanation
            filename: Filename for the saved explanation
            
        Returns:
            str: Path to the saved explanation image
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate explanation
        visualization = self.explain_image(image, task)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"shap_explanation_{task}_{np.random.randint(10000)}.png"
        
        # Ensure the filename has an extension
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename += '.png'
        
        # Save the visualization
        output_path = os.path.join(output_dir, filename)
        visualization.save(output_path)
        
        return output_path
    
    def plot_multiple_explanations(self, image, tasks=None, figsize=(15, 5)):
        """
        Plot SHAP explanations for multiple tasks
        
        Args:
            image: PIL Image or numpy array
            tasks: List of tasks to explain (default: ['gender', 'age', 'race'])
            figsize: Figure size for the plot
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if tasks is None:
            tasks = ['gender', 'age', 'race']
            
        # Generate explanations for each task
        explanations = [self.explain_image(image, task) for task in tasks]
        
        # Create figure
        fig, axes = plt.subplots(1, len(tasks) + 1, figsize=figsize)
        
        # Show original image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError("Image must be a PIL Image or numpy array")
            
        axes[0].imshow(np.array(pil_image))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Show explanations
        for i, (task, explanation) in enumerate(zip(tasks, explanations)):
            axes[i + 1].imshow(np.array(explanation))
            axes[i + 1].set_title(f"{task.capitalize()} Explanation")
            axes[i + 1].axis('off')
            
        plt.tight_layout()
        
        return fig
    
    def get_explanation_as_bytes(self, image, task='gender', format='png'):
        """
        Get a SHAP explanation as bytes
        
        Args:
            image: PIL Image or numpy array
            task: Task to explain ('gender', 'age', or 'race')
            format: Image format ('png', 'jpg', etc.)
            
        Returns:
            bytes: Image bytes
        """
        # Generate explanation
        visualization = self.explain_image(image, task)
        
        # Convert to bytes
        buffer = io.BytesIO()
        visualization.save(buffer, format=format)
        
        return buffer.getvalue() 