import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

class BiasDetector:
    """
    Bias detector class using AI Fairness 360 for detecting bias in facial recognition models
    """
    def __init__(self, protected_attributes=None):
        """
        Initialize the bias detector
        
        Args:
            protected_attributes (list): List of protected attributes to check for bias (e.g., ['race', 'gender'])
        """
        if protected_attributes is None:
            self.protected_attributes = ['race', 'gender']
        else:
            self.protected_attributes = protected_attributes
        
        # Mappings for protected attributes
        self.gender_map = {
            0: "Female", 
            1: "Male"
        }
        
        self.race_map = {
            0: "White", 
            1: "Black", 
            2: "Latino/Hispanic", 
            3: "East Asian", 
            4: "Southeast Asian", 
            5: "Indian", 
            6: "Middle Eastern"
        }
    
    def prepare_dataset(self, predictions, true_labels, protected_attribute='race'):
        """
        Prepare a dataset for bias detection
        
        Args:
            predictions (list): Model predictions
            true_labels (list): Ground truth labels
            protected_attribute (str): Protected attribute to check for bias
            
        Returns:
            BinaryLabelDataset: Dataset prepared for bias detection
        """
        # Convert to numpy arrays
        preds = np.array(predictions)
        labels = np.array(true_labels)
        
        # Create a dataframe
        if protected_attribute == 'race':
            protected_values = np.array([self.race_map[p] for p in true_labels[protected_attribute]])
        elif protected_attribute == 'gender':
            protected_values = np.array([self.gender_map[p] for p in true_labels[protected_attribute]])
        else:
            protected_values = np.array(true_labels[protected_attribute])
        
        df = pd.DataFrame({
            'prediction': preds,
            'label': labels,
            protected_attribute: protected_values
        })
        
        # Prepare dataset for AIF360
        dataset = BinaryLabelDataset(
            df=df,
            label_names=['label'],
            protected_attribute_names=[protected_attribute],
            favorable_label=1,
            unfavorable_label=0
        )
        
        return dataset
    
    def compute_metrics(self, dataset, privileged_groups, unprivileged_groups):
        """
        Compute fairness metrics for a dataset
        
        Args:
            dataset (BinaryLabelDataset): Dataset for bias detection
            privileged_groups (list): List of privileged groups
            unprivileged_groups (list): List of unprivileged groups
            
        Returns:
            dict: Dictionary of fairness metrics
        """
        metrics = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        disparity = metrics.statistical_parity_difference()
        
        return {
            'statistical_parity_difference': disparity,
            'disparate_impact': metrics.disparate_impact(),
            'group_size': {
                'privileged': metrics.num_instances(privileged=True),
                'unprivileged': metrics.num_instances(privileged=False)
            }
        }
    
    def detect_bias(self, model_predictions, true_labels):
        """
        Detect bias in model predictions
        
        Args:
            model_predictions (dict): Dictionary of model predictions
            true_labels (dict): Dictionary of true labels
            
        Returns:
            dict: Dictionary of bias metrics for different protected attributes
        """
        bias_metrics = {}
        
        for attr in self.protected_attributes:
            if attr == 'gender':
                # For gender bias detection
                privileged_groups = [{'gender': 'Male'}]  # Example: males as privileged group
                unprivileged_groups = [{'gender': 'Female'}]  # Example: females as unprivileged group
                
                # Prepare dataset
                dataset = self.prepare_dataset(
                    model_predictions['gender'],
                    true_labels,
                    protected_attribute='gender'
                )
                
                # Compute metrics
                metrics = self.compute_metrics(dataset, privileged_groups, unprivileged_groups)
                bias_metrics['gender'] = metrics
                
            elif attr == 'race':
                # For racial bias detection
                # Example: white as privileged group
                privileged_groups = [{'race': 'White'}]
                
                # Check bias against each racial group
                racial_bias = {}
                for race_id, race_name in self.race_map.items():
                    if race_name == 'White':
                        continue
                    
                    unprivileged_groups = [{'race': race_name}]
                    
                    # Prepare dataset
                    dataset = self.prepare_dataset(
                        model_predictions['race'],
                        true_labels,
                        protected_attribute='race'
                    )
                    
                    # Compute metrics
                    metrics = self.compute_metrics(dataset, privileged_groups, unprivileged_groups)
                    racial_bias[race_name] = metrics
                
                bias_metrics['race'] = racial_bias
        
        return bias_metrics
    
    def interpret_bias_metrics(self, bias_metrics):
        """
        Interpret bias metrics and provide human-readable explanations
        
        Args:
            bias_metrics (dict): Dictionary of bias metrics
            
        Returns:
            dict: Dictionary of bias interpretations
        """
        interpretations = {}
        
        for attr, metrics in bias_metrics.items():
            if attr == 'gender':
                spd = metrics['statistical_parity_difference']
                di = metrics['disparate_impact']
                
                if abs(spd) < 0.05 and 0.95 <= di <= 1.05:
                    bias_level = "Low"
                    description = "The model shows minimal gender bias."
                elif abs(spd) < 0.1 and 0.9 <= di <= 1.1:
                    bias_level = "Moderate"
                    description = "The model shows some gender bias that may need attention."
                else:
                    bias_level = "High"
                    description = "The model shows significant gender bias that requires mitigation."
                
                interpretations['gender'] = {
                    'bias_level': bias_level,
                    'description': description,
                    'metrics': metrics
                }
                
            elif attr == 'race':
                racial_interpretations = {}
                for race, race_metrics in metrics.items():
                    spd = race_metrics['statistical_parity_difference']
                    di = race_metrics['disparate_impact']
                    
                    if abs(spd) < 0.05 and 0.95 <= di <= 1.05:
                        bias_level = "Low"
                        description = f"The model shows minimal bias against {race} individuals."
                    elif abs(spd) < 0.1 and 0.9 <= di <= 1.1:
                        bias_level = "Moderate"
                        description = f"The model shows some bias against {race} individuals that may need attention."
                    else:
                        bias_level = "High"
                        description = f"The model shows significant bias against {race} individuals that requires mitigation."
                    
                    racial_interpretations[race] = {
                        'bias_level': bias_level,
                        'description': description,
                        'metrics': race_metrics
                    }
                
                interpretations['race'] = racial_interpretations
        
        return interpretations 