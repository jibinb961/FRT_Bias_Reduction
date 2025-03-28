import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset
import torch
from torch.utils.data import WeightedRandomSampler

class BiasMitigator:
    """
    Bias mitigator class using AI Fairness 360 for mitigating bias in facial recognition models
    """
    def __init__(self):
        """
        Initialize the bias mitigator
        """
        self.mitigation_techniques = {
            "Reweighting": self.apply_reweighting,
            "Resampling": self.generate_resampling_weights,
            "DisparateImpactRemover": self.apply_disparate_impact_remover
        }
    
    def get_mitigation_techniques(self):
        """
        Get available mitigation techniques
        
        Returns:
            list: List of available mitigation techniques
        """
        return list(self.mitigation_techniques.keys())
    
    def mitigate_bias(self, dataset, technique, protected_attribute, privileged_groups, unprivileged_groups):
        """
        Mitigate bias in a dataset using the specified technique
        
        Args:
            dataset (BinaryLabelDataset): Dataset for bias mitigation
            technique (str): Mitigation technique to use
            protected_attribute (str): Protected attribute to mitigate bias for
            privileged_groups (list): List of privileged groups
            unprivileged_groups (list): List of unprivileged groups
            
        Returns:
            tuple: (Transformed dataset, Additional information)
        """
        if technique not in self.mitigation_techniques:
            raise ValueError(f"Unknown mitigation technique: {technique}. Available techniques: {self.get_mitigation_techniques()}")
        
        return self.mitigation_techniques[technique](dataset, protected_attribute, privileged_groups, unprivileged_groups)
    
    def apply_reweighting(self, dataset, protected_attribute, privileged_groups, unprivileged_groups):
        """
        Apply reweighting to mitigate bias
        
        Args:
            dataset (BinaryLabelDataset): Dataset for bias mitigation
            protected_attribute (str): Protected attribute to mitigate bias for
            privileged_groups (list): List of privileged groups
            unprivileged_groups (list): List of unprivileged groups
            
        Returns:
            tuple: (Transformed dataset, Sample weights)
        """
        reweighing = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        
        transformed_dataset = reweighing.fit_transform(dataset)
        
        # Extract weights for return
        weights = transformed_dataset.instance_weights
        
        return transformed_dataset, weights
    
    def generate_resampling_weights(self, dataset, protected_attribute, privileged_groups, unprivileged_groups):
        """
        Generate weights for resampling to mitigate bias
        
        Args:
            dataset (BinaryLabelDataset): Dataset for bias mitigation
            protected_attribute (str): Protected attribute to mitigate bias for
            privileged_groups (list): List of privileged groups
            unprivileged_groups (list): List of unprivileged groups
            
        Returns:
            tuple: (Original dataset, Sample weights for WeightedRandomSampler)
        """
        # First apply reweighting to get the weights
        _, weights = self.apply_reweighting(dataset, protected_attribute, privileged_groups, unprivileged_groups)
        
        # Normalize weights for resampling
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)
        
        return dataset, weights
    
    def apply_disparate_impact_remover(self, dataset, protected_attribute, privileged_groups, unprivileged_groups, repair_level=1.0):
        """
        Apply disparate impact remover to mitigate bias
        
        Args:
            dataset (BinaryLabelDataset): Dataset for bias mitigation
            protected_attribute (str): Protected attribute to mitigate bias for
            privileged_groups (list): List of privileged groups
            unprivileged_groups (list): List of unprivileged groups
            repair_level (float): Repair level (between 0 and 1)
            
        Returns:
            tuple: (Transformed dataset, None)
        """
        di_remover = DisparateImpactRemover(
            repair_level=repair_level,
            sensitive_attribute=protected_attribute
        )
        
        transformed_dataset = di_remover.fit_transform(dataset)
        
        return transformed_dataset, None
    
    def get_pytorch_sampler(self, weights, replacement=True):
        """
        Create a PyTorch sampler from weights
        
        Args:
            weights (np.ndarray): Sample weights
            replacement (bool): Whether to sample with replacement
            
        Returns:
            WeightedRandomSampler: PyTorch sampler for weighted sampling
        """
        weights = torch.from_numpy(weights.astype(np.float32))
        return WeightedRandomSampler(weights, len(weights), replacement=replacement)
    
    def prepare_dataset_for_mitigation(self, predictions, true_labels, protected_values, protected_attribute):
        """
        Prepare a dataset for bias mitigation
        
        Args:
            predictions (np.ndarray): Model predictions
            true_labels (np.ndarray): Ground truth labels
            protected_values (np.ndarray): Values of the protected attribute
            protected_attribute (str): Name of the protected attribute
            
        Returns:
            BinaryLabelDataset: Dataset prepared for bias mitigation
        """
        # Ensure pandas is imported in this scope
        import pandas as pd
        
        # Create a dataframe
        df = pd.DataFrame({
            'prediction': predictions,
            'label': true_labels,
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