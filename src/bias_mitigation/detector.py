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
        
        # Calculate disparate impact safely with protection against division by zero
        try:
            # Get positive outcome rates for unprivileged and privileged groups
            unpriv_positive_rate = metrics.base_rate(privileged=False)
            priv_positive_rate = metrics.base_rate(privileged=True)
            
            # Print rates for debugging
            print(f"Unprivileged positive rate: {unpriv_positive_rate}")
            print(f"Privileged positive rate: {priv_positive_rate}")
            
            # Calculate disparate impact with a safe division
            if priv_positive_rate > 0:
                disparate_impact = unpriv_positive_rate / priv_positive_rate
            else:
                # Handle division by zero
                if unpriv_positive_rate > 0:
                    disparate_impact = float('inf')  # Unprivileged has positive rate but privileged doesn't
                    print("DI calculation: Privileged group has zero positive rate, unprivileged group has positive rate")
                else:
                    disparate_impact = None  # Both are zero, so we can't calculate a meaningful ratio
                    print("DI calculation: Both privileged and unprivileged groups have zero positive rate")
        except Exception as e:
            print(f"Error calculating disparate impact: {e}")
            disparate_impact = None  # Use None to indicate calculation failed
        
        return {
            'statistical_parity_difference': disparity,
            'disparate_impact': disparate_impact,
            'group_size': {
                'privileged': metrics.num_instances(privileged=True),
                'unprivileged': metrics.num_instances(privileged=False)
            },
            'base_rates': {
                'privileged': metrics.base_rate(privileged=True),
                'unprivileged': metrics.base_rate(privileged=False)
            }
        }
    
    def detect_bias(self, model_predictions, true_labels):
        """
        Detect bias in model predictions based on protected attributes
        
        Args:
            model_predictions (dict): Dictionary of model predictions with keys for 'gender', 'race', etc.
            true_labels (dict): Dictionary of true labels with keys for 'gender', 'race', etc.
            
        Returns:
            dict: Dictionary of bias metrics
        """
        # Ensure pandas is imported in this scope
        import pandas as pd
        
        # Debug print for pandas
        print("Pandas version:", pd.__version__)
        
        bias_metrics = {}
        
        # Check if we have gender predictions and labels
        if 'gender' in model_predictions and 'gender' in true_labels:
            # Get gender values
            gender_values = []
            for gender in model_predictions['gender']:
                # Convert gender to numerical value (privileged class = 1, unprivileged = 0)
                # For FairFace, "Male" is typically considered privileged
                gender_values.append(1 if gender == "Male" else 0)
            
            # Prepare prediction and label values as numerical
            gender_predictions = []
            for pred in model_predictions['gender']:
                # Convert gender predictions to numerical
                gender_predictions.append(1 if pred == "Male" else 0)
                
            gender_labels = []
            for label in true_labels['gender']:
                # Convert gender labels to numerical
                gender_labels.append(1 if label == "Male" else 0)
            
            # Create dataframe with numerical values
            g_df = pd.DataFrame({
                'prediction': gender_predictions,
                'label': gender_labels,
                'gender': gender_values
            })
            
            # Print dataset distribution
            male_count = g_df[g_df['gender'] == 1].shape[0]
            female_count = g_df[g_df['gender'] == 0].shape[0]
            print(f"Gender distribution: Male={male_count}, Female={female_count}")
            
            # Print prediction accuracy
            male_correct = sum((g_df['gender'] == 1) & (g_df['prediction'] == g_df['label']))
            female_correct = sum((g_df['gender'] == 0) & (g_df['prediction'] == g_df['label']))
            print(f"Gender accuracy: Male={male_correct/male_count if male_count > 0 else 0:.2f}, "
                  f"Female={female_correct/female_count if female_count > 0 else 0:.2f}")
            
            # Define privileged and unprivileged groups
            privileged_groups = [{'gender': 1}]  # Male
            unprivileged_groups = [{'gender': 0}]  # Female
            
            # Create dataset
            dataset = BinaryLabelDataset(
                df=g_df,
                label_names=['label'],
                protected_attribute_names=['gender'],
                favorable_label=1,
                unfavorable_label=0
            )
            
            # Check if we have enough samples in both groups
            if male_count < 3 or female_count < 3:
                print(f"WARNING: Not enough samples for gender bias detection. Male={male_count}, Female={female_count}")
                # Return placeholder metrics indicating insufficient data
                bias_metrics['gender'] = {
                    'statistical_parity_difference': 0.0,
                    'disparate_impact': None,
                    'group_size': {
                        'privileged': male_count,
                        'unprivileged': female_count
                    },
                    'base_rates': {
                        'privileged': 0.0 if male_count == 0 else sum(g_df[g_df['gender'] == 1]['prediction']) / male_count,
                        'unprivileged': 0.0 if female_count == 0 else sum(g_df[g_df['gender'] == 0]['prediction']) / female_count
                    },
                    'insufficient_data': True
                }
            else:
                # Compute metrics
                metrics = self.compute_metrics(dataset, privileged_groups, unprivileged_groups)
                metrics['insufficient_data'] = False
                bias_metrics['gender'] = metrics
        
        # Check if we have race predictions and labels
        if 'race' in model_predictions and 'race' in true_labels:
            # Get unique races
            unique_races = list(set(true_labels['race']))
            print(f"Found {len(unique_races)} unique races in ground truth: {unique_races}")
            
            # Define race encoding (map each race to a numerical value)
            race_encoding = {race: i for i, race in enumerate(unique_races)}
            
            # For each race, compute metrics treating it as unprivileged vs. others
            racial_bias = {}
            
            for race in unique_races:
                # Count occurrences of this race in ground truth
                race_count = true_labels['race'].count(race)
                other_races_count = len(true_labels['race']) - race_count
                
                # Skip metrics computation if we don't have enough samples
                if race_count < 3 or other_races_count < 3:
                    print(f"WARNING: Not enough samples for race bias detection. {race}={race_count}, Other races={other_races_count}")
                    racial_bias[race] = {
                        'statistical_parity_difference': 0.0,
                        'disparate_impact': None,
                        'group_size': {
                            'privileged': other_races_count,
                            'unprivileged': race_count
                        },
                        'base_rates': {
                            'privileged': 0.0,
                            'unprivileged': 0.0
                        },
                        'insufficient_data': True
                    }
                    continue
                
                # Use one-hot encoding for races to avoid confusion
                is_this_race_true = [1 if label == race else 0 for label in true_labels['race']]
                is_this_race_pred = [1 if pred == race else 0 for pred in model_predictions['race']]
                
                # Encode binary race indicator (1 if this race, 0 otherwise)
                race_indicators = [1 if label == race else 0 for label in true_labels['race']]
                
                # Use binary predictions here: 1 if the prediction matches the true race, 0 otherwise
                # This evaluates if the model correctly identifies the true race
                prediction_correct = [1 if pred == label else 0 for pred, label in zip(model_predictions['race'], true_labels['race'])]
                
                # Count correct predictions for this race vs other races
                correct_this_race = sum([1 for i, is_race in enumerate(race_indicators) if is_race == 1 and prediction_correct[i] == 1])
                correct_other_races = sum([1 for i, is_race in enumerate(race_indicators) if is_race == 0 and prediction_correct[i] == 1])
                
                print(f"Race accuracy: {race}={correct_this_race/race_count if race_count > 0 else 0:.2f}, "
                      f"Others={correct_other_races/other_races_count if other_races_count > 0 else 0:.2f}")
                
                # Create dataframe with numerical values
                r_df = pd.DataFrame({
                    'prediction': is_this_race_pred,  # Whether predicted as this race (1) or not (0)
                    'label': is_this_race_true,       # Whether actually this race (1) or not (0)
                    'race_indicator': race_indicators # Whether this race (1) or not (0), same as label
                })
                
                # Define privileged (non-specified race) and unprivileged (specified race) groups
                privileged_groups = [{'race_indicator': 0}]  # Not this race
                unprivileged_groups = [{'race_indicator': 1}]  # This race
                
                try:
                    # Create dataset
                    dataset = BinaryLabelDataset(
                        df=r_df,
                        label_names=['label'],
                        protected_attribute_names=['race_indicator'],
                        favorable_label=1,
                        unfavorable_label=0
                    )
                    
                    # Compute metrics
                    metrics = self.compute_metrics(dataset, privileged_groups, unprivileged_groups)
                    metrics['insufficient_data'] = False
                    racial_bias[race] = metrics
                except Exception as e:
                    print(f"Error computing bias metrics for race '{race}': {str(e)}")
                    # Provide placeholder metrics
                    racial_bias[race] = {
                        'statistical_parity_difference': 0.0,
                        'disparate_impact': None,
                        'group_size': {
                            'privileged': other_races_count,
                            'unprivileged': race_count
                        },
                        'base_rates': {
                            'privileged': 0.0,
                            'unprivileged': 0.0
                        },
                        'insufficient_data': True,
                        'error': str(e)
                    }
            
            if racial_bias:  # Only add if we have metrics
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
                
                # Check if DI is None or insufficient data flag is set
                if di is None or metrics.get('insufficient_data', False):
                    bias_level = "Unknown"
                    description = "Insufficient data to determine gender bias level."
                elif abs(spd) < 0.05 and 0.95 <= di <= 1.05:
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
                    
                    # Check if DI is None or insufficient data flag is set
                    if di is None or race_metrics.get('insufficient_data', False):
                        bias_level = "Unknown"
                        description = f"Insufficient data to determine bias level for {race} individuals."
                    elif abs(spd) < 0.05 and 0.95 <= di <= 1.05:
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