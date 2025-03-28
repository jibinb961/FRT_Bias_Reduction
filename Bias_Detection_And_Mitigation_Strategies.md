# Bias Detection and Mitigation Strategies

This document provides an in-depth explanation of how bias detection and mitigation work in our Facial Recognition Technology (FRT) application.

## 1. Understanding Bias in Facial Recognition

Before diving into detection and mitigation, it's important to understand what constitutes bias in facial recognition systems:

### Types of Bias in FRT

1. **Demographic Bias**: Models may perform better for certain demographic groups (based on gender, race, age) than others.
2. **Representation Bias**: When training data under-represents certain groups, leading to poor performance for those groups.
3. **Measurement Bias**: When the data collection process itself introduces biases.
4. **Aggregation Bias**: When combining data across groups masks differences between those groups.
5. **Evaluation Bias**: When test data does not represent the diversity of real-world applications.

### Real-world Implications

Biased facial recognition systems can lead to:
- Unfair treatment of certain demographic groups
- Higher false positive rates for marginalized communities
- Reinforcement of existing social inequalities
- Loss of trust in AI systems
- Potential legal and ethical issues

## 2. Bias Detection Implementation

Our application uses the AI Fairness 360 (AIF360) toolkit to detect bias across different demographic groups. Here's how the system works under the hood:

### Data Collection and Processing

The "Detect Potential Bias" button triggers the following sequence:

1. The system loads images from the `sample_data` directory
2. For each image, it:
   - Processes the image through the selected model (e.g., FairFace)
   - Extracts predictions for gender, race, and age
   - Stores these predictions in a standardized format
3. The predictions are used as both the model outputs and the ground truth labels (in a real system, you would have separate ground truth labels)

### Numerical Conversion

AIF360 requires numerical values for its calculations, so the system converts categorical attributes to numbers:
- Gender: "Male" → 1, "Female" → 0
- Race: For each race, we create binary indicators (1 if the person belongs to that race, 0 otherwise)

### Metrics Calculation

#### For Gender Bias:

1. **Statistical Parity Difference (SPD)**:
   - Formula: `P(Ŷ=1|D=unprivileged) - P(Ŷ=1|D=privileged)`
   - Where Ŷ is the prediction and D is the demographic group
   - Measures the difference in selection rates between groups
   - Values close to 0 indicate minimal bias
   - Negative values indicate bias against unprivileged groups (e.g., females)
   - Positive values indicate bias against privileged groups (e.g., males)

2. **Disparate Impact (DI)**:
   - Formula: `P(Ŷ=1|D=unprivileged) / P(Ŷ=1|D=privileged)`
   - Measures the ratio of selection rates between groups
   - Values close to 1.0 indicate minimal bias
   - Values below 0.8 typically indicate bias against unprivileged groups
   - Values above 1.25 indicate bias against privileged groups

#### For Racial Bias:

Similar metrics are calculated for each race, comparing:
- The specified race (unprivileged group)
- All other races combined (privileged group)

### Code Implementation

The core of the bias detection happens in the `detector.py` file:

```python
# Convert categorical attributes to numerical
gender_values = [1 if gender == "Male" else 0 for gender in predictions['gender']]

# Create dataset for AIF360
dataset = BinaryLabelDataset(
    df=dataframe,
    label_names=['label'],
    protected_attribute_names=['gender'],
    favorable_label=1,
    unfavorable_label=0
)

# Calculate metrics
metrics = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=[{'gender': 0}],  # Female
    privileged_groups=[{'gender': 1}]      # Male
)

# Extract the key fairness metrics
spd = metrics.statistical_parity_difference()
di = metrics.disparate_impact()
```

### Data Sufficiency and Reliability

The system checks if there's enough data for reliable metrics:
- A minimum of 3 samples per group is required
- If insufficient data is available, the system shows placeholder metrics and warnings
- For small sample sizes, "∞" may appear for Disparate Impact if there's a division by zero

## 3. Bias Mitigation Strategies

Our application implements three primary bias mitigation strategies from AIF360:

### 1. Reweighting

**How it works:**
- Weights are assigned to different demographic groups to balance their influence
- The weight formula is: `w = expected_proportion / observed_proportion`
- Groups with higher representation are downweighted
- Groups with lower representation are upweighted

**Technical implementation:**
```python
from aif360.algorithms.preprocessing import Reweighing

reweigher = Reweighing(
    unprivileged_groups=[{'gender': 0}],
    privileged_groups=[{'gender': 1}]
)

# Apply the transformation
transformed_dataset = reweigher.fit_transform(original_dataset)
```

**When to use:**
- When you can modify training data weights
- When the dataset has imbalanced representation
- For addressing representation bias

**Limitations:**
- Doesn't directly modify the model or predictions
- May not address bias in the model architecture itself

### 2. Resampling

**How it works:**
- Creates a new dataset by selectively sampling from the original data
- Oversamples from underrepresented groups
- Undersamples from overrepresented groups
- The goal is to create a balanced dataset across protected attributes

**Technical implementation:**
```python
# For discrimination case: privileged is preferred
if instance_weights[idx] < 1.0:
    # Undersampling
    if np.random.random() < instance_weights[idx]:
        new_dataset.append(instance)
else:
    # Oversampling
    num_copies = int(instance_weights[idx])
    for _ in range(num_copies):
        new_dataset.append(instance)
```

**When to use:**
- When you can modify your training data
- When you have a large enough dataset that can afford to lose some samples
- For addressing performance disparities across groups

**Limitations:**
- Can reduce the overall dataset size
- May lead to overfitting for minority groups
- Less effective with very small datasets

### 3. Disparate Impact Remover

**How it works:**
- Transforms feature values to achieve statistical parity
- Modifies the data distribution while preserving rank-ordering within groups
- Doesn't change the label values, only modifies features

**Technical implementation:**
```python
from aif360.algorithms.preprocessing import DisparateImpactRemover

# Repair level determines how aggressively to transform features (0 to 1)
di_remover = DisparateImpactRemover(repair_level=0.8)
transformed_dataset = di_remover.fit_transform(original_dataset)
```

**When to use:**
- When you want to keep your original model
- When the bias is primarily in the feature representations
- For addressing discrimination during feature preprocessing
- When you need to meet regulatory requirements for statistical parity

**Limitations:**
- May reduce overall model accuracy
- Doesn't address bias in the learning algorithm itself
- More aggressive repair levels can distort data relationships

## 4. Behind the "Apply Mitigation" Button

When you click "Apply Mitigation" in our application, the following happens:

1. The system retrieves the stored predictions and protected attributes
2. Selects the appropriate mitigation algorithm based on your choice
3. Applies the algorithm to transform the data
4. Recalculates the fairness metrics on the transformed data
5. Displays a before/after comparison of the metrics

For any mitigation method, the code follows this general pattern:

```python
# Get mitigator from session state
mitigator = st.session_state['bias_mitigator']

# Apply chosen mitigation method
mitigated_dataset, _ = mitigator.mitigate_bias(
    original_dataset, 
    mitigation_method,  # e.g., "Reweighting"
    protected_attribute, 
    privileged_groups, 
    unprivileged_groups
)

# Compute metrics on mitigated dataset
mitigated_metrics = bias_detector.compute_metrics(
    mitigated_dataset,
    privileged_groups,
    unprivileged_groups
)
```

## 5. Best Practices for Bias Mitigation

To get the most effective bias mitigation in the application:

1. **Use diverse data**: Ensure your `sample_data` directory contains diverse images across different demographic groups.

2. **Choose the right mitigation strategy**:
   - Use Reweighting when you have imbalanced representation
   - Use Resampling when you have a large dataset with performance disparities
   - Use Disparate Impact Remover when you need to meet specific fairness thresholds

3. **Monitor multiple metrics**: Look at both SPD and DI to get a comprehensive view of bias.

4. **Balance fairness with performance**: More aggressive bias mitigation may reduce overall model accuracy.

5. **Apply bias mitigation early**: Addressing bias in data preprocessing is often more effective than post-processing.

## 6. Limitations and Considerations

When using the bias detection and mitigation features, keep in mind:

- **Sample size matters**: Small samples can lead to unreliable metrics.

- **Ground truth quality**: In real applications, you need accurate ground truth labels for reliable bias detection.

- **Multiple types of bias**: The current implementation focuses on demographic parity, but other fairness notions (e.g., equal opportunity, equalized odds) may be more appropriate in certain contexts.

- **Context specificity**: The appropriate level of bias mitigation depends on the specific application context and the potential harm of false positives vs. false negatives.

- **Continuous monitoring**: Bias detection and mitigation should be an ongoing process, not a one-time fix.

## 7. Further Reading

For more in-depth understanding of fairness in machine learning:

1. AI Fairness 360 Documentation: [https://aif360.readthedocs.io/](https://aif360.readthedocs.io/)

2. "Fairness and Machine Learning" by Barocas, Hardt, and Narayanan: [https://fairmlbook.org/](https://fairmlbook.org/)

3. FairFace paper: "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age"

4. IBM Research Trusted AI: [https://www.research.ibm.com/artificial-intelligence/trusted-ai/](https://www.research.ibm.com/artificial-intelligence/trusted-ai/) 