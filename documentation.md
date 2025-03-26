# FRT Bias Reduction Application - Technical Documentation

This document provides detailed explanations of the technical components, methodologies, and features implemented in the Facial Recognition Technology (FRT) Bias Reduction application.

## Table of Contents
1. [Application Overview](#application-overview)
2. [FairFace Models](#fairface-models)
3. [Bias Detection](#bias-detection)
4. [Bias Mitigation Techniques](#bias-mitigation-techniques)
5. [Privacy Preservation](#privacy-preservation)
6. [Model Explainability](#model-explainability)
7. [Implementation Details](#implementation-details)
8. [References](#references)

## Application Overview

The FRT Bias Reduction application demonstrates how bias in facial recognition systems can be detected, mitigated, and explained while preserving privacy. The application consists of four main components:

1. **Home Page**: Introduces the application and allows users to upload facial images for analysis.
2. **Bias Detection Page**: Analyzes uploaded images for potential bias and provides mitigation options.
3. **Privacy Preservation Page**: Demonstrates homomorphic encryption for privacy-preserving facial analysis.
4. **Explainability Page**: Visualizes which facial features influence model predictions.

The application uses the FairFace dataset and models, which are designed for unbiased facial attribute classification.

## FairFace Models

### Model Architecture
The application uses pretrained FairFace models based on ResNet-34. These models are trained on a diverse dataset of 108,501 images, balanced across seven race groups, which helps reduce inherent dataset bias.

Two model variants are available:
- **7-race model**: Classifies race as White, Black, Latino/Hispanic, East Asian, Southeast Asian, Indian, and Middle Eastern
- **4-race model**: Classifies race as White, Black, Asian, and Indian

Both models also predict:
- **Gender**: Male or Female
- **Age**: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+

### Adaptation for Our Application
The original FairFace models are adapted to our application through the `PretrainedFairFaceAdapter` class, which:
1. Loads the pretrained weights
2. Maintains compatibility with our application interfaces
3. Normalizes input images according to the original model's requirements
4. Maps numerical outputs to human-readable categories

## Bias Detection

### Understanding Bias in Facial Recognition
Facial recognition systems can exhibit bias through differential performance across demographic groups, particularly along lines of gender, age, and race. This bias manifests as:

- **False positives**: Incorrectly matching a face to an identity
- **False negatives**: Failing to match a face to its correct identity
- **Attribute classification errors**: Incorrectly predicting demographic attributes

### Bias Metrics
Our application uses the AI Fairness 360 (AIF360) toolkit to calculate several bias metrics:

#### Statistical Parity Difference
Measures the difference in selection rates between privileged and unprivileged groups. 

```
SPD = P(Y_hat=1|D=unprivileged) - P(Y_hat=1|D=privileged)
```

Where:
- Y_hat is the predicted outcome
- D is the protected attribute (e.g., race, gender)

A value of 0 indicates no disparity in selection rates.

#### Disparate Impact
Measures the ratio of selection rates between unprivileged and privileged groups.

```
DI = P(Y_hat=1|D=unprivileged) / P(Y_hat=1|D=privileged)
```

A value of 1 indicates equal selection rates across groups.

### Implementation Details
The `BiasDetector` class:
1. Prepares datasets for bias detection based on model predictions and true labels
2. Computes bias metrics for protected attributes (gender and race)
3. Interprets metrics into human-readable bias levels (Low, Moderate, High)
4. Provides recommendations based on detected bias

## Bias Mitigation Techniques

The application implements three main bias mitigation techniques from AIF360:

### 1. Reweighting
This technique applies different weights to different instances in the training data to ensure fair outcomes across protected groups.

**How it works:**
1. Calculates weights for each (group, label) combination based on their frequencies
2. Assigns higher weights to underrepresented groups
3. Applies these weights during model training to balance influence

**Mathematical formulation:**
```
w(x) = P(Y=y|D=d) / P(Y=y,D=d)
```

Where:
- w(x) is the instance weight
- Y is the true outcome
- D is the protected attribute

**Advantages:**
- Preserves all original data
- Simple to implement
- Doesn't require modifying the model architecture

**Disadvantages:**
- May not entirely eliminate indirect discrimination
- Can increase variance in model predictions

### 2. Resampling
This technique creates a more balanced dataset by sampling instances in a way that achieves fairness.

**How it works:**
1. Uses instance weights from reweighting to create sampling probabilities
2. Oversamples underrepresented groups
3. Undersamples overrepresented groups

**Implementation:**
The application uses WeightedRandomSampler from PyTorch to create a balanced dataset for training.

**Advantages:**
- Creates an actually balanced dataset
- Works with any model architecture
- Can be more effective than reweighting for severe imbalances

**Disadvantages:**
- May lead to overfitting on minority classes
- Can lose information by undersampling majority classes

### 3. Disparate Impact Remover
This technique transforms feature distributions to achieve statistical parity while preserving rank ordering within groups.

**How it works:**
1. Maps the features to a distribution that removes information about protected attributes
2. Preserves ranking within groups (e.g., if person A was ranked higher than person B within their group before transformation, they remain ranked higher after)
3. Applies transformations with varying repair levels (0.0 to 1.0)

**Mathematical basis:**
The technique uses the Bayes optimal estimator to transform features:
```
T(x) = Q(F_d(x))
```

Where:
- T(x) is the transformed feature
- F_d is the CDF of feature values for group d
- Q is the quantile function (inverse CDF) of the target distribution

**Advantages:**
- Modifies features directly, not labels
- Can be applied as a preprocessing step
- Works with any downstream model

**Disadvantages:**
- May reduce model accuracy
- May remove legitimate correlations between features and target
- Not suitable for all types of bias

## Privacy Preservation

### Homomorphic Encryption
Homomorphic encryption allows computations on encrypted data without decryption, preserving privacy while enabling analysis.

#### CKKS Encryption Scheme
Our application uses the CKKS (Cheon-Kim-Kim-Song) encryption scheme through the TenSEAL library, which:
1. Allows approximate arithmetic on encrypted real numbers
2. Supports both addition and multiplication
3. Enables vector operations, ideal for machine learning applications

#### Implementation
The `HomomorphicEncryptor` class provides:
1. **Encryption Context Creation**: Sets parameters like polynomial modulus degree and coefficient modulus
2. **Vector Encryption**: Encrypts feature vectors extracted from face images
3. **Encrypted Computation**: Performs linear transformations on encrypted data
4. **Benchmarking**: Measures encryption/decryption times and performance impact

#### Privacy-Preserving Inference Process
1. Extract facial feature embeddings from the image
2. Encrypt these embeddings using homomorphic encryption
3. Perform model inference on the encrypted data
4. Return encrypted results
5. Decrypt results only on the client side

#### Limitations
1. Computational overhead: Homomorphic operations are much slower than plaintext operations
2. Limited operations: Complex non-linear functions are challenging to implement
3. Precision trade-offs: CKKS scheme introduces small approximation errors

## Model Explainability

### SHAP (SHapley Additive exPlanations)
SHAP is a unified framework for interpreting model predictions based on game-theoretic Shapley values.

#### Theoretical Basis
Shapley values assign each feature a contribution value by considering all possible feature combinations:

```
φᵢ(f,x) = ∑(S⊆N\{i}) [|S|!(|N|-|S|-1)!/|N|!] * [f(S∪{i}) - f(S)]
```

Where:
- φᵢ is the Shapley value for feature i
- N is the set of all features
- S is a subset of features
- f(S) is the prediction with only the features in S

#### DeepExplainer
For deep learning models like our facial recognition system, the application uses DeepExplainer, which:
1. Approximates Shapley values using a background dataset
2. Creates attribution maps showing each pixel's contribution to predictions
3. Visualizes which facial regions most influence specific predictions

#### Visualization
The generated SHAP visualizations use color coding:
- **Red regions**: Features that pushed the prediction toward the predicted class
- **Blue regions**: Features that pushed the prediction away from the predicted class
- **Color intensity**: Magnitude of feature influence

#### Implementation
The `ShapExplainer` class provides:
1. **Background Sample Management**: Uses representative background samples for baseline comparisons
2. **Task-Specific Explanations**: Generates explanations for gender, age, or race predictions
3. **Attribution Aggregation**: Combines attributions across color channels for visualization
4. **Multi-Task Visualization**: Shows explanations for different prediction tasks side-by-side

## Implementation Details

### Key Classes and Their Functions

#### 1. PretrainedFairFaceAdapter (src/models/fairface_adapter.py)
Adapts pretrained FairFace models to work with our application.
- `_load_model()`: Loads and prepares the ResNet-34 model
- `predict()`: Makes predictions on new images
- `from_pretrained()`: Factory method to create an adapter from pretrained weights

#### 2. BiasDetector (src/bias_mitigation/detector.py)
Detects bias in model predictions across protected attributes.
- `prepare_dataset()`: Formats prediction data for bias analysis
- `compute_metrics()`: Calculates statistical parity and disparate impact
- `detect_bias()`: Analyzes bias across multiple protected attributes
- `interpret_bias_metrics()`: Translates metrics into actionable insights

#### 3. BiasMitigator (src/bias_mitigation/mitigator.py)
Implements bias mitigation techniques.
- `mitigate_bias()`: Applies the selected mitigation technique
- `apply_reweighting()`: Implements the reweighting technique
- `generate_resampling_weights()`: Creates weights for balanced sampling
- `apply_disparate_impact_remover()`: Transforms features to remove disparate impact

#### 4. HomomorphicEncryptor (src/privacy/encryption.py)
Provides homomorphic encryption capabilities.
- `_create_context()`: Sets up the encryption context
- `encrypt_vector()`: Encrypts feature vectors
- `linear_transform()`: Performs operations on encrypted data
- `benchmark_encryption()`: Measures encryption performance

#### 5. ShapExplainer (src/explainability/explainer.py)
Generates explanations for model predictions.
- `initialize_explainer()`: Sets up the SHAP explainer with background samples
- `explain_image()`: Generates SHAP explanations for an image
- `_create_visualization()`: Creates visual representations of SHAP values
- `plot_multiple_explanations()`: Shows explanations for multiple prediction tasks

### UI Components (src/ui/components.py)
- `display_bias_metrics()`: Visualizes bias detection results
- `display_shap_explanation()`: Shows SHAP visualizations
- `display_privacy_metrics()`: Presents privacy benchmarking results

## References

### Facial Recognition and Fairness
- [FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age](https://github.com/joojs/fairface)
- Kärkkäinen, K., & Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 1548-1558).

### Bias Detection and Mitigation
- [AI Fairness 360](https://github.com/Trusted-AI/AIF360)
- Bellamy, R. K., et al. (2019). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias. IBM Journal of Research and Development, 63(4/5), 4:1-4:15.

### Homomorphic Encryption
- [TenSEAL](https://github.com/OpenMined/TenSEAL)
- Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017). Homomorphic encryption for arithmetic of approximate numbers. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 409-437). Springer.

### Explainability
- [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap)
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (pp. 4765-4774). 