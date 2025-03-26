# Facial Recognition Technology (FRT) Bias Reduction

A prototype project to reduce bias and ensure privacy in facial recognition technology, focusing on agricultural sector applications.

## Project Overview

This project aims to develop a simplified working prototype that demonstrates:
- Bias detection and mitigation using AI Fairness 360
- Privacy preservation through homomorphic encryption (Microsoft SEAL via TenSEAL)
- Model explainability with SHAP visualizations
- A user-friendly interface using Streamlit

## Setup Instructions

1. Clone the repository
   ```
   git clone https://github.com/yourusername/FRT_Bias_Reduction.git
   cd FRT_Bias_Reduction
   ```

2. Create and activate a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Download the FairFace dataset
   ```
   python scripts/download_dataset.py
   ```
   
   **Note**: Due to Google Drive limitations, the script may fail. If that happens, manually download the FairFace dataset from [https://github.com/joojs/fairface](https://github.com/joojs/fairface) and place the files in the `data/fairface` directory.

## Complete Workflow

### 1. Data Preparation

After downloading the dataset, the processing happens automatically. The script:
- Extracts ZIP files to appropriate directories
- Organizes data into train, validation, and test splits
- Creates CSV files with labels for each split

The `src/data/dataset.py` module handles loading and preprocessing this data for training.

### Using Pretrained FairFace Models

You can skip training and use the pretrained models from the FairFace repository:

1. Download the pretrained models from [https://github.com/joojs/fairface](https://github.com/joojs/fairface)
   - `res34_fair_align_multi_7_20190809.pt`: 7-race classification model
   - `res34_fair_align_multi_4_20190809.pt`: 4-race classification model

2. Place these files in the `models/` directory:
   ```
   mkdir -p models
   mv /path/to/res34_fair_align_multi_7_20190809.pt models/
   mv /path/to/res34_fair_align_multi_4_20190809.pt models/
   ```

3. When running the Streamlit app, select one of the pretrained models from the dropdown menu:
   - "Pretrained FairFace (7 races)"
   - "Pretrained FairFace (4 races)"

The 7-race model is recommended for more detailed bias analysis as it provides finer-grained race classifications.

### 2. Training the Model (Optional if using pretrained models)

To train the facial recognition model:

```
python scripts/train_model.py --data_dir data/fairface --output_dir models --num_epochs 20
```

Parameters you can adjust:
- `--data_dir`: Directory containing the dataset (default: data/fairface)
- `--output_dir`: Directory to save trained models (default: models)
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of epochs to train (default: 20)
- `--learning_rate`: Learning rate (default: 0.001)
- `--device`: Device to use (cuda or cpu, default: auto-detect)

Training output includes:
- `best_model.pth`: Model with the best validation performance
- `final_model.pth`: Model after completing all epochs
- `model_epoch_X.pth`: Checkpoints after each epoch
- `loss_plot.png`: Visualization of training/validation loss

### 3. Bias Detection and Mitigation

Bias detection and mitigation is handled by modules in `src/bias_mitigation`:
- `BiasDetector`: Analyzes model predictions for bias across protected attributes (gender, race)
- `BiasMitigator`: Implements various techniques to reduce bias:
  - Reweighting: Adjusts instance weights to balance outcomes
  - Resampling: Creates balanced training data
  - Disparate Impact Remover: Transforms features to remove bias

### 4. Privacy Preservation

The `src/privacy` module implements homomorphic encryption using TenSEAL:
- Allows computation on encrypted data without decryption
- Secures facial feature vectors during inference
- Provides benchmarking to measure performance impact

### 5. Model Explainability

The `src/explainability` module uses SHAP to explain model predictions:
- Visualizes which facial features influence predictions
- Shows attribution of features to specific outcomes
- Helps identify potential sources of bias

### 6. Running the Demo Application

For a quick start with a prepared demo:

```
./run_demo.sh
```

This script will:
1. Create a virtual environment if needed
2. Install required dependencies
3. Create a dummy model for testing if none exists
4. Run the Streamlit application

Alternatively, you can run each step manually:

```
# Prepare sample data and dummy model
python scripts/prepare_sample_data.py --output_dir sample_data --model_dir models

# Run the Streamlit app
streamlit run app.py
```

### 7. Using the Streamlit Interface

The Streamlit app provides several pages:

1. **Home**: Upload face images for analysis
2. **Bias Detection**: Analyze for bias and apply mitigation techniques
3. **Privacy Preservation**: Demonstrate homomorphic encryption
4. **Explainability**: Visualize SHAP explanations for model predictions

To use the application:
1. Select a model from the sidebar dropdown
2. Upload an image on the home page
3. Navigate to different pages to analyze the image

## Project Structure

```
FRT_Bias_Reduction/
├── data/                   # Dataset storage
├── models/                 # Trained model files
├── notebooks/              # Jupyter notebooks for development
├── scripts/                # Utility scripts
│   ├── download_dataset.py # Script to download FairFace dataset
│   ├── train_model.py      # Script to train the model
│   └── prepare_sample_data.py # Script to prepare sample data
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── models/             # Model architecture and training
│   ├── bias_mitigation/    # Bias detection and mitigation
│   ├── privacy/            # Privacy preservation with encryption
│   ├── explainability/     # Model explainability
│   └── ui/                 # Streamlit user interface
├── tests/                  # Unit tests
├── app.py                  # Main Streamlit application
├── run_demo.sh             # Script to run the demo
├── README.md               # Project documentation
├── DEVLOG.md               # Development log
└── NEXTSTEPS.md            # Notes for future development
```

## Implementation Details

### Model Architecture

The facial recognition model uses a ResNet-18 architecture with:
- Pre-trained ImageNet weights for feature extraction
- Multi-task learning for simultaneous prediction of:
  - Gender (binary classification)
  - Age (9 age groups)
  - Race (7 racial categories)

### Bias Metrics

The bias detection module calculates:
- Statistical Parity Difference: Measures difference in selection rates between groups
- Disparate Impact: Ratio of selection rates between unprivileged and privileged groups
- Group sizes and distributions

### Privacy Implementation

The homomorphic encryption using TenSEAL:
- Uses the CKKS encryption scheme
- Encrypts feature vectors extracted from faces
- Performs encrypted inference on these vectors
- Measures encryption and decryption time impact

### Explainability Approach

SHAP visualizations show:
- Red regions: Features that pushed the prediction toward the predicted class
- Blue regions: Features that pushed the prediction away from the predicted class

## Troubleshooting

1. **Dataset Download Issues**: If the automatic download fails, manually download the FairFace dataset and place files in the `data/fairface` directory.

2. **CUDA/GPU Issues**: If you encounter CUDA errors, try running with CPU by adding `--device cpu` to the training command.

3. **Memory Issues**: If you run out of memory during training, reduce the batch size with `--batch_size 16` or lower.

4. **TenSEAL Compatibility**: If you encounter TenSEAL errors, ensure you have compatible versions of PyTorch and TenSEAL as specified in requirements.txt.

## Future Improvements

See `NEXTSTEPS.md` for planned improvements, including:
- Advanced bias mitigation techniques
- More sophisticated homomorphic encryption approaches
- Extended support for custom models and datasets

## License

[Add your license information here] 