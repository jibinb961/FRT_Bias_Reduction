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

## Project Structure

```
FRT_Bias_Reduction/
├── data/                   # Dataset storage
├── models/                 # Trained model files
├── notebooks/              # Jupyter notebooks for development
├── scripts/                # Utility scripts
├── src/                    # Source code
│   ├── data/               # Data processing modules
│   ├── models/             # Model architecture and training
│   ├── bias_mitigation/    # Bias detection and mitigation
│   ├── privacy/            # Privacy preservation with encryption
│   ├── explainability/     # Model explainability
│   └── ui/                 # Streamlit user interface
├── tests/                  # Unit tests
├── app.py                  # Main Streamlit application
├── README.md               # Project documentation
├── DEVLOG.md               # Development log
└── NEXTSTEPS.md            # Notes for future development
```

## Usage

To run the Streamlit application:
```
streamlit run app.py
```

## Development Stages

1. Project Setup and Initialization
2. Data Preparation (FairFace dataset)
3. Model Development and Training (ResNet-18)
4. Bias Detection and Mitigation (AI Fairness 360)
5. Privacy Preservation (Homomorphic Encryption)
6. Explainability and Testing (SHAP)
7. UI Development (Streamlit)
8. Final Testing and Refinement

## License

[Add your license information here] 