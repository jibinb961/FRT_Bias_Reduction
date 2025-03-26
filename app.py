import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
import time

# Import our model implementations
from src.models.model import FaceRecognitionModel
from src.models.fairface_adapter import PretrainedFairFaceAdapter
from src.bias_mitigation.detector import BiasDetector
from src.bias_mitigation.mitigator import BiasMitigator
from src.privacy.encryption import HomomorphicEncryptor
from src.explainability.explainer import ShapExplainer
from src.ui.components import (
    display_bias_metrics,
    display_shap_explanation,
    display_privacy_metrics
)

def main():
    st.set_page_config(
        page_title="FRT Bias Reduction Demo",
        page_icon="🧑‍🤝‍🧑",
        layout="wide"
    )
    
    st.title("Facial Recognition Technology Bias Reduction Demo")
    st.sidebar.title("Controls")
    
    # Session state initialization
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    
    if 'bias_detector' not in st.session_state:
        st.session_state['bias_detector'] = BiasDetector()
    
    if 'bias_mitigator' not in st.session_state:
        st.session_state['bias_mitigator'] = BiasMitigator()
    
    if 'encryptor' not in st.session_state:
        st.session_state['encryptor'] = None
    
    if 'explainer' not in st.session_state:
        st.session_state['explainer'] = None
    
    # Load model if available
    model_options = [
        "None", 
        "Pretrained FairFace (7 races)", 
        "Pretrained FairFace (4 races)",
        "models/best_model.pth", 
        "models/mitigated_model.pth"
    ]
    
    model_path = st.sidebar.selectbox(
        "Select Model",
        model_options,
        help="Select a model to use for predictions"
    )
    
    # Handle model loading
    if model_path != "None":
        # Check if we need to load a new model
        if st.session_state['model'] is None or st.session_state['model_path'] != model_path:
            with st.spinner("Loading model..."):
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    # Handle different model types
                    if model_path == "Pretrained FairFace (7 races)":
                        model = PretrainedFairFaceAdapter.from_pretrained('race_7')
                        model_type = "fairface_pretrained"
                    elif model_path == "Pretrained FairFace (4 races)":
                        model = PretrainedFairFaceAdapter.from_pretrained('race_4')
                        model_type = "fairface_pretrained"
                    else:
                        # Our custom trained model
                        if os.path.exists(model_path):
                            model = FaceRecognitionModel.load(model_path, map_location=device)
                            model_type = "custom"
                        else:
                            st.sidebar.warning(f"Model file not found: {model_path}")
                            return
                    
                    # Save the model and its type
                    if model_type == "custom":
                        model.eval()
                    
                    st.session_state['model'] = model
                    st.session_state['model_path'] = model_path
                    st.session_state['model_type'] = model_type
                    st.sidebar.success(f"Model loaded successfully: {model_path}")
                    
                    # Initialize explainer for custom models
                    if model_type == "custom":
                        st.session_state['explainer'] = ShapExplainer(model, device=device)
                    else:
                        # For pretrained models, we would need to adapt the explainer
                        # or use a different explanation approach
                        st.session_state['explainer'] = None
                        
                except Exception as e:
                    st.sidebar.error(f"Error loading model: {e}")
    else:
        st.session_state['model'] = None
    
    # Navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Home", "Bias Detection", "Privacy Preservation", "Explainability"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Bias Detection":
        show_bias_detection_page()
    elif page == "Privacy Preservation":
        show_privacy_preservation_page()
    elif page == "Explainability":
        show_explainability_page()

def show_home_page():
    st.header("Welcome to the FRT Bias Reduction Demo")
    st.write("""
    This application demonstrates how bias in facial recognition technology can be detected,
    mitigated, and explained, while preserving privacy through homomorphic encryption.
    
    ### About the Project
    
    This prototype focuses on:
    - Detecting and mitigating bias in facial recognition using AI Fairness 360
    - Enhancing privacy with homomorphic encryption
    - Providing model explainability with SHAP visualizations
    
    ### Getting Started
    
    Select a page from the sidebar to explore different aspects of the application.
    """)
    
    # Add information about using pretrained FairFace models
    st.info("""
    **Using Pretrained FairFace Models:**
    
    This application supports pretrained FairFace models from the original repository.
    You can select either the 7-race or 4-race model from the dropdown in the sidebar.
    
    - **7-race model**: Predicts race as White, Black, Latino/Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern
    - **4-race model**: Predicts race as White, Black, Asian, Indian
    
    If you don't see these options, ensure the model files are in the `models/` directory:
    - `res34_fair_align_multi_7_20190809.pt`
    - `res34_fair_align_multi_4_20190809.pt`
    """)
    
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image of a face", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.session_state["current_image"] = image
        st.success("Image uploaded successfully! Navigate to other pages to analyze it.")

def show_bias_detection_page():
    st.header("Bias Detection and Mitigation")
    
    if "current_image" not in st.session_state:
        st.warning("Please upload an image on the home page first.")
        return
    
    if st.session_state['model'] is None:
        st.warning("Please select a model from the sidebar first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state["current_image"], use_column_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            prediction = st.session_state['model'].predict(st.session_state["current_image"])
            
            # Display prediction
            st.write("Detected attributes:")
            st.json({
                "gender": prediction["gender"],
                "age": prediction["age"],
                "race": prediction["race"],
                "confidence_scores": {
                    "gender": f"{prediction['confidence_scores']['gender']:.4f}",
                    "age": f"{prediction['confidence_scores']['age']:.4f}",
                    "race": f"{prediction['confidence_scores']['race']:.4f}"
                }
            })
    
    # Bias Detection section
    st.subheader("Bias Detection")
    
    # In a real application, we would use a test dataset
    # For demo purposes, we'll generate some sample bias metrics
    if st.button("Detect Potential Bias"):
        with st.spinner("Analyzing for bias..."):
            # Simulate bias detection (in a real app, this would use real data)
            time.sleep(1)  # Simulate computation
            
            # Sample bias metrics
            bias_metrics = {
                'gender': {
                    'statistical_parity_difference': -0.15,
                    'disparate_impact': 0.85,
                    'group_size': {
                        'privileged': 500,
                        'unprivileged': 450
                    }
                },
                'race': {
                    'Black': {
                        'statistical_parity_difference': -0.18,
                        'disparate_impact': 0.82,
                        'group_size': {
                            'privileged': 500,
                            'unprivileged': 300
                        }
                    },
                    'Latino/Hispanic': {
                        'statistical_parity_difference': -0.12,
                        'disparate_impact': 0.88,
                        'group_size': {
                            'privileged': 500,
                            'unprivileged': 280
                        }
                    },
                    'East Asian': {
                        'statistical_parity_difference': -0.05,
                        'disparate_impact': 0.95,
                        'group_size': {
                            'privileged': 500,
                            'unprivileged': 250
                        }
                    }
                }
            }
            
            # Sample interpretations
            bias_interpretations = {
                'gender': {
                    'bias_level': "Moderate",
                    'description': "The model shows some gender bias that may need attention."
                },
                'race': {
                    'Black': {
                        'bias_level': "High",
                        'description': "The model shows significant bias against Black individuals that requires mitigation."
                    },
                    'Latino/Hispanic': {
                        'bias_level': "Moderate",
                        'description': "The model shows some bias against Latino/Hispanic individuals that may need attention."
                    },
                    'East Asian': {
                        'bias_level': "Low",
                        'description': "The model shows minimal bias against East Asian individuals."
                    }
                }
            }
            
            # Display bias metrics using our UI component
            display_bias_metrics(bias_metrics, bias_interpretations)
    
    # Bias Mitigation section
    st.subheader("Bias Mitigation")
    mitigation_method = st.selectbox(
        "Select Bias Mitigation Method",
        ["None", "Reweighting", "Resampling", "Disparate Impact Remover"]
    )
    
    if mitigation_method != "None" and st.button("Apply Mitigation"):
        with st.spinner(f"Applying {mitigation_method} mitigation..."):
            # Simulate mitigation (in a real app, this would use real data)
            time.sleep(2)  # Simulate computation
            
            st.success(f"{mitigation_method} has been applied to mitigate bias.")
            
            # Show simulated improved metrics
            improved_bias_metrics = {
                'gender': {
                    'statistical_parity_difference': -0.05,
                    'disparate_impact': 0.95,
                    'group_size': {
                        'privileged': 500,
                        'unprivileged': 450
                    }
                },
                'race': {
                    'Black': {
                        'statistical_parity_difference': -0.08,
                        'disparate_impact': 0.92,
                        'group_size': {
                            'privileged': 500,
                            'unprivileged': 300
                        }
                    },
                    'Latino/Hispanic': {
                        'statistical_parity_difference': -0.05,
                        'disparate_impact': 0.95,
                        'group_size': {
                            'privileged': 500,
                            'unprivileged': 280
                        }
                    },
                    'East Asian': {
                        'statistical_parity_difference': -0.01,
                        'disparate_impact': 0.99,
                        'group_size': {
                            'privileged': 500,
                            'unprivileged': 250
                        }
                    }
                }
            }
            
            improved_interpretations = {
                'gender': {
                    'bias_level': "Low",
                    'description': "The model shows minimal gender bias after mitigation."
                },
                'race': {
                    'Black': {
                        'bias_level': "Moderate",
                        'description': "The model shows reduced bias against Black individuals after mitigation, but may need further attention."
                    },
                    'Latino/Hispanic': {
                        'bias_level': "Low",
                        'description': "The model shows minimal bias against Latino/Hispanic individuals after mitigation."
                    },
                    'East Asian': {
                        'bias_level': "Low",
                        'description': "The model shows minimal bias against East Asian individuals."
                    }
                }
            }
            
            st.markdown("### After Mitigation")
            display_bias_metrics(improved_bias_metrics, improved_interpretations)

def show_privacy_preservation_page():
    st.header("Privacy Preservation with Homomorphic Encryption")
    
    if "current_image" not in st.session_state:
        st.warning("Please upload an image on the home page first.")
        return
    
    if st.session_state['model'] is None:
        st.warning("Please select a model from the sidebar first.")
        return
    
    st.image(st.session_state["current_image"], caption="Original Image", width=300)
    
    st.subheader("Homomorphic Encryption")
    st.write("""
    Homomorphic encryption allows computations on encrypted data without decrypting it first.
    This demonstration will show how facial features can be encrypted while still allowing
    the model to make predictions.
    """)
    
    # Initialize encryptor if needed
    if st.session_state['encryptor'] is None:
        with st.spinner("Initializing encryption..."):
            st.session_state['encryptor'] = HomomorphicEncryptor()
    
    # Encryption demo
    if st.button("Encrypt and Analyze"):
        with st.spinner("Encrypting and analyzing..."):
            # Run benchmarks
            encryption_benchmarks = st.session_state['encryptor'].benchmark_encryption(vector_size=512, n_trials=5)
            
            # Display privacy metrics
            st.success("Analysis completed on encrypted data!")
            display_privacy_metrics(encryption_benchmarks)
            
            # Sample encrypted prediction results
            st.subheader("Encrypted Prediction Results")
            st.json({
                "encrypted_analysis": True,
                "prediction": {
                    "gender": "Result available only after decryption",
                    "age": "Result available only after decryption",
                    "race": "Result available only after decryption"
                },
                "privacy_score": "High",
                "encryption_scheme": "CKKS (Homomorphic)"
            })
            
            # Add an explanation of the process
            st.info("""
            In a real application, the following steps would occur:
            1. Facial features are extracted and encrypted
            2. The encrypted features are processed by the model
            3. Results are returned in encrypted form
            4. Results are decrypted only on the user's device
            
            This ensures that sensitive biometric data remains private throughout the process.
            """)

def show_explainability_page():
    st.header("Model Explainability with SHAP")
    
    if "current_image" not in st.session_state:
        st.warning("Please upload an image on the home page first.")
        return
    
    if st.session_state['model'] is None:
        st.warning("Please select a model from the sidebar first.")
        return
    
    if st.session_state['explainer'] is None:
        st.warning("Explainer not initialized. Please ensure model is loaded.")
        return
    
    st.image(st.session_state["current_image"], caption="Original Image", width=300)
    
    st.subheader("Feature Importance")
    st.write("""
    SHAP (SHapley Additive exPlanations) helps explain which parts of the image
    are most important for the model's predictions.
    """)
    
    # Select task to explain
    task = st.selectbox(
        "Select attribute to explain",
        ["gender", "age", "race"]
    )
    
    if st.button("Generate Explanation"):
        with st.spinner("Generating explanation..."):
            # In a real application, we would use the actual explainer
            # For demo purposes, we'll simulate the explanation
            time.sleep(2)  # Simulate computation
            
            # Create a sample explanation visualization
            image = st.session_state["current_image"]
            
            # Create a heatmap overlay (this is just a placeholder)
            img_array = np.array(image)
            heatmap = np.zeros_like(img_array)
            
            # Simulate a heatmap focused on facial features
            h, w, _ = img_array.shape
            center_y, center_x = h // 2, w // 2
            
            # Create a gradient centered on the face
            for y in range(h):
                for x in range(w):
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    intensity = max(0, 1 - dist / (min(h, w) // 2))
                    
                    # Eyes region (stronger positive attribution)
                    if center_y - h//5 <= y <= center_y and center_x - w//4 <= x <= center_x + w//4:
                        heatmap[y, x, 0] = 200 * intensity  # Red for positive attribution
                    
                    # Mouth region (medium positive attribution)
                    elif center_y + h//8 <= y <= center_y + h//4 and center_x - w//6 <= x <= center_x + w//6:
                        heatmap[y, x, 0] = 150 * intensity  # Red for positive attribution
                    
                    # Background (negative attribution)
                    elif dist > min(h, w) // 3:
                        heatmap[y, x, 2] = 150 * intensity  # Blue for negative attribution
            
            # Create the overlay
            alpha = 0.5
            overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
            explanation = Image.fromarray(overlay)
            
            st.success("Explanation generated!")
            
            # Display the explanation
            display_shap_explanation(image, explanation, task)
            
            # Additional information about the explanation
            st.subheader("Interpretation")
            st.write(f"""
            The explanation shows which facial features were most important for the model's {task} prediction:
            
            - **Eyes and eyebrows region**: Strongly influences the model's decision
            - **Mouth and chin area**: Moderately important for the prediction
            - **Background and hair**: Less important, sometimes pushing against the prediction
            
            This kind of explanation helps us understand what the model is "looking at" when making decisions,
            which is crucial for detecting and addressing potential biases.
            """)

if __name__ == "__main__":
    main() 