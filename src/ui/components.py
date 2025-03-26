import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64

def display_bias_metrics(bias_metrics, bias_interpretations):
    """
    Display bias metrics in the Streamlit UI
    
    Args:
        bias_metrics (dict): Dictionary of bias metrics
        bias_interpretations (dict): Dictionary of bias interpretations
    """
    st.subheader("Bias Detection Results")
    
    # Gender bias
    if 'gender' in bias_metrics:
        st.write("### Gender Bias")
        
        gender_metrics = bias_metrics['gender']
        gender_interp = bias_interpretations['gender']
        
        # Display interpretation
        st.write(f"**Bias Level**: {gender_interp['bias_level']}")
        st.write(f"**Description**: {gender_interp['description']}")
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Statistical Parity Difference", 
                f"{gender_metrics['statistical_parity_difference']:.4f}",
                help="Measures the difference in selection rates between privileged and unprivileged groups. Value of 0 means no bias."
            )
        
        with col2:
            st.metric(
                "Disparate Impact", 
                f"{gender_metrics['disparate_impact']:.4f}",
                help="Ratio of selection rates between unprivileged and privileged groups. Value of 1 means no bias."
            )
        
        # Display group sizes
        group_sizes = gender_metrics['group_size']
        
        # Create a bar chart for group sizes
        fig = px.bar(
            x=['Male', 'Female'],
            y=[group_sizes['privileged'], group_sizes['unprivileged']],
            labels={'x': 'Gender', 'y': 'Group Size'},
            title='Group Sizes by Gender'
        )
        
        st.plotly_chart(fig)
    
    # Racial bias
    if 'race' in bias_metrics:
        st.write("### Racial Bias")
        
        racial_bias = bias_metrics['race']
        racial_interp = bias_interpretations['race']
        
        # Create tabs for each race
        race_tabs = st.tabs(list(racial_bias.keys()))
        
        for i, (race, tab) in enumerate(zip(racial_bias.keys(), race_tabs)):
            with tab:
                race_metrics = racial_bias[race]
                race_interp = racial_interp[race]
                
                # Display interpretation
                st.write(f"**Bias Level**: {race_interp['bias_level']}")
                st.write(f"**Description**: {race_interp['description']}")
                
                # Display metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Statistical Parity Difference", 
                        f"{race_metrics['statistical_parity_difference']:.4f}",
                        help="Measures the difference in selection rates between White and this racial group. Value of 0 means no bias."
                    )
                
                with col2:
                    st.metric(
                        "Disparate Impact", 
                        f"{race_metrics['disparate_impact']:.4f}",
                        help="Ratio of selection rates between this racial group and White. Value of 1 means no bias."
                    )
                
                # Display group sizes
                group_sizes = race_metrics['group_size']
                
                # Create a bar chart for group sizes
                fig = px.bar(
                    x=['White', race],
                    y=[group_sizes['privileged'], group_sizes['unprivileged']],
                    labels={'x': 'Race', 'y': 'Group Size'},
                    title=f'Group Sizes: White vs. {race}'
                )
                
                st.plotly_chart(fig)

def display_shap_explanation(image, explanation, task='gender'):
    """
    Display SHAP explanation in the Streamlit UI
    
    Args:
        image (PIL.Image): Original image
        explanation (PIL.Image): SHAP explanation visualization
        task (str): Task being explained (gender, age, race)
    """
    st.subheader(f"SHAP Explanation for {task.capitalize()} Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.write("Feature Importance Visualization")
        st.image(explanation, use_column_width=True)
    
    st.write("""
    The visualization shows which parts of the image were most important for the model's prediction.
    - **Red regions**: Features that pushed the prediction toward the predicted class
    - **Blue regions**: Features that pushed the prediction away from the predicted class
    """)

def display_privacy_metrics(encryption_benchmarks):
    """
    Display privacy metrics and encryption benchmarks in the Streamlit UI
    
    Args:
        encryption_benchmarks (dict): Dictionary of encryption benchmarks
    """
    st.subheader("Privacy Metrics")
    
    # Display encryption times
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Average Encryption Time", 
            f"{encryption_benchmarks['avg_encryption_time']:.4f} seconds",
            help="Average time to encrypt data"
        )
    
    with col2:
        st.metric(
            "Average Decryption Time", 
            f"{encryption_benchmarks['avg_decryption_time']:.4f} seconds",
            help="Average time to decrypt data"
        )
    
    # Plot encryption/decryption times
    fig = px.line(
        x=list(range(len(encryption_benchmarks['encryption_time']))),
        y=[encryption_benchmarks['encryption_time'], encryption_benchmarks['decryption_time']],
        labels={'x': 'Trial', 'y': 'Time (seconds)'},
        title='Encryption and Decryption Times',
        line_shape='linear'
    )
    
    fig.update_layout(legend_title_text='Operation')
    fig.data[0].name = 'Encryption'
    fig.data[1].name = 'Decryption'
    
    st.plotly_chart(fig)
    
    # Display information about homomorphic encryption
    st.write("""
    ### About Homomorphic Encryption
    
    Homomorphic encryption allows computations on encrypted data without decrypting it first.
    This ensures privacy by keeping the facial features encrypted while still allowing the model
    to make predictions.
    
    #### Benefits of Homomorphic Encryption in Facial Recognition:
    
    - **Privacy Preservation**: Personal biometric data remains encrypted
    - **Secure Computation**: Analysis can be performed without exposing sensitive information
    - **Reduced Bias Risk**: Minimizes potential for misuse of sensitive demographic information
    
    #### Technical Details:
    
    - **Encryption Scheme**: CKKS (Cheon-Kim-Kim-Song)
    - **Polynomial Modulus Degree**: {poly_modulus_degree}
    - **Vector Size**: {vector_size} elements
    """.format(
        poly_modulus_degree=8192,  # Example value
        vector_size=encryption_benchmarks['vector_size']
    ))

def get_table_download_link(df, filename="data.csv", text="Download data as CSV"):
    """
    Generate a download link for a DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame to download
        filename (str): Filename for the downloaded file
        text (str): Text to display for the download link
        
    Returns:
        str: HTML download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def image_to_base64(img):
    """
    Convert a PIL image to base64 for download
    
    Args:
        img (PIL.Image): Image to convert
        
    Returns:
        str: Base64 representation of the image
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def get_image_download_link(img, filename="image.png", text="Download image"):
    """
    Generate a download link for an image
    
    Args:
        img (PIL.Image): Image to download
        filename (str): Filename for the downloaded file
        text (str): Text to display for the download link
        
    Returns:
        str: HTML download link
    """
    img_str = image_to_base64(img)
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href 