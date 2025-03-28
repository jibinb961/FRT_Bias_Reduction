import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64

def display_bias_metrics(bias_metrics, interpretations=None):
    """
    Display bias metrics in a user-friendly way
    
    Args:
        bias_metrics (dict): Dictionary of bias metrics
        interpretations (dict, optional): Dictionary of bias interpretations
    """
    st.subheader("Bias Metrics")
    
    # Handle no metrics
    if not bias_metrics:
        st.info("No bias metrics available. Please run bias detection first.")
        return
    
    # Gender bias
    if 'gender' in bias_metrics:
        st.markdown("### Gender Bias")
        
        gender_metrics = bias_metrics['gender']
        
        # Check if we have insufficient data
        if gender_metrics.get('insufficient_data', False):
            st.warning("""
            **Insufficient data for reliable gender bias metrics.**
            
            The sample size for one or more gender groups is too small. 
            Add more diverse images to your sample_data directory for better metrics.
            """)
            
            # Still show group sizes
            group_sizes = gender_metrics['group_size']
            st.write(f"Male group size: {group_sizes['privileged']}")
            st.write(f"Female group size: {group_sizes['unprivileged']}")
            
            # Show base rates if available
            if 'base_rates' in gender_metrics:
                base_rates = gender_metrics['base_rates']
                st.write(f"Male positive rate: {base_rates['privileged']:.4f}")
                st.write(f"Female positive rate: {base_rates['unprivileged']:.4f}")
                
            return
        
        # Get metrics
        spd = gender_metrics['statistical_parity_difference']
        di = gender_metrics['disparate_impact']
        
        # Format values for display
        spd_formatted = f"{spd:.4f}"
        
        # Handle potential inf or None values in disparate impact
        if di is None:
            di_formatted = "N/A (insufficient data)"
        elif np.isinf(di):
            di_formatted = "∞ (division by zero)"
        else:
            di_formatted = f"{di:.4f}"
        
        # Show group sizes
        group_sizes = gender_metrics['group_size']
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Statistical Parity Difference", spd_formatted, 
                      delta=None,
                      delta_color="inverse")
            st.write("SPD measures the difference in positive outcome rates between groups.")
            st.write("Target: Close to 0 (range: -1 to 1)")
            
            # Show base rates if available
            if 'base_rates' in gender_metrics:
                base_rates = gender_metrics['base_rates']
                st.write(f"Male positive rate: {base_rates['privileged']:.4f}")
                st.write(f"Female positive rate: {base_rates['unprivileged']:.4f}")
            
            st.write(f"Male group size: {group_sizes['privileged']}")
            st.write(f"Female group size: {group_sizes['unprivileged']}")
            
        with col2:
            st.metric("Disparate Impact", di_formatted,
                      delta=None,
                      delta_color="off")
            st.write("DI measures the ratio of positive outcome rates between groups.")
            st.write("Target: Close to 1.0 (above 0.8 is typically acceptable)")
            
            # Show interpretation if available
            if interpretations and 'gender' in interpretations:
                gender_interp = interpretations['gender']
                bias_level = gender_interp['bias_level']
                description = gender_interp['description']
                
                if bias_level == "Low":
                    st.success(f"Bias Level: {bias_level}")
                elif bias_level == "Moderate":
                    st.warning(f"Bias Level: {bias_level}")
                else:
                    st.error(f"Bias Level: {bias_level}")
                
                st.write(description)
    
    # Race bias
    if 'race' in bias_metrics:
        st.markdown("### Racial Bias")
        
        racial_bias = bias_metrics['race']
        all_insufficient = True
        
        for race, metrics in racial_bias.items():
            st.markdown(f"#### {race}")
            
            # Check if we have insufficient data for this race
            if metrics.get('insufficient_data', False):
                st.warning(f"""
                **Insufficient data for reliable bias metrics for {race}.**
                
                The sample size for this race is too small. 
                Add more images of {race} individuals to your sample_data directory.
                """)
                
                # Still show group sizes
                group_sizes = metrics['group_size']
                st.write(f"Other races group size: {group_sizes['privileged']}")
                st.write(f"{race} group size: {group_sizes['unprivileged']}")
                
                st.markdown("---")
                continue
                
            all_insufficient = False
            
            # Get metrics
            spd = metrics['statistical_parity_difference']
            di = metrics['disparate_impact']
            
            # Format values for display
            spd_formatted = f"{spd:.4f}"
            
            # Handle potential None or inf values in disparate impact
            if di is None:
                di_formatted = "N/A (insufficient data)"
            elif np.isinf(di):
                di_formatted = "∞ (division by zero)"
            else:
                di_formatted = f"{di:.4f}"
            
            # Show group sizes
            group_sizes = metrics['group_size']
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Statistical Parity Difference", spd_formatted, 
                          delta=None,
                          delta_color="inverse")
                
                # Show base rates if available
                if 'base_rates' in metrics:
                    base_rates = metrics['base_rates']
                    st.write(f"Other races positive rate: {base_rates['privileged']:.4f}")
                    st.write(f"{race} positive rate: {base_rates['unprivileged']:.4f}")
                
                st.write(f"Other races group size: {group_sizes['privileged']}")
                st.write(f"{race} group size: {group_sizes['unprivileged']}")
                
            with col2:
                st.metric("Disparate Impact", di_formatted,
                          delta=None,
                          delta_color="off")
                
                # Show interpretation if available
                if interpretations and 'race' in interpretations and race in interpretations['race']:
                    race_interp = interpretations['race'][race]
                    bias_level = race_interp['bias_level']
                    description = race_interp['description']
                    
                    if bias_level == "Low":
                        st.success(f"Bias Level: {bias_level}")
                    elif bias_level == "Moderate":
                        st.warning(f"Bias Level: {bias_level}")
                    else:
                        st.error(f"Bias Level: {bias_level}")
                    
                    st.write(description)
            
            # Add a note if metrics seem problematic
            if di is None or np.isinf(di) or abs(spd) > 0.8:
                st.info(f"""
                Note: The metrics for {race} may be affected by small sample size, class imbalance, or division by zero. 
                Consider adding more diverse sample images to get more accurate bias measurements.
                """)
            
            st.markdown("---")
        
        if all_insufficient:
            st.error("""
            **All racial groups have insufficient data for reliable bias metrics.**
            
            Please add more diverse images to your sample_data directory covering different racial groups.
            """)

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