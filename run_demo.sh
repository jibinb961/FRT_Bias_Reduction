#!/bin/bash
# Script to run the FRT Bias Reduction demo

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if models directory exists, if not create it
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
fi

# Check if dummy model exists, if not create it
if [ ! -f "models/dummy_model.pth" ]; then
    echo "Creating dummy model for testing..."
    python scripts/prepare_sample_data.py --output_dir sample_data --model_dir models
fi

# Run the Streamlit app
echo "Running the Streamlit app..."
streamlit run app.py

# Deactivate virtual environment when done
deactivate 