#!/bin/bash

# Set the environment name
ENV_NAME="automatic_sentiment"

# Create a new conda environment with Python 3.11
conda create -n $ENV_NAME python=3.11 ipykernel -y

# Activate the new environment
source activate $ENV_NAME

# Install pip packages
    pip install numpy pandas scikit-learn matplotlib torch torchvision Pillow tqdm transformers nltk kaggle

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Print success message
echo "Conda environment '$ENV_NAME' has been created and packages have been installed."
echo "To activate this environment, use:"
echo "conda activate $ENV_NAME"