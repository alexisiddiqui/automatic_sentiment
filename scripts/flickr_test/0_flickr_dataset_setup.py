import os 

import subprocess


# create kaggle account and get the api key
# save to ~/.kaggle/kaggle.json on linux/mac or C:\Users\<Windows-username>\.kaggle\kaggle.json on windows

import os
import kaggle
import zipfile

# Set the paths
download_path = os.path.join(os.getcwd(), 'raw_data', 'flickr_test')
zip_file_path = os.path.join(download_path, 'flickr-image-dataset.zip')
extract_path = os.path.join(os.getcwd(), 'data')

# Create the directories if they don't exist
os.makedirs(download_path, exist_ok=True)
os.makedirs(extract_path, exist_ok=True)

# Download the dataset (without unzipping)
kaggle.api.dataset_download_files('hsankesara/flickr-image-dataset', path=download_path, unzip=False)

# Unzip the file to the correct folder
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Optionally, remove the zip file after extraction
os.remove(zip_file_path)