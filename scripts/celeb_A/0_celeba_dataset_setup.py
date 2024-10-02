import os 

import subprocess


# create kaggle account and get the api key
# save to ~/.kaggle/kaggle.json on linux/mac or C:\Users\<Windows-username>\.kaggle\kaggle.json on windows

import os
import kaggle
import zipfile

# Set the paths
download_path = os.path.join(os.getcwd(), 'raw_data', 'celeb_A')
zip_file_path = os.path.join(download_path, 'celeba-dataset.zip')
extract_path = os.path.join(os.getcwd(), 'data')

# Create the directories if they don't exist
os.makedirs(download_path, exist_ok=True)
os.makedirs(extract_path, exist_ok=True)
# kaggle datasets download -d jessicali9530/celeba-dataset
# Download the dataset (without unzipping)
# kaggle.api.dataset_download_files('jessicali9530/celeba-dataset', path=download_path, unzip=False)

# Unzip the file to the correct folder using the subprocess module
subprocess.run(["unzip", zip_file_path, "-d", extract_path])