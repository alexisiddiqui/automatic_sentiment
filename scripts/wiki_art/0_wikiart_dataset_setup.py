import os
import subprocess
import kaggle
import zipfile

# Set the paths
download_path = os.path.join(os.getcwd(), 'raw_data', 'wiki_art')
zip_file_path = os.path.join(download_path, 'wikiart-gangogh-creating-art-gan.zip')
extract_path = os.path.join(os.getcwd(), 'data')

# Create the directories if they don't exist
print(f"Creating directories: {download_path} and {extract_path}")
os.makedirs(download_path, exist_ok=True)
os.makedirs(extract_path, exist_ok=True)

# Download the dataset (without unzipping)
print("Downloading the dataset from Kaggle...")
kaggle.api.dataset_download_files('ipythonx/wikiart-gangogh-creating-art-gan', path=download_path, unzip=False)
print(f"Dataset downloaded to {download_path}")

# Unzip the file to the correct folder
print(f"Unzipping the file {zip_file_path} to {extract_path}")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print(f"Files extracted to {extract_path}")

# # Optionally, remove the zip file after extraction
# print(f"Removing the zip file {zip_file_path}")
# os.remove(zip_file_path)
# print("Zip file removed")