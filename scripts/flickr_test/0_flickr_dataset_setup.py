import os 

import subprocess


path = "/Users/alexi/Documents/ArtxBiology2024/automatic_sentiment/raw_data/flickr_test/flickr_dataset.zip"

# Unzip the file
subprocess.run(["unzip", path, "-d", "/Users/alexi/Documents/ArtxBiology2024/automatic_sentiment/data"])