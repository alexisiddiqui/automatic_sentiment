import csv
import torch
import numpy as np
import pandas as pd
import time
import os
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from PIL import Image

print("CUDA Available:", torch.cuda.is_available())
print("MPS Available:", torch.backends.mps.is_available())
# Use MPS if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading DistilBERT model and tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
model.eval()
print("DistilBERT model and tokenizer loaded.")

def preprocess_label(label):
    # Remove 'painting' from label names
    label = label.replace('-painting', '')
    # Replace specific labels
    label_map = {
        'genre': 'paintings',
        'nude-nu': 'nude'
    }
    return label_map.get(label, label)

def get_distilbert_embeddings(texts, batch_size=128):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.extend(embeddings)
    return all_embeddings

def test_load_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')

        return True
    except:
        print(image_path)
        
        return False

def process_csv(input_file, output_file, image_dir):
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"CSV file read. Total rows: {len(df)}")
    
    start_time = time.time()
    
    # Preprocess labels and test-load images
    valid_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images", unit="image"):
        image_path = os.path.join(image_dir, row['image_name'])
        if test_load_image(image_path):
            row['class_name'] = preprocess_label(row['class_name'])
            valid_rows.append(row)
    
    # Create a new DataFrame with only valid rows
    df_valid = pd.DataFrame(valid_rows)
    print(f"Valid images: {len(df_valid)} out of {len(df)} total")
    
    # Generate embeddings for all preprocessed labels
    embeddings = get_distilbert_embeddings(df_valid['class_name'].tolist())
    
    print("Embedding generation complete. Saving results...")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'class_name', 'labels'] + [f'embedding_{i}' for i in range(768)])
        for (_, row), embedding in tqdm(zip(df_valid.iterrows(), embeddings), total=len(df_valid), desc="Writing results", unit="row"):
            writer.writerow([row['image_name'], row['class_name'], row['labels']] + list(embedding))
    
    total_time = time.time() - start_time
    print(f"Processed {len(df_valid)} valid images out of {len(df)} total in {total_time:.2f} seconds.")
    print(f"Average time per valid image: {total_time/len(df_valid):.2f} seconds")
    print(f"Results saved to {output_file}")

# Usage
input_file = os.path.join(os.getcwd(), "data", "wiki-art", "WikiArt.csv")
output_file = input_file.replace('.csv', '_distilbert_embeddings_valid.csv')
image_dir = os.path.join(os.getcwd(), "data", "wiki-art", "wiki-art")  # Adjust this path to your image directory

print("Starting CSV processing...")
process_csv(input_file, output_file, image_dir)
print("CSV processing complete.")