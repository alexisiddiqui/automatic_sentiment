import csv
import torch
import numpy as np
import pandas as pd
import time
import os
from transformers import DistilBertTokenizer, DistilBertModel

print("CUDA Available:", torch.cuda.is_available())
print("MPS Available:", torch.backends.mps.is_available())

# Use MPS if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading DistilBERT model and tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
model.eval()
print("DistilBERT model and tokenizer loaded.")

def preprocess_celeba_attributes(row):
    # Get positive attributes (value == 1)
    positive_attrs = [col for col, val in row.items() if val == 1]
    # Join positive attributes with spaces
    # print(positive_attrs)
    return ' '.join(positive_attrs)

def get_distilbert_embeddings(texts, batch_size=128):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.extend(embeddings)
    
    return all_embeddings

def process_csv(input_file, output_file):
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"CSV file read. Total rows: {len(df)}")
    
    start_time = time.time()
    
    # Preprocess attributes
    attribute_columns = df.columns[1:]  # Exclude the image_id column
    df['attributes'] = df[attribute_columns].apply(preprocess_celeba_attributes, axis=1)
    
    # Generate embeddings for all preprocessed attributes
    embeddings = get_distilbert_embeddings(df['attributes'].tolist())
    
    print("Embedding generation complete. Saving results...")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name'] + [f'embedding_{i}' for i in range(768)])
        for image_id, embedding in zip(df['image_id'], embeddings):
            writer.writerow([image_id] + list(embedding))
    
    total_time = time.time() - start_time
    print(f"Processed {len(df)} images in {total_time:.2f} seconds.")
    print(f"Average time per image: {total_time/len(df):.2f} seconds")
    print(f"Results saved to {output_file}")

# Usage
input_file = os.path.join(os.getcwd(), "data", "img_align_celeba", "list_attr_celeba.csv")
output_file = input_file.replace('.csv', '_distilbert_embeddings_positive_only.csv')

print("Starting CSV processing...")
process_csv(input_file, output_file)
print("CSV processing complete.")