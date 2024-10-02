import csv
import torch
import numpy as np
import pandas as pd
import time
import os
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

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

def process_csv(input_file, output_file):
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file)
    print(f"CSV file read. Total rows: {len(df)}")

    start_time = time.time()

    # Preprocess labels
    df['class_name'] = df['class_name'].apply(preprocess_label)

    # Generate embeddings for all preprocessed labels
    embeddings = get_distilbert_embeddings(df['class_name'].tolist())

    print("Embedding generation complete. Saving results...")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'class_name', 'labels'] + [f'embedding_{i}' for i in range(768)])
        for (_, row), embedding in tqdm(zip(df.iterrows(), embeddings), total=len(df), desc="Writing results", unit="row"):
            writer.writerow([row['image_path'], row['class_name'], row['labels']] + list(embedding))

    total_time = time.time() - start_time
    print(f"Processed {len(df)} images in {total_time:.2f} seconds.")
    print(f"Average time per image: {total_time/len(df):.2f} seconds")
    print(f"Results saved to {output_file}")

# Usage
input_file = os.path.join(os.getcwd(), "data", "wiki-art", "WikiArt.csv")
output_file = input_file.replace('.csv', '_distilbert_embeddings.csv')
print("Starting CSV processing...")
process_csv(input_file, output_file)
print("CSV processing complete.")