import csv
import torch
import numpy as np
import pandas as pd
import time
import re
import os
from transformers import DistilBertTokenizer, DistilBertModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.tag.mapping import tagset_mapping
import string

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

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

# Get the list of English stopwords
stop_words = set(stopwords.words('english'))

# Get the mapping from Penn Treebank tags to Universal tags
tag_mapping = tagset_mapping('en-ptb', 'universal')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text_no_num = re.sub(r'\d+', '', text_no_punct)
    
    # Tokenize the text
    words = word_tokenize(text_no_num)
    
    # Perform POS tagging with Penn Treebank tags
    penn_tags = pos_tag(words)
    
    # Convert Penn Treebank tags to Universal tags
    universal_tags = [(word, tag_mapping.get(tag, tag)) for word, tag in penn_tags]
    
    # Print the universal tags for debugging
    # print(f"Universal tags: {universal_tags}")
    
    # Remove words that are not emotive or descriptive
    relevant_words = [word for word, tag in universal_tags if tag not in ['CONJ', 'DET', 'ADP', 'PRT']]
    
    # Remove stopwords and single characters
    relevant_words = [word for word in relevant_words if word not in stop_words and len(word) > 1]
    
    # Join the words back into a string
    processed_text = ' '.join(relevant_words)

    # print(f"Original text: {text}")
    # print(f"Processed text: {processed_text}")

    if len(processed_text) == 0:
        # If the processed text is empty, return the original text without punctuation and numbers
        
        print(f"Returning text without punctuation and numbers: {text_no_num}")
        # print(f"Original text: {text}")
        # print(f"Penn Treebank tags: {penn_tags}")
        print(f"Universal tags: {universal_tags}")
    
        return text_no_num
    
    return processed_text


def get_distilbert_embeddings(texts, batch_size=128):
    all_token_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Preprocess and filter out invalid inputs
        valid_texts = []
        for text in batch_texts:
            if isinstance(text, str):
                valid_texts.append(preprocess_text(text))
            elif isinstance(text, (int, float)):
                valid_texts.append(preprocess_text(str(text)))
            else:
                print(f"Skipping invalid input: {text}")
        
        if not valid_texts:
            continue
        
        inputs = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get embeddings for all tokens
        token_embeddings = outputs.last_hidden_state.cpu().numpy()
        all_token_embeddings.extend(token_embeddings)
    
    return all_token_embeddings

def process_csv(input_file, output_file):
    print(f"Reading CSV file: {input_file}")
    df = pd.read_csv(input_file, sep='|', names=['image_name', 'comment_number', 'comment'], header=0)
    print(f"CSV file read. Total rows: {len(df)}")
    
    grouped = df.groupby('image_name')
    print(f"Total unique images: {len(grouped)}")
    
    results = []
    start_time = time.time()
    
    for i, (image_name, group) in enumerate(grouped):
        if i % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Processing image {i+1}/{len(grouped)} - Elapsed time: {elapsed_time:.2f} seconds")
            print(f"Average time per image: {elapsed_time/(i+1):.2f} seconds")
            print(f"Estimated remaining time: {(len(grouped) - i - 1) * elapsed_time/(i+1):.2f} seconds")
        
        comments = group['comment'].tolist()
        
        # Generate embeddings for all comments in this group at once
        token_embeddings = get_distilbert_embeddings(comments)
        
        # Pool embeddings (using mean pooling across all tokens of all comments)
        pooled_embedding = np.mean(np.concatenate(token_embeddings, axis=0), axis=0)
        
        results.append({
            'image_name': image_name,
            'embedding': pooled_embedding
        })
    
    print("Embedding generation complete. Saving results...")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name'] + [f'embedding_{i}' for i in range(768)])
        for result in results:
            writer.writerow([result['image_name']] + list(result['embedding']))
    
    total_time = time.time() - start_time
    print(f"Processed {len(results)} images in {total_time:.2f} seconds.")
    print(f"Average time per image: {total_time/len(results):.2f} seconds")
    print(f"Results saved to {output_file}")

# Usage
input_file = os.getcwd()+"/data/flickr30k_images/results.csv"
output_file = input_file.replace('.csv', '_distilbert_verb_adj_embeddings.csv')

print("Starting CSV processing...")
process_csv(input_file, output_file)
print("CSV processing complete.")