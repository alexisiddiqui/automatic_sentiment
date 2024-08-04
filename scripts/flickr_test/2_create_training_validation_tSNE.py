import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def create_train_val_split(csv_path, eps=0.5, min_samples=5, val_size=0.2, random_state=42):
    print("Starting the train-validation split process...")
    
    print("Loading data from CSV...")
    df = pd.read_csv(csv_path, header=0)
    print(f"Loaded {len(df)} rows of data.")
    
    print("Separating filenames and embeddings...")
    filenames = df['image_name'].values
    embeddings = df.filter(regex='^embedding_').values
    
    print("Normalizing embeddings...")
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)
    
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=3, random_state=random_state, perplexity=50, metric='cosine', n_jobs=-1)
    embeddings_tsne = tsne.fit_transform(embeddings_normalized)
    print("t-SNE completed.")
    
    print("Performing DBSCAN clustering...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    cluster_labels = dbscan.fit_predict(embeddings_tsne)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"DBSCAN completed. Found {n_clusters} clusters.")
    
    print("Performing stratified train-validation split...")
    train_idx, val_idx = train_test_split(
        np.arange(len(filenames)),
        test_size=val_size,
        stratify=cluster_labels,
        random_state=random_state
    )
    
    print("Creating train and validation dataframes...")
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    train_df['cluster'] = cluster_labels[train_idx]
    val_df['cluster'] = cluster_labels[val_idx]
    
    print("Generating visualization...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                          c=cluster_labels, cmap='viridis', marker='o', alpha=0.5)
    plt.colorbar(scatter)
    plt.scatter(embeddings_tsne[val_idx, 0], embeddings_tsne[val_idx, 1],
                facecolors='none', edgecolors='red', s=50, label='Validation')
    plt.legend()
    plt.title('DBSCAN clustering and train-validation split visualization')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig('train_val_split_visualization_dbscan.png')
    plt.close()
    print("Visualization saved as 'train_val_split_visualization_dbscan.png'")
    
    print("Train-validation split process completed.")
    return train_df, val_df

# Usage
print("Starting the script...")
csv_path = os.getcwd() + "/data/flickr30k_images/results_distilbert_embeddings.csv"
train_df, val_df = create_train_val_split(csv_path)

print("Saving the split datasets...")
train_df.to_csv(os.getcwd() + '/data/flickr30k_images/train_data.csv', index=False)
val_df.to_csv(os.getcwd() + '/data/flickr30k_images/val_data.csv', index=False)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Number of clusters: {len(set(train_df['cluster']))}")
print("Script execution completed.")