import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def create_train_val_split(csv_path, n_clusters=500, val_size=0.2, random_state=42):
    # Load the data
    print("Loading data...")
    df = pd.read_csv(csv_path, header=0)
    
    # Separate filenames and embeddings
    filenames = df['image_name'].values
    embeddings = df.filter(regex='^embedding_').values
    
    print("Normalizing embeddings...")
    # Normalize the embeddings
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)
    
    print("Applying PCA...")
    # Apply PCA
    pca = PCA(n_components=100)
    embeddings_pca = pca.fit_transform(embeddings_normalized)
    
    print("Performing k-means clustering...")
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings_pca)
    
    print("Creating stratified split...")
    # Stratified split based on cluster labels
    train_idx, val_idx = train_test_split(
        np.arange(len(filenames)),
        test_size=val_size,
        stratify=cluster_labels,
        random_state=random_state
    )
    
    # Create train and validation dataframes
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    # Add cluster labels to the dataframes
    train_df['cluster'] = cluster_labels[train_idx]
    val_df['cluster'] = cluster_labels[val_idx]
    
    print("Visualizing the split...")
    # Visualize the split
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_pca[train_idx, 0], embeddings_pca[train_idx, 1], c="Green", marker='o', alpha=0.1, label='Train')
    plt.scatter(embeddings_pca[val_idx, 0], embeddings_pca[val_idx, 1], c="Orange", marker='s', alpha=0.1, label='Validation')
    plt.legend()
    plt.title('PCA visualization of train-validation split')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig('celeba_train_val_split_visualization_pca.png')
    plt.close()
    
    return train_df, val_df

# Usage
print("Starting the script...")
csv_path = os.path.join(os.getcwd(), "data", "img_align_celeba", "list_attr_celeba_distilbert_embeddings_positive_only.csv")
train_df, val_df = create_train_val_split(csv_path)

print("Saving the split datasets...")
train_df.to_csv(os.path.join(os.getcwd(), 'data', 'img_align_celeba', 'train_data.csv'), index=False)
val_df.to_csv(os.path.join(os.getcwd(), 'data', 'img_align_celeba', 'val_data.csv'), index=False)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print("Script completed successfully.")