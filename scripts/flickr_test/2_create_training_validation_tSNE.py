import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_train_val_split(csv_path, n_clusters=1000, val_size=0.2, random_state=42):
    # Load the data
    df = pd.read_csv(csv_path, header=0)
    
    # Separate filenames and embeddings
    filenames = df['image_name'].values
    embeddings = df.filter(regex='^embedding_').values

    # Normalize the embeddings
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)

    # Apply t-SNE
    tsne = TSNE(n_components=3, random_state=random_state, perplexity=50, metric='cosine')
    embeddings_tsne = tsne.fit_transform(embeddings_normalized)

    # Perform k-means clustering on t-SNE results
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings_tsne)

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

    # Visualize the split
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_tsne[train_idx, 0], embeddings_tsne[train_idx, 1], 
                          c="Green", marker='o', alpha=0.1, label='Train')
    plt.scatter(embeddings_tsne[val_idx, 0], embeddings_tsne[val_idx, 1], 
                c="Orange", marker='s', alpha=0.1, label='Validation')
    plt.colorbar(scatter)
    plt.legend()
    plt.title('t-SNE visualization of train-validation split')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig('train_val_split_visualization_tsne.png')
    plt.close()

    return train_df, val_df

# Usage
# Usage
csv_path = "/Users/alexi/Documents/ArtxBiology2024/automatic_sentiment/data/flickr30k_images/results_distilbert_embeddings.csv"
train_df, val_df = create_train_val_split(csv_path)

# Save the split datasets
train_df.to_csv('/Users/alexi/Documents/ArtxBiology2024/automatic_sentiment/data/flickr30k_images/train_data.csv', index=False)
val_df.to_csv('/Users/alexi/Documents/ArtxBiology2024/automatic_sentiment/data/flickr30k_images/val_data.csv', index=False)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")