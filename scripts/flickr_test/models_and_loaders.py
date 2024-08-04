

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ImageEmbeddingModelInput:
    image: torch.Tensor
    embedding: torch.Tensor
    filename: str

class ImageEmbeddingDataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_path, header=0)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> ImageEmbeddingModelInput:
        row = self.data.iloc[idx]
        image_name = row['image_name']
        embedding = torch.tensor(row.filter(regex='^embedding_').values.astype('float32'))
        
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return ImageEmbeddingModelInput(image=image, embedding=embedding, filename=image_name)

def collate_fn(batch):
    images = torch.stack([item.image for item in batch])
    embeddings = torch.stack([item.embedding for item in batch])
    filenames = [item.filename for item in batch]
    return ImageEmbeddingModelInput(image=images, embedding=embeddings, filename=filenames)

# def create_dataloader(image_dir: str, csv_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
#     dataset = ImageEmbeddingDataset(image_dir, csv_path)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)


def create_dataloaders(image_dir: str,  batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:

    _dir = os.path.dirname(image_dir)
    train_csv_path = os.path.join(_dir, "train_data.csv")
    val_csv_path = os.path.join(_dir, "val_data.csv")

    
    train_dataset = ImageEmbeddingDataset(image_dir, train_csv_path)
    val_dataset = ImageEmbeddingDataset(image_dir, val_csv_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader


# # Usage example
# if __name__ == "__main__":
#     image_dir = "/Users/alexi/Documents/ArtxBiology2024/automatic_sentiment/data/flickr30k_images/flickr30k_images"
#     csv_path = "/Users/alexi/Documents/ArtxBiology2024/automatic_sentiment/data/flickr30k_images/results_distilbert_embeddings.csv"
    
#     dataloader = create_dataloader(image_dir, csv_path)
    
#     for batch in dataloader:
#         print(f"Batch size: {len(batch.filename)}")
#         print(f"Image shape: {batch.image.shape}")
#         print(f"Embedding shape: {batch.embedding.shape}")
#         print(f"First filename: {batch.filename[0]}")
#         break  # Just print the first batch for this example

import torch.nn as nn

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 28 * 28, latent_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.conv1(x)))
        x = self.dropout(F.relu(self.conv2(x)))
        x = self.dropout(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TextEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class Combiner(nn.Module):
    def __init__(self, image_latent_dim, text_latent_dim, combined_latent_dim):
        super(Combiner, self).__init__()
        self.fc = nn.Linear(image_latent_dim + text_latent_dim, combined_latent_dim * 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, image_z, text_z):
        combined = torch.cat([image_z, text_z], dim=1)
        return self.dropout(self.fc(combined))

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ImageDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 28 * 28)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.fc(x))
        x = x.view(x.size(0), 64, 28, 28)
        x = self.dropout(F.relu(self.deconv1(x)))
        x = self.dropout(F.relu(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        return x

class TextDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(TextDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class ImageEmbeddingVAE(nn.Module):
    def __init__(self, image_latent_dim, text_latent_dim, combined_latent_dim, text_embedding_dim):
        super(ImageEmbeddingVAE, self).__init__()
        self.image_encoder = ImageEncoder(image_latent_dim)
        self.text_encoder = TextEncoder(text_embedding_dim, text_latent_dim)
        self.combiner = Combiner(image_latent_dim, text_latent_dim, combined_latent_dim)
        self.image_decoder = ImageDecoder(combined_latent_dim)
        self.text_decoder = TextDecoder(combined_latent_dim, text_embedding_dim)
        self.combined_latent_dim = combined_latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_data: ImageEmbeddingModelInput):
        image_z = self.image_encoder(input_data.image)
        text_z = self.text_encoder(input_data.embedding)
        
        combined = self.combiner(image_z, text_z)
        mu, logvar = torch.chunk(combined, 2, dim=1)
        
        latent = self.reparameterize(mu, logvar)
        
        reconstructed_image = self.image_decoder(latent)
        reconstructed_embedding = self.text_decoder(latent)
        
        return ImageEmbeddingModelInput(image=reconstructed_image, embedding=reconstructed_embedding, filename=input_data.filename), latent, mu, logvar

    def encode(self, input_data: ImageEmbeddingModelInput):
        image_z = self.image_encoder(input_data.image)
        text_z = self.text_encoder(input_data.embedding)
        combined = self.combiner(image_z, text_z)
        mu, logvar = torch.chunk(combined, 2, dim=1)
        return self.reparameterize(mu, logvar)

    def decode(self, latent):
        reconstructed_image = self.image_decoder(latent)
        reconstructed_embedding = self.text_decoder(latent)
        return ImageEmbeddingModelInput(image=reconstructed_image, embedding=reconstructed_embedding, filename="generated.jpg")
# # Example usage
# if __name__ == "__main__":
#     # Assuming image size is 224x224 and BERT embedding size is 768
#     image_latent_dim = 128
#     text_latent_dim = 64
#     combined_latent_dim = 256
#     text_embedding_dim = 768

#     model = Autoencoder(image_latent_dim, text_latent_dim, combined_latent_dim, text_embedding_dim)

#     # Create a dummy input
#     dummy_image = torch.randn(1, 3, 224, 224)
#     dummy_embedding = torch.randn(1, 768)
#     dummy_input = ImageEmbeddingModelInput(image=dummy_image, embedding=dummy_embedding, filename="dummy.jpg")

#     # Forward pass
#     reconstructed, latent = model(dummy_input)

#     print(f"Input image shape: {dummy_input.image.shape}")
#     print(f"Input embedding shape: {dummy_input.embedding.shape}")
#     print(f"Reconstructed image shape: {reconstructed.image.shape}")
#     print(f"Reconstructed embedding shape: {reconstructed.embedding.shape}")
#     print(f"Latent representation shape: {latent.shape}")
#     print(f"Reconstructed filename: {reconstructed.filename}")