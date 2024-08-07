
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from dataclasses import dataclass
from typing import Tuple
import random

@dataclass
class ImageEmbeddingModelInput:
    image: torch.Tensor
    embedding: torch.Tensor
    filename: str
    image_masked: bool
    text_masked: bool

class ImageEmbeddingDataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str, transform=None, mask_prob=0.5):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_path, header=0)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_prob = mask_prob

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
        
        image_masked = random.random() < self.mask_prob
        text_masked = random.random() < self.mask_prob
        
        if image_masked:
            image = torch.zeros_like(image)
        
        if text_masked:
            embedding = torch.zeros_like(embedding)

        # if both are masked, mask neither
        if image_masked and text_masked:
            image_masked = False
            text_masked = False
        
        
        return ImageEmbeddingModelInput(image=image, embedding=embedding, filename=image_name,
                                        image_masked=image_masked, text_masked=text_masked)

def collate_fn(batch):
    images = torch.stack([item.image for item in batch])
    embeddings = torch.stack([item.embedding for item in batch])
    filenames = [item.filename for item in batch]
    image_masked = torch.tensor([item.image_masked for item in batch])
    text_masked = torch.tensor([item.text_masked for item in batch])
    return ImageEmbeddingModelInput(image=images, embedding=embeddings, filename=filenames,
                                    image_masked=image_masked, text_masked=text_masked)

def create_dataloaders(image_dir: str, batch_size: int = 32, num_workers: int = 4, mask_prob: float = 0.1) -> Tuple[DataLoader, DataLoader]:
    _dir = os.path.dirname(image_dir)
    train_csv_path = os.path.join(_dir, "train_data.csv")
    val_csv_path = os.path.join(_dir, "val_data.csv")
    
    train_dataset = ImageEmbeddingDataset(image_dir, train_csv_path, mask_prob=mask_prob)
    val_dataset = ImageEmbeddingDataset(image_dir, val_csv_path, mask_prob=mask_prob)
    
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
        self.dropout = nn.Dropout(0.05)

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
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class ImageDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ImageDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 28 * 28)
        self.bn1 = nn.BatchNorm1d(128 * 28 * 28)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(self.bn1(self.fc(x)))
        x = x.view(x.size(0), 128, 28, 28)
        x = self.dropout(F.leaky_relu(self.bn2(self.deconv1(x)), 0.2))
        x = self.dropout(F.leaky_relu(self.bn3(self.deconv2(x)), 0.2))
        x = self.deconv3(x)
        return x
    
class TextDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(TextDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
    
    
class Combiner(nn.Module):
    def __init__(self, image_latent_dim, text_latent_dim, combined_latent_dim):
        super(Combiner, self).__init__()
        self.image_latent_dim = image_latent_dim
        self.text_latent_dim = text_latent_dim
        self.combined_latent_dim = combined_latent_dim

        # LSTM to process image and text latent vectors
        self.lstm = nn.LSTM(input_size=max(image_latent_dim, text_latent_dim),
                            hidden_size=combined_latent_dim,
                            num_layers=1,
                            batch_first=True)

        # Final linear layer to produce mu and logvar
        self.fc = nn.Linear(combined_latent_dim, 2 * combined_latent_dim)
        

    def forward(self, image_z, text_z, image_masked, text_masked):
        batch_size = image_z.size(0)
        device = image_z.device

        # Prepare sequences for LSTM
        sequences = []
        seq_lengths = []

        for i in range(batch_size):
            seq = []
            if not image_masked[i]:
                seq.append(F.pad(image_z[i], (0, self.combined_latent_dim - self.image_latent_dim)))
            if not text_masked[i]:
                seq.append(F.pad(text_z[i], (0, self.combined_latent_dim - self.text_latent_dim)))
            
            sequences.append(torch.stack(seq) if seq else torch.zeros(1, self.combined_latent_dim, device=device))
            seq_lengths.append(len(seq))

        # Pad sequences to the same length
        max_length = max(seq_lengths)
        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

        # Pack padded sequences for LSTM
        packed_sequences = torch.nn.utils.rnn.pack_padded_sequence(padded_sequences, seq_lengths, batch_first=True, enforce_sorted=False)

        # Process with LSTM
        _, (h_n, _) = self.lstm(packed_sequences)

        # Use the final hidden state
        combined = h_n.squeeze(0)

        # Final linear layer to produce mu and logvar
        x = self.fc(combined)

        return x

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
        
        combined = self.combiner(image_z, text_z, input_data.image_masked, input_data.text_masked)
        mu, logvar = torch.chunk(combined, 2, dim=1)
        
        latent = self.reparameterize(mu, logvar)
        
        reconstructed_image = self.image_decoder(latent)
        reconstructed_embedding = self.text_decoder(latent)
        
        return ImageEmbeddingModelInput(image=reconstructed_image, embedding=reconstructed_embedding, 
                                        filename=input_data.filename, 
                                        image_masked=input_data.image_masked, 
                                        text_masked=input_data.text_masked), latent, mu, logvar

    def encode(self, input_data: ImageEmbeddingModelInput):
        image_z = self.image_encoder(input_data.image)
        text_z = self.text_encoder(input_data.embedding)
        combined = self.combiner(image_z, text_z, input_data.image_masked, input_data.text_masked)
        mu, logvar = torch.chunk(combined, 2, dim=1)
        return self.reparameterize(mu, logvar)

    def decode(self, latent):
        reconstructed_image = self.image_decoder(latent)
        reconstructed_embedding = self.text_decoder(latent)
        return ImageEmbeddingModelInput(image=reconstructed_image, embedding=reconstructed_embedding, 
                                        filename="generated.jpg", image_masked=False, text_masked=False)
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