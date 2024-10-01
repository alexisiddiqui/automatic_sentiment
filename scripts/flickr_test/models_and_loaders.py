

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

IMAGE_SIZE = 224

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

def create_dataloaders(image_dir: str, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    _dir = os.path.dirname(image_dir)
    train_csv_path = os.path.join(_dir, "train_data.csv")
    val_csv_path = os.path.join(_dir, "val_data.csv")

    # Update the transform to include resizing
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageEmbeddingDataset(image_dir, train_csv_path, transform=transform)
    val_dataset = ImageEmbeddingDataset(image_dir, val_csv_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    
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
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetV2Encoder(nn.Module):
    def __init__(self):
        super(MobileNetV2Encoder, self).__init__()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Remove the classifier
        self.mobilenet.classifier = nn.Identity()
        
        for param in self.mobilenet.parameters():
            param.requires_grad = False

    def forward(self, x, layers=None):
        features = []
        for i, layer in enumerate(self.mobilenet.features):
            x = layer(x)
            features.append(x)
            if layers is not None and i+1 in layers:
                return features
        return features


class MobileNetV2PerceptualLoss(nn.Module):
    def __init__(self):
        super(MobileNetV2PerceptualLoss, self).__init__()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in self.mobilenet.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        input_features = self.mobilenet(input)
        target_features = self.mobilenet(target)
        
        return F.mse_loss(input_features, target_features)
import torch
import torch.nn as nn
import torch.nn.functional as F



class ImageEncoder(nn.Module):
    def __init__(self, mobilenet_encoder, selected_layers):
        super(ImageEncoder, self).__init__()
        self.mobilenet_encoder = mobilenet_encoder
        self.selected_layers = selected_layers
        self.dropout = nn.Dropout(0.1)
        
        # Add a resize transform
        self.resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

    def forward(self, x):
        # Ensure input is the correct size
        x = self.resize(x)
        features = self.mobilenet_encoder(x, self.selected_layers)
        return [self.dropout(F.relu(feature)) for feature in features]
    

class TextEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class FlexibleMultiLayerFlowCombiner(nn.Module):
    def __init__(self, image_feature_shapes, text_latent_dim, combined_latent_dim, num_layers=3):
        super(FlexibleMultiLayerFlowCombiner, self).__init__()
        
        self.image_feature_sizes = [shape[0] * shape[1] * shape[2] for shape in image_feature_shapes]
        self.total_image_features = sum(self.image_feature_sizes)
        
        # Fixed flow layers
        self.flow_layers = nn.ModuleList([
            nn.Linear(self.total_image_features + text_latent_dim if i == 0 else combined_latent_dim, combined_latent_dim)
            for i in range(num_layers)
        ])
        
        self.combined_latent_dim = combined_latent_dim
        self.activation = nn.Tanh()

    def forward(self, image_features, text_z):
        # Flatten and concatenate all image features
        flat_features = [f.view(f.size(0), -1) for f in image_features]
        x = torch.cat(flat_features + [text_z], dim=1)
        
        # Apply flow layers
        for layer in self.flow_layers:
            x = F.relu(layer(x))
        
        return self.activation(x)
class ImageDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ImageDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.dropout = nn.Dropout(0.1)
        
        # Add a final resize to ensure output is correct size
        self.resize = transforms.Resize((224, 224))  # Assuming IMAGE_SIZE is 224

    def forward(self, x):
        x = self.dropout(self.fc(x))
        x = x.view(x.size(0), 256, 7, 7)
        x = self.dropout(F.relu(self.deconv1(x)))
        x = self.dropout(F.relu(self.deconv2(x)))
        x = self.dropout(F.relu(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        # Ensure output is the correct size
        return self.resize(x)

class ImprovedImageDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels=3):
        super(ImprovedImageDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels

        # Initial dense layer
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        
        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList([
            self._make_upsample_block(512, 256),
            self._make_upsample_block(256, 128),
            self._make_upsample_block(128, 64),
            self._make_upsample_block(64, 32)
        ])
        
        # Final convolution to get the desired number of channels
        self.final_conv = nn.Conv2d(32, num_channels, kernel_size=3, padding=1)
        
        # Ensure output size is 224x224
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
        # Activation functions
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def _make_upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Initial dense layer and reshape
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)
        
        # Apply upsampling blocks
        for block in self.upsample_blocks:
            x = block(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Ensure output size is 224x224
        x = self.upsample(x)
        
        # Apply tanh activation to ensure output is in [-1, 1] range
        x = torch.tanh(x)
                
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
class ConvFlowCombiner(nn.Module):
    def __init__(self, image_feature_shapes, text_latent_dim, combined_latent_dim, num_flow_layers=3):
        super(ConvFlowCombiner, self).__init__()
        
        # Convolutional layers for each image feature level
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(shape[0], 32, kernel_size=3, padding=1)
            for shape in image_feature_shapes
        ])
        
        # Global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Calculate the total size after convolutions and pooling
        self.total_conv_features = 32 * len(image_feature_shapes)
        
        # Linear layer to combine convoluted image features and text features
        self.feature_combiner = nn.Linear(self.total_conv_features + text_latent_dim, combined_latent_dim)
        
        # Flow layers
        self.flow_layers = nn.ModuleList([
            nn.Linear(combined_latent_dim, combined_latent_dim)
            for _ in range(num_flow_layers - 1)
        ])
        
        # Final layer to produce mu and logvar
        self.final_layer = nn.Linear(combined_latent_dim, combined_latent_dim * 2)
        
        self.activation = nn.Tanh()

    def forward(self, image_features, text_z):
        # Apply convolutions to image features
        conv_outputs = [self.global_pool(F.relu(conv(feat))).squeeze(-1).squeeze(-1) 
                        for conv, feat in zip(self.conv_layers, image_features)]
        
        # Concatenate all convolved and pooled features
        combined_image_features = torch.cat(conv_outputs, dim=1)
        
        # Concatenate with text features
        combined_features = torch.cat([combined_image_features, text_z], dim=1)
        
        # Initial combination
        x = F.relu(self.feature_combiner(combined_features))
        
        # Apply flow layers
        for layer in self.flow_layers:
            x = F.relu(layer(x))
        
        # Final layer to produce mu and logvar
        x = self.final_layer(x)
        
        return self.activation(x)

class ImageEmbeddingVAE(nn.Module):
    def __init__(self, text_latent_dim=256,
                  combined_latent_dim=256, 
                  text_embedding_dim=768, 
                  selected_layers=list(range(3, 14)),
                  num_combiner_layers=1
                  ):
        super(ImageEmbeddingVAE, self).__init__()
        self.mobilenet_encoder = MobileNetV2Encoder()
        self.image_encoder = ImageEncoder(self.mobilenet_encoder, selected_layers)
        self.text_encoder = TextEncoder(text_embedding_dim, text_latent_dim)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # Assuming IMAGE_SIZE is 224
            image_feature_shapes = [f.shape[1:] for f in self.image_encoder(dummy_input)]
        
        self.combiner = ConvFlowCombiner(
            image_feature_shapes, 
            text_latent_dim, 
            combined_latent_dim,
            num_flow_layers=num_combiner_layers
        )
        
        self.image_decoder = ImprovedImageDecoder(combined_latent_dim)
        self.text_decoder = TextDecoder(combined_latent_dim, text_embedding_dim)
        self.combined_latent_dim = combined_latent_dim
        
        self.perceptual_loss = MobileNetV2PerceptualLoss()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def forward(self, input_data):
    #     image_features = self.image_encoder(input_data.image)
    #     text_z = self.text_encoder(input_data.embedding)
        
    #     combined = self.combiner(image_features, text_z)
    #     mu, logvar = torch.chunk(combined, 2, dim=1)
        
    #     latent = self.reparameterize(mu, logvar)
        
    #     reconstructed_image = self.image_decoder(latent)
    #     reconstructed_embedding = self.text_decoder(latent)
        
        #     return ImageEmbeddingModelInput(image=reconstructed_image, embedding=reconstructed_embedding, filename=input_data.filename), latent, mu, logvar
    def forward(self, input_data):
        # print(f" Input image shape: {input_data.image.shape}")
        
        image_features = self.image_encoder(input_data.image)
        # print(f"Encoded image features shapes: {[f.shape for f in image_features]}")
        
        text_z = self.text_encoder(input_data.embedding)
        # print(f"Encoded text shape: {text_z.shape}")
        
        combined = self.combiner(image_features, text_z)
        mu, logvar = torch.chunk(combined, 2, dim=1)
        # print(f"Mu shape: {mu.shape}, Logvar shape: {logvar.shape}")
        
        latent = self.reparameterize(mu, logvar)
        # print(f"Latent shape: {latent.shape}")
        
        reconstructed_image = self.image_decoder(latent)
        # print(f"Reconstructed image shape: {reconstructed_image.shape}")
        
        reconstructed_embedding = self.text_decoder(latent)
        # print(f"Reconstructed embedding shape: {reconstructed_embedding.shape}")
        
        return ImageEmbeddingModelInput(image=reconstructed_image, embedding=reconstructed_embedding, filename=input_data.filename), latent, mu, logvar

    def encode(self, input_data: ImageEmbeddingModelInput):
        image_features = self.image_encoder(input_data.image)
        text_z = self.text_encoder(input_data.embedding)
        combined = self.combiner(image_features, text_z)
        mu, logvar = torch.chunk(combined, 2, dim=1)
        return self.reparameterize(mu, logvar)

    def decode(self, latent):
        reconstructed_image = self.image_decoder(latent)
        reconstructed_embedding = self.text_decoder(latent)
        return ImageEmbeddingModelInput(image=reconstructed_image, embedding=reconstructed_embedding, filename="generated.jpg")
    

# Example usage
if __name__ == "__main__":
    text_latent_dim = 64
    combined_latent_dim = 256
    text_embedding_dim = 768

    model = ImageEmbeddingVAE(text_latent_dim, combined_latent_dim, text_embedding_dim)

    # Create a dummy input
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_embedding = torch.randn(1, 768)
    dummy_input = ImageEmbeddingModelInput(image=dummy_image, embedding=dummy_embedding, filename="dummy.jpg")

    # Forward pass
    reconstructed, latent, mu, logvar = model(dummy_input)

    print(f"Input image shape: {dummy_input.image.shape}")
    print(f"Input embedding shape: {dummy_input.embedding.shape}")
    print(f"Reconstructed image shape: {reconstructed.image.shape}")
    print(f"Reconstructed embedding shape: {reconstructed.embedding.shape}")
    print(f"Latent representation shape: {latent.shape}")
    print(f"Reconstructed filename: {reconstructed.filename}")