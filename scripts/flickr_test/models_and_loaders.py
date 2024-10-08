

# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import pandas as pd
# import os
# from dataclasses import dataclass
# from typing import Tuple

# @dataclass
# class ImageEmbeddingModelInput:
#     image: torch.Tensor
#     embedding: torch.Tensor
#     filename: str

# IMAGE_SIZE = 224

# class ImageEmbeddingDataset(Dataset):
#     def __init__(self, image_dir: str, csv_path: str, transform=None):
#         self.image_dir = image_dir
#         self.data = pd.read_csv(csv_path, header=0)
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx) -> ImageEmbeddingModelInput:
#         row = self.data.iloc[idx]
#         image_name = row['image_name']
#         embedding = torch.tensor(row.filter(regex='^embedding_').values.astype('float32'))
        
#         image_path = os.path.join(self.image_dir, image_name)
#         image = Image.open(image_path).convert('RGB')
        
#         if self.transform:
#             image = self.transform(image)
        
#         return ImageEmbeddingModelInput(image=image, embedding=embedding, filename=image_name)

# def collate_fn(batch):
#     images = torch.stack([item.image for item in batch])
#     embeddings = torch.stack([item.embedding for item in batch])
#     filenames = [item.filename for item in batch]
#     return ImageEmbeddingModelInput(image=images, embedding=embeddings, filename=filenames)

# # def create_dataloader(image_dir: str, csv_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
# #     dataset = ImageEmbeddingDataset(image_dir, csv_path)
# #     return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

# def create_dataloaders(image_dir: str, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
#     _dir = os.path.dirname(image_dir)
#     train_csv_path = os.path.join(_dir, "train_data.csv")
#     val_csv_path = os.path.join(_dir, "val_data.csv")

#     # Update the transform to include resizing
#     transform = transforms.Compose([
#         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
    
#     train_dataset = ImageEmbeddingDataset(image_dir, train_csv_path, transform=transform)
#     val_dataset = ImageEmbeddingDataset(image_dir, val_csv_path, transform=transform)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    
#     return train_loader, val_loader
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class ImageEmbeddingModelInput:
    image: torch.Tensor
    embedding: torch.Tensor
    filename: str

IMAGE_SIZE = 224

class ImageEmbeddingDataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str, transform=None, labelled=True):
        self.image_dir = image_dir
        if labelled:
            self.data = pd.read_csv(csv_path, header=0)
        else:
            self.data = pd.DataFrame({'image_name': os.listdir(image_dir), 'embedding': [torch.zeros(768) for _ in range(len(os.listdir(image_dir)))]})
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(image_path)
            return None
        if self.transform:
            image = self.transform(image)
        return ImageEmbeddingModelInput(image=image, embedding=embedding, filename=image_name)

def collate_fn(batch):
    batch =  list(filter(lambda x: x is not None, batch))
    images = torch.stack([item.image for item in batch])
    embeddings = torch.stack([item.embedding for item in batch])
    filenames = [item.filename for item in batch]
    return ImageEmbeddingModelInput(image=images, embedding=embeddings, filename=filenames)
def create_dataloaders(
    image_dirs: List[str],
    csv_paths: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    if csv_paths is None:
        csv_paths = []
        for image_dir in image_dirs:
            dir_name = os.path.dirname(image_dir)
            csv_paths.append((
                os.path.join(dir_name, "train_data.csv"),
                os.path.join(dir_name, "val_data.csv")
            ))
    else:
        if len(csv_paths) != len(image_dirs):
            raise ValueError("The number of CSV paths must match the number of image directories")
        csv_paths = [(csv_path, csv_path.replace('train', 'val')) for csv_path in csv_paths]

    # Define separate transforms for training and validation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(p=0.01)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_datasets = []
    val_datasets = []
    for image_dir, (train_csv, val_csv) in zip(image_dirs, csv_paths):
        train_datasets.append(ImageEmbeddingDataset(image_dir, train_csv, transform=train_transform))
        val_datasets.append(ImageEmbeddingDataset(image_dir, val_csv, transform=val_transform))

    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        combined_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


# Usage example:
# image_dirs = ["/path/to/image/dir1", "/path/to/image/dir2", "/path/to/image/dir3"]
# csv_dir = "/path/to/csv/dir"
# train_loader, val_loader = create_dataloaders(image_dirs, csv_dir)

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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetV2Encoder(nn.Module):
    def __init__(self):
        super(MobileNetV2Encoder, self).__init__()
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
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

class ImageEncoder(nn.Module):
    def __init__(self, mobilenet_encoder, selected_layers):
        super(ImageEncoder, self).__init__()
        self.mobilenet_encoder = mobilenet_encoder
        self.selected_layers = selected_layers
        self.dropout = nn.Dropout(0.2)  # Increased dropout rate
        self.resize = transforms.Resize((224, 224))

    def forward(self, x):
        x = self.resize(x)
        features = self.mobilenet_encoder(x, self.selected_layers)
        return [self.dropout(F.relu(feature)) for feature in features]

class TextEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.dropout = nn.Dropout(0.2)  # Increased dropout rate

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class ConvFlowCombiner(nn.Module):
    def __init__(self, image_feature_shapes, text_latent_dim, combined_latent_dim, num_flow_layers=3):
        super(ConvFlowCombiner, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(shape[0], 32, kernel_size=3, padding=1)
            for shape in image_feature_shapes
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.total_conv_features = 32 * len(image_feature_shapes)
        self.feature_combiner = nn.Linear(self.total_conv_features + text_latent_dim, combined_latent_dim)
        self.flow_layers = nn.ModuleList([
            nn.Linear(combined_latent_dim, combined_latent_dim)
            for _ in range(num_flow_layers - 1)
        ])
        self.final_layer = nn.Linear(combined_latent_dim, combined_latent_dim * 2)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(0.2)  # Added dropout

    def forward(self, image_features, text_z):
        conv_outputs = [self.global_pool(F.relu(conv(feat))).squeeze(-1).squeeze(-1) 
                        for conv, feat in zip(self.conv_layers, image_features)]
        combined_image_features = torch.cat(conv_outputs, dim=1)
        combined_features = torch.cat([combined_image_features, text_z], dim=1)
        x = F.relu(self.feature_combiner(combined_features))
        for layer in self.flow_layers:
            x = self.dropout(F.relu(layer(x)))  # Applied dropout
        x = self.final_layer(x)
        return self.activation(x)

# class ImprovedImageDecoder(nn.Module):
#     def __init__(self, latent_dim, num_channels=3):
#         super(ImprovedImageDecoder, self).__init__()
#         self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
#         self.upsample_blocks = nn.ModuleList([
#             self._make_upsample_block(512, 256),
#             self._make_upsample_block(256, 128),
#             self._make_upsample_block(128, 64),
#             self._make_upsample_block(64, 32)
#         ])
#         self.final_conv = nn.Conv2d(32, num_channels, kernel_size=3, padding=1)
#         self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
#         self.lrelu = nn.LeakyReLU(0.2, inplace=True)
#         self.dropout = nn.Dropout(0.2)  # Added dropout

#     def _make_upsample_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.2)  # Added dropout to each upsample block
#         )

#     def forward(self, x):
#         x = self.dropout(self.fc(x))  # Applied dropout
#         x = x.view(x.size(0), 512, 7, 7)
#         for block in self.upsample_blocks:
#             x = block(x)
#         x = self.final_conv(x)
#         x = self.upsample(x)
#         return torch.tanh(x)

class ImprovedImageDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels=3):
        super(ImprovedImageDecoder, self).__init__()
        
        self.initial_features = 256
        self.fc = nn.Linear(latent_dim, self.initial_features * 7 * 7)
        
        self.upsample_blocks = nn.ModuleList([
            self._make_upsample_block(self.initial_features, self.initial_features // 2),
            self._make_upsample_block(self.initial_features // 2, self.initial_features // 4),
            self._make_upsample_block(self.initial_features // 4, self.initial_features // 8),
            self._make_upsample_block(self.initial_features // 8, self.initial_features // 16)
        ])
        
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(self.initial_features // 16)
            for _ in range(2)
        ])
        
        self.final_conv = nn.Conv2d(self.initial_features // 16, num_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.activation = nn.LeakyReLU(0.2)  # Removed inplace=True
        self.dropout = nn.Dropout(0.1)
        
    def _make_upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),  # Removed inplace=True
            nn.Dropout(0.1)
        )
    
    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),  # Removed inplace=True
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2)  # Removed inplace=True
        )

    def forward(self, x):
        x = self.dropout(self.activation(self.fc(x)))
        x = x.view(x.size(0), self.initial_features, 7, 7)
        
        for block in self.upsample_blocks:
            x = block(x)
        
        for residual_block in self.residual_blocks:
            residual = x
            x = residual_block(x)
            x = x + residual  # Changed from += to +
        
        x = self.final_conv(x)
        x = self.upsample(x)
        return torch.tanh(x)
    

class TextDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(TextDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.2)  # Increased dropout rate

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class ImageEmbeddingVAE(nn.Module):
    def __init__(self, text_latent_dim=256, combined_latent_dim=256, text_embedding_dim=768, 
                 selected_layers=list(range(1, 14, 2)), num_combiner_layers=1):
        super(ImageEmbeddingVAE, self).__init__()
        self.mobilenet_encoder = MobileNetV2Encoder()
        self.image_encoder = ImageEncoder(self.mobilenet_encoder, selected_layers)
        self.text_encoder = TextEncoder(text_embedding_dim, text_latent_dim)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
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

    def forward(self, input_data):
        image_features = self.image_encoder(input_data.image)
        text_z = self.text_encoder(input_data.embedding)
        combined = self.combiner(image_features, text_z)
        mu, logvar = torch.chunk(combined, 2, dim=1)
        latent = self.reparameterize(mu, logvar)
        reconstructed_image = self.image_decoder(latent)
        reconstructed_embedding = self.text_decoder(latent)
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