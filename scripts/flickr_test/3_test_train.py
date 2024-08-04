import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models_and_loaders import *
import os
from torchvision.utils import save_image
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch.nn.functional as F

def evaluate_and_save_samples(model, device, epoch, output_dir, num_samples=10):
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased').to(device)
    
    with torch.no_grad():
        epoch_dir = os.path.join(output_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        for i in range(num_samples):
            latent = torch.randn(1, model.combined_latent_dim).to(device)
            output = model.decode(latent)
            
            img_path = os.path.join(epoch_dir, f'latent_{i:02d}_image.png')
            save_image(output.image.cpu(), img_path)
            
            embedding = output.embedding.unsqueeze(0)
            logits = distilbert_model(inputs_embeds=embedding).logits
            predicted_token_ids = torch.argmax(logits, dim=-1).squeeze()
            decoded_text = tokenizer.decode(predicted_token_ids)
            
            text_path = os.path.join(epoch_dir, f'latent_{i:02d}_text.txt')
            with open(text_path, 'w') as f:
                f.write(decoded_text)
            
            print(f"Sample {i} decoded text: {decoded_text}")
    
    print(f"Saved generated samples for epoch {epoch} in {epoch_dir}")

def evaluate_and_save_samples_noise(model, device, epoch, output_dir, val_loader, num_samples=10):
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    distilbert_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased').to(device)
    
    with torch.no_grad():
        epoch_dir = os.path.join(output_dir, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        real_batch = next(iter(val_loader))
        real_batch = ImageEmbeddingModelInput(
            image=real_batch.image.to(device),
            embedding=real_batch.embedding.to(device),
            filename=real_batch.filename
        )
        
        image_z = model.image_encoder(real_batch.image)
        text_z = model.text_encoder(real_batch.embedding)
        combined = model.combiner(image_z, text_z)
        mu, logvar = torch.chunk(combined, 2, dim=1)
        real_latents = model.reparameterize(mu, logvar)
        
        for i in range(num_samples):
            idx = torch.randint(0, real_latents.size(0), (1,)).item()
            latent_var = real_latents[idx:idx+1] + torch.randn_like(real_latents[idx:idx+1]) * 0.1
            
            output = model.decode(latent_var)
            
            print(f"Saving sample {i}...")
            
            img_path = os.path.join(epoch_dir, f'latent_{i:02d}_image.png')
            save_image(output.image.cpu(), img_path)
            
            embedding = output.embedding.unsqueeze(0)
            logits = distilbert_model(inputs_embeds=embedding).logits
            predicted_token_ids = torch.argmax(logits, dim=-1).squeeze()
            decoded_text = tokenizer.decode(predicted_token_ids)
            
            text_path = os.path.join(epoch_dir, f'latent_{i:02d}_text.txt')
            with open(text_path, 'w') as f:
                f.write(decoded_text)
            
            print(f"Sample {i} decoded text: {decoded_text}")
    
    print(f"Saved generated samples for epoch {epoch} in {epoch_dir}")

def train_autoencoder(model: ImageEmbeddingVAE,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      num_epochs: int,
                      learning_rate: float,
                      device: str = 'mps'):
    print(f"Starting training on device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    output_dir = 'generated_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    def kl_annealing_factor(epoch, step, steps_per_epoch, cycle_length=2):
        cycle_progress = (epoch + step / steps_per_epoch) % cycle_length
        return cycle_progress / cycle_length
    
    steps_per_epoch = len(train_loader)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Training')):
            kl_weight = kl_annealing_factor(epoch, batch_idx, steps_per_epoch)
            
            batch = ImageEmbeddingModelInput(
                image=batch.image.to(device),
                embedding=batch.embedding.to(device),
                filename=batch.filename
            )
            
            optimizer.zero_grad()
            recon_batch, latent, mu, logvar = model(batch)
            
            image_loss = F.mse_loss(recon_batch.image, batch.image, reduction='sum')
            
            embedding_scaling = 224 * 224 
            embedding_loss = F.cross_entropy(recon_batch.embedding, batch.embedding.argmax(dim=1))
            
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_loss = image_loss/embedding_scaling + embedding_loss + kl_weight * kl_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f" Batch {batch_idx + 1}/{len(train_loader)}: Loss: {total_loss.item():.4f}, KL weight: {kl_weight:.4f}")
                print(f"Image Loss: {image_loss.item():.4f}, Embedding Loss: {embedding_loss.item():.4f}")
                print(f"KL Loss: {kl_loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        
        print("Starting validation...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                batch = ImageEmbeddingModelInput(
                    image=batch.image.to(device),
                    embedding=batch.embedding.to(device),
                    filename=batch.filename
                )
                recon_batch, latent, mu, logvar = model(batch)
                
                image_loss = F.mse_loss(recon_batch.image, batch.image, reduction='sum')
                embedding_loss =  F.cross_entropy(recon_batch.embedding, batch.embedding.argmax(dim=1))
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = image_loss/embedding_scaling + embedding_loss + kl_weight * kl_loss
                val_loss += total_loss.item()
        
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f"Image Loss: {image_loss.item():.4f}, Embedding Loss: {embedding_loss.item():.4f}")
        print(f"KL Loss: {kl_loss.item():.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_vae_model.pth')
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')
        
        evaluate_and_save_samples(model, device, epoch + 1, output_dir)
        evaluate_and_save_samples_noise(model, device, epoch + 1, output_dir+"_noisy", val_loader)
    
    print('Training completed.')

if __name__ == "__main__":
    print("Starting main program")
    image_dir = "/Users/alexi/Documents/ArtxBiology2024/automatic_sentiment/data/flickr30k_images/flickr30k_images"
    
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(image_dir, batch_size=32, num_workers=4)
    
    print("Initializing model...")
    model = ImageEmbeddingVAE(
        image_latent_dim=256,
        text_latent_dim=128,
        combined_latent_dim=256,
        text_embedding_dim=768 # Assuming DistilBERT embeddings
    )
    
    print("Starting training...")
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=1e-3,
        device='mps' if torch.backends.mps.is_available() else 'cpu'
    )