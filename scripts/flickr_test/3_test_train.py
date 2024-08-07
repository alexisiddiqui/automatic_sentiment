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
import time
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
                f.write(f"{decoded_text}\n")
                f.write(f"Latent: {latent.cpu().numpy().tolist()}")
            
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
            filename=real_batch.filename,
            image_masked=real_batch.image_masked.to(device),
            text_masked=real_batch.text_masked.to(device)
        )
        
        latents = model.encode(real_batch)
        
        for i in range(num_samples):
            idx = torch.randint(0, latents.size(0), (1,)).item()
            latent_var = latents[idx:idx+1] + torch.randn_like(latents[idx:idx+1]) * 0.1
            
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
                f.write(f"{decoded_text}\n")
                f.write(f"Latent: {latent_var.cpu().numpy().tolist()}")
            
            print(f"Sample {i} decoded text: {decoded_text}")
    
    print(f"Saved generated samples for epoch {epoch} in {epoch_dir}")

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def collect_latent_codes(model, loader, device, max_samples=1000):
    model.eval()
    latent_codes = []
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            if sample_count >= max_samples:
                break
            batch = ImageEmbeddingModelInput(
                image=batch.image.to(device),
                embedding=batch.embedding.to(device),
                filename=batch.filename,
                image_masked=batch.image_masked.to(device),
                text_masked=batch.text_masked.to(device)
            )
            _, latent, _, _ = model(batch)
            latent_codes.append(latent.cpu().numpy())
            sample_count += latent.shape[0]
    return np.concatenate(latent_codes)[:max_samples]

def plot_latent_space(latent_codes_dict, epoch, output_dir):
    plt.figure(figsize=(12, 8))
    
    all_latents = np.concatenate(list(latent_codes_dict.values()))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_latents)
    
    start = 0
    colors = ['r', 'g', 'b', 'orange']
    labels = ['Training', 'Validation', 'Eval Random', 'Eval Noise']
    
    for (key, latents), color, label in zip(latent_codes_dict.items(), colors, labels):
        end = start + len(latents)
        plt.scatter(pca_result[start:end, 0], pca_result[start:end, 1], c=color, label=label, alpha=0.6)
        start = end
    
    plt.title(f'Latent Space PCA - Epoch {epoch}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'latent_space_pca_epoch_{epoch}.png'))
    plt.close()

def train_autoencoder(model: ImageEmbeddingVAE,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      num_epochs: int,
                      learning_rate: float,
                      device: str = 'cuda'):
    print(f"Starting training on device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    output_dir = 'generated_samples'
    os.system(f"rm -rf {output_dir}")
    os.system(f"rm -rf {output_dir+'_noisy'}")
    os.makedirs(output_dir, exist_ok=True)
    
    def kl_annealing_factor(epoch, step, steps_per_epoch, cycle_length=2, max_kl_weight=0.25):
        cycle_progress = (epoch + step / steps_per_epoch) % cycle_length
        return max_kl_weight * min(1, cycle_progress / cycle_length)
    
    steps_per_epoch = len(train_loader)
    
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Training')):
            kl_weight = kl_annealing_factor(epoch, batch_idx, steps_per_epoch)
            # kl_weight = 0
            batch = ImageEmbeddingModelInput(
                image=batch.image.to(device),
                embedding=batch.embedding.to(device),
                filename=batch.filename,
                image_masked=batch.image_masked.to(device),
                text_masked=batch.text_masked.to(device)
            )
            
            optimizer.zero_grad()
            recon_batch, latent, mu, logvar = model(batch)
            
            image_loss = F.mse_loss(recon_batch.image, batch.image, reduction='mean')
            
            embedding_scaling = 1
            embedding_loss = embedding_scaling * F.cross_entropy(
                recon_batch.embedding.view(-1, recon_batch.embedding.size(-1)),
                batch.embedding.argmax(dim=-1).view(-1),
                reduction='mean'
            )
            
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_loss = 100*image_loss + embedding_loss + kl_weight * kl_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f" Batch {batch_idx + 1}/{len(train_loader)}: Loss: {total_loss.item():.4f}, KL weight: {kl_weight:.4f}")
                print(f"Image Loss: {image_loss.item():.4f}, Embedding Loss: {embedding_loss.item():.4f} KL Loss: {kl_loss.item():.4f}")


        
        print("Starting validation...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                batch = ImageEmbeddingModelInput(
                    image=batch.image.to(device),
                    embedding=batch.embedding.to(device),
                    filename=batch.filename,
                    image_masked=batch.image_masked.to(device),
                    text_masked=batch.text_masked.to(device)
                )
                recon_batch, latent, mu, logvar = model(batch)
                
                image_loss = F.mse_loss(recon_batch.image, batch.image, reduction='mean')
                embedding_loss = embedding_scaling * F.cross_entropy(
                    recon_batch.embedding.view(-1, recon_batch.embedding.size(-1)),
                    batch.embedding.argmax(dim=-1).view(-1),
                    reduction='mean'
                )
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = image_loss + embedding_loss + kl_weight * kl_loss
                val_loss += total_loss.item()
        

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f"Image Loss: {image_loss.item():.4f}, Embedding Loss: {embedding_loss.item():.4f} KL Loss: {kl_loss.item():.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_vae_model.pth')
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')
        
        evaluate_and_save_samples(model, device, epoch + 1, output_dir)
        evaluate_and_save_samples_noise(model, device, epoch + 1, output_dir+"_noisy", val_loader)

        # Collect latent codes (sample at most 1000 for each)
        train_latents = collect_latent_codes(model, train_loader, device, max_samples=1000)
        val_latents = collect_latent_codes(model, val_loader, device, max_samples=1000)
        
        # Collect latent codes from evaluate_and_save_samples
        random_latents = []
        with torch.no_grad():
            for _ in range(100):  # Generate 100 random samples
                latent = torch.randn(1, model.combined_latent_dim).to(device)
                random_latents.append(latent.cpu().numpy())
        random_latents = np.concatenate(random_latents)
        
        # Collect latent codes from evaluate_and_save_samples_noise
        noise_latents = []
        with torch.no_grad():
            real_batch = next(iter(val_loader))
            real_batch = ImageEmbeddingModelInput(
                image=real_batch.image.to(device),
                embedding=real_batch.embedding.to(device),
                filename=real_batch.filename,
                image_masked=real_batch.image_masked.to(device),
                text_masked=real_batch.text_masked.to(device)
            )
            latents = model.encode(real_batch)
            for _ in range(100):  # Generate 100 noisy samples
                idx = torch.randint(0, latents.size(0), (1,)).item()
                latent_var = latents[idx:idx+1] + torch.randn_like(latents[idx:idx+1]) * 0.1
                noise_latents.append(latent_var.cpu().numpy())
        noise_latents = np.concatenate(noise_latents)
        
        # Plot latent space
        latent_codes_dict = {
            'train': train_latents,
            'val': val_latents,
            'random': random_latents,
            'noise': noise_latents
        }
        plot_latent_space(latent_codes_dict, epoch + 1, output_dir)
        
    end_time = time.time()

    print(f"Total time taken for training: {end_time - start_time} seconds")
    print(f"Total time taken for training: {(end_time - start_time)/60} minutes")
    print('Training completed.')


if __name__ == "__main__":
    print("Starting main program")
    image_dir = os.getcwd()+"/data/flickr30k_images/flickr30k_images"
    
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(image_dir, batch_size=400, num_workers=20, mask_prob=0.3)
    
    print("Initializing model...")
    model = ImageEmbeddingVAE(
        image_latent_dim=256,
        text_latent_dim=512,
        combined_latent_dim=512,
        text_embedding_dim=768 # Assuming DistilBERT embeddings
    )
    
    print("Starting training...")
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=1e-3,
        device='cuda'
    )