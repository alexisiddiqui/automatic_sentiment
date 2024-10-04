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

# from torchvision.models import vgg16
from torchvision import transforms

# class VGGPerceptualLoss(torch.nn.Module):
#     def __init__(self, resize=True):
#         super(VGGPerceptualLoss, self).__init__()
#         vgg = vgg16(pretrained=True)
#         blocks = []
#         blocks.append(vgg.features[:4].eval())
#         blocks.append(vgg.features[4:9].eval())
#         blocks.append(vgg.features[9:16].eval())
        
#         for bl in blocks:
#             for p in bl.parameters():
#                 p.requires_grad = False
        
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.resize = resize
#         self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

#     def forward(self, input, target):
#         if input.shape[1] != 3:
#             input = input.repeat(1, 3, 1, 1)
#             target = target.repeat(1, 3, 1, 1)
#         input = (input - self.mean) / self.std
#         target = (target - self.mean) / self.std
#         if self.resize:
#             input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
#             target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
#         loss = 0.0
#         for block in self.blocks:
#             input = block(input)
#             target = block(target)
#             loss += F.mse_loss(input, target)
#         return loss

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
            torch.cuda.empty_cache()
    
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
            torch.cuda.empty_cache()
    print(f"Saved generated samples for epoch {epoch} in {epoch_dir}")

def train_autoencoder(model: ImageEmbeddingVAE,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      num_epochs: int,
                      learning_rate: float,
                      device: str = 'cuda'):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Starting training on device: {device}")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # perceptual_loss = VGGPerceptualLoss().to(device)
    
    best_val_loss = float('inf')
    output_dir = 'generated_samples'
    os.makedirs(output_dir, exist_ok=True)
    
    def kl_annealing_factor(epoch, step, steps_per_epoch, cycle_length=2):
        cycle_progress = (epoch + step / steps_per_epoch) % cycle_length
        return cycle_progress / cycle_length
    
    steps_per_epoch = len(train_loader)
    
    start_time = time.time()

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
            
            image_loss = 100 * F.mse_loss(recon_batch.image, batch.image, reduction='mean')
            perceptual_loss_value = 10* model.perceptual_loss(recon_batch.image, batch.image)
            
            embedding_scaling = 25
            embedding_loss = embedding_scaling * F.cross_entropy(recon_batch.embedding, batch.embedding.argmax(dim=1), reduction='mean')
            
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            total_loss = image_loss + perceptual_loss_value + embedding_loss + kl_weight * kl_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f" Batch {batch_idx + 1}/{len(train_loader)}: Loss: {total_loss.item():.4f}, KL weight: {kl_weight:.4f}")
                print(f"Image Loss: {image_loss.item():.4f}, Perceptual Loss: {perceptual_loss_value.item():.4f}, Embedding Loss: {embedding_loss.item():.4f}")
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
                
                image_loss = 100 * F.mse_loss(recon_batch.image, batch.image, reduction='mean')
                perceptual_loss_value = 10* model.perceptual_loss(recon_batch.image, batch.image)
                embedding_loss = embedding_scaling * F.cross_entropy(recon_batch.embedding, batch.embedding.argmax(dim=1), reduction='mean')
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = image_loss +  perceptual_loss_value + embedding_loss + kl_weight * kl_loss
                val_loss += total_loss.item()
            torch.cuda.empty_cache()
        
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f"Image Loss: {image_loss.item():.4f}, Perceptual Loss: {perceptual_loss_value.item():.4f}, Embedding Loss: {embedding_loss.item():.4f}")
        print(f"KL Loss: {kl_loss.item():.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_vae_model.pth')
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')
        
        evaluate_and_save_samples(model, device, epoch + 1, output_dir)
        evaluate_and_save_samples_noise(model, device, epoch + 1, output_dir+"_noisy", val_loader)
    
    end_time = time.time()

    print(f"Total time taken for training: {end_time - start_time} seconds")
    print(f"Total time taken for training: {(end_time - start_time)/60} minutes")

    print('Training completed.')

if __name__ == "__main__":
    print("Starting main program")

    flickr_image_dir = os.getcwd()+"/data/flickr30k_images/flickr30k_images"
    celeba_image_dir = os.getcwd()+"/data/img_align_celeba/img_align_celeba"
    wikiart_image_dir = os.getcwd()+"/data/wiki-art/wiki-art"
    image_dirs = [flickr_image_dir, celeba_image_dir, wikiart_image_dir]

    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(image_dirs, batch_size=100, num_workers=8)

    print("Initializing model...")
    model = ImageEmbeddingVAE(
        # text_latent_dim=256,
        # combined_latent_dim=256,
        # text_embedding_dim=768,
        # selected_layers=[1, 3, 6, 13],
        # num_combiner_layers=1  # Adjust this as needed
    )
    torch.compile(model)
    def print_model_parameters_sorted(model, top_k=10):
        param_counts = []
    
        def count_parameters(module, name=''):
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                params = sum(p.numel() for p in child.parameters(recurse=False))
                param_counts.append((full_name, params))
                count_parameters(child, full_name)

        count_parameters(model)
        param_counts.sort(key=lambda x: x[1], reverse=True)

        print(f"Top {top_k} parameter locations:")
        total_params = sum(count for _, count in param_counts)
        for name, count in param_counts[:top_k]:
            print(f"{name:<50}: {count:,} ({count/total_params:.2%})")

        print(f"\nTotal parameters: {total_params:,}")

        # Usage
    print_model_parameters_sorted(model)
    print_model_parameters_sorted(model.combiner)

    print("Starting training...")
    os.system("rm -r generated_samples")
    os.system("rm -r generated_samples_noisy")
    
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=1e-4,
        device='cuda'
    )



    os.system("python scripts/flickr_test/4_inference.py --image_dir test_data/skyscrapers --text_list 'fantastic' 'who' ")