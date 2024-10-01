import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models_and_loaders import ImageEmbeddingVAE, ImageEmbeddingModelInput
from transformers import DistilBertTokenizer, DistilBertModel
from datetime import datetime

def load_model(model_path, device):

    if model_path is None:
        model_path = "best_vae_model.pth"


    model = ImageEmbeddingVAE(
        # text_latent_dim=256,
        # combined_latent_dim=256,
        # text_embedding_dim=768,
        # selected_layers=[3, 6, 13],
        # num_combiner_layers=1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def process_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def get_text_embedding(text, tokenizer, text_model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the VAE model
    model = load_model(args.model_path, device)
    
    # Set up image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Set up text embedding model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    
    output_dir = args.output_dir

    if output_dir is None:
        output_dir = os.path.join("inference_output", os.path.basename(args.image_dir))
    if os.path.exists(output_dir):
        os.system(f"rm -r {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Create the output directory
    output_dir = os.path.join(args.output_dir, os.path.basename(args.image_dir))
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image and text combination
    for image_name in tqdm(os.listdir(args.image_dir), desc="Processing images"):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        image_path = os.path.join(args.image_dir, image_name)
        image_tensor = process_image(image_path, transform).to(device)
        
        for text in args.text_list:
            text_embedding = get_text_embedding(text, tokenizer, text_model, device)
            
            # Perform inference
            with torch.no_grad():
                input_data = ImageEmbeddingModelInput(image=image_tensor, embedding=text_embedding, filename=image_name)
                output, _, _, _ = model(input_data)
            
            # Save the output image
            output_image = output.image.squeeze().cpu().permute(1, 2, 0).numpy()
            output_image = (output_image * 255).clip(0, 255).astype('uint8')
            output_image = Image.fromarray(output_image)
            
            # Create a subfolder for each text prompt
            text_folder = os.path.join(output_dir, text.replace(" ", "_"))
            os.makedirs(text_folder, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create output filename with timestamp
            output_filename = f"{os.path.splitext(image_name)[0]}_{text.replace(' ', '_')}_{timestamp}.png"
            output_path = os.path.join(text_folder, output_filename)
            
            output_image.save(output_path)
    
    print(f"Inference completed. Results saved in {output_dir}")

if __name__ == "__main__":

    # check that the name of the cwd folder is correct
    repo_name = "automatic_sentiment"
    cwd = os.getcwd().split("/")[-1]

    assert cwd == repo_name, f"Current working directory is {cwd} but should be {repo_name}. Please change the directory to the root of the repository."


    parser = argparse.ArgumentParser(description="Perform inference on images with text prompts")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--text_list", nargs='+', required=True, help="List of text prompts")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the trained VAE model - default: best_vae_model.pth")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Directory to save inference results")

    args = parser.parse_args()
    main(args)


    # python inference_script.py --image_dir /path/to/images --text_list "prompt1" "prompt2" "prompt3" --model_path /path/to/best_vae_model.pth --output_dir /path/to/output
    