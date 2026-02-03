import os
import clip
import torch
from PIL import Image
# At the top with your imports
import os
import clip
import torch
from PIL import Image
import logging

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model")
CLIP_DIR = os.path.join(MODEL_DIR, "clip")
CLIP_MODEL_PATH = os.path.join(CLIP_DIR, "ViT-B-32-model.pt")
DOCKER_CLIP_DIR = "/app/model/clip"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Global model variables
model = None
preprocess = None

def load_clip_model():
    global model, preprocess
    
    try:
        # Try Docker path first
        if os.path.exists(DOCKER_CLIP_DIR):
            print(f"Loading CLIP model from Docker path: {DOCKER_CLIP_DIR}")
            os.environ["CLIP_MODEL_DIR"] = DOCKER_CLIP_DIR
            model, preprocess = clip.load("ViT-B/32", device=device)
        # Then try local path
        elif os.path.exists(CLIP_DIR):
            print(f"Loading CLIP model from local path: {CLIP_DIR}")
            os.environ["CLIP_MODEL_DIR"] = CLIP_DIR
            model, preprocess = clip.load("ViT-B/32", device=device)
        # Fall back to download
        else:
            print("Local model directories not found, downloading model...")
            model, preprocess = clip.load("ViT-B/32", device=device)
        
        print("CLIP model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return False

# Load the model when this module is imported
success = load_clip_model()
if not success:
    print("WARNING: Failed to load CLIP model. API endpoints that use this model will fail.")


def get_text_embedding(text_query):
    text = clip.tokenize([text_query]).to(device)
    # Generate the text embedding
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    text_embedding = text_embedding.squeeze().cpu().tolist()

    return text_embedding


def get_image_embedding(image):
    # print("Received image for embedding.")
    image = preprocess(Image.open(image)).unsqueeze(0).to(device)
    # Generate the image embedding
    with torch.no_grad():
        image_embedding = model.encode_image(image)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

    image_embedding = image_embedding.squeeze().cpu().tolist()
    # print(f"Generated embedding: {image_embedding}")
    return image_embedding