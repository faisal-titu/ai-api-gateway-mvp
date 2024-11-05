import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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