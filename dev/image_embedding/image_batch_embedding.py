import os
import clip
import torch
from opensearchpy.helpers import bulk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .open_search_client import client

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image_id = os.path.splitext(os.path.basename(image_path))[0]  # Using file name as image ID
        return image, image_id, image_path

def create_dataloader(image_dir, batch_size=32, num_workers=4):
    dataset = ImageDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader



def generate_embeddings(dataloader):
    embeddings = []
    for images, image_ids, image_paths in dataloader:
        images = images.to(device)
        with torch.no_grad():
            image_embeddings = model.encode_image(images)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings_list = image_embeddings.cpu().tolist()

        # Collect embeddings with IDs and paths
        for i in range(len(image_ids)):
            embeddings.append({
                "image_id": image_ids[i],
                "embedding": image_embeddings_list[i],
                "image_path": image_paths[i]
            })
    return embeddings

def bulk_index_embeddings(client, index_name, embeddings):
    print("index_name: ", index_name)
    print("embeddings length: ", len(embeddings))
    print("client: ", client)
    actions = []
    for i, data in tqdm(enumerate(embeddings), total=len(embeddings), desc="Indexing to OpenSearch"):
        action = {
            "_index": index_name,
            "_id": i,  # Unique ID for each document, could use a more robust unique ID generator if needed
            "_source": {
                "my_vector": data["embedding"],
                "image_id": data["image_id"]
            }
        }
        actions.append(action)

    # Perform the bulk indexing operation
    response = bulk(client, actions)
    return response

