import os
import clip
import torch
from opensearchpy.helpers import bulk
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from open_search_client import client


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def create_dataloader(image_dir, batch_size=20, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(image_dir, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    return dataloader

def prepare_embeddings(dataloader, index_name):

    actions = [] 
    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        with torch.no_grad():
            image_embeddings = model.encode_image(images)

        image_embeddings_list = image_embeddings.cpu().tolist()

        for j, image_embedding in enumerate(image_embeddings_list):
            id = i * len(images) + j

            # Get the file path for the current image
            file_path = dataloader.dataset.samples[i * len(images) + j][0]
            # Use the file name without the .jpg extension as the image_id
            image_id = os.path.splitext(os.path.basename(file_path))[0]

            action = {
                "_index": index_name,
                "_id": id,
                "_source": {
                    "my_vector": image_embedding,
                    "image_id": image_id 
                }
            }
            actions.append(action)
    return actions


def apply_bulk_indexing(client, actions):
    response = bulk(client, actions)
    return response