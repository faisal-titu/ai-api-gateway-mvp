import json
import os

import clip
import numpy as np
import torch
from opensearchpy.helpers import bulk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from dev.image_embedding.open_search_client import client

# Set up device and load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

MAX_CONTEXT_LENGTH = 77  # The max context length in terms of characters

import json
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """Custom Dataset for loading metadata."""
    
    def __init__(self, file_path, max_context_length=77):
        """
        Initializes the dataset by loading the JSON file and handling the max context length.
        
        Args:
            file_path (str): Path to the file containing the metadata in JSON lines format.
            max_context_length (int): Maximum allowed length for the `meta_data` text.
        """
        self.max_context_length = max_context_length
        self.metadata = self.load_file(file_path)
    
    def load_file(self, file_path):
        """Load JSON lines file and return a list of dictionaries."""
        json_objects = []
        with open(file_path, "r", encoding="utf-8") as json_file:
            for line in json_file:
                try:
                    # Attempt to parse each line as JSON and append it to the list
                    data = json.loads(line.strip())  # `strip()` to remove any leading/trailing whitespaces
                    # Truncate meta_data if it exceeds the max context length
                    data["meta_data"] = self.truncate_text(data["meta_data"])
                    json_objects.append(data)
                except json.JSONDecodeError:
                    # Handle any lines that cannot be parsed as JSON
                    print(f"Skipping malformed line: {line}")
                    continue
        print("File loading complete.")
        return json_objects

    def truncate_text(self, text):
        """Truncate text to the maximum allowed context length."""
        if len(text) > self.max_context_length:
            # print(f"Warning: Text truncated to {self.max_context_length} characters.")
            return text[:self.max_context_length]
        return text

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get the metadata for a specific index in the dataset."""
        # Return text_id and the possibly truncated meta_data at the given index
        return self.metadata[idx]["text_id"], self.metadata[idx]["meta_data"]


def create_text_dataloader(text_file_path, batch_size=32, num_workers=4):
    """
    Create a DataLoader for the TextDataset.
    Args:
        text_file_path (str): The file path to the JSON file.
        batch_size (int): The number of samples per batch.
        num_workers (int): The number of worker threads to load data.
    
    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    # Initialize the dataset
    dataset = TextDataset(text_file_path)
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return dataloader

def generate_text_embeddings(dataloader):
    """
    Generate embeddings for text data using the CLIP model.
    Args:
        dataloader (DataLoader): The DataLoader containing the text data.
    
    Returns:
        list: A list of dictionaries containing the embeddings and text metadata.
    """
    embeddings = []
    
    for text_ids, texts in tqdm(dataloader, desc="Generating text embeddings"):
        # Tokenize and truncate the text before passing to the model
        truncated_texts = [clip.tokenize([text])[0] for text in texts]
        
        # Move the tokenized text to the correct device (GPU or CPU)
        truncated_texts = torch.stack(truncated_texts).to(device)
        
        # Generate text embeddings with CLIP
        with torch.no_grad():
            text_embeddings = model.encode_text(truncated_texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        
        text_embeddings_list = text_embeddings.cpu().tolist()
        
        # Collect the embeddings along with their text IDs and content
        for i in range(len(text_ids)):
            embeddings.append({
                "text_id": text_ids[i],
                "embedding": text_embeddings_list[i],
                "text_content": texts[i]
            })
    
    return embeddings


def bulk_index_text_embeddings(client, index_name, embeddings):
    """Index text embeddings into OpenSearch in bulk."""
    actions = []
    for i, data in tqdm(enumerate(embeddings), total=len(embeddings), desc="Indexing text data to OpenSearch"):
        action = {
            "_index": index_name,
            "_id": f"{data['text_id']}_{i}",
            "_source": {
                "my_vector": data["embedding"],
                "text_id": data["text_id"],
                "meta_data": data["text_content"]
            }
        }
        actions.append(action)

    response = bulk(client, actions)
    return response

# Usage Example:
# dataloader = create_dataloader('/path/to/your/text_data.json', batch_size=32)
# embeddings = generate_text_embeddings(dataloader)
# response = bulk_index_text_embeddings(client, 'your_text_index', embeddings)
