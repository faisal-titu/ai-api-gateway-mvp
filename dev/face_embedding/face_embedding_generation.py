from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
from opensearchpy.helpers import bulk
from tqdm import tqdm
from dev.face_embedding.face_detection import generate_face_id
from torch.utils.data import DataLoader


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def generate_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        face_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            embedding = embedding_model(face_tensor).cpu().tolist()[0]
        return embedding
    except Exception as e:
        print(f"Error generating embedding for {image_path}: {e}")
        return None



def generate_embeddings_and_index(client, face_crop_dir: str, index_name: str, dataloader: DataLoader):
    """
    Generate embeddings for cropped faces in batches and index them in OpenSearch.
    """
    actions = []

    # Iterate over batches of images
    for images, image_ids in tqdm(dataloader, desc="Generating embeddings"):
        images = images.to(device)
        
        try:
            # Generate embeddings for the entire batch
            with torch.no_grad():
                embeddings = embedding_model(images).cpu().tolist()

            # Collect actions for bulk indexing
            for image_id, embedding in zip(image_ids, embeddings):
                face_id = generate_face_id()  # Generate a new face_id for each face
                actions.append({
                    "_index": index_name,
                    "_id": image_id,
                    "_source": {
                        "my_vector": embedding,
                        "image_id": image_id,
                        "face_id": face_id
                    }
                })
        except Exception as e:
            print(f"Error processing batch: {e}")

    # If there are actions, perform bulk indexing
    if actions:
        response = bulk(client, actions)
        print(f"Indexed {len(actions)} embeddings successfully.")
        return response
    return None

