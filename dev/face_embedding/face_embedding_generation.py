from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
from opensearchpy.helpers import bulk
from tqdm import tqdm
from dev.face_embedding.face_detection import generate_face_id
from torch.utils.data import DataLoader
from torchvision import transforms
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def preprocess_face(img):
    """Apply consistent preprocessing to face images"""
    # Convert to tensor and normalize 
    if isinstance(img, Image.Image):
        img = transforms.ToTensor()(img)
    
    # Add batch dimension if needed
    if img.dim() == 3:
        img = img.unsqueeze(0)
        
    # Ensure values are in model's expected range
    img = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    return img.to(device)


def generate_embedding_for_single_image(image_path):
    """Generate embedding directly from an image path"""
    try:
        # Open the image
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            # Assume it's a file-like object
            image = Image.open(image_path)
            
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to expected input size
        image = image.resize((160, 160))
        
        # Convert to tensor with proper preprocessing
        tensor = preprocess_face(image)
        
        # Generate face embedding
        with torch.no_grad():
            embedding = embedding_model(tensor).cpu().tolist()[0]
            
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None



def generate_embeddings_and_index(client, face_crop_dir: str, index_name: str, dataloader: DataLoader):
    """
    Generate embeddings for cropped faces in batches and index them in OpenSearch.
    Uses original image filenames (without extension) as image_id.
    """
    actions = []
    indexing_failures = []
    total_embeddings_indexed = 0
    
    # Iterate over batches of images
    for batch in tqdm(dataloader, desc="Generating embeddings"):
        images, image_paths = batch
        images = images.to(device)
        
        try:
            # Generate embeddings for the entire batch
            with torch.no_grad():
                embeddings = embedding_model(images).cpu().tolist()

            # Collect actions for bulk indexing
            for image_path, embedding in zip(image_paths, embeddings):
                try:
                    # Extract original filename from the cropped face path
                    filename = os.path.basename(image_path)
                    
                    # Extract the original image name without extension
                    # If filename is like "person1_face_1.jpg", get "person1"
                    if '_face_' in filename:
                        original_image_name = filename.split('_face_')[0]
                    else:
                        # Otherwise just use filename without extension
                        original_image_name = os.path.splitext(filename)[0]
                    
                    # Generate a unique face_id
                    face_id = generate_face_id()
                    
                    actions.append({
                        "_index": index_name,
                        "_id": face_id,
                        "_source": {
                            "my_vector": embedding,
                            "image_id": original_image_name,  # Original image name without extension
                            "face_id": face_id
                        }
                    })
                    total_embeddings_indexed += 1
                    
                except Exception as e:
                    indexing_failures.append({
                        "image_name": os.path.basename(str(image_path)),
                        "face_id": None,
                        "error_message": str(e)
                    })
                    
        except Exception as e:
            print(f"Error processing batch: {e}")
            for image_path in image_paths:
                indexing_failures.append({
                    "image_name": os.path.basename(str(image_path)),
                    "face_id": None,
                    "error_message": f"Batch error: {str(e)}"
                })

    # If there are actions, perform bulk indexing
    if actions:
        try:
            success, failed = bulk(client, actions, stats_only=True)
            print(f"Indexed {success} embeddings successfully. Failed: {failed}")
        except Exception as e:
            print(f"Bulk indexing error: {e}")
            # Add all actions to failures since we can't determine individual failures
            for action in actions:
                indexing_failures.append({
                    "image_name": action["_source"]["image_id"],
                    "face_id": action["_source"]["face_id"],
                    "error_message": f"Bulk indexing error: {str(e)}"
                })
    
    return total_embeddings_indexed, indexing_failures