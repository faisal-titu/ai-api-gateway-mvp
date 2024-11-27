from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import torchvision.transforms as transforms
from opensearchpy import OpenSearch  # Optional for indexing in OpenSearch
import uuid
from typing import List, Optional
# import the generate_face_id function from the face_detection module
from dev.face_embedding.face_detection import generate_face_id
# import client from the open_search_client module
from dev.image_embedding.open_search_client import client

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
face_detection_model = MTCNN()
embedding_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def detect_faces_in_image(image: Image.Image):
    """
    Detect faces in a given image.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        List[dict]: A list of detected face details including bounding boxes and confidence scores.
    """
    boxes, probs = face_detection_model.detect(image)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No faces detected in the image.")

    detected_faces = []
    for box, prob in zip(boxes, probs):
        if prob >= 0.98:  # Confidence threshold
            x1, y1, x2, y2 = box
            detected_faces.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "probability": prob
            })

    return detected_faces


def generate_face_embeddings(image: Image.Image, detected_faces: List[dict], index_name: Optional[str] = None):
    """
    Generate embeddings for detected faces and optionally index them in OpenSearch.

    Args:
        image (PIL.Image.Image): The input image.
        detected_faces (List[dict]): A list of detected face bounding boxes.
        index_name (str, optional): The OpenSearch index name. Defaults to None.

    Returns:
        List[dict]: A list of embeddings and related details for the detected faces.
    """
    embeddings = []
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor()
    ])

    for face in detected_faces:
        x1, y1, x2, y2 = face["box"]
        face_crop = image.crop((x1, y1, x2, y2))
        face_tensor = transform(face_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = embedding_model(face_tensor).cpu().tolist()[0]

        face_id = generate_face_id()

        if index_name:
            client.index(
                index=index_name,
                body={
                    "my_vector": embedding,
                    "face_id": face_id
                }
            )

        embeddings.append({
            "face_id": face_id,
            "box": face["box"],
            "embedding": embedding
        })

    return embeddings
