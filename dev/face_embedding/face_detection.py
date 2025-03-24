import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import uuid

# Path to models directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "model")
FACE_DIR = os.path.join(MODEL_DIR, "face")
mtcnn_path = os.path.join(FACE_DIR, "mtcnn_model.pt")

# Check if we're running in Docker
docker_mtcnn_path = "/app/model/face/mtcnn_model.pt"

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load face detection model
def load_face_detector():
    """Load face detector from local directory or download if not available"""
    # Check Docker path first
    if os.path.exists(docker_mtcnn_path):
        model_path = docker_mtcnn_path
        print(f"Loading face detector from Docker path: {model_path}")
    elif os.path.exists(mtcnn_path):
        model_path = mtcnn_path
        print(f"Loading face detector from local path: {model_path}")
    else:
        print("Local model not found, initializing MTCNN (will download weights)")
        return MTCNN(keep_all=True, min_face_size=30, device=device)
    
    # Initialize the model
    detector = MTCNN(keep_all=True, min_face_size=30, device=device)
    
    try:
        # Load saved weights
        saved_weights = torch.load(model_path, map_location=device)
        detector.pnet.load_state_dict(saved_weights['pnet'])
        detector.rnet.load_state_dict(saved_weights['rnet'])
        detector.onet.load_state_dict(saved_weights['onet'])
        print("Successfully loaded weights for face detector")
    except Exception as e:
        print(f"Error loading weights: {e}. Using default model.")
    
    return detector

# Load the model
face_detection_model = load_face_detector()


def generate_face_id(num_char=8):
    return uuid.uuid4().hex[:num_char]

def detect_and_crop_faces(image_path: str, crop_folder: str, margin: int = 25):
    """
    Detects faces in an image, crops them with a margin, and saves to a folder.
    """
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        boxes, probs = face_detection_model.detect(image)
        cropped_faces = []

        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob >= 0.98:
                    x1, y1, x2, y2 = box
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(image.width, x2 + margin)
                    y2 = min(image.height, y2 + margin)

                    cropped_face = image.crop((x1, y1, x2, y2))
                    face_id = generate_face_id()
                    cropped_face_path = os.path.join(crop_folder, f"{face_id}.jpg")
                    cropped_face.save(cropped_face_path)
                    cropped_faces.append(cropped_face_path)
        return cropped_faces
    except Exception as e:
        print(f"Error in detect_and_crop_faces: {e}")
        return []
