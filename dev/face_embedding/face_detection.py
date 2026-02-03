import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import uuid
import cv2
import numpy as np

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

def detect_faces_and_save(image_path: str, crop_folder: str, margin: int = 25):
    """Detects faces and saves them with image_id_face_id naming pattern"""
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get face bounding boxes and probabilities
        boxes, probs = face_detection_model.detect(image)
        cropped_faces = []
        
        # Image ID based on original image name
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob >= 0.98:  # Only process faces with high confidence
                    # Enlarge the bounding box to add a margin around the face
                    x1, y1, x2, y2 = box
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin - 10)
                    x2 = min(image.width, x2 + margin)
                    y2 = min(image.height, y2 + margin)

                    # Crop face using the enlarged bounding box
                    cropped_face = image.crop((x1, y1, x2, y2))
                    
                    # Save the cropped face with a unique face ID
                    face_id = generate_face_id()
                    cropped_face_path = os.path.join(crop_folder, f"{image_id}_{face_id}.jpg")
                    cropped_face.save(cropped_face_path)
                    cropped_faces.append(cropped_face_path)
        
        return cropped_faces
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

def align_face(img, landmark, target_size=160):
    """Align face based on landmarks (eyes, nose, etc.)"""
    try:
        # Standard points for alignment (left eye, right eye)
        left_eye = landmark[0]
        right_eye = landmark[1]
        
        # Calculate angle for rotation
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Ensure coordinates are integers for center calculation
        center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_array = np.array(img)
        aligned = cv2.warpAffine(img_array, M, (img.width, img.height))
        
        # Crop to standard size - ensure all coordinates are integers
        face_width = int(np.linalg.norm([dX, dY]) * 2.5)
        x1 = int(max(0, center[0] - face_width // 2))
        y1 = int(max(0, center[1] - face_width // 2))
        x2 = int(min(img.width, x1 + face_width))
        y2 = int(min(img.height, y1 + face_width))
        
        # Verify the coordinates are valid
        if x1 >= x2 or y1 >= y2:
            # Fallback to simple cropping if calculated coordinates are invalid
            return img.resize((target_size, target_size))
            
        cropped = aligned[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (target_size, target_size))
        
        return Image.fromarray(resized)
    except Exception as e:
        print(f"Error in align_face: {e}")
        # Return the original image resized if alignment fails
        return img.resize((target_size, target_size))