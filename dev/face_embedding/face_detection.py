import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import uuid

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
face_detection_model = MTCNN(keep_all=True, min_face_size=30, device=device)

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
