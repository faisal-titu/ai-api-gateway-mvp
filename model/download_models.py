import os
import torch
import clip
from PIL import Image
import logging
from facenet_pytorch import MTCNN, InceptionResnetV1

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create model directory
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(MODEL_DIR, exist_ok=True)
logger.info(f"Using model directory: {MODEL_DIR}")

# Create directory structure
CLIP_DIR = os.path.join(MODEL_DIR, "clip")
FACE_DIR = os.path.join(MODEL_DIR, "face")
os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)

def download_clip_model():
    """Download CLIP model and save to local directory"""
    logger.info("Downloading CLIP model (ViT-B/32)...")
    
    # Set environment variable to change CLIP's download location
    os.environ["CLIP_MODEL_DIR"] = CLIP_DIR
    
    # Force download by loading the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=CLIP_DIR)
    
    # Save model state_dict explicitly
    torch.save(model.state_dict(), os.path.join(CLIP_DIR, "ViT-B-32-model.pt"))
    
    logger.info(f"CLIP model downloaded and saved to {CLIP_DIR}")
    return model, preprocess

def download_face_detection_model():
    """Download facenet_pytorch MTCNN model and save to local directory"""
    logger.info("Downloading facenet_pytorch MTCNN model...")
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize MTCNN
    mtcnn = MTCNN(keep_all=True, min_face_size=30, device=device)
    
    # Also download a face recognition model (optional)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    # Save the models to our custom directory
    mtcnn_path = os.path.join(FACE_DIR, "mtcnn_model.pt")
    resnet_path = os.path.join(FACE_DIR, "resnet_model.pt")
    
    # Save MTCNN model weights
    torch.save({
        'pnet': mtcnn.pnet.state_dict(),
        'rnet': mtcnn.rnet.state_dict(),
        'onet': mtcnn.onet.state_dict(),
    }, mtcnn_path)
    
    # Save face recognition model weights
    torch.save(resnet.state_dict(), resnet_path)
    
    logger.info(f"Face detection model downloaded and saved to {mtcnn_path}")
    logger.info(f"Face recognition model downloaded and saved to {resnet_path}")
    
    # Create a dummy image to verify everything works
    dummy_image = Image.new('RGB', (640, 480))
    boxes, _ = mtcnn.detect(dummy_image)
    logger.info(f"MTCNN detection test on dummy image completed successfully")
    
    return mtcnn, resnet

if __name__ == "__main__":
    logger.info("Starting model download process...")
    clip_model, _ = download_clip_model()
    face_detector, face_recognition = download_face_detection_model()
    logger.info("All models downloaded successfully!")