import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from facenet_pytorch import MTCNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
face_detection_model = MTCNN(keep_all=True, min_face_size=30, device=device)

class CroppedFaceDataset(Dataset):
    def __init__(self, face_crop_dir):
        self.image_paths = [
            os.path.join(face_crop_dir, f) for f in os.listdir(face_crop_dir) if f.lower().endswith(('.jpg', '.png'))
        ]
        print(f"Found {len(self.image_paths)} image(s) in the directory.") 
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        if self.transform:
            image = self.transform(image)
        return image, image_id
