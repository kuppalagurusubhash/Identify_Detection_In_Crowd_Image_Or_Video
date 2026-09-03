import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from utils.preprocessing import preprocess_face_crop

class EmbeddingService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build lightweight deep feature extractor (MobileNetV3 feature backbone)
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Remove final classification layer to get 576-dim feature representation vector
        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # Standard ImageNet transform for deep features
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, face_crop):
        """
        Extracts a normalized 576-dimensional feature vector embedding for a given face crop.
        Returns:
            np.ndarray of shape (576,), L2-normalized float32
        """
        if face_crop is None or face_crop.size == 0:
            return None

        # Preprocess crop (CLAHE + resizing)
        processed = preprocess_face_crop(face_crop, target_size=(112, 112))
        if processed is None:
            return None

        # Convert BGR (cv2) to RGB (PIL)
        rgb_img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        # Tensor transform
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(tensor_img)
            pooled = self.avgpool(features)
            flattened = torch.flatten(pooled, 1).cpu().numpy()[0]

        # L2 Normalization
        norm = np.linalg.norm(flattened)
        if norm > 0:
            embedding = flattened / norm
        else:
            embedding = flattened

        return embedding.astype(np.float32)
